#!/usr/bin/env python3
"""Benchmark legacy or streaming FastWAM action inference on real samples.

Preset selection is intentionally tied to camera count:
  - 2 views: LIBERO data/config + LIBERO release checkpoint.
  - 3 views: RoboTwin data/config + RoboTwin release checkpoint.

Both modes use the same two-stage breakdown: video KV prefill (including VAE)
and action prediction. Streaming measurements are steady-state only, keep one
caller-owned state/RNG per sample, and require explicit streaming model APIs;
the benchmark never substitutes or estimates missing measurements.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.dataset_utils import (  # noqa: E402
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.runtime import _mixed_precision_to_model_dtype  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402

try:  # Support both direct script execution and module-style test imports.
    from .inference_api import (  # type: ignore[import-not-found]
        INFERENCE_MODES,
        InferenceAPI,
        accepts_keyword,
        resolve_inference_api,
        unpack_inference_result,
    )
except ImportError:
    from inference_api import (  # type: ignore[no-redef]
        INFERENCE_MODES,
        InferenceAPI,
        accepts_keyword,
        resolve_inference_api,
        unpack_inference_result,
    )


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    task: str
    checkpoint_file: str
    dataset_stats_file: str
    storage_dataset_parts: tuple[str, ...]
    dataset_loader: str


class StageLatencyRecorder:
    """Record nested inference stages without synchronizing between stages."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.clock = "cuda_event" if self.device.type == "cuda" else "perf_counter"
        self._active: dict[str, Any] = {}
        self._records: dict[str, list[Any]] = defaultdict(list)
        self._finished = False
        self._stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None

    def start(self, name: str) -> None:
        if self._finished:
            raise RuntimeError("Cannot record a stage after finish().")
        if name in self._active:
            raise RuntimeError(f"Latency stage {name!r} is already active.")
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(self._stream)
            self._active[name] = event
        else:
            self._active[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self._active:
            raise RuntimeError(f"Latency stage {name!r} was not started.")
        start = self._active.pop(name)
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(self._stream)
            self._records[name].append((start, end))
        else:
            self._records[name].append((time.perf_counter() - start) * 1000.0)

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("finish() may only be called once.")
        if self._active:
            raise RuntimeError(f"Latency stages still active at finish(): {sorted(self._active)}")
        self._finished = True

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            stages_ms = {
                name: [float(start.elapsed_time(end)) for start, end in records]
                for name, records in self._records.items()
            }
        else:
            stages_ms = {
                name: [float(value) for value in records]
                for name, records in self._records.items()
            }
        return {
            "clock": self.clock,
            "device": str(self.device),
            "stages_ms": stages_ms,
        }


@dataclass
class StreamingRunState:
    """Benchmark-owned state for one independent streaming rollout."""

    model_state: Any = None
    action_generator: torch.Generator | None = None
    cold_start_calls: int = 0


PRESETS = {
    2: BenchmarkPreset(
        name="libero",
        task="libero_uncond_2cam224_1e-4",
        checkpoint_file="libero_uncond_2cam224.pt",
        dataset_stats_file="libero_uncond_2cam224_dataset_stats.json",
        storage_dataset_parts=("data", "libero"),
        dataset_loader="libero_hf_v3",
    ),
    3: BenchmarkPreset(
        name="robotwin",
        task="robotwin_uncond_3cam_384_1e-4",
        checkpoint_file="robotwin_uncond_3cam_384.pt",
        dataset_stats_file="robotwin_uncond_3cam_384_dataset_stats.json",
        storage_dataset_parts=("data", "robotwin2.0", "robotwin2.0"),
        dataset_loader="base_lerobot",
    ),
}


LIBERO_HF_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
LIBERO_HF_STATE_KEY = "observation.state"


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(text)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _expand_path(path: str | Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(path)))
    p = Path(text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _storage_path(*parts: str) -> Path | None:
    storage = os.environ.get("STORAGE")
    if not storage:
        return None
    return Path(storage).expanduser().resolve().joinpath(*parts)


def _default_release_path(filename: str) -> Path:
    candidates = [
        _storage_path("models", "fastwam", filename),
        PROJECT_ROOT / "checkpoints" / "fastwam_release" / filename,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    first = candidates[0] if candidates[0] is not None else candidates[1]
    return first.resolve()


def _default_dataset_path(preset: BenchmarkPreset) -> Path | None:
    return _storage_path(*preset.storage_dataset_parts)


def _load_hydra_cfg(task_name: str) -> DictConfig:
    register_default_resolvers()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "configs"),
        job_name="benchmark_inference_latency",
    ):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    OmegaConf.set_struct(cfg, False)
    return cfg


def _build_processed_sample(
    data_cfg: DictConfig,
    dataset_stats_path: Path,
    sample_index: int,
) -> dict[str, Any]:
    dataset, processor = _build_processed_dataset(
        data_cfg=data_cfg,
        dataset_stats_path=dataset_stats_path,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} out of bounds for dataset length {len(dataset)}.")
    sample = dataset[sample_index]
    sample["_processor"] = processor
    sample["_dataset_length"] = len(dataset)
    return sample


def _build_processed_dataset(
    data_cfg: DictConfig,
    dataset_stats_path: Path,
) -> tuple[BaseLerobotDataset, FastWAMProcessor]:
    processor = _build_processor(data_cfg=data_cfg, dataset_stats_path=dataset_stats_path)

    dataset_dirs_raw = OmegaConf.to_container(data_cfg.dataset_dirs, resolve=True)
    dataset_dirs = [str(_expand_path(path)) for path in dataset_dirs_raw]
    shape_meta = OmegaConf.to_container(data_cfg.shape_meta, resolve=True)
    dataset = BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=int(data_cfg.num_frames),
        action_size=int(data_cfg.num_frames) - 1,
        val_set_proportion=float(data_cfg.get("val_set_proportion", 0.0)),
        is_training_set=False,
        global_sample_stride=int(data_cfg.get("global_sample_stride", 1)),
    )
    dataset._set_return_images(True)
    dataset.set_processor(processor)

    if len(dataset) <= 0:
        raise RuntimeError("Dataset is empty.")
    return dataset, processor


def _build_processor(data_cfg: DictConfig, dataset_stats_path: Path) -> FastWAMProcessor:
    processor: FastWAMProcessor = instantiate(data_cfg.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats_path)))
    return processor


def _decode_hf_image_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_path is not None:
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("HF image dict has neither bytes nor path.")
    else:
        raise TypeError(f"Unsupported HF image value type: {type(value)!r}")
    return transforms_F.to_tensor(image)


class LocalLiberoHfV3Dataset:
    """Minimal local reader for HuggingFaceVLA/libero's LeRobot v3 parquet layout."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        tasks_path = self.root / "meta" / "tasks.parquet"
        episodes_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not info_path.exists() or not tasks_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(
                "Expected HuggingFaceVLA/libero files under "
                f"{self.root}: meta/info.json, meta/tasks.parquet, and meta/episodes/chunk-000/file-000.parquet."
            )

        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.data_path_template = self.info["data_path"]
        self.total_frames = int(self.info["total_frames"])

        task_table = pq.read_table(tasks_path).to_pydict()
        task_text_key = "task" if "task" in task_table else "__index_level_0__"
        self.tasks = {
            int(task_idx): str(task)
            for task_idx, task in zip(task_table["task_index"], task_table[task_text_key], strict=True)
        }

        ep_table = pq.read_table(
            episodes_path,
            columns=[
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "tasks",
            ],
        ).to_pydict()
        self.episode_indices = [int(x) for x in ep_table["episode_index"]]
        self.from_indices = [int(x) for x in ep_table["dataset_from_index"]]
        self.to_indices = [int(x) for x in ep_table["dataset_to_index"]]
        self.data_chunk_indices = [int(x) for x in ep_table["data/chunk_index"]]
        self.data_file_indices = [int(x) for x in ep_table["data/file_index"]]
        self.episode_tasks = ep_table["tasks"]
        self._file_cache: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}

    def __len__(self) -> int:
        return self.total_frames

    def _episode_position(self, sample_index: int) -> int:
        pos = bisect.bisect_right(self.from_indices, sample_index) - 1
        if pos < 0 or sample_index >= self.to_indices[pos]:
            raise IndexError(f"sample_index={sample_index} is outside LIBERO dataset bounds.")
        return pos

    def _load_file_rows(self, chunk_index: int, file_index: int) -> dict[int, dict[str, Any]]:
        cache_key = (int(chunk_index), int(file_index))
        cached = self._file_cache.get(cache_key)
        if cached is not None:
            return cached

        rel_path = self.data_path_template.format(
            chunk_index=int(chunk_index),
            file_index=int(file_index),
        )
        table = pq.read_table(
            self.root / rel_path,
            columns=[
                "index",
                "task_index",
                LIBERO_HF_STATE_KEY,
                *LIBERO_HF_IMAGE_KEYS,
            ],
        ).to_pydict()
        rows: dict[int, dict[str, Any]] = {}
        for row_idx, global_idx in enumerate(table["index"]):
            rows[int(global_idx)] = {
                "task_index": int(table["task_index"][row_idx]),
                LIBERO_HF_STATE_KEY: table[LIBERO_HF_STATE_KEY][row_idx],
                LIBERO_HF_IMAGE_KEYS[0]: table[LIBERO_HF_IMAGE_KEYS[0]][row_idx],
                LIBERO_HF_IMAGE_KEYS[1]: table[LIBERO_HF_IMAGE_KEYS[1]][row_idx],
            }
        self._file_cache[cache_key] = rows
        return rows

    def get_processed_sample(
        self,
        sample_index: int,
        data_cfg: DictConfig,
        processor: FastWAMProcessor,
    ) -> dict[str, Any]:
        ep_pos = self._episode_position(sample_index)
        ep_start = self.from_indices[ep_pos]
        ep_end = self.to_indices[ep_pos]
        query_indices = [min(ep_end - 1, max(ep_start, sample_index + t)) for t in range(int(data_cfg.num_frames))]
        rows = self._load_file_rows(self.data_chunk_indices[ep_pos], self.data_file_indices[ep_pos])

        camera_videos = []
        image_size = data_cfg.processor.val_transforms[1].size
        for image_key in LIBERO_HF_IMAGE_KEYS:
            frames = torch.stack([_decode_hf_image_to_tensor(rows[idx][image_key]) for idx in query_indices])
            frames = transforms_F.resize(
                frames,
                size=[int(image_size[0]), int(image_size[1])],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            camera_videos.append(frames)
        pixel_values = torch.stack(camera_videos, dim=0)

        state = torch.as_tensor(
            [rows[idx][LIBERO_HF_STATE_KEY] for idx in query_indices],
            dtype=torch.float32,
        )
        normed = processor.normalizer.forward({"state": {"default": state.clone()}})
        proprio = normed["state"]["default"]

        first_row = rows[sample_index]
        task = self.tasks.get(int(first_row["task_index"]))
        if task is None and self.episode_tasks[ep_pos]:
            task = str(self.episode_tasks[ep_pos][0])
        if task is None:
            task = ""

        return {
            "idx": sample_index,
            "instruction": task,
            "pixel_values": pixel_values,
            "proprio": proprio,
        }


def _assemble_video_tensor(sample: dict[str, Any], data_cfg: DictConfig) -> torch.Tensor:
    num_frames = int(data_cfg.num_frames)
    ratio = int(data_cfg.action_video_freq_ratio)
    video_indices = list(range(0, num_frames, ratio))

    video = sample["pixel_values"]
    if video.ndim == 5:
        video = video[:, video_indices, :, :, :]
        num_cameras, t_video, channels, height, width = video.shape
    elif video.ndim == 4:
        video = video[video_indices, :, :, :].unsqueeze(0)
        num_cameras, t_video, channels, height, width = video.shape
    else:
        raise ValueError(f"Expected pixel_values to be 4D or 5D, got shape {tuple(video.shape)}")

    del channels, height, width
    concat_mode = data_cfg.get("concat_multi_camera", "horizontal")
    video = video.view(num_cameras, t_video, *video.shape[2:])

    if concat_mode == "robotwin":
        if num_cameras != 3:
            raise ValueError(f"robotwin concat requires 3 cameras, got {num_cameras}.")
        cam_top = transforms_F.resize(
            video[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_left = transforms_F.resize(
            video[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_right = transforms_F.resize(
            video[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        video = torch.cat([cam_top, bottom], dim=-2)
    elif num_cameras > 1:
        if concat_mode == "horizontal":
            video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
        elif concat_mode == "vertical":
            video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
        else:
            raise ValueError(f"Unsupported concat_multi_camera={concat_mode!r}")
    else:
        video = video.squeeze(0)

    video_size = data_cfg.get("video_size", None)
    if video_size is None or len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    resize_transform = ResizeSmallestSideAspectPreserving(
        args={"img_w": int(video_size[1]), "img_h": int(video_size[0])}
    )
    crop_transform = CenterCrop(args={"img_w": int(video_size[1]), "img_h": int(video_size[0])})
    normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
    video = normalize_transform(crop_transform(resize_transform(video)))
    return video


def _load_model(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: str,
    mixed_precision: str,
    load_text_encoder: bool,
) -> torch.nn.Module:
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    cfg.model.load_text_encoder = bool(load_text_encoder)
    cfg.model.redirect_common_files = False
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint_path))
    return model.to(device).eval()


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _summarize_ms(values: list[float]) -> dict[str, float]:
    total = sum(values)
    n = max(len(values), 1)
    return {
        "mean_ms": total / n,
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
        "p50_ms": _percentile(values, 50.0),
        "p90_ms": _percentile(values, 90.0),
        "p95_ms": _percentile(values, 95.0),
    }


BREAKDOWN_STAGES = (
    "video_kv_prefill",
    "action_prediction",
)


def _normalize_breakdown(raw: dict[str, Any]) -> dict[str, Any]:
    stages_raw = raw.get("stages_ms")
    if not isinstance(stages_raw, dict):
        raise ValueError("Profiled inference did not return a stages_ms mapping.")

    def require_stage(name: str, expected_count: int) -> list[float]:
        values = stages_raw.get(name)
        if not isinstance(values, list) or len(values) != expected_count:
            got = None if not isinstance(values, list) else len(values)
            raise ValueError(
                f"Latency stage {name!r} expected {expected_count} record(s), got {got}."
            )
        return [float(value) for value in values]

    return {
        "clock": raw.get("clock"),
        "device": raw.get("device"),
        "stages_ms": {
            name: require_stage(name, 1)[0]
            for name in BREAKDOWN_STAGES
        },
    }


def _summarize_breakdown_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not calls:
        raise ValueError("At least one profiled call is required for latency breakdown.")

    stage_summary = {
        name: _summarize_ms([float(call["stages_ms"][name]) for call in calls])
        for name in BREAKDOWN_STAGES
    }

    return {
        "clock": calls[0]["clock"],
        "device": calls[0]["device"],
        "stages": stage_summary,
        "per_sample": calls,
    }


def _make_action_generator(
    *,
    seed: int | None,
    sample_index: int,
    rand_device: str,
) -> torch.Generator:
    """Create one RNG per benchmark rollout instead of reseeding each call."""

    generator = torch.Generator(device=rand_device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed) + int(sample_index))
    return generator


def _prepare_streaming_call_kwargs(
    method: Any,
    base_kwargs: dict[str, Any],
    run_state: StreamingRunState,
) -> dict[str, Any]:
    """Attach caller-owned state and RNG to the explicit streaming protocol."""

    method_name = getattr(method, "__qualname__", type(method).__name__)
    if not accepts_keyword(method, "streaming_state"):
        raise RuntimeError(
            f"{method_name} must accept streaming_state=; streaming state cannot be model-global."
        )

    kwargs = dict(base_kwargs)
    kwargs["streaming_state"] = run_state.model_state
    # Reusing infer_action(seed=...) on every call would restart the same noise
    # sequence. Streaming owns one generator for the full episode instead.
    kwargs.pop("seed", None)
    if accepts_keyword(method, "action_generator"):
        kwargs["action_generator"] = run_state.action_generator
    elif accepts_keyword(method, "generator"):
        kwargs["generator"] = run_state.action_generator
    else:
        raise RuntimeError(
            f"{method_name} must accept action_generator= or generator= so "
            "streaming noise is episode-owned and is not reset on every call."
        )
    return kwargs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-views",
        type=int,
        choices=sorted(PRESETS),
        required=True,
        help="Camera view count. 2 selects LIBERO; 3 selects RoboTwin.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--dataset-stats", type=str, default=None, help="Override dataset stats JSON path.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset root. Defaults to the preset path under $STORAGE/data.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to benchmark.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of consecutive samples to time.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="Diffusion/flow inference steps.")
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default="legacy",
        help=(
            "Use legacy iterative infer_action or caller-stateful infer_action_streaming. "
            "Streaming mode fails fast if the model does not expose the streaming API."
        ),
    )
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rand-device", type=str, default="cpu")
    parser.add_argument("--seed", type=str, default="0", help="Integer seed, or 'none'.")
    parser.add_argument("--sigma-shift", type=str, default=None, help="Float sigma shift override, or 'none'.")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode/decode.")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable model-level torch.compile for infer_action with max-autotune. Compile time is paid during warmup.",
    )
    parser.add_argument(
        "--include-text-encoder",
        action="store_true",
        help="Load and time prompt encoding inside infer_action. By default a zero dummy context is used.",
    )
    parser.add_argument(
        "--latency-breakdown",
        action="store_true",
        help=(
            "Run an additional eager-instrumented pass reporting only video KV prefill "
            "(including image VAE encoding) and full action prediction/denoising."
        ),
    )
    parser.add_argument(
        "--breakdown-warmup",
        type=int,
        default=1,
        help="Warmup calls for the eager-instrumented breakdown pass.",
    )
    parser.add_argument(
        "--breakdown-iters",
        type=int,
        default=None,
        help="Iterations for the breakdown pass. Defaults to --iters.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write benchmark JSON.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.num_samples <= 0:
        raise ValueError("--warmup must be >= 0, --iters must be > 0, and --num-samples must be > 0.")
    if args.breakdown_warmup < 0:
        raise ValueError("--breakdown-warmup must be >= 0.")
    breakdown_iters = args.iters if args.breakdown_iters is None else int(args.breakdown_iters)
    if breakdown_iters <= 0:
        raise ValueError("--breakdown-iters must be > 0.")

    preset = PRESETS[args.num_views]
    checkpoint_path = (
        _expand_path(args.checkpoint)
        if args.checkpoint is not None
        else _default_release_path(preset.checkpoint_file)
    )
    dataset_stats_path = (
        _expand_path(args.dataset_stats)
        if args.dataset_stats is not None
        else _default_release_path(preset.dataset_stats_file)
    )
    dataset_dir = _expand_path(args.dataset_dir) if args.dataset_dir is not None else _default_dataset_path(preset)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found: {dataset_stats_path}")
    if dataset_dir is not None and not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    cfg = _load_hydra_cfg(preset.task)
    cfg.data.train.pretrained_norm_stats = str(dataset_stats_path)
    cfg.data.train.is_training_set = False
    if "streaming_action" not in cfg.model:
        raise RuntimeError(
            "Model config has no streaming_action section. Use the updated configs/model/fastwam.yaml."
        )
    cfg.model.streaming_action.enabled = args.inference_mode == "streaming"
    cfg.model.torch_compile_infer_action = bool(args.torch_compile)
    cfg.model.torch_compile_mode = "max-autotune"
    cfg.model.torch_compile_dynamic = True
    cfg.model.torch_compile_disable_cudagraphs = True
    if dataset_dir is not None and preset.dataset_loader == "base_lerobot":
        cfg.data.train.dataset_dirs = [str(dataset_dir)]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")

    if preset.dataset_loader == "libero_hf_v3":
        if dataset_dir is None:
            raise RuntimeError("LIBERO HuggingFaceVLA/libero benchmark requires --dataset-dir or $STORAGE.")
        processor = _build_processor(data_cfg=cfg.data.train, dataset_stats_path=dataset_stats_path)
        dataset = LocalLiberoHfV3Dataset(dataset_dir)
        dataset_loader = preset.dataset_loader
    else:
        dataset, processor = _build_processed_dataset(
            data_cfg=cfg.data.train,
            dataset_stats_path=dataset_stats_path,
        )
        dataset_loader = preset.dataset_loader

    sample_start = int(args.sample_index)
    sample_end = sample_start + int(args.num_samples)
    if sample_start < 0 or sample_start >= len(dataset):
        raise IndexError(f"sample_index={sample_start} out of bounds for dataset length {len(dataset)}.")
    if sample_end > len(dataset):
        raise IndexError(
            f"Requested samples [{sample_start}, {sample_end}) exceed dataset length {len(dataset)}."
        )

    action_horizon = int(cfg.data.train.num_frames) - 1
    num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    streaming_num_slots = int(cfg.model.streaming_action.num_slots)
    streaming_chunk_size = int(cfg.model.streaming_action.chunk_size)
    if args.inference_mode == "streaming":
        configured_horizon = streaming_num_slots * streaming_chunk_size
        if configured_horizon != action_horizon:
            raise ValueError(
                "Streaming buffer must cover the dataset action horizon exactly: "
                f"num_slots({streaming_num_slots}) * chunk_size({streaming_chunk_size}) "
                f"= {configured_horizon}, action_horizon={action_horizon}."
            )

    model = _load_model(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        mixed_precision=args.mixed_precision,
        load_text_encoder=bool(args.include_text_encoder),
    )
    compile_targets = list(getattr(model, "torch_compile_targets", []))
    seed = _parse_optional_int(args.seed)
    sigma_shift = _parse_optional_float(args.sigma_shift)
    inference_api = resolve_inference_api(
        model,
        args.inference_mode,
        require_profiled=bool(args.latency_breakdown),
    )

    benchmark_inputs: list[dict[str, Any]] = []
    for sample_idx in range(sample_start, sample_end):
        if isinstance(dataset, LocalLiberoHfV3Dataset):
            sample = dataset.get_processed_sample(sample_idx, data_cfg=cfg.data.train, processor=processor)
        else:
            sample = dataset[sample_idx]
        video = _assemble_video_tensor(sample=sample, data_cfg=cfg.data.train)
        input_image = video[0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        proprio = sample["proprio"][0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        prompt = DEFAULT_PROMPT.format(task=sample["instruction"])
        item: dict[str, Any] = {
            "sample_index": sample_idx,
            "input_image": input_image,
            "proprio": proprio,
            "prompt": prompt,
        }
        if not args.include_text_encoder:
            context_len = int(cfg.data.train.get("context_len", 128))
            item["context"] = torch.zeros(
                (1, context_len, int(model.text_dim)),
                device=model.device,
                dtype=model.torch_dtype,
            )
            item["context_mask"] = torch.ones(
                (1, context_len),
                device=model.device,
                dtype=torch.bool,
            )
        benchmark_inputs.append(item)

    if not benchmark_inputs:
        raise RuntimeError("No benchmark inputs were constructed.")
    input_image_shape = list(benchmark_inputs[0]["input_image"].shape)

    accepts_num_video_frames = accepts_keyword(inference_api.canonical, "num_video_frames")

    def build_infer_kwargs(item: dict[str, Any]) -> dict[str, Any]:
        infer_kwargs: dict[str, Any] = {
            "input_image": item["input_image"],
            "action_horizon": action_horizon,
            "proprio": item["proprio"],
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "num_inference_steps": int(args.num_inference_steps),
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": args.rand_device,
            "tiled": bool(args.tiled),
        }
        if args.include_text_encoder:
            infer_kwargs["prompt"] = item["prompt"]
        else:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = item["context"]
            infer_kwargs["context_mask"] = item["context_mask"]
        if accepts_num_video_frames:
            infer_kwargs["num_video_frames"] = num_video_frames
        return infer_kwargs

    def make_streaming_runs() -> dict[int, StreamingRunState]:
        if not inference_api.stateful:
            return {}
        return {
            int(item["sample_index"]): StreamingRunState(
                action_generator=_make_action_generator(
                    seed=seed,
                    sample_index=int(item["sample_index"]),
                    rand_device=args.rand_device,
                )
            )
            for item in benchmark_inputs
        }

    def call_once(
        item: dict[str, Any],
        *,
        api: InferenceAPI,
        run_state: StreamingRunState | None,
        profiled: bool,
    ) -> tuple[Any, dict[str, Any] | None]:
        method = api.profiled if profiled else api.canonical
        if method is None:
            expected_name = api.profiled_name or f"{api.canonical_name}_profiled"
            raise RuntimeError(f"Missing required profiled method {expected_name}().")

        infer_kwargs = build_infer_kwargs(item)
        if api.stateful:
            if run_state is None:
                raise RuntimeError("Streaming inference requires benchmark-owned run state.")
            infer_kwargs = _prepare_streaming_call_kwargs(method, infer_kwargs, run_state)

        recorder = StageLatencyRecorder(model.device) if profiled else None
        if recorder is not None:
            infer_kwargs["latency_recorder"] = recorder
        with torch.inference_mode():
            result = method(**infer_kwargs)
        action, new_model_state = unpack_inference_result(result, api.mode)
        if api.stateful:
            assert run_state is not None
            run_state.model_state = new_model_state
        return action, recorder.finish() if recorder is not None else None

    def require_steady_action(action: Any, *, sample_index: int, phase: str) -> None:
        if action is None:
            raise RuntimeError(
                f"Streaming API returned action=None during {phase} for sample {sample_index}. "
                "Only cold-start calls may omit an action; no placeholder latency is recorded."
            )

    def prime_streaming_runs(runs: dict[int, StreamingRunState]) -> None:
        if not inference_api.stateful:
            return
        for item in benchmark_inputs:
            sample_index = int(item["sample_index"])
            run = runs[sample_index]
            for _ in range(streaming_num_slots + 1):
                action, _ = call_once(
                    item,
                    api=inference_api,
                    run_state=run,
                    profiled=False,
                )
                run.cold_start_calls += 1
                if action is not None:
                    break
            else:
                raise RuntimeError(
                    "Streaming cold start did not produce an action within "
                    f"num_slots+1={streaming_num_slots + 1} calls for sample {sample_index}."
                )

    canonical_runs = make_streaming_runs()
    prime_streaming_runs(canonical_runs)
    for warmup_idx in range(args.warmup):
        item = benchmark_inputs[warmup_idx % len(benchmark_inputs)]
        sample_index = int(item["sample_index"])
        action, _ = call_once(
            item,
            api=inference_api,
            run_state=canonical_runs.get(sample_index),
            profiled=False,
        )
        if inference_api.stateful:
            require_steady_action(action, sample_index=sample_index, phase="canonical warmup")
    _sync_if_needed(model.device)

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    latencies_ms: list[float] = []
    per_sample: list[dict[str, float | int]] = []
    total_timed_calls = int(args.iters) * len(benchmark_inputs)
    for _ in range(args.iters):
        for item in benchmark_inputs:
            sample_index = int(item["sample_index"])
            _sync_if_needed(model.device)
            t0 = time.perf_counter()
            action, _ = call_once(
                item,
                api=inference_api,
                run_state=canonical_runs.get(sample_index),
                profiled=False,
            )
            _sync_if_needed(model.device)
            if inference_api.stateful:
                require_steady_action(action, sample_index=sample_index, phase="canonical timing")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)
            per_sample.append(
                {
                    "sample_index": sample_index,
                    "latency_ms": float(latency_ms),
                }
            )

    latency_breakdown: dict[str, Any] = {"enabled": False}
    if args.latency_breakdown:
        profiled_runs = make_streaming_runs()
        prime_streaming_runs(profiled_runs)
        for warmup_idx in range(args.breakdown_warmup):
            item = benchmark_inputs[warmup_idx % len(benchmark_inputs)]
            sample_index = int(item["sample_index"])
            action, _ = call_once(
                item,
                api=inference_api,
                run_state=profiled_runs.get(sample_index),
                profiled=True,
            )
            if inference_api.stateful:
                require_steady_action(action, sample_index=sample_index, phase="breakdown warmup")

        profiled_calls: list[dict[str, Any]] = []
        for _ in range(breakdown_iters):
            for item in benchmark_inputs:
                sample_index = int(item["sample_index"])
                action, raw_breakdown = call_once(
                    item,
                    api=inference_api,
                    run_state=profiled_runs.get(sample_index),
                    profiled=True,
                )
                if inference_api.stateful:
                    require_steady_action(action, sample_index=sample_index, phase="breakdown timing")
                if raw_breakdown is None:
                    raise RuntimeError("Profiled inference returned no real latency records.")
                normalized = _normalize_breakdown(raw=raw_breakdown)
                normalized["sample_index"] = sample_index
                profiled_calls.append(normalized)

        canonical_is_compiled = bool(
            args.torch_compile
            and any(
                target == inference_api.canonical_name
                or target.endswith(f".{inference_api.canonical_name}")
                for target in compile_targets
            )
        )
        action_stage_definition = (
            "Inference schedule construction plus the full iterative ActionDiT denoising loop, "
            "including scheduler updates; output D2H is excluded"
            if args.inference_mode == "legacy"
            else (
                "One ActionDiT prediction over the caller-owned rolling buffer plus the shifted "
                "per-slot scheduler update, emission, shift, and noise append; output D2H is excluded"
            )
        )
        latency_breakdown = {
            "enabled": True,
            "inference_mode": args.inference_mode,
            "execution_mode": "eager_instrumented",
            "canonical_latency_execution_mode": "compiled" if canonical_is_compiled else "eager",
            "same_execution_path_as_canonical_latency": not canonical_is_compiled,
            "measurement_note": (
                "CUDA stages use events on the current stream with one synchronization at call end. "
                "The model must emit both named stages; the benchmark never derives or estimates either "
                "stage from end-to-end latency. Host-side argument/state validation, caller RNG/noise "
                "preparation, and output D2H are excluded. A compiled canonical method is profiled "
                "through its explicit eager-instrumented companion."
            ),
            "stage_definitions": {
                "video_kv_prefill": (
                    "Image device transfer, VAE encoding, conditioning, video token/mask preparation, "
                    "and the video transformer pass that materializes the KV cache"
                ),
                "action_prediction": action_stage_definition,
            },
            "warmup": int(args.breakdown_warmup),
            "iters": int(breakdown_iters),
            "total_profiled_calls": int(breakdown_iters) * len(benchmark_inputs),
            "input_devices": {
                "image": str(benchmark_inputs[0]["input_image"].device),
                "proprio": str(benchmark_inputs[0]["proprio"].device),
                "context": (
                    str(benchmark_inputs[0]["context"].device)
                    if "context" in benchmark_inputs[0]
                    else None
                ),
            },
            **_summarize_breakdown_calls(calls=profiled_calls),
        }

    checkpoint_metadata = getattr(
        model, "loaded_checkpoint_streaming_action", None
    )
    checkpoint_declares_streaming = bool(
        isinstance(checkpoint_metadata, dict)
        and checkpoint_metadata.get("enabled", False)
    )
    mode_matches_checkpoint = (
        bool(inference_api.stateful) == checkpoint_declares_streaming
        if checkpoint_metadata is not None
        else not inference_api.stateful
    )
    effective_action_infer_shift = (
        float(model.infer_action_scheduler.shift)
        if sigma_shift is None
        else float(sigma_shift)
    )
    shift_matches_checkpoint: bool | None = None
    if checkpoint_declares_streaming:
        shift_matches_checkpoint = (
            checkpoint_metadata.get("action_infer_shift")
            == effective_action_infer_shift
        )
    checkpoint_contract_matches = bool(
        mode_matches_checkpoint and shift_matches_checkpoint is not False
    )
    if checkpoint_metadata is None:
        checkpoint_compatibility = (
            "legacy_initialization_only; latency_not_task_quality"
            if inference_api.stateful
            else "legacy_checkpoint_without_objective_metadata"
        )
    elif mode_matches_checkpoint and shift_matches_checkpoint is False:
        checkpoint_compatibility = "streaming_schedule_mismatches_checkpoint"
    elif mode_matches_checkpoint:
        checkpoint_compatibility = (
            "streaming_metadata_validated; latency_not_task_quality"
            if inference_api.stateful
            else "legacy_objective_metadata_present"
        )
    else:
        checkpoint_compatibility = "inference_mode_mismatches_checkpoint_objective"

    streaming_result = {
        "enabled": bool(inference_api.stateful),
        "state_ownership": "caller",
        "rng_ownership": "caller_per_sample",
        "num_slots": streaming_num_slots if inference_api.stateful else None,
        "chunk_size": streaming_chunk_size if inference_api.stateful else None,
        "buffer_horizon": (
            streaming_num_slots * streaming_chunk_size if inference_api.stateful else None
        ),
        "cold_start_calls_by_sample": (
            {
                str(sample_index): int(run.cold_start_calls)
                for sample_index, run in canonical_runs.items()
            }
            if inference_api.stateful
            else {}
        ),
        "timed_phase": "steady_state" if inference_api.stateful else "full_legacy_call",
        "checkpoint_compatibility": checkpoint_compatibility,
        "checkpoint_metadata": checkpoint_metadata,
        "mode_matches_checkpoint": bool(mode_matches_checkpoint),
        "shift_matches_checkpoint": shift_matches_checkpoint,
        "checkpoint_contract_matches": checkpoint_contract_matches,
    }

    result = {
        "schema_version": "3.0",
        "preset": preset.name,
        "num_views": int(args.num_views),
        "task": preset.task,
        "checkpoint": str(checkpoint_path),
        "dataset_stats": str(dataset_stats_path),
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
        "dataset_loader": dataset_loader,
        "dataset_length": int(len(dataset)),
        "sample_start": int(sample_start),
        "num_samples": int(args.num_samples),
        "device": str(model.device),
        "mixed_precision": args.mixed_precision,
        "inference_mode": args.inference_mode,
        "inference_api": {
            "canonical": inference_api.canonical_name,
            "profiled": inference_api.profiled_name,
        },
        "include_text_encoder": bool(args.include_text_encoder),
        "context_source": "text_encoder" if args.include_text_encoder else "zero_dummy",
        "num_inference_steps": (
            int(args.num_inference_steps) if args.inference_mode == "legacy" else None
        ),
        "sigma_shift_override": sigma_shift,
        "effective_action_infer_shift": effective_action_infer_shift,
        "action_horizon": int(action_horizon),
        "num_video_frames": int(num_video_frames),
        "input_image_shape": input_image_shape,
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "total_timed_calls": int(total_timed_calls),
        "streaming_action": streaming_result,
        "torch_compile": {
            "enabled": bool(getattr(model, "torch_compile_infer_action", False)),
            "canonical_method": inference_api.canonical_name,
            "mode": getattr(model, "torch_compile_mode", None) if args.torch_compile else None,
            "dynamic": getattr(model, "torch_compile_dynamic", None) if args.torch_compile else None,
            "disable_cudagraphs": getattr(model, "torch_compile_disable_cudagraphs", None)
            if args.torch_compile
            else None,
            "options": getattr(model, "torch_compile_options", None) if args.torch_compile else None,
            "targets": compile_targets,
        },
        "latency": _summarize_ms(latencies_ms),
        "latency_breakdown": latency_breakdown,
        "per_sample": per_sample,
    }
    if model.device.type == "cuda":
        result["cuda_memory"] = {
            "max_allocated_gb": torch.cuda.max_memory_allocated(model.device) / (1024**3),
            "max_reserved_gb": torch.cuda.max_memory_reserved(model.device) / (1024**3),
        }

    result_json = json.dumps(result, indent=2, allow_nan=False)
    print(result_json)
    if args.output_json is not None:
        output_path = _expand_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
