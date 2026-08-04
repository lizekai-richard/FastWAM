import math
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .streaming_action import (
    StreamingActionState,
    cold_start_token_values,
    initialize_streaming_action_state,
    slot_block_causal_action_mask,
    validate_streaming_config,
)

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        streaming_action_enabled: bool = False,
        streaming_action_num_slots: int = 4,
        streaming_action_chunk_size: int = 16,
        torch_compile_infer_action: bool = False,
        torch_compile_mode: str = "max-autotune",
        torch_compile_dynamic: Optional[bool] = None,
        torch_compile_disable_cudagraphs: Optional[bool] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.streaming_action_enabled = bool(streaming_action_enabled)
        self.streaming_action_num_slots = int(streaming_action_num_slots)
        self.streaming_action_chunk_size = int(streaming_action_chunk_size)
        if self.streaming_action_enabled and type(self) is not FastWAM:
            raise NotImplementedError(
                "This integration currently supports base FastWAM only; "
                f"{type(self).__name__} needs its own video/action streaming contract."
            )
        if self.streaming_action_num_slots <= 0:
            raise ValueError(
                "`streaming_action_num_slots` must be positive, "
                f"got {self.streaming_action_num_slots}."
            )
        if self.streaming_action_chunk_size <= 0:
            raise ValueError(
                "`streaming_action_chunk_size` must be positive, "
                f"got {self.streaming_action_chunk_size}."
            )
        if (
            self.streaming_action_enabled
            and float(self.train_action_scheduler.shift)
            != float(self.infer_action_scheduler.shift)
        ):
            raise ValueError(
                "Streaming action requires identical action train/infer shifts "
                "so slot stages have one schedule contract, got "
                f"train={self.train_action_scheduler.shift}, "
                f"infer={self.infer_action_scheduler.shift}."
            )
        self.torch_compile_infer_action = bool(torch_compile_infer_action)
        self.torch_compile_mode = str(torch_compile_mode)
        self.torch_compile_dynamic = (
            (None if self.streaming_action_enabled else True)
            if torch_compile_dynamic is None
            else bool(torch_compile_dynamic)
        )
        self.torch_compile_disable_cudagraphs = (
            (False if self.streaming_action_enabled else True)
            if torch_compile_disable_cudagraphs is None
            else bool(torch_compile_disable_cudagraphs)
        )
        self.torch_compile_options: Optional[dict[str, Any]] = None
        self.torch_compile_targets: list[str] = []
        self.torch_compile_scope = "none"
        self.loaded_checkpoint_streaming_action: Optional[dict[str, Any]] = None

        self.to(self.device)
        if self.torch_compile_infer_action:
            self._compile_infer_action_entrypoint()

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        streaming_action_enabled: bool = False,
        streaming_action_num_slots: int = 4,
        streaming_action_chunk_size: int = 16,
        torch_compile_infer_action: bool = False,
        torch_compile_mode: str = "max-autotune",
        torch_compile_dynamic: Optional[bool] = None,
        torch_compile_disable_cudagraphs: Optional[bool] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            streaming_action_enabled=streaming_action_enabled,
            streaming_action_num_slots=streaming_action_num_slots,
            streaming_action_chunk_size=streaming_action_chunk_size,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            torch_compile_infer_action=torch_compile_infer_action,
            torch_compile_mode=torch_compile_mode,
            torch_compile_dynamic=torch_compile_dynamic,
            torch_compile_disable_cudagraphs=torch_compile_disable_cudagraphs,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        # Canonicalize generic devices such as ``cuda`` to the concrete device
        # selected by PyTorch (for example ``cuda:0``). Streaming state carries
        # real tensors and should be checked against that resolved device.
        try:
            self.device = next(self.action_expert.parameters()).device
        except StopIteration:
            pass
        return self

    def _move_tensor_tree_to_device(self, value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(
                self._move_tensor_tree_to_device(item, device) for item in value
            )
        if isinstance(value, list):
            return [self._move_tensor_tree_to_device(item, device) for item in value]
        return value

    def _prepare_compile_inputs(self) -> None:
        for expert_name in ("video_expert", "action_expert"):
            expert = getattr(self, expert_name, None)
            if expert is not None and hasattr(expert, "freqs"):
                expert.freqs = self._move_tensor_tree_to_device(
                    expert.freqs, self.device
                )

    def _torch_compile_kwargs(self) -> dict[str, Any]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile.")

        kwargs: dict[str, Any] = {}
        if self.torch_compile_dynamic is not None:
            kwargs["dynamic"] = self.torch_compile_dynamic
        mode = self.torch_compile_mode
        if self.torch_compile_disable_cudagraphs:
            options: dict[str, Any] = {}
            if mode:
                list_mode_options = getattr(
                    getattr(torch, "_inductor", None),
                    "list_mode_options",
                    None,
                )
                if list_mode_options is not None:
                    options.update(
                        list_mode_options(mode, dynamic=self.torch_compile_dynamic)
                    )
                elif mode == "max-autotune":
                    options.update(
                        {
                            "max_autotune": True,
                            "coordinate_descent_tuning": True,
                        }
                    )
                else:
                    raise RuntimeError(
                        "torch.compile mode options are unavailable for "
                        f"mode={mode!r} in this PyTorch build."
                    )
            options["triton.cudagraphs"] = False
            options["triton.cudagraph_trees"] = False
            self.torch_compile_options = dict(options)
            kwargs["options"] = options
        elif mode:
            kwargs["mode"] = mode
            self.torch_compile_options = None
        return kwargs

    def _compile_infer_action_entrypoint(self) -> None:
        self._prepare_compile_inputs()
        torch.set_float32_matmul_precision("high")
        compile_kwargs = self._torch_compile_kwargs()
        if self.streaming_action_enabled:
            # Match FlashVLA's eager dispatcher with fixed-shape compiled
            # inference kernels. The stateful VAE remains eager; its tensor
            # output then flows through one compiled video/MoT prefill kernel
            # and one of the compiled cold/steady action kernels.
            method_names = (
                "_streaming_video_kv_prefill_kernel",
                "_streaming_action_cold_start_kernel",
                "_streaming_action_steady_kernel",
            )
            self.torch_compile_scope = "streaming_inference_kernels"
        else:
            method_names = ("infer_action",)
            self.torch_compile_scope = "legacy_infer_action"

        for method_name in method_names:
            eager_method = getattr(self, method_name)
            eager_alias = (
                "_infer_action_eager"
                if method_name == "infer_action"
                else f"{method_name}_eager"
            )
            setattr(self, eager_alias, eager_method)
            compiled_method = torch.compile(eager_method, **compile_kwargs)
            setattr(self, method_name, compiled_method)
            target = f"{self.__class__.__name__}.{method_name}"
            self.torch_compile_targets.append(target)
            logger.info(
                "Enabled torch.compile for %s with mode=%s dynamic=%s "
                "disable_cudagraphs=%s.",
                target,
                self.torch_compile_mode,
                self.torch_compile_dynamic,
                self.torch_compile_disable_cudagraphs,
            )

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(
        self,
        sample,
        tiled: bool = False,
        first_frame_only: bool = False,
    ):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        if first_frame_only:
            # The streaming objective consumes only the clean visual prefix.
            # Slice on the host so the unused rollout frames are not copied to
            # the accelerator before VAE encoding.
            video = video[:, :, :1]
        input_video = video.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        if action_attention_mask is None:
            mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        else:
            if action_attention_mask.ndim not in (2, 3):
                raise ValueError(
                    "`action_attention_mask` must be [Sa,Sa] or [B,Sa,Sa], "
                    f"got {tuple(action_attention_mask.shape)}."
                )
            if action_attention_mask.shape[-2:] != (action_seq_len, action_seq_len):
                raise ValueError(
                    "`action_attention_mask` shape mismatch: expected trailing shape "
                    f"({action_seq_len}, {action_seq_len}), got "
                    f"{tuple(action_attention_mask.shape)}."
                )
            prefix_shape = action_attention_mask.shape[:-2]
            mask = torch.zeros(
                (*prefix_shape, total_seq_len, total_seq_len),
                dtype=torch.bool,
                device=device,
            )

        # video -> video
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[..., :video_seq_len, :video_seq_len] = video_mask
        # action -> action
        if action_attention_mask is None:
            mask[..., video_seq_len:, video_seq_len:] = True
        else:
            mask[..., video_seq_len:, video_seq_len:] = action_attention_mask.to(
                device=device, dtype=torch.bool
            )
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[..., video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _build_streaming_training_batch(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Construct every padded cold-start configuration for one observation.

        Config ``k`` contains ``k + 1`` real slots. Its real slots occupy the
        final ``k + 1`` intervals of the shifted flow schedule, matching the
        buffer states encountered during streaming cold start.
        """
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,H,A], got {tuple(action.shape)}.")
        batch_size, horizon, action_dim = action.shape
        num_slots = self.streaming_action_num_slots
        chunk_size = self.streaming_action_chunk_size
        validate_streaming_config(
            num_slots=num_slots,
            chunk_size=chunk_size,
            action_horizon=horizon,
        )

        if action_is_pad is None:
            base_valid = torch.ones(
                (batch_size, horizon), dtype=torch.bool, device=action.device
            )
        else:
            if action_is_pad.shape != (batch_size, horizon):
                raise ValueError(
                    "`action_is_pad` must match [B,H], got "
                    f"{tuple(action_is_pad.shape)} vs {(batch_size, horizon)}."
                )
            base_valid = ~action_is_pad.to(device=action.device, dtype=torch.bool)

        config_index = torch.arange(num_slots, device=action.device)
        slot_index = torch.arange(num_slots, device=action.device)
        num_real = config_index + 1
        config_slot_valid = slot_index.unsqueeze(0) < num_real.unsqueeze(1)

        # The left-aligned real slots use the suffix of the clean->noisy stage
        # sequence: N=4 gives [3,P,P,P], [2,3,P,P], ... [0,1,2,3].
        stage = num_slots - num_real.unsqueeze(1) + slot_index.unsqueeze(0)
        stage = stage.clamp(max=num_slots - 1)
        stage = stage.unsqueeze(0).expand(batch_size, -1, -1)

        # Uniformly sample in the raw flow interval, then apply FastWAM's
        # shifted sigma map. All C tokens in a slot share the same timestep.
        interval_sample = torch.rand(
            (batch_size, num_slots, num_slots),
            device=action.device,
            dtype=torch.float32,
        )
        u = (stage.to(torch.float32) + interval_sample) / float(num_slots)
        sigma_per_slot = self.train_action_scheduler._phi(
            u, self.train_action_scheduler.shift
        )
        sigma_per_slot = torch.where(
            config_slot_valid.unsqueeze(0),
            sigma_per_slot,
            torch.ones_like(sigma_per_slot),
        )
        sigma = (
            sigma_per_slot.unsqueeze(-1)
            .expand(-1, -1, -1, chunk_size)
            .reshape(batch_size, num_slots * horizon)
        )
        timestep = sigma * float(self.train_action_scheduler.num_train_timesteps)

        config_valid = (
            config_slot_valid.unsqueeze(-1)
            .expand(-1, -1, chunk_size)
            .reshape(num_slots, horizon)
        )
        valid = config_valid.unsqueeze(0) & base_valid.unsqueeze(1)

        clean = action.unsqueeze(1).expand(-1, num_slots, -1, -1).clone()
        clean = clean.masked_fill(~valid.unsqueeze(-1), 0)
        clean = clean.reshape(batch_size, num_slots * horizon, action_dim)
        valid = valid.reshape(batch_size, num_slots * horizon)

        noise = torch.randn_like(clean)
        sigma_model = sigma.to(device=action.device, dtype=action.dtype).unsqueeze(-1)
        noisy = (1.0 - sigma_model) * clean + sigma_model * noise
        target = noise - clean
        return {
            "clean": clean,
            "noisy": noisy,
            "target": target,
            "timestep": timestep.to(dtype=action.dtype),
            "valid": valid,
        }

    def _build_streaming_training_action_mask(
        self,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Block config branches while retaining slot-causal attention."""
        if valid.ndim != 2:
            raise ValueError(f"`valid` must be [B,N*H], got {tuple(valid.shape)}.")
        batch_size, total_action_len = valid.shape
        num_slots = self.streaming_action_num_slots
        chunk_size = self.streaming_action_chunk_size
        horizon = num_slots * chunk_size
        expected = num_slots * horizon
        if total_action_len != expected:
            raise ValueError(
                f"Streaming training action length must be N*H={expected}, got {total_action_len}."
            )

        local_causal = slot_block_causal_action_mask(
            num_slots=num_slots,
            chunk_size=chunk_size,
            device=valid.device,
        )
        mask = torch.zeros(
            (batch_size, total_action_len, total_action_len),
            dtype=torch.bool,
            device=valid.device,
        )
        for config_idx in range(num_slots):
            start = config_idx * horizon
            end = start + horizon
            branch_valid = valid[:, start:end]
            branch_mask = (
                local_causal.unsqueeze(0)
                & branch_valid.unsqueeze(1)
                & branch_valid.unsqueeze(2)
            )
            mask[:, start:end, start:end] = branch_mask
        return mask

    def training_loss_streaming_action(self, sample, tiled: bool = False):
        """Train aligned video diffusion and FlashVLA-style action streaming.

        The full video window keeps the standard FastWAM diffusion objective.
        The aligned action horizon is expanded into all ``N`` padded streaming
        configurations, while action attention remains limited to the clean
        first video frame so future ground-truth video cannot leak into policy
        prediction.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        image_is_pad = inputs["image_is_pad"]

        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "Streaming video/action training requires "
                "`video_attention_mask_mode='first_frame_causal'` so the clean "
                "prefix cannot absorb future ground-truth video tokens."
            )
        future_video_latent_steps = int(input_latents.shape[2]) - 1
        if future_video_latent_steps != self.streaming_action_num_slots:
            raise ValueError(
                "Streaming video/action alignment requires one future video latent "
                "per action-buffer slot: got "
                f"future_video_latent_steps={future_video_latent_steps}, "
                f"num_slots={self.streaming_action_num_slots}."
            )
        if inputs["first_frame_latents"] is None:
            raise ValueError(
                "Streaming video/action training requires a clean first-frame latent "
                "prefix (`fuse_vae_embedding_in_latents=true`)."
            )

        stream = self._build_streaming_training_batch(
            action=action,
            action_is_pad=inputs["action_is_pad"],
        )

        batch_size = input_latents.shape[0]
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        latents_video = self.train_video_scheduler.add_noise(
            input_latents,
            noise_video,
            timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents,
            noise_video,
            timestep_video,
        )
        latents_video[:, :, :1] = inputs["first_frame_latents"]

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=stream["noisy"],
            timestep=stream["timestep"],
            context=context,
            context_mask=context_mask,
        )
        # Each training branch represents the same action horizon and must use
        # the same action positions; do not let RoPE positions run across branch
        # boundaries in the concatenated shared-observation sequence.
        horizon = self.streaming_action_num_slots * self.streaming_action_chunk_size
        base_freqs = self.action_expert.freqs[:horizon].view(horizon, 1, -1)
        action_pre["freqs"] = base_freqs.repeat(
            self.streaming_action_num_slots, 1, 1
        ).to(device=action_pre["tokens"].device)

        action_attention_mask = self._build_streaming_training_action_mask(
            valid=stream["valid"]
        )
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            action_attention_mask=action_attention_mask,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video
        ).to(loss_video_per_sample.device, dtype=loss_video_per_sample.dtype)
        loss_video = (loss_video_per_sample * video_weight).mean()

        token_loss = F.mse_loss(
            pred_action.float(), stream["target"].float(), reduction="none"
        ).mean(dim=2)
        valid_float = stream["valid"].to(dtype=token_loss.dtype)
        token_weight = self.train_action_scheduler.training_weight(
            stream["timestep"]
        ).to(device=token_loss.device, dtype=token_loss.dtype)
        valid_sum = valid_float.sum(dim=1).clamp(min=1.0)
        loss_per_sample = (token_loss * token_weight * valid_float).sum(dim=1) / valid_sum
        loss_action = loss_per_sample.mean()
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_streaming_action": self.loss_lambda_action
            * float(loss_action.detach().item()),
        }

    def training_loss(self, sample, tiled: bool = False):
        if self.streaming_action_enabled:
            return self.training_loss_streaming_action(sample=sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _streaming_video_kv_prefill_kernel(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Tokenize VAE latents and run the tensor-only MoT video prefix.

        Wan VAE encoding stays in the eager dispatcher because its encoder
        mutates Python feature-cache lists. Everything after that boundary is
        fixed-shape tensor work and is safe to specialize and CUDA-graph.
        The returned cache is consumed immediately by the action kernel in the
        same control step; it is never persisted as caller-owned state.
        """
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=first_frame_latents.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )
        return video_kv_cache

    @torch.no_grad()
    def _streaming_action_denoise_buffer(
        self,
        buffer: torch.Tensor,
        valid_mask: torch.Tensor,
        timestep_action: torch.Tensor,
        delta_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run one tensor-only action prediction/update over the rolling buffer."""
        pred_action = self._predict_action_noise_with_cache(
            latents_action=buffer.to(dtype=self.torch_dtype),
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        updated = buffer + pred_action.float() * delta_action.float().unsqueeze(-1)
        return torch.where(valid_mask.unsqueeze(-1), updated, buffer)

    @torch.no_grad()
    def _streaming_action_cold_start_kernel(
        self,
        buffer: torch.Tensor,
        valid_mask: torch.Tensor,
        new_noise: torch.Tensor,
        timestep_action: torch.Tensor,
        delta_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cold-start tensor kernel compiled once for every tensor-valued stage.

        The eager caller validates shapes/state and constructs the tensor-valued
        timestep, delta, and validity inputs. This kernel contains no random
        sampling or Python rollout state, so every cold-start phase reuses one
        fixed-shape Dynamo graph just like FlashVLA's compiled ``_cold_start``.
        """
        updated_buffer = self._streaming_action_denoise_buffer(
            buffer=buffer,
            valid_mask=valid_mask,
            timestep_action=timestep_action,
            delta_action=delta_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )

        batch_size, horizon, action_dim = buffer.shape
        num_slots = self.streaming_action_num_slots
        chunk_size = self.streaming_action_chunk_size
        slot_valid = valid_mask.reshape(batch_size, num_slots, chunk_size).all(dim=2)
        first_invalid = (~slot_valid).to(dtype=torch.int64).argmax(dim=1)
        slot_index = torch.arange(num_slots, device=buffer.device)
        target_slots = slot_index.unsqueeze(0) == first_invalid.unsqueeze(1)
        target_tokens = target_slots.repeat_interleave(chunk_size, dim=1)
        tiled_noise = (
            new_noise.float()
            .unsqueeze(1)
            .expand(-1, num_slots, -1, -1)
            .reshape(batch_size, horizon, action_dim)
        )
        next_buffer = torch.where(
            target_tokens.unsqueeze(-1), tiled_noise, updated_buffer
        )
        next_valid_mask = valid_mask | target_tokens
        return next_buffer, next_valid_mask

    @torch.no_grad()
    def _streaming_action_steady_kernel(
        self,
        buffer: torch.Tensor,
        valid_mask: torch.Tensor,
        new_noise: torch.Tensor,
        timestep_action: torch.Tensor,
        delta_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed-shape steady-state action kernel for CUDA-graph replay."""
        updated_buffer = self._streaming_action_denoise_buffer(
            buffer=buffer,
            valid_mask=valid_mask,
            timestep_action=timestep_action,
            delta_action=delta_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        chunk_size = self.streaming_action_chunk_size
        emitted = updated_buffer[:, :chunk_size, :].clone()
        next_buffer = torch.cat(
            (updated_buffer[:, chunk_size:, :], new_noise.float()), dim=1
        )
        next_valid_mask = torch.ones_like(valid_mask)
        return emitted, next_buffer, next_valid_mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        latency_recorder: Optional[Any] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        if latency_recorder is not None:
            latency_recorder.start("video_kv_prefill")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        if latency_recorder is not None:
            latency_recorder.stop("video_kv_prefill")
            latency_recorder.start("action_prediction")

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        if latency_recorder is not None:
            latency_recorder.stop("action_prediction")

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_profiled(
        self,
        latency_recorder: Any,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the legacy action path while recording the canonical two stages."""
        if latency_recorder is None:
            raise ValueError("`latency_recorder` is required for profiled inference.")
        eager_method = getattr(self, "_infer_action_eager", self.infer_action)
        return eager_method(latency_recorder=latency_recorder, **kwargs)

    @torch.no_grad()
    def initialize_action_stream_state(
        self,
        initial_noise: torch.Tensor,
        sigma_shift: Optional[float] = None,
    ) -> StreamingActionState:
        """Initialize caller-owned padded streaming state from one noise chunk."""
        if not self.streaming_action_enabled:
            raise RuntimeError(
                "Streaming action is disabled. Set model.streaming_action.enabled=true."
            )
        effective_shift = (
            float(self.infer_action_scheduler.shift)
            if sigma_shift is None
            else float(sigma_shift)
        )
        if not math.isfinite(effective_shift) or effective_shift <= 0:
            raise ValueError(f"`sigma_shift` must be positive, got {effective_shift}.")
        state = initialize_streaming_action_state(
            initial_noise=initial_noise.to(device=self.device, dtype=torch.float32),
            num_slots=self.streaming_action_num_slots,
            chunk_size=self.streaming_action_chunk_size,
            schedule_shift=effective_shift,
        )
        if state.buffer.shape[2] != int(self.action_expert.action_dim):
            raise ValueError(
                "Initial streaming noise action dim mismatch: "
                f"expected {self.action_expert.action_dim}, got {state.buffer.shape[2]}."
            )
        return state

    def _sample_streaming_noise_chunk(
        self,
        batch_size: int,
        action_generator: torch.Generator,
    ) -> torch.Tensor:
        if not isinstance(action_generator, torch.Generator):
            raise TypeError(
                "`action_generator` must be a caller-owned torch.Generator."
            )
        generator_device = action_generator.device
        return torch.randn(
            (
                batch_size,
                self.streaming_action_chunk_size,
                int(self.action_expert.action_dim),
            ),
            generator=action_generator,
            device=generator_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def infer_action_stream_step(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        state: StreamingActionState,
        new_noise: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        sigma_shift: Optional[float] = None,
        tiled: bool = False,
        latency_recorder: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Advance one FlashVLA action-buffer stage for one control call.

        This low-level API performs no random sampling. Both ``state`` and the
        next noise chunk are supplied by the caller, and the returned state is
        a new FP32 value rather than model-global mutable state.
        """
        self.eval()
        if not self.streaming_action_enabled:
            raise RuntimeError(
                "Streaming action is disabled. Set model.streaming_action.enabled=true."
            )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_stream_step` requires "
                "`video_attention_mask_mode='first_frame_causal'`."
            )
        if not isinstance(state, StreamingActionState):
            raise TypeError(
                f"`state` must be StreamingActionState, got {type(state).__name__}."
            )

        batch_size, action_horizon, action_dim = state.buffer.shape
        validate_streaming_config(
            num_slots=self.streaming_action_num_slots,
            chunk_size=self.streaming_action_chunk_size,
            action_horizon=action_horizon,
        )
        if batch_size != 1:
            raise ValueError(
                "Current image VAE action inference supports batch_size=1, "
                f"got streaming state batch {batch_size}."
            )
        if action_dim != int(self.action_expert.action_dim):
            raise ValueError(
                f"Streaming state action dim must be {self.action_expert.action_dim}, got {action_dim}."
            )
        if state.buffer.device != self.device:
            raise ValueError(
                f"Streaming state must be on model device {self.device}, got {state.buffer.device}."
            )
        effective_shift = (
            float(self.infer_action_scheduler.shift)
            if sigma_shift is None
            else float(sigma_shift)
        )
        if not math.isfinite(effective_shift) or effective_shift <= 0:
            raise ValueError(f"`sigma_shift` must be positive, got {effective_shift}.")
        if (
            state.schedule_shift is not None
            and float(state.schedule_shift) != effective_shift
        ):
            raise ValueError(
                "Cannot change `sigma_shift` while reusing a streaming state: "
                f"state={state.schedule_shift}, requested={effective_shift}. Reset the state first."
            )
        if new_noise.shape != (
            batch_size,
            self.streaming_action_chunk_size,
            action_dim,
        ):
            raise ValueError(
                "`new_noise` must be [B,C,A], got "
                f"{tuple(new_noise.shape)}."
            )
        new_noise = new_noise.to(device=self.device, dtype=torch.float32)

        # State validity must be a slot-aligned prefix and remain synchronized
        # with the per-episode step counter.
        slot_valid_tokens = state.valid_mask.reshape(
            batch_size,
            self.streaming_action_num_slots,
            self.streaming_action_chunk_size,
        )
        slot_all = slot_valid_tokens.all(dim=2)
        slot_any = slot_valid_tokens.any(dim=2)
        if not torch.equal(slot_all, slot_any):
            raise ValueError("Streaming valid_mask must mark complete action slots.")
        expected_prefix = (
            torch.arange(self.streaming_action_num_slots, device=self.device).unsqueeze(0)
            < (state.steps + 1)
            .clamp(max=self.streaming_action_num_slots)
            .unsqueeze(1)
        )
        if not torch.equal(slot_all, expected_prefix):
            raise ValueError(
                "Streaming valid_mask is inconsistent with state.steps; reset or restore both together."
            )
        buffer_was_full = bool(state.valid_mask.all().item())

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "`input_image` must be resized before infer, expected multiples "
                f"of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if latency_recorder is not None:
            latency_recorder.start("video_kv_prefill")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
                raise ValueError("Context batch size must match streaming state batch size.")
            context = context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            context_mask = context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if (
            self.device.type == "cuda"
            and bool(getattr(self, "torch_compile_infer_action", False))
            and not bool(getattr(self, "torch_compile_disable_cudagraphs", True))
        ):
            # One control step invokes the compiled video graph followed by a
            # compiled action graph. Mark only the outer step so the cache
            # remains live across that parent/child CUDA-graph boundary.
            mark_step_begin = getattr(
                getattr(torch, "compiler", None),
                "cudagraph_mark_step_begin",
                None,
            )
            if mark_step_begin is None:
                raise RuntimeError(
                    "Streaming CUDA-graph compile requires "
                    "torch.compiler.cudagraph_mark_step_begin()."
                )
            mark_step_begin()
        video_kv_cache = self._streaming_video_kv_prefill_kernel(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
        )
        if not video_kv_cache:
            raise RuntimeError(
                "Compiled video prefill returned an empty MoT K/V cache."
            )
        video_seq_len = int(video_kv_cache[0]["k"].shape[1])
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        video_tokens_per_frame = (
            int(first_frame_latents.shape[3]) // patch_h
        ) * (
            int(first_frame_latents.shape[4]) // patch_w
        )
        if latency_recorder is not None:
            latency_recorder.stop("video_kv_prefill")
            latency_recorder.start("action_prediction")

        slot_timesteps, slot_deltas = self.infer_action_scheduler.build_streaming_slot_schedule(
            num_slots=self.streaming_action_num_slots,
            device=self.device,
            dtype=torch.float32,
            shift_override=effective_shift,
        )
        timestep_action = cold_start_token_values(
            clean_to_noisy_slot_values=slot_timesteps,
            chunk_size=self.streaming_action_chunk_size,
            step=state.steps,
            padding_value=slot_timesteps[-1],
            batch_size=batch_size,
        ).to(dtype=self.torch_dtype)
        delta_action = cold_start_token_values(
            clean_to_noisy_slot_values=slot_deltas,
            chunk_size=self.streaming_action_chunk_size,
            step=state.steps,
            padding_value=0.0,
            batch_size=batch_size,
        )
        base_action_mask = slot_block_causal_action_mask(
            num_slots=self.streaming_action_num_slots,
            chunk_size=self.streaming_action_chunk_size,
            device=self.device,
        )
        action_attention_mask = (
            base_action_mask.unsqueeze(0)
            & state.valid_mask.unsqueeze(1)
            & state.valid_mask.unsqueeze(2)
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=video_tokens_per_frame,
            device=self.device,
            action_attention_mask=action_attention_mask,
        )
        emitted: Optional[torch.Tensor]
        if buffer_was_full:
            emitted, next_buffer, next_valid_mask = (
                self._streaming_action_steady_kernel(
                    buffer=state.buffer,
                    valid_mask=state.valid_mask,
                    new_noise=new_noise,
                    timestep_action=timestep_action,
                    delta_action=delta_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
            )
        else:
            emitted = None
            next_buffer, next_valid_mask = (
                self._streaming_action_cold_start_kernel(
                    buffer=state.buffer,
                    valid_mask=state.valid_mask,
                    new_noise=new_noise,
                    timestep_action=timestep_action,
                    delta_action=delta_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
            )
        # CUDA-graph outputs use replay-owned storage. Persist caller-owned
        # state in ordinary eager allocations so a later cold/steady replay
        # cannot overwrite this rollout (or another interleaved rollout).
        next_buffer = next_buffer.clone()
        next_valid_mask = next_valid_mask.clone()
        next_state = StreamingActionState(
            buffer=next_buffer,
            valid_mask=next_valid_mask,
            steps=state.steps + 1,
            schedule_shift=effective_shift,
        )
        if latency_recorder is not None:
            latency_recorder.stop("action_prediction")

        action_out = None
        if emitted is not None:
            action_out = emitted[0].detach().to(device="cpu", dtype=torch.float32)
        return {"action": action_out, "state": next_state}

    @torch.no_grad()
    def infer_action_streaming(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        streaming_state: Optional[StreamingActionState],
        action_generator: torch.Generator,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 1,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        latency_recorder: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Caller-stateful public streaming protocol used by policies/benchmarks."""
        del negative_prompt, text_cfg_scale, num_inference_steps, seed, rand_device
        expected_horizon = (
            self.streaming_action_num_slots * self.streaming_action_chunk_size
        )
        if int(action_horizon) != expected_horizon:
            raise ValueError(
                f"Streaming action_horizon must be {expected_horizon}, got {action_horizon}."
            )
        if streaming_state is None:
            initial_noise = self._sample_streaming_noise_chunk(
                batch_size=1,
                action_generator=action_generator,
            )
            streaming_state = self.initialize_action_stream_state(
                initial_noise,
                sigma_shift=sigma_shift,
            )
        new_noise = self._sample_streaming_noise_chunk(
            batch_size=streaming_state.buffer.shape[0],
            action_generator=action_generator,
        )
        result = self.infer_action_stream_step(
            prompt=prompt,
            input_image=input_image,
            state=streaming_state,
            new_noise=new_noise,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            sigma_shift=sigma_shift,
            tiled=tiled,
            latency_recorder=latency_recorder,
        )
        return {
            "action": result["action"],
            "streaming_state": result["state"],
        }

    @torch.no_grad()
    def infer_action_streaming_profiled(
        self,
        latency_recorder: Any,
        **kwargs,
    ) -> dict[str, Any]:
        if latency_recorder is None:
            raise ValueError("`latency_recorder` is required for profiled inference.")
        eager_method = getattr(
            self,
            "_infer_action_streaming_eager",
            self.infer_action_streaming,
        )
        return eager_method(
            latency_recorder=latency_recorder,
            **kwargs,
        )

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def streaming_action_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.streaming_action_enabled,
            "objective": (
                "flashvla_video_action_streaming_v2"
                if self.streaming_action_enabled
                else "legacy"
            ),
            "num_slots": (
                self.streaming_action_num_slots
                if self.streaming_action_enabled
                else None
            ),
            "chunk_size": (
                self.streaming_action_chunk_size
                if self.streaming_action_enabled
                else None
            ),
            "video_train_shift": float(self.train_video_scheduler.shift),
            "video_num_train_timesteps": int(
                self.train_video_scheduler.num_train_timesteps
            ),
            "video_temporal_downsample_factor": int(
                self.vae.temporal_downsample_factor
            ),
            "loss_lambda_video": self.loss_lambda_video,
            "loss_lambda_action": self.loss_lambda_action,
            "action_train_shift": float(self.train_action_scheduler.shift),
            "action_infer_shift": float(self.infer_action_scheduler.shift),
            "action_num_train_timesteps": int(
                self.train_action_scheduler.num_train_timesteps
            ),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "streaming_action": self.streaming_action_checkpoint_metadata(),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Checkpoint payload must be a dict, got {type(payload).__name__}: {path}"
            )
        if "mot" not in payload and "dit" not in payload:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        streaming_metadata = payload.get("streaming_action")
        if streaming_metadata is not None and not isinstance(streaming_metadata, dict):
            raise ValueError(
                "Checkpoint `streaming_action` metadata must be a dict, got "
                f"{type(streaming_metadata).__name__}."
            )
        if streaming_metadata is not None and not isinstance(
            streaming_metadata.get("enabled"), bool
        ):
            raise ValueError(
                "Checkpoint `streaming_action.enabled` metadata must be boolean."
            )
        checkpoint_is_streaming = bool(
            streaming_metadata is not None
            and streaming_metadata.get("enabled", False)
        )
        if checkpoint_is_streaming:
            expected_objective = "flashvla_video_action_streaming_v2"
            if streaming_metadata.get("objective") != expected_objective:
                raise ValueError(
                    "Unsupported streaming checkpoint objective: "
                    f"{streaming_metadata.get('objective')!r}; expected {expected_objective!r}."
                )
            if self.streaming_action_enabled:
                expected_values = {
                    "num_slots": self.streaming_action_num_slots,
                    "chunk_size": self.streaming_action_chunk_size,
                    "video_train_shift": float(self.train_video_scheduler.shift),
                    "video_num_train_timesteps": int(
                        self.train_video_scheduler.num_train_timesteps
                    ),
                    "video_temporal_downsample_factor": int(
                        self.vae.temporal_downsample_factor
                    ),
                    "loss_lambda_video": self.loss_lambda_video,
                    "loss_lambda_action": self.loss_lambda_action,
                    "action_train_shift": float(self.train_action_scheduler.shift),
                    "action_infer_shift": float(self.infer_action_scheduler.shift),
                    "action_num_train_timesteps": int(
                        self.train_action_scheduler.num_train_timesteps
                    ),
                }
                mismatches = {
                    key: (streaming_metadata.get(key), expected)
                    for key, expected in expected_values.items()
                    if streaming_metadata.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        "Streaming checkpoint/config metadata mismatch: "
                        f"{mismatches}."
                    )

        # Mutate weights only after all objective/config compatibility checks
        # above have succeeded.
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        else:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        self.loaded_checkpoint_streaming_action = (
            None if streaming_metadata is None else dict(streaming_metadata)
        )

        if checkpoint_is_streaming and not self.streaming_action_enabled:
            logger.warning(
                "Loading a streaming-trained checkpoint while streaming_action is disabled; "
                "legacy action inference does not match its training objective."
            )
        elif not checkpoint_is_streaming and self.streaming_action_enabled:
            logger.warning(
                "Streaming action is enabled, but checkpoint %s has no compatible streaming "
                "training metadata. Treat it only as legacy initialization and fine-tune before deployment.",
                path,
            )
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
