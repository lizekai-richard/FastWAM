from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf

from fastwam.datasets.lerobot import base_lerobot_dataset
from fastwam.datasets.lerobot import robot_video_dataset
from fastwam import runtime


def _shape_meta():
    return {
        "images": [{"key": "default", "raw_shape": [3, 8, 8]}],
        "state": [{"key": "default", "raw_shape": 2}],
        "action": [{"key": "default", "raw_shape": 3}],
    }


class _FakeMetadata:
    def __init__(self, repo_id, root):
        del root
        self.repo_id = repo_id
        self.fps = 10
        self.total_episodes = 1


class _FakeInnerDataset:
    episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([1]),
    }


class _FakeMultiDataset:
    last_delta_timestamps = None

    def __init__(self, dataset_dirs, episodes, delta_timestamps):
        del dataset_dirs, episodes
        type(self).last_delta_timestamps = delta_timestamps
        self._datasets = [_FakeInnerDataset()]
        self.num_frames = 1


class StreamingDatasetHorizonTest(unittest.TestCase):
    def test_base_dataset_samples_observations_and_actions_independently(self) -> None:
        with (
            mock.patch.object(
                base_lerobot_dataset,
                "LeRobotDatasetMetadata",
                _FakeMetadata,
            ),
            mock.patch.object(
                base_lerobot_dataset,
                "MultiLeRobotDataset",
                _FakeMultiDataset,
            ),
        ):
            dataset = base_lerobot_dataset.BaseLerobotDataset(
                dataset_dirs=["fake"],
                shape_meta=_shape_meta(),
                obs_size=33,
                action_size=64,
                val_set_proportion=0,
            )

        self.assertEqual(dataset.obs_size, 33)
        self.assertEqual(dataset.action_size, 64)
        timestamps = _FakeMultiDataset.last_delta_timestamps
        self.assertEqual(len(timestamps["observation.images"]), 33)
        self.assertEqual(len(timestamps["observation.state"]), 33)
        self.assertEqual(len(timestamps["action"]), 64)

    def test_base_dataset_rejects_non_positive_horizons_before_io(self) -> None:
        for action_size, obs_size in ((0, 33), (64, 0), (-1, 33), (64, -1)):
            with self.subTest(action_size=action_size, obs_size=obs_size):
                with self.assertRaises(ValueError):
                    base_lerobot_dataset.BaseLerobotDataset(
                        dataset_dirs=["not-read"],
                        shape_meta=_shape_meta(),
                        obs_size=obs_size,
                        action_size=action_size,
                    )

    def test_robot_video_dataset_passes_explicit_action_horizon_to_lerobot(self) -> None:
        captured = {}

        class _FakeBaseDataset:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def _set_return_images(self, flag):
                captured["return_images"] = flag

        shape_meta = OmegaConf.create(_shape_meta())
        with (
            mock.patch.object(
                robot_video_dataset,
                "BaseLerobotDataset",
                _FakeBaseDataset,
            ),
            mock.patch.object(robot_video_dataset, "ResizeSmallestSideAspectPreserving"),
            mock.patch.object(robot_video_dataset, "CenterCrop"),
            mock.patch.object(robot_video_dataset, "Normalize"),
        ):
            dataset = robot_video_dataset.RobotVideoDataset(
                dataset_dirs=["fake"],
                shape_meta=shape_meta,
                num_frames=33,
                action_video_freq_ratio=4,
                action_horizon=64,
            )

        self.assertEqual(captured["obs_size"], 33)
        self.assertEqual(captured["action_size"], 64)
        self.assertEqual(dataset.action_horizon, 64)
        self.assertEqual(dataset.video_sample_indices, list(range(0, 33, 4)))

    def test_robot_video_dataset_keeps_legacy_default_action_horizon(self) -> None:
        captured = {}

        class _FakeBaseDataset:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def _set_return_images(self, flag):
                del flag

        with (
            mock.patch.object(
                robot_video_dataset,
                "BaseLerobotDataset",
                _FakeBaseDataset,
            ),
            mock.patch.object(robot_video_dataset, "ResizeSmallestSideAspectPreserving"),
            mock.patch.object(robot_video_dataset, "CenterCrop"),
            mock.patch.object(robot_video_dataset, "Normalize"),
        ):
            dataset = robot_video_dataset.RobotVideoDataset(
                dataset_dirs=["fake"],
                shape_meta=OmegaConf.create(_shape_meta()),
                num_frames=33,
                action_video_freq_ratio=4,
            )

        self.assertEqual(captured["action_size"], 32)
        self.assertEqual(dataset.action_horizon, 32)


class StreamingDatasetRuntimeTest(unittest.TestCase):
    def test_build_datasets_passes_action_horizon_to_train_and_val(self) -> None:
        data_cfg = OmegaConf.create(
            {
                "train": {"_target_": "fake.Train", "pretrained_norm_stats": None},
                "val": {"_target_": "fake.Val", "pretrained_norm_stats": "/tmp/stats.json"},
            }
        )
        created = []

        def _instantiate(config, **kwargs):
            result = object()
            created.append((config._target_, kwargs, result))
            return result

        with (
            mock.patch.object(runtime, "instantiate", side_effect=_instantiate),
            mock.patch.object(runtime.misc, "get_work_dir", return_value="/tmp/work"),
        ):
            train_ds, val_ds = runtime.build_datasets(data_cfg, action_horizon=64)

        self.assertIs(train_ds, created[0][2])
        self.assertIs(val_ds, created[1][2])
        self.assertEqual(created[0][1], {"action_horizon": 64})
        self.assertEqual(
            created[1][1],
            {
                "pretrained_norm_stats": "/tmp/stats.json",
                "action_horizon": 64,
            },
        )

    def test_run_training_derives_action_horizon_from_streaming_model(self) -> None:
        model = SimpleNamespace(
            streaming_action_enabled=True,
            streaming_action_num_slots=4,
            streaming_action_chunk_size=16,
        )
        trainer = mock.Mock()
        trainer_cls = mock.Mock(return_value=trainer)

        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.create(
                {
                    "output_dir": output_dir,
                    "mixed_precision": "no",
                    "model": {"_target_": "fake.Model"},
                    "data": {"train": {"_target_": "fake.Dataset"}},
                }
            )
            with (
                mock.patch.object(runtime, "setup_logging"),
                mock.patch.object(runtime.misc, "register_work_dir"),
                mock.patch.object(runtime, "_resolve_train_device", return_value="cpu"),
                mock.patch.object(runtime, "instantiate", return_value=model),
                mock.patch.object(
                    runtime,
                    "build_datasets",
                    return_value=("train", "val"),
                ) as build_datasets,
                mock.patch.object(runtime, "Wan22Trainer", trainer_cls),
            ):
                runtime.run_training(cfg)

        build_datasets.assert_called_once_with(cfg.data, action_horizon=64)
        trainer.train.assert_called_once_with()

    def test_eval_batching_preserves_64_step_action_and_video_pad_masks(self) -> None:
        action_is_pad = torch.zeros(64, dtype=torch.bool)
        action_is_pad[-7:] = True
        image_is_pad = torch.zeros(9, dtype=torch.bool)
        image_is_pad[-1] = True
        sample = {
            "video": torch.zeros(3, 9, 16, 16),
            "prompt": "test",
            "action": torch.zeros(64, 2),
            "proprio": torch.zeros(32, 3),
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

        batched = runtime.Wan22Trainer._to_batched_eval_sample(sample)

        self.assertEqual(batched["action"].shape, (1, 64, 2))
        self.assertEqual(batched["action_is_pad"].shape, (1, 64))
        self.assertEqual(batched["image_is_pad"].shape, (1, 9))
        torch.testing.assert_close(batched["action_is_pad"][0], action_is_pad)
        torch.testing.assert_close(batched["image_is_pad"][0], image_is_pad)

    def test_streaming_training_rejects_stepwise_action_normalization(self) -> None:
        model = SimpleNamespace(
            streaming_action_enabled=True,
            streaming_action_num_slots=4,
            streaming_action_chunk_size=16,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.create(
                {
                    "output_dir": output_dir,
                    "mixed_precision": "no",
                    "model": {"_target_": "fake.Model"},
                    "data": {
                        "train": {
                            "_target_": "fake.Dataset",
                            "processor": {"use_stepwise_action_norm": True},
                        }
                    },
                }
            )
            with (
                mock.patch.object(runtime, "setup_logging"),
                mock.patch.object(runtime.misc, "register_work_dir"),
                mock.patch.object(runtime, "_resolve_train_device", return_value="cpu"),
                mock.patch.object(runtime, "instantiate", return_value=model),
                mock.patch.object(runtime, "build_datasets") as build_datasets,
            ):
                with self.assertRaisesRegex(ValueError, "global action normalization"):
                    runtime.run_training(cfg)

        build_datasets.assert_not_called()


if __name__ == "__main__":
    unittest.main()
