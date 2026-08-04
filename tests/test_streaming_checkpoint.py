from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _checkpoint_shell(*, enabled: bool, num_slots: int = 4) -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Linear(2, 2)
    model.video_expert = torch.nn.Linear(2, 2)
    model.proprio_encoder = None
    model.vae = type("VAE", (), {"temporal_downsample_factor": 4})()
    model.torch_dtype = torch.float32
    model.loss_lambda_video = 1.0
    model.loss_lambda_action = 1.0
    model.streaming_action_enabled = enabled
    model.streaming_action_num_slots = num_slots
    model.streaming_action_chunk_size = 2
    model.train_video_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
    model.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
    model.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
    model.loaded_checkpoint_streaming_action = None
    return model


class StreamingCheckpointTests(unittest.TestCase):
    def test_streaming_checkpoint_round_trip_records_contract(self) -> None:
        source = _checkpoint_shell(enabled=True)
        target = _checkpoint_shell(enabled=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "streaming.pt"
            source.save_checkpoint(path, step=7)
            payload = target.load_checkpoint(path)

        metadata = payload["streaming_action"]
        self.assertEqual(
            metadata["objective"],
            "flashvla_video_action_streaming_v2",
        )
        self.assertEqual(metadata["num_slots"], 4)
        self.assertEqual(metadata["chunk_size"], 2)
        self.assertEqual(metadata["video_temporal_downsample_factor"], 4)
        self.assertEqual(metadata["video_num_train_timesteps"], 1000)
        self.assertEqual(metadata["action_num_train_timesteps"], 1000)
        self.assertEqual(target.loaded_checkpoint_streaming_action, metadata)

    def test_mismatch_is_rejected_before_weights_are_mutated(self) -> None:
        source = _checkpoint_shell(enabled=True, num_slots=4)
        target = _checkpoint_shell(enabled=True, num_slots=8)
        before = {key: value.clone() for key, value in target.mot.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "streaming.pt"
            source.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                target.load_checkpoint(path)

        for key, value in target.mot.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_legacy_checkpoint_is_initialization_only_for_streaming(self) -> None:
        source = _checkpoint_shell(enabled=False)
        target = _checkpoint_shell(enabled=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            torch.save({"mot": source.mot.state_dict()}, path)
            target.load_checkpoint(path)

        self.assertIsNone(target.loaded_checkpoint_streaming_action)


if __name__ == "__main__":
    unittest.main()
