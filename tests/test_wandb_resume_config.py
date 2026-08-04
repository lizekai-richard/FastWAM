from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from fastwam.trainer import Wan22Trainer


class WandbResumeConfigTests(unittest.TestCase):
    def test_init_forwards_stable_run_id_and_resume_policy(self) -> None:
        run = object()
        wandb_init = Mock(return_value=run)
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.wandb_enabled = True
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.output_dir = "/tmp/fastwam-run"
        trainer.cfg = OmegaConf.create(
            {
                "wandb": {
                    "workspace": None,
                    "project": "fastwam",
                    "name": "robotwin",
                    "group": "robotwin_clean_flashvla",
                    "mode": "online",
                    "id": "fw-stable-id",
                    "resume": "allow",
                }
            }
        )

        with patch.dict(sys.modules, {"wandb": SimpleNamespace(init=wandb_init)}):
            trainer._init_wandb()

        self.assertIs(trainer.wandb_run, run)
        wandb_init.assert_called_once_with(
            entity=None,
            project="fastwam",
            name="robotwin",
            group="robotwin_clean_flashvla",
            mode="online",
            id="fw-stable-id",
            resume="allow",
            dir="/tmp/fastwam-run",
        )


if __name__ == "__main__":
    unittest.main()
