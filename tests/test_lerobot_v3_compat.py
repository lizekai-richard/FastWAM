from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata


def _metadata(
    root: Path,
    *,
    version: str,
    data_path: str,
    video_path: str,
) -> LeRobotDatasetMetadata:
    metadata = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    metadata.root = root
    metadata.info = {
        "codebase_version": version,
        "chunks_size": 1000,
        "data_path": data_path,
        "video_path": video_path,
    }
    return metadata


class LeRobotV3CompatibilityPathTest(unittest.TestCase):
    def test_v21_episode_templates_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = _metadata(
                Path(temp_dir),
                version="v2.1",
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path=(
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
            )

            self.assertEqual(
                metadata.get_data_file_path(1005),
                Path("data/chunk-001/episode_001005.parquet"),
            )
            self.assertEqual(
                metadata.get_video_file_path(1005, "observation.images.cam_high"),
                Path(
                    "videos/chunk-001/observation.images.cam_high/"
                    "episode_001005.mp4"
                ),
            )

    def test_v3_uses_existing_per_episode_compatibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = Path("data/chunk-000/episode_000042.parquet")
            video_path = Path(
                "videos/chunk-000/observation.images.cam_high/episode_000042.mp4"
            )
            (root / data_path).parent.mkdir(parents=True)
            (root / data_path).touch()
            (root / video_path).parent.mkdir(parents=True)
            (root / video_path).touch()
            metadata = _metadata(
                root,
                version="v3.0",
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path=(
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
            )

            self.assertEqual(metadata.get_data_file_path(42), data_path)
            self.assertEqual(
                metadata.get_video_file_path(42, "observation.images.cam_high"),
                video_path,
            )

    def test_v3_does_not_infer_file_index_without_compatibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grouped_data = root / "data/chunk-000/file-000.parquet"
            grouped_video = (
                root
                / "videos/observation.images.cam_high/chunk-000/file-000.mp4"
            )
            grouped_data.parent.mkdir(parents=True)
            grouped_data.touch()
            grouped_video.parent.mkdir(parents=True)
            grouped_video.touch()
            metadata = _metadata(
                root,
                version="v3.0",
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path=(
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
            )

            with self.assertRaisesRegex(ValueError, "refusing to infer file_index"):
                metadata.get_data_file_path(42)
            with self.assertRaisesRegex(ValueError, "refusing to infer file_index"):
                metadata.get_video_file_path(42, "observation.images.cam_high")


if __name__ == "__main__":
    unittest.main()
