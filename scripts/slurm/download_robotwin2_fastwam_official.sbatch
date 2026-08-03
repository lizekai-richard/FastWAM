#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --job-name=rt2-fastwam-data
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/data/rt2-fastwam-data-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/data/rt2-fastwam-data-%j.err

set -euo pipefail

STORAGE="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail"
ENV_PREFIX="${STORAGE}/miniforge3/envs/fastwam"
DEST="${STORAGE}/data/robotwin2.0-fastwam-official"
DATASET_ROOT="${DEST}/robotwin2.0"
REPO_ID="yuanty/robotwin2.0-fastwam"
REVISION="aac262c35d02cc71b2f6ef670bd65fd9f2bb2547"

export HF_HOME="${STORAGE}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=3600
export PYTHONUNBUFFERED=1

mkdir -p "${DEST}" "${STORAGE}/logs/FastWAM/data"

exec 9>"${DEST}/.download.lock"
if ! flock -n 9; then
  echo "Another official FastWAM dataset download is already running." >&2
  exit 1
fi

echo "[download] repo=${REPO_ID} revision=${REVISION} destination=${DEST}"
"${ENV_PREFIX}/bin/python" - "${DEST}" "${REPO_ID}" "${REVISION}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

dest, repo_id, revision = sys.argv[1:]
path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    local_dir=dest,
    allow_patterns=[
        "README.md",
        "dataset_stats.json",
        "robotwin2.0.tar.gz.part-*",
    ],
    max_workers=8,
)
print(f"[download] snapshot={path}")
PY

"${ENV_PREFIX}/bin/python" - "${DEST}" "${REPO_ID}" "${REVISION}" <<'PY'
import os
import sys

from huggingface_hub import HfApi

dest, repo_id, revision = sys.argv[1:]
info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
expected = {
    f.rfilename: f.size
    for f in info.siblings
    if f.rfilename.startswith("robotwin2.0.tar.gz.part-")
}
if len(expected) != 8:
    raise RuntimeError(f"Expected 8 archive parts, found {len(expected)}")
for name, size in sorted(expected.items()):
    path = os.path.join(dest, name)
    actual = os.path.getsize(path)
    if actual != size:
        raise RuntimeError(f"Size mismatch for {name}: expected={size}, actual={actual}")
    print(f"[verify-download] {name} bytes={actual}")
PY

if [[ ! -f "${DATASET_ROOT}/.FASTWAM_DATASET_READY" ]]; then
  echo "[extract] streaming split archive into ${DEST}"
  cat "${DEST}"/robotwin2.0.tar.gz.part-* | tar -xzf - -C "${DEST}"
fi

"${ENV_PREFIX}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
required = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)
for relpath in required:
    path = os.path.join(root, relpath)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Missing or empty required metadata: {path}")

with open(os.path.join(root, "meta/info.json"), encoding="utf-8") as f:
    info = json.load(f)

expected = {
    "codebase_version": "v2.1",
    "total_episodes": 27500,
    "total_frames": 6075103,
}
for key, value in expected.items():
    if info.get(key) != value:
        raise RuntimeError(f"Unexpected {key}: expected={value!r}, actual={info.get(key)!r}")

print(
    "[verify-dataset] "
    f"version={info['codebase_version']} "
    f"episodes={info['total_episodes']} frames={info['total_frames']}"
)
PY

data_files=$(find "${DATASET_ROOT}/data" -type f -name '*.parquet' | wc -l)
video_files=$(find "${DATASET_ROOT}/videos" -type f -name '*.mp4' | wc -l)
if [[ "${data_files}" -ne 27500 ]]; then
  echo "Expected 27500 parquet files, found ${data_files}." >&2
  exit 1
fi
if [[ "${video_files}" -lt 1 ]]; then
  echo "No video files found under ${DATASET_ROOT}/videos." >&2
  exit 1
fi

printf 'revision=%s\ncompleted_at=%s\n' "${REVISION}" "$(date --iso-8601=seconds)" \
  > "${DATASET_ROOT}/.FASTWAM_DATASET_READY"

echo "[complete] data_files=${data_files} video_files=${video_files}"
du -sh "${DEST}" "${DATASET_ROOT}"
