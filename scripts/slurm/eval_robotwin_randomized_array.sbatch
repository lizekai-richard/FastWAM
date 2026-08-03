#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --job-name=robotwin_rand_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-49%8
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_randomized/%x-%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_randomized/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
STORAGE="${STORAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail}"
ENV_PREFIX="${ENV_PREFIX:-${STORAGE}/miniforge3/envs/fastwam}"

RUN_ID="${RUN_ID:-robotwin_randomized_joint_step021040}"
RUN_ROOT="${RUN_ROOT:-${STORAGE}/evaluate_results/${RUN_ID}}"
TASK="${TASK:-robotwin_clean_joint_3cam_384_1e-4}"
CKPT="${CKPT:-${STORAGE}/runs/FastWAM/robotwin_clean_joint_3cam_384_1e-4/slurm_requeue/checkpoints/weights/step_021040.pt}"
DATASET_STATS="${DATASET_STATS:-${STORAGE}/runs/FastWAM/robotwin_clean_joint_3cam_384_1e-4/slurm_requeue/dataset_stats.json}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
FASTWAM_SEED="${FASTWAM_SEED:-42}"

TASK_LIST_FILE="${RUN_ROOT}/task_list.txt"
SUMMARY_LOCK="${RUN_ROOT}/summary.lock"
EVAL_PID=""

source "${STORAGE}/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

PY_SITE="$("${ENV_PREFIX}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
TORCH_LIB="${PY_SITE}/torch/lib"
NVIDIA_LIBS="$(
  find "${PY_SITE}/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib 2>/dev/null \
    | paste -sd ':' -
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

export STORAGE
export PYTHONUNBUFFERED=1
export FASTWAM_ROBOTWIN_DIRECT_OUTPUT_DIR=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${STORAGE}/models/diffsynth}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton-autotune}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${XDG_CACHE_HOME}/pycache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

mkdir -p \
  "${RUN_ROOT}" \
  "$(dirname "${SLURM_STDOUT:-${STORAGE}/logs/FastWAM/robotwin_randomized/dummy}")" \
  "${STORAGE}/logs/FastWAM/robotwin_randomized" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${PIP_CACHE_DIR}" \
  "${MPLCONFIGDIR}"

cd "${PROJECT_DIR}"

TASK_NAME="$("${ENV_PREFIX}/bin/python" - "${SLURM_ARRAY_TASK_ID:-0}" "${TASK_LIST_FILE}" <<'PY'
import sys
from pathlib import Path
import yaml

idx = int(sys.argv[1])
task_list_file = Path(sys.argv[2])
task_file = Path("third_party/RoboTwin/task_config/_eval_step_limit.yml")
with task_file.open("r", encoding="utf-8") as f:
    tasks = list(dict.fromkeys(yaml.safe_load(f).keys()))
if idx < 0 or idx >= len(tasks):
    raise SystemExit(f"array index {idx} outside task count {len(tasks)}")
task_list_file.parent.mkdir(parents=True, exist_ok=True)
if not task_list_file.exists():
    task_list_file.write_text("\n".join(tasks) + "\n", encoding="utf-8")
print(tasks[idx])
PY
)"

TASK_ROOT="${RUN_ROOT}/tasks/${TASK_NAME}"
PROGRESS_FILE="${TASK_ROOT}/progress_random.json"
RESULT_FILE="${TASK_ROOT}/${TASK_NAME}/_result_random.txt"
mkdir -p "${TASK_ROOT}"

run_summary() {
  (
    flock -x 200
    "${ENV_PREFIX}/bin/python" experiments/robotwin/summarize_randomized_eval.py \
      --run-root "${RUN_ROOT}" \
      --target-episodes "${EVAL_EPISODES}" || true
  ) 200>"${SUMMARY_LOCK}"
}

descendant_pids() {
  local pid="$1"
  local children
  children="$(pgrep -P "${pid}" 2>/dev/null || true)"
  local child
  for child in ${children}; do
    echo "${child}"
    descendant_pids "${child}"
  done
}

signal_process_tree() {
  local pid="$1"
  local sig="$2"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  local child
  for child in $(descendant_pids "${pid}"); do
    kill -s "${sig}" "${child}" 2>/dev/null || true
  done
  kill -s "${sig}" "${pid}" 2>/dev/null || true
}

handle_usr1() {
  trap - USR1
  echo "[requeue] received USR1 task=${TASK_NAME} job=${SLURM_JOB_ID:-unknown} restart=${SLURM_RESTART_COUNT:-0}" >&2
  if [[ -n "${EVAL_PID}" ]] && kill -0 "${EVAL_PID}" 2>/dev/null; then
    signal_process_tree "${EVAL_PID}" TERM
    wait "${EVAL_PID}" || true
  fi
  run_summary

  if [[ -f "${RESULT_FILE}" ]]; then
    echo "[requeue] result already exists, not requeueing: ${RESULT_FILE}" >&2
    exit 0
  fi
  if (( ${SLURM_RESTART_COUNT:-0} >= MAX_RESTARTS )); then
    echo "[requeue] restart limit reached task=${TASK_NAME} max=${MAX_RESTARTS}" >&2
    exit 1
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}

trap handle_usr1 USR1

if [[ -f "${RESULT_FILE}" ]]; then
  echo "[skip] task=${TASK_NAME} existing result=${RESULT_FILE}" >&2
  run_summary
  exit 0
fi

echo "[start] run_root=${RUN_ROOT}" >&2
echo "[start] task=${TASK_NAME} array=${SLURM_ARRAY_TASK_ID:-0} restart=${SLURM_RESTART_COUNT:-0}" >&2
echo "[start] progress=${PROGRESS_FILE}" >&2
echo "[start] result=${RESULT_FILE}" >&2

"${ENV_PREFIX}/bin/python" experiments/robotwin/eval_robotwin_single.py \
  task="${TASK}" \
  ckpt="${CKPT}" \
  seed="${FASTWAM_SEED}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS}" \
  EVALUATION.task_name="${TASK_NAME}" \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes="${EVAL_EPISODES}" \
  EVALUATION.output_dir="${TASK_ROOT}" \
  EVALUATION.progress_file="${PROGRESS_FILE}" \
  EVALUATION.num_inference_steps="${INFERENCE_STEPS}" \
  EVALUATION.replan_steps="${REPLAN_STEPS}" \
  gpu_id=0 &
EVAL_PID="$!"

set +e
wait "${EVAL_PID}"
RC="$?"
set -e
EVAL_PID=""

run_summary

if [[ "${RC}" != "0" ]]; then
  echo "[failed] task=${TASK_NAME} rc=${RC}" >&2
  exit "${RC}"
fi
echo "[done] task=${TASK_NAME} result=${RESULT_FILE}" >&2
