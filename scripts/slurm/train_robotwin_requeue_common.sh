#!/usr/bin/env bash

set -euo pipefail

: "${TASK_NAME:?TASK_NAME must be set by the sbatch wrapper}"

PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
STORAGE="${STORAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail}"
ENV_PREFIX="${ENV_PREFIX:-${STORAGE}/fastwam-env}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-${ENV_PREFIX}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
PRECOMPUTE_TEXT_EMBEDS="${PRECOMPUTE_TEXT_EMBEDS:-1}"
PRECOMPUTE_GPUS="${PRECOMPUTE_GPUS:-${NPROC_PER_NODE}}"
RESUME_CKPT="${RESUME_CKPT:-${STORAGE}/models/fastwam/robotwin_uncond_3cam_384.pt}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${STORAGE}/data/RoboTwin2.0-lerobotv3.0}"
ROBOTWIN_NORM_STATS="${ROBOTWIN_NORM_STATS:-${ROBOTWIN_DATA_ROOT}/_pooled_stats/clean_50tasks_fastwam_dataset_stats.json}"
RUN_ID="${RUN_ID:-slurm_requeue}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-1}"
REQUEUE_SIGNAL_WAIT_SECONDS="${REQUEUE_SIGNAL_WAIT_SECONDS:-780}"
RUNS_ROOT="${RUNS_ROOT:-${STORAGE}/runs/FastWAM}"
MIN_STATE_FILES="${MIN_STATE_FILES:-21}"

TRAIN_PID=""
REQUEUE_REQUESTED=0

cd "${PROJECT_DIR}"
mkdir -p logs

RUN_ROOT="${RUNS_ROOT}/${TASK_NAME}/${RUN_ID}"
STATE_ROOT="${RUN_ROOT}/checkpoints/state"

latest_state_checkpoint() {
  if [[ ! -d "${STATE_ROOT}" ]]; then
    return 0
  fi

  local checkpoint
  while IFS= read -r checkpoint; do
    local file_count=0
    local state_file
    local readable=1

    while IFS= read -r -d '' state_file; do
      file_count=$((file_count + 1))
      if [[ ! -s "${state_file}" ]] || ! head -c 1 "${state_file}" >/dev/null 2>&1; then
        echo "[resume] skipping unreadable state file: ${state_file}" >&2
        readable=0
        break
      fi
    done < <(find "${checkpoint}" -type f -print0 2>/dev/null)

    if [[ "${readable}" == "1" && "${file_count}" -ge "${MIN_STATE_FILES}" ]]; then
      printf '%s\n' "${checkpoint}"
      return 0
    fi
    echo "[resume] skipping incomplete state checkpoint: ${checkpoint} (files=${file_count})" >&2
  done < <(
    find "${STATE_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' \
      -exec test -f '{}/trainer_state.json' ';' -print 2>/dev/null \
      | sort -Vr
  )
}

signal_process_tree() {
  local pid="$1"
  local sig="$2"

  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  local children
  children="$(pgrep -P "${pid}" 2>/dev/null || true)"
  local child
  for child in ${children}; do
    signal_process_tree "${child}" "${sig}"
  done

  kill -s "${sig}" "${pid}" 2>/dev/null || true
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

signal_training_workers() {
  local sig="$1"
  local matched=0
  local pid

  if [[ -z "${TRAIN_PID}" ]] || ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    return 0
  fi

  for pid in $(descendant_pids "${TRAIN_PID}"); do
    local args
    local parent_args
    local ppid
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    ppid="$(ps -p "${pid}" -o ppid= 2>/dev/null | tr -d '[:space:]')"
    parent_args="$(ps -p "${ppid}" -o args= 2>/dev/null || true)"

    # Signal only the rank processes launched directly by Accelerate. The
    # launcher and DataLoader workers also contain scripts/train.py in their
    # command line, but terminating either prevents a graceful checkpoint.
    if [[ "${args}" == *"scripts/train.py"* && "${parent_args}" == *"accelerate launch"* ]]; then
      matched=1
      echo "[requeue] sending ${sig} to training rank PID ${pid}" >&2
      kill -s "${sig}" "${pid}" 2>/dev/null || true
    fi
  done

  if [[ "${matched}" == "0" ]]; then
    echo "[requeue] no scripts/train.py worker process found under PID ${TRAIN_PID}" >&2
  fi
}

wait_for_training_exit() {
  local deadline=$((SECONDS + REQUEUE_SIGNAL_WAIT_SECONDS))

  while [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "[requeue] training did not exit after SIGUSR1; sending SIGTERM" >&2
      signal_process_tree "${TRAIN_PID}" TERM
      sleep 30
      if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "[requeue] training still running after SIGTERM; sending SIGKILL" >&2
        signal_process_tree "${TRAIN_PID}" KILL
      fi
      break
    fi
    sleep 5
  done

  if [[ -n "${TRAIN_PID}" ]]; then
    wait "${TRAIN_PID}" || true
  fi
}

request_requeue() {
  trap - USR1
  REQUEUE_REQUESTED=1

  echo "[requeue] received USR1 for job ${SLURM_JOB_ID:-unknown}; requesting graceful checkpoint" >&2
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    signal_training_workers USR1
    wait_for_training_exit
  fi

  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[requeue] SLURM_JOB_ID is not set; cannot requeue" >&2
    exit 1
  fi

  local node_rank="${NODE_RANK:-${SLURM_PROCID:-0}}"
  if [[ "${node_rank}" == "0" ]]; then
    echo "[requeue] requeueing job ${SLURM_JOB_ID}" >&2
    scontrol requeue "${SLURM_JOB_ID}"
  else
    echo "[requeue] node_rank=${node_rank}; rank 0 will requeue job ${SLURM_JOB_ID}" >&2
  fi
  exit 0
}

trap request_requeue USR1

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
test -x "${ENV_PREFIX}/bin/python"
test -x "${FFMPEG_PREFIX}/bin/ffmpeg"

export PATH="${FFMPEG_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
"${ENV_PREFIX}/bin/python" -c \
  'import torch, torchcodec; print(f"[video] torch={torch.__version__} torchcodec={torchcodec.__version__}")'

CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-}"
if [[ -z "${CUDA_TOOLKIT_ROOT}" ]]; then
  for candidate in \
    /cm/shared/apps/cuda-latest/toolkit/* \
    /cm/shared/apps/cuda*/toolkit/* \
    /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      CUDA_TOOLKIT_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CUDA_TOOLKIT_ROOT}" || ! -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ]]; then
  echo "[cuda] CUDA toolkit not found; set CUDA_TOOLKIT_ROOT to a valid toolkit path" >&2
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-${CUDA_TOOLKIT_ROOT}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
echo "[cuda] toolkit=${CUDA_HOME}" >&2

export PYTHONUNBUFFERED=1
export STORAGE
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${STORAGE}/models/diffsynth}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE}/.cache}"
export HF_HOME="${HF_HOME:-${STORAGE}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton-autotune}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${XDG_CACHE_HOME}/pycache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${RUNS_ROOT}/wandb}"
if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_TOKEN:-}" ]]; then
  export WANDB_API_KEY="${WANDB_TOKEN}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_ID
export FASTWAM_OUTPUT_DIR="${RUN_ROOT}"

mkdir -p \
  "${RUN_ROOT}" \
  "${WANDB_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${PIP_CACHE_DIR}" \
  "${MPLCONFIGDIR}"

test -d "${ROBOTWIN_DATA_ROOT}"
test -f "${ROBOTWIN_NORM_STATS}"
if [[ ! -f "${RESUME_CKPT}" && ! -d "${RESUME_CKPT}" ]]; then
  echo "[resume] RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
  exit 1
fi

CLEAN_TASK_COUNT="$(
  find "${ROBOTWIN_DATA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name 'aloha-agilex_clean_50' | wc -l
)"
if [[ "${CLEAN_TASK_COUNT}" -lt 1 ]]; then
  echo "[data] no clean LeRobot task directories found under ${ROBOTWIN_DATA_ROOT}" >&2
  exit 1
fi
echo "[data] clean_lerobot_dirs=${CLEAN_TASK_COUNT} stats=${ROBOTWIN_NORM_STATS}" >&2
echo "[data] num_workers_per_rank=${NUM_WORKERS}" >&2

LATEST_STATE="$(latest_state_checkpoint)"
if [[ -n "${LATEST_STATE}" ]]; then
  TRAIN_RESUME="${LATEST_STATE}"
  PRECOMPUTE_TEXT_EMBEDS=0
  echo "[resume] using latest training state: ${TRAIN_RESUME}" >&2
else
  if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
    echo "[resume] no complete training state found under ${STATE_ROOT} after requeue; refusing to restart from initial weights" >&2
    exit 1
  fi
  TRAIN_RESUME="${RESUME_CKPT}"
  echo "[resume] no training state found; loading initial weights: ${TRAIN_RESUME}" >&2
fi

WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-${RUN_ROOT}/wandb_run_id}"
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
elif [[ -s "${WANDB_RUN_ID_FILE}" ]]; then
  WANDB_RUN_ID="$(tr -d '[:space:]' < "${WANDB_RUN_ID_FILE}")"
else
  WANDB_RUN_ID_SEED="${TASK_NAME}-${RUN_ID}-${SLURM_JOB_ID:-manual}-$(date +%Y%m%d%H%M%S)"
  WANDB_RUN_ID="fw-$(printf '%s' "${WANDB_RUN_ID_SEED}" | sha1sum | awk '{print substr($1, 1, 24)}')"
  printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
fi
export WANDB_RUN_ID
export WANDB_RESUME="${WANDB_RESUME:-allow}"
echo "[wandb] run_id=${WANDB_RUN_ID} resume=${WANDB_RESUME} id_file=${WANDB_RUN_ID_FILE}" >&2

if [[ "${PRECOMPUTE_TEXT_EMBEDS}" == "1" ]]; then
  torchrun --standalone --nproc_per_node="${PRECOMPUTE_GPUS}" \
    scripts/precompute_text_embeds.py \
    task="${TASK_NAME}" \
    +overwrite=false \
    model.redirect_common_files=false
fi

if [[ "${FASTWAM_PRECOMPUTE_ONLY:-0}" == "1" ]]; then
  echo "[precompute] FASTWAM_PRECOMPUTE_ONLY=1; skipping training launch" >&2
  exit 0
fi

set +e
bash scripts/train_zero2.sh "${NPROC_PER_NODE}" \
  task="${TASK_NAME}" \
  model.redirect_common_files=false \
  model.skip_dit_load_from_pretrain=true \
  model.action_dit_pretrained_path=null \
  resume="${TRAIN_RESUME}" \
  save_every="${SAVE_EVERY}" \
  num_workers="${NUM_WORKERS}" \
  wandb.enabled=true \
  wandb.project=fastwam \
  wandb.group="${WANDB_GROUP:-robotwin_joint_idm}" \
  wandb.id="${WANDB_RUN_ID}" \
  wandb.resume="${WANDB_RESUME}" \
  "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e

if [[ "${REQUEUE_REQUESTED}" == "1" ]]; then
  exit 0
fi

exit "${TRAIN_RC}"
