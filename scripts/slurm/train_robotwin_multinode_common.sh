#!/usr/bin/env bash

set -euo pipefail

: "${TASK_NAME:?TASK_NAME must be set by the sbatch wrapper}"

PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
STORAGE="${STORAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail}"
RUN_ID="${RUN_ID:-slurm_requeue}"
RUNS_ROOT="${RUNS_ROOT:-${STORAGE}/runs/FastWAM}"
NNODES="${NNODES:-${SLURM_NNODES:-2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
PRECOMPUTE_TEXT_EMBEDS="${PRECOMPUTE_TEXT_EMBEDS:-1}"
PRECOMPUTE_GPUS="${PRECOMPUTE_GPUS:-${NPROC_PER_NODE}}"

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "[multinode] SLURM_JOB_NODELIST is not set; submit this script through sbatch" >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
RUN_ROOT="${RUNS_ROOT}/${TASK_NAME}/${RUN_ID}"
STATE_ROOT="${RUN_ROOT}/checkpoints/state"
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-${RUN_ROOT}/wandb_run_id}"

latest_state_checkpoint() {
  if [[ ! -d "${STATE_ROOT}" ]]; then
    return 0
  fi

  find "${STATE_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' \
    -exec test -f '{}/trainer_state.json' ';' -print \
    | sort -V \
    | tail -n 1
}

if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  mkdir -p "${RUN_ROOT}"
  printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
elif [[ -n "$(latest_state_checkpoint)" && -s "${WANDB_RUN_ID_FILE}" ]]; then
  WANDB_RUN_ID="$(tr -d '[:space:]' < "${WANDB_RUN_ID_FILE}")"
else
  mkdir -p "${RUN_ROOT}"
  WANDB_RUN_ID_SEED="${TASK_NAME}-${RUN_ID}-${SLURM_JOB_ID}-$(date +%Y%m%d%H%M%S)"
  WANDB_RUN_ID="fw-$(printf '%s' "${WANDB_RUN_ID_SEED}" | sha1sum | awk '{print substr($1, 1, 24)}')"
  printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
fi

export PROJECT_DIR
export STORAGE
export RUN_ID
export RUNS_ROOT
export NNODES
export NPROC_PER_NODE
export MASTER_ADDR
export MASTER_PORT
export TASK_NAME
export WANDB_RUN_ID
export WANDB_RUN_ID_FILE

cd "${PROJECT_DIR}"
mkdir -p logs

echo "[multinode] task=${TASK_NAME} nodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} run_id=${RUN_ID}" >&2
echo "[multinode] run_root=${RUN_ROOT} wandb_run_id=${WANDB_RUN_ID}" >&2

if [[ "${PRECOMPUTE_TEXT_EMBEDS}" == "1" ]]; then
  echo "[multinode] running text embedding precompute once on $(hostname)" >&2
  FASTWAM_PRECOMPUTE_ONLY=1 \
  PRECOMPUTE_TEXT_EMBEDS=1 \
  PRECOMPUTE_GPUS="${PRECOMPUTE_GPUS}" \
  NNODES=1 \
  NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 \
  bash -lc 'source "${PROJECT_DIR}/scripts/slurm/train_robotwin_requeue_common.sh" "$@"' bash "$@"
fi

export PRECOMPUTE_TEXT_EMBEDS=0

srun \
  --label \
  --kill-on-bad-exit=1 \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  bash "${PROJECT_DIR}/scripts/slurm/train_robotwin_multinode_worker.sh" "$@"
