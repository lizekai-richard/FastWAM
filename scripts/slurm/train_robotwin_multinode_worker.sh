#!/usr/bin/env bash

set -euo pipefail

: "${TASK_NAME:?TASK_NAME must be set}"
: "${PROJECT_DIR:?PROJECT_DIR must be set}"
: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:?MASTER_PORT must be set}"

export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}"
export PRECOMPUTE_TEXT_EMBEDS="${PRECOMPUTE_TEXT_EMBEDS:-0}"

echo "[multinode-worker] node=$(hostname) node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}" >&2

source "${PROJECT_DIR}/scripts/slurm/train_robotwin_requeue_common.sh" "$@"
