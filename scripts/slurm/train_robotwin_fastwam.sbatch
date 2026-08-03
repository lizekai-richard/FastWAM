#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --job-name=fastwam_robotwin_base
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_train/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_train/%x-%j.err

TASK_NAME="robotwin_clean_uncond_3cam_384_1e-4"
WANDB_GROUP="${WANDB_GROUP:-robotwin_clean}"
PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
source "${PROJECT_DIR}/scripts/slurm/train_robotwin_requeue_common.sh" "$@"
