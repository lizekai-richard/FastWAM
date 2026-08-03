#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --job-name=fastwam_robotwin_idm
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --requeue
#SBATCH --signal=USR1@900
#SBATCH --open-mode=append
#SBATCH --output=/home/zekail/FastWAM/logs/%x-%j.out
#SBATCH --error=/home/zekail/FastWAM/logs/%x-%j.err

TASK_NAME="robotwin_clean_idm_3cam_384_1e-4"
PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
source "${PROJECT_DIR}/scripts/slurm/train_robotwin_multinode_common.sh" "$@"
