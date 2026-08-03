#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --job-name=robotwin_eval_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_eval_smoke/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/logs/FastWAM/robotwin_eval_smoke/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/zekail/FastWAM}"
STORAGE="${STORAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail}"
ENV_PREFIX="${ENV_PREFIX:-${STORAGE}/miniforge3/envs/fastwam}"

TASK="${TASK:-robotwin_clean_joint_3cam_384_1e-4}"
CKPT="${CKPT:-${STORAGE}/runs/FastWAM/robotwin_clean_joint_3cam_384_1e-4/slurm_requeue/checkpoints/weights/step_021040.pt}"
DATASET_STATS="${DATASET_STATS:-${STORAGE}/runs/FastWAM/robotwin_clean_joint_3cam_384_1e-4/slurm_requeue/dataset_stats.json}"
ROBOTWIN_TASK="${ROBOTWIN_TASK:-click_alarmclock}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
export ROBOTWIN_TASK ROBOTWIN_TASK_CONFIG

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

cd "${PROJECT_DIR}"

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

cd "${PROJECT_DIR}/third_party/RoboTwin"
python - <<'PY'
import sys
import os
import yaml
sys.path.append("./")
from envs import CONFIGS_PATH
from envs.robot.planner import CuroboPlanner
from script.eval_policy import class_decorator, get_embodiment_config
print("CuroboPlanner import OK", CuroboPlanner)

task_name = os.environ.get("ROBOTWIN_TASK", "click_alarmclock")
task_config = os.environ.get("ROBOTWIN_TASK_CONFIG", "demo_clean")
with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)
args["task_name"] = task_name
args["task_config"] = task_config
args["policy_name"] = "fastwam_policy"

with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
    embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

def get_embodiment_file(embodiment_type):
    robot_file = embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise RuntimeError("No embodiment files")
    return robot_file

embodiment_type = args["embodiment"]
if len(embodiment_type) == 1:
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
elif len(embodiment_type) == 3:
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
    args["embodiment_dis"] = embodiment_type[2]
    args["dual_arm_embodied"] = False
else:
    raise RuntimeError("embodiment items should be 1 or 3")

args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
env = class_decorator(task_name)
env.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
env.close_env()
print("RoboTwin setup_demo preflight OK")
PY
python script/test_render.py

cd "${PROJECT_DIR}"
python experiments/robotwin/eval_robotwin_single.py \
  task="${TASK}" \
  ckpt="${CKPT}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS}" \
  EVALUATION.task_name="${ROBOTWIN_TASK}" \
  EVALUATION.task_config="${ROBOTWIN_TASK_CONFIG}" \
  EVALUATION.eval_num_episodes="${EVAL_EPISODES}" \
  EVALUATION.num_inference_steps="${INFERENCE_STEPS}" \
  gpu_id=0
