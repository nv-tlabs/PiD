#!/usr/bin/env bash
# Multi-node Slurm example: PiD v1.5 FLUX teacher, 2 nodes x 4 GPUs.
# Slurm launches one task per GPU; each task starts one training process.
#
# Before submitting, make sure WANDB_API_KEY and HF_TOKEN are available in the
# environment if the run needs them:
#   sbatch scripts/multinode_slurm.sh

#SBATCH --job-name=pid-teacher-flux-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=batch_long,batch
#SBATCH --account=nvr_torontoai_videogen
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --wckey=p0
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_videogen/users/yiflu/workspace/pid/imaginaire4/logs/slurm-%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_videogen/users/yiflu/workspace/pid/imaginaire4/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_videogen/users/yiflu/workspace/pid"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_videogen/users/yiflu/workspace/pid/docker/pid_docker_arm64.sqsh"
CONFIG_FILE="pid/_src/configs/pid_training/config.py"
EXPERIMENT="pid_v1pt5_teacher_flux_h1024_d4_fix_backbone_res_2048"
JOB_GROUP="multinode_slurm"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export WORLD_SIZE="$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
export SLURM_LOG_DIR="${REPO_ROOT}/imaginaire4/logs/${SLURM_JOB_NAME}/${SLURM_JOB_ID}"

export IMAGINAIRE_OUTPUT_ROOT="${REPO_ROOT}/imaginaire4/imaginaire4-output"
export IMAGINAIRE_CACHE_DIR="${REPO_ROOT}/imaginaire4/imaginaire4-cache"
export HF_HOME="${IMAGINAIRE_CACHE_DIR}/huggingface"
export TORCH_HOME="${IMAGINAIRE_CACHE_DIR}"
export WANDB_CACHE_DIR="${IMAGINAIRE_CACHE_DIR}/wandb/cache"
export WANDB_DATA_DIR="${IMAGINAIRE_CACHE_DIR}/wandb/data"
export WANDB_ENTITY="nvidia-toronto"

export PYTHONPATH="${REPO_ROOT}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

RUN_NAME="${EXPERIMENT}_slurm_${SLURM_JOB_ID}"

mkdir -p \
  "${IMAGINAIRE_OUTPUT_ROOT}" \
  "${IMAGINAIRE_CACHE_DIR}" \
  "${HF_HOME}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${SLURM_LOG_DIR}"

export REPO_ROOT CONFIG_FILE EXPERIMENT JOB_GROUP RUN_NAME

echo "Nodes:         ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "Tasks/node:    ${SLURM_NTASKS_PER_NODE} (one task per GPU)"
echo "World size:    ${WORLD_SIZE}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Experiment:    ${EXPERIMENT}"
echo "Output root:   ${IMAGINAIRE_OUTPUT_ROOT}"

# This cluster's Slurm/Pyxis setup launches one task per GPU. Map the Slurm task
# ids explicitly to torch.distributed's env:// variables, then start one Python
# training process in each task.
srun \
  --kill-on-bad-exit=1 \
  --output="${SLURM_LOG_DIR}/rank-%t.out" \
  --error="${SLURM_LOG_DIR}/rank-%t.err" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/lustre:/lustre:rw,/run/shm:/run/shm:rw" \
  bash -lc '
    set -euo pipefail
    cd "${REPO_ROOT}"

    export RANK="${SLURM_PROCID}"
    export LOCAL_RANK="${SLURM_LOCALID}"

    PYTHONPATH=. python -m scripts.train \
      --config="${CONFIG_FILE}" \
      -- \
      experiment="${EXPERIMENT}" \
      job.group="${JOB_GROUP}" \
      job.name="${RUN_NAME}"
  '
