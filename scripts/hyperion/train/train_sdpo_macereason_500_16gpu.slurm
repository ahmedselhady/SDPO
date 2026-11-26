#!/bin/bash
#SBATCH --job-name=mace500-sdpox16gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=400gb
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/asalem/repos/SDPO/.slurm/%x-%j.out
#SBATCH --error=/scratch/asalem/repos/SDPO/.slurm/%x-%j.err

set -euo pipefail

export SDPO_NNODES="$SLURM_JOB_NUM_NODES"
export ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-4}"
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
exec bash "${SDPO_DIR:-/scratch/asalem/repos/SDPO}/scripts/hyperion/train/train_sdpo_macereason_500_8gpu.slurm"
