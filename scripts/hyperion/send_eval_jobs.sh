#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_tasks.sh"

# Load the user's sbatch aliases (including `sr`) from .bashrc. The shell is
# non-interactive here, so enable alias expansion explicitly.
shopt -s expand_aliases
source "$HOME/.bashrc"

# .bashrc changes to /scratch/asalem; submit from this script's directory so
# the relative Slurm script and output paths resolve consistently.
cd "$SCRIPT_DIR"
mkdir -p .slurm

models=(
    # "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-9B"
)

for model in "${models[@]}"; do
    array_range="0-$(( ${#EVAL_TASKS[@]} - 1 ))"
    echo "Submitting ${#EVAL_TASKS[@]} evaluation tasks for $model (array $array_range)"
    sr --array="$array_range" --export="ALL,MODEL=$model" run_eval_task_model.sh
done
