#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Load the user's sbatch aliases (including `sr`) from .bashrc. The shell is
# non-interactive here, so enable alias expansion explicitly.
shopt -s expand_aliases
source "$HOME/.bashrc"
NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"

TASKS_FILE="$SCRIPT_DIR/eval_tasks.sh"
if ! source "$TASKS_FILE" || (( ${#EVAL_TASKS[@]} == 0 )); then
    "$NOTIFY" "ERROR: could not load a nonempty evaluation task list from $TASKS_FILE" || true
    exit 1
fi

# .bashrc changes to /scratch/asalem; submit from this script's directory so
# the relative Slurm script and output paths resolve consistently.
cd "$SCRIPT_DIR"
mkdir -p .slurm

models=(
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
)

task_count="${#EVAL_TASKS[@]}"
array_range="0-$(( task_count - 1 ))"
for model in "${models[@]}"; do
    "$NOTIFY" "Submitting $task_count evaluation tasks for $model (array $array_range)" || true
    sb --array="$array_range" --export="ALL,MODEL=$model" run_eval_task_model.sh
done
