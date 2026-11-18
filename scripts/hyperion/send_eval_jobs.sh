#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Load the user's sbatch aliases (including `sr`) from .bashrc. The shell is
# non-interactive here, so enable alias expansion explicitly.
shopt -s expand_aliases
source "$HOME/.bashrc"

# .bashrc changes to /scratch/asalem; submit from this script's directory so
# the relative Slurm script and output paths resolve consistently.
cd "$SCRIPT_DIR"

models=(
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-9B"
)

tasks=(
"macereason_math_en"
  "macereason_math_bn"
  "macereason_math_de"
  "macereason_math_es"
  "macereason_math_fr"
  "macereason_math_it"
  "macereason_math_ja"
  "macereason_math_ko"
  "macereason_math_pt"
  "macereason_math_ru"
  "macereason_math_sw"
  "macereason_math_te"
  "macereason_math_th"
  "macereason_math_zh"
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        sr run_eval_task_model.sh "$model" "$task"
    done
done
