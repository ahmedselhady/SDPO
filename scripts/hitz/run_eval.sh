#!/bin/bash
#SBATCH --job-name=macereason_math
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail


NOTIFY=~/bin/slack_notify.py
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '$NOTIFY "⚠️ $TAG was terminated (cancelled or hit time limit)"; kill $pid 2>/dev/null; exit 143' TERM
$NOTIFY "🚀 $TAG started"

WORKDIR="${WORKDIR:-/gscratch5/users/asalem/RL/SDPO}"
LM_EVAL_DIR="${LM_EVAL_DIR:-/gscratch5/users/asalem/repos/lm-evaluation-harness}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/eval_results/macereason_math}"

ENV_FILE="$WORKDIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE; set HF_TOKEN in this file." >&2
    exit 1
fi
set -a
source "$ENV_FILE"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in $ENV_FILE" >&2
    exit 1
fi

if [[ ! -d "$LM_EVAL_DIR" ]]; then
    echo "ERROR: lm-evaluation-harness checkout not found at $LM_EVAL_DIR" >&2
    exit 1
fi

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

if [[ -f "$LM_EVAL_DIR/.venv/bin/activate" ]]; then
    # Use the harness checkout's environment when it has one.
    source "$LM_EVAL_DIR/.venv/bin/activate"
fi

# The A100 hitz nodes are Zen3; use their matching compiler for Triton helpers.
TRITON_GCC="/scicomp/builds/Rocky/8.7/Zen3/software/GCCcore/11.3.0/bin/gcc"
TRITON_BINUTILS="/scicomp/builds/Rocky/8.7/Zen3/software/binutils/2.38-GCCcore-11.3.0/bin"
if [[ ! -x "$TRITON_GCC" || ! -x "$TRITON_BINUTILS/as" ]]; then
    echo "ERROR: Zen3 GCC/binutils required by Triton are unavailable." >&2
    exit 1
fi
export CC="$TRITON_GCC"
export PATH="$TRITON_BINUTILS:$PATH"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "ERROR: lm_eval is not on PATH. Install lm-evaluation-harness or activate its environment." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$LM_EVAL_DIR"


models=(
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
    # "Qwen/Qwen3.5-9B"
    # "Qwen/Qwen3.5-27B"
)

tasks=(
    "macereason_math_en"
    "macereason_math"
)

$NOTIFY "🚀 $TAG started"
for model in "${models[@]}"; do
    model_name="${model##*/}"
    model_output="$RESULTS_DIR/${model_name}"

    for task in "${tasks[@]}"; do
        echo "Evaluating $model on $task"
        $NOTIFY "🚀 $TAG evaluating $model on $task"
        lm_eval \
            --model hf \
            --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,parallelize=True" \
            --tasks "$task" \
            --batch_size auto \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --log_samples \
            --output_path "$model_output"
    done
done
$NOTIFY "✅ $TAG finished successfully"
