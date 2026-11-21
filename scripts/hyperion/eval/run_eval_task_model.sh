#!/bin/bash
#SBATCH --job-name=mace-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=48gb
#SBATCH --time=3-00:00:00
#SBATCH --output=.slurm/%x-%A_%a.out
#SBATCH --error=.slurm/%x-%A_%a.err

set -euo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '"$NOTIFY" "⚠️ $TAG was terminated (cancelled or hit time limit)" || true; exit 143' TERM
"$NOTIFY" "🚀 $TAG started" || true

WORKDIR="${WORKDIR:-/scratch/asalem/repos/SDPO}"
LM_EVAL_DIR="${LM_EVAL_DIR:-/scratch/asalem/repos/lm-evaluation-harness}"
LM_EVAL_ENV="${LM_EVAL_ENV:-/scratch/asalem/lm-evaluation-harness-venv}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/eval_results/macereason_math}"
EVAL_TASKS_FILE="${EVAL_TASKS_FILE:-$WORKDIR/scripts/hyperion/eval/eval_tasks.sh}"
if [[ ! -r "$EVAL_TASKS_FILE" ]]; then
    "$NOTIFY" "ERROR: eval task list not found at $EVAL_TASKS_FILE" || true
    exit 1
fi
source "$EVAL_TASKS_FILE"

ENV_FILE="$WORKDIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    "$NOTIFY" "ERROR: .env not found at $ENV_FILE; set HF_TOKEN in this file." || true
    exit 1
fi
set -a
source "$ENV_FILE"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
    "$NOTIFY" "ERROR: HF_TOKEN is not set in $ENV_FILE" || true
    exit 1
fi

if [[ ! -d "$LM_EVAL_DIR" ]]; then
    "$NOTIFY" "ERROR: lm-evaluation-harness checkout not found at $LM_EVAL_DIR" || true
    exit 1
fi
if [[ ! -x "$LM_EVAL_ENV/bin/lm_eval" || ! -f "$LM_EVAL_ENV/bin/activate" ]]; then
    "$NOTIFY" "ERROR: lm-evaluation-harness environment not found at $LM_EVAL_ENV" || true
    exit 1
fi

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/asalem/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${EVAL_HF_CACHE:-}" ]]; then
    export HF_HUB_CACHE="$EVAL_HF_CACHE/hub"
    export HF_XET_CACHE="$EVAL_HF_CACHE/xet"
    unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
fi

module load CUDA/12.8.0

source "$LM_EVAL_ENV/bin/activate"

# A100 nodes in hitz-exclusive use AMD EPYC 75F3 (Zen 3) CPUs. Keep GCC
# matched to that CPU family, but prefer the system assembler: the GCCcore
# binutils 2.38 assembler crashes on these nodes, while system `as` works.
TRITON_GCC="/scicomp/builds/Rocky/8.7/Zen3/software/GCCcore/11.3.0/bin/gcc"
if [[ ! -x "$TRITON_GCC" || ! -x /usr/bin/as ]]; then
    "$NOTIFY" "ERROR: Zen3 GCC and system assembler required by Triton are unavailable." || true
    exit 1
fi
export CC="$TRITON_GCC"
export PATH="/usr/bin:$PATH"

mkdir -p "$RESULTS_DIR"
cd "$LM_EVAL_DIR"

model="${MODEL:-${1:-}}"
if [[ -z "$model" ]]; then
    "$NOTIFY" "ERROR: set MODEL or pass the model as the first argument." || true
    exit 1
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_index="$SLURM_ARRAY_TASK_ID"
    if [[ ! "$task_index" =~ ^[0-9]+$ ]] || (( task_index >= ${#EVAL_TASKS[@]} )); then
        "$NOTIFY" "ERROR: invalid array task index '$task_index' (expected 0-$(( ${#EVAL_TASKS[@]} - 1 )))." || true
        exit 1
    fi
    task="${EVAL_TASKS[$task_index]}"
else
    task="${2:-}"
fi
if [[ -z "$task" ]]; then
    "$NOTIFY" "ERROR: set SLURM_ARRAY_TASK_ID or pass the task as the second argument." || true
    exit 1
fi

model_name="${model##*/}"
model_output="$RESULTS_DIR/${model_name}"

"$NOTIFY" "⚙️ $TAG evaluating $model on $task" || true
"$LM_EVAL_ENV/bin/lm_eval" \
    --model hf \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,parallelize=True" \
    --tasks "$task" \
    --batch_size auto \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path "$model_output"

"$NOTIFY" "✅ $TAG finished successfully" || true
