#!/bin/bash
#SBATCH --job-name=mace-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=ahmed.salemmohamed@ehu.eus
#SBATCH --mail-type=FAIL
#SBATCH --output=.slurm/%x-%A_%a.out
#SBATCH --error=.slurm/%x-%A_%a.err

set -euo pipefail

NOTIFY=~/bin/slack_notify.py
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '$NOTIFY "⚠️ $TAG was terminated (cancelled or hit time limit)"; kill $pid 2>/dev/null; exit 143' TERM
$NOTIFY "🚀 $TAG started"

WORKDIR="${WORKDIR:-/scratch/asalem/repos/SDPO}"
LM_EVAL_DIR="${LM_EVAL_DIR:-/scratch/asalem/repos/lm-evaluation-harness}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/eval_results/macereason_math}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_tasks.sh"

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
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/asalem/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

module load CUDA/12.8.0

if [[ -f "$LM_EVAL_DIR/.venv/bin/activate" ]]; then
    # Use the harness checkout's environment when it has one.
    source "$LM_EVAL_DIR/.venv/bin/activate"
fi

# A100 nodes in hitz-exclusive use AMD EPYC 75F3 (Zen 3) CPUs. Keep GCC
# matched to that CPU family, but prefer the system assembler: the GCCcore
# binutils 2.38 assembler crashes on these nodes, while system `as` works.
TRITON_GCC="/scicomp/builds/Rocky/8.7/Zen3/software/GCCcore/11.3.0/bin/gcc"
if [[ ! -x "$TRITON_GCC" || ! -x /usr/bin/as ]]; then
    echo "ERROR: Zen3 GCC and system assembler required by Triton are unavailable." >&2
    exit 1
fi
export CC="$TRITON_GCC"
export PATH="/usr/bin:$PATH"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "ERROR: lm_eval is not on PATH. Install lm-evaluation-harness or activate its environment." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$LM_EVAL_DIR"

model="${MODEL:-${1:-}}"
if [[ -z "$model" ]]; then
    echo "ERROR: set MODEL or pass the model as the first argument." >&2
    exit 1
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_index="$SLURM_ARRAY_TASK_ID"
    if [[ ! "$task_index" =~ ^[0-9]+$ ]] || (( task_index >= ${#EVAL_TASKS[@]} )); then
        echo "ERROR: invalid array task index '$task_index' (expected 0-$(( ${#EVAL_TASKS[@]} - 1 )))." >&2
        exit 1
    fi
    task="${EVAL_TASKS[$task_index]}"
else
    task="${2:-}"
fi
if [[ -z "$task" ]]; then
    echo "ERROR: set SLURM_ARRAY_TASK_ID or pass the task as the second argument." >&2
    exit 1
fi

model_name="${model##*/}"
model_output="$RESULTS_DIR/${model_name}"

echo "Evaluating $model on $task"
$NOTIFY "🚀 $TAG evaluating $model on $task"
lm_eval \
    --model hf \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True,parallelize=True" \
    --tasks "$task" \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path "$model_output"

$NOTIFY "✅ $TAG finished successfully"
