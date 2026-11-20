#!/bin/bash
#SBATCH --job-name=sdpo-qwen3-06b-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=100gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=ahmed.salemmohamed@ehu.eus
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/asalem/repos/SDPO/.slurm/%x-%j.out
#SBATCH --error=/scratch/asalem/repos/SDPO/.slurm/%x-%j.err

set -euo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
SDPO_DIR="/scratch/asalem/repos/SDPO"
ENV_DIR="${SDPO_ENV:-/scratch/asalem/SDPO-venv}"
DATA_DIR="$SDPO_DIR/datasets/mAceReason-Math-en-es-10"
MODEL_PATH="suayptalha/Qwen3-0.6B-Math-Expert"
EXPERIMENT="sdpo-qwen3-06b-math-expert-probe"
CACHE_DIR="${CACHE_DIR:-/scratch/asalem/.cache/huggingface}"
CHECKPOINT_DIR="$SDPO_DIR/checkpoints/$EXPERIMENT"

notify_failure() {
    local rc=$?
    trap - ERR
    if ! "$NOTIFY" "❌ $TAG failed with exit code $rc"; then
        printf '%s\n' 'Failed to send failure notification' >&2
    fi
    exit "$rc"
}
trap notify_failure ERR
trap '"$NOTIFY" "⚠️ $TAG was terminated" || true; exit 143' TERM

if [[ ! -f "$SDPO_DIR/.env" ]]; then
    "$NOTIFY" "ERROR: .env not found at $SDPO_DIR/.env; set HF_TOKEN in this file." || true
    exit 1
fi
set -a
source "$SDPO_DIR/.env"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
    "$NOTIFY" "ERROR: HF_TOKEN is not set in $SDPO_DIR/.env" || true
    exit 1
fi
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    "$NOTIFY" "ERROR: prepared probe data is missing from $DATA_DIR" || true
    exit 1
fi
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    "$NOTIFY" "ERROR: SDPO environment not found at $ENV_DIR" || true
    exit 1
fi

export USER="${USER:-$(id -un)}"
export TASK="mAceReason-Math-en-es-10"
export EXPERIMENT
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

module load CUDA/12.8.0
source "$ENV_DIR/bin/activate"
TRITON_GCC="/scicomp/builds/Rocky/8.7/Zen3/software/GCCcore/11.3.0/bin/gcc"
TRITON_BINUTILS="/scicomp/builds/Rocky/8.7/Zen3/software/binutils/2.38-GCCcore-11.3.0/bin"
if [[ ! -x "$TRITON_GCC" || ! -x "$TRITON_BINUTILS/as" ]]; then
    "$NOTIFY" "ERROR: Zen3 GCC/binutils required by Triton are unavailable." || true
    exit 1
fi
export CC="$TRITON_GCC"
export PATH="$TRITON_BINUTILS:$PATH"
ulimit -c 0

cd "$SDPO_DIR"
mkdir -p "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
"$NOTIFY" "Probe job $TAG" || true
"$NOTIFY" "Model: $MODEL_PATH" || true
"$NOTIFY" "Data: $DATA_DIR" || true
"$NOTIFY" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" || true
"$NOTIFY" "🚀 $TAG Starting one-step SDPO probe with $MODEL_PATH" || true

python -m verl.trainer.main_ppo --config-name sdpo \
    "vars.dir=$SDPO_DIR" \
    "vars.task=$TASK" \
    "data.train_files=[$DATA_DIR/train.parquet]" \
    "data.val_files=[$DATA_DIR/test.parquet]" \
    "data.train_batch_size=2" \
    "data.max_prompt_length=2048" \
    "data.max_response_length=1024" \
    "data.parallel_group_batching.enabled=True" \
    "data.parallel_group_batching.group_key=parallel_id" \
    "max_model_len=8192" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.rollout.n=1" \
    "actor_rollout_ref.rollout.max_model_len=8192" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=2" \
    "actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096" \
    "trainer.project_name=SDPO-probe" \
    "trainer.group_name=qwen3-06b-math-expert-probe" \
    "trainer.experiment_name=$EXPERIMENT" \
    "trainer.nnodes=1" \
    "trainer.n_gpus_per_node=1" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=1" \
    "trainer.save_freq=-1" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=False" \
    "trainer.resume_mode=disable" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.max_actor_ckpt_to_keep=1"

"$NOTIFY" "✅ $TAG completed the one-step SDPO probe" || true
