#!/bin/bash
#SBATCH --job-name=sdpo-mace500
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail

NOTIFY=~/bin/slack_notify.py
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '$NOTIFY "⚠️ $TAG was terminated (cancelled or hit time limit)"; kill $pid 2>/dev/null; exit 143' TERM

notify_failure() {
    local rc=$?
    trap - ERR
    if ! $NOTIFY "❌ $TAG failed with exit code $rc"; then
        echo "Failed to send failure notification" >&2
    fi
    exit "$rc"
}
trap notify_failure ERR

ROOT_DIR="/gscratch5/users/asalem/RL"
SDPO_DIR="${SDPO_DIR:-$ROOT_DIR/SDPO}"
DATA_DIR="${DATA_DIR:-$SDPO_DIR/datasets/mAceReason-Math-500x14}"
SOURCE_DATASET="${SOURCE_DATASET:-ahmedselhady/mAceReason-Math}"
ENV_DIR="${SDPO_ENV:-$HOME/environments/sdpo}"
MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-Research-Reasoning-Qwen-1.5B}"
EXPERIMENT="${EXPERIMENT:-macereason-math-500x14-nemotron-1.5b}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SDPO_DIR/checkpoints/$EXPERIMENT}"

ENV_FILE="$SDPO_DIR/.env"
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

export USER="${USER:-$(id -un)}"
export TASK="mAceReason-Math-500x14"
export EXPERIMENT
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_USE_V1=1
export PYTHONBUFFERED=1



if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "SDPO environment not found at $ENV_DIR" >&2
    echo "Create it using the setup instructions in SDPO/README.md." >&2
    exit 1
fi

module load cuda/13.1
source "$ENV_DIR/bin/activate"
ulimit -c 0

cd "$SDPO_DIR"
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

$NOTIFY "🚀 $TAG Preparing 500 parallel problem IDs across all 14 languages from $SOURCE_DATASET"

python data/prepare_macereason_math.py \
    --dataset "$SOURCE_DATASET" \
    --output-dir "$DATA_DIR" \
    --max-train-groups 500 \
    --seed 42

$NOTIFY "🚀 $TAG Starting SDPO multilingual subset training"
$NOTIFY "🚀 $TAG Model: $MODEL_PATH"
$NOTIFY "🚀 $TAG Data: $DATA_DIR"
$NOTIFY "🚀 $TAG Experiment: $EXPERIMENT"
$NOTIFY "🚀 $TAG Checkpoint directory: $CHECKPOINT_DIR"

$NOTIFY "🚀 $TAG started training"
python -m verl.trainer.main_ppo --config-name sdpo \
    "vars.dir=$SDPO_DIR" \
    "vars.task=$TASK" \
    "data.train_files=[$DATA_DIR/train.parquet]" \
    "data.val_files=[$DATA_DIR/test.parquet]" \
    "data.train_batch_size=28" \
    "data.parallel_group_batching.enabled=True" \
    "data.parallel_group_batching.group_key=parallel_id" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.rollout.n=8" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=28" \
    "trainer.project_name=SDPO-multilingual" \
    "trainer.group_name=macereason-math-500" \
    "trainer.experiment_name=$EXPERIMENT" \
    "trainer.nnodes=1" \
    "trainer.n_gpus_per_node=2" \
    "trainer.total_epochs=1" \
    "trainer.save_freq=100" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=False" \
    "trainer.resume_mode=disable" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.max_actor_ckpt_to_keep=2"

$NOTIFY "✅ $TAG finished successfully"
