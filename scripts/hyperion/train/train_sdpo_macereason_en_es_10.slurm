#!/bin/bash
#SBATCH --job-name=mace-en-es-10-sdpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=100gb
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/asalem/repos/SDPO/.slurm/%x-%j.out
#SBATCH --error=/scratch/asalem/repos/SDPO/.slurm/%x-%j.err

set -euo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
printf 'Starting %s\n' "$TAG"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '"$NOTIFY" "⚠️ $TAG was terminated (cancelled or hit time limit)" || true; exit 143' TERM

notify_failure() {
    local rc=$?
    trap - ERR
    printf 'ERROR: %s failed with exit code %s\n' "$TAG" "$rc" >&2
    if ! "$NOTIFY" "❌ $TAG failed with exit code $rc"; then
        printf '%s\n' 'Failed to send failure notification' >&2
    fi
    exit "$rc"
}
trap notify_failure ERR

fail() {
    printf '%s\n' "$1" >&2
    "$NOTIFY" "$1" || true
    exit 1
}

ROOT_DIR="/scratch/asalem/repos"
SDPO_DIR="${SDPO_DIR:-$ROOT_DIR/SDPO}"
DATA_DIR="${DATA_DIR:-$SDPO_DIR/datasets/mAceReason-Math-en-es-10}"
SOURCE_DATASET="${SOURCE_DATASET:-ahmedselhady/mAceReason-Math}"
ENV_DIR="${SDPO_ENV:-/scratch/asalem/SDPO-venv}"
MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-Research-Reasoning-Qwen-1.5B}"
EXPERIMENT="${EXPERIMENT:-macereason-math-en-es-10-nemotron-1.5b}"
CACHE_DIR="${CACHE_DIR:-/scratch/asalem/.cache/huggingface}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SDPO_DIR/checkpoints/$EXPERIMENT}"

ENV_FILE="$SDPO_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    fail "ERROR: .env not found at $ENV_FILE; set HF_TOKEN in this file."
fi
set -a
source "$ENV_FILE"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
    fail "ERROR: HF_TOKEN is not set in $ENV_FILE"
fi

export USER="${USER:-$(id -un)}"
export TASK="mAceReason-Math-en-es-10"
export EXPERIMENT
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-8}"
RAY_OBJECT_STORE_MEMORY=$((20 * 1024 * 1024 * 1024))

if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    fail "ERROR: SDPO environment not found at $ENV_DIR. See the setup instructions in SDPO/README.md."
fi

module load CUDA/12.8.0
source "$ENV_DIR/bin/activate"
SDPO_PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$SDPO_PYTHON" ]] || ! "$SDPO_PYTHON" --version >/dev/null; then
    if [[ -L "$SDPO_PYTHON" ]]; then
        printf 'Python symlink target: %s\n' "$(readlink "$SDPO_PYTHON")" >&2
    fi
    fail "ERROR: $SDPO_PYTHON cannot run on $(hostname). Set SDPO_ENV to an environment with a compute-node Python."
fi
# The CUDA job may inherit ROCr's device mask from the cluster environment.
# verl/Ray rejects it when CUDA_VISIBLE_DEVICES is also set.
unset ROCR_VISIBLE_DEVICES
# A100 nodes in hitz-exclusive use AMD EPYC 75F3 (Zen 3) CPUs.
# Triton builds a small CUDA helper with the host C compiler on first use;
# keep the compiler and assembler matched to the node CPU family.
TRITON_GCC="/scicomp/builds/Rocky/8.7/Zen3/software/GCCcore/11.3.0/bin/gcc"
TRITON_BINUTILS="/scicomp/builds/Rocky/8.7/Zen3/software/binutils/2.38-GCCcore-11.3.0/bin"
if [[ ! -x "$TRITON_GCC" || ! -x "$TRITON_BINUTILS/as" ]]; then
    fail "ERROR: Zen3 GCC/binutils required by Triton are unavailable."
fi
export CC="$TRITON_GCC"
export PATH="$TRITON_BINUTILS:$PATH"
ulimit -c 0

cd "$SDPO_DIR"
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

"$NOTIFY" "Preparing dataset: source=$SOURCE_DATASET output=$DATA_DIR languages=en,es max_train_groups=10 seed=42" || true

"$SDPO_PYTHON" data/prepare_macereason_math.py \
    --dataset "$SOURCE_DATASET" \
    --output-dir "$DATA_DIR" \
    --languages en,es \
    --max-train-groups 10 \
    --seed 42
"$NOTIFY" "Dataset preparation complete" || true

"$NOTIFY" "🚀 $TAG Starting SDPO English/Spanish training (10 steps)" || true
"$NOTIFY" "🚀 $TAG Model: $MODEL_PATH" || true
"$NOTIFY" "🚀 $TAG Data: $DATA_DIR" || true
"$NOTIFY" "🚀 $TAG Experiment: $EXPERIMENT" || true
"$NOTIFY" "🚀 $TAG Checkpoint directory: $CHECKPOINT_DIR" || true

"$NOTIFY" "Launching verl.trainer.main_ppo; live training progress follows" || true
"$NOTIFY" "Ray resources: cpus=$RAY_NUM_CPUS object_store_memory=${RAY_OBJECT_STORE_MEMORY}B" || true
"$SDPO_PYTHON" -m verl.trainer.main_ppo --config-name sdpo \
    "vars.dir=$SDPO_DIR" \
    "vars.task=$TASK" \
    "data.train_files=[$DATA_DIR/train.parquet]" \
    "data.val_files=[$DATA_DIR/test.parquet]" \
    "data.train_batch_size=2" \
    "data.parallel_group_batching.enabled=True" \
    "data.parallel_group_batching.group_key=parallel_id" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.rollout.n=8" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=2" \
    "trainer.project_name=SDPO-multilingual" \
    "trainer.group_name=macereason-math-en-es" \
    "trainer.experiment_name=$EXPERIMENT" \
    "trainer.nnodes=1" \
    "trainer.n_gpus_per_node=2" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=10" \
    "trainer.save_freq=100" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=False" \
    "trainer.resume_mode=disable" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
    "+ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY"

"$NOTIFY" "✅ $TAG finished successfully" || true
