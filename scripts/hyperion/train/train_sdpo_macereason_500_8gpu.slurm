#!/bin/bash
#SBATCH --job-name=mace500-sdpox8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=200gb
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/asalem/repos/SDPO/.slurm/%x-%j.out
#SBATCH --error=/scratch/asalem/repos/SDPO/.slurm/%x-%j.err

set -euo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"
TAG="$SLURM_JOB_NAME (#$SLURM_JOB_ID) on $(hostname)"
# Fires on scancel or time limit (Slurm sends SIGTERM first)
trap '"$NOTIFY" "⚠️ $TAG was terminated (cancelled or hit time limit)" || true; exit 143' TERM

notify_failure() {
    local rc=$?
    trap - ERR
    if ! "$NOTIFY" "❌ $TAG failed with exit code $rc"; then
        printf '%s\n' 'Failed to send failure notification' >&2
    fi
    exit "$rc"
}
trap notify_failure ERR

ROOT_DIR="/scratch/asalem/repos"
SDPO_DIR="${SDPO_DIR:-$ROOT_DIR/SDPO}"
# Each parallel group contains one example per language. A batch needs four groups.
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-500}"
if [[ ! "$MAX_TRAIN_GROUPS" =~ ^([4-9]|[1-9][0-9]+)$ ]]; then
    printf '%s\n' 'ERROR: MAX_TRAIN_GROUPS must be an integer of at least 4.' >&2
    exit 1
fi
# Use comma-separated language codes.
DEFAULT_LANGUAGES="bn,en,es,te,th,zh"
dataset_languages="${LANGUAGES-$DEFAULT_LANGUAGES}"
if [[ ! "$dataset_languages" =~ ^[a-z]{2,3}(,[a-z]{2,3})*$ ]]; then
    printf '%s\n' 'ERROR: LANGUAGES must be comma-separated lowercase language codes.' >&2
    exit 1
fi
IFS=, read -r -a language_codes <<< "$dataset_languages"
seen_languages=","
for language in "${language_codes[@]}"; do
    if [[ "$seen_languages" == *",$language,"* ]]; then
        printf 'ERROR: duplicate language code %s.\n' "$language" >&2
        exit 1
    fi
    seen_languages+="$language,"
done

# Use en to prefer successful English references, or any to use any successful language.
reference_language="${SOLUTION_REFERENCE_LANGUAGE-en}"
case "$reference_language" in
    en|any) ;;
    *)
        printf '%s\n' 'ERROR: SOLUTION_REFERENCE_LANGUAGE must be en or any.' >&2
        exit 1
        ;;
esac

# Keep four complete parallel groups in each training batch.
TRAIN_BATCH_SIZE=$((4 * ${#language_codes[@]}))
DATASET_TAG="${MAX_TRAIN_GROUPS}x${#language_codes[@]}"
if [[ "$dataset_languages" != "$DEFAULT_LANGUAGES" ]]; then
    DATASET_TAG+="-${dataset_languages//,/-}"
fi
DATA_DIR="${DATA_DIR:-$SDPO_DIR/datasets/mAceReason-Math-$DATASET_TAG}"
SOURCE_DATASET="${SOURCE_DATASET:-ahmedselhady/mAceReason-Math}"
CONTAINER_IMAGE="${SDPO_CONTAINER_IMAGE:-/scratch/asalem/SDPO-qwen35-vllm-0.19.1.sif}"
CONTAINER_ENV="${SDPO_CONTAINER_ENV:-/scratch/asalem/SDPO-qwen35-container-venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
EXPERIMENT="${EXPERIMENT:-macereason-math-${DATASET_TAG}-qwen-3.5-2b-8gpu}"
CACHE_DIR="${CACHE_DIR:-/scratch/asalem/.cache/huggingface}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SDPO_DIR/checkpoints/$EXPERIMENT}"

ENV_FILE="$SDPO_DIR/.env"
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

export USER="${USER:-$(id -un)}"
export TASK="mAceReason-Math-${DATASET_TAG}"
export EXPERIMENT
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# Ray needs 24 CPUs for the eight GPU workers, plus CPUs for the trainer.
RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
RAY_OBJECT_STORE_MEMORY=$((40 * 1024 * 1024 * 1024))
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# The CUDA job may inherit ROCr's device mask from the cluster environment.
# verl/Ray rejects it when CUDA_VISIBLE_DEVICES is also set.
unset ROCR_VISIBLE_DEVICES
if [[ -n "${SDPO_ENV:-}" ]]; then
    if [[ ! -f "$SDPO_ENV/bin/activate" ]]; then
        printf 'ERROR: SDPO environment not found at %s\n' "$SDPO_ENV" >&2
        exit 1
    fi
    module load CUDA/12.8.0
    source "$SDPO_ENV/bin/activate"
    export CC=/usr/bin/gcc
    export PATH="$SDPO_ENV/bin:/usr/bin:/bin:$PATH"
    python_cmd=(python)
else
    if [[ ! -r "$CONTAINER_IMAGE" || ! -f "$CONTAINER_ENV/pyvenv.cfg" ]]; then
        printf 'ERROR: Qwen3.5 runtime requires %s and the Python environment at %s.\n' "$CONTAINER_IMAGE" "$CONTAINER_ENV" >&2
        exit 1
    fi
    # Hyperion's host glibc is older than the vLLM wheel requires.
    module load Apptainer/1.4.5
    python_cmd=(
        apptainer exec --nv --bind /scratch
        --env CUDA_HOME=/usr/local/cuda,CC=/usr/bin/gcc
        "$CONTAINER_IMAGE" "$CONTAINER_ENV/bin/python"
    )
fi
ulimit -c 0

cd "$SDPO_DIR"
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

"$NOTIFY" "Preparing dataset: source=$SOURCE_DATASET output=$DATA_DIR max_train_groups=$MAX_TRAIN_GROUPS languages=$dataset_languages seed=42" || true

"${python_cmd[@]}" data/prepare_macereason_math.py \
    --dataset "$SOURCE_DATASET" \
    --output-dir "$DATA_DIR" \
    --languages "$dataset_languages" \
    --max-train-groups "$MAX_TRAIN_GROUPS" \
    --seed 42
"$NOTIFY" "Dataset preparation complete" || true

"$NOTIFY" "🚀 $TAG Starting SDPO multilingual subset training" || true
"$NOTIFY" "🚀 $TAG Model: $MODEL_PATH" || true
"$NOTIFY" "🚀 $TAG Data: $DATA_DIR" || true
"$NOTIFY" "🚀 $TAG Experiment: $EXPERIMENT" || true
"$NOTIFY" "🚀 $TAG Checkpoint directory: $CHECKPOINT_DIR" || true

"$NOTIFY" "Launching verl.trainer.main_ppo; live training progress follows" || true
"$NOTIFY" "Ray resources: cpus=$RAY_NUM_CPUS object_store_memory=${RAY_OBJECT_STORE_MEMORY}B" || true
"${python_cmd[@]}" -m verl.trainer.main_ppo --config-name sdpo \
    "vars.dir=$SDPO_DIR" \
    "vars.task=$TASK" \
    "data.train_files=[$DATA_DIR/train.parquet]" \
    "data.val_files=[$DATA_DIR/test.parquet]" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
    "data.parallel_group_batching.enabled=True" \
    "data.parallel_group_batching.group_key=parallel_id" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.model.use_remove_padding=False" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.n=8" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE" \
    "actor_rollout_ref.actor.self_distillation.solution_reference_language=$reference_language" \
    "trainer.project_name=SDPO-multilingual" \
    "trainer.group_name=macereason-math-$MAX_TRAIN_GROUPS" \
    "trainer.experiment_name=$EXPERIMENT" \
    "trainer.nnodes=1" \
    "trainer.n_gpus_per_node=8" \
    "trainer.total_epochs=1" \
    "trainer.save_freq=25" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=False" \
    "trainer.resume_mode=disable" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
    "+ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY"

"$NOTIFY" "✅ $TAG finished successfully" || true
