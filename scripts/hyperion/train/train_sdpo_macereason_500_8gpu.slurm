#!/bin/bash
#SBATCH --job-name=mace500-sdpox8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=400gb
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
# Each parallel group contains one example per language.
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-500}"
TRAIN_GROUPS_PER_BATCH="${TRAIN_GROUPS_PER_BATCH:-4}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"
NNODES="${SDPO_NNODES:-1}"
GPUS_PER_NODE=8
for setting in MAX_TRAIN_GROUPS TRAIN_GROUPS_PER_BATCH ROLLOUTS_PER_PROMPT MAX_RESPONSE_TOKENS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'ERROR: %s must be a positive integer.\n' "$setting" >&2
        exit 1
    fi
done
if (( MAX_TRAIN_GROUPS < TRAIN_GROUPS_PER_BATCH || ROLLOUTS_PER_PROMPT < 2 )); then
    printf '%s\n' 'ERROR: The dataset must cover one full batch, and GRPO needs at least two rollouts per prompt.' >&2
    exit 1
fi
# Use comma-separated language codes.
DEFAULT_LANGUAGES="bn,en,es,te,th,zh" # "bn,de,en,es,fr,it,ja,ko,pt,ru,sw,te,th,zh"
dataset_languages="${LANGUAGES-$DEFAULT_LANGUAGES}"
if [[ "$dataset_languages" == all ]]; then
    dataset_languages="bn,de,en,es,fr,it,ja,ko,pt,ru,sw,te,th,zh"
fi
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

reference_language="${SOLUTION_REFERENCE_LANGUAGE:-en_only}"
if [[ "$reference_language" != en_only && "$reference_language" != any ]]; then
    printf '%s\n' 'ERROR: SOLUTION_REFERENCE_LANGUAGE must be en_only or any.' >&2
    exit 1
fi
if [[ "$reference_language" == en_only && "$seen_languages" != *,en,* ]]; then
    printf '%s\n' 'ERROR: SOLUTION_REFERENCE_LANGUAGE=en_only requires en in LANGUAGES.' >&2
    exit 1
fi

TRAIN_BATCH_SIZE=$((TRAIN_GROUPS_PER_BATCH * ${#language_codes[@]}))
ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUTS_PER_PROMPT))
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
if (( ROLLOUT_BATCH_SIZE % TOTAL_GPUS != 0 )); then
    printf 'ERROR: %s rollout samples per batch must be divisible by %s GPUs.\n' "$ROLLOUT_BATCH_SIZE" "$TOTAL_GPUS" >&2
    exit 1
fi
DATASET_TAG="${MAX_TRAIN_GROUPS}x${#language_codes[@]}"
if [[ "$dataset_languages" != "$DEFAULT_LANGUAGES" ]]; then
    DATASET_TAG+="-${dataset_languages//,/-}"
fi
DATA_DIR="${DATA_DIR:-$SDPO_DIR/datasets/mAceReason-Math-$DATASET_TAG}"
SOURCE_DATASET="${SOURCE_DATASET:-ahmedselhady/mAceReason-Math}"
CONTAINER_IMAGE="${SDPO_CONTAINER_IMAGE:-/scratch/asalem/SDPO-qwen35-vllm-0.19.1.sif}"
CONTAINER_ENV="${SDPO_CONTAINER_ENV:-/scratch/asalem/SDPO-qwen35-container-venv}"
MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-Research-Reasoning-Qwen-1.5B}"
TEACHER_PROMPT_TOKENS=10752
SDPO_ALPHA="${SDPO_ALPHA:-1}"
if [[ ! "$SDPO_ALPHA" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    printf 'ERROR: SDPO_ALPHA must be a decimal between 0 and 1.\n' >&2
    exit 1
fi
alpha_tag=""
if [[ "$SDPO_ALPHA" != 0.5 ]]; then
    alpha_tag="-alpha${SDPO_ALPHA}"
fi
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
case "${USE_KL_LOSS,,}" in
    true) kl_tag="kl${KL_LOSS_COEF}" ;;
    false) kl_tag="no-kl" ;;
    *)
        printf 'ERROR: USE_KL_LOSS must be True or False.\n' >&2
        exit 1
        ;;
esac
GRPO_ON_SUCCESS="${GRPO_ON_SUCCESS:-False}"
case "${GRPO_ON_SUCCESS,,}" in
    true) grpo_success_tag="-gs" ;;
    false) grpo_success_tag="" ;;
    *)
        printf 'ERROR: GRPO_ON_SUCCESS must be True or False.\n' >&2
        exit 1
        ;;
esac
batch_tag=""
model_tag="nemotron-1.5b"
if (( NNODES != 1 || TRAIN_GROUPS_PER_BATCH != 4 || ROLLOUTS_PER_PROMPT != 8 || MAX_RESPONSE_TOKENS != 8192 )); then
    batch_tag="-groups${TRAIN_GROUPS_PER_BATCH}-n${ROLLOUTS_PER_PROMPT}-resp${MAX_RESPONSE_TOKENS}"
    if [[ "$MODEL_PATH" != nvidia/Nemotron-Research-Reasoning-Qwen-1.5B ]]; then
        model_tag="${MODEL_PATH##*/}"
    fi
fi
EXPERIMENT="${EXPERIMENT:-macereason-math-${DATASET_TAG}-${model_tag}-${TOTAL_GPUS}gpu${batch_tag}-lr${LEARNING_RATE}-${kl_tag}-full-ref${TEACHER_PROMPT_TOKENS}-k512-frozen}"
CACHE_DIR="${CACHE_DIR:-/scratch/asalem/.cache/huggingface}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SDPO_DIR/checkpoints/$EXPERIMENT-$reference_language${alpha_tag}${grpo_success_tag}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$CHECKPOINT_DIR/rollouts}"
LOG_TEACHER_PROMPTS="${LOG_TEACHER_PROMPTS:-3}"

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
        printf 'ERROR: The container runtime requires %s and the Python environment at %s.\n' "$CONTAINER_IMAGE" "$CONTAINER_ENV" >&2
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

ray_steps=()
stop_sdpo_ray_cluster() {
    local step
    for step in "${ray_steps[@]}"; do
        kill "$step" 2>/dev/null || true
    done
    for step in "${ray_steps[@]}"; do
        wait "$step" 2>/dev/null || true
    done
    ray_steps=()
}

ray_init_overrides=(
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS"
    "+ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY"
)
if (( NNODES > 1 )); then
    trap stop_sdpo_ray_cluster EXIT
    ray_head_ip=$("${python_cmd[@]}" -c 'from ray.util import get_node_ip_address; print(get_node_ip_address())')
    ray_port=$((20000 + SLURM_JOB_ID % 20000))
    ray_address="$ray_head_ip:$ray_port"
    export RAY_ADDRESS="$ray_address"
    ray_step=(srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="$RAY_NUM_CPUS" --gres="gpu:$GPUS_PER_NODE")
    ray_start=("${python_cmd[@]}" -m ray.scripts.scripts start --block
        --num-cpus="$RAY_NUM_CPUS" --num-gpus="$GPUS_PER_NODE" --object-store-memory="$RAY_OBJECT_STORE_MEMORY")
    "${ray_step[@]}" --nodelist="$SLURMD_NODENAME" "${ray_start[@]}" \
        --head --node-ip-address="$ray_head_ip" --port="$ray_port" --include-dashboard=false &
    ray_steps+=("$!")
    timeout 120 "${python_cmd[@]}" - "$ray_head_ip" "$ray_port" <<'PY'
import socket
import sys
import time

while True:
    try:
        with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
            break
    except OSError:
        time.sleep(1)
PY
    node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    while IFS= read -r node; do
        [[ "$node" != "$SLURMD_NODENAME" ]] || continue
        "${ray_step[@]}" --nodelist="$node" "${ray_start[@]}" --address="$ray_address" &
        ray_steps+=("$!")
    done <<< "$node_list"
    timeout 180 "${python_cmd[@]}" - "$ray_address" "$NNODES" "$GPUS_PER_NODE" <<'PY'
import sys
import time

import ray

ray.init(address=sys.argv[1])
nodes_expected, gpus_per_node = map(int, sys.argv[2:])
while True:
    gpu_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU") == gpus_per_node]
    if len(gpu_nodes) == nodes_expected:
        print(f"Ray ready with {nodes_expected} nodes and {nodes_expected * gpus_per_node} GPUs", flush=True)
        break
    time.sleep(2)
ray.shutdown()
PY
    ray_init_overrides=("+ray_kwargs.ray_init.address=$ray_address")
fi

"$NOTIFY" "Launching verl.trainer.main_ppo; live training progress follows" || true
printf 'Training batch: %s groups, %s prompts, %s rollouts per prompt, %s response tokens, %s GPUs\n' \
    "$TRAIN_GROUPS_PER_BATCH" "$TRAIN_BATCH_SIZE" "$ROLLOUTS_PER_PROMPT" "$MAX_RESPONSE_TOKENS" "$TOTAL_GPUS"
printf 'Actor learning rate=%s reference KL enabled=%s coefficient=%s\n' "$LEARNING_RATE" "$USE_KL_LOSS" "$KL_LOSS_COEF"
printf 'SDPO alpha=%s\n' "$SDPO_ALPHA"
printf 'Training rollout dumps: %s\n' "$ROLLOUT_DATA_DIR"
"$NOTIFY" "Ray resources: cpus=$RAY_NUM_CPUS object_store_memory=${RAY_OBJECT_STORE_MEMORY}B" || true
training_status=0
"${python_cmd[@]}" -m verl.trainer.main_ppo --config-name sdpo \
    "vars.dir=$SDPO_DIR" \
    "vars.task=$TASK" \
    "data.train_files=[$DATA_DIR/train.parquet]" \
    "data.val_files=[$DATA_DIR/test.parquet]" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.max_response_length=$MAX_RESPONSE_TOKENS" \
    "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
    "data.parallel_group_batching.enabled=True" \
    "data.parallel_group_batching.group_key=parallel_id" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.model.use_remove_padding=False" \
    "actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-True}" \
    "actor_rollout_ref.actor.fsdp_config.use_meta_init=False" \
    "actor_rollout_ref.ref.fsdp_config.use_meta_init=False" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.n=$ROLLOUTS_PER_PROMPT" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE" \
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE" \
    "actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS" \
    "actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF" \
    "actor_rollout_ref.actor.calculate_entropy=True" \
    "actor_rollout_ref.actor.entropy_coeff=0" \
    "actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=False" \
    "actor_rollout_ref.actor.self_distillation.reference_reasoning_only=False" \
    "actor_rollout_ref.actor.self_distillation.alpha=$SDPO_ALPHA" \
    "actor_rollout_ref.actor.self_distillation.distillation_topk=512" \
    "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True" \
    "actor_rollout_ref.actor.self_distillation.grpo_on_success=$GRPO_ON_SUCCESS" \
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0" \
    "actor_rollout_ref.actor.self_distillation.max_reprompt_len=$TEACHER_PROMPT_TOKENS" \
    "actor_rollout_ref.actor.self_distillation.solution_reference_language=$reference_language" \
    "actor_rollout_ref.actor.self_distillation.log_teacher_prompts=$LOG_TEACHER_PROMPTS" \
    "trainer.project_name=SDPO-multilingual" \
    "trainer.group_name=macereason-math-$MAX_TRAIN_GROUPS" \
    "trainer.experiment_name=$EXPERIMENT${alpha_tag}" \
    "trainer.nnodes=$NNODES" \
    "trainer.n_gpus_per_node=$GPUS_PER_NODE" \
    "trainer.total_epochs=1" \
    "trainer.save_freq=50" \
    "trainer.test_freq=50" \
    "trainer.val_before_train=False" \
    "trainer.resume_mode=disable" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.rollout_data_dir=$ROLLOUT_DATA_DIR" \
    "trainer.max_actor_ckpt_to_keep=null" \
    "${ray_init_overrides[@]}" || training_status=$?
if (( NNODES > 1 )); then
    stop_sdpo_ray_cluster
fi

printf '[checkpoint upload] Trainer exited with status %s. Checking %s\n' "$training_status" "$CHECKPOINT_DIR"
exporter="$SDPO_DIR/scripts/hyperion/eval/export_checkpoint_to_hub.sh"
uploaded=0
upload_failures=0
for checkpoint in "$CHECKPOINT_DIR"/global_step_*; do
    step_name="${checkpoint##*/}"
    [[ "$step_name" =~ ^global_step_[0-9]+$ && -d "$checkpoint/actor" ]] || continue

    export_dir="$checkpoint-hf"
    export_args=()
    if [[ -d "$export_dir" ]]; then
        export_args+=(--upload-only)
    fi
    printf '[checkpoint upload] Processing %s\n' "$checkpoint"
    # Derive a separate Hub repository ID from each checkpoint's path.
    if env -u HF_REPO_ID HF_OUTPUT_DIR="$export_dir" HF_HUB_DISABLE_PROGRESS_BARS=1 \
        SDPO_CONTAINER_IMAGE="$CONTAINER_IMAGE" SDPO_CONTAINER_ENV="$CONTAINER_ENV" \
        "$exporter" "${export_args[@]}" "$checkpoint"; then
        ((uploaded += 1))
        printf '[checkpoint upload] Uploaded %s\n' "$checkpoint"
    else
        ((upload_failures += 1))
        printf '[checkpoint upload] Failed for %s\n' "$checkpoint" >&2
    fi
done
printf '[checkpoint upload] Uploaded %s checkpoint(s), %s failed.\n' "$uploaded" "$upload_failures"

if (( training_status != 0 || upload_failures != 0 )); then
    "$NOTIFY" "❌ $TAG trainer exit=$training_status, checkpoints uploaded=$uploaded, upload failures=$upload_failures" || true
    if (( training_status != 0 )); then
        exit "$training_status"
    fi
    exit 1
fi

"$NOTIFY" "✅ $TAG finished successfully, checkpoints uploaded=$uploaded" || true
