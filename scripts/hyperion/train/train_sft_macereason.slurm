#!/bin/bash
#SBATCH --job-name=mace-sft
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --mem=100gb
#SBATCH --time=1-00:00:00
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"

SDPO_DIR="${SDPO_DIR:-/scratch/asalem/repos/SDPO}"
SDPO_ENV="${SDPO_ENV:-/scratch/asalem/SDPO-venv}"
SOURCE_DATASET="${SOURCE_DATASET:-ahmedselhady/mAceReason-Math}"
LANGUAGES="${LANGUAGES-bn,de,en,es,fr,it,ja,ko,pt,ru,sw,te,th,zh}"
MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-Research-Reasoning-Qwen-1.5B}"
TRAIN_PERCENTAGE="${TRAIN_PERCENTAGE:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$NUM_GPUS}"
EXPERIMENT="${EXPERIMENT:-macereason-sft-${SLURM_JOB_ID:-local-$$}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SDPO_DIR/checkpoints/$EXPERIMENT}"

cd "$SDPO_DIR"
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
export HF_HOME="${HF_HOME:-/scratch/asalem/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

module load CUDA/12.8.0
PYTHON="$SDPO_ENV/bin/python"
if ! "$PYTHON" --version; then
    "$NOTIFY" "ERROR: set SDPO_ENV to a training environment whose Python runs on this compute node." || true
    exit 1
fi
source "$SDPO_ENV/bin/activate"

DATA_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/macereason-sft.XXXXXXXX")"
trap 'rm -r -- "$DATA_DIR"' EXIT

"$NOTIFY" "Preparing $TRAIN_PERCENTAGE% of $SOURCE_DATASET train (languages $LANGUAGES, seed $SEED)" || true
"$PYTHON" data/prepare_macereason_math.py \
    --dataset "$SOURCE_DATASET" \
    --output-dir "$DATA_DIR" \
    --languages "$LANGUAGES" \
    --train-percentage "$TRAIN_PERCENTAGE" \
    --seed "$SEED"

"$PYTHON" - "$DATA_DIR" "$TRAIN_BATCH_SIZE" "$NUM_GPUS" "$LEARNING_RATE" <<'PY'
import math
from pathlib import Path
import sys

from pyarrow.parquet import ParquetFile

data_dir, batch_size, num_gpus, learning_rate = sys.argv[1:]
batch_size, num_gpus = int(batch_size), int(num_gpus)
learning_rate = float(learning_rate)
if not math.isfinite(learning_rate) or learning_rate <= 0:
    sys.exit("LEARNING_RATE must be finite and positive.")
if batch_size < num_gpus or batch_size % num_gpus:
    sys.exit("TRAIN_BATCH_SIZE must be a positive multiple of the number of GPUs.")
for split, minimum in (("train", batch_size), ("test", num_gpus)):
    rows = ParquetFile(Path(data_dir) / f"{split}.parquet").metadata.num_rows
    if rows < minimum:
        sys.exit(f"{split} has {rows} rows but needs at least {minimum} for one batch.")
PY

"$NOTIFY" "Starting full-parameter SFT on $MODEL_PATH with learning rate $LEARNING_RATE" || true
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    "data.train_files=$DATA_DIR/train.parquet" \
    "data.val_files=$DATA_DIR/test.parquet" \
    "data.prompt_key=prompt" \
    "data.prompt_dict_keys=[content]" \
    "data.response_key=reward_model" \
    "data.response_dict_keys=[ground_truth]" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.micro_batch_size_per_gpu=1" \
    "data.max_length=$MAX_LENGTH" \
    "model.partial_pretrain=$MODEL_PATH" \
    "model.attn_implementation=sdpa" \
    "model.lora_rank=0" \
    "optim.lr=$LEARNING_RATE" \
    "trainer.n_gpus_per_node=$NUM_GPUS" \
    "trainer.total_epochs=$EPOCHS" \
    "trainer.seed=$SEED" \
    "trainer.project_name=macereason-sft" \
    "trainer.experiment_name=$EXPERIMENT" \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "trainer.logger=[console]" \
    "trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]"

"$NOTIFY" "Saved SFT checkpoint under $CHECKPOINT_DIR" || true
