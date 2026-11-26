#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Load the user's sbatch alias `sb` from .bashrc. The shell is
# non-interactive here, so enable alias expansion explicitly.
shopt -s expand_aliases
source "$HOME/.bashrc"
NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"

TASKS_FILE="$SCRIPT_DIR/eval_tasks.sh"
if ! source "$TASKS_FILE" || (( ${#EVAL_TASKS[@]} == 0 )); then
    "$NOTIFY" "ERROR: could not load a nonempty evaluation task list from $TASKS_FILE" || true
    exit 1
fi

# .bashrc changes to /scratch/asalem; submit from this script's directory so
# the relative Slurm script and output paths resolve consistently.
cd "$SCRIPT_DIR"
mkdir -p .slurm

# The publisher keeps each run's default revision (main) on its latest checkpoint.
models=(
    # "ahmedselhady/XSDPO-Nemotron-1.5B-1500x2-en-bn-lr3e-6-kl0.001-reasoning-ref10752-k512-gs-en_only"
    # "ahmedselhady/XSDPO-Nemotron-1.5B-1500x2-en-bn-lr3e-6-no-kl-en"
    "ahmedselhady/XSDPO-Nemotron-1.5B-300x5-lr3e-6-kl0.001-full-ref10752-k512-frozen-en_only-alpha1"
    "ahmedselhady/XSDPO-Nemotron-1.5B-500x5-lr3e-6-kl0.001-full-ref10752-k512-frozen-en_only-alpha1"
    "ahmedselhady/XSDPO-Nemotron-1.5B-750x5-lr3e-6-kl0.001-full-ref10752-k512-frozen-en_only-alpha1.0"
    # "ahmedselhady/XSDPO-Nemotron-1.5B-500x5-lr3e-6-kl0.001-reasoning-ref10752-k512-gs-en_only"
)

task_count="${#EVAL_TASKS[@]}"
# Run one task at a time per array and up to two model arrays concurrently.
array_range="0-$(( task_count - 1 ))%1"
max_parallel_models=2
cache_root="${HF_HOME:-/scratch/asalem/.cache/huggingface}/eval-arrays"
if ! mkdir -p -- "$cache_root" || ! cache_root="$(cd -- "$cache_root" && pwd -P)"; then
    "$NOTIFY" "ERROR: could not prepare evaluation cache directory $cache_root" || true
    exit 1
fi

status=0
slot_jobs=()
for model_index in "${!models[@]}"; do
    model="${models[$model_index]}"
    slot=$(( model_index % max_parallel_models ))
    dependency=()
    if [[ -n "${slot_jobs[$slot]:-}" ]]; then
        dependency=(--dependency="afterany:${slot_jobs[$slot]}")
    fi

    if ! cache_dir="$(mktemp -d -- "$cache_root/${model##*/}.XXXXXXXX")"; then
        "$NOTIFY" "ERROR: could not allocate a cache for $model under $cache_root" || true
        status=1
        continue
    fi

    if ! submission="$(EVAL_HF_CACHE="$cache_dir" sb --parsable --array="$array_range" \
        "${dependency[@]}" --export="ALL,MODEL=$model" run_eval_task_model.sh)"; then
        rmdir -- "$cache_dir" || true
        "$NOTIFY" "ERROR: failed to submit the evaluation array for $model" || true
        status=1
        continue
    fi
    job_id="${submission%%;*}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        "$NOTIFY" "ERROR: could not parse the array ID for $model from '$submission'. Cache: $cache_dir" || true
        exit 1
    fi

    slot_jobs[$slot]="$job_id"
    "$NOTIFY" "Submitted $task_count evaluation tasks for $model (job $job_id, array $array_range)" || true
    # The sb alias adds an A100 constraint. Cleanup only needs a CPU.
    if ! cleanup_submission="$(EVAL_HF_CACHE="$cache_dir" command sbatch --parsable \
        --account=hitz-exclusive --partition=hitz-exclusive --dependency="afterany:$job_id" \
        --job-name=mace-eval-cleanup --time=00:15:00 --mem=512M \
        --output=.slurm/%x-%j.out --error=.slurm/%x-%j.err \
        --export=ALL --wrap='rm -rf -- "${EVAL_HF_CACHE:?}"')"; then
        "$NOTIFY" "ERROR: could not schedule cache cleanup for $model (array $job_id). Cache: $cache_dir" || true
        status=1
        continue
    fi
    cleanup_id="${cleanup_submission%%;*}"
    if [[ ! "$cleanup_id" =~ ^[0-9]+$ ]]; then
        "$NOTIFY" "ERROR: could not parse the cleanup job ID for $model from '$cleanup_submission'. Cache: $cache_dir" || true
        exit 1
    fi
    slot_jobs[$slot]="$job_id:$cleanup_id"
done
exit "$status"
