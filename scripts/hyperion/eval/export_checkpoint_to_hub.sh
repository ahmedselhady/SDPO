#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: export_checkpoint_to_hub.sh [--local-only | --upload-only] CHECKPOINT_DIR

Merge an SDPO FSDP actor checkpoint into a Hugging Face model and upload it.
CHECKPOINT_DIR is a global_step directory containing actor/. Quote paths with spaces.

  --local-only   Export without contacting the Hub.
  --upload-only  Upload an existing export from HF_OUTPUT_DIR or the default output path.
  HF_REPO_ID     Override the default <account>/<experiment>-<step> model repo.
  HF_OUTPUT_DIR  Override the default <step>-hf directory beside the checkpoint.
  HF_TOKEN       Use a write token from the environment, repo .env, or cached login.

New Hub repositories are private. A trailing " copy" is stripped from the step name.
EOF
}

local_only=false
upload_only=false
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --local-only) local_only=true; shift ;;
    --upload-only) upload_only=true; shift ;;
esac
if [[ $# != 1 || ! -d "$1" ]]; then
    usage >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
checkpoint="$(realpath -- "$1")"
actor_dir="$checkpoint/actor"
if [[ ! -f "$actor_dir/fsdp_config.json" || ! -f "$actor_dir/huggingface/config.json" ]]; then
    printf 'Expected an FSDP actor checkpoint at %s\n' "$actor_dir" >&2
    exit 1
fi

experiment="$(basename "$(dirname "$checkpoint")")"
step="$(basename "$checkpoint")"
step="${step% copy}"
output_dir="$(realpath -m -- "${HF_OUTPUT_DIR:-$(dirname "$checkpoint")/$step-hf}")"
if [[ "$upload_only" == true ]]; then
    if [[ ! -f "$output_dir/config.json" || ! -f "$output_dir/tokenizer.json" ]] ||
        ! compgen -G "$output_dir/model*.safetensors" > /dev/null; then
        printf 'Expected an exported Hugging Face model at %s\n' "$output_dir" >&2
        exit 1
    fi
elif [[ -e "$output_dir" ]]; then
    printf 'Output already exists at %s. Set HF_OUTPUT_DIR to a new path.\n' "$output_dir" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/scratch/asalem/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
if [[ "$local_only" == true ]]; then
    export HF_HUB_OFFLINE=1
elif [[ -z "${HF_TOKEN:-}" && -f "$repo_root/.env" ]]; then
    set -a
    source "$repo_root/.env"
    set +a
fi

module load Apptainer/1.4.5
image="${SDPO_CONTAINER_IMAGE:-/scratch/asalem/SDPO-qwen35-vllm-0.19.1.sif}"
runtime="${SDPO_CONTAINER_ENV:-/scratch/asalem/SDPO-qwen35-container-venv}"
cd "$repo_root"
python_cmd=(apptainer exec --bind /scratch --bind "$(dirname "$output_dir")" "$image" "$runtime/bin/python")

repo_id="${HF_REPO_ID:-}"
if [[ "$local_only" == false && -z "$repo_id" ]]; then
    account="$("${python_cmd[@]}" -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')"
    repo_id="$account/$experiment-$step"
fi

if [[ "$upload_only" == false ]]; then
    staging_dir="$(mktemp -d -- "$output_dir.tmp.XXXXXXXX")"
    trap 'rm -rf -- "$staging_dir"' EXIT
    "${python_cmd[@]}" -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$staging_dir"
    mv -T --no-clobber -- "$staging_dir" "$output_dir"
    if [[ -d "$staging_dir" ]]; then
        printf 'Output appeared during export at %s.\n' "$output_dir" >&2
        exit 1
    fi
    printf 'Exported Hugging Face model to %s\n' "$output_dir"
fi

if [[ "$local_only" == false ]]; then
    "${python_cmd[@]}" - "$repo_id" "$output_dir" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, folder = sys.argv[1:]
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=folder)
files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
assert {"config.json", "tokenizer.json"} <= files and any(
    name.startswith("model") and name.endswith(".safetensors") for name in files
), "The uploaded model is missing weights, config, or tokenizer files"
print(f"Uploaded to https://huggingface.co/{repo_id}")
PY
fi
