#!/bin/bash
#SBATCH --job-name=eval-toolchain-check
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --mem=16gb
#SBATCH --time=00:10:00
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -uo pipefail

NOTIFY="${NOTIFY:-$HOME/bin/slack_notify.py}"

WORKDIR="${WORKDIR:-/scratch/asalem/repos/SDPO}"
LM_EVAL_DIR="${LM_EVAL_DIR:-/scratch/asalem/repos/lm-evaluation-harness}"
ENV_DIR="$LM_EVAL_DIR/.venv"

"$NOTIFY" "=== Node and allocation ===" || true
date --iso-8601=seconds
hostname
scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true
lscpu | rg '^(Architecture|CPU op-mode|Model name|Vendor ID|CPU family|Model:|Flags):' || true
nvidia-smi || true

"$NOTIFY" "=== Modules and toolchain ===" || true
module load CUDA/12.8.0
module list 2>&1 || true

CPU_FAMILY="${CPU_FAMILY:-Zen3}"
TRITON_GCC="/scicomp/builds/Rocky/8.7/$CPU_FAMILY/software/GCCcore/11.3.0/bin/gcc"
TRITON_BINUTILS="/scicomp/builds/Rocky/8.7/$CPU_FAMILY/software/binutils/2.38-GCCcore-11.3.0/bin"
AS_DIR="${AS_DIR:-$TRITON_BINUTILS}"
if [[ ! -x "$TRITON_GCC" || ! -x "$TRITON_BINUTILS/as" ]]; then
    "$NOTIFY" "ERROR: GCC/binutils toolchain for CPU family $CPU_FAMILY is unavailable." || true
    exit 2
fi
export CC="$TRITON_GCC"
export PATH="$AS_DIR:$(dirname "$TRITON_GCC"):$PATH"

"$NOTIFY" "CC=$CC" || true
"$NOTIFY" "gcc=$(command -v gcc || true)" || true
"$NOTIFY" "as=$(command -v as || true)" || true
gcc --version || true
as --version || true

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cat > "$tmpdir/probe.c" <<'EOF'
int main(void) { return 0; }
EOF
"$NOTIFY" "=== Minimal GCC/assembler compile ===" || true
gcc -v -O3 "$tmpdir/probe.c" -o "$tmpdir/probe"
gcc_rc=$?
"$NOTIFY" "minimal_compile_exit_code=$gcc_rc" || true
if [[ $gcc_rc -eq 0 ]]; then
    "$tmpdir/probe"
    "$NOTIFY" "minimal_binary_exit_code=$?" || true
fi

"$NOTIFY" "=== lm-eval uv environment and Triton probes ===" || true
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$NOTIFY" "ERROR: expected Python executable missing: $ENV_DIR/bin/python" || true
    exit 2
fi
cat > "$tmpdir/triton_probe.py" <<'PY'
import sys
import torch
import triton
import transformers
import accelerate
import lm_eval

print("python:", sys.version.split()[0], sys.executable)
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("triton:", triton.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("lm_eval:", lm_eval.__file__)
print("torch.cuda.is_available:", torch.cuda.is_available())

try:
    from triton.runtime import driver
    print("triton current device:", driver.active.get_current_device())
except Exception:
    print("TRITON_DRIVER_HELPER_FAILED")
    raise

import triton.language as tl

@triton.jit
def add_one(x_ptr, y_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets, offsets < n, other=0)
    tl.store(y_ptr + offsets, values + 1, offsets < n)

try:
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(1,)](x, y, 128, 128)
    torch.cuda.synchronize()
    assert torch.equal(y, x + 1)
    print("TRITON_KERNEL_PASSED")
except Exception:
    print("TRITON_KERNEL_FAILED")
    raise
PY
"$ENV_DIR/bin/python" "$tmpdir/triton_probe.py"
python_rc=$?
"$NOTIFY" "triton_probe_exit_code=$python_rc" || true

if [[ $gcc_rc -ne 0 || $python_rc -ne 0 ]]; then
    exit 1
fi
"$NOTIFY" "All compiler and Triton probes passed." || true
