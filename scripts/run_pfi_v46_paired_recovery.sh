#!/usr/bin/env bash
# Run from the repository containing this launcher.
set -euo pipefail
pfi_code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pfi_code_dir"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
config="${PFI_RECOVERY_CONFIG:-configs/vton_v46_pfi_1024_paired_recovery.yaml}"
out=logs/vton-v46-pfi-1024-paired-recovery
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config "$config" "${args[@]}" "$@"
