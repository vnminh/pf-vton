#!/usr/bin/env bash
# Isolated patched source avoids concurrent writes by the repository migration.
set -euo pipefail
cd /workspace/pfi-resume-audit/code
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
config=configs/v46-recovery-server.yaml
out=/workspace/patch-forcing-vton/logs/vton-v46-pfi-1024-paired-recovery
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config "$config" "${args[@]}" "$@"
