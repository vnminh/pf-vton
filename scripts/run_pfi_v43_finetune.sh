#!/usr/bin/env bash
set -euo pipefail
cd /workspace/patch-forcing-vton
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
config=configs/vton_v43_pfi_1024_24gb.yaml
pilot_dir=logs/vton-v43-pfi-1024-24gb
args=()
if [[ -f "$pilot_dir/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/ai/bin/python -m vton_ext.pfi_train --config "$config" "${args[@]}" "$@"
