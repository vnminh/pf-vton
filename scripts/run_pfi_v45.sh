#!/usr/bin/env bash
# V45 on the 48 GB A6000: resume latest.pt when present, otherwise start from V44 step-1250 weights.
set -euo pipefail
cd /workspace/patch-forcing-vton
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
config=configs/vton_v45_pfi_1024.yaml
out=logs/vton-v45-pfi-1024
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -W ignore -m vton_ext.pfi_train --config "$config" "${args[@]}" "$@"
