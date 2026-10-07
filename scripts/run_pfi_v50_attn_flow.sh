#!/usr/bin/env bash
# V50 attention flow: resume the run's own latest.pt when present, else start from V49 latest.pt weights.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=logs/vton-v50-attn-flow
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config configs/vton_v50_attn_flow.yaml "${args[@]}" "$@"
