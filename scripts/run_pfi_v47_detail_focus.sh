#!/usr/bin/env bash
# V47 detail focus: resume the run's own latest.pt when present, else start from V46 update 8250.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=logs/vton-v47-detail-focus
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config configs/vton_v47_detail_focus.yaml "${args[@]}" "$@"
