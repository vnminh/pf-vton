#!/usr/bin/env bash
# V49 post-norm fine garment: resume the run's own latest.pt when present, else start from V48 latest.pt weights.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=logs/vton-v49-fine-postnorm
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config configs/vton_v49_fine_postnorm.yaml "${args[@]}" "$@"
