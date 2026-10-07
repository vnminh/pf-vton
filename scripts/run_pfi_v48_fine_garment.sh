#!/usr/bin/env bash
# V48 fine garment: resume the run's own latest.pt when present, else start from V47 latest.pt weights.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=logs/vton-v48-fine-garment
args=()
if [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
exec /venv/main/bin/python -u -m vton_ext.pfi_train --config configs/vton_v48_fine_garment.yaml "${args[@]}" "$@"
