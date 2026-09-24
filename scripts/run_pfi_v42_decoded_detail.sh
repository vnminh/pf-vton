#!/usr/bin/env bash
# V42 decoded-detail candidate; launch only after reviewing the V41 full audit.
set -euo pipefail
cd /workspace/patch-forcing-vton
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
pilot_dir=logs/vton-v42-pfi-decoded-detail
if [[ -f "$pilot_dir/latest.pt" ]]; then
    resume="$pilot_dir/latest.pt"
elif [[ -f checkpoints/pfi-safety/v41-step6000-resume.pt ]]; then
    resume=checkpoints/pfi-safety/v41-step6000-resume.pt
else
    resume=logs/vton-v41-pfi-detail/latest.pt
fi
exec /venv/ai/bin/python -m vton_ext.pfi_train \
    --config configs/vton_v42_pfi_decoded_detail.yaml --resume "$resume"
