#!/usr/bin/env bash
# Server launcher: resume the pilot's latest save after an interrupted run.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

resume_checkpoint=logs/vton-v41-pfi-detail/latest.pt
if [[ ! -f "$resume_checkpoint" ]]; then
    resume_checkpoint=logs/vton-v40-pfi-coral/step0005000-resume.pt
fi
if [[ ! -f "$resume_checkpoint" ]]; then
    printf 'Missing resume checkpoint: %s\n' "$resume_checkpoint" >&2
    exit 1
fi

exec "${PFI_PYTHON:-/venv/ai/bin/python}" -u -m vton_ext.pfi_train \
    --config configs/vton_v41_pfi_detail.yaml \
    --resume "$resume_checkpoint" "$@"
