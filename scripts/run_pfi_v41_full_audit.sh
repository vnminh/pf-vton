#!/usr/bin/env bash
# Wait for the bounded V41 pilot, then audit all 256 paired dev cases.
set -euo pipefail
cd /workspace/patch-forcing-vton
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

pilot_dir=logs/vton-v41-pfi-detail
audit_dir="$pilot_dir/full-dev-audit"
mkdir -p "$audit_dir"

while true; do
    # The fixed eight-case preview at step 5500 may beat the later endpoint.
    # Save its optimizer state before latest.pt is replaced at step 5750.
    if [[ -f "$pilot_dir/step0005500.pt" && ! -f "$pilot_dir/step0005500-resume.pt" ]]; then
        ln "$pilot_dir/latest.pt" "$pilot_dir/step0005500-resume.pt"
    fi
    # supervisorctl returns 3 for an exited program, including expected exit 0.
    status="$(supervisorctl status pfi-v41-detail || true)"
    if [[ "$status" == *RUNNING* || "$status" == *STARTING* ]]; then
        sleep 30
        continue
    fi
    if [[ "$status" == *EXITED* && -f "$pilot_dir/step0006000.pt" ]] &&
       tail -n 1 "$pilot_dir/eval.jsonl" | grep -q '"step": 6000'; then
        break
    fi
    echo "V41 did not finish successfully: $status" >&2
    exit 1
done

# Keep the completed pilot accessible even if log-directory cleanup runs.
mkdir -p checkpoints/pfi-safety
if [[ ! -f checkpoints/pfi-safety/v41-step6000-resume.pt ]]; then
    ln "$pilot_dir/latest.pt" checkpoints/pfi-safety/v41-step6000-resume.pt
fi
if [[ ! -f checkpoints/pfi-safety/v41-step6000.pt ]]; then
    ln "$pilot_dir/step0006000.pt" checkpoints/pfi-safety/v41-step6000.pt
fi

# The V40 step-5000 checkpoint was deleted. Compare the two retained V41
# points under one fixed protocol and keep V40's eight-case records separate.
baseline_name=v41-step5500
baseline_label='V41 step 5500'
baseline_config=configs/vton_v41_pfi_detail.yaml
baseline_checkpoint=checkpoints/pfi-safety/v41-step5500.pt
candidate_label='V41 step 6000'

if [[ ! -f "$audit_dir/$baseline_name/complete.json" ]]; then
    /venv/ai/bin/python scripts/audit_pfi_timesteps.py \
        --config "$baseline_config" \
        --checkpoint "$baseline_checkpoint" \
        --output "$audit_dir/$baseline_name" --batch-size 4
fi

if [[ ! -f "$audit_dir/v41-step6000/complete.json" ]]; then
    /venv/ai/bin/python scripts/audit_pfi_timesteps.py \
        --config configs/vton_v41_pfi_detail.yaml \
        --checkpoint checkpoints/pfi-safety/v41-step6000.pt \
        --output "$audit_dir/v41-step6000" --batch-size 4
fi

/venv/ai/bin/python scripts/compare_pfi_audits.py \
    --baseline "$audit_dir/$baseline_name" \
    --candidate "$audit_dir/v41-step6000" \
    --output "$audit_dir/comparison"

/venv/ai/bin/python scripts/plot_pfi_logo_comparison.py \
    --baseline "$audit_dir/$baseline_name" \
    --candidate "$audit_dir/v41-step6000" \
    --baseline-label "$baseline_label" --candidate-label "$candidate_label" \
    --output "$audit_dir/comparison/logo-crops.png"

/venv/ai/bin/python scripts/plot_pfi_time_coverage.py \
    --baseline-config "$baseline_config" \
    --candidate-config configs/vton_v41_pfi_detail.yaml \
    --comparison "$audit_dir/comparison/comparison.json" \
    --baseline-label "$baseline_label" --candidate-label "$candidate_label" \
    --output "$audit_dir/comparison/time-coverage-vs-error.png"
