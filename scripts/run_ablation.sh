#!/usr/bin/env bash
# Train one ablation branch; resumes its own latest.pt when present.
#
#   bash scripts/run_ablation.sh r0|e1|e2|e3|e4 [config overrides...]
#   bash scripts/run_ablation.sh e3 --bench        # 100-update speed/memory probe, separate output dir
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/venv/main/bin/python}
case "${1:-}" in
    r0) cfg=configs/ablation/r0_v50_recipe.yaml ;;
    e1) cfg=configs/ablation/e1_shared_timestep.yaml ;;
    e2) cfg=configs/ablation/e2_ltg_only.yaml ;;
    e3) cfg=configs/ablation/e3_full_attention.yaml ;;
    e4) cfg=configs/ablation/e4_no_decoded_loss.yaml ;;
    *) echo "usage: $0 r0|e1|e2|e3|e4 [--bench] [overrides...]" >&2; exit 2 ;;
esac
name=$1; shift
out=$(grep -m1 "output_dir:" "$cfg" | awk '{print $2}')
args=()
if [[ "${1:-}" == "--bench" ]]; then
    shift
    out=logs/ablation/bench-$name
    args=(train.output_dir=$out train.stop_at_step=100 eval.at_start=false eval.every=1000000
          train.save_every=1000000 train.keep_every=1000000)
elif [[ -f "$out/latest.pt" ]]; then
    args=(--resume auto)
fi
mkdir -p logs/ablation
"$PY" -u -m vton_ext.pfi_train --config "$cfg" "${args[@]}" "$@" 2>&1 | tee -a "logs/ablation/$(basename "$out").log"
