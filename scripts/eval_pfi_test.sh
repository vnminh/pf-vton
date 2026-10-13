#!/usr/bin/env bash
# Đánh giá một checkpoint PFI-VTON trên tập test VITON-HD (paired + unpaired), trên server.
#
#   bash scripts/eval_pfi_test.sh <checkpoint.pt> [config.yaml] [các tùy chọn khác của eval_pfi_test.py]
#
# Ví dụ:
#   bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt
#   bash scripts/eval_pfi_test.sh logs/ablation/e1_shared_timestep/latest.pt configs/ablation/e1_shared_timestep.yaml \
#       --sampler euler
#   CFG_SCALES=1,1.5,2 SEED=123 bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt
#   LIMIT=16 bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt      # chạy thử 16 mẫu
#
# Biến môi trường (có giá trị mặc định): PY, DATA_ROOT, SAMPLER, NFE, CFG_SCALES, SEED,
# BATCH_SIZE, LIMIT, OUT (thư mục kết quả; mặc định outputs/eval/<run>-<sampler><nfe>-cfg<w>-seed<s>).
# Chạy lại cùng lệnh sẽ tiếp tục từ chỗ dừng (bỏ qua các mẫu đã có ảnh và dòng CSV).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 || "$1" == -* ]]; then
    sed -n '2,16p' "$0" >&2
    exit 2
fi
CKPT=$1; shift
CONFIG=configs/vton_v50_attn_flow.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG=$1; shift
fi
[[ -f "$CKPT" ]] || { echo "không thấy checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "không thấy config: $CONFIG" >&2; exit 1; }

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-../high-resolution-viton-zalando-dataset}
SAMPLER=${SAMPLER:-look_ahead}
NFE=${NFE:-50}
CFG_SCALES=${CFG_SCALES:-1.5}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-8}
LIMIT=${LIMIT:-0}
run=$(basename "$(dirname "$CKPT")")
OUT=${OUT:-outputs/eval/${run}-${SAMPLER}${NFE}-cfg${CFG_SCALES//,/_}-seed${SEED}$([[ $LIMIT != 0 ]] && echo "-limit$LIMIT" || true)}

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "checkpoint: $CKPT | config: $CONFIG | output: $OUT"
"$PY" scripts/eval_pfi_test.py \
    --checkpoint "$CKPT" --config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUT" \
    --sampler "$SAMPLER" --nfe "$NFE" --cfg-scales "$CFG_SCALES" --seed "$SEED" \
    --batch-size "$BATCH_SIZE" --limit "$LIMIT" "$@" 2>&1 | tee -a "$OUT/eval.log"
