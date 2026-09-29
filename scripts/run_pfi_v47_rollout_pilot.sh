#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

case "${1:-pilot}" in
  probe)
    exec /venv/main/bin/python -u -m vton_ext.pfi_train \
      --config configs/vton_v47_rollout_repair_pilot.yaml \
      train.stop_at_step=2
    ;;
  pilot)
    exec /venv/main/bin/python -u -m vton_ext.pfi_train \
      --config configs/vton_v47_rollout_repair_pilot.yaml --resume auto
    ;;
  *)
    echo "usage: $0 {probe|pilot}" >&2
    exit 2
    ;;
esac
