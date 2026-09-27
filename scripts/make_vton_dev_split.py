#!/usr/bin/env python3
"""Create a deterministic fit/development split without touching official test pairs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

# Preserve direct-script usage as well as `python -m scripts.make_vton_dev_split`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vton_ext.pairs import read_pairs


def stable_key(row: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{row}".encode("utf-8")).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fit-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--dev-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.input)
    rows = [f"{person} {garment}" for person, garment in read_pairs(source)]
    if not 0 < args.dev_size < len(rows):
        raise ValueError(f"dev-size must be between 1 and {len(rows) - 1}")

    ranked = sorted(range(len(rows)), key=lambda i: stable_key(rows[i], args.seed))
    dev_indices = set(ranked[: args.dev_size])
    fit_rows = [row for i, row in enumerate(rows) if i not in dev_indices]
    dev_rows = [row for i, row in enumerate(rows) if i in dev_indices]

    fit_path = Path(args.fit_output)
    dev_path = Path(args.dev_output)
    if fit_path.exists() or dev_path.exists():
        raise FileExistsError("Keep existing split files; use prepare_pfi_pairs for an audited repair")
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    fit_path.write_text("\n".join(fit_rows) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(dev_rows) + "\n", encoding="utf-8")
    print(f"source={len(rows)} fit={len(fit_rows)} dev={len(dev_rows)} seed={args.seed}")
    print(f"fit={fit_path.resolve()}")
    print(f"dev={dev_path.resolve()}")


if __name__ == "__main__":
    main()
