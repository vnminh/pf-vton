"""Score every in-shop garment by how much of it is print, logo or text.

    PYTHONPATH=. python scripts/score_garment_detail.py \
        --root ../high-resolution-viton-zalando-dataset --output checkpoints/garment_detail_scores.json

Score = fraction of the garment interior (mask eroded by 8 px) whose grey
level differs from its 5x5 local mean by more than 0.08 in [-1, 1], on the
512x384 in-shop image: fine edges of letters, logos and prints, not smooth
shading or the garment outline. Used by data.detail_sampling to oversample
garments whose details the model still gets wrong.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SIZE = (384, 512)  # PIL (width, height)
ERODE = 17         # 8 px interior margin
EDGE = 0.08        # |grey - 5x5 box blur| in [-1, 1]


def score(cloth_path: Path, mask_path: Path) -> float:
    rgb = np.asarray(Image.open(cloth_path).convert("L").resize(SIZE, Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    mask = np.asarray(Image.open(mask_path).convert("L").resize(SIZE, Image.NEAREST)) > 127
    # Interior only: the garment outline against the background is not detail.
    inner = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(ERODE)))> 127
    if inner.sum() < 100:
        return 0.0
    blur = np.asarray(Image.fromarray(((rgb + 1) * 127.5).astype(np.uint8)).filter(ImageFilter.BoxBlur(2)),
                      dtype=np.float32) / 127.5 - 1.0
    edges = np.abs(rgb - blur) > EDGE
    return float(edges[inner].mean())


def _job(args):
    name, cloth_dir, mask_dir = args
    stem = Path(name).stem
    masks = [p for p in (mask_dir / name, mask_dir / f"{stem}.png", mask_dir / f"{stem}.jpg") if p.is_file()]
    if not masks:
        raise FileNotFoundError(f"no cloth mask for {name}")
    return name, score(cloth_dir / name, masks[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    base = Path(args.root) / "train"
    cloth_dir, mask_dir = base / "cloth", base / "cloth-mask"
    names = sorted(p.name for p in cloth_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with ProcessPoolExecutor(args.workers) as pool:
        scores = dict(pool.map(_job, [(n, cloth_dir, mask_dir) for n in names], chunksize=64))
    Path(args.output).write_text(json.dumps(scores, indent=0, sort_keys=True))
    values = np.array(list(scores.values()))
    print(f"{len(values)} garments; score quantiles 25/50/75/90%: "
          f"{np.percentile(values, [25, 50, 75, 90]).round(4).tolist()}")


if __name__ == "__main__":
    main()
