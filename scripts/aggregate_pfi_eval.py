"""Mean and standard deviation over sampling seeds of infer_modal.py results.

    python scripts/aggregate_pfi_eval.py --root /path/to/experiments [--mode full] [--markdown]

Reads every <root>/<output_tag>/<mode>/metadata.json and the per-CFG
pair/unpair metrics/summary.json beside it. Runs that differ only in their
sampling seed (same checkpoint, config, sampler, NFE, p, context ratio and CFG
scale) are grouped; each metric is reported as mean, sample std over seeds and
the number of seeds. KID's own subset std is a different quantity and is not
mixed in here. Copy the JSON summaries off the Modal volume first, e.g.
``modal volume get VTON-HD experiments ./experiments``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = {
    "pair": ("mse_mean", "ssim_mean", "lpips_mean", "fid", "kid", "sample_seconds_per_image",
             "denoiser_calls_per_image"),
    "unpair": ("fid", "kid", "sample_seconds_per_image", "denoiser_calls_per_image"),
}
SETTING_KEYS = ("checkpoint", "checkpoint_step", "config", "sampler", "nfe", "p", "context_ratio")


def collect(root: Path, mode: str):
    groups = defaultdict(lambda: defaultdict(list))  # setting -> (cfg, split, metric) -> values
    seeds = defaultdict(set)
    for meta_path in sorted(root.glob(f"*/{mode}/metadata.json")):
        meta = json.loads(meta_path.read_text())
        for scale in meta["cfg_scales"]:
            setting = tuple((k, meta.get(k)) for k in SETTING_KEYS) + (("cfg", scale),)
            for split in ("pair", "unpair"):
                summary_path = meta_path.parent / f"cfg{scale:g}" / split / "metrics" / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                for metric in METRICS[split]:
                    if summary.get(metric) is not None:
                        groups[setting][(split, metric)].append(float(summary[metric]))
            seeds[setting].add(meta["seed"])
    return groups, seeds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding <output_tag>/<mode>/...")
    ap.add_argument("--mode", default="full", choices=("full", "sample"))
    ap.add_argument("--markdown", action="store_true", help="one markdown table row per setting")
    args = ap.parse_args(argv)
    groups, seeds = collect(Path(args.root), args.mode)
    if not groups:
        raise SystemExit(f"no {args.mode} results under {args.root}")
    for setting, values in groups.items():
        s = dict(setting)
        name = (f"{Path(str(s['config'])).stem} step {s['checkpoint_step']} | {s['sampler']} NFE {s['nfe']} "
                f"p {s['p']} ratio {s['context_ratio']} | CFG {s['cfg']:g} | seeds {sorted(seeds[setting])}")
        cells = []
        for (split, metric), vals in values.items():
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            scale = 1000.0 if metric == "kid" else 1.0  # report KID x 1000
            label = f"{split}.{metric}{'x1000' if metric == 'kid' else ''}"
            cells.append((label, mean * scale, std * scale, len(vals)))
        if args.markdown:
            print("| " + name + " | " + " | ".join(f"{m:.4g} ± {sd:.2g}" for _, m, sd, _ in cells) + " |")
        else:
            print(name)
            for label, m, sd, n in cells:
                print(f"    {label:<34} {m:.5g} ± {sd:.2g}  (n={n})")


if __name__ == "__main__":
    main()
