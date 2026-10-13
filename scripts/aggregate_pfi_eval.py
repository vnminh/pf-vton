"""Mean and standard deviation over sampling seeds of PFI-VTON test evaluations.

    python scripts/aggregate_pfi_eval.py --root /path/to/experiments [--mode full] [--markdown]

Reads two layouts under --root:
* Modal (docs/experiment/infer_modal.py): <tag>/<mode>/metadata.json and
  cfg<w>/<pair|unpair>/metrics/summary.json;
* server (scripts/eval_pfi_test.py): <run>/metadata.json and
  cfg<w>/<pair|unpair>/summary.json (only with --mode full; runs with --limit are skipped). Runs that differ only in their
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
    "pair": ("mse_mean", "mse_mask_mean", "ssim_mean", "lpips_mean", "fid", "kid", "sample_seconds_per_image",
             "denoiser_calls_per_image"),
    "unpair": ("fid", "kid", "sample_seconds_per_image", "denoiser_calls_per_image"),
}
SETTING_KEYS = ("checkpoint", "checkpoint_step", "config", "sampler", "nfe", "p", "context_ratio")


def collect(root: Path, mode: str):
    groups = defaultdict(lambda: defaultdict(list))  # setting -> (cfg, split, metric) -> values
    seeds = defaultdict(set)
    metas = sorted(root.glob(f"*/{mode}/metadata.json"))
    if mode == "full":
        metas += [m for m in sorted(root.glob("*/metadata.json")) if not json.loads(m.read_text()).get("limit")]
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        scales = meta.get("cfg_scales") or sorted(float(d.name[3:]) for d in meta_path.parent.glob("cfg*") if d.is_dir())
        for scale in scales:
            setting = tuple((k, meta.get(k)) for k in SETTING_KEYS) + (("cfg", scale),)
            for split in ("pair", "unpair"):
                summary_path = meta_path.parent / f"cfg{scale:g}" / split / "metrics" / "summary.json"
                if not summary_path.exists():
                    summary_path = meta_path.parent / f"cfg{scale:g}" / split / "summary.json"
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
