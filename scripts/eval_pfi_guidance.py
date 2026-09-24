"""Compare PFI samplers and garment-CFG scales on fixed paired dev cases.

    PYTHONPATH=. python scripts/eval_pfi_guidance.py --config C --checkpoint K \
        --output DIR --specs dual_loop:8:1.0 dual_loop:8:2.0 euler:8:2.0

Spec = sampler:nfe:cfg[:lo-hi], lo-hi = token-time interval where CFG applies. Same noise for every spec. Metrics are per-image means
(L1, high-pass L1 and a target-derived contrast-region proxy, not OCR).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from scripts.audit_pfi_timesteps import contrast_mask, highpass, per_image_mean
from vton_ext.pfi_sample import composite, generate, load_for_inference, prepare_inputs
from vton_ext.pfi_train import load_cfg, make_dataset
from vton_ext.vae import decode_latents, load_sd_vae


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="default: eval.indices")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    device = torch.device("cuda")
    model, step = load_for_inference(cfg, args.checkpoint, device, use_ema=False)
    vae = load_sd_vae(cfg.weights.vae, device)
    indices = args.indices or list(cfg.eval.indices)
    ds = Subset(make_dataset(cfg, cfg.data.test_pairs_file, augment=False), indices)
    specs = []
    for s in args.specs:
        parts = s.split(":")
        interval = tuple(float(v) for v in parts[3].split("-")) if len(parts) > 3 else (0.0, 1.0)
        specs.append((parts[0], int(parts[1]), float(parts[2]), interval))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sums, rows = {}, []
    for bi, batch in enumerate(DataLoader(ds, batch_size=args.batch_size, shuffle=False)):
        gen = torch.Generator(device=device).manual_seed(int(cfg.eval.seed) + bi)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = prepare_inputs(vae, batch, model, device, mask_open_px=int(cfg.eval.mask_open_px))
            noise = torch.randn(inputs["known"].shape, device=device, generator=gen)
            latents = [generate(model, inputs, nfe=nfe, sampler=name, p=float(cfg.eval.p),
                                n_inner=int(cfg.eval.n_inner), cfg_scale=scale, cfg_interval=iv, noise=noise)
                       for name, nfe, scale, iv in specs]
        target = batch["image"].to(device)
        cloth = batch["clothing_mask"].to(device)
        masks = {"edit": inputs["pixel_mask"], "cloth": cloth, "contrast": contrast_mask(target, cloth)}
        row = [batch["garment"].to(device), target]
        for (name, nfe, scale, iv), lat in zip(specs, latents):
            rgb = composite(decode_latents(vae, lat.float()).float(), inputs["agnostic_rgb"], inputs["pixel_mask"])
            key = f"{name}{nfe}_cfg{scale:g}" + ("" if iv == (0.0, 1.0) else f"_t{iv[0]:g}-{iv[1]:g}")
            err, hp = (rgb - target).abs(), (highpass(rgb) - highpass(target)).abs()
            for mname, m in masks.items():
                for metric, values in ((f"{mname}_l1", err), (f"{mname}_highpass_l1", hp)):
                    v = per_image_mean(values, m)
                    v = v[torch.isfinite(v)]
                    acc = sums.setdefault(key, {}).setdefault(metric, [0.0, 0])
                    acc[0] += float(v.sum())
                    acc[1] += int(v.numel())
            row.append(rgb)
        rows.append(torch.stack(row, 1).flatten(0, 1))
    save_image((torch.cat(rows) + 1) / 2, out / "grid.jpg", nrow=2 + len(specs))
    result = {"step": step, "indices": indices, "columns": ["garment", "target"] + list(sums),
        "metrics": {k: {m: s / max(n, 1) for m, (s, n) in v.items()} for k, v in sums.items()}}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    for k, v in result["metrics"].items():
        print(k, " ".join(f"{m}={x:.4f}" for m, x in v.items()))


if __name__ == "__main__":
    main()
