"""Compare a resumed PFI checkpoint with a backup on identical paired cases.

The baseline may be transferred from another resolution with --transfer-baseline.
That comparison assesses a recovery initializer, not the pre-interruption model.
Outputs contain numeric metrics, with an optional selected-case preview.
Neither checkpoint nor optimizer is modified.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from vton_ext.pairs import split_manifest
from vton_ext.pfi_train import build_model, load_cfg, make_dataset
from vton_ext.pfi_sample import composite, generate, load_for_inference, prepare_inputs
from vton_ext.vae import decode_latents, load_sd_vae


def masked_per_image(error, mask):
    weights = mask.expand_as(error)
    return (error * weights).flatten(1).sum(1) / weights.flatten(1).sum(1).clamp_min(1)


def highpass(rgb):
    return rgb - F.avg_pool2d(F.pad(rgb, (2, 2, 2, 2), mode="reflect"), 5, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--transfer-baseline", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--preview-output", help="optional comparison image; columns: garment, agnostic, target, latest, baseline")
    ap.add_argument("--preview-count", type=int, default=3)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    fit = make_dataset(cfg, cfg.data.pairs_file, augment=False)
    dev = make_dataset(cfg, cfg.data.test_pairs_file, augment=False)
    fingerprints = split_manifest(fit.pair_path, dev.pair_path)
    indices = list(cfg.eval.indices)
    batches = list(DataLoader(Subset(dev, indices), batch_size=1, shuffle=False))
    vae = load_sd_vae(cfg.weights.vae, device)
    cached, records, preview = [], [], {"latest": [], "baseline": []}
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for label, path in (("latest", args.checkpoint), ("baseline", args.baseline)):
        if label == "baseline" and args.transfer_baseline:
            cfg.weights.init_from = path
            model = build_model(cfg).to(device).eval()
            step = None
        else:
            model, step = load_for_inference(cfg, path, device, use_ema=False)
        nonfinite = [name for name, value in model.state_dict().items()
                     if value.is_floating_point() and not bool(torch.isfinite(value).all())]
        if nonfinite:
            raise ValueError(f"Nonfinite model tensors: {nonfinite[:5]}")
        for position, batch in enumerate(batches):
            if label == "latest":
                torch.manual_seed(int(cfg.eval.seed) + position)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    inputs = prepare_inputs(vae, batch, model, device, mask_open_px=int(cfg.eval.mask_open_px))
                noise = torch.randn(inputs["known"].shape, device=device,
                                    generator=torch.Generator(device=device).manual_seed(int(cfg.eval.seed) + position))
                cached.append((inputs, noise))
            inputs, noise = cached[position]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = generate(model, inputs, sampler="dual_loop", nfe=8, p=float(cfg.eval.p),
                                  n_inner=int(cfg.eval.n_inner), cfg_scale=1.0, noise=noise, time_shift=1.0)
            rgb = composite(decode_latents(vae, latent.float()).float(), inputs["agnostic_rgb"], inputs["pixel_mask"],
                            seam_px=float(cfg.eval.seam_px), feather_px=float(cfg.eval.feather_px))
            target, cloth = batch["image"].to(device), batch["clothing_mask"].to(device)
            values = {"cloth_l1": masked_per_image((rgb - target).abs(), cloth),
                      "edit_l1": masked_per_image((rgb - target).abs(), inputs["pixel_mask"]),
                      "cloth_highpass_l1": masked_per_image((highpass(rgb) - highpass(target)).abs(), cloth)}
            record = {"label": label, "step": step, "index": indices[position],
                      **{k: float(v.item()) for k, v in values.items()}}
            records.append(record)
            if args.preview_output and position < args.preview_count:
                preview[label].append(rgb[0].cpu())
            print(json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    keys = ("cloth_l1", "edit_l1", "cloth_highpass_l1")
    summary = {label: {k: sum(r[k] for r in records if r["label"] == label) / len(indices) for k in keys}
               for label in ("latest", "baseline")}
    result = {"pairs": fingerprints, "sampler": "dual_loop8_cfg1_shift1", "indices": indices,
              "baseline_transferred": args.transfer_baseline, "checkpoints": {"latest": args.checkpoint, "baseline": args.baseline},
              "summary": summary, "cases": records}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.preview_output:
        panels = []
        for i in range(min(args.preview_count, len(indices))):
            panels.extend([batches[i]["garment"][0], cached[i][0]["agnostic_rgb"][0].cpu(),
                           batches[i]["image"][0], preview["latest"][i], preview["baseline"][i]])
        image_path = Path(args.preview_output)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # A 512px-high comparison is sufficient for inspection and limits transfer size.
        grid = F.interpolate(torch.stack(panels), size=(512, 384), mode="area")
        save_image((grid + 1) / 2, image_path, nrow=5)
    print(json.dumps({"summary": summary, "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
