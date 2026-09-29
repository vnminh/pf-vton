"""Fixed-seed matched, swapped, and absent-garment inference on dev previews.

This diagnostic tests whether the trained model's output actually responds to
the garment condition. Its eight preview cases do not establish full-dev quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from scripts.audit_pfi_timesteps import as_float, contrast_mask, rgb_metrics
from vton_ext.pfi_sample import composite, generate, load_for_inference, prepare_inputs
from vton_ext.pfi_train import load_cfg, make_dataset
from vton_ext.vae import decode_latents, encode_images, load_sd_vae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    model, step = load_for_inference(cfg, args.checkpoint, device)
    vae = load_sd_vae(cfg.weights.vae, device)
    ds = make_dataset(cfg, cfg.data.test_pairs_file, augment=False)
    indices = list(cfg.eval.indices)
    if len(indices) < 2 or len(indices) % 2:
        raise ValueError("Expected an even number of preview cases")
    shift = len(indices) // 2
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results, panels = [], []
    labels = ["garment", "target", "VAE target", "matched", "swapped", "absent"]

    for pos, batch in enumerate(DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=0)):
        idx = indices[pos]
        donor_idx = indices[(pos + shift) % len(indices)]
        donor = ds[donor_idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = prepare_inputs(vae, batch, model, device, mask_open_px=int(cfg.eval.mask_open_px))
            donor_latent = encode_images(vae, donor["garment"][None].to(device)).float()
        donor_mask = F.adaptive_avg_pool2d(donor["garment_mask"][None].to(device).float(), model.latent_hw)
        target = batch["image"].to(device).float()
        cloth = batch["clothing_mask"].to(device).float()
        masks = {"cloth": cloth, "edit": inputs["pixel_mask"],
                 "contrast": contrast_mask(target, cloth) * inputs["pixel_mask"]}
        vae_target = decode_latents(vae, encode_images(vae, target).float())
        noise = torch.randn(inputs["known"].shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(args.seed + idx))
        modes = {
            "matched": (inputs["garment"], inputs["garment_mask"]),
            "swapped": (donor_latent, donor_mask),
            "absent": (torch.zeros_like(inputs["garment"]), torch.zeros_like(inputs["garment_mask"])),
        }
        images = {}
        for name, (garment, garment_mask) in modes.items():
            conditioned = {**inputs, "garment": garment, "garment_mask": garment_mask}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = generate(model, conditioned, sampler="dual_loop", nfe=8,
                                  p=float(cfg.eval.p), n_inner=int(cfg.eval.n_inner),
                                  noise=noise, time_shift=1.0, cfg_scale=1.0)
            images[name] = composite(decode_latents(vae, latent.float()), inputs["agnostic_rgb"],
                                     inputs["pixel_mask"], seam_px=float(cfg.eval.seam_px),
                                     feather_px=float(cfg.eval.feather_px))
            values = rgb_metrics(images[name], target, masks)
            results.append({"index": idx, "person_name": batch["person_name"][0],
                            "donor_index": donor_idx, "mode": name,
                            **{key: as_float(value[0]) for key, value in values.items()}})
        row = [batch["garment"].to(device), target, vae_target,
               images["matched"], images["swapped"], images["absent"]]
        panels.extend(F.interpolate(torch.cat(row), size=(512, 384), mode="area").cpu())
        print(json.dumps({"completed": pos + 1, "total": len(indices),
                          "index": idx, "donor_index": donor_idx}), flush=True)

    save_image((torch.stack(panels) + 1) / 2, out / "matched-swapped-absent.jpg", nrow=len(labels))
    keys = ("cloth_l1", "cloth_highpass_l1", "contrast_l1", "contrast_highpass_l1", "edit_l1")
    summary = {mode: {key: sum(r[key] for r in results if r["mode"] == mode and
                               r[key] is not None) / max(sum(r["mode"] == mode and
                               r[key] is not None for r in results), 1)
                      for key in keys} for mode in modes}
    data = {"checkpoint": args.checkpoint, "step": step, "seed": args.seed,
            "time_shift": 1.0, "nfe": 8, "indices": indices, "columns": labels,
            "donor_rule": "preview index shifted by half the preview list",
            "summary": summary, "cases": results,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "metrics.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
