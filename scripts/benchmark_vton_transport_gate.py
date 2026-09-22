#!/usr/bin/env python3
"""Ablate raw-latent attention transport without retraining the checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from vton_ext.data import VitonHDDataset
from vton_ext.sample import build_model, sample_one
from vton_ext.vae import load_sd_vae


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.broadcast_to(mask, value.shape)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vton_joint_512x384.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--phase", default="train", choices=("train", "test"))
    parser.add_argument("--output", default="outputs/vton_transport_gate_benchmark")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--gates", type=float, nargs="+",
        default=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, -0.1),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    model = build_model(cfg, args.checkpoint, device)
    vae = load_sd_vae(str(cfg.weights.vae), device=device, dtype=torch.float32)
    dataset = VitonHDDataset(
        root=str(cfg.data.root), phase=args.phase, order="paired",
        size=tuple(cfg.model.image_hw), pairs_file=args.pairs,
        require_cloth_mask=bool(cfg.data.require_cloth_mask),
        require_parse=bool(cfg.data.get("require_parse", False)),
        clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    shuffled = dict(batch)
    shuffled["garment"] = batch["garment"].roll(1, dims=0)
    shuffled["garment_mask"] = batch["garment_mask"].roll(1, dims=0)
    identity = dict(batch)
    identity["image"] = batch["garment"]
    identity["agnostic"] = torch.ones_like(batch["agnostic"])
    identity["densepose"] = -torch.ones_like(batch["densepose"])
    identity["agnostic_mask"] = batch["garment_mask"]
    identity["clothing_mask"] = batch["garment_mask"]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gt = batch["image"].to(device)
    clothing = batch["clothing_mask"].to(device)
    garment = batch["garment"].to(device)
    garment_mask = batch["garment_mask"].to(device)
    summary: dict[str, dict[str, float]] = {}
    rows = []

    for gate in args.gates:
        model.attention_transport_gate.data.fill_(gate)
        with torch.autocast("cuda", dtype=precision):
            paired = sample_one(
                model, vae, batch, steps=1, device=device, seed=args.seed,
                start_from_warp=False, solver="euler",
            )
            unpaired = sample_one(
                model, vae, shuffled, steps=1, device=device, seed=args.seed,
                start_from_warp=False, solver="euler",
            )
            copied = sample_one(
                model, vae, identity, steps=1, device=device, seed=args.seed,
                start_from_warp=False, solver="euler",
            )
        key = f"{gate:g}"
        summary[key] = {
            "paired_clothing_l1": masked_mean((paired - gt).abs(), clothing).item(),
            "identity_clothing_l1": masked_mean(
                (copied - garment).abs(), garment_mask
            ).item(),
            "garment_sensitivity": masked_mean(
                (paired - unpaired).abs(), clothing
            ).item(),
        }
        for index in range(paired.shape[0]):
            rows.extend([
                batch["garment"][index:index + 1],
                batch["image"][index:index + 1],
                paired[index:index + 1].float().cpu(),
                copied[index:index + 1].float().cpu(),
                unpaired[index:index + 1].float().cpu(),
            ])

    save_image(
        make_grid(torch.cat(rows), nrow=5, normalize=True, value_range=(-1, 1), padding=2),
        output / "comparison.png",
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
