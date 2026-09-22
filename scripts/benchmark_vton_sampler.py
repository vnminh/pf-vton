#!/usr/bin/env python3
"""Compare ODE solvers/step counts on the exact same VTON examples and noise."""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--output", default="outputs/vton_sampler_benchmark")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("euler:1", "euler:5", "euler:10", "euler:25", "euler:50", "heun:5", "heun:10", "heun:25"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    model = build_model(cfg, args.checkpoint, device)
    vae = load_sd_vae(str(cfg.weights.vae), device=device, dtype=torch.float32)
    dataset = VitonHDDataset(
        root=str(cfg.data.root),
        phase=args.phase,
        order="paired",
        size=tuple(cfg.model.image_hw),
        pairs_file=args.pairs,
        require_cloth_mask=bool(cfg.data.require_cloth_mask),
        require_parse=bool(cfg.data.get("require_parse", False)),
        clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gt = batch["image"].to(device)
    mask = batch["clothing_mask"].to(device)
    summary: dict[str, dict[str, float]] = {}
    comparison = []

    for specification in args.variants:
        solver, raw_steps = specification.split(":", 1)
        steps = int(raw_steps)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=precision):
            generated = sample_one(
                model,
                vae,
                batch,
                steps=steps,
                device=device,
                seed=args.seed,
                source_noise_std=float(cfg.flow.get("source_noise_std", 0.0)),
                source_confidence_erode_px=int(cfg.flow.get("source_confidence_erode_px", 4)),
                start_from_warp=bool(cfg.flow.get("start_from_warp", False)),
                solver=solver,
            )
        elapsed = time.perf_counter() - started
        l1 = masked_mean((generated - gt).abs(), mask).item()
        mse = masked_mean((generated - gt).square(), mask).item()
        summary[specification] = {
            "clothing_l1": l1,
            "clothing_psnr": -10.0 * torch.log10(torch.tensor(max(mse, 1e-12))).item(),
            "seconds": elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
        save_image(
            generated.float().cpu(),
            output / f"{solver}_{steps:03d}.png",
            nrow=args.batch_size,
            normalize=True,
            value_range=(-1, 1),
        )
        for index in range(generated.shape[0]):
            comparison.extend([
                batch["garment"][index:index + 1],
                batch["image"][index:index + 1],
                generated[index:index + 1].float().cpu(),
            ])

    grid = make_grid(
        torch.cat(comparison),
        nrow=3,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    save_image(grid, output / "comparison.png")
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
