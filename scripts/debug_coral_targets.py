#!/usr/bin/env python3
"""Visualize the training-only DINO correspondence targets used by CORAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from vton_ext.coral import DINOv3CoralTeacher
from vton_ext.data import VitonHDDataset
from vton_ext.utils import latent_edit_mask, pool_valid_mask, token_edit_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vton_joint_512x384.yaml")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--phase", default="train", choices=("train", "test"))
    parser.add_argument("--output", default="outputs/coral_target_debug")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    person_hw = (
        int(cfg.model.latent_hw[0]) // int(cfg.model.patch_size),
        int(cfg.model.latent_hw[1]) // int(cfg.model.patch_size),
    )
    garment_hw = (
        int(cfg.model.latent_hw[0]) // int(cfg.model.garment_patch_size),
        int(cfg.model.latent_hw[1]) // int(cfg.model.garment_patch_size),
    )
    dataset = VitonHDDataset(
        root=str(cfg.data.root), phase=args.phase, order="paired",
        size=tuple(cfg.model.image_hw), pairs_file=args.pairs,
        require_cloth_mask=bool(cfg.data.require_cloth_mask),
        require_parse=bool(cfg.data.get("require_parse", False)),
        clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    teacher = DINOv3CoralTeacher(
        str(cfg.weights.dino),
        min_similarity=float(cfg.coral.min_similarity),
        cycle_radius=float(cfg.coral.cycle_radius),
    ).to(device)
    person = batch["image"].to(device)
    garment = batch["garment"].to(device)
    clothing = batch["clothing_mask"].to(device)
    garment_mask = batch["garment_mask"].to(device)
    edit = token_edit_mask(
        latent_edit_mask(clothing, tuple(cfg.model.latent_hw)),
        int(cfg.model.patch_size),
    )
    garment_valid = pool_valid_mask(garment_mask, garment_hw)
    with torch.no_grad():
        targets = teacher.build_targets(
            person_rgb=person,
            garment_rgb=garment,
            edit_token_mask=edit,
            garment_valid=garment_valid,
            person_hw=person_hw,
            garment_hw=garment_hw,
        )

    b = person.shape[0]
    grid = targets.garment_coord.reshape(b, *person_hw, 2)[..., [1, 0]] * 2.0 - 1.0
    routed = F.grid_sample(
        garment.float(), grid.float(), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    routed = F.interpolate(routed, size=person.shape[-2:], mode="nearest")
    reliable = targets.reliable.reshape(b, 1, *person_hw).float()
    reliable = F.interpolate(reliable, size=person.shape[-2:], mode="nearest")
    visible = reliable * routed + (1.0 - reliable)
    similarity = targets.similarity.reshape(b, 1, *person_hw)
    similarity = F.interpolate(similarity, size=person.shape[-2:], mode="nearest")
    similarity_rgb = similarity.clamp(0, 1).expand(-1, 3, -1, -1) * 2.0 - 1.0

    rows = []
    for index in range(b):
        rows.extend([
            garment[index:index + 1].cpu(),
            person[index:index + 1].cpu(),
            routed[index:index + 1].cpu(),
            visible[index:index + 1].cpu(),
            similarity_rgb[index:index + 1].cpu(),
        ])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_image(
        make_grid(torch.cat(rows), nrow=5, normalize=True, value_range=(-1, 1), padding=2),
        output / "targets.png",
    )
    metrics = {
        "reliable_all_fraction": float(targets.reliable.float().mean()),
        "reliable_clothing_fraction": float(
            targets.reliable.float().sum() / edit.float().sum().clamp_min(1.0)
        ),
        "reliable_similarity": float(
            (targets.similarity * targets.reliable).sum()
            / targets.reliable.sum().clamp_min(1)
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
