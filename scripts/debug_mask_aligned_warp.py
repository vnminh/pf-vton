#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from vton_ext.data import VitonHDDataset
from vton_ext.geometry import erode_confidence, mask_aligned_base_grid


def warp(garment, garment_mask, target_mask):
    grid = mask_aligned_base_grid(garment_mask, target_mask, garment.shape[-2:])
    rgb = F.grid_sample(
        garment.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    mask = F.grid_sample(
        garment_mask.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    confidence = erode_confidence(mask * target_mask, radius=10)
    rgb = rgb * mask + (-1.0) * (1.0 - mask)
    return rgb, confidence


def main():
    cfg = OmegaConf.load("configs/vton_crossattn_512x384.yaml")
    ds = VitonHDDataset(
        root=str(cfg.data.root), phase="test", order="paired",
        size=tuple(cfg.model.image_hw), pairs_file=cfg.data.test_pairs_file,
        require_cloth_mask=True, require_parse=False,
    )
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)))
    shuffled_garment = batch["garment"].roll(1, 0)
    shuffled_mask = batch["garment_mask"].roll(1, 0)
    paired_warp, _ = warp(batch["garment"], batch["garment_mask"], batch["agnostic_mask"])
    unpaired_warp, confidence = warp(
        shuffled_garment, shuffled_mask, batch["agnostic_mask"]
    )
    source = confidence * unpaired_warp + (1.0 - confidence) * batch["agnostic"]
    rows = []
    for i in range(4):
        confidence_rgb = confidence[i:i+1].expand(-1, 3, -1, -1) * 2.0 - 1.0
        rows.extend([
            batch["agnostic"][i:i+1], batch["image"][i:i+1],
            batch["garment"][i:i+1], paired_warp[i:i+1],
            shuffled_garment[i:i+1], unpaired_warp[i:i+1],
            confidence_rgb, source[i:i+1],
        ])
    grid = make_grid(torch.cat(rows), nrow=8, normalize=True, value_range=(-1, 1), padding=2)
    output = Path("outputs/debug_mask_aligned_warp.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output)
    print(output.resolve())
    print("columns: agnostic | GT | paired garment | paired warp | shuffled garment | shuffled warp | confidence | shuffled source")


if __name__ == "__main__":
    main()
