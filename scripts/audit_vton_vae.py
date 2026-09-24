"""Measure the VAE reconstruction limit on held-out lettering, without a DiT."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from vton_ext.data import VitonHDDataset
from vton_ext.vae import load_sd_vae, encode_images, decode_latents


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/vton_joint_512x384.yaml')
    p.add_argument('--output', default='outputs/vae-logo-audit')
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    vae = load_sd_vae(str(cfg.weights.vae), device)
    indices = [23, 44, 65, 93, 115, 152, 154, 161, 163, 209, 232, 241]
    ds = VitonHDDataset(root=str(cfg.data.root), phase='train', order='paired',
                       size=tuple(cfg.model.image_hw), pairs_file=str(cfg.data.test_pairs_file),
                       require_cloth_mask=True, require_parse=True,
                       clothing_labels=list(cfg.data.clothing_labels), augment=False)
    rows, metrics = [], []
    for index, batch in zip(indices, DataLoader(Subset(ds, indices), batch_size=1)):
        source, target = batch['garment'].to(device), batch['image'].to(device)
        mask = batch['clothing_mask'].to(device)
        recon = decode_latents(vae, encode_images(vae, torch.cat([source, target]))).clamp(-1, 1)
        error = (recon[1:] - target).abs()
        hp = lambda x: x - F.avg_pool2d(x, 5, 1, 2)
        metrics.append({'index': index,
                        'clothing_l1': ((error * mask).sum() / (3 * mask.sum())).item(),
                        'highpass_l1': (((hp(recon[1:]) - hp(target)).abs() * mask).sum()
                                         / (3 * mask.sum())).item()})
        rows.extend([source.cpu(), recon[:1].cpu(), target.cpu(), recon[1:].cpu()])
    save_image(torch.cat(rows), output / 'source_reconstruction_target_reconstruction.png',
               nrow=4, normalize=True, value_range=(-1, 1))
    result = {'per_sample': metrics, 'mean': {
        key: sum(row[key] for row in metrics) / len(metrics)
        for key in ('clothing_l1', 'highpass_l1')}}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
