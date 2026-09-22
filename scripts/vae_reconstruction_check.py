#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from vton_ext.vae import decode_latents, encode_images, load_sd_vae

p = argparse.ArgumentParser(description="Check whether the SD VAE itself preserves a small garment logo/text.")
p.add_argument("image")
p.add_argument("--vae", default="checkpoints/sd_ae.ckpt")
p.add_argument("--output", default="vae_reconstruction.png")
a = p.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is recommended/required for this repository")
device = torch.device("cuda")
im = Image.open(a.image).convert("RGB").resize((384, 512), Image.Resampling.LANCZOS)
x = TF.pil_to_tensor(im).float().unsqueeze(0).to(device) / 127.5 - 1.0
vae = load_sd_vae(a.vae, device)
z = encode_images(vae, x)
y = decode_latents(vae, z).clamp(-1, 1)
save_image(torch.cat([x, y], dim=0), a.output, nrow=2, normalize=True, value_range=(-1, 1))
print("saved", Path(a.output).resolve())
print("left=original, right=VAE reconstruction")
