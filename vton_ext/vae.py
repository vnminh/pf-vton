from __future__ import annotations

import torch

from jutils.nn.kl_autoencoder import AutoencoderKL


def load_sd_vae(checkpoint: str, device: torch.device, dtype: torch.dtype = torch.float32) -> AutoencoderKL:
    # The upstream loader restores tensors to the checkpoint's saved CUDA
    # device before copying them into its CPU module. Load on CPU explicitly
    # to avoid transient duplicate GPU allocations during model startup.
    vae = AutoencoderKL(ckpt_path=None)
    vae.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    vae = vae.to(device=device, dtype=dtype).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images(vae: AutoencoderKL, images_m11: torch.Tensor) -> torch.Tensor:
    # jutils AutoencoderKL includes the SD 0.18215 latent scaling internally.
    return vae.encode(images_m11)


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents)


def decode_latents_with_grad(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Same decode as jutils AutoencoderKL.decode, but allows gradient w.r.t. latents.

    VAE parameters remain frozen. This is used only for optional RGB/edge supervision.
    """
    z = latents / vae.scale + vae.shift
    z = vae.post_quant_conv(z)
    return vae.decoder(z)
