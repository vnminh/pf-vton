from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | os.PathLike[str]):
    return OmegaConf.load(path)


def save_config(cfg, path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def expand_patch_values(values: torch.Tensor, patch_size: int, hw: Tuple[int, int]) -> torch.Tensor:
    """(B, gh*gw) -> (B,1,H,W), repeating each patch scalar spatially."""
    b, n = values.shape
    h, w = hw
    gh, gw = h // patch_size, w // patch_size
    if n != gh * gw:
        raise ValueError(f"Expected {gh*gw} patch values for latent {hw}, got {n}")
    x = values.view(b, 1, gh, gw)
    return x.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)


def latent_edit_mask(pixel_mask: torch.Tensor, latent_hw: Tuple[int, int]) -> torch.Tensor:
    """Conservatively convert a pixel edit mask to latent-cell occupancy.

    A latent cell is editable when *any* source pixel covered by it is editable.
    Area/nearest resizing can silently drop thin sleeves and boundary strips;
    adaptive max pooling preserves that support. Pixel-space compositing still
    guarantees exact preservation outside the requested RGB mask.
    """
    if pixel_mask.ndim == 3:
        pixel_mask = pixel_mask[:, None]
    pooled = F.adaptive_max_pool2d(pixel_mask.float(), latent_hw)
    return (pooled > 0.0).to(pixel_mask.dtype)


def token_edit_mask(latent_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """A PFT token is editable if any part of its latent patch is editable."""
    pooled = F.max_pool2d(latent_mask, kernel_size=patch_size, stride=patch_size)
    return (pooled.flatten(1) > 1e-4)


def pool_valid_mask(mask: torch.Tensor | None, grid_hw: Tuple[int, int], threshold: float = 0.02) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, None]
    pooled = F.adaptive_avg_pool2d(mask.float(), grid_hw)
    return pooled.flatten(1) > threshold


def edge_strength(rgb_m11: torch.Tensor) -> torch.Tensor:
    """Cheap differentiable high-frequency weighting map, no extra network."""
    gray = rgb_m11.mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy).clamp_min(0)


def normalized_edge_weights(
    rgb_m11: torch.Tensor,
    target_hw: Tuple[int, int],
    boost: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    e = edge_strength(rgb_m11)
    e = F.adaptive_avg_pool2d(e, target_hw)
    flat = e.flatten(1)
    denom = flat.quantile(0.95, dim=1, keepdim=True).clamp_min(1e-4)
    e = (e / denom[:, :, None, None]).clamp(0, 1)
    w = 1.0 + boost * e
    if mask is not None:
        if mask.shape[-2:] != target_hw:
            mask = F.adaptive_avg_pool2d(mask.float(), target_hw)
        w = w * mask
    return w


def rectangular_pos_from_square(src: torch.Tensor, dst_hw: Tuple[int, int]) -> torch.Tensor:
    """Bicubic-interpolate a pretrained fixed square token positional embedding."""
    if src.ndim != 3 or src.shape[0] != 1:
        raise ValueError(f"Expected [1,N,D] positional embedding, got {tuple(src.shape)}")
    n = src.shape[1]
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"Source positional embedding is not square: N={n}")
    x = src.reshape(1, side, side, src.shape[-1]).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=dst_hw, mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(1, dst_hw[0] * dst_hw[1], src.shape[-1])


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Accept compact release checkpoints and Lightning training checkpoints."""
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    else:
        sd = checkpoint
    if not isinstance(sd, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")

    keys = list(sd.keys())
    # Compact release is already clean. Training checkpoints may contain model./ema_model.
    for prefix in ("ema_model.", "model."):
        matches = [k for k in keys if k.startswith(prefix)]
        if matches and len(matches) >= max(1, len(keys) // 3):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return dict(sd)


def count_trainable(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def detach_metrics(metrics: Mapping[str, torch.Tensor | float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v.detach().float().mean().cpu())
        else:
            out[k] = float(v)
    return out
