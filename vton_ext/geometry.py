from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_mask_bounds(
    mask: torch.Tensor,
    threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample (left, right, top, bottom) bounds in [-1, 1]."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}")
    b, _, h, w = mask.shape
    valid = mask[:, 0].float() > float(threshold)
    xs = torch.linspace(-1.0, 1.0, w, device=mask.device, dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=torch.float32)
    xmap = xs.view(1, 1, w).expand(b, h, w)
    ymap = ys.view(1, h, 1).expand(b, h, w)
    inf = torch.full((), float("inf"), device=mask.device)
    ninf = torch.full((), float("-inf"), device=mask.device)
    left = torch.where(valid, xmap, inf).amin(dim=(1, 2))
    right = torch.where(valid, xmap, ninf).amax(dim=(1, 2))
    top = torch.where(valid, ymap, inf).amin(dim=(1, 2))
    bottom = torch.where(valid, ymap, ninf).amax(dim=(1, 2))
    has_valid = valid.flatten(1).any(dim=1)
    left = torch.where(has_valid, left, torch.full_like(left, -1.0))
    right = torch.where(has_valid, right, torch.full_like(right, 1.0))
    top = torch.where(has_valid, top, torch.full_like(top, -1.0))
    bottom = torch.where(has_valid, bottom, torch.full_like(bottom, 1.0))
    return left, right, top, bottom


def mask_aligned_base_grid(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    output_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Aspect-preserving source-mask to target-mask backward affine grid.

    The whole source garment is uniformly fitted inside the target edit-mask
    bounding box. Unlike an identity base, this does not assume that shop cloth
    and the person's torso have identical scale or vertical placement. Uniform
    scale avoids stretching lettering differently along x and y.
    """
    if source_mask.shape[0] != target_mask.shape[0]:
        raise ValueError("source_mask and target_mask batch sizes must match")
    h, w = output_hw or tuple(target_mask.shape[-2:])
    sl, sr, st, sb = _normalized_mask_bounds(source_mask)
    tl, tr, tt, tb = _normalized_mask_bounds(target_mask)
    source_w = (sr - sl).clamp_min(2.0 / max(source_mask.shape[-1] - 1, 1))
    source_h = (sb - st).clamp_min(2.0 / max(source_mask.shape[-2] - 1, 1))
    target_w = (tr - tl).clamp_min(2.0 / max(target_mask.shape[-1] - 1, 1))
    target_h = (tb - tt).clamp_min(2.0 / max(target_mask.shape[-2] - 1, 1))
    # Forward source->target scale. "Contain" preserves the complete logo and
    # garment without anisotropically stretching its typography.
    scale = torch.minimum(target_w / source_w, target_h / source_h).clamp(0.25, 4.0)
    source_cx, source_cy = (sl + sr) * 0.5, (st + sb) * 0.5
    target_cx, target_cy = (tl + tr) * 0.5, (tt + tb) * 0.5

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=target_mask.device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, w, device=target_mask.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = (xx[None] - target_cx[:, None, None]) / scale[:, None, None] + source_cx[:, None, None]
    gy = (yy[None] - target_cy[:, None, None]) / scale[:, None, None] + source_cy[:, None, None]
    # Do not clamp. Out-of-source coordinates must become invalid through
    # grid_sample padding instead of repeating a garment-border pixel.
    return torch.stack([gx, gy], dim=-1)


def erode_confidence(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Soft morphological erosion used only as conservative copy confidence."""
    radius = int(radius)
    if radius <= 0:
        return mask.float().clamp(0.0, 1.0)
    kernel = 2 * radius + 1
    x = mask.float().clamp(0.0, 1.0)
    # min-pool via negated max-pool; explicit zero padding makes image borders
    # uncertain rather than accidentally confident.
    x = F.pad(x, (radius, radius, radius, radius), mode="constant", value=0.0)
    x = -F.max_pool2d(-x, kernel_size=kernel, stride=1)
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
