from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, AutoModel


@dataclass
class CoralTargets:
    garment_index: torch.Tensor
    garment_coord: torch.Tensor
    reliable: torch.Tensor
    similarity: torch.Tensor
    garment_valid: torch.Tensor | None = None


def grid_coordinates(hw: Tuple[int, int], device, dtype=torch.float32) -> torch.Tensor:
    h, w = hw
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([yy, xx], dim=-1).reshape(h * w, 2)


class DINOv3CoralTeacher(nn.Module):
    """Training-only DINOv3 correspondence teacher for CORAL routing loss."""

    def __init__(
        self,
        model_name_or_path: str,
        min_similarity: float = 0.20,
        cycle_radius: float = 0.10,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path).eval()
        self.model.requires_grad_(False)
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)
        patch_size = getattr(self.model.config, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size)
        self.min_similarity = float(min_similarity)
        self.cycle_radius = float(cycle_radius)

    @torch.no_grad()
    def features(self, rgb_m11: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = (rgb_m11.float() + 1.0) * 0.5
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state
        gh = rgb_m11.shape[-2] // self.patch_size
        gw = rgb_m11.shape[-1] // self.patch_size
        n = gh * gw
        # DINO variants can have CLS/register tokens; spatial patch tokens are at the tail.
        tokens = tokens[:, -n:, :]
        feat = tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1]).permute(0, 3, 1, 2)
        if (gh, gw) != tuple(target_hw):
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        feat = feat.flatten(2).transpose(1, 2)
        return F.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def build_targets(
        self,
        person_rgb: torch.Tensor,
        garment_rgb: torch.Tensor,
        edit_token_mask: torch.Tensor,
        garment_valid: torch.Tensor | None,
        person_hw: Tuple[int, int],
        garment_hw: Tuple[int, int],
    ) -> CoralTargets:
        pf = self.features(person_rgb, person_hw)
        gf = self.features(garment_rgb, garment_hw)
        sim = torch.matmul(pf, gf.transpose(-2, -1))
        if garment_valid is not None:
            sim = sim.masked_fill(~garment_valid[:, None, :], -1e4)

        score, best_j = sim.max(dim=-1)
        # Cycle consistency: selected garment token should map back near the source person token.
        best_person_for_garment = sim.argmax(dim=1)  # [B, Ng]
        cycle_i = best_person_for_garment.gather(1, best_j)
        pcoords = grid_coordinates(person_hw, sim.device)
        gcoords = grid_coordinates(garment_hw, sim.device)
        src = pcoords[None].expand(sim.shape[0], -1, -1)
        cycled = pcoords[cycle_i]
        cycle_dist = torch.linalg.vector_norm(src - cycled, dim=-1)

        target_coord = gcoords[best_j]
        reliable = edit_token_mask.bool() & (score >= self.min_similarity) & (cycle_dist <= self.cycle_radius)
        return CoralTargets(best_j, target_coord, reliable, score, garment_valid)


def coral_routing_loss(
    attention_maps: Dict[int, torch.Tensor],
    targets: CoralTargets,
    garment_hw: Tuple[int, int],
    query_weights: torch.Tensor | None = None,
    gaussian_sigma: float = 0.035,
    checkpoint_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not attention_maps:
        zero = targets.garment_coord.sum() * 0.0
        return zero, zero, {"coral_reliable": zero, "coral_target_mass": zero}

    # Preserve heads while optimizing. Averaging first allowed several
    # diffuse/wrong heads to hide behind one useful head and mix unrelated
    # sleeve/body colors in the value payload.
    layers = [a.float() for _, a in sorted(attention_maps.items())]
    coords = grid_coordinates(garment_hw, layers[0].device, layers[0].dtype)
    target_coord = targets.garment_coord.to(coords.dtype)
    distance2 = ((coords[None, None] - target_coord[:, :, None]) ** 2).sum(dim=-1)
    sigma = max(float(gaussian_sigma), 1e-4)
    target_prob = torch.exp(-distance2 / (2.0 * sigma * sigma))
    if targets.garment_valid is not None:
        target_prob = target_prob * targets.garment_valid[:, None].to(target_prob.dtype)
    target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Supervise every selected layer rather than only the barycenter of a layer
    # average. The old barycenter objective allowed broad attention that mixed
    # sleeve/body colors while still landing at the correct mean coordinate.
    def layer_terms(attention, target_prob):
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        p = attention.clamp_min(1e-8)
        logp = p.log()
        return (-(target_prob[:, None] * logp).sum(dim=-1).mean(dim=1),
                -(p * logp).sum(dim=-1) / math.log(max(p.shape[-1], 2)))

    layer_ce, layer_entropy = [], []
    for attention in layers:
        if checkpoint_loss and torch.is_grad_enabled() and attention.requires_grad:
            ce_i, ent_i = checkpoint(layer_terms, attention, target_prob, use_reentrant=False)
        else:
            ce_i, ent_i = layer_terms(attention, target_prob)
        layer_ce.append(ce_i)
        layer_entropy.append(ent_i)
    ce = torch.stack(layer_ce).mean(dim=0) / math.log(max(layers[0].shape[-1], 2))
    entropy = torch.stack(layer_entropy).mean(dim=(0, 2))
    # Diagnostics do not need a backward graph or a stack of all B*H*N*N maps.
    with torch.no_grad():
        attention = sum(a.mean(dim=1) for a in layers) / len(layers)

    reliable = targets.reliable.to(attention.dtype)
    weights = reliable
    if query_weights is not None:
        weights = weights * query_weights.to(attention.dtype)
    denom = weights.sum().clamp_min(1.0)
    corr_loss = (ce * weights).sum() / denom
    ent_loss = (entropy * weights).sum() / denom

    target_mass = attention.gather(-1, targets.garment_index[..., None]).squeeze(-1)
    local_mass = (attention * (distance2 <= (2.0 * sigma) ** 2).to(attention.dtype)).sum(dim=-1)
    metrics = {
        "coral_reliable": reliable.mean(),
        "coral_target_mass": (target_mass * reliable).sum() / reliable.sum().clamp_min(1.0),
        "coral_local_mass": (local_mass * reliable).sum() / reliable.sum().clamp_min(1.0),
        "coral_entropy": (entropy * reliable).sum() / reliable.sum().clamp_min(1.0),
        "coral_similarity": (targets.similarity * reliable).sum() / reliable.sum().clamp_min(1.0),
    }
    return corr_loss, ent_loss, metrics


def transport_routing_loss(
    transport_coord: torch.Tensor | None,
    targets: CoralTargets,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the dense local garment sampler with reliable CORAL matches."""
    if transport_coord is None:
        return targets.garment_coord.sum() * 0.0
    error = ((transport_coord.float() - targets.garment_coord.float()) ** 2).sum(dim=-1)
    weights = targets.reliable.to(error.dtype)
    if query_weights is not None:
        weights = weights * query_weights.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)
