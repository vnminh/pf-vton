from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from patch_flow.timestep_schedules import LogitNormalTruncatedGaussian

from .coral import DINOv3CoralTeacher, coral_routing_loss
from .data import VitonHDDataset
from .model import VTONPatchForcingDiT
from .sample import build_warp_source_latent, sample_one, warp_source_rgb
from .utils import (
    count_trainable,
    detach_metrics,
    expand_patch_values,
    latent_edit_mask,
    normalized_edge_weights,
    token_edit_mask,
)
from .vae import decode_latents_with_grad, encode_images, load_sd_vae


def _sample_times(cfg, batch: int, tokens: int, device, dtype, step: int = 0) -> tuple[torch.Tensor, dict]:
    sampler = LogitNormalTruncatedGaussian(
        std=float(cfg.flow.ltg_std),
        loc=float(cfg.flow.ltg_loc),
        scale=float(cfg.flow.ltg_scale),
    )
    asynchronous_t = sampler((batch, tokens), device=device, dtype=dtype)
    if bool(cfg.flow.get("synchronous_uniform", True)):
        synchronous_base = torch.rand(batch, 1, device=device, dtype=dtype)
    else:
        synchronous_base = sampler.get_t_bar(batch, device=device, dtype=dtype)[:, None]
    synchronous_t = synchronous_base.expand(-1, tokens)
    synchronous_probability = float(cfg.flow.get("synchronous_probability", 1.0))
    synchronous = torch.rand(batch, device=device) < synchronous_probability
    t = torch.where(synchronous[:, None], synchronous_t, asynchronous_t)

    # Modes must be mutually exclusive. The legacy "pure_noise" config name now
    # means an exact t=0 warp-source endpoint; detail selects late refinement.
    grounding = step < int(cfg.flow.get("grounding_steps", 0))
    pure_probability = float(
        cfg.flow.get("grounding_pure_noise_probability", cfg.flow.pure_noise_probability)
        if grounding else cfg.flow.pure_noise_probability
    )
    detail_probability = float(
        cfg.flow.get("grounding_detail_refine_probability", cfg.flow.detail_refine_probability)
        if grounding else cfg.flow.detail_refine_probability
    )
    if pure_probability + detail_probability > 1.0:
        raise ValueError("pure_noise_probability + detail_refine_probability must be <= 1")
    mode = torch.rand(batch, device=device)
    pure = mode < pure_probability
    detail = (mode >= pure_probability) & (mode < pure_probability + detail_probability)
    t[pure] = 0.0

    if detail.any():
        lo, hi = map(float, cfg.flow.detail_time_range)
        base = torch.empty(int(detail.sum()), 1, device=device, dtype=dtype).uniform_(lo, hi)
        if bool(cfg.flow.get("synchronous_detail", True)):
            t[detail] = base.expand(-1, tokens)
        else:
            local = (
                base
                - torch.rand(int(detail.sum()), tokens, device=device, dtype=dtype) * 0.12
            ).clamp(0, 1)
            t[detail] = local
    return t, {
        "time_pure_fraction": pure.float().mean(),
        "time_detail_fraction": detail.float().mean(),
        "time_token_mean": t.mean(),
        "time_synchronous_fraction": synchronous.float().mean(),
        "grounding_active": torch.tensor(float(grounding), device=device),
    }


def _masked_mean(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < x.ndim:
        weight = weight.unsqueeze(1)
    weight = torch.broadcast_to(weight, x.shape)
    return (x * weight).sum() / weight.sum().clamp_min(1.0)


def _edge_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    def grads(x):
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy
    pdx, pdy = grads(pred)
    tdx, tdy = grads(target)
    mx = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    my = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    return _masked_mean((pdx - tdx).abs(), mx) + _masked_mean((pdy - tdy).abs(), my)


def _chroma(x: torch.Tensor) -> torch.Tensor:
    """Two opponent-color channels; unlike grayscale edges these retain color blocks."""
    r, g, b = ((x.float() + 1.0) * 0.5).unbind(dim=1)
    cb = b - 0.5 * (r + g)
    cr = r - 0.5 * (g + b)
    return torch.stack([cb, cr], dim=1)


def _block_color_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match regional color and chroma boundaries at sleeve/logo/body scales."""
    loss = pred.sum() * 0.0
    pred_chroma, target_chroma = _chroma(pred), _chroma(target)
    for kernel in (8, 16, 32):
        pooled_mask = F.avg_pool2d(mask.float(), kernel, kernel)
        loss = loss + _masked_mean(
            (F.avg_pool2d(pred, kernel, kernel) - F.avg_pool2d(target, kernel, kernel)).abs(),
            pooled_mask,
        )
    loss = loss / 3.0
    return loss + _edge_loss(pred_chroma, target_chroma, mask)


def _counterfactual_appearance_pair(
    target: torch.Tensor,
    garment: torch.Tensor,
    target_clothing_mask: torch.Tensor,
    garment_mask: torch.Tensor,
    *,
    probability: float,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one random pointwise color map to both views of the same garment.

    A VITON-HD person is observed with only one garment, so ordinary paired
    reconstruction admits a person->appearance shortcut. Applying the same
    random RGB map to the shop garment and its worn pixels creates a valid
    counterfactual target without estimating a warp. Geometry, lettering and
    edges are unchanged; only their appearance is varied. Consequently the
    network must read colors from the garment condition instead of memorizing
    the person/garment pair.
    """
    probability = float(probability)
    strength = max(float(strength), 0.0)
    selected = torch.rand(target.shape[0], device=target.device) < probability
    if not selected.any() or strength == 0.0:
        return target, garment, selected

    out_target = target.clone()
    out_garment = garment.clone()
    eye = torch.eye(3, device=target.device, dtype=torch.float32)
    for i in selected.nonzero(as_tuple=False).flatten().tolist():
        # Mostly retain channel identity, but permit enough cross-channel mixing
        # to turn a fixed training garment into genuinely different colorways.
        diagonal = torch.empty(3, device=target.device).uniform_(
            1.0 - 0.45 * strength, 1.0 + 0.45 * strength
        )
        mixing = torch.empty(3, 3, device=target.device).uniform_(
            -0.18 * strength, 0.18 * strength
        )
        matrix = eye * diagonal[:, None] + mixing * (1.0 - eye)
        bias = torch.empty(3, device=target.device).uniform_(
            -0.18 * strength, 0.18 * strength
        )
        gamma = torch.empty(3, device=target.device).uniform_(
            max(0.45, 1.0 - 0.45 * strength), 1.0 + 0.55 * strength
        )
        tone_amplitude = torch.empty(3, device=target.device).uniform_(
            -0.06 * strength, 0.06 * strength
        )
        tone_phase = torch.empty(3, device=target.device).uniform_(0.0, 2.0 * math.pi)

        def remap(image: torch.Tensor) -> torch.Tensor:
            rgb = ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            rgb = torch.einsum("cd,dhw->chw", matrix, rgb) + bias[:, None, None]
            rgb = rgb.clamp(0.0, 1.0).pow(gamma[:, None, None])
            rgb = rgb + tone_amplitude[:, None, None] * torch.sin(
                2.0 * math.pi * rgb + tone_phase[:, None, None]
            )
            return rgb.clamp(0.0, 1.0) * 2.0 - 1.0

        mapped_target = remap(target[i])
        mapped_garment = remap(garment[i])
        tm = target_clothing_mask[i].float()
        gm = garment_mask[i].float()
        out_target[i] = tm * mapped_target + (1.0 - tm) * target[i]
        out_garment[i] = gm * mapped_garment + (1.0 - gm) * garment[i]
    return out_target, out_garment, selected


def _laplacian_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Multi-scale high-pass loss for logos, lettering, seams, and fabric texture."""
    total = pred.sum() * 0.0
    p, q, m = pred, target, mask
    for level in range(3):
        p_low = F.avg_pool2d(p, 3, 1, 1)
        q_low = F.avg_pool2d(q, 3, 1, 1)
        total = total + _masked_mean(((p - p_low) - (q - q_low)).abs(), m)
        if level < 2:
            p = F.avg_pool2d(p, 2, 2)
            q = F.avg_pool2d(q, 2, 2)
            m = F.avg_pool2d(m.float(), 2, 2)
    return total / 3.0


def _transport_smoothness(offset: torch.Tensor | None) -> torch.Tensor:
    """Keep neighboring samples coherent so lettering is bent rather than shredded."""
    if offset is None:
        return torch.tensor(0.0)
    dx = offset[:, :, :, 1:] - offset[:, :, :, :-1]
    dy = offset[:, :, 1:, :] - offset[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def _joint_garment_attention_loss(
    attention_maps: dict[int, torch.Tensor],
    clothing_tokens: torch.Tensor,
    garment_valid: torch.Tensor,
    *,
    token_hw: tuple[int, int],
    minimum_mass: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the shared transformer from silently ignoring garment tokens.

    This deliberately does not prescribe a shop-to-body coordinate mapping:
    such a target does not exist in VITON-HD and was the source of the previous
    warped bias. It only imposes a lower bound on how much clothing-query
    attention reaches valid garment pixels. Reconstruction decides *where*.
    Entropy and spatial TV are reported for diagnosis, not optimized.
    """
    if not attention_maps:
        raise ValueError("joint garment attention loss requires an attention map")
    query_mask = clothing_tokens.float()
    valid = garment_valid.float()
    losses, masses, entropies, tvs = [], [], [], []
    h, w = token_hw
    for weights in attention_maps.values():
        routed = weights.float() * valid[:, None, :]
        mass = routed.sum(dim=-1).clamp_min(1e-8)
        # A hinge, instead of -log(mass), leaves useful person self-attention
        # intact after the garment route becomes active.
        loss = _masked_mean(
            F.relu(float(minimum_mass) - mass).square(), query_mask
        )
        conditional = routed / mass[..., None]
        entropy = -(conditional * conditional.clamp_min(1e-8).log()).sum(dim=-1)
        normalizer = valid.sum(dim=-1).clamp_min(2.0).log()[:, None]
        entropy = entropy / normalizer

        # Diagnostic only: lower values mean neighboring body queries select
        # spatially coherent regions in the unwarped garment.
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=weights.device),
            torch.linspace(0, 1, w, device=weights.device),
            indexing="ij",
        )
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        expected = torch.matmul(conditional, coords).reshape(-1, h, w, 2)
        q = query_mask.reshape(-1, h, w)
        tv_x = _masked_mean(
            (expected[:, :, 1:] - expected[:, :, :-1]).abs(),
            (q[:, :, 1:] * q[:, :, :-1])[..., None],
        )
        tv_y = _masked_mean(
            (expected[:, 1:] - expected[:, :-1]).abs(),
            (q[:, 1:] * q[:, :-1])[..., None],
        )
        losses.append(loss)
        masses.append(_masked_mean(mass, query_mask))
        entropies.append(_masked_mean(entropy, query_mask))
        tvs.append(tv_x + tv_y)
    route_loss = torch.stack(losses).mean()
    return route_loss, {
        "joint_attention_mass": torch.stack(masses).mean(),
        "joint_attention_entropy": torch.stack(entropies).mean(),
        "joint_attention_tv": torch.stack(tvs).mean(),
    }


def _set_learning_rates(optimizer, base_lrs: list[float], step: int, max_steps: int, cfg) -> None:
    warmup = int(cfg.optim.get("warmup_steps", 0))
    min_ratio = float(cfg.optim.get("min_lr_ratio", 1.0))
    if warmup > 0 and step < warmup:
        scale = max(step + 1, 1) / warmup
    else:
        denom = max(max_steps - warmup, 1)
        progress = min(max((step - warmup) / denom, 0.0), 1.0)
        scale = min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def build_model(cfg) -> VTONPatchForcingDiT:
    m = VTONPatchForcingDiT(
        image_hw=tuple(cfg.model.image_hw),
        latent_hw=tuple(cfg.model.latent_hw),
        patch_size=int(cfg.model.patch_size),
        garment_patch_size=int(cfg.model.garment_patch_size),
        cross_blocks=list(cfg.model.cross_blocks),
        coral_blocks=list(cfg.model.coral_blocks),
        qk_norm=bool(cfg.model.qk_norm),
        cross_logit_scale=float(cfg.model.cross_logit_scale),
        cross_init_gate=float(cfg.model.cross_init_gate),
        use_dense_transport=bool(cfg.model.get("use_dense_transport", True)),
        transport_mask_aligned_base=bool(
            cfg.model.get("transport_mask_aligned_base", False)
        ),
        transport_geometry_only=bool(
            cfg.model.get("transport_geometry_only", False)
        ),
        transport_learned_residual=bool(
            cfg.model.get("transport_learned_residual", True)
        ),
        transport_max_offset=float(cfg.model.get("transport_max_offset", 1.0)),
        transport_fixed_gate=float(cfg.model.get("transport_fixed_gate", 0.02)),
        transport_blocks=list(cfg.model.get("transport_blocks", [20, 24, 27])),
        transport_block_gate=float(cfg.model.get("transport_block_gate", 0.15)),
        transport_output_scale=float(cfg.model.get("transport_output_scale", 1.0)),
        cross_attention_scale=float(cfg.model.get("cross_attention_scale", 0.25)),
        source_condition_scale=float(cfg.model.get("source_condition_scale", 1.0)),
        joint_garment_tokens=bool(cfg.model.get("joint_garment_tokens", False)),
        joint_attention_blocks=list(cfg.model.get("joint_attention_blocks", [24])),
        joint_highres_cross_attention=bool(
            cfg.model.get("joint_highres_cross_attention", False)
        ),
        attention_transport=bool(cfg.model.get("attention_transport", False)),
        attention_transport_block=int(
            cfg.model.get("attention_transport_block", 27)
        ),
        attention_transport_init_gate=float(
            cfg.model.get("attention_transport_init_gate", 0.0)
        ),
        dedicated_router=bool(cfg.model.get("dedicated_router", True)),
        router_dim=int(cfg.model.get("router_dim", 64)),
        final_detail_attention=bool(
            cfg.model.get("final_detail_attention", False)
        ),
        final_detail_dim=int(cfg.model.get("final_detail_dim", 256)),
        final_detail_heads=int(cfg.model.get("final_detail_heads", 8)),
    )
    report = m.load_pretrained_pft(str(cfg.weights.pft))
    print(f"Loaded {report['loaded_keys']} compatible pretrained PFT tensors")
    m.configure_trainable(
        train_self_attention=bool(cfg.optim.train_self_attention),
        train_final_layer=bool(cfg.optim.train_final_layer),
        train_transport_flow=bool(cfg.optim.get("train_transport_flow", True)),
        train_transport_output=bool(cfg.optim.get("train_transport_output", True)),
        train_source_condition=bool(cfg.optim.get("train_source_condition", True)),
        train_garment_embedder=bool(cfg.optim.get("train_garment_embedder", True)),
        train_cross_attention=bool(cfg.optim.get("train_cross_attention", True)),
        train_transport_features=bool(cfg.optim.get("train_transport_features", True)),
        self_attention_blocks=list(cfg.optim.get("self_attention_blocks", [])) or None,
    )
    lora_rank = int(cfg.optim.get("lora_rank", 0))
    if lora_rank > 0:
        lora_report = m.add_backbone_lora(
            rank=lora_rank,
            alpha=float(cfg.optim.get("lora_alpha", lora_rank)),
            dropout=float(cfg.optim.get("lora_dropout", 0.0)),
            include_adaln=bool(cfg.optim.get("lora_include_adaln", True)),
        )
        print(
            f"Added LoRA to {lora_report.modules} modules across all "
            f"{len(m.blocks)} blocks ({lora_report.parameters:,} parameters)"
        )
    return m




def _preview_grid(
    batch,
    paired_source: torch.Tensor,
    generated: torch.Tensor,
    t0_reconstruction: torch.Tensor,
    paired_warped: torch.Tensor,
    shuffled_garment: torch.Tensor,
    shuffled_source: torch.Tensor,
    shuffled_warped: torch.Tensor,
    shuffled: torch.Tensor,
) -> torch.Tensor:
    """Expose both paired quality and each stage of the unpaired copy route."""
    rows = []
    count = generated.shape[0]
    for i in range(count):
        rows.extend([
            batch["agnostic"][i:i+1].to(generated.device),
            batch["garment"][i:i+1].to(generated.device),
            batch["densepose"][i:i+1].to(generated.device),
            batch["image"][i:i+1].to(generated.device),
            paired_source[i:i+1],
            generated[i:i+1],
            t0_reconstruction[i:i+1],
            paired_warped[i:i+1],
            shuffled_garment[i:i+1].to(generated.device),
            shuffled_source[i:i+1],
            shuffled_warped[i:i+1],
            shuffled[i:i+1],
        ])
    images = torch.cat(rows, dim=0).detach().float().cpu()
    return make_grid(images, nrow=12, normalize=True, value_range=(-1, 1), padding=2)


def _noise_preview_grid(
    batch,
    generated: torch.Tensor,
    t0_reconstruction: torch.Tensor,
    shuffled_garment: torch.Tensor,
    shuffled: torch.Tensor,
) -> torch.Tensor:
    """Preview the actual pure-noise task without misleading warp columns."""
    rows = []
    for i in range(generated.shape[0]):
        rows.extend([
            batch["agnostic"][i:i+1].to(generated.device),
            batch["garment"][i:i+1].to(generated.device),
            batch["densepose"][i:i+1].to(generated.device),
            batch["image"][i:i+1].to(generated.device),
            t0_reconstruction[i:i+1],
            generated[i:i+1],
            shuffled_garment[i:i+1].to(generated.device),
            shuffled[i:i+1],
        ])
    images = torch.cat(rows, dim=0).detach().float().cpu()
    return make_grid(images, nrow=8, normalize=True, value_range=(-1, 1), padding=2)


def _identity_preview_grid(batch, generated: torch.Tensor) -> torch.Tensor:
    """Reference/reconstruction pairs for the exact-copy auxiliary task."""
    rows = []
    for i in range(generated.shape[0]):
        rows.extend([
            batch["garment"][i:i+1].to(generated.device),
            generated[i:i+1],
        ])
    return make_grid(
        torch.cat(rows, dim=0).detach().float().cpu(),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )


@torch.no_grad()
def _t0_transport_preview(
    model,
    vae,
    batch,
    *,
    device,
    seed: int,
    source_noise_std: float,
    source_confidence_erode_px: int,
    start_from_warp: bool,
):
    """One-step endpoint prediction from the same source used by inference."""
    agnostic = batch["agnostic"].to(device)
    pose = batch["densepose"].to(device)
    mask_px = batch["agnostic_mask"].to(device)
    garment = batch["garment"].to(device)
    garment_mask_px = batch["garment_mask"].to(device)

    all_z = encode_images(vae, torch.cat([agnostic, pose, garment], dim=0))
    z_agnostic, z_pose, z_garment = all_z.chunk(3, dim=0)
    mask_lat = latent_edit_mask(mask_px, model.latent_hw)
    garment_mask_lat = latent_edit_mask(garment_mask_px, model.latent_hw)
    edit_tokens = token_edit_mask(mask_lat, model.patch_size)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        z_agnostic.shape, generator=generator, device=device, dtype=z_agnostic.dtype
    )
    z_source = source_rgb = warped = source_confidence = None
    if start_from_warp:
        z_source, source_rgb, warped, _, source_confidence = build_warp_source_latent(
            model,
            vae,
            agnostic=agnostic,
            pose=pose,
            mask_px=mask_px,
            garment=garment,
            garment_mask_px=garment_mask_px,
            z_agnostic=z_agnostic,
            z_pose=z_pose,
            z_garment=z_garment,
            mask_lat=mask_lat,
            garment_mask_lat=garment_mask_lat,
            confidence_erode_px=source_confidence_erode_px,
        )
        sigma = min(max(float(source_noise_std), 0.0), 0.999)
        source_scale = (1.0 - sigma * sigma) ** 0.5
        trusted_source = source_scale * z_source + sigma * noise
        x0 = source_confidence * trusted_source + (1.0 - source_confidence) * noise
    else:
        x0 = noise
    xt = mask_lat * x0 + (1.0 - mask_lat) * z_agnostic
    t = torch.where(
        edit_tokens,
        torch.zeros_like(edit_tokens, dtype=xt.dtype),
        torch.ones_like(edit_tokens, dtype=xt.dtype),
    )
    out = model(
        xt=xt,
        t=t,
        agnostic_latent=z_agnostic,
        densepose_latent=z_pose,
        agnostic_mask_latent=mask_lat,
        garment_latent=z_garment,
        garment_mask_latent=garment_mask_lat,
        garment_rgb=garment,
        garment_mask_rgb=garment_mask_px,
        source_latent=z_source,
        source_confidence_latent=source_confidence,
        edit_token_mask=edit_tokens,
        return_uncertainty=True,
    )
    predicted_z = xt + mask_lat * out["velocity"]
    reconstructed = decode_latents_with_grad(vae, predicted_z.float()).clamp(-1, 1)
    reconstructed = mask_px * reconstructed + (1.0 - mask_px) * agnostic

    if warped is not None:
        warped = mask_px * warped + (1.0 - mask_px) * agnostic
    return reconstructed, warped, source_rgb


@torch.no_grad()
def _write_preview(model, vae, batch, *, step: int, cfg, writer, out_dir: Path, device):
    unwrapped = model
    was_training = unwrapped.training
    unwrapped.eval()
    precision = str(cfg.train.mixed_precision).lower()
    shuffled_batch = dict(batch)
    shuffled_batch["garment"] = batch["garment"].roll(1, dims=0)
    shuffled_batch["garment_mask"] = batch["garment_mask"].roll(1, dims=0)
    start_from_warp = bool(cfg.flow.get("start_from_warp", True))
    identity_probability = float(
        cfg.loss.get("garment_identity_probability", 0.0)
    )
    identity_batch = None
    if identity_probability > 0:
        identity_batch = dict(batch)
        identity_batch["image"] = batch["garment"]
        identity_batch["agnostic"] = torch.ones_like(batch["agnostic"])
        identity_batch["densepose"] = -torch.ones_like(batch["densepose"])
        identity_batch["agnostic_mask"] = batch["garment_mask"]
        identity_batch["clothing_mask"] = batch["garment_mask"]

    def generate_pair():
        erode_px = int(cfg.flow.get("source_confidence_erode_px", 10))
        generated, paired_diag = sample_one(
            unwrapped, vae, batch,
            steps=int(cfg.train.get("preview_steps", 50)),
            device=device,
            seed=int(cfg.train.get("preview_seed", 12345)),
            source_noise_std=float(cfg.flow.get("source_noise_std", 0.15)),
            source_confidence_erode_px=erode_px,
            start_from_warp=start_from_warp,
            solver=str(cfg.flow.get("solver", "euler")),
            return_diagnostics=True,
        )
        shuffled, shuffled_diag = sample_one(
            unwrapped, vae, shuffled_batch,
            steps=int(cfg.train.get("preview_steps", 50)),
            device=device,
            seed=int(cfg.train.get("preview_seed", 12345)),
            source_noise_std=float(cfg.flow.get("source_noise_std", 0.15)),
            source_confidence_erode_px=erode_px,
            start_from_warp=start_from_warp,
            solver=str(cfg.flow.get("solver", "euler")),
            return_diagnostics=True,
        )
        t0_reconstruction, _, _ = _t0_transport_preview(
            unwrapped,
            vae,
            batch,
            device=device,
            seed=int(cfg.train.get("preview_seed", 12345)),
            source_noise_std=float(cfg.flow.get("source_noise_std", 0.15)),
            source_confidence_erode_px=erode_px,
            start_from_warp=start_from_warp,
        )
        identity_generated = None
        if identity_batch is not None:
            identity_generated = sample_one(
                unwrapped, vae, identity_batch,
                steps=1,
                device=device,
                seed=int(cfg.train.get("preview_seed", 12345)),
                start_from_warp=False,
                solver="euler",
            )
        return (
            generated, t0_reconstruction, shuffled, identity_generated,
            paired_diag, shuffled_diag,
        )

    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype):
            (
                generated, t0_reconstruction, shuffled, identity_generated,
                paired_diag, shuffled_diag,
            ) = generate_pair()
    else:
        (
            generated, t0_reconstruction, shuffled, identity_generated,
            paired_diag, shuffled_diag,
        ) = generate_pair()

    mask = batch["agnostic_mask"].to(device)
    agnostic = batch["agnostic"].to(device)
    if start_from_warp:
        paired_warped = mask * paired_diag["warped_rgb"] + (1.0 - mask) * agnostic
        shuffled_warped = mask * shuffled_diag["warped_rgb"] + (1.0 - mask) * agnostic
        grid = _preview_grid(
            batch,
            paired_diag["source_rgb"],
            generated,
            t0_reconstruction,
            paired_warped,
            shuffled_batch["garment"],
            shuffled_diag["source_rgb"],
            shuffled_warped,
            shuffled,
        )
        preview_tag = "preview/agnostic_pgarment_pose_gt_psource_paired_t0_pwarp_sgarment_ssource_swarp_shuffled"
    else:
        paired_warped = None
        grid = _noise_preview_grid(
            batch, generated, t0_reconstruction, shuffled_batch["garment"], shuffled
        )
        preview_tag = "preview/agnostic_pgarment_pose_gt_t0_paired_sgarment_shuffled"
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, preview_dir / f"step_{step:07d}.png")
    if identity_generated is not None:
        identity_grid = _identity_preview_grid(batch, identity_generated)
        save_image(
            identity_grid,
            preview_dir / f"identity_step_{step:07d}.png",
        )
    if writer is not None:
        writer.add_image(preview_tag, grid, step)
        clothing = batch["clothing_mask"].to(device)
        writer.add_scalar("preview/paired_l1", _masked_mean((generated - batch["image"].to(device)).abs(), mask), step)
        writer.add_scalar("preview/clothing_l1", _masked_mean((generated - batch["image"].to(device)).abs(), clothing), step)
        writer.add_scalar("preview/t0_clothing_l1", _masked_mean((t0_reconstruction - batch["image"].to(device)).abs(), clothing), step)
        writer.add_scalar("preview/garment_sensitivity", _masked_mean((generated - shuffled).abs(), clothing), step)
        if identity_generated is not None:
            garment_mask = batch["garment_mask"].to(device)
            garment = batch["garment"].to(device)
            writer.add_image("preview/identity_gt_reconstruction", identity_grid, step)
            writer.add_scalar(
                "preview/identity_clothing_l1",
                _masked_mean((identity_generated - garment).abs(), garment_mask),
                step,
            )
        if start_from_warp:
            writer.add_scalar("preview/source_clothing_l1", _masked_mean((paired_diag["source_rgb"] - batch["image"].to(device)).abs(), clothing), step)
            writer.add_scalar("preview/warped_clothing_l1", _masked_mean((paired_warped - batch["image"].to(device)).abs(), clothing), step)
            writer.add_scalar("preview/source_confidence", paired_diag["source_confidence"].float().mean(), step)
        writer.flush()
    if was_training:
        unwrapped.train()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vton_crossattn_512x384.yaml")
    parser.add_argument("--resume", default=None, help="Resume weights + optimizer; config/trainable set must match")
    parser.add_argument("--load", default=None, help="Load VTON delta weights only; use for stage changes")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    set_seed(int(cfg.seed))

    accelerator = Accelerator(
        mixed_precision=str(cfg.train.mixed_precision),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
    )
    device = accelerator.device

    dataset = VitonHDDataset(
        root=str(cfg.data.root),
        phase="train",
        order="paired",
        size=tuple(cfg.model.image_hw),
        pairs_file=cfg.data.get("pairs_file", None),
        require_cloth_mask=bool(cfg.data.require_cloth_mask),
        require_parse=bool(cfg.data.get("require_parse", False)),
        clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.data.num_workers) > 0,
        drop_last=True,
    )

    model = build_model(cfg)
    vae = load_sd_vae(str(cfg.weights.vae), device=device, dtype=torch.float32)
    teacher = None
    if float(cfg.coral.weight) > 0:
        teacher = DINOv3CoralTeacher(
            str(cfg.weights.dino),
            min_similarity=float(cfg.coral.min_similarity),
            cycle_radius=float(cfg.coral.cycle_radius),
        ).to(device)

    trainable, total = count_trainable(model)
    print(f"Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    # Separate newly-added modules from optional pretrained self-attention/final layer.
    # When the final output projection is explicitly reset, it is new for this
    # objective as well and must not be throttled by the pretrained LR multiplier.
    reset_final_output = bool(cfg.optim.get("reset_final_output_on_load", False))
    new_params, lora_params, pretrained_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".lora_down." in name or ".lora_up." in name:
            lora_params.append(p)
        elif name.startswith((
            "person_condition_embedder", "source_condition_embedder",
            "garment_embedder", "garment_cross_attn",
            "garment_router",
            "attention_transport",
            "final_detail",
            "garment_transport", "garment_rgb_transport",
            "joint_garment_segment",
        )) or (reset_final_output and name.startswith("final_layer.linear")):
            new_params.append(p)
        else:
            pretrained_params.append(p)
    groups = []
    if new_params:
        groups.append({"params": new_params, "lr": float(cfg.optim.lr)})
    if lora_params:
        groups.append({
            "params": lora_params,
            "lr": float(cfg.optim.get("lora_lr", cfg.optim.lr)),
            "weight_decay": float(cfg.optim.get("lora_weight_decay", 0.0)),
        })
    if pretrained_params:
        groups.append({"params": pretrained_params, "lr": float(cfg.optim.lr) * float(cfg.optim.pretrained_lr_multiplier)})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.optim.weight_decay), betas=(0.9, 0.95))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    start_step = 0
    # Stage checkpoints must remain self-contained relative to the original PFT
    # weights. Preserve tensors inherited from a prior VTON delta even when V7
    # freezes them; otherwise the next sample/resume silently loses that stage.
    inherited_delta_names: set[str] = set()
    if args.load and args.resume:
        raise ValueError("Use either --load (weights only) or --resume (weights + optimizer), not both")
    if args.load or args.resume:
        state = torch.load(args.load or args.resume, map_location="cpu", weights_only=False)
        # Raw weights only. EMA is deliberately neither restored nor preferred.
        delta = state.get("model_delta", state.get("model"))
        delta_source = "model_delta" if "model_delta" in state else "model"
        if delta is None:
            raise KeyError("Checkpoint has neither model_delta nor model")
        include_prefixes = list(cfg.optim.get("load_include_prefixes", []))
        if args.load and include_prefixes:
            delta = {
                name: value for name, value in delta.items()
                if any(name.startswith(prefix) for prefix in include_prefixes)
            }
            if not delta:
                raise ValueError(
                    f"No checkpoint tensors matched load_include_prefixes={include_prefixes}"
                )
            print(
                f"Selective stage load: {len(delta)} tensors matching "
                f"{include_prefixes}"
            )
        inherited_delta_names = set(delta)
        missing, unexpected = model.load_state_dict(delta, strict=False)
        print(
            f"Loaded VTON checkpoint {delta_source}: {len(delta)} tensors; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if args.load and bool(cfg.optim.get("reset_final_output_on_load", False)):
            # The source distribution changed from Gaussian noise to an already
            # recognizable warped garment. Reusing the old noise-to-image output
            # projection immediately destroys that source. Zero velocity is the
            # correct identity/refinement initialization; all pretrained features,
            # attention adapters, and transport weights remain loaded.
            with torch.no_grad():
                model.final_layer.linear.weight.zero_()
                model.final_layer.linear.bias.zero_()
            print("Reset final velocity/logvar projection for warp-residual training")
        if args.load and bool(cfg.optim.get("reset_transport_flow_output_on_load", False)):
            # A residual learned around the old identity grid has a different
            # meaning around the mask-aligned affine grid. Preserve the flow
            # feature extractor, but begin the two-channel residual at zero so
            # step zero exactly matches the verified geometric base warp.
            with torch.no_grad():
                model.garment_transport_flow[-1].weight.zero_()
                model.garment_transport_flow[-1].bias.zero_()
            print("Reset transport residual projection for mask-aligned base warp")
        if args.load and bool(cfg.optim.get("reset_transport_geometry_input_on_load", False)):
            # Input channels 9:13 held VAE appearance latents in V5, but hold a
            # signed silhouette in geometry-only V6. Their old filters have no
            # compatible meaning. Preserve channels 0:9 (person/pose/mask) and
            # all later flow features while relearning only this input slice.
            with torch.no_grad():
                model.garment_transport_flow[0].weight[:, 9:13].zero_()
            print("Reset appearance-to-geometry transport input kernels")
        if args.resume:
            if "optimizer" not in state:
                raise KeyError("--resume checkpoint has no optimizer state")
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state.get("step", 0))

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if teacher is not None:
        teacher.eval()
    vae.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    out_dir = Path(str(cfg.train.output_dir))
    writer = None
    fixed_preview_batch = None
    preview_every = int(cfg.train.get("preview_every", 1000))
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "resolved_config.yaml")
        if bool(cfg.train.get("tensorboard", True)):
            writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        if preview_every > 0:
            preview_ds = VitonHDDataset(
                root=str(cfg.data.root),
                phase=str(cfg.train.get("preview_phase", "test")),
                order=str(cfg.train.get("preview_order", "paired")),
                size=tuple(cfg.model.image_hw),
                pairs_file=cfg.data.get("test_pairs_file", None),
                require_cloth_mask=bool(cfg.data.require_cloth_mask),
                require_parse=bool(cfg.data.get("require_parse", False)),
                clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
            )
            preview_loader = DataLoader(
                preview_ds,
                batch_size=int(cfg.train.get("preview_num_samples", 2)),
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            fixed_preview_batch = next(iter(preview_loader))

    max_steps = int(cfg.train.max_steps)
    pbar = tqdm(total=max_steps, initial=start_step, disable=not accelerator.is_local_main_process)
    step = start_step
    _set_learning_rates(optimizer, base_lrs, step, max_steps, cfg)
    data_iter = iter(loader)

    if preview_every > 0 and bool(cfg.train.get("preview_at_start", True)) and start_step == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and fixed_preview_batch is not None:
            _write_preview(
                unwrapped_model, vae, fixed_preview_batch,
                step=0, cfg=cfg, writer=writer, out_dir=out_dir, device=device,
            )
        accelerator.wait_for_everyone()

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            gt = batch["image"].to(device)
            agnostic = batch["agnostic"].to(device)
            pose = batch["densepose"].to(device)
            mask_px = batch["agnostic_mask"].to(device)
            garment = batch["garment"].to(device)
            garment_mask_px = batch["garment_mask"].to(device)
            clothing_mask_px = batch["clothing_mask"].to(device)

            # Paired VITON data contains only one garment for each person. A
            # network can therefore lower reconstruction loss by associating
            # person context with a plausible garment instead of learning an
            # exact, compositional copy path. On a configurable subset, turn
            # the same example into a canonical garment self-reconstruction
            # task. The target is the original shop image, not a synthetic
            # warp; a blank pose distinguishes this auxiliary task. This makes
            # logos, text, texture and color blocks necessary to solve the
            # objective while sharing all garment encoders/attention adapters.
            identity_probability = float(
                cfg.loss.get("garment_identity_probability", 0.0)
            )
            identity_task = (
                torch.rand(gt.shape[0], device=device) < identity_probability
            )
            if identity_task.any():
                choose_rgb = identity_task[:, None, None, None]
                choose_mask = identity_task[:, None, None, None]
                gt = torch.where(choose_rgb, garment, gt)
                agnostic = torch.where(choose_rgb, torch.ones_like(agnostic), agnostic)
                pose = torch.where(choose_rgb, -torch.ones_like(pose), pose)
                mask_px = torch.where(choose_mask, garment_mask_px, mask_px)
                clothing_mask_px = torch.where(
                    choose_mask, garment_mask_px, clothing_mask_px
                )

            # Counterfactual paired supervision prevents the one-person/one-
            # garment dataset from teaching a mean or memorized colorway. This
            # is a pointwise appearance transform, not a spatial garment warp.
            gt, garment, appearance_counterfactual = _counterfactual_appearance_pair(
                gt,
                garment,
                clothing_mask_px,
                garment_mask_px,
                probability=float(
                    cfg.loss.get("appearance_counterfactual_probability", 0.0)
                ),
                strength=float(cfg.loss.get("appearance_counterfactual_strength", 1.0)),
            )

            with torch.no_grad():
                all_rgb = torch.cat([gt, agnostic, pose, garment], dim=0)
                all_z = encode_images(vae, all_rgb)
                z_gt, z_agnostic, z_pose, z_garment = all_z.chunk(4, dim=0)

            latent_hw = tuple(cfg.model.latent_hw)
            p = int(cfg.model.patch_size)
            mask_lat = latent_edit_mask(mask_px, latent_hw)
            garment_mask_lat = latent_edit_mask(garment_mask_px, latent_hw)
            clothing_mask_lat = latent_edit_mask(clothing_mask_px, latent_hw)
            edit_tokens = token_edit_mask(mask_lat, p)
            clothing_tokens = token_edit_mask(clothing_mask_lat, p) & edit_tokens

            t, time_metrics = _sample_times(
                cfg, gt.shape[0], edit_tokens.shape[1], device, z_gt.dtype, step=step
            )
            # The production sampler is one-step from t=0. Canonical identity
            # examples specifically teach that exact source-to-output path.
            t[identity_task] = 0.0
            # Outside inpainting region is always data-time t=1.
            t = torch.where(edit_tokens, t, torch.ones_like(t))
            t_map = expand_patch_values(t, p, latent_hw)
            # Pixel/latent-exact outside handling: even boundary patches keep non-mask pixels at t=1.
            t_map = torch.where(mask_lat > 1e-4, t_map, torch.ones_like(t_map))

            noise = torch.randn_like(z_gt)
            start_from_warp = bool(cfg.flow.get("start_from_warp", True))
            z_source = source_rgb = source_confidence = None
            if start_from_warp:
                z_source, source_rgb, _, _, source_confidence = build_warp_source_latent(
                    accelerator.unwrap_model(model),
                    vae,
                    agnostic=agnostic,
                    pose=pose,
                    mask_px=mask_px,
                    garment=garment,
                    garment_mask_px=garment_mask_px,
                    z_agnostic=z_agnostic,
                    z_pose=z_pose,
                    z_garment=z_garment,
                    mask_lat=mask_lat,
                    garment_mask_lat=garment_mask_lat,
                    confidence_erode_px=int(
                        cfg.flow.get("source_confidence_erode_px", 10)
                    ),
                )
                source_noise_std = min(
                    max(float(cfg.flow.get("source_noise_std", 0.15)), 0.0), 0.999
                )
                source_scale = math.sqrt(1.0 - source_noise_std * source_noise_std)
                trusted_source = source_scale * z_source + source_noise_std * noise
                x0 = source_confidence * trusted_source + (1.0 - source_confidence) * noise
            else:
                # Standard rectified flow: no geometric answer is placed in the
                # initial state. The model must infer garment layout through the
                # clean reference tokens available in every transformer block.
                x0 = noise
            x0 = mask_lat * x0 + (1.0 - mask_lat) * z_agnostic
            xt = t_map * z_gt + (1.0 - t_map) * x0
            # Never leak the target through the model input: outside the edit mask use the
            # agnostic/source latent, exactly as inference does. Loss outside the mask is zero.
            xt = mask_lat * xt + (1.0 - mask_lat) * z_agnostic
            target_v = z_gt - x0

            out = model(
                xt=xt,
                t=t,
                agnostic_latent=z_agnostic,
                densepose_latent=z_pose,
                agnostic_mask_latent=mask_lat,
                garment_latent=z_garment,
                garment_mask_latent=garment_mask_lat,
                garment_rgb=garment,
                garment_mask_rgb=garment_mask_px,
                source_latent=z_source,
                source_confidence_latent=source_confidence,
                edit_token_mask=edit_tokens,
                return_uncertainty=True,
                return_attention=(
                    float(cfg.coral.weight) > 0
                    or float(cfg.loss.get("joint_attention_weight", 0.0)) > 0
                ),
            )
            pred_v = out["velocity"]

            edge_w = normalized_edge_weights(gt, latent_hw, boost=float(cfg.loss.edge_flow_boost))
            clothing_boost = float(cfg.loss.get("clothing_boost", 0.0))
            flow_weight = mask_lat * edge_w * (1.0 + clothing_boost * clothing_mask_lat)
            flow_sq = (pred_v - target_v).pow(2)
            flow_loss = _masked_mean(flow_sq, flow_weight)
            total_loss = flow_loss
            metrics = {
                "loss_flow": flow_loss,
                "garment_identity_fraction": identity_task.float().mean(),
                "appearance_counterfactual_fraction": appearance_counterfactual.float().mean(),
                **time_metrics,
            }
            if start_from_warp:
                metrics.update({
                    "source_clothing_l1": _masked_mean(
                        (source_rgb - gt).abs(), clothing_mask_px
                    ),
                    "source_latent_l1": _masked_mean(
                        (z_source - z_gt).abs(), clothing_mask_lat
                    ),
                    "source_confidence": _masked_mean(source_confidence, mask_lat),
                })
            for metric_name, output_name in (
                ("transport_token_rms", "transport_tokens"),
                ("transport_rgb_feature_rms", "rgb_transport_tokens"),
                ("transport_velocity_rms", "transport_velocity"),
                ("attention_transport_velocity_rms", "attention_transport_velocity"),
                ("attention_transport_confidence", "attention_transport_confidence"),
            ):
                value = out.get(output_name)
                if value is not None:
                    metrics[metric_name] = value.float().square().mean().sqrt()

            warped_garment = out.get("warped_garment")
            if warped_garment is not None:
                # This objective directly teaches garment-to-body deformation. V2
                # only supervised the route indirectly, allowing the generator to
                # replace source logos and color blocks with semantic approximations.
                transport_latent_loss = _masked_mean(
                    (warped_garment.float() - z_gt.float()).abs(),
                    clothing_mask_lat,
                )
                smoothness_loss = _transport_smoothness(out.get("transport_offset"))
                offset = out.get("transport_offset")
                if offset is None:
                    magnitude_loss = transport_latent_loss * 0.0
                else:
                    magnitude_loss = offset.float().square().mean()
                total_loss = (
                    total_loss
                    + float(cfg.loss.get("transport_latent_weight", 0.0)) * transport_latent_loss
                    + float(cfg.loss.get("transport_smoothness_weight", 0.0)) * smoothness_loss
                    + float(cfg.loss.get("transport_offset_weight", 0.0)) * magnitude_loss
                )
                metrics.update({
                    "loss_transport_latent": transport_latent_loss,
                    "loss_transport_smoothness": smoothness_loss,
                    "loss_transport_offset": magnitude_loss,
                })

                # Supervise the same deformation at full image resolution. Unlike
                # DINO/CORAL or 1/8 latents, this sees individual letters and exact
                # sleeve/body color boundaries.
                warped_source_rgb, warped_source_mask = warp_source_rgb(
                    garment, garment_mask_px, out["transport_grid"]
                )
                transport_rgb_loss = _masked_mean(
                    (warped_source_rgb - gt).abs(), clothing_mask_px
                )
                transport_edge_loss = _edge_loss(
                    warped_source_rgb, gt, clothing_mask_px
                )
                transport_laplacian_loss = _laplacian_loss(
                    warped_source_rgb, gt, clothing_mask_px
                )
                transport_color_loss = _block_color_loss(
                    warped_source_rgb, gt, clothing_mask_px
                )
                total_loss = total_loss + (
                    float(cfg.loss.get("transport_rgb_weight", 0.0)) * transport_rgb_loss
                    + float(cfg.loss.get("transport_edge_weight", 0.0)) * transport_edge_loss
                    + float(cfg.loss.get("transport_laplacian_weight", 0.0)) * transport_laplacian_loss
                    + float(cfg.loss.get("transport_color_weight", 0.0)) * transport_color_loss
                )
                metrics.update({
                    "loss_transport_rgb": transport_rgb_loss,
                    "loss_transport_edge": transport_edge_loss,
                    "loss_transport_laplacian": transport_laplacian_loss,
                    "loss_transport_color": transport_color_loss,
                    "transport_coverage": _masked_mean(warped_source_mask, clothing_mask_px),
                })

            if out["logvar"] is not None and float(cfg.loss.uncertainty_weight) > 0:
                logvar = out["logvar"].clamp(-10, 10)
                nll = 0.5 * ((pred_v.detach() - target_v).pow(2).mean(1, keepdim=True) * torch.exp(-logvar) + logvar)
                uncertainty_loss = _masked_mean(nll, mask_lat)
                total_loss = total_loss + float(cfg.loss.uncertainty_weight) * uncertainty_loss
                metrics["loss_uncertainty"] = uncertainty_loss

            joint_attention_weight = float(
                cfg.loss.get("joint_attention_weight", 0.0)
            )
            if out.get("joint_attention") and joint_attention_weight > 0:
                joint_route_loss, joint_metrics = _joint_garment_attention_loss(
                    out["joint_attention"],
                    clothing_tokens,
                    out["garment_valid"],
                    token_hw=accelerator.unwrap_model(model).token_hw,
                    minimum_mass=float(
                        cfg.loss.get("joint_attention_minimum_mass", 0.35)
                    ),
                )
                total_loss = total_loss + joint_attention_weight * joint_route_loss
                metrics.update({
                    "loss_joint_attention": joint_route_loss,
                    **joint_metrics,
                })

            if out["attention"] and float(cfg.coral.weight) > 0:
                if teacher is None:
                    raise RuntimeError("CORAL weight is positive but its teacher was not initialized")
                with torch.no_grad():
                    targets = teacher.build_targets(
                        person_rgb=gt,
                        garment_rgb=garment,
                        edit_token_mask=clothing_tokens,
                        # The native-resolution cross route is 64x48; the joint
                        # garment stream is only 32x24. Use the mask belonging
                        # to the attention map being supervised.
                        garment_valid=out["cross_garment_valid"],
                        person_hw=accelerator.unwrap_model(model).token_hw,
                        garment_hw=accelerator.unwrap_model(model).garment_token_hw,
                    )
                    q_weight = normalized_edge_weights(
                        gt,
                        accelerator.unwrap_model(model).token_hw,
                        boost=float(cfg.coral.detail_edge_boost),
                        mask=F.adaptive_avg_pool2d(clothing_mask_px, accelerator.unwrap_model(model).token_hw),
                    ).flatten(1)
                corr_loss, ent_loss, coral_metrics = coral_routing_loss(
                    out["attention"],
                    targets,
                    accelerator.unwrap_model(model).garment_token_hw,
                    query_weights=q_weight,
                    gaussian_sigma=float(cfg.coral.get("gaussian_sigma", 0.035)),
                )
                total_loss = (
                    total_loss
                    + float(cfg.coral.weight) * corr_loss
                    + float(cfg.coral.entropy_weight) * ent_loss
                )
                metrics.update({
                    "loss_coral": corr_loss,
                    "loss_entropy": ent_loss,
                    **coral_metrics,
                })

            # Supervise the endpoint estimate at every time used by inference.
            # Previously only t=0 received an endpoint/image objective, while 50-step
            # sampling visits 49 nonzero times. That let the vector field preserve a
            # low flow MSE yet drift away from the source logo during integration.
            pred_clean_z = xt + mask_lat * (1.0 - t_map) * pred_v
            one_minus_t = (
                ((1.0 - t_map) * mask_lat).sum()
                / mask_lat.sum().clamp_min(1.0)
            )
            correction = 1.0 / one_minus_t.detach().clamp_min(
                float(cfg.loss.decoded_time_floor)
            )
            latent_reconstruction_loss = _masked_mean(
                (pred_clean_z.float() - z_gt.float()).abs(), clothing_mask_lat
            )
            total_loss = total_loss + correction * float(
                cfg.loss.get("decoded_latent_weight", 0.0)
            ) * latent_reconstruction_loss
            metrics.update({
                "loss_decoded_latent": latent_reconstruction_loss,
                "decoded_correction": correction,
            })

            edited_times = t.masked_select(edit_tokens)
            pure_batch = bool(
                edited_times.numel() > 0 and edited_times.abs().max().item() == 0.0
            )
            do_decode = (
                float(cfg.loss.decoded_probability) > 0
                and (
                    pure_batch
                    or (
                        not bool(cfg.loss.get("decoded_pure_only", True))
                        and torch.rand((), device=device) < float(cfg.loss.decoded_probability)
                    )
                )
            )
            decoded_weights = [
                float(cfg.loss.get("decoded_rgb_weight", 0.0)),
                float(cfg.loss.get("decoded_edge_weight", 0.0)),
                float(cfg.loss.get("decoded_laplacian_weight", 0.0)),
                float(cfg.loss.get("decoded_color_weight", 0.0)),
            ]
            if do_decode and any(w > 0 for w in decoded_weights):
                pred_rgb = decode_latents_with_grad(vae, pred_clean_z.float())
                detail_mask = mask_px * (1.0 + clothing_boost * clothing_mask_px)
                rgb_loss = _masked_mean((pred_rgb - gt).abs(), detail_mask)
                ed_loss = _edge_loss(pred_rgb, gt, detail_mask)
                lap_loss = _laplacian_loss(pred_rgb, gt, detail_mask)
                color_loss = _block_color_loss(pred_rgb, gt, detail_mask)
                total_loss = total_loss + correction * (
                    float(cfg.loss.decoded_rgb_weight) * rgb_loss
                    + float(cfg.loss.decoded_edge_weight) * ed_loss
                    + float(cfg.loss.get("decoded_laplacian_weight", 0.0)) * lap_loss
                    + float(cfg.loss.get("decoded_color_weight", 0.0)) * color_loss
                )
                metrics.update({
                    "loss_rgb": rgb_loss,
                    "loss_edge": ed_loss,
                    "loss_laplacian": lap_loss,
                    "loss_color": color_loss,
                })
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), float(cfg.optim.clip_grad_norm))
                metrics["grad_norm"] = grad_norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics["loss_total"] = total_loss
        for key, value in detach_metrics(metrics).items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_counts[key] = metric_counts.get(key, 0) + 1

        if accelerator.sync_gradients:
            step += 1
            pbar.update(1)
            _set_learning_rates(optimizer, base_lrs, step, max_steps, cfg)
            if accelerator.is_main_process and step % int(cfg.train.log_every) == 0:
                vals = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}
                text = " ".join(f"{k}={v:.4f}" for k, v in vals.items())
                pbar.set_postfix_str(text[:220])
                if writer is not None:
                    for key, value in vals.items():
                        writer.add_scalar(f"train/{key}", value, step)
                    for gi, group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"train/lr_group_{gi}", group["lr"], step)
                    for name, p in unwrapped_model.named_parameters():
                        if name.endswith(".gate"):
                            writer.add_scalar(f"train_gate/{name}", p.detach().float(), step)
                    writer.add_scalar(
                        "train_gate/transport_fixed_gate",
                        unwrapped_model.transport_fixed_gate,
                        step,
                    )
                    writer.add_scalar(
                        "train_gate/transport_block_gate",
                        unwrapped_model.transport_block_gate,
                        step,
                    )
                    writer.add_scalar(
                        "train_gate/transport_output_scale",
                        unwrapped_model.transport_output_scale,
                        step,
                    )
                    writer.add_scalar(
                        "train_gate/cross_attention_scale",
                        unwrapped_model.cross_attention_scale,
                        step,
                    )
                    writer.add_scalar(
                        "train_gate/attention_transport_gate",
                        unwrapped_model.attention_transport_gate.detach().float(),
                        step,
                    )
                metric_sums.clear()
                metric_counts.clear()

            if preview_every > 0 and step % preview_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process and fixed_preview_batch is not None:
                    _write_preview(
                        unwrapped_model, vae, fixed_preview_batch,
                        step=step, cfg=cfg, writer=writer, out_dir=out_dir, device=device,
                    )
                accelerator.wait_for_everyone()

            if accelerator.is_main_process and step % int(cfg.train.save_every) == 0:
                unwrapped = accelerator.unwrap_model(model)
                trainable_names = {
                    name for name, p in unwrapped.named_parameters() if p.requires_grad
                }
                required_names = trainable_names | inherited_delta_names
                full_state = unwrapped.state_dict()
                delta = {
                    k: v.detach().cpu()
                    for k, v in full_state.items()
                    if k in required_names
                }
                ckpt = {
                    "step": step,
                    "model_delta": delta,
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                torch.save(ckpt, out_dir / f"vton_crossattn_step{step:07d}.pt")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        trainable_names = {
            name for name, p in unwrapped.named_parameters() if p.requires_grad
        }
        required_names = trainable_names | inherited_delta_names
        full_state = unwrapped.state_dict()
        delta = {
            k: v.detach().cpu()
            for k, v in full_state.items()
            if k in required_names
        }
        torch.save(
            {
                "step": step,
                "model_delta": delta,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            out_dir / "vton_crossattn_final.pt",
        )
    if writer is not None:
        writer.flush()
        writer.close()
    pbar.close()


if __name__ == "__main__":
    main()
