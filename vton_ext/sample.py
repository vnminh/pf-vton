from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .data import VitonHDDataset
from .geometry import erode_confidence
from .model import VTONPatchForcingDiT
from .utils import latent_edit_mask, token_edit_mask
from .vae import decode_latents, encode_images, load_sd_vae


def build_model(cfg, checkpoint: str, device) -> VTONPatchForcingDiT:
    model = VTONPatchForcingDiT(
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
    model.load_pretrained_pft(str(cfg.weights.pft))
    lora_rank = int(cfg.optim.get("lora_rank", 0))
    if lora_rank > 0:
        model.add_backbone_lora(
            rank=lora_rank,
            alpha=float(cfg.optim.get("lora_alpha", lora_rank)),
            dropout=float(cfg.optim.get("lora_dropout", 0.0)),
            include_adaln=bool(cfg.optim.get("lora_include_adaln", True)),
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    delta = state.get("model_delta", state.get("model"))
    if delta is None:
        raise KeyError("VTON checkpoint has neither model_delta nor model")
    model.load_state_dict(delta, strict=False)
    return model.to(device).eval()


def warp_source_rgb(
    garment: torch.Tensor,
    garment_mask: torch.Tensor,
    latent_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp the original garment pixels with the learned latent-resolution grid."""
    rgb_grid = F.interpolate(
        latent_grid.permute(0, 3, 1, 2),
        size=garment.shape[-2:],
        mode="bilinear",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    warped = F.grid_sample(
        garment.float(), rgb_grid.float(), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    warped_mask = F.grid_sample(
        garment_mask.float(), rgb_grid.float(), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).clamp(0.0, 1.0)
    # RGB is normalized to [-1, 1], so invalid source pixels are black (-1).
    warped = warped * warped_mask + (-1.0) * (1.0 - warped_mask)
    return warped, warped_mask


@torch.no_grad()
def build_warp_source_latent(
    model,
    vae,
    *,
    agnostic: torch.Tensor,
    pose: torch.Tensor,
    mask_px: torch.Tensor,
    garment: torch.Tensor,
    garment_mask_px: torch.Tensor,
    z_agnostic: torch.Tensor,
    z_pose: torch.Tensor,
    z_garment: torch.Tensor,
    mask_lat: torch.Tensor,
    garment_mask_lat: torch.Tensor,
    confidence_erode_px: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the deterministic source endpoint for warp-to-person residual flow.

    The old sampler started from unrelated Gaussian noise and exposed the good
    garment warp only through adapters. The transformer therefore learned to
    synthesize a plausible garment and routinely replaced exact source logos.
    Here the full-resolution warped pixels are composited with the agnostic
    person and encoded once, so the flow starts with the identity information it
    must preserve and only needs to learn pose/occlusion/lighting refinement.
    """
    grid, _ = model.predict_transport_grid(
        z_agnostic, z_pose, mask_lat, z_garment, garment_mask_lat
    )
    warped_rgb, warped_mask = warp_source_rgb(
        garment, garment_mask_px, grid
    )
    support_mask = (warped_mask * mask_px.float()).clamp(0.0, 1.0)
    copy_confidence = erode_confidence(
        support_mask,
        radius=int(confidence_erode_px),
    )
    # Preserve the complete warped garment in the explicit source condition.
    # Confidence controls only the ODE start state; it must not erase boundary
    # appearance from the source that the model sees at later timesteps.
    source_rgb = support_mask * warped_rgb + (1.0 - support_mask) * agnostic.float()
    source_latent = encode_images(vae, source_rgb)
    confidence_latent = F.interpolate(
        copy_confidence, size=model.latent_hw, mode="area"
    ).clamp(0.0, 1.0)
    return source_latent, source_rgb, warped_rgb, warped_mask, confidence_latent


@torch.no_grad()
def sample_one(
    model,
    vae,
    batch,
    steps: int,
    device,
    seed: int,
    source_noise_std: float = 0.15,
    source_confidence_erode_px: int = 10,
    start_from_warp: bool = True,
    solver: str = "euler",
    return_diagnostics: bool = False,
):
    agnostic = batch["agnostic"].to(device)
    pose = batch["densepose"].to(device)
    mask_px = batch["agnostic_mask"].to(device)
    garment = batch["garment"].to(device)
    garment_mask_px = batch["garment_mask"].to(device)

    all_z = encode_images(vae, torch.cat([agnostic, pose, garment], dim=0))
    z_agnostic, z_pose, z_garment = all_z.chunk(3, dim=0)
    latent_hw = model.latent_hw
    mask_lat = latent_edit_mask(mask_px, latent_hw)
    garment_mask_lat = latent_edit_mask(garment_mask_px, latent_hw)
    edit_tokens = token_edit_mask(mask_lat, model.patch_size)

    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        z_agnostic.shape, generator=gen, device=device, dtype=z_agnostic.dtype
    )
    z_source = source_rgb = warped_rgb = warped_mask = source_confidence = None
    if start_from_warp:
        z_source, source_rgb, warped_rgb, warped_mask, source_confidence = build_warp_source_latent(
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
        x = source_confidence * trusted_source + (1.0 - source_confidence) * noise
    else:
        x = noise
    # Clean agnostic context is fixed outside the inpaint region from the first step onward.
    x = mask_lat * x + (1.0 - mask_lat) * z_agnostic

    schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)
    solver = str(solver).lower()
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unknown solver '{solver}'; expected 'euler' or 'heun'")

    def predict_velocity(state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        time_tokens = torch.where(
            edit_tokens,
            torch.full_like(edit_tokens, time_value, dtype=state.dtype),
            torch.ones_like(edit_tokens, dtype=state.dtype),
        )
        return model(
            xt=state,
            t=time_tokens,
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
        )

    for s in range(steps):
        t0, t1 = schedule[s], schedule[s + 1]
        dt = (t1 - t0).to(x.dtype)
        velocity = predict_velocity(x, t0)
        if solver == "heun" and s + 1 < steps:
            proposal = x + mask_lat * dt * velocity
            proposal = mask_lat * proposal + (1.0 - mask_lat) * z_agnostic
            next_velocity = predict_velocity(proposal, t1)
            velocity = 0.5 * (velocity + next_velocity)
        x = x + mask_lat * dt * velocity
        x = mask_lat * x + (1.0 - mask_lat) * z_agnostic

    rgb = decode_latents(vae, x.float()).clamp(-1, 1)
    # Exact pixel-space preservation outside the requested inpaint mask.
    rgb = mask_px * rgb + (1.0 - mask_px) * agnostic
    if return_diagnostics:
        return rgb, {
            "source_rgb": source_rgb,
            "warped_rgb": warped_rgb,
            "warped_mask": warped_mask,
            "source_confidence": source_confidence,
        }
    return rgb


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vton_crossattn_512x384.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/vton_samples")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PFT-XL sampling")
    device = torch.device("cuda")

    model = build_model(cfg, args.checkpoint, device)
    vae = load_sd_vae(str(cfg.weights.vae), device=device, dtype=torch.float32)
    ds = VitonHDDataset(
        root=str(cfg.data.root),
        phase=str(cfg.data.get("test_phase", "test")),
        order=str(cfg.data.test_order),
        size=tuple(cfg.model.image_hw),
        pairs_file=cfg.data.get("test_pairs_file", None),
        require_cloth_mask=bool(cfg.data.get("require_cloth_mask", False)),
        require_parse=bool(cfg.data.get("require_parse", False)),
        clothing_labels=list(cfg.data.get("clothing_labels", [5, 6, 7])),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for i, batch in enumerate(tqdm(dl, total=min(args.num_samples, len(ds)))):
        if i >= args.num_samples:
            break
        with torch.autocast("cuda", dtype=autocast_dtype):
            result = sample_one(
                model,
                vae,
                batch,
                args.steps,
                device,
                args.seed + i,
                source_noise_std=float(cfg.flow.get("source_noise_std", 0.15)),
                source_confidence_erode_px=int(
                    cfg.flow.get("source_confidence_erode_px", 10)
                ),
                start_from_warp=bool(cfg.flow.get("start_from_warp", True)),
                solver=str(cfg.flow.get("solver", "euler")),
            )
        person = batch["person_name"][0]
        garment = Path(batch["garment_name"][0]).stem
        save_image(result, out_dir / f"{Path(person).stem}__{garment}.png", normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
