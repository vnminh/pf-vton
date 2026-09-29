"""Train the Patch-Forcing Inpainting DiT (PFI) with CoRAL attention supervision.

    PYTHONPATH=. python -m vton_ext.pfi_train --config configs/vton_v40_pfi_coral.yaml [key=value ...]

Per example:
  * editable tokens (agnostic mask) get patch-forcing times: pure noise,
    synchronous logit-normal, or heterogeneous LTG around a logit-normal mean;
  * known tokens and garment tokens are clean context at t=1;
  * loss = flow MSE on editable latents + uncertainty NLL (SRM, PFT recipe)
           + CoRAL CE/entropy on person->garment attention at coral blocks;
           optionally, sparse decoded RGB/high-pass garment-detail loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from patch_flow.timestep_schedules import LogitNormalTruncatedGaussian
from vton_ext.coral import DINOv3CoralTeacher, coral_routing_loss
from vton_ext.data import VitonHDDataset
from vton_ext.pairs import split_manifest, validate_resume_pairs
from vton_ext.pfi_model import VTONInpaintDiT
from vton_ext.pfi_sample import (composite, generate, prepare_inputs, shift_time,
                                 token_uncertainty, uncertain_tokens)
from vton_ext.utils import expand_patch_values, pool_valid_mask, seed_everything
from vton_ext.vae import decode_latents, decode_latents_with_grad, load_sd_vae


# ------------------------------------------------------------------ config

def load_cfg(path: str, overrides=()):
    cfg = OmegaConf.load(path)
    if "base_config" in cfg:
        base = load_cfg(cfg.pop("base_config"))
        cfg = OmegaConf.merge(base, cfg)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def build_model(cfg, load_pretrained: bool = True) -> VTONInpaintDiT:
    m = cfg.model
    model = VTONInpaintDiT(
        latent_hw=tuple(m.latent_hw),
        patch_size=int(m.patch_size),
        coral_blocks=list(m.coral_blocks),
        coral_heads=int(m.coral_heads),
        pos_embed=str(m.get("pos_embed", "interpolate")),
        pos_embed_trainable=bool(m.get("pos_embed_trainable", False)),
    )
    model.time_shift = float(cfg.flow.get("time_shift", 1.0))
    if load_pretrained:
        report = model.load_pretrained_pft(cfg.weights.pft)
        print(f"loaded PFT: {report['loaded']} tensors, missing {report['missing']}")
        init_from = cfg.weights.get("init_from")
        if init_from:
            report = model.load_weights_any_resolution(init_from)
            print(f"initialised from {init_from}: {report}")
    return model


def validate_config(cfg):
    image_hw, latent_hw = tuple(cfg.model.image_hw), tuple(cfg.model.latent_hw)
    patch = int(cfg.model.patch_size)
    if len(image_hw) != 2 or len(latent_hw) != 2 or patch <= 0:
        raise ValueError("Expected two image/latent dimensions and a positive patch size")
    if any(i <= 0 or i % 8 or i // 8 != l or l % patch for i, l in zip(image_hw, latent_hw)):
        raise ValueError("image_hw must be 8 * latent_hw; latent dimensions must divide by patch_size")
    if int(cfg.train.batch_size) < 1 or int(cfg.train.gradient_accumulation_steps) < 1:
        raise ValueError("Training batch size and accumulation must be positive")
    rcfg = cfg.get("rollout")
    if rcfg and bool(rcfg.get("enabled", False)):
        if not 0 <= float(rcfg.probability) <= 1:
            raise ValueError("rollout.probability must be in [0,1]")
        if not 0 < int(rcfg.min_calls) <= int(rcfg.max_calls) < int(rcfg.nfe):
            raise ValueError("rollout calls must satisfy 0 < min <= max < nfe")
        if int(rcfg.nfe) % int(rcfg.n_inner) or int(rcfg.n_inner) < 2:
            raise ValueError("rollout.nfe must divide by rollout.n_inner >= 2")
        if not 0 < float(rcfg.p) < 1 or float(rcfg.time_shift) <= 0:
            raise ValueError("rollout.p must be in (0,1) and time_shift positive")
        if float(rcfg.latent_weight) < 0 or float(rcfg.decoded_weight) < 0:
            raise ValueError("rollout loss weights must be nonnegative")
        if not float(rcfg.latent_weight) + float(rcfg.decoded_weight) > 0:
            raise ValueError("at least one rollout loss weight must be positive")


def resolve_resume(resume, output_dir):
    if resume is None:
        return None
    path = Path(output_dir) / "latest.pt" if resume == "auto" else Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {path}")
    invalid = path.with_name(path.name + ".invalid-data.json")
    if invalid.exists():
        raise ValueError(f"Checkpoint flagged for invalid training data: {invalid}. Use a verified clean backup in a new run.")
    return path


# ------------------------------------------------------------------- times

class EditTimeSampler:
    """Editable-token times with a garment-grounding -> refinement curriculum.

    Each example is one of:
      * pure noise   t=0 everywhere: the inference start state. Only the garment
                     and pose branches can explain the target, so this is what
                     teaches the model to use the garment condition;
      * synchronous  one logit-normal time for all editable tokens (Euler);
      * detail       half-normal lags below a late maximum in ``detail_range``;
                     ``detail_std`` optionally caps the lag standard deviation
                     for this branch alone (omitting it preserves legacy LTG);
      * otherwise    LTG below a logit-normal maximum (patch forcing).

    Mixture weights start at ``start`` and move to ``end`` over ``ramp_steps``
    once the garment condition is good enough: an EMA of ``gate_metric``
    reaches ``gate_threshold`` after ``gate_min_steps``, or ``gate_max_steps``
    is reached regardless.
    """

    KINDS = ("pure_noise", "synchronous", "detail")

    def __init__(self, cfg):
        f = cfg.flow
        c = f.curriculum
        self.ltg = LogitNormalTruncatedGaussian(std=float(f.ltg_std), loc=float(f.ltg_loc), scale=float(f.ltg_scale))
        self.start = torch.tensor([float(c.start[k]) for k in self.KINDS])
        self.end = torch.tensor([float(c.end[k]) for k in self.KINDS])
        if self.start.sum() > 1 or self.end.sum() > 1:
            raise ValueError("curriculum probabilities must sum to <= 1")
        self.detail_range = tuple(float(v) for v in c.detail_range)
        if len(self.detail_range) != 2 or not 0 <= self.detail_range[0] <= self.detail_range[1] <= 1:
            raise ValueError("detail_range must satisfy 0 <= low <= high <= 1")
        detail_std = c.get("detail_std")
        self.detail_std = None if detail_std is None else float(detail_std)
        if self.detail_std is not None and (not math.isfinite(self.detail_std) or self.detail_std <= 0):
            raise ValueError("detail_std must be a finite positive number")
        self.gate_metric = str(c.gate_metric)
        self.gate_threshold = float(c.gate_threshold)
        self.gate_min_steps = int(c.gate_min_steps)
        self.gate_max_steps = int(c.gate_max_steps)
        self.ramp_steps = max(int(c.ramp_steps), 1)
        # Applied after every branch, so branch ranges above are pre-shift.
        self.time_shift = float(f.get("time_shift", 1.0))
        # The resolution shift moves every time toward noise. The detail branch
        # exists to cover near-clean refinement in absolute terms, so it can be
        # exempted: its detail_range is then the actual patch-time range.
        self.detail_unshifted = bool(c.get("detail_unshifted", False))
        # Optional coarse-to-fine drift of the logit-normal location used by the
        # synchronous and LTG branches: ltg_loc_start early (t near 0, global
        # structure) -> ltg_loc_end once the ramp completes (t near 1, detail).
        loc = f.ltg_loc
        self.loc_start = float(c.get("ltg_loc_start", loc))
        self.loc_end = float(c.get("ltg_loc_end", loc))
        self.metric_ema = None
        self.ramp_start = None

    def observe(self, step: int, metrics: dict) -> None:
        value = metrics.get(self.gate_metric)
        if value is not None:
            self.metric_ema = value if self.metric_ema is None else 0.98 * self.metric_ema + 0.02 * value
        if self.ramp_start is None:
            good = self.metric_ema is not None and self.metric_ema >= self.gate_threshold
            if (good and step >= self.gate_min_steps) or step >= self.gate_max_steps:
                self.ramp_start = step
                print(f"curriculum: garment gate opened at step {step} ({self.gate_metric} ema={self.metric_ema})")

    def progress(self, step: int) -> float:
        if self.ramp_start is None:
            return 0.0
        return min(max((step - self.ramp_start) / self.ramp_steps, 0.0), 1.0)

    def probabilities(self, step: int) -> torch.Tensor:
        return torch.lerp(self.start, self.end, self.progress(step))

    def ltg_loc(self, step: int) -> float:
        return self.loc_start + (self.loc_end - self.loc_start) * self.progress(step)

    def state_dict(self) -> dict:
        return {"metric_ema": self.metric_ema, "ramp_start": self.ramp_start}

    def load_state_dict(self, state: dict) -> None:
        self.metric_ema = state.get("metric_ema")
        self.ramp_start = state.get("ramp_start")

    def __call__(self, b: int, n: int, device, step: int = 0) -> torch.Tensor:
        p_noise, p_sync, p_detail = self.probabilities(step).tolist()
        u = torch.rand(b, device=device)
        noise = u < p_noise
        sync = (u >= p_noise) & (u < p_noise + p_sync)
        detail = (u >= p_noise + p_sync) & (u < p_noise + p_sync + p_detail)

        self.ltg.loc = self.ltg_loc(step)
        t_bar = self.ltg.get_t_bar(b, device=device)
        lo, hi = self.detail_range
        t_bar = torch.where(detail, lo + (hi - lo) * torch.rand(b, device=device), t_bar)
        # t_bar is an upper bound, not the actual mean of patch times. A narrow
        # detail-only lag keeps these examples near clean without changing the
        # ordinary LTG distribution or adding random draws to the other branches.
        if self.detail_std is None:
            t = self.ltg.get_time_with_mean(t_bar, n)
        else:
            std_cap = torch.where(detail, self.detail_std, self.ltg.std)
            std = torch.minimum(t_bar / 2, std_cap)
            t = t_bar[:, None] - torch.randn(b, n, device=device, dtype=t_bar.dtype).abs() * std[:, None]
            # Same negative-tail replacement and RNG draw order as legacy LTG.
            rand = torch.rand_like(t)
            t = torch.where(t < 0, rand * t_bar[:, None], t)
        t = t.clamp(0, 1)
        t = torch.where(sync[:, None], t_bar[:, None].expand(-1, n), t)
        t = torch.where(noise[:, None], torch.zeros_like(t), t)
        shifted = shift_time(t, self.time_shift)
        return torch.where(detail[:, None], t, shifted) if self.detail_unshifted else shifted


# ------------------------------------------------------------------- utils

def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.expand_as(x)
    return (x * w).sum() / w.sum().clamp_min(1.0)


def _rgb_highpass(rgb: torch.Tensor) -> torch.Tensor:
    return rgb - F.avg_pool2d(F.pad(rgb, (2, 2, 2, 2), mode="reflect"), 5, 1)


def graphic_saliency(rgb: torch.Tensor, garment_mask: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) soft map of pixels that differ from the garment's base colour.

    Target-derived weighting for logos, text and prints (distance from the
    per-image median colour inside ``garment_mask``). Never an inference input.
    """
    bases = []
    for img, v in zip(rgb, garment_mask):
        use = v[0] > 0.5
        bases.append(img[:, use].median(1).values if use.any() else img.new_zeros(3))
    base = torch.stack(bases)[:, :, None, None]
    distance = (rgb - base).square().mean(1, keepdim=True).sqrt()
    return torch.sigmoid((distance - 0.16) / 0.035)


def _decode_windows(endpoint, clothing_mask, crop, margin, factor=8):
    """Latent windows (y0, y1, x0, x1) centred on each image's clothing, plus
    ``margin`` latents of decoder context on each side where the image allows."""
    h, w = endpoint.shape[-2:]
    ch, cw = min(int(crop[0]), h), min(int(crop[1]), w)
    windows = []
    for cloth in clothing_mask:
        ys, xs = torch.nonzero(cloth[0] > 0.5, as_tuple=True)
        cy = int(ys.float().mean() / factor) if len(ys) else h // 2
        cx = int(xs.float().mean() / factor) if len(xs) else w // 2
        cy += int(torch.randint(-ch // 4, ch // 4 + 1, ()))
        cx += int(torch.randint(-cw // 4, cw // 4 + 1, ()))
        y0 = min(max(cy - ch // 2, 0), h - ch)
        x0 = min(max(cx - cw // 2, 0), w - cw)
        inner = (y0, y0 + ch, x0, x0 + cw)
        outer = (max(y0 - margin, 0), min(y0 + ch + margin, h), max(x0 - margin, 0), min(x0 + cw + margin, w))
        windows.append((inner, outer))
    return windows


def decoded_detail_loss(vae, endpoint, target, clothing_mask, edit_mask, cfg, selected):
    """Optional decoded RGB/detail supervision on selected training examples.

    The target-derived color contrast mask weights the loss only; it is never
    passed to the model or an inference sampler and is not an OCR label.
    With ``loss.decoded_crop_latent`` = [h, w], only a latent window around the
    clothing is decoded (plus ``decoded_crop_margin`` latents of context whose
    pixels are excluded from the loss): full 1024x768 decoder backward does not
    fit next to the model on a 24 GB card.
    """
    zero = endpoint.sum() * 0.0
    if not selected.any():
        return zero, {"loss_decoded_rgb": zero, "loss_decoded_highpass": zero,
                      "loss_decoded_total": zero}
    with torch.autocast(endpoint.device.type, enabled=False):
        def decode(z):
            return decode_latents_with_grad(
                vae, z, checkpoint_decoder=bool(cfg.loss.get("decoded_checkpoint", False))
            ).float()
        z = endpoint[selected].float()
        truth = target[selected].float()
        cloth = clothing_mask[selected].float()
        edit = edit_mask[selected].float()
        crop = cfg.loss.get("decoded_crop_latent")
        if crop:
            margin = int(cfg.loss.get("decoded_crop_margin", 4))
            preds, truths, cloths, edits, valids = [], [], [], [], []
            for i, (inner, outer) in enumerate(_decode_windows(z, cloth, crop, margin)):
                oy0, oy1, ox0, ox1 = outer
                pred = decode(z[i:i + 1, :, oy0:oy1, ox0:ox1])
                sl = (slice(None), slice(None), slice(oy0 * 8, oy1 * 8), slice(ox0 * 8, ox1 * 8))
                valid = torch.zeros_like(pred[:, :1])
                valid[..., (inner[0] - oy0) * 8:(inner[1] - oy0) * 8, (inner[2] - ox0) * 8:(inner[3] - ox0) * 8] = 1
                preds.append(pred); truths.append(truth[i:i + 1][sl]); cloths.append(cloth[i:i + 1][sl])
                edits.append(edit[i:i + 1][sl]); valids.append(valid)
            # Windows share one size except at image borders; handle one by one.
            parts = list(zip(preds, truths, cloths, edits, valids))
        else:
            parts = [(decode(z), truth, cloth, edit, torch.ones_like(cloth))]
        boost = float(cfg.loss.get("decoded_graphic_boost", 0.0))
        num_rgb, num_hp, den = zero, zero, zero
        for pred, tr, cl, ed, valid in parts:
            mask = cl * ed
            if boost:
                mask = mask * (1.0 + boost * graphic_saliency(tr, cl).detach())
            hp_err = (_rgb_highpass(pred) - _rgb_highpass(tr)).abs()
            mask = (mask * valid).expand_as(pred)
            num_rgb = num_rgb + ((pred - tr).abs() * mask).sum()
            num_hp = num_hp + (hp_err * mask).sum()
            den = den + mask.sum()
        rgb_loss = num_rgb / den.clamp_min(1.0)
        highpass_loss = num_hp / den.clamp_min(1.0)
        total = (float(cfg.loss.get("decoded_rgb_weight", 0.0)) * rgb_loss +
                 float(cfg.loss.get("decoded_highpass_weight", 0.0)) * highpass_loss)
    return total, {"loss_decoded_rgb": rgb_loss, "loss_decoded_highpass": highpass_loss,
                   "loss_decoded_total": total}


@torch.no_grad()
def update_ema(ema, model, decay: float):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1.0 - decay)


def lr_factor(step: int, cfg) -> float:
    o = cfg.optim
    if step < o.warmup_steps:
        return (step + 1) / o.warmup_steps
    progress = min((step - o.warmup_steps) / max(cfg.train.max_steps - o.warmup_steps, 1), 1.0)
    return o.min_lr_ratio + (1 - o.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def prune_snapshots(out_dir: Path, keep_last: int) -> list:
    """Delete all but the newest ``keep_last`` weight snapshots (0 = keep all)."""
    snaps = sorted(Path(out_dir).glob("step[0-9]*.pt"))
    doomed = snaps[:-keep_last] if keep_last > 0 else []
    for p in doomed:
        p.unlink()
    return doomed


def detail_sampling_weights(rows, scores: dict, boost: float, reference: float) -> torch.Tensor:
    """Per-row sampling weight 1 + boost * min(score / reference, 1).

    ``scores`` maps a garment file name to the fraction of its pixels that are
    graphics/text (scripts/score_garment_detail.py). Missing garments get 1.
    """
    if boost < 0 or reference <= 0:
        raise ValueError("detail_sampling needs boost >= 0 and reference > 0")
    s = torch.tensor([float(scores.get(g, 0.0)) for _, g in rows])
    return 1.0 + boost * (s / reference).clamp(0, 1)


def detail_sampler(train_ds, cfg):
    """Oversample garments with prints/text; None keeps plain shuffling."""
    dcfg = cfg.data.get("detail_sampling")
    if not dcfg or not bool(dcfg.get("enabled", False)):
        return None
    scores = json.loads(Path(dcfg.scores).read_text())
    weights = detail_sampling_weights(train_ds.rows, scores, float(dcfg.boost), float(dcfg.reference))
    known = sum(g in scores for _, g in train_ds.rows)
    if known < len(train_ds.rows):
        raise ValueError(f"detail scores cover {known}/{len(train_ds.rows)} training garments")
    share = float((weights > 1.5).float().mean())
    print(f"detail sampling: mean weight {float(weights.mean()):.3f}, {share:.1%} of garments boosted >1.5x",
          flush=True)
    return torch.utils.data.WeightedRandomSampler(weights.double(), len(weights), replacement=True)


def infinite(loader):
    while True:
        yield from loader


def make_dataset(cfg, phase_pairs: str, augment: bool):
    d = cfg.data
    return VitonHDDataset(
        d.root, phase="train", order="paired", size=tuple(cfg.model.image_hw), pairs_file=phase_pairs,
        require_cloth_mask=True, require_parse=True, clothing_labels=list(d.clothing_labels),
        augment=augment,
        person_translate_fraction=d.person_translate_fraction, person_scale_range=list(d.person_scale_range),
        garment_translate_fraction=d.garment_translate_fraction, garment_scale_range=list(d.garment_scale_range),
    )


# -------------------------------------------------------------------- loss

def training_loss(model, teacher, vae, batch, cfg, time_sampler, device, step: int = 0):
    b = batch["image"].shape[0]
    drop = torch.rand(b, device=device) < float(cfg.flow.garment_dropout)
    lo, hi = (int(v) for v in cfg.data.mask_open_px_range)
    open_px = torch.randint(lo, hi + 1, (b,), device=device)
    inputs = prepare_inputs(vae, batch, model, device, drop_garment=drop, mask_open_px=open_px)
    with torch.no_grad():
        from vton_ext.vae import encode_images
        x1 = encode_images(vae, batch["image"].to(device)).float()
    edit_tok, edit_pix = inputs["edit_tokens"], inputs["edit_pixels"]

    t_edit = time_sampler(b, model.num_tokens, device, step)
    t = torch.where(edit_tok, t_edit, torch.ones_like(t_edit))
    x0 = torch.randn_like(x1)
    t_pix = expand_patch_values(t, model.patch_size, model.latent_hw)
    # Known tokens hold the agnostic latent, exactly what inference provides.
    xt = torch.where(edit_pix.bool(), t_pix * x1 + (1 - t_pix) * x0, inputs["known"])
    ut = x1 - x0

    use_coral = teacher is not None and float(cfg.coral.weight) > 0
    # Compute the frozen teacher before retaining the transformer's backward
    # graph. At high resolution its dense similarity matrix is substantial.
    targets = None
    if use_coral:
        grid = model.token_hw
        clothing_tok = pool_valid_mask(batch["clothing_mask"].to(device), grid, threshold=0.25)
        garment_valid = pool_valid_mask(batch["garment_mask"].to(device), grid, threshold=0.25)
        with torch.no_grad():
            targets = teacher.build_targets(
                batch["image"].to(device), batch["garment"].to(device),
                edit_tok & clothing_tok & ~drop[:, None], garment_valid, grid, grid,
            )
    out = model(
        xt, t, inputs["cond"], edit_tok, garment_latent=inputs["garment"], garment_mask=inputs["garment_mask"],
        return_uncertainty=True, return_attention=use_coral,
    )
    v, logvar = out[0].float(), out[1].float()

    clothing = F.adaptive_max_pool2d(batch["clothing_mask"].to(device).float(), model.latent_hw)
    w = edit_pix * (1.0 + float(cfg.loss.clothing_boost) * clothing)
    flow = masked_mean((v - ut).square(), w)
    nll = 0.5 * (logvar + (ut - v.detach()).square().mean(1, keepdim=True) / logvar.exp())
    nll = masked_mean(nll, edit_pix)
    loss = flow + float(cfg.loss.uncertainty_weight) * nll
    edit_count = edit_tok.sum().clamp_min(1)
    metrics = {
        "loss_flow": flow,
        "loss_nll": nll,
        "t_edit_mean": (t * edit_tok).sum() / edit_count,
        "t_edit_gt_08": ((t > 0.8) & edit_tok).sum() / edit_count,
        "t_edit_gt_09": ((t > 0.9) & edit_tok).sum() / edit_count,
    }

    if use_coral:
        corr, ent, cm = coral_routing_loss(
            out[2], targets, grid, gaussian_sigma=float(cfg.coral.gaussian_sigma),
            checkpoint_loss=bool(cfg.coral.get("checkpoint_loss", False)),
        )
        loss = loss + float(cfg.coral.weight) * corr + float(cfg.coral.entropy_weight) * ent
        metrics.update(loss_coral=corr, loss_coral_entropy=ent, **cm)
    # Optionally ramp the fine-detail objective with the time curriculum: while
    # pure noise dominates the model is learning to read the garment, and
    # sharpening its endpoint would only sharpen wrong content.
    decoded_scale = time_sampler.progress(step) if bool(cfg.loss.get("decoded_follow_curriculum", False)) else 1.0
    if decoded_scale > 0 and (float(cfg.loss.get("decoded_rgb_weight", 0.0)) > 0 or
                              float(cfg.loss.get("decoded_highpass_weight", 0.0)) > 0):
        p = float(cfg.loss.get("decoded_probability", 1.0))
        max_images = int(cfg.loss.get("decoded_max_images", 1))
        if not 0 <= p <= 1 or max_images < 1:
            raise ValueError("decoded_probability must be in [0,1] and decoded_max_images >= 1")
        t_mean = (t * edit_tok).sum(1) / edit_tok.sum(1).clamp_min(1)
        eligible = ((t_mean >= float(cfg.loss.get("decoded_min_time", 0.0))) & ~drop &
                    (batch["clothing_mask"].to(device).flatten(1).sum(1) > 0))
        if p < 1:
            eligible &= torch.rand(b, device=device) < p
        candidates = eligible.nonzero(as_tuple=False).flatten()
        if len(candidates) > max_images:
            chosen = candidates[torch.randperm(len(candidates), device=device)[:max_images]]
            eligible = torch.zeros_like(eligible)
            eligible[chosen] = True
        endpoint = xt + (1 - t_pix) * v
        decoded_loss, decoded_metrics = decoded_detail_loss(
            vae, endpoint, batch["image"].to(device), batch["clothing_mask"].to(device),
            inputs["pixel_mask"], cfg, eligible,
        )
        loss = loss + decoded_scale * decoded_loss
        metrics.update(**decoded_metrics, decoded_scale=decoded_scale, decoded_selected=eligible.float().sum(),
                       decoded_time=(t_mean * eligible).sum() / eligible.sum().clamp_min(1))
    return loss, metrics


@torch.no_grad()
def short_dual_loop_rollout(model, inputs: dict, calls: int, nfe: int = 8,
                            n_inner: int = 2, p: float = 0.7, time_shift: float = 1.0,
                            noise: torch.Tensor | None = None):
    """Return the exact partial state after `calls` dual-loop model calls.

    The garment and observed person context are kept as in inference. Gradients
    are deliberately stopped through the rollout; only its endpoint correction
    receives training gradients.
    """
    if not 0 < calls <= nfe or nfe % n_inner or n_inner < 2:
        raise ValueError("rollout calls must be within nfe and nfe divisible by n_inner >= 2")
    edit, edit_pix = inputs["edit_tokens"], inputs["edit_pixels"]
    known = inputs["known"]
    x = torch.where(edit_pix.bool(), torch.randn_like(known) if noise is None else noise, known).float()
    t = torch.where(edit, 0.0, 1.0).to(known.dtype)
    garment_kv = model.encode_garment(inputs["garment"], inputs["garment_mask"])
    levels = shift_time(torch.linspace(0, 1, nfe // n_inner + 1, device=known.device), time_shift)
    count = 0

    def advance(state, times, velocity, dt_tok):
        dt_tok = dt_tok * edit
        return (state + expand_patch_values(dt_tok, model.patch_size, model.latent_hw) * velocity,
                (times + dt_tok).clamp(max=1.0))

    for i in range(nfe // n_inner):
        dt = float(levels[i + 1] - levels[i])
        v, logvar = model(x, t, inputs["cond"], edit, garment_kv=garment_kv,
                          return_uncertainty=True)
        hard = uncertain_tokens(token_uncertainty(logvar, model.patch_size), edit, p)
        step = torch.where(hard, dt / n_inner, dt)
        x, t = advance(x, t, v.float(), step)
        count += 1
        if count == calls:
            return x.detach(), t.detach()
        for _ in range(n_inner - 1):
            v = model(x, t, inputs["cond"], edit, garment_kv=garment_kv).float()
            x, t = advance(x, t, v, hard.to(t.dtype) * (dt / n_inner))
            count += 1
            if count == calls:
                return x.detach(), t.detach()
    raise AssertionError("unreachable rollout call count")


def rollout_endpoint_loss(model, vae, batch, cfg, device):
    """Supervise an endpoint predicted from a state reached by inference.

    A randomly selected paired example is rolled out for configured calls of the eight
    inference steps. The model then predicts the clean endpoint from that state,
    with full decoded garment RGB/high-pass loss and an editable clothing-latent
    anchor. This complements the ordinary straight-interpolation flow loss.
    """
    rcfg = cfg.rollout
    b = batch["image"].shape[0]
    j = int(torch.randint(b, (1,), device=device))
    example = {k: v[j:j + 1] for k, v in batch.items() if torch.is_tensor(v)}
    calls = int(torch.randint(int(rcfg.min_calls), int(rcfg.max_calls) + 1, (1,), device=device))
    # Keep no-grad rollout casts out of the gradient-bearing autocast cache.
    # Reusing those casts can make checkpoint recomputation save different
    # tensors from the forward pass on CUDA.
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        open_px = int(torch.randint(int(cfg.data.mask_open_px_range[0]),
                                    int(cfg.data.mask_open_px_range[1]) + 1, (1,), device=device))
        inputs = prepare_inputs(vae, example, model, device, mask_open_px=open_px)
        from vton_ext.vae import encode_images
        truth = encode_images(vae, example["image"].to(device)).float()
        x, t = short_dual_loop_rollout(
            model, inputs, calls, nfe=int(rcfg.nfe), n_inner=int(rcfg.n_inner),
            p=float(rcfg.p), time_shift=float(rcfg.time_shift),
        )

    edit_pix = inputs["edit_pixels"]
    t_pix = expand_patch_values(t, model.patch_size, model.latent_hw)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        v = model(x, t, inputs["cond"], inputs["edit_tokens"],
                  garment_latent=inputs["garment"], garment_mask=inputs["garment_mask"]).float()
    endpoint = torch.where(edit_pix.bool(), x + (1 - t_pix) * v, inputs["known"])
    clothing = F.adaptive_avg_pool2d(example["clothing_mask"].to(device).float(), model.latent_hw)
    latent_mask = clothing * edit_pix
    latent_loss = masked_mean((endpoint - truth).abs(), latent_mask)
    decoded_loss, decoded_metrics = decoded_detail_loss(
        vae, endpoint, example["image"].to(device), example["clothing_mask"].to(device),
        inputs["pixel_mask"], cfg, torch.ones(1, device=device, dtype=torch.bool),
    )
    total = float(rcfg.latent_weight) * latent_loss + float(rcfg.decoded_weight) * decoded_loss
    return total, {"rollout_loss": total.detach(), "rollout_latent_l1": latent_loss.detach(),
                   "rollout_decoded_rgb": decoded_metrics["loss_decoded_rgb"].detach(),
                   "rollout_decoded_highpass": decoded_metrics["loss_decoded_highpass"].detach(),
                   "rollout_time": t[inputs["edit_tokens"]].mean().detach(),
                   "rollout_calls": float(calls)}


# ----------------------------------------------------------------- preview

@torch.no_grad()
def evaluate(model, vae, batches, cfg, device, out_dir: Path, step: int) -> dict:
    model.eval()
    ev = cfg.eval
    results, rows = {}, []
    for bi, batch in enumerate(batches):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = prepare_inputs(vae, batch, model, device, mask_open_px=int(ev.mask_open_px))
            noise = torch.cat([
                torch.randn((1, *inputs["known"].shape[1:]), device=device,
                            generator=torch.Generator(device=device).manual_seed(
                                int(ev.seed) + bi * int(ev.batch_size) + j))
                for j in range(inputs["known"].shape[0])
            ])
            images = []
            for spec in ev.samplers:
                lat = generate(model, inputs, nfe=int(spec.nfe), sampler=spec.name, p=float(ev.p),
                               n_inner=int(ev.n_inner), cfg_scale=float(spec.get("cfg", 1.0)),
                               cfg_interval=tuple(spec.get("cfg_interval", [0.0, 1.0])), noise=noise,
                               time_shift=spec.get("time_shift"))
                images.append(lat)
        target = batch["image"].to(device)
        mask = inputs["pixel_mask"]
        cloth = batch["clothing_mask"].to(device)
        row = [batch["garment"].to(device), inputs["agnostic_rgb"], target]
        for spec, lat in zip(ev.samplers, images):
            rgb = composite(decode_latents(vae, lat.float()).float(), inputs["agnostic_rgb"], mask,
                            seam_px=float(ev.get("seam_px", 0.0)), feather_px=float(ev.get("feather_px", 0.0)))
            key = f"{spec.name}{spec.nfe}" + (f"_cfg{float(spec.cfg):g}" if float(spec.get("cfg", 1.0)) != 1.0 else "")
            if spec.get("time_shift") is not None:
                key += f"_shift{float(spec.time_shift):g}"
            err = (rgb - target).abs().mean(1, keepdim=True)
            results.setdefault(f"{key}_mask_l1", []).append(float(masked_mean(err, mask)))
            results.setdefault(f"{key}_cloth_l1", []).append(float(masked_mean(err, cloth)))
            # Detail-sensitive scores: plain L1 prefers blur, so also track the
            # error on the garment's graphics/text and on its high frequencies.
            graphic = cloth * graphic_saliency(target, cloth)
            results.setdefault(f"{key}_graphic_l1", []).append(float(masked_mean(err, graphic)))
            hp = (_rgb_highpass(rgb) - _rgb_highpass(target)).abs().mean(1, keepdim=True)
            results.setdefault(f"{key}_cloth_hp", []).append(float(masked_mean(hp, cloth)))
            row.append(rgb)
        rows.append(torch.stack(row, 1).flatten(0, 1).cpu())
    grid = torch.cat(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, out_dir / f"step{step:07d}.jpg", nrow=3 + len(ev.samplers))
    model.train()
    return {k: sum(v) / len(v) for k, v in results.items()}


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for <output_dir>/latest.pt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config, args.overrides)
    validate_config(cfg)
    resume = resolve_resume(args.resume, cfg.train.output_dir)
    # A bounded pilot must not shorten the cosine LR horizon in train.max_steps.
    stop_at_step = int(cfg.train.get("stop_at_step", cfg.train.max_steps))
    if not 0 < stop_at_step <= int(cfg.train.max_steps):
        raise ValueError("train.stop_at_step must be positive and <= train.max_steps")
    # Validate supervision before allocating the large model or touching run logs.
    train_ds = make_dataset(cfg, cfg.data.pairs_file, augment=bool(cfg.data.augment))
    dev_base = make_dataset(cfg, cfg.data.test_pairs_file, augment=False)
    data_pairs = split_manifest(train_ds.pair_path, dev_base.pair_path)
    if any(not 0 <= int(i) < len(dev_base) for i in cfg.eval.indices):
        raise ValueError("eval.indices must be valid indices in the development split")
    ckpt = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if ckpt is not None:
        if "optimizer" not in ckpt:
            raise ValueError("--resume requires a full optimizer checkpoint; use weights.init_from for weights only")
        if not validate_resume_pairs(ckpt.get("data_pairs"), data_pairs):
            if bool(cfg.train.get("require_pair_fingerprints", False)):
                raise ValueError("This run requires checkpoint pair fingerprints. Use verified clean weights.init_from for a legacy checkpoint.")
            print("WARNING: legacy checkpoint has no pair fingerprints; its original split cannot be verified.", flush=True)
    seed_everything(int(cfg.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    model = build_model(cfg, load_pretrained=resume is None).to(device)
    model.gradient_checkpointing = bool(cfg.train.gradient_checkpointing)
    model.train()
    ema = None
    if float(cfg.train.ema_decay) > 0:
        ema = copy.deepcopy(model).eval().requires_grad_(False)

    vae = load_sd_vae(cfg.weights.vae, device)
    teacher = None
    if float(cfg.coral.weight) > 0:
        teacher = DINOv3CoralTeacher(
            cfg.weights.dino, min_similarity=float(cfg.coral.min_similarity), cycle_radius=float(cfg.coral.cycle_radius)
        ).to(device)

    o = cfg.optim
    groups = [
        {"params": [p for _, p in model.new_parameters()], "lr": float(o.lr), "weight_decay": 0.0},
        {"params": [p for _, p in model.pretrained_parameters() if p.requires_grad],
         "lr": float(o.lr) * float(o.pretrained_lr_multiplier), "weight_decay": float(o.weight_decay)},
    ]
    for g in groups:
        g["base_lr"] = g["lr"]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=True)

    step = 0
    time_sampler_state = None
    if resume:
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        time_sampler_state = ckpt.get("time_sampler")
        print(f"resumed {resume} at step {step}")
        del ckpt

    sampler = detail_sampler(train_ds, cfg)
    loader = DataLoader(train_ds, batch_size=int(cfg.train.batch_size), shuffle=sampler is None,
                        sampler=sampler, drop_last=True,
                        num_workers=int(cfg.data.num_workers), pin_memory=True,
                        persistent_workers=int(cfg.data.num_workers) > 0)
    dev_ds = Subset(dev_base, list(cfg.eval.indices))
    dev_batches = list(DataLoader(dev_ds, batch_size=int(cfg.eval.batch_size), shuffle=False))
    time_sampler = EditTimeSampler(cfg)
    if time_sampler_state:
        time_sampler.load_state_dict(time_sampler_state)
    accum = int(cfg.train.gradient_accumulation_steps)
    it = infinite(loader)
    log_path = out_dir / "metrics.jsonl"
    running, t0 = {}, time.time()

    keep_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[str(cfg.train.get("keep_dtype", "fp32"))]
    keep_last = int(cfg.train.get("keep_last", 0))
    min_free_gb = float(cfg.train.get("min_free_gb", 0.0))

    def save(name: str, with_optimizer: bool, dtype=None):
        # Also catch an external pair-file replacement during a running job.
        validate_resume_pairs(data_pairs, split_manifest(train_ds.pair_path, dev_base.pair_path))
        weights = model.state_dict()
        if dtype is not None:  # weights-only snapshots; loading casts back to fp32
            weights = {k: v.to(dtype) if v.is_floating_point() else v for k, v in weights.items()}
        state = {"step": step, "model": weights, "ema": ema.state_dict() if ema is not None else None,
                 "time_sampler": time_sampler.state_dict(),
                 "data_pairs": data_pairs,
                 "config": OmegaConf.to_container(cfg)}
        if with_optimizer:
            state["optimizer"] = optimizer.state_dict()
        tmp = out_dir / f".{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(out_dir / name)

    if bool(cfg.eval.at_start) and step == 0:
        rec = {"step": 0, **evaluate(ema or model, vae, dev_batches, cfg, device, out_dir / "previews", 0)}
        print(json.dumps(rec), flush=True)
        with (out_dir / "eval.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        t0 = time.time()

    while step < stop_at_step:
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * lr_factor(step, cfg)
        step_metrics = {}
        for _ in range(accum):
            batch = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = training_loss(model, teacher, vae, batch, cfg, time_sampler, device, step)
            (loss / accum).backward()
            # Backpropagate separately so the full primary training graph is
            # released before the extra rollout and decoder graphs are built.
            del loss
            metrics = {k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
            rcfg = cfg.get("rollout")
            rollout_selected = bool(rcfg and rcfg.get("enabled", False) and
                                    torch.rand((), device=device) < float(rcfg.probability))
            metrics["rollout_selected"] = float(rollout_selected)
            if rollout_selected:
                auxiliary, rollout_metrics = rollout_endpoint_loss(model, vae, batch, cfg, device)
                (auxiliary / accum).backward()
                metrics.update(rollout_metrics)
                del auxiliary
            for k, v in metrics.items():
                step_metrics.setdefault(k, []).append(float(v.detach()) if torch.is_tensor(v) else float(v))
        step_metrics = {k: sum(v) / len(v) for k, v in step_metrics.items()}
        for k, v in step_metrics.items():
            running.setdefault(k, []).append(v)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(o.clip_grad_norm), error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        time_sampler.observe(step, step_metrics)
        if ema is not None:
            update_ema(ema, model, float(cfg.train.ema_decay) if step > int(cfg.train.ema_start) else 0.0)

        if step % int(cfg.train.log_every) == 0:
            rec = {"step": step, **{k: sum(v) / len(v) for k, v in running.items()},
                   "grad_norm": float(grad_norm), "lr": optimizer.param_groups[1]["lr"],
                   "curriculum": time_sampler.progress(step), "ltg_loc": time_sampler.ltg_loc(step),
                   **{f"p_{k}": v for k, v in zip(EditTimeSampler.KINDS, time_sampler.probabilities(step).tolist())},
                   "sec_per_step": (time.time() - t0) / int(cfg.train.log_every),
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            running, t0 = {}, time.time()
        if (step % int(cfg.train.save_every) == 0 or step == stop_at_step
                or step == int(cfg.train.get("save_first_step", -1))):
            save("latest.pt", with_optimizer=True)
        if step % int(cfg.train.keep_every) == 0:
            # Snapshots are expendable; latest.pt (needed to resume) is not.
            # Skip a snapshot rather than let the disk fill during a later save.
            free_gb = shutil.disk_usage(out_dir).free / 2**30
            if free_gb < min_free_gb:
                print(f"skip snapshot at step {step}: {free_gb:.1f} GB free < {min_free_gb} GB", flush=True)
            else:
                save(f"step{step:07d}.pt", with_optimizer=False, dtype=keep_dtype)
                for p in prune_snapshots(out_dir, keep_last):
                    print(f"pruned {p.name}", flush=True)
        if step % int(cfg.eval.every) == 0 or step == stop_at_step:
            res = evaluate(ema or model, vae, dev_batches, cfg, device, out_dir / "previews", step)
            print(json.dumps({"step": step, **res}), flush=True)
            with (out_dir / "eval.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **res}) + "\n")


if __name__ == "__main__":
    main()
