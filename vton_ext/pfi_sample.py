"""Patch-forcing inference for the PFI try-on model.

Information available at inference, and how each sampler uses it:

* known person patches and every garment patch are clean (t=1) for the whole
  trajectory; garment K/V are computed once and reused for every call;
* only tokens inside the agnostic mask are integrated, starting from N(0, I);
* ``dual_loop``: after each full evaluation, patches the model is confident
  about (predicted log-variance) take the whole step, uncertain patches take
  ``n_inner`` smaller steps, each evaluated with the cleaner confident context;
* ``look_ahead``: confident patches are extrapolated ahead to provide a
  cleaner context for a second evaluation of the uncertain ones.

Every sampler is budgeted by sampler steps ``nfe``, not by outer loops. With
garment guidance (cfg_scale != 1) each step costs two person-denoiser calls;
``return_stats`` reports both counts. The garment encoder runs once (twice with
guidance) per image and is not counted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from vton_ext.utils import expand_patch_values, latent_edit_mask, token_edit_mask


SAMPLERS = ("euler", "dual_loop", "look_ahead")


# ----------------------------------------------------------------- inputs

def open_mask(mask: torch.Tensor, pixels) -> torch.Tensor:
    """Dilate a (B,1,H,W) binary mask by a per-sample radius in pixels.

    A slightly-too-tight agnostic mask leaves old-garment pixels at its border;
    opening it lets the model regenerate that ring instead of copying them.
    """
    radii = torch.as_tensor(pixels, device=mask.device).long().reshape(-1).expand(mask.shape[0])
    out = mask.clone()
    for r in radii.unique().tolist():
        if r > 0:
            sel = radii == r
            out[sel] = F.max_pool2d(mask[sel].float(), 2 * r + 1, stride=1, padding=r).to(mask.dtype)
    return out


@torch.no_grad()
def prepare_inputs(vae, batch, model, device, drop_garment=None, mask_open_px=0):
    """VAE-encode one batch into model inputs. The target image is never used here.

    ``mask_open_px``: int or (B,) radii. The opened ring is grayed out in the
    agnostic image (VITON-HD fill value 0 in [-1, 1]) and becomes editable.
    """
    from vton_ext.vae import encode_images

    latent_hw = model.latent_hw
    mask = open_mask(batch["agnostic_mask"].to(device), mask_open_px)
    agnostic_rgb = batch["agnostic"].to(device) * (1 - mask)
    agnostic = encode_images(vae, agnostic_rgb).float()
    pose = encode_images(vae, batch["densepose"].to(device)).float()
    garment = encode_images(vae, batch["garment"].to(device)).float()
    mask_lat = latent_edit_mask(mask, latent_hw)
    garment_mask = F.adaptive_avg_pool2d(batch["garment_mask"].to(device).float(), latent_hw)
    if drop_garment is not None:
        keep = (~drop_garment).float()[:, None, None, None]
        garment = garment * keep
        garment_mask = garment_mask * keep
    edit_tokens = token_edit_mask(mask_lat, model.patch_size)
    edit_pixels = expand_patch_values(edit_tokens.float(), model.patch_size, latent_hw)
    return {
        "known": agnostic,
        "cond": torch.cat([agnostic, mask_lat, pose], 1),
        "garment": garment,
        "garment_mask": garment_mask,
        "edit_tokens": edit_tokens,
        "edit_pixels": edit_pixels,
        "pixel_mask": mask,
        "agnostic_rgb": agnostic_rgb,
    }


def shift_time(t, a: float):
    """Resolution shift t' = t / (t + a (1 - t)); a > 1 spends more time near noise."""
    if a == 1.0:
        return t
    return t / (t + a * (1 - t))


def token_uncertainty(logvar: torch.Tensor, patch_size: int) -> torch.Tensor:
    return F.avg_pool2d(logvar.float().exp(), patch_size).flatten(1)


def uncertain_tokens(uq: torch.Tensor, edit_tokens: torch.Tensor, p: float) -> torch.Tensor:
    """True for editable tokens above the p-quantile of editable uncertainty."""
    masked = uq.masked_fill(~edit_tokens, float("nan"))
    thresh = torch.nanquantile(masked.double(), p, dim=1, keepdim=True).to(uq.dtype)
    return edit_tokens & (uq >= thresh)


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def generate(
    model,
    inputs: dict,
    nfe: int = 8,
    sampler: str = "dual_loop",
    p: float = 0.7,
    n_inner: int = 2,
    context_ratio: float = 1.5,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    noise: torch.Tensor | None = None,
    return_stats: bool = False,
    time_shift: float | None = None,
):
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}")
    ps = model.patch_size
    a = float(getattr(model, "time_shift", 1.0) if time_shift is None else time_shift)
    edit_tok = inputs["edit_tokens"]
    edit_pix = inputs["edit_pixels"]
    known = inputs["known"]
    b = known.shape[0]

    garment_kv = model.encode_garment(inputs["garment"], inputs["garment_mask"])
    null_kv = None
    if cfg_scale != 1.0:
        null_kv = model.encode_garment(torch.zeros_like(inputs["garment"]), torch.zeros_like(inputs["garment_mask"]))

    calls = {"n": 0, "denoiser": 0}

    def predict(x, t):
        calls["n"] += 1
        calls["denoiser"] += 1
        v, logvar = model(x, t, inputs["cond"], edit_tok, garment_kv=garment_kv, return_uncertainty=True)
        if null_kv is not None:
            calls["denoiser"] += 1
            # Garment guidance per token, only while its own time is inside
            # cfg_interval: guiding the noisiest steps mostly shifts colours.
            lo, hi = cfg_interval
            w = torch.where((t >= lo) & (t <= hi), cfg_scale, 1.0)
            v0 = model(x, t, inputs["cond"], edit_tok, garment_kv=null_kv)
            v = v0 + expand(w) * (v - v0)
        return v.float(), logvar.float()

    def expand(tok):  # (B,N) -> (B,1,h,w)
        return expand_patch_values(tok.to(known.dtype), ps, model.latent_hw)

    if noise is None:
        noise = torch.randn_like(known)
    x = torch.where(edit_pix.bool(), noise, known)
    t = torch.where(edit_tok, 0.0, 1.0).to(known.dtype)

    def advance(x, t, v, dt_tok):
        dt_tok = dt_tok * edit_tok
        return x + expand(dt_tok) * v, (t + dt_tok).clamp(max=1.0)

    uq_trace = []
    if sampler == "euler":
        levels = shift_time(torch.linspace(0, 1, nfe + 1, device=known.device), a)
        for i in range(nfe):
            v, logvar = predict(x, t)
            x, t = advance(x, t, v, torch.full_like(t, float(levels[i + 1] - levels[i])))
            uq_trace.append(token_uncertainty(logvar, ps))

    elif sampler == "dual_loop":
        if nfe % n_inner:
            raise ValueError("dual_loop needs nfe divisible by n_inner")
        outer = nfe // n_inner
        levels = shift_time(torch.linspace(0, 1, outer + 1, device=known.device), a)
        for i in range(outer):
            dt = float(levels[i + 1] - levels[i])
            v, logvar = predict(x, t)
            uq = token_uncertainty(logvar, ps)
            uq_trace.append(uq)
            hard = uncertain_tokens(uq, edit_tok, p)
            step = torch.where(hard, dt / n_inner, dt)
            x, t = advance(x, t, v, step)
            for _ in range(n_inner - 1):
                v, _ = predict(x, t)
                x, t = advance(x, t, v, hard.to(t.dtype) * (dt / n_inner))

    else:  # look_ahead: two evaluations per step
        if nfe % 2:
            raise ValueError("look_ahead needs an even nfe")
        steps = nfe // 2
        levels = shift_time(torch.linspace(0, 1, steps + 1, device=known.device), a)
        for i in range(steps):
            t_now, dt = float(levels[i]), float(levels[i + 1] - levels[i])
            v, logvar = predict(x, t)
            uq = token_uncertainty(logvar, ps)
            uq_trace.append(uq)
            hard = uncertain_tokens(uq, edit_tok, p)
            easy = edit_tok & ~hard
            jump = min(max(t_now, dt) * context_ratio, 1.0) - t_now
            x_ctx, t_ctx = advance(x, t, v, easy.to(t.dtype) * jump)
            v_ctx, _ = predict(x_ctx, t_ctx)
            v_mix = torch.where(expand(hard).bool(), v_ctx, v)
            x, t = advance(x, t, v_mix, torch.full_like(t, dt))

    x = torch.where(edit_pix.bool(), x, known)
    if return_stats:
        return x, {"nfe": calls["n"], "denoiser_calls": calls["denoiser"], "uncertainty": uq_trace}
    return x


def composite(decoded: torch.Tensor, agnostic_rgb: torch.Tensor, agnostic_mask: torch.Tensor) -> torch.Tensor:
    """Keep every observed person pixel exactly; generated pixels only inside the mask."""
    m = agnostic_mask.to(decoded.dtype)
    return (m * decoded + (1 - m) * agnostic_rgb).clamp(-1, 1)


# --------------------------------------------------------------------- CLI

def load_for_inference(cfg, checkpoint_path: str, device, use_ema: bool = True):
    from vton_ext.pfi_train import build_model

    model = build_model(cfg, load_pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "ema" if use_ema and ckpt.get("ema") is not None else "model"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval(), ckpt.get("step", -1)


def main(argv=None):
    from vton_ext.data import VitonHDDataset
    from vton_ext.vae import decode_latents, load_sd_vae

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase", default="test")
    ap.add_argument("--order", default="unpaired", choices=["paired", "unpaired"])
    ap.add_argument("--pairs-file", default=None)
    ap.add_argument("--num-samples", type=int, default=0, help="0 = whole split")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--sampler", default="dual_loop", choices=SAMPLERS)
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--n-inner", type=int, default=2)
    ap.add_argument("--cfg-scale", type=float, default=None, help="default: eval.cfg_scale or 1")
    ap.add_argument("--cfg-interval", type=float, nargs=2, default=None, help="default: eval.cfg_interval or 0 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--mask-open-px", type=int, default=None, help="default: eval.mask_open_px")
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    model, step = load_for_inference(cfg, args.checkpoint, device, use_ema=not args.no_ema)
    vae = load_sd_vae(cfg.weights.vae, device)
    pairs = args.pairs_file or (
        cfg.data.official_unpaired_pairs_file if args.order == "unpaired" else cfg.data.official_paired_pairs_file
    )
    ds = VitonHDDataset(
        cfg.data.root, phase=args.phase, order=args.order, size=tuple(cfg.model.image_hw),
        pairs_file=pairs, require_cloth_mask=True, require_parse=False,
    )
    if args.num_samples:
        ds = torch.utils.data.Subset(ds, range(min(args.num_samples, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else float(cfg.eval.get("cfg_scale", 1.0))
    cfg_interval = tuple(args.cfg_interval or cfg.eval.get("cfg_interval", [0.0, 1.0]))
    open_px = args.mask_open_px if args.mask_open_px is not None else int(cfg.eval.mask_open_px)
    print(f"step {step}: {len(ds)} samples, {args.sampler} {args.nfe} NFE -> {out}")
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = prepare_inputs(vae, batch, model, device, mask_open_px=open_px)
            noise = torch.randn(inputs["known"].shape, device=device, generator=gen)
            lat = generate(model, inputs, nfe=args.nfe, sampler=args.sampler, p=args.p,
                           n_inner=args.n_inner, cfg_scale=cfg_scale, cfg_interval=cfg_interval, noise=noise)
        rgb = decode_latents(vae, lat.float()).float()
        rgb = composite(rgb, inputs["agnostic_rgb"], inputs["pixel_mask"])
        for img, name in zip(rgb, batch["person_name"]):
            save_image((img + 1) / 2, out / f"{Path(name).stem}.png")


if __name__ == "__main__":
    main()
