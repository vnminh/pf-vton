"""Evaluate PFI on paired dev cases and profile denoising time without retraining.

Final metrics use equal per-image weighting, RGB in [-1, 1], and the same noise
for every sampler. Teacher profiles use ground-truth interpolations; rollout
profiles use actual generated states. Endpoint error is meaningful in both;
the original straight-path velocity is supervised only on teacher states.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.utils import save_image

from vton_ext.pfi_sample import composite, generate, load_for_inference, prepare_inputs, shift_time
from vton_ext.pairs import split_manifest, validate_resume_pairs
from vton_ext.pfi_train import load_cfg, make_dataset
from vton_ext.utils import expand_patch_values
from vton_ext.vae import decode_latents, encode_images, load_sd_vae


def per_image_mean(values, mask):
    weights = mask.expand_as(values)
    den = weights.flatten(1).sum(1)
    result = (values * weights).flatten(1).sum(1) / den.clamp_min(1)
    return torch.where(den > 0, result, torch.full_like(result, float("nan")))


def highpass(rgb, kernel=5):
    return rgb - F.avg_pool2d(F.pad(rgb, (kernel // 2,) * 4, mode="reflect"), kernel, 1)


def contrast_mask(target, clothing):
    """A target-derived contrast proxy, NOT a semantic text/logo detector."""
    interior = 1 - F.max_pool2d(1 - clothing, 5, 1, 2)
    masks = []
    for rgb, valid in zip(target, interior):
        selected = valid[0] > 0.5
        base = rgb[:, selected].median(1).values if selected.any() else rgb.new_zeros(3)
        distance = (rgb - base[:, None, None]).square().mean(0, keepdim=True).sqrt()
        masks.append((distance > 0.16).float() * valid)
    return torch.stack(masks)


def rgb_metrics(pred, target, masks, include_ssim=False):
    err = (pred - target).abs()
    hp_err = (highpass(pred) - highpass(target)).abs()
    result = {}
    for name, mask in masks.items():
        result[f"{name}_l1"] = per_image_mean(err, mask)
        result[f"{name}_highpass_l1"] = per_image_mean(hp_err, mask)
    if include_ssim:
        _, ssim_map = structural_similarity_index_measure(
            (pred + 1) / 2, (target + 1) / 2, data_range=1.0,
            reduction="none", return_full_image=True,
        )
        result["image_ssim"] = ssim_map.flatten(1).mean(1)
        result["cloth_ssim"] = per_image_mean(ssim_map, masks["cloth"])
    return result


def as_float(value):
    result = float(value)
    return result if np.isfinite(result) else None


def profile_time(call, nfe, time_shift):
    """Actual synchronous token time, including the inference grid shift."""
    return round(float(shift_time(call / nfe, time_shift)), 6)


class ObservedModel:
    """Transparent inference wrapper; its observer cannot change sampler output."""

    def __init__(self, model, observer):
        self.model, self.observer, self.calls = model, observer, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, x, t, cond, edit, **kwargs):
        out = self.model(x, t, cond, edit, **kwargs)
        self.observer(self.calls, x, t, out[0], out[1], kwargs["garment_kv"])
        self.calls += 1
        return out


def summarize(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    excluded = set(group_keys) | {"index", "person_name", "call", "step"}
    result = []
    for group, parts in sorted(groups.items()):
        record = dict(zip(group_keys, group))
        record["num_cases"] = len(parts)
        metrics = {}
        for key in parts[0]:
            if key in excluded:
                continue
            values = [p[key] for p in parts if isinstance(p.get(key), (int, float)) and np.isfinite(p[key])]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)), "count": len(values),
                    "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                }
        record["metrics"] = metrics
        result.append(record)
    return result


def plot_profiles(summary, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("teacher_velocity_mse", "Teacher velocity MSE"),
        ("latent_cloth_endpoint_l1", "Latent clothing endpoint L1"),
        ("cloth_l1", "Decoded clothing endpoint L1"),
        ("cloth_highpass_l1", "Decoded clothing high-pass error"),
        ("contrast_l1", "Contrast-region endpoint L1 (proxy)"),
        ("uncertainty_spearman", "Uncertainty vs endpoint-error rank"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    series = sorted({(r["sampler"], r["state"]) for r in summary})
    colors = {"euler8": "#1673b1", "euler50": "#d35400"}
    for ax, (metric, label) in zip(axes.flat, specs):
        for sampler, state in series:
            points = sorted(
                [r for r in summary if r["sampler"] == sampler and r["state"] == state and metric in r["metrics"]],
                key=lambda r: r["time"],
            )
            if not points:
                continue
            x = np.array([p["time"] for p in points])
            y = np.array([p["metrics"][metric]["mean"] for p in points])
            sem = np.array([p["metrics"][metric]["sem"] for p in points])
            color = colors.get(sampler)
            line, = ax.plot(x, y, "--" if state == "teacher" else "-", label=f"{sampler} {state}", color=color)
            ax.fill_between(x, y - sem, y + sem, color=line.get_color(), alpha=0.10)
        ax.set(title=label, xlabel="Time (0 = noise; 1 = clean)", xlim=(0, 1))
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle(title + "\nEqual per-image means; bands = ±1 SEM; teacher endpoint error shrinks with (1−t)")
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


@torch.no_grad()
def run(args):
    cfg = load_cfg(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError(f"A completed audit already exists: {output}")
    device = torch.device(args.device)
    model, step = load_for_inference(cfg, args.checkpoint, device)
    time_shift = args.time_shift if args.time_shift is not None else float(
        cfg.eval.get("time_shift", getattr(model, "time_shift", 1.0)))
    seam_px = float(cfg.eval.get("seam_px", 0.0))
    feather_px = float(cfg.eval.get("feather_px", 0.0))
    pairs = split_manifest(Path(cfg.data.pairs_file), Path(cfg.data.test_pairs_file))
    checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    validate_resume_pairs(checkpoint_metadata.get("data_pairs"), pairs)
    del checkpoint_metadata
    vae = load_sd_vae(cfg.weights.vae, device)
    dataset = make_dataset(cfg, cfg.data.test_pairs_file, augment=False)
    indices = list(range(len(dataset))) if args.indices is None else args.indices
    if args.num_samples:
        indices = indices[:args.num_samples]
    preview_set = set(int(i) for i in cfg.eval.indices)
    if args.profile_count < 0:
        profile_indices = []
    elif args.profile_count == 0 or args.profile_count >= len(indices):
        profile_indices = list(indices)
    else:
        featured = [i for i in indices if i in preview_set]
        if len(featured) > args.profile_count:
            featured = featured[:args.profile_count]
        others = [i for i in indices if i not in featured]
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(others, size=args.profile_count - len(featured), replace=False).tolist()
        profile_indices = featured + sorted(chosen)
        # Keep profiled batches together; per-case noise and paired statistics
        # stay unchanged, while the extra teacher/decoder work runs on only a
        # representative 64-case subset rather than every long sample.
        profile_set = set(profile_indices)
        indices = profile_indices + [i for i in indices if i not in profile_set]
    profile_set = set(profile_indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "step": step,
        "config": args.config, "seed": args.seed, "indices": indices,
        "profile_indices": profile_indices,
        "samplers": args.samplers,
        "profile_samplers": [f"euler{n}" for n in (8, 50) if f"euler:{n}" in args.samplers],
        "metric_weighting": "equal per image; empty regions excluded with explicit counts",
        "rgb_range": [-1, 1], "cfg_scale": 1.0,
        "time_shift": time_shift, "seam_px": seam_px, "feather_px": feather_px,
        "data_pairs": pairs, "torch_version": str(torch.__version__),
        "notes": [
            "contrast is a color-deviation proxy, not text recognition",
            "teacher states contain ground-truth signal; endpoint error shrinks mechanically with remaining time",
            "rollout velocity is not compared to the original straight-path velocity target",
            "profile observer and teacher/decoder work are excluded from denoiser NFE",
            "one fixed noise realization per dev case; not a multi-seed statistical estimate",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    final_rows, profile_rows = [], []
    offset, started = 0, time.monotonic()
    with (output / "final_metrics.jsonl").open("w") as final_file, (output / "time_profiles.jsonl").open("w") as time_file:
        for batch in loader:
            b = len(batch["person_name"])
            case_indices = indices[offset:offset + b]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                inputs = prepare_inputs(vae, batch, model, device, mask_open_px=int(cfg.eval.mask_open_px))
            target = batch["image"].to(device).float()
            truth = encode_images(vae, target).float()
            cloth = batch["clothing_mask"].to(device).float()
            masks = {"edit": inputs["pixel_mask"], "cloth": cloth,
                     "known": 1 - inputs["pixel_mask"],
                     "contrast": contrast_mask(target, cloth) * inputs["pixel_mask"]}
            latent_cloth = F.adaptive_avg_pool2d(cloth, model.latent_hw) * inputs["edit_pixels"]
            noise = torch.stack([
                torch.randn(inputs["known"].shape[1:], device=device,
                            generator=torch.Generator(device=device).manual_seed(args.seed + i))
                for i in case_indices
            ])
            originals = {"garment": batch["garment"].to(device), "target": target,
                         "vae": decode_latents(vae, truth)}
            outputs = {}

            def append_rows(metrics, common, destination, handle, selected_indices=None):
                for j, index in enumerate(case_indices):
                    if selected_indices is not None and index not in selected_indices:
                        continue
                    row = {"index": index, "person_name": batch["person_name"][j], **common,
                           **{k: as_float(v[j]) for k, v in metrics.items()}}
                    destination.append(row)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()

            for spec in args.samplers:
                sampler, nfe_text = spec.split(":")
                nfe = int(nfe_text)
                key = f"{sampler}{nfe}"
                profile = sampler == "euler" and nfe in (8, 50) and bool(profile_set.intersection(case_indices))
                selected_calls = set(range(nfe)) if nfe == 8 else set(range(0, 41, 5)) | {42, 44, 46, 48, 49}

                def observer(call, x, t, v, logvar, garment_kv):
                    if not profile or call not in selected_calls:
                        return
                    tpix = expand_patch_values(t.float(), model.patch_size, model.latent_hw)
                    teacher_x = torch.where(inputs["edit_pixels"].bool(), tpix * truth + (1 - tpix) * noise, inputs["known"])
                    tv, tlv = model(teacher_x, t, inputs["cond"], inputs["edit_tokens"],
                                    garment_kv=garment_kv, return_uncertainty=True)
                    # This observer profiles synchronous Euler only. Use its
                    # exact grid position to avoid splitting groups by float
                    # reduction noise for different mask sizes/batch partitions.
                    mean_time = profile_time(call, nfe, time_shift)
                    for state, sx, sv, slv in [("rollout", x, v, logvar), ("teacher", teacher_x, tv, tlv)]:
                        endpoint = torch.where(inputs["edit_pixels"].bool(), sx.float() + (1 - tpix) * sv.float(), inputs["known"])
                        with torch.autocast(device.type, enabled=False):
                            rgb = composite(decode_latents(vae, endpoint.float()), inputs["agnostic_rgb"], inputs["pixel_mask"],
                                            seam_px=seam_px, feather_px=feather_px)
                            measures = rgb_metrics(rgb, target, masks)
                        measures["latent_cloth_endpoint_l1"] = per_image_mean((endpoint - truth).abs(), latent_cloth)
                        if state == "teacher":
                            measures["teacher_velocity_mse"] = per_image_mean((sv.float() - (truth - noise)).square(), latent_cloth)
                        err = F.avg_pool2d((endpoint - truth).square().mean(1, keepdim=True), model.patch_size).flatten(1)
                        uq = F.avg_pool2d(slv.float().exp(), model.patch_size).flatten(1)
                        correlations = []
                        for j in range(b):
                            selected = inputs["edit_tokens"][j]
                            u, e = uq[j, selected].cpu().numpy(), err[j, selected].cpu().numpy()
                            corr = spearmanr(u, e).statistic if len(u) > 2 and np.ptp(u) > 0 and np.ptp(e) > 0 else np.nan
                            correlations.append(corr)
                        measures["uncertainty_spearman"] = correlations
                        append_rows(measures, {"sampler": key, "state": state, "time": mean_time, "call": call},
                                    profile_rows, time_file, selected_indices=profile_set)

                wrapped = ObservedModel(model, observer) if profile else model
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    latent, stats = generate(wrapped, inputs, nfe=nfe, sampler=sampler,
                                             p=float(cfg.eval.p), n_inner=int(cfg.eval.n_inner), noise=noise,
                                             return_stats=True, time_shift=time_shift)
                assert stats["nfe"] == nfe
                rgb = composite(decode_latents(vae, latent.float()), inputs["agnostic_rgb"], inputs["pixel_mask"],
                                seam_px=seam_px, feather_px=feather_px)
                outputs[key] = rgb
                measures = rgb_metrics(rgb, target, masks, include_ssim=True)
                append_rows(measures, {"sampler": key, "nfe": nfe}, final_rows, final_file)
                if not args.no_images:
                    image_dir = output / "samples" / key
                    image_dir.mkdir(parents=True, exist_ok=True)
                    for j, index in enumerate(case_indices):
                        save_image((rgb[j] + 1) / 2, image_dir / f"{index:03d}_{Path(batch['person_name'][j]).stem}.png")
            vae_measures = rgb_metrics(originals["vae"], target, masks, include_ssim=True)
            append_rows(vae_measures, {"sampler": "vae_reconstruction", "nfe": 0}, final_rows, final_file)
            for j, index in enumerate(case_indices):
                if index in preview_set:
                    preview_dir = output / "previews"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    row = [v[j] for v in originals.values()] + [v[j] for v in outputs.values()]
                    save_image((torch.stack(row) + 1) / 2, preview_dir / f"case{index:03d}.png", nrow=len(row))
            offset += b
            print(json.dumps({"step": step, "completed_cases": offset, "total_cases": len(indices), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    final_summary = summarize(final_rows, ["sampler"])
    time_summary = summarize(profile_rows, ["sampler", "state", "time"])
    (output / "summary.json").write_text(json.dumps({"final": final_summary, "time": time_summary}, indent=2, allow_nan=False) + "\n")
    plot_profiles(time_summary, output / "time-errors.png",
                  f"PFI step {step}: {len(profile_indices)} profiled of {len(indices)} paired dev cases")
    (output / "complete.json").write_text(json.dumps({"step": step, "num_cases": len(indices), "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--time-shift", type=float, help="Inference grid shift; defaults to cfg.eval.time_shift")
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--profile-count", type=int, default=64,
                   help="Number of cases for expensive teacher/rollout time profiles; 0 profiles all, -1 skips")
    p.add_argument("--indices", type=int, nargs="+")
    p.add_argument("--samplers", nargs="+", default=["euler:8", "dual_loop:8", "euler:25", "euler:50"])
    p.add_argument("--no-images", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
