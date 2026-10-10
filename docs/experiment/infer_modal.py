"""VITON-HD evaluation on Modal for a named checkpoint/sampler/seed setting.

``--mode sample`` is a three-image smoke test; ``--mode full`` evaluates all
2032 paired and unpaired cases. Full evaluation records MSE, SSIM, LPIPS,
FID, KID, actual denoiser calls and synchronized sampler time. Outputs are
under /data/experiments/<output_tag>/<mode>/<cfg>/<pair|unpair>/.

Metric references
-----------------
MSE  : mean squared error in [0,1]^3, averaged over H*W*C.
       Matches pixel-L2 used in VITON, HR-VITON, DCI-VTON papers.
SSIM : Wang et al. (2004) IEEE TIP.
       window=11, sigma=1.5, K1=0.01, K2=0.03, data_range=1.0.
       Matches torchmetrics StructuralSimilarityIndexMeasure defaults
       used in HR-VITON / LaDI-VTON / DCI-VTON benchmark tables.
LPIPS: Zhang et al. (2018) CVPR.  AlexNet backbone (net_type='alex').
       torchmetrics LearnedPerceptualImagePatchSimilarity, reduction='mean'.
       Lower is better.  Standard in HR-VITON, DCI-VTON, CatVTON, StableVTON.
FID  : Heusel et al. (2017) NeurIPS.
       Inception-v3 pool3 (2048-d), bicubic-299 resize, via
       torch-fidelity - same implementation as most VTON papers.
       Paired:   real = GT try-on images,   fake = predictions.
       Unpaired: real = person test images,  fake = predictions.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

import modal

# -----------------------------------------------------------------------
# Modal infrastructure
# -----------------------------------------------------------------------

APP_NAME = "pf-vton-eval-v50"
VOLUME_NAME = "VTON-HD"
VOLUME_MOUNT = "/data"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "lightning>=2.5.0",
        "accelerate>=1.10.0",
        "timm>=1.0.17",
        "hydra-core",
        "torch-fidelity",
        "torchdiffeq",
        "torchmetrics[image]>=1.0.0",
        "lpips",
        "numpy",
        "scipy",
        "pandas",
        "pillow>=10.0",
        "einops",
        "jaxtyping",
        "tqdm",
        "omegaconf>=2.3",
        "opencv-python-headless",
        "diffusers[torch]",
        "transformers>=4.56",
        "huggingface_hub>=0.34",
        "git+https://github.com/joh-schb/jutils.git",
    )
    .add_local_dir(
        Path(__file__).resolve().parents[2],
        remote_path="/workspace/patch-forcing-vton",
        ignore=[
            "logs/**", "checkpoints/**", ".git/**",
            "__pycache__/**", "*.pt", "*.ckpt",
            "docs/experiment/paired/**", "docs/experiment/unpaired/**",
            "docs/experiment/*.jpg", "docs/assets/**", "docs/*.pdf",
        ],
    )
)

app = modal.App(name=APP_NAME, image=image)

# -----------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------

NFE = 30
SAMPLER = "look_ahead"
BATCH_SIZE = 20
SEED = 42

DATA_ROOT = VOLUME_MOUNT
CHECKPOINT = f"{VOLUME_MOUNT}/logs/vton-v50-attn-flow/latest.pt"
CONFIG_PATH = "/workspace/patch-forcing-vton/configs/vton_v50_attn_flow.yaml"
UNPAIRED_PAIRS = f"{VOLUME_MOUNT}/test_pairs.txt"


# =======================================================================
# Metric engine
# =======================================================================

class VTONMetrics:
    """Paper-accurate metric computation.

    MSE  - Mean Squared Error in [0,1] pixel space, over H*W*C.
    SSIM - Wang et al. IEEE TIP 2004.  window=11, sigma=1.5,
           K1=0.01, K2=0.03, data_range=1.0.
    LPIPS- Zhang et al. CVPR 2018.  AlexNet backbone (net_type='alex'),
           input normalised to [-1,1].  Lower is better.
    FID  - Heusel et al. NeurIPS 2017 via torch-fidelity
           (Inception-v3 pool3, 2048-d features).
    """

    def __init__(self, device: str = "cuda"):
        import lpips
        import torch
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.device = torch.device(device)
        # Exact Wang et al. 2004 defaults: 11x11 Gaussian, sigma=1.5
        self.ssim_fn = StructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=11,
            sigma=1.5,
            k1=0.01,
            k2=0.03,
            reduction="none",   # per-image; we aggregate manually
        ).to(self.device)
        # Zhang et al. 2018 - AlexNet backbone is the VTON benchmark standard
        # Dùng trực tiếp official lpips (nn.Module thuần túy, không tích lũy state/leak VRAM)
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()

    @staticmethod
    def _to_01(t):
        """Convert [-1,1] -> [0,1] with safety clamp."""
        return ((t + 1.0) / 2.0).clamp(0.0, 1.0)

    def mse_ssim_lpips(self, pred, real) -> Tuple[float, float, float]:
        """MSE, SSIM, and LPIPS for one image pair.  pred/real: (C,H,W) in [-1,1]."""
        import torch
        p01 = self._to_01(pred.to(self.device)).unsqueeze(0)   # (1,C,H,W) in [0,1]
        r01 = self._to_01(real.to(self.device)).unsqueeze(0)
        mse = float(torch.mean((p01 - r01) ** 2).item())
        ssim = float(self.ssim_fn(p01, r01).mean().item())
        # LPIPS expects [-1,1]; clamp for safety
        p11 = pred.to(self.device).clamp(-1.0, 1.0).unsqueeze(0)
        r11 = real.to(self.device).clamp(-1.0, 1.0).unsqueeze(0)
        with torch.no_grad():
            lpips_v = float(self.lpips_fn(p11, r11).mean().item())
        return mse, ssim, lpips_v

    @staticmethod
    def distribution_metrics(real_dir: Path, fake_dir: Path, device: str = "cuda") -> dict:
        """FID and KID on the same image sets and Inception features.

        Uses torch-fidelity with Inception-v3 pool3 (2048-d), bicubic
        resize to 299x299.  Matches the FID protocol used in HR-VITON,
        DCI-VTON, LaDI-VTON, and similar VTON benchmark papers.
        """
        import torch_fidelity
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        n_real = sum(p.is_file() and p.suffix.lower() in image_exts for p in real_dir.iterdir())
        n_fake = sum(p.is_file() and p.suffix.lower() in image_exts for p in fake_dir.iterdir())
        if n_real < 2 or n_fake < 2:
            raise ValueError("FID/KID need at least two images in each directory")
        # The three-image smoke test is too small for a useful KID estimate.
        kid_enabled = min(n_real, n_fake) >= 1000
        out = torch_fidelity.calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            cuda=(device == "cuda"),
            isc=False,
            fid=True,
            kid=kid_enabled,
            kid_subset_size=1000,
            kid_subsets=100,
            rng_seed=42,
            prc=False,
            verbose=False,
        )
        return {
            "fid": float(out["frechet_inception_distance"]),
            "kid": float(out["kernel_inception_distance_mean"]) if kid_enabled else None,
            "kid_subset_std": float(out["kernel_inception_distance_std"]) if kid_enabled else None,
            "kid_subset_size": 1000 if kid_enabled else None,
            "kid_subsets": 100 if kid_enabled else None,
        }


# =======================================================================
# Config / model helpers
# =======================================================================

def _load_cfg(config_path: str, data_root: str):
    """Resolve merged OmegaConf config (base_config chain) and
    override data.root with the volume mount path."""
    import os
    workspace = Path("/workspace/patch-forcing-vton")
    if workspace.is_dir():
        os.chdir(workspace)
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))

    from omegaconf import OmegaConf
    from vton_ext.pfi_train import load_cfg

    cfg = load_cfg(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"root": data_root}}))

    # Resolve checkpoints/sd_ae.ckpt if relative and not present locally
    if hasattr(cfg, "weights") and hasattr(cfg.weights, "vae"):
        vae_p = Path(cfg.weights.vae)
        if not vae_p.exists():
            for cand in [Path(VOLUME_MOUNT) / cfg.weights.vae,
                         Path(VOLUME_MOUNT) / "checkpoints" / vae_p.name]:
                if cand.exists():
                    cfg.weights.vae = str(cand)
                    break
    return cfg


def _load_model_and_vae(cfg, checkpoint_path: str, device):
    from vton_ext.pfi_sample import load_for_inference
    from vton_ext.vae import load_sd_vae

    model, step = load_for_inference(cfg, checkpoint_path, device, use_ema=True)
    vae = load_sd_vae(cfg.weights.vae, device)
    print(f"Loaded checkpoint (step={step})", flush=True)
    return model, vae, step


def _build_dataset(cfg, pairs_file: str, order: str, num_samples: int, seed: int):
    """Build VitonHDDataset; sub-sample randomly if num_samples > 0."""
    import torch
    from vton_ext.data import VitonHDDataset

    ds = VitonHDDataset(
        cfg.data.root,
        phase="test",
        order=order,
        size=tuple(cfg.model.image_hw),
        pairs_file=pairs_file,
        require_cloth_mask=True,
        require_parse=False,
    )
    if num_samples and num_samples < len(ds):
        idx = random.Random(seed).sample(range(len(ds)), num_samples)
        ds = torch.utils.data.Subset(ds, sorted(idx))
    return ds


# =======================================================================
# Inference
# =======================================================================

def _infer_and_save(model, vae, cfg, dataset, pred_dir: Path,
                    cfg_scale: float, nfe: int, device,
                    detail_dir: Path | None = None,
                    metric_engine: VTONMetrics | None = None,
                    runtime: dict | None = None,
                    seed: int = SEED, sampler: str = SAMPLER,
                    context_ratio: float = 1.5) -> tuple[list[float], list[float], list[float], list[dict]]:
    """Run model inference, save predictions as PNGs in pred_dir,
    and side-by-side composite details in detail_dir.
    If metric_engine is provided (paired evaluation), compute MSE, SSIM, LPIPS directly
    from in-memory float tensors without reading back from disk."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from vton_ext.pfi_sample import composite, generate, prepare_inputs
    from vton_ext.vae import decode_latents

    pred_dir.mkdir(parents=True, exist_ok=True)
    if detail_dir is not None:
        detail_dir.mkdir(parents=True, exist_ok=True)

    ecfg = cfg.get("eval", {})
    open_px = int(ecfg.get("mask_open_px", 0))
    cfg_iv = tuple(ecfg.get("cfg_interval", [0.0, 1.0]))
    time_shift = ecfg.get("time_shift", None)
    seam_px = float(ecfg.get("seam_px", 0.0))
    feather_px = float(ecfg.get("feather_px", 0.0))
    p = float(ecfg.get("p", 0.7))          # uncertain-token quantile (look_ahead / dual_loop)
    n_inner = int(ecfg.get("n_inner", 2))  # inner steps per outer step (dual_loop)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    total = len(dataset)
    done = 0

    mse_list, ssim_list, lpips_list, detail_rows = [], [], [], []
    if runtime is not None:
        runtime.update(seconds=0.0, images=0, model_calls=0, denoiser_calls=0)

    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = prepare_inputs(vae, batch, model, device,
                                    mask_open_px=open_px)
            noise = torch.randn(inputs["known"].shape, device=device,
                                generator=gen)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            lat, sample_stats = generate(
                model, inputs,
                nfe=nfe,
                sampler=sampler,
                p=p,
                n_inner=n_inner,
                context_ratio=context_ratio,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_iv,
                noise=noise,
                time_shift=time_shift,
                return_stats=True,
            )
            torch.cuda.synchronize(device)
            if runtime is not None:
                batch_size = len(batch["person_name"])
                runtime["seconds"] += time.perf_counter() - started
                runtime["images"] += batch_size
                runtime["model_calls"] += sample_stats["nfe"] * batch_size
                runtime["denoiser_calls"] += sample_stats["denoiser_calls"] * batch_size
        rgb = decode_latents(vae, lat.float()).float()
        rgb = composite(rgb, inputs["agnostic_rgb"], inputs["pixel_mask"],
                        seam_px=seam_px, feather_px=feather_px)

        for i, (img, name) in enumerate(zip(rgb, batch["person_name"])):
            stem = Path(name).stem
            # 1. Save prediction to disk for inspection
            save_image((img + 1.0) / 2.0,
                       pred_dir / f"{stem}.png")

            # 2. Save detail strip if requested
            if detail_dir is not None:
                # 4 images concatenated horizontally: cloth, agnostic-v3.2, image, pred-image
                cloth_t = batch["garment"][i].to(img.device)
                agnostic_t = batch["agnostic"][i].to(img.device)
                image_t = batch["image"][i].to(img.device)
                four_strip = torch.cat([cloth_t, agnostic_t, image_t, img], dim=-1)
                save_image(((four_strip + 1.0) / 2.0).clamp(0.0, 1.0),
                           detail_dir / f"{stem}.png")

            # 3. Compute metrics directly from RAM/VRAM tensor (lossless, no disk reload)
            if metric_engine is not None:
                real_t = batch["image"][i].to(device)  # (C,H,W) in [-1,1]
                assert img.shape == real_t.shape, (
                    f"Shape mismatch for {stem}: pred {img.shape} vs real {real_t.shape}"
                )
                mse_v, ssim_v, lpips_v = metric_engine.mse_ssim_lpips(img, real_t)
                mse_list.append(mse_v)
                ssim_list.append(ssim_v)
                lpips_list.append(lpips_v)
                detail_rows.append({
                    "person": name,
                    "garment": batch["garment_name"][i],
                    "mse": mse_v,
                    "ssim": ssim_v,
                    "lpips": lpips_v,
                })

            done += 1
        print(f"    progress: {done}/{total}", flush=True)

    return mse_list, ssim_list, lpips_list, detail_rows


# =======================================================================
# Ground-truth export for FID
# =======================================================================

def _export_real_for_fid(dataset, real_dir: Path) -> None:
    """Export GT person images as PNG for FID (idempotent)."""
    from torchvision.utils import save_image

    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))
    expected = {Path(base.rows[i][0]).stem + ".png" for i in indices}
    if real_dir.exists() and {p.name for p in real_dir.glob("*.png")} == expected:
        return
    if real_dir.exists() and any(real_dir.iterdir()):
        raise ValueError(f"Stale/incomplete real images in {real_dir}; choose a new output tag")
    real_dir.mkdir(parents=True, exist_ok=True)

    def _item(i):
        if hasattr(dataset, "dataset"):   # Subset
            return dataset.dataset[dataset.indices[i]]
        return dataset[i]

    for i in range(len(dataset)):
        item = _item(i)
        stem = Path(item["person_name"]).stem
        save_image((item["image"] + 1.0) / 2.0,
                   real_dir / f"{stem}.png")


def _verify_prediction_set(dataset, pred_dir: Path) -> None:
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))
    expected = {Path(base.rows[i][0]).stem + ".png" for i in indices}
    actual = {p.name for p in pred_dir.glob("*.png")}
    if actual != expected:
        raise ValueError(f"Prediction set mismatch in {pred_dir}: expected {len(expected)}, found {len(actual)}. Choose a new output tag.")


def _keep_review_predictions(pred_dir: Path, keep: int = 16) -> int:
    """After distribution metrics, retain a small deterministic visual audit set."""
    files = sorted(pred_dir.glob("*.png"))
    chosen = set(files[:keep])
    for path in files:
        if path not in chosen:
            path.unlink()
    return min(len(files), keep)


def _real_distribution_dir(dataset, metrics_dir: Path, full: bool) -> Path:
    if full:
        base = dataset.dataset if hasattr(dataset, "dataset") else dataset
        names = {p.name for p in base.image_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
        expected = {row[0] for row in base.rows}
        if names != expected or len(dataset) != len(base.rows):
            raise ValueError("Full real-image directory does not match the test pair list")
        return base.image_dir
    real_dir = metrics_dir / "real_gt"
    _export_real_for_fid(dataset, real_dir)
    return real_dir


# =======================================================================
# Paired evaluation  (MSE + SSIM + LPIPS + FID)
# =======================================================================

def _eval_paired(model, vae, cfg, dataset, out_dir: Path,
                 cfg_scale: float, nfe: int, device,
                 seed: int = SEED, sampler: str = SAMPLER, context_ratio: float = 1.5,
                 full: bool = True) -> dict:
    pred_dir = out_dir / "pred-image"
    detail_dir = None if full else out_dir / "pred-image-detail"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- generate and compute in-memory MSE, SSIM & LPIPS ----
    print(f"  [pair] Generating {len(dataset)} images & computing in-memory metrics ...", flush=True)
    engine = VTONMetrics(device=str(device))
    runtime = {}
    mse_list, ssim_list, lpips_list, detail_rows = _infer_and_save(
        model, vae, cfg, dataset, pred_dir,
        cfg_scale=cfg_scale, nfe=nfe, device=device,
        detail_dir=detail_dir, metric_engine=engine, runtime=runtime,
        seed=seed, sampler=sampler, context_ratio=context_ratio,
    )

    # ---- FID (distribution-level; reads saved images from disk) ----
    real_dir = _real_distribution_dir(dataset, metrics_dir, full)
    _verify_prediction_set(dataset, pred_dir)
    print(f"  [pair] Computing FID ...", flush=True)
    dist = VTONMetrics.distribution_metrics(real_dir, pred_dir, device=str(device))

    n = len(mse_list)
    summary = {
        "num_samples": n,
        "mse_mean": sum(mse_list) / n if n else float("nan"),
        "ssim_mean": sum(ssim_list) / n if n else float("nan"),
        "lpips_mean": sum(lpips_list) / n if n else float("nan"),
        **dist,
        "sample_seconds_per_image": runtime["seconds"] / runtime["images"],
        "model_calls_per_image": runtime["model_calls"] / runtime["images"],
        "denoiser_calls_per_image": runtime["denoiser_calls"] / runtime["images"],
    }
    summary["predictions_retained"] = _keep_review_predictions(pred_dir) if full else len(dataset)

    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(metrics_dir / "detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["person", "garment", "mse", "ssim", "lpips"])
        w.writeheader()
        w.writerows(detail_rows)

    print(
        f"  [pair] MSE={summary['mse_mean']:.6f}  "
        f"SSIM={summary['ssim_mean']:.4f}  "
        f"LPIPS={summary['lpips_mean']:.4f}  FID={dist['fid']:.4f}  KID={dist['kid']}",
        flush=True,
    )
    return summary


# =======================================================================
# Unpaired evaluation  (FID only)
# =======================================================================

def _eval_unpaired(model, vae, cfg, dataset, out_dir: Path,
                   cfg_scale: float, nfe: int, device,
                   seed: int = SEED, sampler: str = SAMPLER, context_ratio: float = 1.5,
                   full: bool = True) -> dict:
    pred_dir = out_dir / "pred-image"
    detail_dir = None if full else out_dir / "pred-image-detail"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [unpair] Generating {len(dataset)} images ...", flush=True)
    runtime = {}
    _infer_and_save(model, vae, cfg, dataset, pred_dir,
                    cfg_scale=cfg_scale, nfe=nfe, device=device,
                    detail_dir=detail_dir, runtime=runtime,
                    seed=seed, sampler=sampler, context_ratio=context_ratio)

    real_dir = _real_distribution_dir(dataset, metrics_dir, full)
    _verify_prediction_set(dataset, pred_dir)
    print(f"  [unpair] Computing FID ...", flush=True)
    dist = VTONMetrics.distribution_metrics(real_dir, pred_dir, device=str(device))

    summary = {"num_samples": len(dataset), **dist,
               "sample_seconds_per_image": runtime["seconds"] / runtime["images"],
               "model_calls_per_image": runtime["model_calls"] / runtime["images"],
               "denoiser_calls_per_image": runtime["denoiser_calls"] / runtime["images"]}
    summary["predictions_retained"] = _keep_review_predictions(pred_dir) if full else len(dataset)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [unpair] FID={dist['fid']:.4f}  KID={dist['kid']}", flush=True)
    return summary


# =======================================================================
# Paired pairs-file resolver
# =======================================================================

def _paired_pairs_file(cfg, data_root: str) -> str:
    """Return path to a paired test pairs file.

    Search order:
      1. cfg.data.official_paired_pairs_file (from config yaml)
      2. VTON-HD/test_pairs_paired.txt
      3. Rows in test_pairs.txt where person_stem == garment_stem
      4. Intersection of test/image and test/cloth directories
    """
    root = Path(data_root)

    explicit = cfg.data.get("official_paired_pairs_file")
    if explicit:
        p = (
            Path(explicit)
            if Path(explicit).is_absolute()
            else root / explicit
        )
        if p.exists():
            return str(p)

    for cand in [
        root / "test_pairs_paired.txt",
        root / "test" / "test_pairs_paired.txt",
    ]:
        if cand.exists():
            return str(cand)

    up = Path(UNPAIRED_PAIRS)
    if up.exists():
        lines = [
            ln
            for ln in up.read_text().splitlines()
            if len(ln.split()) == 2
            and Path(ln.split()[0]).stem == Path(ln.split()[1]).stem
        ]
        if lines:
            tmp = root / "test_pairs_paired_auto.txt"
            tmp.write_text("\n".join(lines) + "\n")
            return str(tmp)

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    img_dir = root / "test" / "image"
    clo_dir = root / "test" / "cloth"
    if img_dir.is_dir() and clo_dir.is_dir():
        imgs = {
            p.stem: p.name
            for p in img_dir.iterdir()
            if p.suffix.lower() in EXTS
        }
        clos = {
            p.stem: p.name
            for p in clo_dir.iterdir()
            if p.suffix.lower() in EXTS
        }
        common = sorted(set(imgs) & set(clos))
        if common:
            tmp = root / "test_pairs_paired_auto.txt"
            tmp.write_text(
                "\n".join(f"{imgs[s]} {clos[s]}" for s in common) + "\n"
            )
            return str(tmp)

    raise FileNotFoundError(
        "Cannot find or construct a paired test pairs file. "
        "Expected VTON-HD/test_pairs_paired.txt."
    )


# =======================================================================
# Result table printer
# =======================================================================

def _print_table(all_results: dict) -> None:
    hdr = f"{'CFG':<8}  {'Mode':<8}  {'MSE':>12}  {'SSIM':>8}  {'LPIPS':>8}  {'FID':>8}  {'KIDx100':>8}  {'sec/img':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label in sorted(all_results):
        for mode, s in all_results[label].items():
            mse_s = (
                f"{s['mse_mean']:.6f}" if "mse_mean" in s else "     -    "
            )
            ssim_s = (
                f"{s['ssim_mean']:.4f}" if "ssim_mean" in s else "    -   "
            )
            lpips_s = (
                f"{s['lpips_mean']:.4f}" if "lpips_mean" in s else "    -   "
            )
            fid_s = f"{s['fid']:.4f}"
            kid_s = f"{100*s['kid']:.4f}" if s['kid'] is not None else "-"
            sec_s = f"{s['sample_seconds_per_image']:.3f}"
            print(
                f"{label:<8}  {mode:<8}  {mse_s:>12}  {ssim_s:>8}  {lpips_s:>8}  {fid_s:>8}  {kid_s:>8}  {sec_s:>8}"
            )


# =======================================================================
# Background auto-commit thread
# =======================================================================

def _auto_commit(stop: threading.Event, interval: int = 120) -> None:
    while not stop.wait(interval):
        try:
            volume.commit()
            print("[auto-commit] volume synced.", flush=True)
        except Exception as e:
            print(f"[auto-commit] warning: {e}", flush=True)


# =======================================================================
# Core evaluation runner  (runs inside Modal container)
# =======================================================================

def _run_evaluation(mode: str, nfe: int, cfg_scales: tuple[float, ...], seed: int,
                    sampler: str, context_ratio: float, checkpoint: str,
                    config_path: str, output_tag: str) -> None:
    """Evaluate a named, reproducible inference setting on both test pairings."""
    import os
    workspace = Path("/workspace/patch-forcing-vton")
    if workspace.is_dir():
        os.chdir(workspace)
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))
        ckpt_dir = workspace / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        vol_ckpt = Path(VOLUME_MOUNT) / "checkpoints"
        if vol_ckpt.is_dir():
            for item in vol_ckpt.iterdir():
                dst = ckpt_dir / item.name
                if dst.is_symlink() or dst.exists():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                try:
                    dst.symlink_to(item)
                    print(f"Linked checkpoint: {dst.name} -> {item}", flush=True)
                except Exception as e:
                    print(f"Warning linking {dst.name}: {e}", flush=True)
    else:
        sys.path.insert(0, "/workspace/patch-forcing-vton")

    import torch

    device = torch.device("cuda")
    num_samples = 3 if mode == "sample" else 0
    output_root = Path(VOLUME_MOUNT) / "experiments" / output_tag / mode

    sep = "=" * 65
    print(sep)
    print(
        f"VTON-HD Evaluation | mode={mode} | "
        f"NFE={nfe} | sampler={sampler} | seed={seed}"
    )
    print(f"Checkpoint : {checkpoint}")
    print(f"Output     : {output_root}")
    print(sep, flush=True)

    cfg = _load_cfg(config_path, DATA_ROOT)
    model, vae, step = _load_model_and_vae(cfg, checkpoint, device)
    checkpoint_stat = Path(checkpoint).stat()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint": checkpoint, "checkpoint_step": step, "config": config_path,
                "checkpoint_size_bytes": checkpoint_stat.st_size,
                "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                "seed": seed, "nfe": nfe, "sampler": sampler,
                "context_ratio": context_ratio, "cfg_scales": list(cfg_scales), "mode": mode}
    metadata_path = output_root / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError(f"Output tag {output_tag} was used for a different setting")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    all_results: dict = {}

    for cfg_scale in cfg_scales:
        label = f"cfg{cfg_scale:g}"
        print(f"\n{sep}")
        print(f"CFG scale = {cfg_scale}  ({label})")
        print(sep, flush=True)

        cfg_out = output_root / label

        # Paired
        print("\n--- PAIRED ---", flush=True)
        pair_pairs = _paired_pairs_file(cfg, DATA_ROOT)
        ds_pair = _build_dataset(cfg, pair_pairs, "paired", num_samples, seed)
        pair_sum = _eval_paired(
            model, vae, cfg, ds_pair,
            cfg_out / "pair", cfg_scale, nfe, device,
            seed=seed, sampler=sampler, context_ratio=context_ratio,
            full=(mode == "full"),
        )

        # Unpaired
        print("\n--- UNPAIRED ---", flush=True)
        ds_up = _build_dataset(
            cfg, UNPAIRED_PAIRS, "unpaired", num_samples, seed
        )
        up_sum = _eval_unpaired(
            model, vae, cfg, ds_up,
            cfg_out / "unpair", cfg_scale, nfe, device,
            seed=seed, sampler=sampler, context_ratio=context_ratio,
            full=(mode == "full"),
        )

        all_results[label] = {"pair": pair_sum, "unpair": up_sum}

    print(f"\n{sep}")
    print("EVALUATION COMPLETE")
    print(sep)
    _print_table(all_results)
    print(f"\nEvaluation finished.", flush=True)


# =======================================================================
# Modal function
# =======================================================================

@app.function(
    gpu="H100",
    volumes={VOLUME_MOUNT: volume},
    timeout= 24 * 3600,
)
def run_eval_modal(mode: str = "full", nfe: int = NFE, cfg_scales: tuple[float, ...] = (2.0,),
                   seed: int = SEED, sampler: str = SAMPLER, context_ratio: float = 1.5,
                   checkpoint: str = CHECKPOINT, config_path: str = CONFIG_PATH,
                   output_tag: str = "v50-lookahead30-cfg2-seed42"):
    """Evaluate VTON-HD on Modal with the requested reproducible setting.

    Args:
        mode: "sample" - 3 random images (quick sanity check)
              "full"   - entire VITON-HD test set
    """
    stop = threading.Event()
    t = threading.Thread(
        target=_auto_commit, args=(stop, 120), daemon=True
    )
    t.start()
    try:
        _run_evaluation(mode, nfe, cfg_scales, seed, sampler, context_ratio,
                        checkpoint, config_path, output_tag)
    finally:
        stop.set()
        t.join(timeout=5)
        volume.commit()
        print("Final volume commit done.", flush=True)


# =======================================================================
# Local entrypoint
# =======================================================================

@app.local_entrypoint()
def main(mode: str = "full", nfe: int = NFE, cfg_scales: str = "2.0",
         seed: int = SEED, sampler: str = SAMPLER, context_ratio: float = 1.5,
         checkpoint: str = CHECKPOINT, config_path: str = CONFIG_PATH,
         output_tag: str = "v50-lookahead30-cfg2-seed42"):
    """
    Submit VTON-HD evaluation job to Modal (50 NFE Lookahead).

    Usage:
      modal run infer_modal.py                # full test set
      modal run infer_modal.py --mode sample  # 3-sample quick check
    """
    if mode not in ("sample", "full"):
        print(f"ERROR: --mode must be 'sample' or 'full', got '{mode}'")
        sys.exit(1)
    if sampler not in ("euler", "dual_loop", "look_ahead"):
        raise ValueError(f"Invalid sampler: {sampler}")
    if nfe < 1 or (sampler == "look_ahead" and nfe % 2):
        raise ValueError("NFE must be positive and even for look_ahead")
    if context_ratio <= 0 or not output_tag or "/" in output_tag or ".." in output_tag:
        raise ValueError("context_ratio must be positive and output_tag a single directory name")
    scales = tuple(float(s) for s in cfg_scales.split(","))
    if not scales or any(s <= 0 for s in scales):
        raise ValueError("cfg_scales must be positive comma-separated numbers")

    sep = "=" * 70
    print()
    print(sep)
    print(f"VTON-HD EVALUATION  |  mode={mode}")
    print(sep)
    print()

    call = run_eval_modal.spawn(mode, nfe, scales, seed, sampler, context_ratio,
                                checkpoint, config_path, output_tag)

    print(f"Job submitted!  Function Call ID: {call.object_id}")
    print()
    print(f"  mode       : {mode}")
    print(f"  CFG scales : {scales}")
    print(f"  NFE        : {nfe}  sampler: {sampler}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  output     : {VOLUME_MOUNT}/experiments/{output_tag}/{mode}")
    print()
    print("Monitor at: https://modal.com/apps")
    print(sep)
    print()
