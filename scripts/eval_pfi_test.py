"""Đánh giá một checkpoint PFI-VTON trên tập test VITON-HD, chạy trực tiếp trên server.

Cùng quy trình với docs/experiment/infer_modal.py (bản Modal), nhưng giữ lại
**mọi** ảnh dự đoán và ghi thước đo cho **từng mẫu**:

    PYTHONPATH=. python scripts/eval_pfi_test.py \
        --checkpoint logs/vton-v50-attn-flow/latest.pt \
        --config configs/vton_v50_attn_flow.yaml \
        --data-root ../high-resolution-viton-zalando-dataset \
        --output-dir outputs/eval/v50-la50-cfg1.5-seed42 \
        --sampler look_ahead --nfe 50 --cfg-scales 1.5 --seed 42

Đầu ra cho mỗi CFG và mỗi thiết lập (pair = paired, unpair = unpaired):

    <output-dir>/
      metadata.json                    cấu hình đầy đủ của lần chạy (để kiểm tra khi chạy tiếp)
      summary.json                     thước đo trung bình + FID/KID của mọi CFG và thiết lập
      cfg<w>/<pair|unpair>/
        pred/<person>.png              ảnh dự đoán 1024x768 (đã ghép với ảnh agnostic)
        strip/<person>.jpg             (tùy chọn --save-strips) áo | agnostic | ảnh người | dự đoán
        per_sample.csv                 một dòng cho mỗi mẫu
        summary.json                   trung bình, FID, KID, thời gian

Thước đo:
* Paired, từng mẫu: MSE trên ảnh [0,1]; MSE chỉ trong vùng mặt nạ edit; SSIM
  (cửa sổ Gauss 11, sigma 1.5); LPIPS AlexNet (thư viện lpips chính thức).
* Unpaired không có ground truth: từng mẫu chỉ ghi tên ảnh và thời gian.
* FID và KID (torch-fidelity, cùng đặc trưng Inception-v3 pool3; KID 100 tập
  con x 1000 ảnh) cho cả tập. Ảnh thật là test/image: ground truth ở paired,
  ảnh người thật ở unpaired. KID cần ít nhất 1000 ảnh, nên tắt khi --limit < 1000.
* Thời gian: giây GPU đã đồng bộ cho mỗi ảnh (chỉ phần sampler, bỏ batch đầu vì
  khởi động CUDA), số bước sampler và số lượt gọi denoiser mỗi ảnh (CFG làm
  gấp đôi số lượt gọi).

Nhiễu ban đầu của mỗi mẫu được sinh từ (seed, chỉ số mẫu), không phụ thuộc
batch size. Nhờ vậy khi chạy lại cùng --output-dir, các mẫu đã có ảnh và dòng
CSV được bỏ qua mà kết quả không đổi (dùng --overwrite để chạy lại từ đầu).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chạy được cả khi không đặt PYTHONPATH

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from vton_ext.data import VitonHDDataset
from vton_ext.pfi_sample import SAMPLERS, composite, generate, load_for_inference, prepare_inputs
from vton_ext.pfi_train import load_cfg
from vton_ext.vae import decode_latents, load_sd_vae

PAIR_FIELDS = ["index", "person", "garment", "mse", "mse_mask", "ssim", "lpips", "sample_seconds"]
UNPAIR_FIELDS = ["index", "person", "garment", "sample_seconds"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- metrics

class PairMetrics:
    """MSE, MSE trong mặt nạ, SSIM, LPIPS cho từng mẫu (ảnh vào trong [-1, 1])."""

    def __init__(self, device):
        import lpips
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03, reduction="none").to(device)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, real, mask):
        """pred/real (B,3,H,W) trong [-1,1]; mask (B,1,H,W) trong {0,1}. Trả về các tensor (B,)."""
        p01, r01 = ((pred + 1) / 2).clamp(0, 1), ((real + 1) / 2).clamp(0, 1)
        sq = (p01 - r01).square()
        mse = sq.mean(dim=(1, 2, 3))
        m = mask.expand_as(sq)
        mse_mask = (sq * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
        ssim = self.ssim(p01, r01)
        lp = self.lpips(pred.clamp(-1, 1), real.clamp(-1, 1)).flatten()
        return mse, mse_mask, ssim, lp


def distribution_metrics(real_dir: Path, fake_dir: Path, kid: bool) -> dict:
    """FID và KID trên cùng đặc trưng Inception (torch-fidelity)."""
    import torch_fidelity

    out = torch_fidelity.calculate_metrics(
        input1=str(real_dir), input2=str(fake_dir), cuda=torch.cuda.is_available(), isc=False,
        fid=True, kid=kid, kid_subsets=100, kid_subset_size=1000, rng_seed=42, prc=False, verbose=False)
    return {
        "fid": float(out["frechet_inception_distance"]),
        "kid": float(out["kernel_inception_distance_mean"]) if kid else None,
        "kid_subset_std": float(out["kernel_inception_distance_std"]) if kid else None,
    }


# ------------------------------------------------------------------- data

def pairs_file(cfg, root: Path, setting: str) -> Path:
    key = "official_paired_pairs_file" if setting == "pair" else "official_unpaired_pairs_file"
    default = "test_pairs_paired.txt" if setting == "pair" else "test_pairs.txt"
    cand = [Path(str(cfg.data.get(key)))] if cfg.data.get(key) else []
    cand += [root / default, root / "test" / default]
    for c in cand:
        c = c if c.is_absolute() else root / c
        if c.exists():
            return c
    raise FileNotFoundError(f"không tìm thấy danh sách cặp {setting} ({default}) trong {root}")


def complete_paired_list(listed: Path, root: Path, out_root: Path) -> tuple[Path, dict]:
    """Paired = mỗi người test mặc lại đúng áo của mình (2032 cặp ở VITON-HD).

    Một số bản tải về có test_pairs_paired.txt thiếu dòng (vd. 1953/2032). Khi
    danh sách không phủ hết test/image, tạo danh sách đầy đủ từ các định danh có
    cả ảnh người và ảnh áo, ghi vào thư mục kết quả và dùng danh sách đó.
    """
    def stems(folder):
        return {p.stem: p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS}

    people, cloth = stems(root / "test" / "image"), stems(root / "test" / "cloth")
    rows = [line.split() for line in listed.read_text().splitlines() if line.strip()]
    if {Path(r[0]).stem for r in rows} == set(people):
        return listed, {"paired_list": str(listed), "paired_rows": len(rows)}
    common = sorted(set(people) & set(cloth))
    full = out_root / "test_pairs_paired_full.txt"
    full.write_text("".join(f"{people[k]} {cloth[k]}\n" for k in common))
    print(f"CẢNH BÁO: {listed} có {len(rows)} dòng nhưng test/image có {len(people)} người; "
          f"dùng danh sách đầy đủ {len(common)} cặp: {full}", flush=True)
    return full, {"paired_list": str(full), "paired_rows": len(common),
                  "paired_list_original": str(listed), "paired_rows_original": len(rows)}


def build_dataset(cfg, root: Path, setting: str, limit: int, pairs: Path):
    ds = VitonHDDataset(
        str(root), phase="test", order="paired" if setting == "pair" else "unpaired",
        size=tuple(cfg.model.image_hw), pairs_file=str(pairs),
        require_cloth_mask=True, require_parse=False)
    if limit and limit < len(ds):
        ds = Subset(ds, list(range(limit)))  # các mẫu đầu tiên: tái lập được, không phụ thuộc seed
    return ds


def rows_of(ds):
    base = ds.dataset if isinstance(ds, Subset) else ds
    idx = ds.indices if isinstance(ds, Subset) else range(len(ds))
    return base, [(i, base.rows[i]) for i in idx]


class IndexedDataset(torch.utils.data.Dataset):
    """Trả về mẫu kèm chỉ số gốc trong danh sách cặp (dùng để sinh nhiễu theo mẫu)."""

    def __init__(self, base, indices):
        self.base, self.indices = base, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        item = self.base[self.indices[k]]
        item = {key: v for key, v in item.items() if v is not None}
        item["index"] = self.indices[k]
        return item


def real_dir_for(ds, out_dir: Path) -> Path:
    """Toàn bộ tập test: dùng thẳng test/image. Tập con: chép đúng các ảnh thật tương ứng."""
    base, rows = rows_of(ds)
    if len(rows) == len(base.rows):
        names = {p.name for p in base.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        if names == {r[0] for _, r in rows}:
            return base.image_dir
    real = out_dir / "real_subset"
    real.mkdir(parents=True, exist_ok=True)
    for i, (person, _) in rows:
        dst = real / f"{Path(person).stem}.png"
        if not dst.exists():
            save_image((base[i]["image"] + 1) / 2, dst)
    return real


# -------------------------------------------------------------- inference

def save_jpeg(img, path: Path, quality: int = 92) -> None:
    """(3,H,W) trong [-1,1] -> JPEG (save_image của torchvision không nhận quality)."""
    from PIL import Image

    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, quality=quality)


def per_sample_noise(indices, shape, seed: int, device):
    """Nhiễu của mỗi mẫu chỉ phụ thuộc (seed, chỉ số mẫu)."""
    out = []
    for i in indices.tolist():
        g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + int(i))
        out.append(torch.randn(shape, generator=g, device=device))
    return torch.stack(out)


def read_done(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open() as f:
        return {int(r["index"]): r for r in csv.DictReader(f)}


def evaluate_setting(model, vae, cfg, ds, setting: str, cfg_scale: float, args, device, out_dir: Path,
                     metrics: PairMetrics | None) -> dict:
    pred_dir = out_dir / "pred"
    strip_dir = out_dir / "strip"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_strips:
        strip_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_sample.csv"
    fields = PAIR_FIELDS if setting == "pair" else UNPAIR_FIELDS

    base, rows = rows_of(ds)
    done = read_done(csv_path)
    todo = [i for i, (person, _) in rows
            if not (i in done and (pred_dir / f"{Path(person).stem}.png").exists())]
    print(f"[{setting} cfg{cfg_scale:g}] {len(rows)} mẫu, đã có {len(rows) - len(todo)}, còn {len(todo)}", flush=True)

    ecfg = cfg.get("eval", {})
    p = float(ecfg.get("p", 0.7)) if args.p is None else args.p
    kw = dict(nfe=args.nfe, sampler=args.sampler, p=p, n_inner=int(ecfg.get("n_inner", 2)),
              context_ratio=args.context_ratio, cfg_scale=cfg_scale,
              cfg_interval=tuple(ecfg.get("cfg_interval", [0.0, 1.0])),
              time_shift=ecfg.get("time_shift", None), return_stats=True)
    open_px = int(ecfg.get("mask_open_px", 0))
    seam_px, feather_px = float(ecfg.get("seam_px", 0.0)), float(ecfg.get("feather_px", 0.0))

    timing = {"seconds": 0.0, "timed_images": 0, "nfe": None, "denoiser_calls": None}
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        if new_file:
            writer.writeheader()
        loader = DataLoader(IndexedDataset(base, todo), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        for b_i, batch in enumerate(loader):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                inputs = prepare_inputs(vae, batch, model, device, mask_open_px=open_px)
                noise = per_sample_noise(batch["index"], inputs["known"].shape[1:], args.seed, device)
                torch.cuda.synchronize(device)
                tic = time.perf_counter()
                lat, stats = generate(model, inputs, noise=noise, **kw)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - tic
            with torch.no_grad():
                rgb = decode_latents(vae, lat.float()).float()
                rgb = composite(rgb, inputs["agnostic_rgb"], inputs["pixel_mask"], seam_px=seam_px,
                                feather_px=feather_px)
            n = rgb.shape[0]
            per_img = elapsed / n
            if b_i > 0 or len(loader) == 1:
                timing["seconds"] += elapsed
                timing["timed_images"] += n
            timing["nfe"], timing["denoiser_calls"] = stats["nfe"], stats["denoiser_calls"]

            if metrics is not None:
                real = batch["image"].to(device).float()
                mse, mse_mask, ssim, lp = metrics(rgb, real, inputs["pixel_mask"].float())
            for k in range(n):
                stem = Path(batch["person_name"][k]).stem
                save_image((rgb[k] + 1) / 2, pred_dir / f"{stem}.png")
                if args.save_strips:
                    strip = torch.cat([batch["garment"][k].to(device), batch["agnostic"][k].to(device),
                                       batch["image"][k].to(device), rgb[k]], dim=-1)
                    save_jpeg(strip, strip_dir / f"{stem}.jpg")
                row = {"index": int(batch["index"][k]), "person": batch["person_name"][k],
                       "garment": batch["garment_name"][k], "sample_seconds": f"{per_img:.4f}"}
                if metrics is not None:
                    row.update(mse=f"{float(mse[k]):.6f}", mse_mask=f"{float(mse_mask[k]):.6f}",
                               ssim=f"{float(ssim[k]):.6f}", lpips=f"{float(lp[k]):.6f}")
                writer.writerow(row)
            fcsv.flush()
            print(f"  {min((b_i + 1) * args.batch_size, len(todo))}/{len(todo)}  {per_img:.2f} s/ảnh", flush=True)

    # Tổng hợp trên toàn bộ CSV (gồm cả các mẫu của lần chạy trước).
    done = read_done(csv_path)
    wanted = {i for i, _ in rows}
    missing = wanted - set(done)
    if missing:
        raise RuntimeError(f"thiếu {len(missing)} mẫu trong {csv_path}")
    summary = {"setting": setting, "cfg_scale": cfg_scale, "num_samples": len(rows)}
    if setting == "pair":
        for key in ("mse", "mse_mask", "ssim", "lpips"):
            vals = [float(done[i][key]) for i in wanted]
            summary[f"{key}_mean"] = sum(vals) / len(vals)
    expected = {f"{Path(r[0]).stem}.png" for _, r in rows}
    if not expected <= {p.name for p in pred_dir.glob("*.png")}:
        raise RuntimeError(f"thiếu ảnh dự đoán trong {pred_dir}")
    if args.no_fid:
        summary.update(fid=None, kid=None, kid_subset_std=None)
    else:
        print(f"[{setting} cfg{cfg_scale:g}] tính FID/KID ...", flush=True)
        summary.update(distribution_metrics(real_dir_for(ds, out_dir), pred_dir, kid=len(rows) >= 1000))
    if timing["timed_images"]:
        summary["sample_seconds_per_image"] = timing["seconds"] / timing["timed_images"]
        summary["sampler_steps_per_image"] = timing["nfe"]
        summary["denoiser_calls_per_image"] = timing["denoiser_calls"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint .pt (latest.pt hoặc stepXXXXXXX.pt)")
    ap.add_argument("--config", required=True, help="config YAML của checkpoint (vd. configs/vton_v50_attn_flow.yaml)")
    ap.add_argument("--data-root", default=None, help="thư mục VITON-HD; mặc định là data.root trong config")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--settings", default="pair,unpair", help="pair, unpair hoặc pair,unpair")
    ap.add_argument("--sampler", default="look_ahead", choices=SAMPLERS)
    ap.add_argument("--nfe", type=int, default=50, help="số bước sampler (look_ahead cần số chẵn)")
    ap.add_argument("--cfg-scales", default="1.5", help="danh sách hệ số CFG, vd. 1,1.5,2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p", type=float, default=None, help="phân vị độ bất định của look-ahead (mặc định eval.p = 0.7)")
    ap.add_argument("--context-ratio", type=float, default=1.5, help="khoảng nhìn trước của look-ahead")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0: chỉ đánh giá N mẫu đầu (chạy thử)")
    ap.add_argument("--save-strips", action="store_true", help="lưu thêm ảnh ghép 4 cột (JPEG) cho mỗi mẫu")
    ap.add_argument("--no-fid", action="store_true", help="bỏ qua FID/KID")
    ap.add_argument("--overwrite", action="store_true", help="xóa per_sample.csv cũ và chạy lại mọi mẫu")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("cần GPU CUDA")
    if args.sampler == "look_ahead" and args.nfe % 2:
        raise SystemExit("look_ahead cần --nfe chẵn")
    if args.p is not None and not 0 <= args.p < 1:
        raise SystemExit("--p phải trong [0, 1)")
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if not settings or any(s not in ("pair", "unpair") for s in settings):
        raise SystemExit("--settings chỉ nhận pair và/hoặc unpair")
    scales = [float(v) for v in args.cfg_scales.split(",")]

    cfg = load_cfg(args.config)
    root = Path(args.data_root or cfg.data.root).resolve()
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"root": str(root)}}))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint).resolve()
    meta = {"checkpoint": str(ckpt), "checkpoint_size_bytes": ckpt.stat().st_size,
            "checkpoint_mtime_ns": ckpt.stat().st_mtime_ns, "config": args.config, "data_root": str(root),
            "sampler": args.sampler, "nfe": args.nfe, "seed": args.seed,
            "p": float(cfg.eval.get("p", 0.7)) if args.p is None else args.p,
            "context_ratio": args.context_ratio, "limit": args.limit}
    meta_path = out_root / "metadata.json"
    lists = {"pair": None, "unpair": None}
    if "pair" in settings:
        lists["pair"], info = complete_paired_list(pairs_file(cfg, root, "pair"), root, out_root)
        meta.update(info)
    if "unpair" in settings:
        lists["unpair"] = pairs_file(cfg, root, "unpair")
        meta["unpaired_list"] = str(lists["unpair"])
    if meta_path.exists() and not args.overwrite:
        old = json.loads(meta_path.read_text())
        # Danh sách cặp là giá trị suy ra (có thể chưa có nếu lần trước chỉ chạy một thiết lập).
        diff = {k for k in meta if old.get(k) != meta[k] and not k.startswith(("paired_", "unpaired_"))}
        if diff:
            raise SystemExit(f"{out_root} đã dùng cho cấu hình khác ({sorted(diff)}); đổi --output-dir hoặc dùng --overwrite")
    if args.overwrite:
        for f in out_root.glob("cfg*/*/per_sample.csv"):
            f.unlink()

    model, step = load_for_inference(cfg, str(ckpt), device)
    vae = load_sd_vae(cfg.weights.vae, device)
    meta["checkpoint_step"] = step
    meta["gpu"] = torch.cuda.get_device_name(device)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"checkpoint {ckpt} (bước {step}) | {args.sampler} NFE {args.nfe} | CFG {scales} | seed {args.seed}", flush=True)

    metrics = PairMetrics(device) if "pair" in settings else None
    results = {}
    for setting in settings:
        ds = build_dataset(cfg, root, setting, args.limit, lists[setting])
        for w in scales:
            out_dir = out_root / f"cfg{w:g}" / setting
            res = evaluate_setting(model, vae, cfg, ds, setting, w, args, device, out_dir,
                                   metrics if setting == "pair" else None)
            results[f"cfg{w:g}/{setting}"] = res
            (out_root / "summary.json").write_text(json.dumps({"metadata": meta, "results": results},
                                                              indent=2, ensure_ascii=False))

    print("\nKết quả:")
    for key, r in results.items():
        cells = [f"{k}={r[k]:.5g}" for k in ("mse_mean", "mse_mask_mean", "ssim_mean", "lpips_mean", "fid")
                 if r.get(k) is not None]
        if r.get("kid") is not None:
            cells.append(f"KIDx1000={1000 * r['kid']:.3f}")
        if r.get("sample_seconds_per_image") is not None:
            cells.append(f"{r['sample_seconds_per_image']:.2f} s/ảnh")
        print(f"  {key:<14} " + "  ".join(cells))
    print(f"\nĐã lưu ở {out_root}")


if __name__ == "__main__":
    main()
