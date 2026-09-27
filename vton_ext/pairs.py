"""VITON-HD pairing checks; no tensor or GPU dependencies."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random


def read_pairs(path: Path, *, paired: bool = True):
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (1, 2) or any(Path(p).name != p for p in parts):
            raise ValueError(f"Invalid pair row at {path}:{line_no}")
        person, garment = parts[0], parts[-1]
        # VITON-HD's paired in-shop garment has the same identity as its person.
        if paired and Path(person).stem != Path(garment).stem:
            raise ValueError(
                f"Unpaired garment at {path}:{line_no}: {person} {garment}. "
                "Paired supervision requires the garment worn by the target. "
                "Run python -m scripts.prepare_pfi_pairs --root <data-root>; "
                "do not split the archive's unpaired train_pairs.txt."
            )
        rows.append((person, garment))
    if not rows:
        raise ValueError(f"No pairs found in {path}")
    return rows


def manifest(rows, path: Path):
    canonical = "".join(f"{p} {g}\n" for p, g in rows).encode()
    return {"count": len(rows), "sha256": hashlib.sha256(canonical).hexdigest(),
            "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def split_manifest(fit_path: Path, dev_path: Path):
    fit, dev = read_pairs(fit_path), read_pairs(dev_path)
    for label, rows in (("fit", fit), ("dev", dev)):
        people = [Path(p).stem for p, _ in rows]
        if len(set(people)) != len(people):
            raise ValueError(f"Duplicate person identities in {label} split")
    for column, label in ((0, "person"), (1, "garment")):
        overlap = {Path(r[column]).stem for r in fit} & {Path(r[column]).stem for r in dev}
        if overlap:
            raise ValueError(f"Fit/dev {label} overlap: {sorted(overlap)[:5]}")
    return {"fit": manifest(fit, fit_path), "dev": manifest(dev, dev_path)}


def validate_resume_pairs(saved, current):
    if saved is None:
        return False  # Legacy checkpoints cannot certify the original split.
    for split in ("fit", "dev"):
        if any(saved[split][key] != current[split][key] for key in ("count", "sha256")):
            raise ValueError(
                f"The {split} pairs differ from the resume checkpoint (including row order). "
                "Restore the original split before --resume. For an intentional data change, "
                "use weights.init_from in a new output directory with a fresh optimizer."
            )
    return True


def paired_split(root: Path, seed: int = 2026, dev_size: int = 256, method: str = "stable-hash"):
    def index(folder):
        result = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if path.stem in result:
                raise ValueError(f"Ambiguous identity {path.stem} in {folder}")
            result[path.stem] = path.name
        return result

    root = Path(root)
    people, garments = index(root / "train/image"), index(root / "train/cloth")
    missing = set(people) - set(garments)
    if missing:
        raise ValueError(f"Missing matching garments: {sorted(missing)[:5]}")
    rows = [(people[k], garments[k]) for k in sorted(people)]
    if not 0 < dev_size < len(rows):
        raise ValueError("dev_size must be positive and smaller than the training set")
    if method == "stable-hash":
        # Matches the original make_vton_dev_split.py recipe, including row order.
        ranked = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
            f"{seed}:{rows[i][0]} {rows[i][1]}".encode()).digest())
        selected = set(ranked[:dev_size])
    elif method == "random":
        selected = set(random.Random(seed).sample(range(len(rows)), dev_size))
    else:
        raise ValueError(f"Unknown split method: {method}")
    return ([r for i, r in enumerate(rows) if i not in selected],
            [r for i, r in enumerate(rows) if i in selected])
