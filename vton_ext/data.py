from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from vton_ext.pairs import read_pairs


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _find_named(directory: Path, source_name: str, extra_stems: Sequence[str] = ()) -> Path:
    stem = Path(source_name).stem
    candidates = [directory / source_name]
    for s in (stem, *extra_stems):
        for ext in _IMAGE_EXTS:
            candidates.append(directory / f"{s}{ext}")
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError(f"Could not resolve '{source_name}' in {directory}")
    return found


def _resolve_dir(base: Path, names: Sequence[str], required: bool = True) -> Path | None:
    for name in names:
        p = base / name
        if p.is_dir():
            return p
    if required:
        raise FileNotFoundError(f"None of the expected folders exist under {base}: {names}")
    return None


def _rgb(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    im = Image.open(path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    x = TF.pil_to_tensor(im).float() / 127.5 - 1.0
    return x


def _mask(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    im = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    x = TF.pil_to_tensor(im).float() / 255.0
    return (x > 0.5).float()


def _parse_mask(path: Path, size_hw: Tuple[int, int], labels: Sequence[int]) -> torch.Tensor:
    """Return a binary mask for selected human-parsing labels."""
    h, w = size_hw
    im = Image.open(path).convert("P").resize((w, h), Image.Resampling.NEAREST)
    labels_t = torch.as_tensor(tuple(int(v) for v in labels), dtype=torch.uint8)
    parse = TF.pil_to_tensor(im).squeeze(0)
    return (parse[..., None] == labels_t).any(dim=-1).float().unsqueeze(0)


class VitonHDDataset(Dataset):
    """VITON-HD loader for the PFT VTON extension.

    Model inputs: agnostic image, agnostic mask, DensePose, in-shop garment.
    The original person image is returned only as training ground truth / DINO teacher input.
    """

    def __init__(
        self,
        root: str,
        phase: Literal["train", "test"] = "train",
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        pairs_file: str | None = None,
        require_cloth_mask: bool = False,
        require_parse: bool = False,
        clothing_labels: Sequence[int] = (5, 6, 7),
        preserve_skin: bool = False,
        preserve_known_context: bool = False,
        augment: bool = False,
        person_translate_fraction: float = 0.0,
        person_scale_range: Tuple[float, float] = (1.0, 1.0),
        garment_translate_fraction: float = 0.0,
        garment_scale_range: Tuple[float, float] = (1.0, 1.0),
    ):
        self.root = Path(root)
        self.phase = phase
        self.order = order
        self.size = tuple(size)
        self.clothing_labels = tuple(int(v) for v in clothing_labels)
        self.preserve_skin = bool(preserve_skin)
        self.preserve_known_context = bool(preserve_known_context)
        self.augment = bool(augment)
        self.person_translate_fraction = max(float(person_translate_fraction), 0.0)
        self.person_scale_range = tuple(float(v) for v in person_scale_range)
        self.garment_translate_fraction = max(float(garment_translate_fraction), 0.0)
        self.garment_scale_range = tuple(float(v) for v in garment_scale_range)
        for name, scale_range in (
            ("person_scale_range", self.person_scale_range),
            ("garment_scale_range", self.garment_scale_range),
        ):
            if len(scale_range) != 2 or not 0 < scale_range[0] <= scale_range[1]:
                raise ValueError(f"Invalid {name}: {scale_range}")
        self.base = self.root / phase
        if not self.base.is_dir():
            raise FileNotFoundError(f"Missing VITON-HD split folder: {self.base}")

        self.image_dir = _resolve_dir(self.base, ("image",))
        self.cloth_dir = _resolve_dir(self.base, ("cloth", "garment"))
        self.agnostic_dir = _resolve_dir(self.base, ("agnostic-v3.2", "agnostic", "image-agnostic"))
        self.mask_dir = _resolve_dir(self.base, ("agnostic-mask", "agnostic_mask", "mask"))
        self.pose_dir = _resolve_dir(self.base, ("image-densepose", "densepose", "dense-pose"))
        self.cloth_mask_dir = _resolve_dir(self.base, ("cloth-mask", "cloth_mask"), required=False)
        self.parse_dir = _resolve_dir(
            self.base, ("image-parse-v3", "image-parse", "parse"), required=False
        )
        if require_cloth_mask and self.cloth_mask_dir is None:
            raise FileNotFoundError("require_cloth_mask=true but no cloth-mask folder was found")
        if require_parse and self.parse_dir is None:
            raise FileNotFoundError("require_parse=true but no human-parse folder was found")
        if (self.preserve_skin or self.preserve_known_context) and self.parse_dir is None:
            raise FileNotFoundError("preserving known person context requires human parsing")

        if pairs_file is None:
            candidates = []
            if order == "unpaired":
                candidates.extend([
                    self.root / f"{phase}_pairs_unpaired.txt",
                    self.base / f"{phase}_pairs_unpaired.txt",
                ])
            else:
                candidates.extend([
                    self.root / f"{phase}_pairs_paired.txt",
                    self.base / f"{phase}_pairs_paired.txt",
                ])
            candidates.extend([
                self.root / f"{phase}_pairs.txt",
                self.base / f"{phase}_pairs.txt",
            ])
            pair_path = _first_existing(candidates)
        else:
            p = Path(pairs_file)
            pair_path = p if p.is_absolute() else _first_existing((self.root / p, self.base / p, p))
        if pair_path is None:
            raise FileNotFoundError("Could not find a VITON-HD pair list. Pass data.pairs_file explicitly.")

        self.rows = read_pairs(pair_path, paired=order == "paired")
        self.pair_path = pair_path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        person_name, garment_name = self.rows[index]
        person_path = _find_named(self.image_dir, person_name)
        garment_path = _find_named(self.cloth_dir, garment_name)
        agnostic_path = _find_named(self.agnostic_dir, person_name)
        pose_path = _find_named(self.pose_dir, person_name)
        stem = Path(person_name).stem
        mask_path = _find_named(
            self.mask_dir,
            person_name,
            extra_stems=(f"{stem}_mask", f"{stem}_agnostic_mask"),
        )

        cloth_mask = None
        if self.cloth_mask_dir is not None:
            gstem = Path(garment_name).stem
            try:
                cloth_mask_path = _find_named(
                    self.cloth_mask_dir,
                    garment_name,
                    extra_stems=(f"{gstem}_mask",),
                )
                cloth_mask = _mask(cloth_mask_path, self.size)
            except FileNotFoundError:
                cloth_mask = None

        if cloth_mask is None:
            cloth_mask = torch.ones(1, *self.size, dtype=torch.float32)

        clothing_mask = None
        skin_mask = None
        known_context_mask = None
        if self.parse_dir is not None:
            try:
                parse_path = _find_named(self.parse_dir, person_name)
                clothing_mask = _parse_mask(parse_path, self.size, self.clothing_labels)
                if self.preserve_skin:
                    # LIP/VITON-HD labels 13=face, 14/15=arms. Only copy skin
                    # already visible in the input person image; no garment RGB.
                    skin_mask = _parse_mask(parse_path, self.size, (13, 14, 15))
                if self.preserve_known_context:
                    # Use only confidently non-garment parse classes. Retain an
                    # editable 10px halo around the old shirt for sleeve changes.
                    known_context_mask = _parse_mask(
                        parse_path, self.size,
                        (0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19),
                    )
            except FileNotFoundError:
                clothing_mask = None
        if self.preserve_skin and skin_mask is None:
            raise FileNotFoundError(f"Missing person parse for skin preservation: {person_name}")
        if self.preserve_known_context and known_context_mask is None:
            raise FileNotFoundError(f"Missing person parse for context preservation: {person_name}")
        if clothing_mask is None:
            # Safe fallback for datasets without parsing: focus on the editable area.
            clothing_mask = _mask(mask_path, self.size)

        image = _rgb(person_path, self.size)
        agnostic = _rgb(agnostic_path, self.size)
        agnostic_mask = _mask(mask_path, self.size)
        densepose = _rgb(pose_path, self.size)
        garment = _rgb(garment_path, self.size)

        if self.augment:
            h, w = self.size

            def parameters(translate_fraction, scale_range):
                max_dx = int(round(w * translate_fraction))
                max_dy = int(round(h * translate_fraction))
                dx = int(torch.randint(-max_dx, max_dx + 1, ()).item()) if max_dx else 0
                dy = int(torch.randint(-max_dy, max_dy + 1, ()).item()) if max_dy else 0
                scale = float(
                    torch.empty(()).uniform_(scale_range[0], scale_range[1]).item()
                )
                return [dx, dy], scale

            person_translate, person_scale = parameters(
                self.person_translate_fraction, self.person_scale_range
            )
            garment_translate, garment_scale = parameters(
                self.garment_translate_fraction, self.garment_scale_range
            )

            def affine_rgb(x, translate, scale, fill):
                return TF.affine(
                    x,
                    angle=0.0,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=fill,
                )

            def affine_mask(x, translate, scale):
                return TF.affine(
                    x,
                    angle=0.0,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )

            image = affine_rgb(image, person_translate, person_scale, 1.0)
            agnostic = affine_rgb(agnostic, person_translate, person_scale, 1.0)
            # DensePose visualization uses semantic left/right colors. Translation
            # and isotropic scale are safe; raw horizontal flipping is not.
            densepose = affine_rgb(densepose, person_translate, person_scale, -1.0)
            agnostic_mask = affine_mask(
                agnostic_mask, person_translate, person_scale
            )
            clothing_mask = affine_mask(
                clothing_mask, person_translate, person_scale
            )
            if skin_mask is not None:
                skin_mask = affine_mask(skin_mask, person_translate, person_scale)
            if known_context_mask is not None:
                known_context_mask = affine_mask(
                    known_context_mask, person_translate, person_scale
                )
            garment = affine_rgb(
                garment, garment_translate, garment_scale, 1.0
            )
            cloth_mask = affine_mask(
                cloth_mask, garment_translate, garment_scale
            )

        if known_context_mask is not None:
            shirt_halo = torch.nn.functional.max_pool2d(
                clothing_mask, kernel_size=21, stride=1, padding=10
            )
            known_context_mask = known_context_mask * (1.0 - shirt_halo)
        if skin_mask is not None:
            # Conservative interior avoids leaking shirt pixels at arm borders.
            skin_mask = 1.0 - torch.nn.functional.max_pool2d(
                1.0 - skin_mask, kernel_size=3, stride=1, padding=1
            )
            known_context_mask = (
                skin_mask if known_context_mask is None
                else torch.maximum(known_context_mask, skin_mask)
            )
        if known_context_mask is not None:
            agnostic = known_context_mask * image + (1.0 - known_context_mask) * agnostic
            agnostic_mask = agnostic_mask * (1.0 - known_context_mask)

        return {
            "image": image,
            "agnostic": agnostic,
            "agnostic_mask": agnostic_mask,
            "densepose": densepose,
            "garment": garment,
            "garment_mask": cloth_mask,
            "clothing_mask": clothing_mask,
            "person_name": person_name,
            "garment_name": garment_name,
        }
