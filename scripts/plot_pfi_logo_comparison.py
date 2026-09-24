"""Make a native-resolution, unenhanced comparison of fixed dev logo crops."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CASES = ((65, "Lee"), (44, "GAP"), (163, "FILA"), (93, "Wrangler"))
IMAGE_HW = (512, 384)
PADDING = 2
# One fixed upper-torso crop for all models in each case. Each original person
# panel is 384 x 512; no resize or sharpening is applied to the crop itself.
BOX = (35, 70, 350, 320)
def panel(image, col):
    x = PADDING + col * (IMAGE_HW[1] + PADDING)
    y = PADDING
    return image.crop((x + BOX[0], y + BOX[1], x + BOX[2], y + BOX[3]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--candidate-label", default="Candidate")
    args = p.parse_args()
    columns = (("Garment", "baseline", 0), ("Target", "baseline", 1),
               ("VAE", "baseline", 2), (f"{args.baseline_label} Euler 8", "baseline", 3),
               (f"{args.candidate_label} Euler 8", "candidate", 3),
               (f"{args.baseline_label} Dual 8", "baseline", 4),
               (f"{args.candidate_label} Dual 8", "candidate", 4),
               (f"{args.baseline_label} Euler 50", "baseline", 6),
               (f"{args.candidate_label} Euler 50", "candidate", 6))
    w, h = BOX[2] - BOX[0], BOX[3] - BOX[1]
    header, gutter, row_label = 42, 4, 120
    canvas = Image.new("RGB", (row_label + len(columns) * (w + gutter), header + len(CASES) * (h + gutter)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for c, (name, _, _) in enumerate(columns):
        draw.text((row_label + c * (w + gutter) + 3, 10), name, fill="#111111", font=font)
    for row, (index, label) in enumerate(CASES):
        images = {side: Image.open(path / "previews" / f"case{index:03d}.png").convert("RGB")
                  for side, path in (("baseline", args.baseline), ("candidate", args.candidate))}
        y = header + row * (h + gutter)
        draw.text((5, y + 10), f"{index} {label}", fill="#111111", font=font)
        for c, (_, side, source_col) in enumerate(columns):
            canvas.paste(panel(images[side], source_col), (row_label + c * (w + gutter), y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
