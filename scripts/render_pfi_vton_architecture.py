"""Render the code-checked PFI-VTON architecture figure for the Vietnamese paper."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
BG = "#f8fafc"
INK = "#162338"
MUTED = "#526178"
BLUE = "#216ac4"
TEAL = "#098676"
ORANGE = "#bb6a16"
PURPLE = "#7657a5"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD if bold else FONT
    if not Path(path).exists():
        raise FileNotFoundError(f"Install Noto Sans or adjust FONT in {__file__}: {path}")
    return ImageFont.truetype(path, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/assets/pfi-vton-architecture.png",
    )
    args = parser.parse_args()
    img = Image.new("RGB", (3000, 1720), BG)
    d = ImageDraw.Draw(img)

    def label(x: int, y: int, s: str, size: int = 31, color: str = INK, bold: bool = False) -> None:
        d.text((x, y), s, font=font(size, bold), fill=color)

    def card(box: tuple[int, int, int, int], title: str, lines: list[str],
             *, accent: str = BLUE, fill: str = "#ffffff", title_size: int = 43,
             body_size: int = 32, gap: int = 48) -> None:
        x0, y0, x1, y1 = box
        d.rounded_rectangle(box, radius=24, fill=fill, outline=accent, width=4)
        d.rounded_rectangle((x0, y0, x0 + 11, y1), radius=5, fill=accent)
        label(x0 + 33, y0 + 22, title, title_size, accent, True)
        y = y0 + 89
        for line in lines:
            label(x0 + 33, y, line, body_size, MUTED)
            y += gap
        if y > y1 + 20:
            raise ValueError(f"Text overflow in {title}")

    def arrow(points: list[tuple[int, int]], color: str = BLUE, width: int = 7,
              dashed: bool = False) -> None:
        if dashed:
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if not dist:
                    continue
                ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
                p = 0.0
                while p < dist - 23:
                    a, b = p, min(p + 17, dist - 23)
                    d.line((x0 + ux*a, y0 + uy*a, x0 + ux*b, y0 + uy*b),
                           fill=color, width=width)
                    p += 30
        else:
            d.line(points, fill=color, width=width, joint="curve")
        x1, y1 = points[-1]
        x0, y0 = points[-2]
        dx, dy = x1-x0, y1-y0
        norm = max((dx*dx + dy*dy) ** 0.5, 1)
        ux, uy = dx/norm, dy/norm
        wx, wy = -uy, ux
        d.polygon([
            (x1, y1),
            (x1 - ux*24 + wx*12, y1 - uy*24 + wy*12),
            (x1 - ux*24 - wx*12, y1 - uy*24 - wy*12),
        ], fill=color)

    label(88, 53, "PFI-VTON: hai luồng, một bộ trọng số PFT-XL/2", 70, INK, True)
    label(89, 139, "Khi suy luận, áo được mã hóa một lần; chỉ vùng cần thay được khử nhiễu theo patch.", 36, MUTED)
    d.rounded_rectangle((80, 213, 2908, 1616), radius=36, outline="#d6e1ec", width=4)

    card((110, 248, 690, 418), "Lịch thời gian (train)", [
        "Thuần nhiễu · đồng bộ · LTG · chi tiết",
        "V46: ramp theo bước 0–6000",
    ], accent=PURPLE, fill="#f5f0fc", body_size=28, gap=42)

    card((110, 500, 690, 786), "Người cần thử áo", [
        "A: người đã xóa áo   [B,3,H,W]",
        "M: mask vùng cần sửa  [B,1,H,W]",
        "P: DensePose         [B,3,H,W]",
        "Ngoài M: giữ pixel người ban đầu",
    ], accent=BLUE, fill="#eef5fd")
    card((110, 906, 690, 1170), "Áo tham chiếu", [
        "G: ảnh sản phẩm       [B,3,H,W]",
        "M_G: mask áo         [B,1,H,W]",
        "Áo là ngữ cảnh sạch: t = 1",
    ], accent=TEAL, fill="#ecf8f5")
    card((110, 1320, 690, 1515), "Ảnh đích I (chỉ train)", [
        "Áo trong I phải đúng là G",
        "Không được đưa I vào suy luận",
    ], accent=ORANGE, fill="#fff6e9", body_size=31)

    card((795, 500, 1305, 786), "VAE đóng băng", [
        "E(A) + mask + E(P)",
        "Điều kiện: [B,9,h,w]",
        "h=H/8; w=W/8",
        "Known: E(A); edit: nhiễu",
    ], accent=BLUE, fill="#eef5fd")
    card((795, 906, 1305, 1170), "VAE đóng băng", [
        "E(G) + mask áo",
        "Đầu vào áo: [B,5,h,w]",
        "Không phụ thuộc người",
    ], accent=TEAL, fill="#ecf8f5")

    card((1410, 500, 1900, 786), "Token người", [
        "Conv 4->1152 + Conv 9->1152",
        "Vị trí + role EDIT/KNOWN",
        "[B,N,1152]; N=HW/256",
        "Thời gian t_i cho từng patch",
    ], accent=BLUE, fill="#eef5fd", body_size=29)
    card((1410, 906, 1900, 1170), "Token áo", [
        "Conv 5->1152 + vị trí",
        "Role GARMENT; t=1",
        "[B,N,1152]",
    ], accent=TEAL, fill="#ecf8f5", body_size=30)

    d.rounded_rectangle((2000, 460, 2495, 1190), radius=31,
                        fill="#ffffff", outline="#8796ad", width=5)
    label(2025, 477, "28 block PFT dùng chung", 36, INK, True)
    label(2025, 535, "width 1152 · 16 head", 30, MUTED)
    card((2030, 609, 2465, 821), "Người đọc", [
        "Q người đọc K/V người + áo",
        "CoRAL: block 8,12,16,20",
    ], accent=BLUE, fill="#eef5fd", title_size=38, body_size=27, gap=42)
    card((2030, 914, 2465, 1135), "Áo tự đọc áo", [
        "K/V cache: 28 tầng",
        "Tái dùng ở 8 lần gọi người",
    ], accent=TEAL, fill="#ecf8f5", title_size=38, body_size=27, gap=43)
    arrow([(2230, 914), (2230, 838), (2210, 838), (2210, 821)], TEAL, 7)
    label(2260, 844, "K/V", 27, TEAL, True)

    card((2580, 490, 2890, 750), "Đầu ra", [
        "v: [B,4,h,w]",
        "logvar: [B,1,h,w]",
        "chỉ nhánh người",
    ], accent=BLUE, fill="#eef5fd", title_size=40, body_size=28)
    card((2580, 835, 2890, 1083), "Lấy mẫu", [
        "dual-loop, 8 call",
        "không CFG",
        "chỉ cập nhật EDIT",
    ], accent=PURPLE, fill="#f5f0fc", title_size=40, body_size=28)
    card((2580, 1170, 2890, 1485), "Ảnh thử đồ", [
        "VAE D(x) đóng băng",
        "Ghép lại ngoài mask",
        "[B,3,H,W]",
    ], accent=TEAL, fill="#ecf8f5", title_size=40, body_size=28)

    d.rounded_rectangle((795, 1310, 2495, 1515), radius=24,
                        fill="#fff6e9", outline=ORANGE, width=4)
    label(830, 1326, "Chỉ khi train: I + áo G qua DINOv3 / CoRAL", 38, ORANGE, True)
    label(830, 1390, "Flow MSE + NLL + CoRAL CE/entropy + RGB/high-pass (chọn mẫu)", 30, MUTED)
    label(830, 1440, "Target I dùng để tạo loss, không đi vào điều kiện của DiT.", 30, MUTED)

    arrow([(690, 643), (795, 643)], BLUE)
    arrow([(1305, 643), (1410, 643)], BLUE)
    arrow([(1900, 643), (2030, 643)], BLUE)
    arrow([(2495, 716), (2580, 620)], BLUE)
    arrow([(690, 1035), (795, 1035)], TEAL)
    arrow([(1305, 1035), (1410, 1035)], TEAL)
    arrow([(1900, 1035), (2030, 1035)], TEAL)
    arrow([(2734, 750), (2734, 835)], PURPLE)
    arrow([(2734, 1083), (2734, 1170)], PURPLE)
    arrow([(690, 1418), (795, 1418)], ORANGE, dashed=True)
    arrow([(2580, 750), (2535, 750), (2535, 1400), (2495, 1400)],
          ORANGE, dashed=True)
    arrow([(398, 418), (398, 450), (1655, 450), (1655, 500)], PURPLE, dashed=True)
    label(725, 1550, "Xanh: người  ·  Lục: áo  ·  Tím: thời gian/lấy mẫu  ·  Cam nét đứt: chỉ huấn luyện", 30, MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output, optimize=True, dpi=(300, 300))
    print(f"{args.output}: {img.width}x{img.height}")


if __name__ == "__main__":
    main()
