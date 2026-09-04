"""Generate a small, privacy-safe demo dataset for Matching Clothes.

The images are deliberately synthetic. They contain no customer photos,
supplier information, or assets copied from the user's production library.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "style_library"
QUERIES = ROOT / "store_photos"

STYLES = [
    ("demo_orbit_01.png", "ORBIT 7", "orbit", (229, 231, 235), (32, 48, 72)),
    ("demo_mountain_02.png", "NORTH 23", "mountain", (58, 63, 69), (232, 224, 205)),
    ("demo_grid_03.png", "GRID LAB", "grid", (212, 198, 171), (48, 45, 42)),
    ("demo_echo_04.png", "ECHO", "echo", (32, 35, 41), (225, 230, 236)),
    ("demo_wave_05.png", "WAVE 11", "wave", (205, 216, 220), (37, 75, 91)),
    ("demo_block_06.png", "BLOCK 5", "block", (238, 232, 218), (114, 51, 49)),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "msyhbd.ttc" if bold else "msyh.ttc"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def shirt_polygon(cx: int, top: int, width: int, height: int) -> list[tuple[int, int]]:
    left = cx - width // 2
    right = cx + width // 2
    shoulder = top + int(height * 0.08)
    sleeve_bottom = top + int(height * 0.34)
    hem = top + height
    return [
        (cx - int(width * .18), top),
        (left + int(width * .12), shoulder),
        (left, sleeve_bottom),
        (left + int(width * .18), sleeve_bottom + int(height * .06)),
        (left + int(width * .24), hem),
        (right - int(width * .24), hem),
        (right - int(width * .18), sleeve_bottom + int(height * .06)),
        (right, sleeve_bottom),
        (right - int(width * .12), shoulder),
        (cx + int(width * .18), top),
    ]


def draw_design(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, ink: tuple[int, int, int]) -> None:
    if kind == "orbit":
        for radius in (38, 62, 88):
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=ink, width=9)
        draw.line((cx-105, cy+80, cx+110, cy-75), fill=ink, width=13)
    elif kind == "mountain":
        draw.line((cx-125, cy+75, cx-35, cy-55, cx+15, cy+15, cx+75, cy-90, cx+135, cy+75), fill=ink, width=14)
        draw.line((cx-145, cy+78, cx+145, cy+78), fill=ink, width=8)
    elif kind == "grid":
        for offset in range(-105, 106, 42):
            draw.line((cx+offset, cy-105, cx+offset, cy+105), fill=ink, width=6)
            draw.line((cx-105, cy+offset, cx+105, cy+offset), fill=ink, width=6)
        draw.rectangle((cx-62, cy-62, cx+62, cy+62), outline=ink, width=14)
    elif kind == "echo":
        points = []
        for x in range(-140, 141, 8):
            y = int(math.sin(x / 18) * (75 - abs(x) * .18))
            points.append((cx+x, cy+y))
        draw.line(points, fill=ink, width=12)
        draw.ellipse((cx-22, cy-22, cx+22, cy+22), fill=ink)
    elif kind == "wave":
        for row in range(5):
            points = []
            for x in range(-135, 136, 8):
                points.append((cx+x, cy-70+row*36+int(math.sin(x/24+row)*13)))
            draw.line(points, fill=ink, width=8)
    else:
        for x, y in [(-100, -85), (15, -85), (-100, 30), (15, 30)]:
            draw.rounded_rectangle((cx+x, cy+y, cx+x+86, cy+y+86), radius=12, outline=ink, width=12)


def render_style(
    title: str,
    kind: str,
    garment: tuple[int, int, int],
    ink: tuple[int, int, int],
    *,
    query: bool = False,
    annotation: str = "",
) -> Image.Image:
    size = (780, 1080) if query else (720, 960)
    bg = (220, 218, 211) if query else (248, 248, 246)
    image = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(image)

    if query:
        draw.rectangle((0, 0, size[0], 170), fill=(190, 189, 185))
        for x in (90, 690):
            draw.rectangle((x, 0, x+18, size[1]), fill=(177, 177, 174))
        draw.line((size[0]//2, 18, size[0]//2, 118), fill=(75, 75, 75), width=8)
        draw.arc((size[0]//2-70, 62, size[0]//2+70, 160), 195, 345, fill=(75, 75, 75), width=8)

    cx = size[0] // 2
    top = 125 if query else 95
    width = 670 if query else 620
    height = 820 if query else 735
    poly = shirt_polygon(cx, top, width, height)
    draw.polygon(poly, fill=garment, outline=(92, 92, 92))
    draw.line(poly + [poly[0]], fill=(92, 92, 92), width=5, joint="curve")
    draw.ellipse((cx-92, top-25, cx+92, top+82), fill=bg, outline=(92, 92, 92), width=5)

    design_y = top + int(height * .48)
    draw_design(draw, kind, cx, design_y, ink)
    title_font = font(48 if query else 44, bold=True)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text((cx-(box[2]-box[0])//2, design_y+135), title, font=title_font, fill=ink)

    if annotation:
        label_font = font(54, bold=True)
        draw.multiline_text((70, 235), annotation, font=label_font, fill=(245, 52, 52), spacing=10)
    return image


def main() -> None:
    LIBRARY.mkdir(parents=True, exist_ok=True)
    QUERIES.mkdir(parents=True, exist_ok=True)
    for filename, title, kind, garment, ink in STYLES:
        render_style(title, kind, garment, ink).save(LIBRARY / filename, optimize=True)

    demos = [
        ("query_orbit_gray_M1_XL1.jpg", STYLES[0], (190, 194, 201), "GRAY\nM1\nXL1"),
        ("query_echo_black_L2.jpg", STYLES[3], (25, 27, 31), "BLACK\nL2"),
        ("query_wave_blue_2XL1.jpg", STYLES[4], (101, 132, 146), "BLUE\n2XL1"),
    ]
    rows = []
    for query_name, style, garment, annotation in demos:
        filename, title, kind, _, ink = style
        render_style(title, kind, garment, ink, query=True, annotation=annotation).save(
            QUERIES / query_name, quality=92, optimize=True
        )
        rows.append((query_name, filename))

    with (ROOT / "expected_matches.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["store_photo", "expected_style"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
