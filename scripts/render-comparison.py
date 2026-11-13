"""Compose the before/after contact sheet for the ring-breaking change.

    python scripts/render-comparison.py <before.json> <after.json> <out.png>

Reuses the renderer in render-heights.py (loaded by path, because the file name has a dash
and is not importable). The font is the one the page already ships, so the Chinese labels
render instead of turning into boxes.
"""
import importlib.util
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("render_heights", os.path.join(HERE, "render-heights.py"))
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)

CELL = 430
PAD = 16
LABEL_H = 30
BACKGROUND = (12, 14, 10)
INK = (198, 203, 182)


def font(size):
    path = os.path.join(ROOT, "public", "fonts", "MiSansVF.ttf")
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit(image, width):
    scale = width / image.width
    return image.resize((width, int(image.height * scale)), Image.LANCZOS)


def main():
    if len(sys.argv) != 4:
        print("usage: python scripts/render-comparison.py <before.json> <after.json> <out.png>")
        return 2
    before = render.load(sys.argv[1])
    after = render.load(sys.argv[2])
    rows = [
        (
            "改动前 · 俯视高度图：能数出同心圈",
            "改动后 · 同心圈读不出来了",
            render.top_down(*before),
            render.top_down(*after),
            1,
        ),
        (
            "改动前 · 30 级同心台地",
            "改动后 · 同一条径向包络，表面被打散",
            render.isometric(*before, scale=60.0),
            render.isometric(*after, scale=60.0),
            3,
        ),
    ]
    prepared = []
    for left_label, right_label, left, right, shrink_factor in rows:
        if shrink_factor > 1:
            # The isometric panels are rasterised at 3x; downsampling is the only
            # anti-aliasing ImageDraw offers.
            left = left.resize((left.width // shrink_factor, left.height // shrink_factor), Image.LANCZOS)
            right = right.resize((right.width // shrink_factor, right.height // shrink_factor), Image.LANCZOS)
        prepared.append((left_label, right_label, fit(left, CELL), fit(right, CELL)))

    row_heights = [max(a.height, b.height) + LABEL_H for _, _, a, b in prepared]
    size = (CELL * 2 + PAD * 3, sum(row_heights) + PAD * (len(prepared) + 1))
    sheet = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    text = font(17)
    y = PAD
    for (left_label, right_label, left, right), row_height in zip(prepared, row_heights):
        draw.text((PAD, y), left_label, font=text, fill=INK)
        draw.text((PAD + CELL + PAD, y), right_label, font=text, fill=INK)
        y += LABEL_H
        sheet.paste(left, (PAD, y))
        sheet.paste(right, (PAD + CELL + PAD, y))
        y += row_height - LABEL_H + PAD
    sheet.save(sys.argv[3])
    print(f"{sys.argv[3]}: {sheet.size[0]}x{sheet.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
