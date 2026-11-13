# Rasterizes docs/bench.svg just far enough to check the layout: rects, lines and text at
# their stated positions. Not a general SVG renderer -- it exists because no browser can be
# launched in this sandbox, and a chart nobody has looked at is a chart that might be empty.
#
#   python scripts/svg-preview.py docs/bench.svg docs/bench-preview.png
#
# The output is for looking at, not for shipping: fonts and text widths differ from a
# browser's, so trust it for overlap and spacing, not for typography.
import re
import sys
from PIL import Image, ImageDraw, ImageFont

src = open(sys.argv[1], encoding="utf-8").read()
W = int(re.search(r'width="(\d+)"', src).group(1))
H = int(re.search(r'height="(\d+)"', src).group(1))
bg = re.search(r'<rect width="100%" height="100%" fill="(#\w+)"', src)
img = Image.new("RGB", (W, H), bg.group(1) if bg else "white")
draw = ImageDraw.Draw(img, "RGBA")

FONTS = {}
SIZES = {"title": 16, "h": 13, "note": 10.5, "label": 12, "value": 12, "pct": 12, "tick": 10}
BOLD = {"title", "h", "value"}


def font(cls):
    size, bold = SIZES.get(cls, 12), cls in BOLD
    key = (round(size), bold)
    if key not in FONTS:
        name = "consolab.ttf" if bold else "consola.ttf"
        try:
            FONTS[key] = ImageFont.truetype(name, round(size))
        except OSError:
            FONTS[key] = ImageFont.load_default(round(size))
    return FONTS[key]


def color(value, default=(219, 230, 240)):
    if not value or value == "none":
        return None
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


body = src.split("<rect width=\"100%\"", 1)[1]
body = body.split(">", 1)[1]
count = 0
for tag in re.finditer(r"<(rect|line|text)\b([^>]*?)(?:/>|>(.*?)</text>)", body, re.S):
    kind, attrs, text = tag.group(1), tag.group(2), tag.group(3)
    a = dict(re.findall(r'(\S+?)="([^"]*)"', attrs))
    cls = a.get("class", "")

    if kind == "line":
        draw.line(
            [float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])],
            fill=color(a.get("stroke"), (60, 60, 60)),
            width=int(float(a.get("stroke-width", 1))),
        )
        count += 1

    elif kind == "rect":
        x, y = float(a["x"]), float(a["y"])
        w, h = float(a["width"]), float(a["height"])
        fill = color(a.get("fill"))
        alpha = float(a.get("fill-opacity", 1))
        if fill:
            draw.rounded_rectangle(
                [x, y, x + w, y + h], radius=float(a.get("rx", 0)),
                fill=fill + (int(alpha * 255),),
            )
        stroke = color(a.get("stroke"))
        if stroke:
            draw.rounded_rectangle(
                [x, y, x + w, y + h], radius=float(a.get("rx", 0)),
                outline=stroke + (int(float(a.get("stroke-opacity", 1)) * 255),),
                width=int(float(a.get("stroke-width", 1))),
            )
        count += 1

    else:
        value = (text or "").strip()
        f = font(cls)
        x, y = float(a["x"]), float(a["y"])
        left, top, right, bottom = draw.textbbox((0, 0), value, font=f)
        width = right - left
        anchor = a.get("text-anchor", "start")
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        draw.text((x, y - bottom), value, font=f, fill=color(a.get("fill"), (255, 255, 255)))
        count += 1

img.save(sys.argv[2])
print(f"{count} primitives -> {sys.argv[2]} ({W}x{H})")
