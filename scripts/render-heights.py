"""Draw a dumped height field as the reactor would show it: a shaded top-down map and an
isometric view of the same 709 rods.

    python scripts/render-heights.py <dump.json> <out-prefix>

The top-down map is the diagnostic one: a ring percept is an iso-luminance band, and it is
impossible to miss when it is there. The isometric one answers "does it look like the app".
"""
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

BACKGROUND = (20, 24, 16)
TOP = np.array([216, 220, 194], dtype=float)
SIDE_LIGHT = np.array([92, 99, 76], dtype=float)
SIDE_DARK = np.array([62, 68, 50], dtype=float)


def load(path):
    with open(path, encoding="utf-8") as handle:
        dump = json.load(handle)
    rods = dump["rods"]
    return (
        np.array([r["x"] for r in rods]),
        np.array([r["z"] for r in rods]),
        np.array([r["height"] for r in rods]),
    )


def top_down(x, z, h, cell=26):
    n = 33
    size = n * cell
    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    low, high = float(h.min()), float(h.max())
    span = max(high - low, 1e-6)
    for i in range(len(x)):
        px = (x[i] + n / 2) * cell
        py = (z[i] + n / 2) * cell
        t = (h[i] - low) / span
        # A plain luminance ramp: any hue-coding would hide a band the eye can see.
        level = 40 + 200 * t
        colour = (int(level * 0.98), int(level), int(level * 0.86))
        draw.rectangle([px, py, px + cell - 2, py + cell - 2], fill=colour)
    return image


def isometric(x, z, h, scale=20.0, height_scale=1.0, width=1.0):
    cos30 = math.cos(math.radians(30))
    half = width / 2
    corners = [
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
    ]

    def project(px, pz, py):
        sx = (px - pz) * cos30
        sy = (px + pz) * 0.5 - py * height_scale
        return sx, sy

    rods = []
    for i in range(len(x)):
        base = [project(x[i] + cx, z[i] + cz, 0.0) for cx, cz in corners]
        top = [project(x[i] + cx, z[i] + cz, h[i]) for cx, cz in corners]
        rods.append((x[i] + z[i], base, top, float(h[i])))

    all_y = [p[1] for _, base, top, _ in rods for p in base + top]
    all_x = [p[0] for _, base, top, _ in rods for p in base + top]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    margin = 20
    size = (int((max_x - min_x) * scale) + 2 * margin, int((max_y - min_y) * scale) + 2 * margin)
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)

    def to_screen(p):
        return ((p[0] - min_x) * scale + margin, (p[1] - min_y) * scale + margin)

    high = max(float(h.max()), 1e-6)
    # Painter's algorithm: larger x+z projects lower on the screen, so it is nearer.
    for depth, base, top, height in sorted(rods, key=lambda rod: rod[0]):
        # Every side quad is (base_k, base_k+1, top_k+1, top_k); the two whose centres sit
        # lowest on the screen are the ones facing the camera. Picking them by geometry,
        # rather than by assuming a corner order, is what keeps the shading consistent.
        faces = []
        for k in range(4):
            next_k = (k + 1) % 4
            quad = [base[k], base[next_k], top[next_k], top[k]]
            faces.append((sum(p[1] for p in quad) / 4.0, quad))
        faces.sort(key=lambda face: face[0])
        for index, (_, quad) in enumerate(faces[-2:]):
            shade = SIDE_LIGHT if index else SIDE_DARK
            draw.polygon([to_screen(p) for p in quad], fill=tuple(int(v) for v in shade))
        lid = 0.62 + 0.38 * (height / high)
        draw.polygon([to_screen(p) for p in top], fill=tuple(int(v) for v in TOP * lid))
    return image


def main():
    if len(sys.argv) != 3:
        print("usage: python scripts/render-heights.py <dump.json> <out-prefix>")
        return 2
    x, z, h = load(sys.argv[1])
    prefix = sys.argv[2]
    top_down(x, z, h).save(f"{prefix}-top.png")
    # Render at 3x and downsample: the polygon edges are the only anti-aliasing we get.
    big = isometric(x, z, h, scale=60.0)
    big.resize((big.width // 3, big.height // 3), Image.LANCZOS).save(f"{prefix}-iso.png")
    print(f"{prefix}-top.png, {prefix}-iso.png  (heights {h.min():.2f}..{h.max():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
