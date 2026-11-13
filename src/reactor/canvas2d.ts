// Canvas 2D backend: the original renderer, kept as the reference implementation.
//
// It draws the same scene as the GPU backends — same projection, same flowing
// palette, same clockwise wipe — but pays for it in the main thread: a painter's
// sort over 709 rods and five filled polygons each, every frame. That cost is
// exactly why the other two backends exist, so it is worth being able to compare
// them side by side (the 主线程耗时 readout usually shows the difference).

import type { Scene, View } from './draw';
import type { CoreRenderer, RodSource } from './renderer';

const TAU = Math.PI * 2;
const DEG = 180 / Math.PI;
/** Mirrors `FLOW_RATE` in the GPU backends: radians per millisecond. */
const FLOW_RATE = 0.0009;
const FRONT_WIDTH = 0.045;
const ROD_BASE = 0.07;
const HEIGHT_SCALE = 4.8;
const ROD_HALF = 0.44;
/** Ring radius, stroke width in CSS pixels, plain colour, hue offset in the RGB palette. */
const RINGS: Array<{ radius: number; width: number; plain: string; hue: number; light: number }> = [
  { radius: 16.4, width: 20, plain: '#060b07', hue: NaN, light: 0 },
  { radius: 16.5, width: 2, plain: '#505948', hue: 0, light: 0.55 },
  { radius: 16.0, width: 1, plain: '#878969', hue: 120, light: 0.62 },
  { radius: 17.3, width: 1, plain: '#303a2d', hue: 240, light: 0.4 },
];

/** Hue in degrees, saturation and lightness in 0..1, to an `rgb()` string. */
function hsl(hue: number, saturation: number, lightness: number): string {
  const h = (((hue % 360) + 360) % 360) / 360;
  const q = lightness < 0.5
    ? lightness * (1 + saturation)
    : lightness + saturation - lightness * saturation;
  const p = 2 * lightness - q;
  const channel = (t: number) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  const to255 = (value: number) => Math.round(Math.max(0, Math.min(1, value)) * 255);
  return `rgb(${to255(channel(h + 1 / 3))},${to255(channel(h))},${to255(channel(h - 1 / 3))})`;
}

export class Canvas2dRenderer implements CoreRenderer {
  readonly kind = 'canvas2d' as const;
  private readonly canvas: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D;
  private readonly source: RodSource;
  private readonly order: Uint16Array;

  constructor(canvas: HTMLCanvasElement, source: RodSource) {
    const context = canvas.getContext('2d');
    if (!context) throw new Error('这个浏览器没有可用的 Canvas 2D 上下文。');
    this.canvas = canvas;
    this.g = context;
    this.source = source;
    this.order = new Uint16Array(source.count);
  }

  resize(): void {
    // The engine sizes the backing store; this backend owns the DPR transform
    // because nobody else may ask the canvas for a context.
    const dpr = this.canvas.clientWidth > 0 ? this.canvas.width / this.canvas.clientWidth : 1;
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  dispose(): void {
    this.g.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawCore(width: number, height: number, view: View, scene: Scene): void {
    const g = this.g;
    if (width <= 0 || height <= 0) return;
    const scale = Math.min(width / 45, height / 37) * view.zoom;
    const cos = Math.cos(view.angle);
    const sin = Math.sin(view.angle);
    const flow = scene.phase * FLOW_RATE;

    g.clearRect(0, 0, width, height);

    if (scene.rgbOn || scene.rgbSweep < 1) this.halo(g, width, height, view, scene, scale, flow);

    const settled = scene.rgbOn && scene.rgbSweep >= 1;
    const hue = flow * DEG;
    RINGS.forEach((ring, index) => {
      const color = Number.isNaN(ring.hue) || !settled ? ring.plain : hsl(hue + ring.hue, 0.95, ring.light);
      this.ring(g, width, height, scale, cos, sin, view.tilt, ring.radius, ring.width, color);
      void index;
    });
    this.ticks(g, width, height, scale, cos, sin, view.tilt, view, scene, flow);

    this.rods(g, width, height, scale, cos, sin, view, scene, flow);
  }

  /** The oblique projection every backend shares: rotate, squash, lift. */
  private project(
    width: number,
    height: number,
    scale: number,
    cos: number,
    sin: number,
    tilt: number,
    x: number,
    z: number,
    elevation = 0,
  ): [number, number] {
    const rx = x * cos - z * sin;
    const rz = x * sin + z * cos;
    return [width / 2 + rx * scale, height * 0.55 + rz * scale * tilt - elevation * scale];
  }

  private ring(
    g: CanvasRenderingContext2D,
    width: number,
    height: number,
    scale: number,
    cos: number,
    sin: number,
    tilt: number,
    radius: number,
    lineWidth: number,
    color: string,
  ): void {
    g.beginPath();
    for (let i = 0; i <= 100; i++) {
      const a = (i / 100) * TAU;
      const [x, y] = this.project(width, height, scale, cos, sin, tilt, Math.cos(a) * radius, Math.sin(a) * radius);
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.strokeStyle = color;
    g.lineWidth = lineWidth;
    g.stroke();
  }

  private ticks(
    g: CanvasRenderingContext2D,
    width: number,
    height: number,
    scale: number,
    cos: number,
    sin: number,
    tilt: number,
    view: View,
    scene: Scene,
    flow: number,
  ): void {
    for (let i = 0; i < 80; i++) {
      const a = (i / 80) * TAU;
      const b = a + 0.034;
      const points: Array<[number, number]> = [
        this.project(width, height, scale, cos, sin, tilt, Math.cos(a) * 15.7, Math.sin(a) * 15.7),
        this.project(width, height, scale, cos, sin, tilt, Math.cos(b) * 15.7, Math.sin(b) * 15.7),
        this.project(width, height, scale, cos, sin, tilt, Math.cos(b) * 16.15, Math.sin(b) * 16.15),
        this.project(width, height, scale, cos, sin, tilt, Math.cos(a) * 16.15, Math.sin(a) * 16.15),
      ];
      // Same world-anchored front as the rods: subtract the view rotation.
      const sector = fract((a - view.angle) / TAU);
      let color = i % 2 ? '#484d31' : '#adac64';
      if (sectorLit(sector, scene)) {
        const front = frontStrength(sector, scene.rgbSweep);
        const comet = 0.5 + 0.5 * Math.sin(a * 3 - flow * 3);
        color = hsl(flowHue(a, 15.7, flow), Math.max(0, 0.95 - front * 0.55), (22 + comet * 48 + front * 26) / 100);
      }
      this.polygon(g, points, color);
    }
  }

  private rods(
    g: CanvasRenderingContext2D,
    width: number,
    height: number,
    scale: number,
    cos: number,
    sin: number,
    view: View,
    scene: Scene,
    flow: number,
  ): void {
    const { rods, strides } = { rods: this.source.rods, strides: this.source.stride };
    const depths = this.order.subarray(0, scene.heights.length);
    for (let i = 0; i < depths.length; i++) depths[i] = i;
    // Painter's algorithm: front-most rods (larger rotated z) are drawn last.
    depths.sort((a, b) => (rods[a * strides] * sin + rods[a * strides + 1] * cos)
      - (rods[b * strides] * sin + rods[b * strides + 1] * cos));

    const shades = [0.44, 0.38, 0.36, 0.4];
    const corners: Array<[number, number]> = [[-ROD_HALF, -ROD_HALF], [ROD_HALF, -ROD_HALF], [ROD_HALF, ROD_HALF], [-ROD_HALF, ROD_HALF]];

    for (const index of depths) {
      const offset = index * strides;
      const x = rods[offset];
      const z = rods[offset + 1];
      const radius = rods[offset + 2];
      const seed = rods[offset + 3];
      const rodHeight = scene.heights[index];
      const amp = Math.max(0, (rodHeight - ROD_BASE) / HEIGHT_SCALE);

      const sector = fract((Math.atan2(z, x) - view.angle) / TAU);
      const lit = sectorLit(sector, scene);
      const angle = Math.atan2(z, x);
      const front = lit ? frontStrength(sector, scene.rgbSweep) : 0;
      const comet = 0.5 + 0.5 * Math.sin(angle * 3 - flow * 3);

      const top = corners.map(([cx, cz]) =>
        this.project(width, height, scale, cos, sin, view.tilt, x + cx, z + cz, rodHeight));
      const base = corners.map(([cx, cz]) =>
        this.project(width, height, scale, cos, sin, view.tilt, x + cx, z + cz));

      const plain = seed > 0.92
        ? [143, 109, 81]
        : seed > 0.85
          ? [97, 118, 104]
          : [139 + seed * 29, 148 + seed * 22, 119 + seed * 21];
      const plainTop = `rgb(${Math.min(255, plain[0] + amp * 43) | 0},${Math.min(255, plain[1] + amp * 40) | 0},${Math.min(255, plain[2] + amp * 24) | 0})`;

      for (let i = 0; i < 4; i++) {
        const next = (i + 1) % 4;
        const shade = shades[i];
        const color = lit
          ? hsl(flowHue(angle, radius, flow), Math.max(0, 0.95 - front * 0.62), ((20 + comet * 10 + Math.min(10, amp * 16)) / 100) * shade + 0.04)
          : dim(plainTop, shade);
        this.polygon(g, [top[i], base[i], base[next], top[next]], color, '#141b12');
      }

      const topColor = lit
        ? hsl(flowHue(angle, radius, flow), Math.max(0, 0.95 - front * 0.62), (30 + comet * 34 + Math.min(18, amp * 30) + front * 26) / 100)
        : plainTop;
      this.polygon(g, top, topColor, '#232d20');
    }
  }

  private halo(
    g: CanvasRenderingContext2D,
    width: number,
    height: number,
    view: View,
    scene: Scene,
    scale: number,
    flow: number,
  ): void {
    const radius = 17.9 * scale;
    g.save();
    g.translate(width / 2, height * 0.55);
    g.scale(1, view.tilt);
    // Angular front in local (screen) space: the disc-space origin sits at the
    // view angle, so the wipe turns with the reactor exactly like the rods.
    const from = view.angle + (scene.rgbOn ? 0 : scene.rgbSweep * TAU);
    const to = view.angle + (scene.rgbOn ? scene.rgbSweep * TAU : TAU);
    const gradient = g.createConicGradient(view.angle + flow, 0, 0);
    for (let step = 0; step <= 12; step++) {
      const a = (step / 12) * TAU;
      gradient.addColorStop(step / 12, hsl(flow * DEG + (a - view.angle) * DEG + 90, 1, 0.58));
    }
    g.globalCompositeOperation = 'lighter';
    g.globalAlpha = Math.min(0.42, 0.1 + scene.energy * 0.45);
    g.beginPath();
    g.moveTo(0, 0);
    g.arc(0, 0, radius, from, to);
    g.closePath();
    g.fillStyle = gradient;
    g.fill();
    g.globalAlpha = Math.min(0.8, 0.22 + scene.energy * 0.6);
    g.lineWidth = Math.max(1.5, scale * 0.1);
    g.strokeStyle = gradient;
    g.beginPath();
    g.arc(0, 0, radius, from, to);
    g.stroke();

    const spot = 3.8 * scale;
    for (let i = 0; i < 3; i++) {
      const world = flow + (i / 3) * TAU;
      if (!sectorLit(fract(world / TAU), scene)) continue;
      const local = world + view.angle;
      const x = Math.cos(local) * 16.6 * scale;
      const y = Math.sin(local) * 16.6 * scale;
      const blob = g.createRadialGradient(x, y, 0, x, y, spot);
      blob.addColorStop(0, hsl(flow * DEG + world * DEG, 1, 0.82));
      blob.addColorStop(1, 'rgba(0,0,0,0)');
      g.globalAlpha = Math.min(0.85, 0.32 + scene.energy * 0.5);
      g.fillStyle = blob;
      g.beginPath();
      g.arc(x, y, spot, 0, TAU);
      g.fill();
    }
    g.restore();
    g.globalCompositeOperation = 'source-over';
    g.globalAlpha = 1;
  }

  private polygon(
    g: CanvasRenderingContext2D,
    points: Array<[number, number]>,
    fill: string,
    stroke?: string,
  ): void {
    g.beginPath();
    points.forEach(([x, y], i) => (i === 0 ? g.moveTo(x, y) : g.lineTo(x, y)));
    g.closePath();
    g.fillStyle = fill;
    g.fill();
    if (stroke) {
      g.strokeStyle = stroke;
      g.lineWidth = 0.55;
      g.stroke();
    }
  }
}

/** `rgb(r,g,b)` darkened by a factor, for the shaded sides of a rod. */
function dim(color: string, factor: number): string {
  const match = /rgb\((\d+),(\d+),(\d+)\)/.exec(color);
  if (!match) return color;
  const scale = (value: string) => Math.round(Number(value) * factor);
  return `rgb(${scale(match[1])},${scale(match[2])},${scale(match[3])})`;
}

function fract(value: number): number {
  return value - Math.floor(value);
}

function sectorLit(sector: number, scene: Scene): boolean {
  return scene.rgbOn ? sector <= scene.rgbSweep : sector > scene.rgbSweep;
}

function frontStrength(sector: number, sweep: number): number {
  if (sweep >= 1) return 0;
  const delta = Math.abs(sector - sweep);
  return Math.max(0, 1 - Math.min(delta, 1 - delta) / FRONT_WIDTH);
}

function flowHue(angle: number, radius: number, flow: number): number {
  return (angle * 2 + radius * 0.21 - flow * 2) * DEG;
}
