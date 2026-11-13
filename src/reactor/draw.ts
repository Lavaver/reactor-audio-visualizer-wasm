// Scene description shared by the WebGL backend, plus the mini spectrum.
//
// The core scene lives in `gl.ts` now; what stays here is the data contract
// between the engine and any renderer, and the 48-bar spectrum, which is a
// second, tiny canvas where 2D rectangles are still the simplest thing that
// works.

export interface View {
  angle: number;
  tilt: number;
  zoom: number;
}

export interface Scene {
  /** Eased rod heights, straight out of the wasm kernel's memory. */
  heights: Float32Array;
  /** Idle "preview" animation phase in milliseconds, used before any audio. */
  phase: number;
  /**
   * Where the RGB palette is heading. It never lands in one go: the light
   * arrives (or leaves) as a clockwise wipe, so this is the target and
   * `rgbSweep` is how far the front has travelled.
   */
  rgbOn: boolean;
  /** 0..1 clockwise wipe progress; 1 means no transition is left to draw. */
  rgbSweep: number;
  /** 0..1 loudness that brightens the halo; ignored while the palette is off. */
  energy: number;
}

/**
 * 48-bar mini spectrum. Reads whatever spectrum the active kernel publishes:
 * v1 exposes a byte spectrum, v2 a calibrated 0..1 array, so `scale` says how to
 * turn one entry into a 0..1 bar height.
 */
export function drawSpectrum(
  g: CanvasRenderingContext2D,
  width: number,
  height: number,
  bins: ArrayLike<number>,
  scale: number,
  playing: boolean,
): void {
  g.clearRect(0, 0, width, height);
  if (!bins.length || width <= 0 || height <= 0) return;

  const count = 48;
  const slot = width / count;
  // A fixed 3px gap is fine on a wide panel and fatal on a narrow one: at 136px
  // the bar width would come out negative and `fillRect` would draw hairlines
  // instead of bars. Scale the gap with the slot and keep a floor under the bar.
  const gap = Math.min(3, Math.max(0, slot * 0.3));
  const barWidth = Math.max(0.75, slot - gap);

  for (let i = 0; i < count; i++) {
    const index = Math.min(bins.length - 1, Math.floor(Math.pow(i / (count - 1), 2) * 700));
    const value = playing ? Math.max(0, Math.min(1, bins[index] * scale)) : 0;
    g.fillStyle = i < 12 ? '#d8b975' : i < 30 ? '#c3cf87' : '#95b1a1';
    const barHeight = Math.max(2, value * height);
    g.fillRect(i * slot, height - barHeight, barWidth, barHeight);
  }
}
