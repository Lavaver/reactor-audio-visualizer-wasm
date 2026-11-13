// Loads the v2 kernel and keeps every view into its linear memory.
//
// v2 is *optional*: the module only exists once someone ran `pnpm run build:wasm:v2`,
// and the page has to keep working with v1 when it does not. That rules out a
// static `import ... ?url`, which would fail the build for a missing file, so the
// module is fetched from `public/` and the wasm magic number is what decides
// whether it is really there.

/** Served verbatim from `public/`, like the analysis worklet. */
export const KERNEL_V2_FILE = 'reactor_kernel_v2.wasm';
export const KERNEL_V2_URL = `${import.meta.env.BASE_URL}${KERNEL_V2_FILE}`;
/** `\0asm`, the four bytes every wasm module starts with. */
const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d];

interface KernelV2Exports {
  memory: WebAssembly.Memory;
  rk2_version(): number;
  rk2_init(): number;
  rk2_configure(fftSize: number, sampleRate: number): void;
  rk2_reset(): void;
  rk2_clear_spectrum(): void;
  rk2_push(): number;
  rk2_step(dtSeconds: number, gain: number, smoothing: number): void;
  rk2_set_channels(channels: number): void;
  rk2_channels(): number;
  rk2_fft_size(): number;
  rk2_bin_count(): number;
  rk2_rod_count(): number;
  rk2_rod_stride(): number;
  rk2_rod_ptr(): number;
  rk2_heights_ptr(): number;
  rk2_channel_amps_ptr(): number;
  rk2_band_rms_ptr(): number;
  rk2_wave_ptr(): number;
  rk2_bin_power_ptr(): number;
  rk2_window_ptr(): number;
  rk2_bin_db_ptr(): number;
  rk2_bars_ptr(): number;
  rk2_report_ptr(): number;
  rk2_sine_ref(): number;
  rk2_processed(): number;
  rk2_dropped(): number;
}

/** Report words behind `rk2_report_ptr`, in the order the kernel writes them. */
export const REPORT_PEAK_HZ = 0;
export const REPORT_CENTROID_HZ = 1;
export const REPORT_RMS_DB = 2;
export const REPORT_PEAK_DB = 3;

let pending: Promise<ArrayBuffer | null> | null = null;

/**
 * Fetch the v2 module once and report whether it can be used.
 *
 * A missing file is not an error: `vite preview` and most static servers answer
 * an unknown path with the SPA's `index.html` and a 200, so the response has to
 * be checked for the wasm magic number rather than trusted on `response.ok`.
 */
export function probeKernelV2(): Promise<ArrayBuffer | null> {
  if (!pending) {
    pending = (async () => {
      try {
        const response = await fetch(KERNEL_V2_URL);
        if (!response.ok) return null;
        const bytes = await response.arrayBuffer();
        if (bytes.byteLength < 8) return null;
        const head = new Uint8Array(bytes, 0, WASM_MAGIC.length);
        return WASM_MAGIC.every((value, index) => head[index] === value) ? bytes : null;
      } catch {
        return null;
      }
    })();
  }
  return pending;
}

export class KernelV2 {
  readonly version = 'v2' as const;
  readonly rodCount: number;
  readonly stride: number;
  readonly fftSize: number;
  readonly binCount: number;
  /** Peak bin power of a full-scale sine: the reference the bin dB values use. */
  readonly sineRef: number;

  private views: Views | null = null;
  /** Buffer identity the current views belong to; a growth invalidates them. */
  private buffer: ArrayBuffer | null = null;
  private sampleRate = -1;

  private constructor(private readonly exports: KernelV2Exports) {
    this.rodCount = exports.rk2_rod_count();
    this.stride = exports.rk2_rod_stride();
    this.fftSize = exports.rk2_fft_size();
    this.binCount = exports.rk2_bin_count();
    this.sineRef = exports.rk2_sine_ref();
  }

  /**
   * Instantiate v2, or return null when it was never built.
   *
   * The version export is checked as well: a stale file from an older layout
   * would otherwise load and then misbehave in ways that look like a maths bug.
   */
  static async load(): Promise<KernelV2 | null> {
    const bytes = await probeKernelV2();
    if (!bytes) return null;
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exports = instance.exports as unknown as KernelV2Exports;
    if (exports.rk2_version() !== 2) return null;
    if (exports.rk2_init() !== 709) return null;
    exports.rk2_configure(exports.rk2_fft_size(), 48_000);
    return new KernelV2(exports);
  }

  /**
   * Views are rebuilt whenever the module's memory grew.
   *
   * `rk2_push` hands a block to a channel, which allocates; an allocator that
   * grows the heap replaces `memory.buffer` and detaches every existing view. A
   * cached `Float32Array` would then silently read as empty, so every access goes
   * through here instead.
   */
  private view(): Views {
    const buffer = this.exports.memory.buffer;
    if (this.views && this.buffer === buffer) return this.views;
    this.buffer = buffer;
    this.views = {
      // The two f32 arrays are the GPU's: geometry and rod heights.
      rods: new Float32Array(buffer, this.exports.rk2_rod_ptr(), this.rodCount * this.stride),
      heights: new Float32Array(buffer, this.exports.rk2_heights_ptr(), this.rodCount),
      wave: new Float32Array(buffer, this.exports.rk2_wave_ptr(), this.fftSize),
      // Everything that is *data* is f64, read in place.
      binPower: new Float64Array(buffer, this.exports.rk2_bin_power_ptr(), this.binCount),
      binDb: new Float64Array(buffer, this.exports.rk2_bin_db_ptr(), this.binCount),
      bars: new Float64Array(buffer, this.exports.rk2_bars_ptr(), this.binCount),
      window: new Float64Array(buffer, this.exports.rk2_window_ptr(), this.fftSize),
      bandRms: new Float64Array(buffer, this.exports.rk2_band_rms_ptr(), 64),
      amps: new Float64Array(buffer, this.exports.rk2_channel_amps_ptr(), 64),
      report: new Float64Array(buffer, this.exports.rk2_report_ptr(), 4),
    };
    return this.views;
  }

  get rods(): Float32Array<ArrayBuffer> {
    return this.view().rods;
  }

  /** Rod heights, `f32`: uploaded to the GPU, where the 7th decimal is invisible. */
  get heights(): Float32Array<ArrayBuffer> {
    return this.view().heights;
  }

  /** Linear power per bin, `f64` — the raw measurement, untouched. */
  get binPower(): Float64Array<ArrayBuffer> {
    return this.view().binPower;
  }

  /** Bin power in dB relative to a full-scale on-bin sine, `f64`. */
  get binDb(): Float64Array<ArrayBuffer> {
    return this.view().binDb;
  }

  /** Smoothed 0..1 bar values, one per bin, `f64`. */
  get bars(): Float64Array<ArrayBuffer> {
    return this.view().bars;
  }

  /** The window in use, `f64`; exposed for exactness checks. */
  get window(): Float64Array<ArrayBuffer> {
    return this.view().window;
  }

  /** Linear RMS per band, `f64`. */
  get bandRms(): Float64Array<ArrayBuffer> {
    return this.view().bandRms;
  }

  /** Per-band 0..1 amounts the rods react to, `f64`. */
  get channelAmps(): Float64Array<ArrayBuffer> {
    return this.view().amps;
  }

  /** `[peak_hz, centroid_hz, rms_db, peak_db]`, all `f64`. */
  get report(): Float64Array<ArrayBuffer> {
    return this.view().report;
  }

  get channels(): number {
    return this.exports.rk2_channels();
  }

  setChannels(channels: number): void {
    this.exports.rk2_set_channels(channels);
  }

  /**
   * (Re)tune the transform and the band layout for a sample rate.
   *
   * The kernel ignores a call that does not change anything, so this is safe to
   * make defensive; only a real change drops the pipeline.
   */
  configure(sampleRate: number): void {
    if (sampleRate === this.sampleRate) return;
    this.sampleRate = sampleRate;
    this.exports.rk2_configure(this.fftSize, sampleRate);
  }

  /**
   * Read one block straight into wasm memory and let the asynchronous stage run.
   *
   * The analyser writes into the staging buffer directly, so a block costs no copy
   * on this side; `rk2_push` then moves it through the channel.
   */
  pushFrom(analyser: AnalyserNode): number {
    const views = this.view();
    // `getFloatTimeDomainData` needs an array at least as long as the analyser's
    // fftSize, and throws otherwise. The engine keeps the two in step; this only
    // repairs the case where a stale analyser survived a backend switch.
    if (analyser.fftSize !== this.fftSize) analyser.fftSize = this.fftSize;
    analyser.getFloatTimeDomainData(views.wave);
    return this.exports.rk2_push();
  }

  step(dtSeconds: number, gain: number, smoothing: number): void {
    this.exports.rk2_step(dtSeconds, gain, smoothing);
  }

  reset(): void {
    this.exports.rk2_reset();
  }

  clearSpectrum(): void {
    this.exports.rk2_clear_spectrum();
  }

  /** Blocks the asynchronous stage has consumed; `dropped` counts backpressure. */
  get counters(): { processed: number; dropped: number } {
    return { processed: this.exports.rk2_processed(), dropped: this.exports.rk2_dropped() };
  }
}

interface Views {
  rods: Float32Array<ArrayBuffer>;
  heights: Float32Array<ArrayBuffer>;
  wave: Float32Array<ArrayBuffer>;
  binPower: Float64Array<ArrayBuffer>;
  binDb: Float64Array<ArrayBuffer>;
  bars: Float64Array<ArrayBuffer>;
  window: Float64Array<ArrayBuffer>;
  bandRms: Float64Array<ArrayBuffer>;
  amps: Float64Array<ArrayBuffer>;
  report: Float64Array<ArrayBuffer>;
}
