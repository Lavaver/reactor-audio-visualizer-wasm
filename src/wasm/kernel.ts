// Loads the wasm kernel and keeps every view into its linear memory in one
// place. The module never allocates after instantiation, so these typed arrays
// stay valid for the lifetime of the page.
//
// The kernel owns the analysis chain: the browser copies time-domain samples
// into `wave`, calls `analyse()`, and only then reads `bins`. Nothing here
// allocates per frame, and no data is copied back out of wasm.
import wasmUrl from '../generated/reactor_kernel.wasm?url';

/** Transform size the app uses; also the length of the time-domain block. */
export const FFT_SIZE = 2048;

interface KernelExports {
  memory: WebAssembly.Memory;
  rk_init(): number;
  rk_rod_count(): number;
  rk_rod_stride(): number;
  rk_set_channels(channels: number): void;
  rk_channels(): number;
  rk_set_fft(fftSize: number): void;
  rk_fft_size(): number;
  rk_bin_count(): number;
  rk_bins_ptr(): number;
  rk_wave_ptr(): number;
  rk_analyse(): void;
  rk_heights_ptr(): number;
  rk_rod_x_ptr(): number;
  rk_channel_amps_ptr(): number;
  rk_step(dtFrames: number, sampleRate: number, gain: number, smoothing: number): void;
}

export class Kernel {
  /** Rod heights, one f32 per rod. */
  readonly heights: Float32Array<ArrayBuffer>;
  /** Rod geometry, `stride` f32 per rod: x, z, radius, seed, channel. */
  readonly rods: Float32Array<ArrayBuffer>;
  /** Time-domain input: the browser writes one block here each frame. */
  readonly wave: Float32Array<ArrayBuffer>;
  /** Byte spectrum produced by `analyse()`; the bars read the same array. */
  readonly bins: Uint8Array<ArrayBuffer>;
  /** Per-channel energy, 0..1. */
  readonly channelAmps: Float32Array<ArrayBuffer>;
  readonly rodCount: number;
  readonly stride: number;
  readonly fftSize: number;

  private constructor(private readonly exports: KernelExports) {
    const memory = exports.memory;
    this.rodCount = exports.rk_rod_count();
    this.stride = exports.rk_rod_stride();
    this.fftSize = exports.rk_fft_size();
    this.rods = new Float32Array(memory.buffer, exports.rk_rod_x_ptr(), this.rodCount * this.stride);
    this.heights = new Float32Array(memory.buffer, exports.rk_heights_ptr(), this.rodCount);
    this.wave = new Float32Array(memory.buffer, exports.rk_wave_ptr(), this.fftSize);
    this.bins = new Uint8Array(memory.buffer, exports.rk_bins_ptr(), exports.rk_bin_count());
    this.channelAmps = new Float32Array(memory.buffer, exports.rk_channel_amps_ptr(), 64);
  }

  static async load(): Promise<Kernel> {
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`无法加载 WASM 内核（HTTP ${response.status}）`);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exports = instance.exports as unknown as KernelExports;
    exports.rk_init();
    // Fix the transform size *before* the views are created, so `wave` and
    // `bins` have the right lengths from the start.
    exports.rk_set_fft(FFT_SIZE);
    return new Kernel(exports);
  }

  get channels(): number {
    return this.exports.rk_channels();
  }

  setChannels(channels: number): void {
    this.exports.rk_set_channels(channels);
  }

  /** Window + FFT + smoothing + decibel mapping, all inside wasm. */
  analyse(): void {
    this.exports.rk_analyse();
  }

  step(dtFrames: number, sampleRate: number, gain: number, smoothing: number): void {
    this.exports.rk_step(dtFrames, sampleRate, gain, smoothing);
  }

  clearBins(): void {
    this.bins.fill(0);
  }
}
