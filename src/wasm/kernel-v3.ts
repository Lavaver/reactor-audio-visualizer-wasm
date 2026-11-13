// Loads the v3 hybrid kernel and keeps every view into its linear memory.
//
// Same rules as v2: the module is optional (fetched from `public/` and validated by its
// wasm magic number, because a static import would fail the build when it was never
// compiled), and every view is rebuilt whenever the module's memory grew — `rk3_push`
// hands a block to a channel, which allocates.

export const KERNEL_V3_FILE = 'reactor_kernel_v3.wasm';
export const KERNEL_V3_URL = `${import.meta.env.BASE_URL}${KERNEL_V3_FILE}`;
const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d];

interface KernelV3Exports {
  memory: WebAssembly.Memory;
  rk3_version(): number;
  rk3_init(): number;
  rk3_configure(fftSize: number, sampleRate: number): void;
  rk3_reset(): void;
  rk3_clear_spectrum(): void;
  rk3_push(): number;
  rk3_step(dtSeconds: number, gain: number, smoothing: number): void;
  rk3_set_channels(channels: number): void;
  rk3_channels(): number;
  rk3_fft_size(): number;
  rk3_bin_count(): number;
  rk3_rod_count(): number;
  rk3_rod_stride(): number;
  rk3_rod_ptr(): number;
  rk3_heights_ptr(): number;
  rk3_wave_ptr(): number;
  rk3_bin_power_ptr(): number;
  rk3_bin_db_ptr(): number;
  rk3_window_ptr(): number;
  rk3_bars_ptr(): number;
  rk3_report_ptr(): number;
  rk3_channel_amps_ptr(): number;
  rk3_band_drive_ptr(): number;
  rk3_band_rms_ptr(): number;
  rk3_band_rms_fft_ptr(): number;
  rk3_band_rms_cqt_ptr(): number;
  rk3_mix_weight_ptr(): number;
  rk3_cqt_bins(): number;
  rk3_cqt_first_band(): number;
  rk3_cqt_freq_ptr(): number;
  rk3_cqt_power_ptr(): number;
  rk3_cqt_db_ptr(): number;
  rk3_cqt_report_ptr(): number;
  rk3_sine_ref(): number;
  rk3_processed(): number;
  rk3_dropped(): number;
  rk3_cqt_skipped(): number;
}

/** Report words behind `rk3_report_ptr`: the FFT path only, same as v2's estimators. */
export const REPORT_PEAK_HZ = 0;
export const REPORT_CENTROID_HZ = 1;
export const REPORT_RMS_DB = 2;
export const REPORT_PEAK_DB = 3;
/** Report words behind `rk3_cqt_report_ptr`: [peak_hz, peak_rms, bin_index, bins]. */
export const CQT_REPORT_PEAK_HZ = 0;

let pending: Promise<ArrayBuffer | null> | null = null;

/** Fetch the v3 module once and report whether it can be used. */
export function probeKernelV3(): Promise<ArrayBuffer | null> {
  if (!pending) {
    pending = (async () => {
      try {
        const response = await fetch(KERNEL_V3_URL);
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

export class KernelV3 {
  readonly version = 'v3' as const;
  readonly rodCount: number;
  readonly stride: number;
  readonly fftSize: number;
  readonly binCount: number;
  readonly sineRef: number;
  /** How many bands the CQT runs, and the first band it covers. */
  private readonly cqtBands: number;
  private readonly cqtFirstBand: number;

  private views: Views | null = null;
  private buffer: ArrayBuffer | null = null;
  private sampleRate = -1;

  private constructor(private readonly exports: KernelV3Exports) {
    this.rodCount = exports.rk3_rod_count();
    this.stride = exports.rk3_rod_stride();
    this.fftSize = exports.rk3_fft_size();
    this.binCount = exports.rk3_bin_count();
    this.sineRef = exports.rk3_sine_ref();
    this.cqtBands = exports.rk3_cqt_bins();
    this.cqtFirstBand = exports.rk3_cqt_first_band();
  }

  static async load(): Promise<KernelV3 | null> {
    const bytes = await probeKernelV3();
    if (!bytes) return null;
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exports = instance.exports as unknown as KernelV3Exports;
    if (exports.rk3_version() !== 3) return null;
    if (exports.rk3_init() !== 709) return null;
    // 2048 explicitly: `rk3_fft_size()` is 0 until the first configure, because the whole
    // kernel state starts zero-filled so it can live in .bss instead of the wasm file.
    exports.rk3_configure(2048, 48_000);
    return new KernelV3(exports);
  }

  /** Rebuild the views whenever the module's memory grew. */
  private view(): Views {
    const buffer = this.exports.memory.buffer;
    if (this.views && this.buffer === buffer) return this.views;
    this.buffer = buffer;
    const x = this.exports;
    this.views = {
      // The two f32 arrays are the GPU's.
      rods: new Float32Array(buffer, x.rk3_rod_ptr(), this.rodCount * this.stride),
      heights: new Float32Array(buffer, x.rk3_heights_ptr(), this.rodCount),
      wave: new Float32Array(buffer, x.rk3_wave_ptr(), this.fftSize),
      // Everything the panel or a test reads as data is f64.
      binPower: new Float64Array(buffer, x.rk3_bin_power_ptr(), this.binCount),
      binDb: new Float64Array(buffer, x.rk3_bin_db_ptr(), this.binCount),
      bars: new Float64Array(buffer, x.rk3_bars_ptr(), this.binCount),
      window: new Float64Array(buffer, x.rk3_window_ptr(), this.fftSize),
      bandRms: new Float64Array(buffer, x.rk3_band_rms_ptr(), 64),
      bandRmsFft: new Float64Array(buffer, x.rk3_band_rms_fft_ptr(), 64),
      bandRmsCqt: new Float64Array(buffer, x.rk3_band_rms_cqt_ptr(), 64),
      mixWeight: new Float64Array(buffer, x.rk3_mix_weight_ptr(), 64),
      cqtPower: new Float64Array(buffer, x.rk3_cqt_power_ptr(), 64),
      cqtFreq: new Float64Array(buffer, x.rk3_cqt_freq_ptr(), 64),
      amps: new Float64Array(buffer, x.rk3_channel_amps_ptr(), 64),
      drive: new Float64Array(buffer, x.rk3_band_drive_ptr(), 64),
      report: new Float64Array(buffer, x.rk3_report_ptr(), 4),
      cqtReport: new Float64Array(buffer, x.rk3_cqt_report_ptr(), 4),
    };
    return this.views;
  }

  get rods(): Float32Array<ArrayBuffer> {
    return this.view().rods;
  }

  get heights(): Float32Array<ArrayBuffer> {
    return this.view().heights;
  }

  get bars(): Float64Array<ArrayBuffer> {
    return this.view().bars;
  }

  get binPower(): Float64Array<ArrayBuffer> {
    return this.view().binPower;
  }

  get bandRms(): Float64Array<ArrayBuffer> {
    return this.view().bandRms;
  }

  get bandRmsFft(): Float64Array<ArrayBuffer> {
    return this.view().bandRmsFft;
  }

  get bandRmsCqt(): Float64Array<ArrayBuffer> {
    return this.view().bandRmsCqt;
  }

  get mixWeight(): Float64Array<ArrayBuffer> {
    return this.view().mixWeight;
  }

  get cqtPower(): Float64Array<ArrayBuffer> {
    return this.view().cqtPower;
  }

  get cqtFreq(): Float64Array<ArrayBuffer> {
    return this.view().cqtFreq;
  }

  get channelAmps(): Float64Array<ArrayBuffer> {
    return this.view().amps;
  }

  get report(): Float64Array<ArrayBuffer> {
    return this.view().report;
  }

  get cqtReport(): Float64Array<ArrayBuffer> {
    return this.view().cqtReport;
  }

  /** Bands driven by the CQT, for the panel's source label. */
  get cqtRange(): { first: number; count: number } {
    return { first: this.cqtFirstBand, count: this.cqtBands };
  }

  get channels(): number {
    return this.exports.rk3_channels();
  }

  setChannels(channels: number): void {
    this.exports.rk3_set_channels(channels);
  }

  configure(sampleRate: number): void {
    if (sampleRate === this.sampleRate) return;
    this.sampleRate = sampleRate;
    this.exports.rk3_configure(this.fftSize, sampleRate);
  }

  /** Read one block straight into wasm memory and let the pipeline run. */
  pushFrom(analyser: AnalyserNode): number {
    const views = this.view();
    if (analyser.fftSize !== this.fftSize) analyser.fftSize = this.fftSize;
    analyser.getFloatTimeDomainData(views.wave);
    return this.exports.rk3_push();
  }

  step(dtSeconds: number, gain: number, smoothing: number): void {
    this.exports.rk3_step(dtSeconds, gain, smoothing);
  }

  reset(): void {
    this.exports.rk3_reset();
  }

  clearSpectrum(): void {
    this.exports.rk3_clear_spectrum();
  }

  get counters(): { processed: number; dropped: number; cqtSkipped: number } {
    return {
      processed: this.exports.rk3_processed(),
      dropped: this.exports.rk3_dropped(),
      cqtSkipped: this.exports.rk3_cqt_skipped(),
    };
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
  bandRmsFft: Float64Array<ArrayBuffer>;
  bandRmsCqt: Float64Array<ArrayBuffer>;
  mixWeight: Float64Array<ArrayBuffer>;
  cqtPower: Float64Array<ArrayBuffer>;
  cqtFreq: Float64Array<ArrayBuffer>;
  amps: Float64Array<ArrayBuffer>;
  drive: Float64Array<ArrayBuffer>;
  report: Float64Array<ArrayBuffer>;
  cqtReport: Float64Array<ArrayBuffer>;
}
