// The one analysis interface the engine talks to, and the two implementations.
//
// v1 and v2 are not two flavours of the same thing — v1 reproduces `AnalyserNode`
// (byte spectrum, −100…−30 dB window, per-frame smoothing) while v2 reports
// calibrated physics (dBFS, Hz, Parseval-conserving band energy). What they share
// is the *shape* the reactor needs: rod geometry, rod heights, a 0..1 spectrum,
// per-band amounts and readouts. Keeping that shape in one interface is what lets
// the engine, the renderers and the panels stay unaware of which one is running.

import { Kernel } from './kernel';
import { KernelV2, REPORT_CENTROID_HZ, REPORT_PEAK_DB, REPORT_PEAK_HZ, REPORT_RMS_DB } from './kernel-v2';

export type KernelVersion = 'v1' | 'v2';

/** Where the spectrum is computed, shown in the panel so it is not a guess. */
export type AnalysisMode = 'worklet' | 'main';

export interface BackendReadout {
  /** RMS level of the block, dB. */
  levelDb: number | null;
  /** True sample peak, dBFS. The engine holds and decays this for the meter. */
  samplePeakDb: number | null;
  peakHz: number | null;
  centroidHz: number | null;
}

export interface AnalysisBackend {
  readonly version: KernelVersion;
  readonly analysis: AnalysisMode;
  /** Whether an audio-thread analyser can feed this backend at all. */
  readonly workletCapable: boolean;
  readonly rodCount: number;
  readonly stride: number;
  /** Transform size, and therefore the length of the block the analyser hands over. */
  readonly fftSize: number;
  /** Rod geometry, `stride` f32 per rod: x, z, radius, seed, channel. */
  readonly rods: Float32Array<ArrayBuffer>;
  /** Rod heights, one f32 per rod. */
  readonly heights: Float32Array<ArrayBuffer>;
  /** Display spectrum; multiply by `spectrumScale` for a 0..1 value. */
  readonly spectrum: ArrayLike<number>;
  readonly spectrumScale: number;
  readonly channels: number;
  setChannels(channels: number): void;
  /**
   * Drop the spectrum but keep the rods, so a resume does not flash and a stop can
   * ease the rods down instead of snapping them to the floor.
   *
   * This is the only "start over" the reactor needs. A kernel that also zeroed the
   * rod heights would collapse all 709 of them in one frame — the difference is
   * visible, so it is deliberately not part of the interface.
   */
  clearSpectrum(): void;
  setWorkletActive(active: boolean): void;
  /** Byte spectrum from the audio thread; false when this backend cannot use it. */
  ingestBins(bins: Uint8Array): boolean;
  /** Take one block from the analyser and run the analysis chain. */
  process(analyser: AnalyserNode, sampleRate: number): void;
  /** Advance one frame: rod easing and display smoothing. */
  step(dtFrames: number, sampleRate: number, gain: number, smoothing: number): void;
  readouts(analyser: AnalyserNode | null, sampleRate: number): BackendReadout;
  /** 0..1 loudness for the RGB halo. */
  energy(): number;
}

const NULL_READOUT: BackendReadout = {
  levelDb: null,
  samplePeakDb: null,
  peakHz: null,
  centroidHz: null,
};

/**
 * The byte spectrum is a decibel value mapped onto 0..255; these two constants
 * must match `MIN_DB` / `MAX_DB` in `wasm/src/lib.rs`. v2 has no equivalent —
 * it reports dBFS directly.
 */
const MIN_DB = -100;
const MAX_DB = -30;

/**
 * Byte -> linear power, precomputed once.
 *
 * Weighting the centroid by raw bytes is wrong: a silent high bin still reads
 * ~80/255, and there are far more high bins than low ones, so the noise floor
 * wins and the average lands around 8 kHz on bass-heavy music. Converting to
 * power first drops an empty bin to ~1e-8 and lets real content decide.
 */
const BIN_POWER = new Float32Array(256);
for (let byte = 0; byte < 256; byte++) {
  const decibels = MIN_DB + (byte / 255) * (MAX_DB - MIN_DB);
  BIN_POWER[byte] = Math.pow(10, decibels / 10);
}

/** v1: the `AnalyserNode`-compatible kernel. */
export class KernelV1Backend implements AnalysisBackend {
  readonly version = 'v1' as const;
  readonly workletCapable = true;
  readonly spectrumScale = 1 / 255;
  private worklet = false;

  constructor(private readonly kernel: Kernel) {}

  get analysis(): AnalysisMode {
    return this.worklet ? 'worklet' : 'main';
  }

  get rodCount(): number {
    return this.kernel.rodCount;
  }

  get stride(): number {
    return this.kernel.stride;
  }

  get fftSize(): number {
    return this.kernel.fftSize;
  }

  get rods(): Float32Array<ArrayBuffer> {
    return this.kernel.rods;
  }

  get heights(): Float32Array<ArrayBuffer> {
    return this.kernel.heights;
  }

  get spectrum(): ArrayLike<number> {
    return this.kernel.bins;
  }

  get channels(): number {
    return this.kernel.channels;
  }

  setChannels(channels: number): void {
    this.kernel.setChannels(channels);
  }

  clearSpectrum(): void {
    this.kernel.clearBins();
  }

  setWorkletActive(active: boolean): void {
    this.worklet = active;
  }

  ingestBins(bins: Uint8Array): boolean {
    this.kernel.bins.set(bins.subarray(0, this.kernel.bins.length));
    return true;
  }

  process(analyser: AnalyserNode): void {
    // With the worklet running the bins arrive by message; the main thread would
    // only repeat work the audio thread has already done.
    if (this.worklet) return;
    analyser.getFloatTimeDomainData(this.kernel.wave);
    this.kernel.analyse();
  }

  step(dtFrames: number, sampleRate: number, gain: number, smoothing: number): void {
    this.kernel.step(dtFrames, sampleRate, gain, smoothing);
  }

  readouts(analyser: AnalyserNode | null, sampleRate: number): BackendReadout {
    if (!analyser) return NULL_READOUT;
    // Level comes from the same time-domain block the kernel just analysed, so the
    // meter and the bars never disagree about what was measured.
    analyser.getFloatTimeDomainData(this.kernel.wave);
    const wave = this.kernel.wave;
    let energy = 0;
    let samplePeak = 0;
    for (const sample of wave) {
      energy += sample * sample;
      const magnitude = Math.abs(sample);
      if (magnitude > samplePeak) samplePeak = magnitude;
    }
    energy = Math.sqrt(energy / wave.length);
    const levelDb = Math.round(20 * Math.log10(Math.max(energy, 0.001)));
    const samplePeakDb = 20 * Math.log10(Math.max(samplePeak, 1e-4));

    const bins = this.kernel.bins;
    // Bin 0 is the DC term, never a pitch: skip it in both the peak search and the
    // centroid weighting.
    let peak = 0;
    let peakIndex = 1;
    let weight = 0;
    let weightedBin = 0;
    for (let i = 1; i < bins.length; i++) {
      const byte = bins[i];
      if (byte > peak) {
        peak = byte;
        peakIndex = i;
      }
      const power = BIN_POWER[byte];
      weight += power;
      weightedBin += power * i;
    }
    const binHz = sampleRate / this.kernel.fftSize;
    return {
      levelDb,
      samplePeakDb,
      peakHz: peak > 0 ? Math.round(peakIndex * binHz) : null,
      centroidHz: weight > 0 ? Math.round((weightedBin / weight) * binHz) : null,
    };
  }

  /**
   * 0..1 loudness driving the RGB halo. The byte spectrum is a decibel curve, so
   * a silent bin still reads ~80/255; subtracting that floor is what keeps the
   * glow from being permanently on.
   */
  energy(): number {
    const bins = this.kernel.bins;
    let peak = 0;
    for (let i = 0; i < bins.length; i += 4) {
      if (bins[i] > peak) peak = bins[i];
    }
    return Math.max(0, Math.min(1, (peak / 255 - 0.3) / 0.55));
  }
}

/** Below this the kernel reports its silence sentinel rather than a number. */
const SILENCE_DB = -143;

/**
 * v2: the asynchronous, calibrated kernel. Runs on the main thread only.
 *
 * Everything it exposes as *data* is a `Float64Array` view over wasm memory:
 * linear bin power, dB bins, bars, band RMS, band amounts and the readout report.
 * A JS number is an `f64` too, so there is no conversion step between the kernel
 * and the panel in which a digit could be lost. Only rod geometry and rod heights
 * are `f32`, because the GPU is what reads them.
 */
export class KernelV2Backend implements AnalysisBackend {
  readonly version = 'v2' as const;
  readonly analysis = 'main' as const;
  // v2 has no byte spectrum to accept, and its analysis is already asynchronous
  // inside wasm, so an audio-thread feed would buy nothing.
  readonly workletCapable = false;
  readonly spectrumScale = 1;

  constructor(private readonly kernel: KernelV2) {}

  get rodCount(): number {
    return this.kernel.rodCount;
  }

  get stride(): number {
    return this.kernel.stride;
  }

  get fftSize(): number {
    return this.kernel.fftSize;
  }

  get rods(): Float32Array<ArrayBuffer> {
    return this.kernel.rods;
  }

  get heights(): Float32Array<ArrayBuffer> {
    return this.kernel.heights;
  }

  get spectrum(): ArrayLike<number> {
    return this.kernel.bars;
  }

  get channels(): number {
    return this.kernel.channels;
  }

  setChannels(channels: number): void {
    this.kernel.setChannels(channels);
  }

  clearSpectrum(): void {
    this.kernel.clearSpectrum();
  }

  setWorkletActive(): void {
    /* v2 never runs on the audio thread */
  }

  ingestBins(): boolean {
    return false;
  }

  process(analyser: AnalyserNode, sampleRate: number): void {
    // The band layout is frequency-based, so it has to know the rate it is
    // measuring at; the kernel ignores a call that changes nothing.
    this.kernel.configure(sampleRate);
    this.kernel.pushFrom(analyser);
  }

  step(dtFrames: number, sampleRate: number, gain: number, smoothing: number): void {
    this.kernel.configure(sampleRate);
    // `dtFrames` counts 60 Hz frames by definition, so one frame is exactly 1/60 s
    // and the conversion needs no magic constant. The kernel takes seconds: display
    // smoothing and rod easing are both exponentials of real time, so a 120 Hz
    // screen behaves like a 60 Hz one.
    this.kernel.step(dtFrames / 60, gain, smoothing);
  }

  readouts(): BackendReadout {
    // Everything below was measured inside the block that was just analysed, so
    // no second read of the analyser is needed (or wanted: it would sample a
    // different block and disagree with the bars).
    //
    // No rounding happens here. These are the `f64` values the kernel wrote, read
    // straight out of its memory, and a JS number is an `f64`, so the value that
    // reaches the panel is the value that was computed. Formatting is the panel's
    // job, and it must not be confused with the measurement.
    const report = this.kernel.report;
    const level = report[REPORT_RMS_DB];
    const peak = report[REPORT_PEAK_DB];
    const peakHz = report[REPORT_PEAK_HZ];
    const centroidHz = report[REPORT_CENTROID_HZ];
    return {
      levelDb: level <= SILENCE_DB ? null : level,
      samplePeakDb: peak <= SILENCE_DB ? null : peak,
      peakHz: peakHz > 0 ? peakHz : null,
      centroidHz: centroidHz > 0 ? centroidHz : null,
    };
  }

  /**
   * 0..1 loudness for the halo. The bars are already calibrated 0..1, so unlike
   * v1 there is no noise floor to subtract — only the display curve.
   */
  energy(): number {
    const bars = this.kernel.bars;
    let peak = 0;
    for (let i = 0; i < bars.length; i += 4) {
      if (bars[i] > peak) peak = bars[i];
    }
    return Math.max(0, Math.min(1, (peak - 0.3) / 0.6));
  }
}

export { probeKernelV2 } from './kernel-v2';
