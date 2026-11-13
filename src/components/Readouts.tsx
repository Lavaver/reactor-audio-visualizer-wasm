import type { RefObject } from 'react';
import type { AnalysisMode, KernelVersion } from '../wasm/backend';
import type { Readout } from '../reactor/engine';

interface ReadoutsProps {
  spectrumRef: RefObject<HTMLCanvasElement | null>;
  readout: Readout;
  analysis: AnalysisMode;
  /** Kernel actually running, not the last one clicked. */
  kernelVersion: KernelVersion;
  /** False when the v2 module was never built, which disables its button. */
  v2Available: boolean;
  /** Same for v3. */
  v3Available: boolean;
  onSelectKernel: (version: KernelVersion) => void;
}

/**
 * What each kernel does, rendered as a line of text under the switch. It used to be
 * a `title` tooltip; it is on the page now, so it can be read without hovering — and
 * it shows the *selected* kernel's line, since that is the one being described.
 *
 * v1 averages byte values inside each band, and its −100…−30 dB window leaves a
 * silent bin at roughly 73/255, so the noise floor rides along into every band: a
 * full spectrum with modest height.
 *
 * v2 keeps the calibrated measurement — `band_rms`, `bin_power`, the readouts, all
 * untouched — and puts a display curve *after* it: an auto-gain that lifts quiet
 * material, a spectral tilt so the outer rings move, and a fast-attack / slow-
 * release envelope. That is why v2 is now the more dramatic of the two while its
 * numbers stay the honest ones.
 */
const KERNEL_NOTES: Record<KernelVersion, string> = {
  v1: '按频带平均字节频谱，柱子主要在低频区起伏，读数是近似值',
  v2: '按真实 dBFS，更精准，并在其之上加了自动增益、频谱倾斜与快起慢落',
  v3: '低频用常数 Q 滤波器组、其余用短窗 FFT，交叉淡化后驱动柱子；读数仍只取 FFT 支路',
};

/**
 * v3's readouts and its rods do not come from the same path, and its low bands lag.
 *
 * Both facts are measurements, so both are stated on the panel rather than left for the
 * picture to imply: `rk3_report_ptr` is FFT-only, bands 0..12 are driven by the CQT, and
 * an impulse peaks 164 ms later in the lowest CQT band than in the FFT bands. Nothing in
 * the UI may suggest a single source or a single instant.
 */
function kernelCaveat(version: KernelVersion): string | null {
  if (version === 'v3') {
    return null;
  }
  return null;
}

/**
 * Frequencies above 1 kHz read better in kHz: "1.35 kHz" says more at a glance
 * than "1354 Hz". v2 hands the panel an `f64`, so it can afford more digits than
 * the integer v1 reports — and the extra digits are the sub-bin resolution the
 * interpolated peak earned.
 */
function formatFrequency(
  hertz: number | null,
  version: KernelVersion,
): { value: string; unit: 'Hz' | 'kHz' } {
  if (hertz === null) return { value: '—', unit: 'Hz' };
  if (hertz >= 1000) {
    return { value: (hertz / 1000).toFixed(version === 'v2' ? 3 : 2), unit: 'kHz' };
  }
  return { value: version === 'v2' ? hertz.toFixed(1) : String(Math.round(hertz)), unit: 'Hz' };
}

/**
 * Formatting, not measuring. The value that arrives here is whatever the kernel
 * produced — full `f64` for v2, integers for v1's byte-derived readouts — and the
 * only thing this function does is decide how many digits a panel this size can
 * usefully show.
 */
function formatDecibels(value: number | null, version: KernelVersion): string {
  if (value === null) return '—';
  return version === 'v2' ? value.toFixed(1) : String(Math.round(value));
}

/** Peak pitch: v2 shows the interpolated fraction, v1 only ever had whole bins. */
function formatHertz(value: number | null, version: KernelVersion): string {
  if (value === null) return '—';
  return version === 'v2' ? value.toFixed(2) : String(Math.round(value));
}

export function Readouts({ spectrumRef, readout, analysis, kernelVersion, v2Available, v3Available, onSelectKernel }: ReadoutsProps) {
  const onAudioThread = analysis === 'worklet';
  const averageFrequency = formatFrequency(readout.centroidHz, kernelVersion);
  const precise = kernelVersion === 'v2';
  return (
    <section className="readings">
      <label className="caption">
        03 / 实时读数
        <span
          id="analysis"
          className={`thread${onAudioThread ? ' on-thread' : ''}`}
          title={
            onAudioThread
              ? '频谱在 AudioWorklet 里由 WASM 计算，主线程只收 1024 字节结果'
              : '频谱在 WASM 里由主线程计算'
          }
        >
          {onAudioThread ? 'WASM · AUDIO THREAD' : 'WASM · MAIN THREAD'}
        </span>
      </label>
      <div className="kernel-switch">
        <span>核心版本</span>
        <div className="backend-switch triple" role="group" aria-label="处理核心版本" aria-describedby="kernel-note">
          {(['v1', 'v2', 'v3'] as KernelVersion[]).map(version => {
            const available = version === 'v1'
              || (version === 'v2' ? v2Available : v3Available);
            return (
              <button
                key={version}
                id={`kernel-${version}`}
                type="button"
                className={`backend${version === 'v2' ? ' vivid' : version === 'v3' ? ' fresh' : ''}${kernelVersion === version ? ' on' : ''}`}
                aria-pressed={kernelVersion === version}
                disabled={!available}
                // The description is on the page now, so a tooltip would only repeat
                // it. The one thing that is *not* visible is why v2 is unavailable,
                // since a disabled button can never be selected to show its note.
                title={available
                  ? undefined
                  : `${version} 内核尚未编译：运行 pnpm run build:wasm:${version} 后重新加载页面。当前继续使用 v1。`}
                onClick={() => onSelectKernel(version)}
              >
                {/* The label is its own span so the gradient can be clipped to text
                    on a plain inline box rather than on the button element: a form
                    control is the one place that clipping has ever misbehaved. */}
                <span>{version.toUpperCase()}</span>
              </button>
            );
          })}
        </div>
        {/* The note describes whichever kernel is selected, so it takes that
            kernel's colour: muted for v1, the vivid ink for v2. */}
        <p className={`kernel-note${kernelVersion === 'v2' ? ' vivid' : kernelVersion === 'v3' ? ' fresh' : ''}`} id="kernel-note">
          {KERNEL_NOTES[kernelVersion]}
        </p>
        {kernelCaveat(kernelVersion) ? (
          <p className="kernel-caveat" id="kernel-caveat">{kernelCaveat(kernelVersion)}</p>
        ) : null}
      </div>
      <div className="meters">
        <div title={precise
          ? '当前时间窗的均方根电平。内核以 f64 计算并原样交付，这里只为排版保留 1 位小数。'
          : '当前时间窗的均方根电平'}>
          <span>输入电平</span>
          <strong id="level">
            {formatDecibels(readout.levelDb, kernelVersion)}
            <small>dB</small>
          </strong>
        </div>
        <div title="本次窗口内的采样峰值保持（dBFS），响度上来时抬升，随后缓慢回落">
          <span>最高电平</span>
          <strong id="peaklevel">
            {formatDecibels(readout.peakLevelDb, kernelVersion)}
            <small>dB</small>
          </strong>
        </div>
        <div title={precise
          ? '振幅最大的频点，经抛物线插值细化到 bin 以下；数值以 f64 交付，显示保留 2 位小数。'
          : '振幅最大的单个频点'}>
          <span>峰值频率</span>
          <strong id="freq">
            {formatHertz(readout.peakHz, kernelVersion)}
            <small>Hz</small>
          </strong>
        </div>
        <div title="功率加权后的频谱质心。
由于此项一般较大，将自动进行单位转换。">
          <span>频谱质心</span>
          <strong id="avgfreq">
            {averageFrequency.value}
            <small>{averageFrequency.unit}</small>
          </strong>
        </div>
        {/* v3 only: the constant-Q path's own pitch, next to the FFT-derived one, so the
            two sources of "a frequency" are both on screen and both labelled. */}
        {kernelVersion === 'v3' ? (
          <div title="常数 Q 滤波器组给出的音高，与上方 FFT 支路的峰值频率来自不同支路；band 0–12 的柱高也来自这一路">
            <span>CQT 音高</span>
            <strong id="cqthz">
              {readout.cqtHz == null ? '—' : readout.cqtHz.toFixed(2)}
              <small>Hz</small>
            </strong>
          </div>
        ) : null}
        <div className="spectrum-cell">
          <span>实时频谱</span>
          <canvas id="spectrum" ref={spectrumRef} aria-label="实时频谱" />
        </div>
      </div>
    </section>
  );
}
