import type { RefObject } from 'react';
import type { AnalysisMode, Readout } from '../reactor/engine';

interface ReadoutsProps {
  spectrumRef: RefObject<HTMLCanvasElement | null>;
  readout: Readout;
  analysis: AnalysisMode;
}

/**
 * Frequencies above 1 kHz read better in kHz: "1.35 kHz" says more at a glance
 * than "1354 Hz". Two decimals keep roughly 10 Hz of resolution.
 */
function formatFrequency(hertz: number | null): { value: string; unit: 'Hz' | 'kHz' } {
  if (hertz === null) return { value: '—', unit: 'Hz' };
  return hertz >= 1000
    ? { value: (hertz / 1000).toFixed(2), unit: 'kHz' }
    : { value: String(hertz), unit: 'Hz' };
}

export function Readouts({ spectrumRef, readout, analysis }: ReadoutsProps) {
  const onAudioThread = analysis === 'worklet';
  const averageFrequency = formatFrequency(readout.centroidHz);
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
              : 'AudioWorklet 不可用，频谱在 WASM 中由主线程计算'
          }
        >
          {onAudioThread ? 'WASM · AUDIO THREAD' : 'WASM · MAIN THREAD'}
        </span>
      </label>
      <div className="meters">
        <div title="当前时间窗的均方根电平">
          <span>输入电平</span>
          <strong id="level">
            {readout.levelDb ?? '—'}
            <small>dB</small>
          </strong>
        </div>
        <div title="本次窗口内的采样峰值保持（dBFS），响度上来时抬升，随后缓慢回落">
          <span>最高电平</span>
          <strong id="peaklevel">
            {readout.peakLevelDb ?? '—'}
            <small>dB</small>
          </strong>
        </div>
        <div title="振幅最大的单个频点">
          <span>峰值频率</span>
          <strong id="freq">
            {readout.peakHz ?? '—'}
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
        <div className="spectrum-cell">
          <span>实时频谱</span>
          <canvas id="spectrum" ref={spectrumRef} aria-label="实时频谱" />
        </div>
      </div>
    </section>
  );
}
