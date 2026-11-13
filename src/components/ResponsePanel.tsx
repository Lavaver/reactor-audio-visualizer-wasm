import type { EngineSnapshot, ReactorEngine } from '../reactor/engine';

interface ResponsePanelProps {
  engine: ReactorEngine;
  snapshot: EngineSnapshot;
}

export function ResponsePanel({ engine, snapshot }: ResponsePanelProps) {
  const channelsLabel = `${snapshot.autoChannels ? 'AUTO · ' : ''}${snapshot.channels} CH`;

  return (
    <section>
      <label className="caption">02 / 控制棒响应</label>

      <div className="slider-title" title="调整越大，控制棒振动的幅度越大，反之亦然。">
        <label htmlFor="gain">震动幅度</label>
        <output id="gainval">{snapshot.gain.toFixed(1)}×</output>
      </div>
      <input
        id="gain"
        type="range"
        min={0.3}
        max={3}
        step={0.1}
        value={snapshot.gain}
        onChange={event => engine.setGain(Number(event.target.value))}
      />

      <div className="slider-title" title="调整控制棒的运动平滑度以改善视觉体验。">
        <label htmlFor="smooth">运动平滑度</label>
        <output id="smoothval">{Math.round(snapshot.smoothing * 100)}%</output>
      </div>
      <input
        id="smooth"
        type="range"
        min={0}
        max={95}
        value={Math.round(snapshot.smoothing * 100)}
        onChange={event => engine.setSmoothing(Number(event.target.value) / 100)}
      />

      <div className="slider-title">
        <label htmlFor="volume">播放音量</label>
        <output id="volumeval">{Math.round(snapshot.volume * 100)}%</output>
      </div>
      <input
        id="volume"
        type="range"
        min={0}
        max={100}
        value={Math.round(snapshot.volume * 100)}
        onChange={event => engine.setVolume(Number(event.target.value) / 100)}
      />

      <div
        className="slider-title"
        title="频道越少，控制棒越能成环联动，反之亦然。"
      >
        <label htmlFor="channels">频谱频道数</label>
        <span className="subtle">
          <output id="channelsval">{channelsLabel}</output>
          <button
            id="autoch"
            type="button"
            className={`auto-ch${snapshot.autoChannels ? ' on' : ''}`}
            aria-pressed={snapshot.autoChannels}
            title="按当前可视尺寸自动调整频道数；拖动滑块会切回手动"
            onClick={() => engine.setAutoChannels(!snapshot.autoChannels)}
          >
            AUTO
          </button>
        </span>
      </div>
      <input
        id="channels"
        type="range"
        min={4}
        max={64}
        step={1}
        value={snapshot.channels}
        onChange={event => engine.setChannels(Number(event.target.value))}
      />

      <div className="legend">
        <span>
          <i className="low" />
          低频
        </span>
        <span>
          <i className="mid" />
          中频
        </span>
        <span>
          <i className="high" />
          高频
        </span>
      </div>
    </section>
  );
}
