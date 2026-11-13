import type { Readout } from '../reactor/engine';
import { RENDERER_LABELS, RENDERER_TITLES, type RendererKind } from '../reactor/renderer';

const ORDER: RendererKind[] = ['canvas2d', 'webgl', 'webgpu'];

interface PerfPanelProps {
  readout: Readout;
  /** Backend actually drawing; the buttons follow this, not the last click. */
  renderer: RendererKind;
  /** Set when the requested backend could not be used. */
  note: string | null;
  availability: Record<RendererKind, boolean>;
  onSelect: (kind: RendererKind) => void;
}

export function PerfPanel({ readout, renderer, note, availability, onSelect }: PerfPanelProps) {
  return (
    <section>
      <label className="caption">04 / 性能与渲染</label>

      <div className="perf-grid">
        <div title="本页动画回调平均每帧占用的主线程时间。60 fps 的预算是 16.7 ms，超过它就会掉帧。换后端时这一格变化最明显。">
          <span>主线程耗时</span>
          <strong id="frametime">
            {readout.frameMs === null ? '—' : readout.frameMs.toFixed(1)}
            <small>ms</small>
          </strong>
        </div>
        <div title="最近约 2 秒内实际呈现的帧率（120 帧滑动窗口）">
          <span>呈现 FPS</span>
          <strong id="fps">
            {readout.fps ?? '—'}
            <small>fps</small>
          </strong>
        </div>
        <div title="第 99 百分位帧时间换算出的帧率，也就是常说的「1% low」：它只反映卡顿，不反映平均值。">
          <span>P99 FPS</span>
          <strong id="p99">
            {readout.p99Fps ?? '—'}
            <small>fps</small>
          </strong>
        </div>
      </div>

      <div className="perf-grid">
        <div>
          <span>渲染后端选择</span>
        </div>
      </div>
      <div className="backend-switch" role="group" aria-label="渲染后端">
        {ORDER.map(kind => (
          <button
            key={kind}
            id={`backend-${kind}`}
            type="button"
            className={`backend${renderer === kind ? ' on' : ''}`}
            aria-pressed={renderer === kind}
            disabled={!availability[kind]}
            title={
              availability[kind]
                ? RENDERER_TITLES[kind]
                : `${RENDERER_TITLES[kind]}（这个浏览器里不可用）`
            }
            onClick={() => onSelect(kind)}
          >
            {RENDERER_LABELS[kind]}
          </button>
        ))}
      </div>
      {note ? (
        <p className="backend-note" id="rendernote" role="status">
          {note}
        </p>
      ) : null}
    </section>
  );
}
