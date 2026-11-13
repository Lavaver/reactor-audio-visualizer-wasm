import { useEffect, type RefObject } from 'react';
import type { ReactorEngine } from '../reactor/engine';
import type { RendererKind } from '../reactor/renderer';
import type { GpuState } from '../hooks/useGpu';

interface StageProps {
  engine: ReactorEngine;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  status: string;
  channels: number;
  rods: number;
  autoRotate: boolean;
  rgbLights: boolean;
  /** Changing this remounts the canvas: one canvas can only hold one context type. */
  rendererKind: RendererKind;
  gpu: GpuState;
  onEditGpu: () => void;
}

export function Stage({
  engine,
  canvasRef,
  status,
  channels,
  rods,
  autoRotate,
  rgbLights,
  rendererKind,
  gpu,
  onEditGpu,
}: StageProps) {
  // Wheel must be non-passive to stop the page from scrolling while zooming, so
  // it is registered directly instead of through React's synthetic handler. The
  // canvas is replaced on every backend switch, hence the extra dependency.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      engine.zoomBy(event.deltaY);
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [canvasRef, engine, rendererKind]);

  return (
    <section className="stage">
      <div className="stage-title">
        <div className="eyebrow">REAL-TIME AUDIO VISUALIZER</div>
        <h1>
          声场反应堆 on WebGL
          <span>
            /{' '}
            <span
              id="gpu"
              role="button"
              tabIndex={0}
              className={gpu.approx ? 'approx' : undefined}
              title={gpu.title || '点击设置 GPU 型号'}
              onClick={onEditGpu}
              onKeyDown={event => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onEditGpu();
                }
              }}
            >
              {gpu.approx ? '≈ ' : ''}
              {gpu.label}
            </span>
          </span>
        </h1>
      </div>
      <div className="live">
        <i />
        <span id="status">{status}</span>
      </div>
      {/* View switches live up here rather than in the bottom bar: the stage is
          stretched to the panel's height, so its bottom edge is often below the
          fold and anything parked there is unreachable without scrolling. */}
      <div className="stage-toggles">
        <button
          id="autorotate"
          type="button"
          className={`plain toggle${autoRotate ? ' on' : ''}`}
          aria-pressed={autoRotate}
          aria-label="自动旋转"
          title="让反应堆本体缓慢自转；拖动视角时自动暂停，松手后接着转"
          onClick={() => engine.setAutoRotate(!autoRotate)}
        >
          ⟳ <span className="wide">自动旋转</span>
        </button>
        <button
          id="rgb"
          type="button"
          className={`plain toggle${rgbLights ? ' on rgb' : ''}`}
          aria-pressed={rgbLights}
          aria-label="RGB 光污染"
          title="本模式可能导致审美崩坏与精神污♂染，请谨慎开启 :)"
          onClick={() => engine.setRgbLights(!rgbLights)}
        >
          ✦ <span className="wide">RGB 效果</span>
        </button>
        <button id="reset" className="plain" aria-label="重置视角" onClick={() => engine.resetView()}>
          ⤾ <span className="wide">重置视角</span>
        </button>
      </div>
      <canvas
        id="core"
        key={rendererKind}
        ref={canvasRef}
        aria-label="圆形控制棒阵列，随音频频谱起伏的三维可视化"
        onPointerDown={event => {
          engine.pointerDown(event.clientX, event.clientY);
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={event => engine.pointerMove(event.clientX, event.clientY)}
        onPointerUp={() => engine.pointerUp()}
        onPointerCancel={() => engine.pointerUp()}
      />
      <div className="axis top">
        CONTROL ROD ARRAY · <span id="count">{rods}</span> RODS ·{' '}
        <span id="channelsLabel">{channels}</span> CHANNELS
      </div>
      <div className="stage-bottom">
        <span>
          ↔ 拖动旋转视角 <span className="desktop"> / 滚轮缩放</span>
        </span>
      </div>
      <div className="scale">
        <span>振幅</span>
        <div />
        <span>高</span>
        <span>低</span>
      </div>
    </section>
  );
}
