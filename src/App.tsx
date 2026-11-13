import { useEffect, useRef, useState } from 'react';
import { Kernel } from './wasm/kernel';
import { KernelV2 } from './wasm/kernel-v2';
import { KernelV3 } from './wasm/kernel-v3';
import { KernelV1Backend, KernelV2Backend, KernelV3Backend, probeKernelV2, probeKernelV3 } from './wasm/backend';
import type { AnalysisBackend, KernelVersion } from './wasm/backend';
import { ReactorEngine, type EngineSnapshot, type Readout } from './reactor/engine';
import { webgl2Available, webgpuAvailable, type RendererKind } from './reactor/renderer';
import { useGpu } from './hooks/useGpu';
import { Stage } from './components/Stage';
import { SourcePanel } from './components/SourcePanel';
import { ResponsePanel } from './components/ResponsePanel';
import { Readouts } from './components/Readouts';
import { PerfPanel } from './components/PerfPanel';

export default function App() {
  const coreRef = useRef<HTMLCanvasElement | null>(null);
  const spectrumRef = useRef<HTMLCanvasElement | null>(null);
  const [engine, setEngine] = useState<ReactorEngine | null>(null);
  const [snapshot, setSnapshot] = useState<EngineSnapshot | null>(null);
  const [readout, setReadout] = useState<Readout>({
    levelDb: null,
    peakLevelDb: null,
    peakHz: null,
    centroidHz: null,
    frameMs: null,
    fps: null,
    p99Fps: null,
  });
  const [loadError, setLoadError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [gpu, editGpu] = useGpu();
  const [renderer, setRenderer] = useState<RendererKind>('webgl');
  /** Whether the optional kernels were compiled. Absent means the button stays disabled. */
  const [v2Available, setV2Available] = useState(false);
  const [v3Available, setV3Available] = useState(false);
  /** All backends stay alive once built, so switching back is instant. */
  const backends = useRef<{ v1: AnalysisBackend | null; v2: AnalysisBackend | null; v3: AnalysisBackend | null }>(
    { v1: null, v2: null, v3: null },
  );
  const [availability, setAvailability] = useState<Record<RendererKind, boolean>>({
    canvas2d: true,
    webgl: true,
    webgpu: false,
  });

  // Ask each backend whether it can run here. WebGPU answers asynchronously and
  // only in a secure context, so the button starts disabled until it replies.
  useEffect(() => {
    setAvailability(current => ({ ...current, webgl: webgl2Available() }));
    let cancelled = false;
    void webgpuAvailable().then(ok => {
      if (!cancelled) setAvailability(current => ({ ...current, webgpu: ok }));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load the v1 wasm kernel once, then hand it to the imperative engine.
  useEffect(() => {
    let cancelled = false;
    let created: ReactorEngine | null = null;
    Kernel.load()
      .then(kernel => {
        if (cancelled) return;
        const v1 = new KernelV1Backend(kernel);
        backends.current.v1 = v1;
        created = new ReactorEngine(v1);
        // Publish the first snapshot in the same commit that installs `engine`.
        // Otherwise the next render is still the loading screen, the canvases do
        // not exist yet, and the attach effect below never finds its refs.
        setSnapshot(created.snapshot());
        setEngine(created);
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      });
    // v2 and v3 are optional, so their probes run alongside and never block the boot: a
    // miss only leaves one button disabled.
    void probeKernelV2().then(bytes => {
      if (!cancelled) setV2Available(bytes !== null);
    });
    void probeKernelV3().then(bytes => {
      if (!cancelled) setV3Available(bytes !== null);
    });
    return () => {
      cancelled = true;
      created?.dispose();
    };
  }, []);

  /**
   * Switch the analysis kernel.
   *
   * v1 is already in memory; v2 is instantiated on first use, which is why this
   * is async. If the module turns out to be unusable after the probe said it was,
   * the promise resolves to null and the page simply stays on v1 instead of
   * ending up with no backend at all.
   */
  const selectKernel = async (version: KernelVersion) => {
    if (!engine || engine.kernelVersion === version) return;
    if (version === 'v1') {
      if (backends.current.v1) engine.setBackend(backends.current.v1);
      return;
    }
    try {
      if (version === 'v2') {
        if (!backends.current.v2) {
          const kernel = await KernelV2.load();
          if (!kernel) {
            setV2Available(false);
            return;
          }
          backends.current.v2 = new KernelV2Backend(kernel);
        }
        engine.setBackend(backends.current.v2);
        return;
      }
      if (!backends.current.v3) {
        const kernel = await KernelV3.load();
        if (!kernel) {
          setV3Available(false);
          return;
        }
        backends.current.v3 = new KernelV3Backend(kernel);
      }
      engine.setBackend(backends.current.v3);
    } catch {
      // A broken module must not leave the reactor without an analyser.
      if (version === 'v2') setV2Available(false);
      else setV3Available(false);
    }
  };

  // React is a subscriber, not a driver: the engine pushes throttled snapshots.
  useEffect(() => {
    if (!engine) return;
    const unsubscribe = engine.subscribe(setSnapshot);
    const offReadout = engine.onReadout(setReadout);
    const onPageHide = () => engine.dispose();
    window.addEventListener('pagehide', onPageHide);
    return () => {
      unsubscribe();
      offReadout();
      window.removeEventListener('pagehide', onPageHide);
    };
  }, [engine]);

  // The canvases only exist once the UI renders, so attaching happens here. The
  // canvas element changes identity when the backend switches (React remounts it
  // through `key`), so the guard is the element, not a boolean.
  const attachedTo = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const core = coreRef.current;
    if (!engine || !core || !spectrumRef.current) return;
    if (attachedTo.current === core) return;
    attachedTo.current = core;
    try {
      engine.attach(core, spectrumRef.current, renderer);
    } catch (error) {
      // Most likely a missing WebGL2 or 2D context. Do not retry the same canvas
      // on every render; surface it the same way a failed kernel load is shown.
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, [engine, snapshot, renderer]);

  // If a backend fell back (WebGPU without an adapter, say), follow the engine so
  // the buttons stop claiming something that is not running.
  useEffect(() => {
    if (snapshot?.rendererNote && snapshot.renderer !== renderer) setRenderer(snapshot.renderer);
  }, [snapshot, renderer]);

  useEffect(() => {
    const onChange = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      /* fullscreen is best-effort */
    }
  };

  if (loadError) {
    return (
      <div className="boot boot-error">
        <p>反应堆启动失败</p>
        <code>{loadError}</code>
        <p>
          若是 WASM 相关，请运行 pnpm run build:wasm，或确认 src/generated/reactor_kernel.wasm 存在；
          若是 WebGL 相关，请换用支持 WebGL2 的浏览器，或打开硬件加速。
        </p>
      </div>
    );
  }

  if (!engine || !snapshot) {
    return (
      <div className="boot">
        <p>正在装载 WASM 内核…</p>
      </div>
    );
  }

  return (
    <>
      <header>
        <a className="brand" href="./">
          <span className="brand-icon">▦</span> REACTOR
          <span className="edition">AUDIO LAB / 01</span>
        </a>
        <span className="top-note">声音，进入堆芯。</span>
        <button id="fullscreen" className="plain" onClick={() => void toggleFullscreen()}>
          {fullscreen ? '退出全屏 ↙' : '全屏 ↗'}
        </button>
      </header>
      <main>
        <Stage
          engine={engine}
          canvasRef={coreRef}
          status={engine.statusText()}
          channels={snapshot.channels}
          rods={engine.rodCount}
          autoRotate={snapshot.autoRotate}
          rgbLights={snapshot.rgbLights}
          rendererKind={renderer}
          gpu={gpu}
          onEditGpu={editGpu}
        />
        <aside>
          <div className="panel-title">
            <span>控制台</span>
            <span className="eyebrow">CONTROL PANEL</span>
          </div>
          <SourcePanel engine={engine} snapshot={snapshot} />
          <ResponsePanel engine={engine} snapshot={snapshot} />
          <Readouts
            spectrumRef={spectrumRef}
            readout={readout}
            analysis={snapshot.analysis}
            kernelVersion={snapshot.kernelVersion}
            v2Available={v2Available}
            v3Available={v3Available}
            onSelectKernel={version => void selectKernel(version)}
          />
          <PerfPanel
            readout={readout}
            renderer={snapshot.renderer}
            note={snapshot.rendererNote}
            availability={availability}
            onSelect={setRenderer}
          />
          <p id="message" role="status">
            {snapshot.message}
          </p>
        </aside>
      </main>
      <footer>
        <span>
          <b>◈</b> REACTOR EXPERIMENT
        </span>
        <span></span>
        <span>Powered by React + Vite + WebAssembly</span>
      </footer>
    </>
  );
}
