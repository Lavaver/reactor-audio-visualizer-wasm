// Audio graph + animation loop, kept outside React.
//
// React owns the panel and the input state machine's *display*; this class owns
// the AudioContext, the source lifecycle, the wasm kernel and the 60 fps draw
// loop. Updates reach React through two throttled callbacks so a 120 Hz display
// never turns into 120 React renders per second.

import { drawSpectrum, type Scene, type View } from './draw';
import { Canvas2dRenderer } from './canvas2d';
import { WebglRenderer } from './gl';
import { WebgpuRenderer } from './webgpu';
import type { CoreRenderer, RendererKind } from './renderer';
import type { AnalysisBackend, AnalysisMode, KernelVersion } from '../wasm/backend';
import wasmUrl from '../generated/reactor_kernel.wasm?url';

export type SourceMode = 'demo' | 'mic' | 'system' | 'file';

export type { AnalysisMode, KernelVersion };

export interface DeviceOption {
  id: string;
  label: string;
  loopback: boolean;
}

export interface EngineSnapshot {
  mode: SourceMode;
  playing: boolean;
  hasStarted: boolean;
  trackName: string;
  trackSub: string;
  durationLabel: string;
  elapsedLabel: string;
  message: string;
  seekPercent: number;
  seekEnabled: boolean;
  gain: number;
  smoothing: number;
  volume: number;
  channels: number;
  autoChannels: boolean;
  /** The core slowly turns on its own; dragging the canvas pauses it. */
  autoRotate: boolean;
  /** "RGB light pollution" on the stage. */
  rgbLights: boolean;
  devices: DeviceOption[] | null;
  deviceHint: string | null;
  analysis: AnalysisMode;
  /** Analysis kernel actually running; v2 is only offered when it was built. */
  kernelVersion: KernelVersion;
  /** Backend actually drawing right now; falls back if the requested one fails. */
  renderer: RendererKind;
  /** Why the requested backend was not used, if it was not. */
  rendererNote: string | null;
}

export interface Readout {
  levelDb: number | null;
  /** Sample-peak hold (dBFS) that falls back slowly so it stays readable. */
  peakLevelDb: number | null;
  peakHz: number | null;
  /** Spectral centroid: the power-weighted mean frequency. */
  centroidHz: number | null;
  /**
   * Milliseconds this page spends inside its own animation frame, averaged over
   * the last readout window. A browser cannot see the machine's CPU, so this is
   * the honest version of "how expensive is this page": compare it to the 16.7 ms
   * a 60 fps frame gets.
   */
  frameMs: number | null;
  /** Frames presented per second over the last couple of seconds. */
  fps: number | null;
  /** FPS implied by the 99th percentile frame time, i.e. the "1% low". */
  p99Fps: number | null;
}

const DEMO_FREQS = [55, 82.41, 110, 164.81, 220, 329.63, 440, 659.25];
const IS_FIREFOX = /firefox|fxios/i.test(navigator.userAgent);
/** Served verbatim from `public/`, because a worklet must not be transformed. */
const WORKLET_URL = `${import.meta.env.BASE_URL}reactor-analysis-worklet.js`;
/** Escape hatch for debugging and for the fallback regression: `?noworklet=1`. */
const WORKLET_DISABLED = /(?:^|[?&])noworklet=1/.test(window.location.search);
/**
 * `?preserve=1` keeps the WebGL drawing buffer readable after compositing.
 * The regression harness samples canvas pixels from outside the render loop and
 * cannot do that otherwise; production leaves it off because preserving the
 * buffer costs a copy every frame.
 */
const PRESERVE_DRAWING_BUFFER = /(?:^|[?&])preserve=1/.test(window.location.search);
const LOOPBACK = /monitor|stereo.?mix|what u hear|wave out mix|loopback|blackhole|soundflower|立体声混音|回环/i;

/** How far the peak hold drops per readout tick (~150 ms, so ≈10 dB/s). */
const PEAK_DECAY_DB = 1.5;
/**
 * Radians per second the core turns while auto-rotate is on. Slow on purpose:
 * a full turn takes about 80 s, so it reads as a drift rather than a spin and
 * never fights the drag gesture.
 */
const AUTO_SPIN = 0.078;
/**
 * How long the RGB palette takes to sweep once around the core. Turning it on
 * and turning it off both run the same clockwise front, so the second animation
 * is not the first one played backwards.
 */
const RGB_WIPE_MS = 1100;
/** Frame-time samples kept for the FPS readouts: 120 frames ≈ 2 s at 60 fps. */
const FRAME_SAMPLES = 120;
/**
 * How long a kernel switch takes to interpolate the rods from one version's shape
 * to the other's, in seconds. This is a *dwell*, not an animation of its own: the
 * interpolation is the rod easing the reactor always has, held at a slow setting
 * for this long so 709 rods visibly morph instead of snapping.
 */
const SWITCH_SETTLE_S = 0.6;
/**
 * The smoothing value forced during [`SWITCH_SETTLE_S`]. Near the top of the
 * kernel's range (0..0.95), where one frame covers ~6% of the remaining distance
 * instead of ~15% — about 0.6 s to visually settle.
 */
const SWITCH_SETTLE_SMOOTHING = 0.94;
/**
 * One frame at the 60 Hz the animation is defined against, in milliseconds.
 * Written as a division rather than `16.667` so the unit the kernel is handed is
 * exactly one sixtieth of a second and not an approximation of it.
 */
const FRAME_MS = 1000 / 60;

function formatTime(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  return `${Math.floor(safe / 60).toString().padStart(2, '0')}:${Math.floor(safe % 60)
    .toString()
    .padStart(2, '0')}`;
}

const MODE_LABEL: Record<SourceMode, string> = {
  demo: '演示信号',
  mic: '实时输入',
  system: '机内声音',
  file: '音频分析',
};

/** What to do when no loopback device shows up, phrased for the host OS. */
const LOOPBACK_HINT = /Windows/i.test(navigator.userAgent)
  ? '未发现明显的回环设备；请在 Windows「声音设置 → 更多声音设置 → 录制」中启用「立体声混音 / Stereo Mix」，再重新点击「机内声音」。'
  : /Mac/i.test(navigator.userAgent)
    ? '未发现明显的回环设备；请安装 BlackHole / Loopback 等虚拟音频设备，再重新点击「机内声音」。'
    : '未发现明显的回环设备；请确认 PulseAudio / PipeWire 的 monitor 源可用（如 pavucontrol），再重新点击「机内声音」。';

export class ReactorEngine {
  /** Created in `attach()`, once the canvas exists. */
  renderer: CoreRenderer | null = null;
  /** Backend the engine is running; may differ from the request after a fallback. */
  activeRenderer: RendererKind = 'webgl';
  private rendererNote: string | null = null;
  /**
   * The analysis kernel. Swappable at runtime: v1 and v2 fill the same rod
   * lattice, so a switch only replaces the arrays the frame loop reads.
   */
  private backend: AnalysisBackend;
  private disposed = false;

  private canvas: HTMLCanvasElement | null = null;
  private spectrum: HTMLCanvasElement | null = null;
  private spectrumCtx: CanvasRenderingContext2D | null = null;
  private observer: ResizeObserver | null = null;
  private raf = 0;
  private lastFrameTime = 0;
  private lastUiPush = 0;
  private lastReadout = 0;
  /** Ring buffer of real frame intervals, in milliseconds. */
  private readonly frameIntervals = new Float32Array(FRAME_SAMPLES);
  private frameCursor = 0;
  private frameFilled = 0;
  private lastFrameAt = 0;
  /** Milliseconds spent inside our own frame callbacks since the last readout. */
  private busyMs = 0;
  private framesSinceReadout = 0;
  private busySince = 0;
  private frameMs: number | null = null;
  private fps: number | null = null;
  private p99Fps: number | null = null;

  private audio: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private master: GainNode | null = null;
  /** Sink for the analysis worklet: its output is silent, it only has to be pulled. */
  private workletSink: GainNode | null = null;
  /** Main-thread tap used when the worklet is unavailable or disabled. */
  private worklet: AudioWorkletNode | null = null;
  private workletReady = false;
  private workletTried = false;
  private source: AudioBufferSourceNode | MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private demoNodes: AudioNode[] = [];
  private buffer: AudioBuffer | null = null;

  private loading = false;
  /** True from `play()` entry until it settles, so rapid seeks keep restarting. */
  private playPending = false;
  /** Loudest meter reading since playback started, cleared whenever it stops. */
  private peakLevelDb: number | null = null;
  /**
   * Seconds left of the slow easing that interpolates the rods across a kernel
   * switch. Zero means the reactor is easing at the user's own setting.
   */
  private switchSettle = 0;
  private session = 0;
  private startedAt = 0;
  private offset = 0;
  private message = '音频仅在本机处理，不会上传。';

  mode: SourceMode = 'demo';
  playing = false;
  hasStarted = false;
  gain = 1.4;
  smoothing = 0.75;
  volume = 0.4;
  channels = 32;
  autoChannels = false;
  autoRotate = true;
  rgbLights = false;
  /** 0..1 clockwise wipe progress; 1 means the RGB palette has settled. */
  private rgbSweep = 1;
  devices: DeviceOption[] | null = null;
  deviceHint: string | null = null;
  view: View = { angle: -0.32, tilt: 0.58, zoom: 1 };

  private trackName = 'REACTOR / 内置信号';
  private durationLabel = 'DEMO / LOOP';
  private listeners = new Set<(snapshot: EngineSnapshot) => void>();
  private readoutListeners = new Set<(readout: Readout) => void>();

  constructor(backend: AnalysisBackend) {
    this.backend = backend;
  }

  /** Rods in the lattice the active kernel builds. */
  get rodCount(): number {
    return this.backend.rodCount;
  }

  /** Which analysis kernel is running right now. */
  get kernelVersion(): KernelVersion {
    return this.backend.version;
  }

  /**
   * Swap the analysis kernel without touching the audio graph.
   *
   * Both versions build the identical 709-rod lattice (the v2 check asserts it), so
   * the renderer's GPU buffers stay valid: only the arrays the frame loop reads
   * change identity, and those are re-read every frame.
   *
   * The rods are carried across instead of being reset. The two kernels disagree by
   * up to ~50% on how tall a band should be — v2's display curve is louder on
   * purpose — so a reset would collapse all 709 rods to the floor and rebuild them,
   * which reads as a glitch. Handing the old shape to the new kernel as its starting
   * point lets its own per-rod easing interpolate from "what v1 looked like" to
   * "what v2 wants", and `switchSettle` below slows that easing down so the
   * interpolation takes long enough to be seen.
   */
  setBackend(backend: AnalysisBackend): void {
    if (this.backend.version === backend.version) return;
    // The shape to interpolate from, captured before the old kernel is dropped.
    const carried = Float32Array.from(this.backend.heights);
    // Stop feeding the old kernel before it is dropped, so a worklet message
    // cannot land in an instance nobody reads.
    this.backend.setWorkletActive(false);
    this.backend = backend;
    this.backend.setChannels(this.channels);
    // The worklet node outlives the swap, so a kernel that can use it must be
    // told it is still there; a kernel that cannot (v2) ignores the call.
    if (this.worklet && this.workletReady) this.backend.setWorkletActive(true);
    // Clear the spectrum — it belongs to the kernel that measured it — but keep the
    // rods, and seed them with the shape the previous kernel had reached.
    this.backend.clearSpectrum();
    this.backend.heights.set(carried);
    this.switchSettle = SWITCH_SETTLE_S;
    this.push();
  }

  // --- wiring ---------------------------------------------------------------

  attach(core: HTMLCanvasElement, spectrum: HTMLCanvasElement, kind: RendererKind): void {
    // Switching backends remounts the canvas (a canvas can only ever hand out one
    // context type), so this runs again with a fresh element every time.
    this.detachCore();
    this.canvas = core;
    this.spectrum = spectrum;
    this.spectrumCtx = spectrum.getContext('2d');
    this.activeRenderer = kind;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(core);
    this.observer.observe(spectrum);
    this.resize();
    this.createRenderer(kind);
    this.lastFrameTime = performance.now();
    this.busySince = this.lastFrameTime;
    this.raf = requestAnimationFrame(this.frame);
    this.push();
    // Warm the graph in the background: compiling the kernel for the worklet and
    // loading its module takes ~1 s on a cold cache, and that latency belongs to
    // page load, not to the user's first press of play. The context starts
    // suspended until the gesture, which is exactly what the autoplay policy wants.
    void this.ensureAudio().catch(() => {
      /* a suspended context or a blocked worklet is handled on play */
    });
  }

  /** Tear down the loop, the observers and the current backend. */
  private detachCore(): void {
    cancelAnimationFrame(this.raf);
    this.observer?.disconnect();
    this.observer = null;
    this.renderer?.dispose();
    this.renderer = null;
  }

  /**
   * Build the requested backend on the current canvas.
   *
   * WebGPU can only be created asynchronously, so it lands a moment later; the
   * frame loop simply draws nothing until it does. If it fails, the engine
   * reports the WebGL fallback in the snapshot and the UI switches the request
   * back, which remounts the canvas with a clean context.
   */
  private createRenderer(kind: RendererKind): void {
    const canvas = this.canvas;
    if (!canvas) return;
    const source = { rods: this.backend.rods, stride: this.backend.stride, count: this.backend.rodCount };
    this.rendererNote = null;

    if (kind === 'webgpu') {
      WebgpuRenderer.create(canvas, source)
        .then(renderer => {
          if (this.disposed || this.canvas !== canvas) {
            renderer.dispose();
            return;
          }
          this.renderer = renderer;
          renderer.resize();
          this.push();
        })
        .catch((error: unknown) => {
          if (this.disposed || this.canvas !== canvas) return;
          this.rendererNote = error instanceof Error ? error.message : String(error);
          this.activeRenderer = 'webgl';
          this.renderer = null;
          this.push();
        });
      return;
    }

    this.renderer = kind === 'canvas2d'
      ? new Canvas2dRenderer(canvas, source)
      : new WebglRenderer(canvas, source, PRESERVE_DRAWING_BUFFER);
    this.renderer.resize();
  }

  dispose(): void {
    // Called from both the React cleanup and `pagehide`, so make it idempotent.
    if (this.disposed) return;
    this.disposed = true;
    this.detachCore();
    this.stop();
    this.worklet?.port.close();
    this.worklet?.disconnect();
    this.workletSink?.disconnect();
    this.worklet = null;
    this.workletSink = null;
    void this.audio?.close();
    this.audio = null;
  }

  subscribe(listener: (snapshot: EngineSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot());
    return () => this.listeners.delete(listener);
  }

  onReadout(listener: (readout: Readout) => void): () => void {
    this.readoutListeners.add(listener);
    return () => this.readoutListeners.delete(listener);
  }

  snapshot(): EngineSnapshot {
    return {
      mode: this.mode,
      playing: this.playing,
      hasStarted: this.hasStarted,
      trackName: this.trackName,
      trackSub: this.playing ? '控制棒随频谱实时响应' : this.idleHint(),
      durationLabel: this.durationLabel,
      elapsedLabel: formatTime(this.elapsed()),
      message: this.message,
      seekPercent: this.seekPercent(),
      seekEnabled: this.mode === 'file' && this.buffer !== null,
      gain: this.gain,
      smoothing: this.smoothing,
      volume: this.volume,
      channels: this.channels,
      autoChannels: this.autoChannels,
      autoRotate: this.autoRotate,
      rgbLights: this.rgbLights,
      devices: this.devices,
      deviceHint: this.deviceHint,
      analysis: this.backend.analysis,
      kernelVersion: this.backend.version,
      renderer: this.activeRenderer,
      rendererNote: this.rendererNote,
    };
  }

  private push(): void {
    const snapshot = this.snapshot();
    for (const listener of this.listeners) listener(snapshot);
  }

  private idleHint(): string {
    if (this.mode === 'system') return IS_FIREFOX ? '点击机内声音，选择系统回环设备' : '点击播放，选择共享内容并勾选音频';
    if (this.mode === 'file') return '点击播放，启动声场';
    if (this.mode === 'mic') return '点击播放，允许麦克风后开始分析';
    return '点击播放，启动声场';
  }

  private elapsed(): number {
    if (!this.audio || !this.playing) return this.offset;
    return this.offset + Math.max(0, this.audio.currentTime - this.startedAt);
  }

  private seekPercent(): number {
    if (this.mode !== 'file' || !this.buffer) return 0;
    return Math.min(100, (this.elapsed() / this.buffer.duration) * 100);
  }

  // --- controls -------------------------------------------------------------

  setGain(value: number): void {
    this.gain = value;
    this.push();
  }

  setSmoothing(value: number): void {
    this.smoothing = value;
    this.push();
  }

  setVolume(value: number): void {
    this.volume = value;
    if (this.master && this.audio && this.mode !== 'mic' && this.mode !== 'system') {
      this.master.gain.setTargetAtTime(value, this.audio.currentTime, 0.03);
    }
    this.push();
  }

  setChannels(value: number): void {
    this.applyChannels(Math.round(value));
    this.autoChannels = false;
    this.push();
  }

  setAutoChannels(enabled: boolean): void {
    this.autoChannels = enabled;
    if (enabled) this.applyChannels(this.autoChannelTarget());
    this.push();
  }

  setAutoRotate(enabled: boolean): void {
    this.autoRotate = enabled;
    this.push();
  }

  setRgbLights(enabled: boolean): void {
    if (this.rgbLights === enabled) return;
    this.rgbLights = enabled;
    // Restart the front from the same origin in the same direction, so clearing
    // the pollution replays the wipe rather than undoing it.
    this.rgbSweep = 0;
    this.push();
  }

  private applyChannels(value: number): void {
    this.channels = Math.max(4, Math.min(64, value));
    this.backend.setChannels(this.channels);
  }

  private autoChannelTarget(): number {
    const width = this.canvas?.clientWidth ?? 1080;
    const height = this.canvas?.clientHeight ?? 900;
    const scale = Math.min(width / 45, height / 37) * this.view.zoom;
    return Math.max(8, Math.min(64, Math.round((scale * 15) / 12)));
  }

  // --- view manipulation ----------------------------------------------------

  pointerDown(x: number, y: number): void {
    this.drag = { x, y };
  }

  pointerMove(x: number, y: number): void {
    if (!this.drag) return;
    this.view.angle -= (x - this.drag.x) * 0.006;
    this.view.tilt = Math.max(0.25, Math.min(0.85, this.view.tilt + (y - this.drag.y) * 0.002));
    this.drag = { x, y };
  }

  pointerUp(): void {
    this.drag = null;
  }

  private drag: { x: number; y: number } | null = null;

  zoomBy(deltaY: number): void {
    this.view.zoom = Math.max(0.65, Math.min(1.5, this.view.zoom - deltaY * 0.001));
    this.refreshAutoChannels();
  }

  resetView(): void {
    this.view = { angle: -0.32, tilt: 0.58, zoom: 1 };
    this.refreshAutoChannels();
  }

  private refreshAutoChannels(): void {
    if (!this.autoChannels) return;
    const target = this.autoChannelTarget();
    if (target !== this.channels) {
      this.applyChannels(target);
      this.push();
    }
  }

  // --- source selection -----------------------------------------------------

  async selectMode(mode: SourceMode): Promise<void> {
    this.stop();
    this.mode = mode;
    this.offset = 0;
    this.buffer = null;
    this.devices = null;
    this.deviceHint = null;
    this.message = '音频仅在本机处理，不会上传。';
    if (mode === 'demo') {
      this.trackName = 'REACTOR / 内置信号';
      this.durationLabel = 'DEMO / LOOP';
    } else if (mode === 'mic') {
      this.trackName = 'MIC / 麦克风输入';
      this.durationLabel = 'LIVE';
    } else if (mode === 'system') {
      this.trackName = 'SYSTEM / 机内声音';
      this.durationLabel = 'LIVE';
      if (IS_FIREFOX) {
        // Firefox cannot capture display audio; it can only read a loopback
        // device through getUserMedia, so surface the picker before playing.
        this.message = 'Firefox 不能直接共享系统声音；请选择系统回环输入设备，或改用 Chrome / Edge。';
        this.push();
        await this.probeDevices();
        return;
      }
      this.message = '共享标签页时请勾选「分享标签页音频」。';
    }
    this.push();
    await this.play();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else void this.play();
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.elapsed();
    this.stop();
    this.push();
  }

  async loadFile(file: File): Promise<void> {
    this.stop();
    const id = this.session;
    this.loading = true;
    this.message = '正在读取音频…';
    this.push();
    try {
      await this.ensureAudio();
      const decoded = await this.audio!.decodeAudioData(await file.arrayBuffer());
      if (id !== this.session) return;
      this.buffer = decoded;
      this.mode = 'file';
      this.offset = 0;
      this.trackName = file.name;
      this.durationLabel = formatTime(decoded.duration);
      this.loading = false;
      this.push();
      await this.play();
    } catch {
      this.message = '无法读取该文件，请尝试 MP3 或 WAV 音频。';
      this.push();
    } finally {
      this.loading = false;
    }
  }

  seekTo(percent: number): void {
    if (this.mode !== 'file' || !this.buffer) return;
    // A seek may arrive while the previous `play()` is still awaiting the audio
    // context; treat that as "was playing" or the second seek would leave the
    // player paused.
    const wasPlaying = this.playing || this.playPending;
    this.stop();
    this.offset = (percent / 100) * this.buffer.duration;
    if (wasPlaying) void this.play();
    else this.push();
  }

  // --- device picker (Firefox) ---------------------------------------------

  async probeDevices(): Promise<void> {
    this.devices = null;
    this.deviceHint = '正在请求音频输入权限…';
    this.push();
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
      probe.getTracks().forEach(track => track.stop());
      const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
      const options = inputs
        .map(d => ({ id: d.deviceId, label: d.label || '未命名音频输入', loopback: LOOPBACK.test(d.label) }))
        .sort((a, b) => Number(b.loopback) - Number(a.loopback));
      this.devices = options;
      if (!options.length) {
        this.deviceHint = '没有发现音频输入设备；请检查系统音频设备，或改用 Chrome / Edge。';
      } else if (options.some(o => o.loopback)) {
        this.deviceHint = '已优先列出系统回环设备；选择后点击开始监听。';
      } else {
        this.deviceHint = LOOPBACK_HINT;
      }
    } catch {
      this.deviceHint = '无法访问音频输入。请允许 Firefox 使用麦克风，或改用 Chrome / Edge。';
    }
    this.push();
  }

  async startWithDevice(deviceId: string): Promise<void> {
    this.devices = null;
    this.deviceHint = null;
    this.stop();
    this.mode = 'system';
    this.trackName = 'SYSTEM / 机内声音';
    this.durationLabel = 'LIVE';
    this.offset = 0;
    this.message = '正在打开所选系统回环设备…';
    this.push();
    this.pendingDeviceId = deviceId;
    await this.play();
  }

  private pendingDeviceId: string | null = null;

  // --- audio graph ----------------------------------------------------------

  private async ensureAudio(): Promise<void> {
    if (!this.audio) {
      const audio = new AudioContext();
      const analyser = audio.createAnalyser();
      // Only used as a tap: `fftSize` is the length of the block that
      // `getFloatTimeDomainData` hands us, and the kernel does its own FFT.
      analyser.fftSize = this.backend.fftSize;
      const master = audio.createGain();
      master.gain.value = this.volume;
      analyser.connect(master);
      master.connect(audio.destination);
      this.audio = audio;
      this.analyser = analyser;
      this.master = master;
      await this.ensureWorklet(audio);
    }
    // Before any gesture the context is created suspended and `resume()` is a
    // no-op promise; a rejection here must not be reported as a play failure.
    if (this.audio.state !== 'running') {
      try {
        await this.audio.resume();
      } catch {
        /* autoplay policy: playback will resume on the next gesture */
      }
    }
  }

  /**
   * Bring up the audio-thread analyser.
   *
   * The wasm module is compiled here, on the main thread, because a worklet
   * scope has no `fetch`; the compiled `WebAssembly.Module` is structured-cloned
   * into the processor. Any failure (old browser, blocked module, bad path)
   * silently falls back to main-thread analysis, which is why `analysis` is part
   * of the snapshot instead of an assumption.
   */
  private async ensureWorklet(audio: AudioContext): Promise<void> {
    if (this.workletTried || !audio.audioWorklet) return;
    this.workletTried = true;
    // v2 analyses inside its own asynchronous stage, in wasm, and has no byte
    // spectrum for an audio-thread analyser to hand back.
    if (!this.backend.workletCapable) return;
    if (WORKLET_DISABLED) {
      this.backend.setWorkletActive(false);
      return;
    }
    try {
      const bytes = await (await fetch(wasmUrl)).arrayBuffer();
      const module = await WebAssembly.compile(bytes);
      await audio.audioWorklet.addModule(WORKLET_URL);
      const node = new AudioWorkletNode(audio, 'reactor-analysis', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: 'explicit',
        processorOptions: { module, fftSize: this.backend.fftSize },
      });
      // A zero-gain sink keeps the node in the pulled graph without adding sound.
      const sink = audio.createGain();
      sink.gain.value = 0;
      node.connect(sink);
      sink.connect(audio.destination);
      node.port.onmessage = (event: MessageEvent) => this.onWorkletMessage(event.data);
      this.worklet = node;
      this.workletSink = sink;
    } catch {
      this.backend.setWorkletActive(false);
    }
  }

  private onWorkletMessage(data: unknown): void {
    if (!data || typeof data !== 'object') return;
    const message = data as { type?: string; bins?: ArrayBuffer };
    if (message.type === 'bins' && message.bins) {
      // Handed to the backend, which decides what to do with a byte spectrum:
      // v1 copies it into wasm memory, v2 refuses it.
      if (!this.backend.ingestBins(new Uint8Array(message.bins))) return;
      if (!this.workletReady) {
        this.workletReady = true;
        this.backend.setWorkletActive(true);
        this.push();
      }
    } else if (message.type === 'error') {
      this.workletReady = false;
      this.backend.setWorkletActive(false);
    }
  }

  /** Send a node's audio to the level meter and to the active analyser. */
  private tap(node: AudioNode): void {
    node.connect(this.analyser!);
    if (this.worklet) node.connect(this.worklet);
  }

  async play(): Promise<void> {
    if (this.loading || this.playing) return;
    if (this.mode === 'system' && IS_FIREFOX && !this.pendingDeviceId) return;
    this.loading = true;
    this.playPending = true;
    const attempt = ++this.session;
    try {
      await this.ensureAudio();
      if (attempt !== this.session) return;
      if (this.mode === 'mic' || this.mode === 'system') {
        const stream = this.mode === 'mic'
          ? await navigator.mediaDevices.getUserMedia({
              audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
            })
          : await this.openSystemStream();
        if (attempt !== this.session) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }
        if (this.mode === 'system' && !stream.getAudioTracks().length) {
          stream.getTracks().forEach(track => track.stop());
          this.stop();
          this.message = IS_FIREFOX
            ? '无法从所选设备获取音频；请重新选择系统回环设备。'
            : '没有收到音频。共享标签页时请勾选「分享标签页音频」；共享整个屏幕时请勾选「分享系统音频」。';
          this.push();
          return;
        }
        this.stream = stream;
        this.master!.gain.value = 0;
        const node = this.audio!.createMediaStreamSource(stream);
        this.tap(node);
        this.source = node;
        stream.getTracks().forEach(track =>
          track.addEventListener('ended', () => {
            if (this.stream === stream) {
              this.stop();
              this.push();
            }
          }),
        );
        this.message = this.mode === 'mic'
          ? '正在监听麦克风；停止或切换输入即可关闭。'
          : IS_FIREFOX
            ? '正在分析机内声音（系统回环设备）；停止或切换输入即可结束。'
            : '正在分析机内声音；点击浏览器的「停止共享」即可结束。';
      } else {
        this.master!.gain.value = this.volume;
        this.pendingDeviceId = null;
        if (this.mode === 'file' && this.buffer) {
          if (this.offset >= this.buffer.duration) this.offset = 0;
          const node = this.audio!.createBufferSource();
          node.buffer = this.buffer;
          this.tap(node);
          node.onended = () => {
            this.stop();
            this.offset = 0;
            this.push();
          };
          node.start(0, this.offset);
          this.source = node;
        } else {
          this.startDemoSignal();
        }
        this.message = '音频仅在本机处理，不会上传。';
      }
      this.backend.clearSpectrum();
      this.startedAt = this.audio!.currentTime;
      this.playing = true;
      this.hasStarted = true;
      this.pendingDeviceId = null;
      this.push();
    } catch (error) {
      this.stop();
      this.message = this.mode === 'mic'
        ? '无法访问麦克风，请检查浏览器权限，或导入音频。'
        : this.mode === 'system'
          ? IS_FIREFOX
            ? '无法打开所选音频设备；请重新选择系统回环设备。'
            : '无法拾取机内声音，请使用 Chrome 或 Edge，并在共享时勾选音频。'
          : '无法播放此音频，请尝试其他文件。';
      if (error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'AbortError')) {
        this.message = this.mode === 'system' ? '已取消共享机内声音。' : this.message;
      }
      this.push();
    } finally {
      this.loading = false;
      this.playPending = false;
    }
  }

  private async openSystemStream(): Promise<MediaStream> {
    if (IS_FIREFOX) {
      const deviceId = this.pendingDeviceId;
      if (!deviceId) throw new Error('no-device');
      return navigator.mediaDevices.getUserMedia({
        audio: { deviceId: { exact: deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    }
    if (!navigator.mediaDevices.getDisplayMedia) throw new Error('unsupported');
    // `systemAudio` / `selfBrowserSurface` are real Chromium options but are not
    // in the DOM lib's `DisplayMediaStreamOptions` yet, hence the widening cast.
    const options = {
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      systemAudio: 'include',
      selfBrowserSurface: 'exclude',
    } as unknown as DisplayMediaStreamOptions;
    return navigator.mediaDevices.getDisplayMedia(options);
  }

  private startDemoSignal(): void {
    // One bus for all eight voices: a single tap reaches both analysers.
    const bus = this.audio!.createGain();
    bus.gain.value = 1;
    this.tap(bus);
    this.demoNodes.push(bus);
    for (const [index, frequency] of DEMO_FREQS.entries()) {
      const audio = this.audio!;
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      const lfo = audio.createOscillator();
      const lfoGain = audio.createGain();
      osc.type = index < 3 ? 'triangle' : 'sine';
      osc.frequency.value = frequency;
      gain.gain.value = 0.025;
      lfo.frequency.value = 0.22 + index * 0.087;
      lfoGain.gain.value = 0.023;
      lfo.connect(lfoGain);
      lfoGain.connect(gain.gain);
      osc.connect(gain);
      gain.connect(bus);
      osc.start();
      lfo.start();
      this.demoNodes.push(osc, lfo);
    }
  }

  stop(): void {
    this.session++;
    this.loading = false;
    this.playPending = false;
    if (this.source) {
      const node = this.source;
      if ('onended' in node) node.onended = null;
      if ('stop' in node) {
        try {
          node.stop();
        } catch {
          /* already stopped */
        }
      }
      node.disconnect();
      this.source = null;
    }
    for (const node of this.demoNodes) {
      // The bus is a GainNode, the voices are oscillators; only some can stop.
      if (node instanceof OscillatorNode) {
        try {
          node.stop();
        } catch {
          /* already stopped */
        }
      }
      node.disconnect();
    }
    this.demoNodes = [];
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.playing = false;
    this.pendingDeviceId = null;
    // A new playback session starts with a fresh peak hold.
    this.peakLevelDb = null;
    // Drop the worklet's partial window so a resume never analyses stale audio.
    this.worklet?.port.postMessage({ type: 'reset' });
    // Zero the *spectrum*, and only the spectrum: the band drive falls to zero, the
    // rod heights are left exactly where they are, so the next `step` eases them
    // down over ~0.3 s. Anything that also cleared the heights would collapse all
    // 709 rods in a single frame, which reads as a glitch rather than as a stop.
    this.backend.clearSpectrum();
  }

  // --- frame loop -----------------------------------------------------------

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const canvas of [this.canvas, this.spectrum]) {
      if (!canvas) continue;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
    }
    // Only the spectrum is known to be 2D. Asking the core canvas for a 2D
    // context here would *create* one and lock out WebGL/WebGPU for good, so each
    // core backend installs its own transform in `renderer.resize()`.
    this.spectrum?.getContext('2d')?.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.renderer?.resize();
    this.refreshAutoChannels();
  }

  private frame = (time: number): void => {
    const frameStart = performance.now();
    // Real interval between presented frames, kept separate from the clamped
    // `dt` the animation uses: the FPS readouts must see a stall as a stall.
    if (this.lastFrameAt > 0) {
      const interval = time - this.lastFrameAt;
      if (interval > 0 && interval < 1000) {
        this.frameIntervals[this.frameCursor] = interval;
        this.frameCursor = (this.frameCursor + 1) % FRAME_SAMPLES;
        this.frameFilled = Math.min(FRAME_SAMPLES, this.frameFilled + 1);
      }
    }
    this.lastFrameAt = time;

    const dt = Math.min(Math.max((time - this.lastFrameTime) / FRAME_MS, 0.1), 3) || 1;
    this.lastFrameTime = time;
    const canvas = this.canvas;
    const spectrum = this.spectrum;
    const renderer = this.renderer;
    const sg = this.spectrumCtx;

    if (canvas && spectrum && renderer && sg) {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const analyser = this.analyser;
      const sampleRate = this.audio?.sampleRate ?? 48000;
      // A kernel switch raises the easing time constant for a moment, so the two
      // versions' shapes interpolate instead of cutting. The kernels do the
      // interpolation per rod; this only decides how long they take over it.
      const smoothing = this.switchSettle > 0
        ? Math.max(this.smoothing, SWITCH_SETTLE_SMOOTHING)
        : this.smoothing;
      if (this.switchSettle > 0) this.switchSettle -= dt / 60;

      if (this.playing && analyser) {
        // The backend owns its own capture: v1 pulls a block and transforms it
        // here, v2 writes straight into wasm memory and lets its asynchronous
        // stage run. With the v1 worklet active, `process` is a no-op and the
        // spectrum arrives by message instead.
        this.backend.process(analyser, sampleRate);
        this.backend.step(dt, sampleRate, this.gain, smoothing);
      } else if (this.hasStarted) {
        // Paused: keep easing towards the baseline instead of freezing.
        this.backend.step(dt, sampleRate, this.gain, smoothing);
      } else {
        this.previewHeights(time);
      }

      // Auto-rotate yields to the pointer: dragging sets `drag`, so the core
      // stops turning under the user's hand and picks the drift back up after.
      if (this.autoRotate && !this.drag) {
        this.view.angle += (AUTO_SPIN * dt) / 60;
      }
      if (this.rgbSweep < 1) {
        this.rgbSweep = Math.min(1, this.rgbSweep + (dt * FRAME_MS) / RGB_WIPE_MS);
      }

      const scene: Scene = {
        heights: this.backend.heights,
        phase: time,
        rgbOn: this.rgbLights,
        rgbSweep: this.rgbSweep,
        energy: this.rgbLights || this.rgbSweep < 1 ? this.backend.energy() : 0,
      };
      renderer.drawCore(width, height, this.view, scene);
      drawSpectrum(sg, spectrum.clientWidth, spectrum.clientHeight, this.backend.spectrum,
        this.backend.spectrumScale, this.playing);

      if (time - this.lastReadout > 150) {
        this.lastReadout = time;
        this.emitReadout(analyser);
      }
      if (this.playing && time - this.lastUiPush > 150) {
        this.lastUiPush = time;
        this.push();
      }
    }
    this.busyMs += performance.now() - frameStart;
    this.framesSinceReadout++;
    this.raf = requestAnimationFrame(this.frame);
  };

  /**
   * Refresh the performance readouts.
   *
   * `frameMs` is the mean cost of our own frame callback; `fps` is the mean
   * presented frame rate over the ring buffer; `p99Fps` is the rate implied by
   * the 99th percentile frame time, which is what "1% low" means in practice —
   * one number for the hitch, not the average.
   */
  private updatePerformance(now: number): void {
    if (now > this.busySince) {
      const frames = Math.max(1, this.framesSinceReadout);
      this.frameMs = this.framesSinceReadout > 0 ? this.busyMs / frames : null;
      this.busyMs = 0;
      this.framesSinceReadout = 0;
      this.busySince = now;
    }

    const count = this.frameFilled;
    if (count < 8) return;
    let total = 0;
    for (let i = 0; i < count; i++) total += this.frameIntervals[i];
    const mean = total / count;
    this.fps = mean > 0 ? Math.round(1000 / mean) : null;

    const sorted = Array.from(this.frameIntervals.subarray(0, count)).sort((a, b) => a - b);
    const slow = sorted[Math.min(count - 1, Math.floor(count * 0.99))];
    this.p99Fps = slow > 0 ? Math.round(1000 / slow) : null;
  }

  /**
   * Idle "demo preview": before any audio has played, write a slow travelling
   * wave straight into the kernel's height buffer. This is the one place JS
   * writes kernel memory, and it exists because there is no spectrum yet.
   */
  private previewHeights(time: number): void {
    if (this.mode !== 'demo') return;
    const heights = this.backend.heights;
    const rods = this.backend.rods;
    const stride = this.backend.stride;
    const phase = time * 0.0007;
    for (let i = 0; i < this.backend.rodCount; i++) {
      const offset = i * stride;
      const channel = rods[offset + 4];
      const bandRadius = ((channel + 0.5) / this.channels) * 15;
      const amp = 0.1 + 0.2 * (0.5 + 0.5 * Math.sin(bandRadius * 0.6 - phase + rods[offset] * 0.1));
      heights[i] = 0.07 + amp * 4.8;
    }
  }

  private emitReadout(analyser: AnalyserNode | null): void {
    // The performance figures do not depend on audio, so they keep updating
    // while the reactor is idle.
    this.updatePerformance(performance.now());
    const perf = { frameMs: this.frameMs, fps: this.fps, p99Fps: this.p99Fps };
    if (!this.playing || !analyser) {
      for (const listener of this.readoutListeners) {
        listener({ levelDb: null, peakLevelDb: null, peakHz: null, centroidHz: null, ...perf });
      }
      return;
    }
    // Measured by the backend, from the block it just analysed, so the meter and
    // the bars can never disagree about what was measured.
    const measured = this.backend.readouts(analyser, this.audio?.sampleRate ?? 48000);
    // Peak hold of the *sample* peak, not of the RMS: it reacts to transients,
    // then falls back slowly instead of freezing at the loudest moment forever.
    // It lives here rather than in a kernel because it is presentation, not
    // measurement, and both versions share it. Nothing is rounded on the way: the
    // value keeps whatever width the backend produced, and the panel decides how
    // many digits to show.
    const samplePeakDb = measured.samplePeakDb;
    if (samplePeakDb === null) {
      this.peakLevelDb = this.peakLevelDb === null
        ? null
        : this.peakLevelDb - PEAK_DECAY_DB;
    } else {
      const decayed = this.peakLevelDb === null ? samplePeakDb : this.peakLevelDb - PEAK_DECAY_DB;
      this.peakLevelDb = Math.max(samplePeakDb, decayed);
    }
    for (const listener of this.readoutListeners) {
      listener({
        levelDb: measured.levelDb,
        peakLevelDb: this.peakLevelDb,
        peakHz: measured.peakHz,
        centroidHz: measured.centroidHz,
        ...perf,
      });
    }
  }

  /** Text used by the status pill. */
  statusText(): string {
    if (this.playing) return `运行中 · ${MODE_LABEL[this.mode]}`;
    if (this.mode === 'system') return IS_FIREFOX ? '待机 · 待选择设备' : '待机 · 等待共享';
    return `待机 · ${this.mode === 'demo' ? '演示预览' : '已暂停'}`;
  }
}
