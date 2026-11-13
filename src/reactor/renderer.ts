// Renderer abstraction: three interchangeable backends for the core scene.
//
// Each backend owns its own context type, and one canvas can only ever hand out
// one of them (`getContext('2d')` then `getContext('webgl2')` returns null), so
// switching backends remounts the canvas element. React does that with a `key`,
// and the engine swaps the renderer in `attach()`.

import type { Scene, View } from './draw';

export type RendererKind = 'canvas2d' | 'webgl' | 'webgpu';

export const RENDERER_LABELS: Record<RendererKind, string> = {
  canvas2d: '传统',
  webgl: 'WebGL',
  webgpu: 'WebGPU',
};

/**
 * One sentence per backend covering how it draws, who can run it, and what it
 * costs — the three things that actually differ between them.
 */
export const RENDERER_TITLES: Record<RendererKind, string> = {
  canvas2d: '每帧在主线程排序 709 根棒并逐个填充多边形；所有浏览器都支持，但性能弱。',
  webgl: '实例化绘制配深度缓冲，排序与配色均在 GPU 上计算；现代设备与浏览器均可使用，性能优良。',
  webgpu: 'WGSL 管线加 4× MSAA，一次写 uniform、一次提交；需要现代设备和高版本 Chromium 内核浏览器，（理论上）性能最佳。',
};

export interface RodSource {
  rods: Float32Array;
  stride: number;
  count: number;
}

export interface CoreRenderer {
  readonly kind: RendererKind;
  /** Called after the backing store was resized; refreshes the viewport. */
  resize(): void;
  drawCore(width: number, height: number, view: View, scene: Scene): void;
  dispose(): void;
}

/** Whether a backend can be created at all, cheap enough to ask on every render. */
let webgl2: boolean | null = null;
export function webgl2Available(): boolean {
  if (webgl2 === null) {
    try {
      const probe = document.createElement('canvas');
      webgl2 = Boolean(probe.getContext('webgl2'));
    } catch {
      webgl2 = false;
    }
  }
  return webgl2;
}

/**
 * WebGPU needs an adapter, which can only be requested asynchronously. The result
 * is cached because the answer does not change for a page lifetime, and because
 * requesting an adapter twice is wasteful.
 */
let webgpuProbe: Promise<boolean> | null = null;
export function webgpuAvailable(): Promise<boolean> {
  if (!('gpu' in navigator) || !navigator.gpu) return Promise.resolve(false);
  if (!webgpuProbe) {
    webgpuProbe = navigator.gpu
      .requestAdapter()
      .then(adapter => adapter !== null)
      .catch(() => false);
  }
  return webgpuProbe;
}
