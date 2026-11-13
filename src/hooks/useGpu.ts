// GPU name detection with a manual override.
//
// Firefox deliberately generalises the WebGL renderer string ("... or similar")
// so a page cannot fingerprint the exact card. We show the sanitised name with a
// `≈` marker and let the user type the real one once; it is remembered locally.
import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'reactor.gpu';

export interface GpuState {
  label: string;
  title: string;
  /** True when the browser gave us a generalised name. */
  approx: boolean;
  /** True when the label came from the stored override. */
  manual: boolean;
}

const EMPTY: GpuState = { label: '检测中…', title: '', approx: false, manual: false };

// WebGPU is not part of the DOM lib yet, so only the two fields we read are
// declared here. `info` may be a promise in some implementations.
interface GpuAdapterInfo {
  description?: string;
  device?: string;
}
interface GpuAdapter {
  info?: GpuAdapterInfo | Promise<GpuAdapterInfo>;
}
interface GpuNavigator {
  gpu?: { requestAdapter(): Promise<GpuAdapter | null> };
}

function cleanGpu(value: string): string {
  const source = value.trim();
  const isAngle = /^ANGLE\s*\(/i.test(source);
  const cleaned = source
    .replace(/^ANGLE\s*\(/i, '')
    .replace(/\)\s*$/, '')
    .replace(/\s+Direct3D\d+.*$/i, '')
    .replace(/\s*\(0x[0-9a-f]+\)\s*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
  return isAngle ? cleaned.replace(/^[^,]+,\s*/, '') : cleaned;
}

function readOverride(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

function writeOverride(value: string): void {
  try {
    if (value) localStorage.setItem(STORAGE_KEY, value);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage may be blocked; the label just will not persist */
  }
}

async function detectGpu(): Promise<GpuState> {
  let webglLabel = '';
  let raw = '';
  let sanitized = false;
  try {
    const probe = document.createElement('canvas');
    const gl = probe.getContext('webgl2') ?? probe.getContext('webgl');
    if (gl) {
      const extension = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = String(
        extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) ?? '',
      );
      const vendor = extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      sanitized = /\bor similar\b/i.test(renderer);
      raw = [vendor, renderer].filter(Boolean).join(' · ');
      webglLabel = cleanGpu(renderer);
    }
  } catch {
    /* WebGL unavailable; fall through to WebGPU and the placeholder */
  }

  const generic = !webglLabel || /^(mozilla|webkit|webkit webgl|webgl)$/i.test(webglLabel);
  if (webglLabel && !generic && !sanitized) {
    return { label: webglLabel, title: raw || webglLabel, approx: false, manual: false };
  }

  const gpuApi = (navigator as Navigator & GpuNavigator).gpu;
  if (gpuApi) {
    try {
      const adapter = await gpuApi.requestAdapter();
      // `adapter.info` is synchronous in the current spec, but some builds still
      // hand back a promise, so accept either shape.
      let info: unknown = adapter?.info;
      if (info && typeof (info as { then?: unknown }).then === 'function') {
        info = await (info as Promise<unknown>);
      }
      const resolved = info as GpuAdapterInfo | undefined;
      const name = resolved?.description || resolved?.device || '';
      if (name) return { label: name, title: name, approx: false, manual: false };
    } catch {
      /* no WebGPU adapter; keep the WebGL label */
    }
  }

  if (webglLabel) {
    return {
      label: webglLabel,
      title: sanitized
        ? `Firefox 会泛化 GPU 型号（原始字符串：${raw}）；点击手动设置真实型号。`
        : raw || webglLabel,
      approx: sanitized,
      manual: false,
    };
  }
  return { label: '未检测到', title: 'WebGL / WebGPU 均不可用；点击手动设置', approx: false, manual: false };
}

export function useGpu(): [GpuState, () => void] {
  const [gpu, setGpu] = useState<GpuState>(EMPTY);

  useEffect(() => {
    const override = readOverride();
    if (override) {
      setGpu({ label: override, title: '手动设置的 GPU 型号；点击修改', approx: false, manual: true });
      return;
    }
    let cancelled = false;
    void detectGpu().then(state => {
      if (!cancelled) setGpu(state);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const edit = useCallback(() => {
    const current = readOverride() || gpu.label;
    const value = window.prompt('设置 GPU 型号（留空恢复自动检测）', current);
    if (value === null) return;
    const trimmed = value.trim();
    if (trimmed) {
      writeOverride(trimmed);
      setGpu({ label: trimmed, title: '手动设置的 GPU 型号；点击修改', approx: false, manual: true });
      return;
    }
    writeOverride('');
    setGpu(EMPTY);
    void detectGpu().then(setGpu);
  }, [gpu.label]);

  return [gpu, edit];
}
