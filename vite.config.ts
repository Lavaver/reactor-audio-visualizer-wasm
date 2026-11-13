import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // `127.0.0.1` instead of Vite's default `localhost`: on Windows that name can
  // resolve to ::1 only, which breaks simple IPv4 health checks and previews.
  server: { host: '127.0.0.1', port: 4174, strictPort: true },
  preview: { host: '127.0.0.1', port: 4174, strictPort: true },
  // The wasm kernel is fetched at runtime, so it must be emitted as an asset
  // rather than inlined; `assetsInlineLimit: 0` keeps `.wasm` out of base64.
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true, assetsInlineLimit: 0 },
});
