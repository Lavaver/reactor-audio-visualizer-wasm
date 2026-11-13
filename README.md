# Reactor Audio Visualizer (WASM)

以切尔诺贝利反应堆堆芯的控制棒阵列为视觉原型的实时音频可视化实验，把频谱分析整条链路搬进了 WebAssembly。

网页在浏览器本机分析音频，将不同频段的能量映射为 709 根控制棒的升降运动。支持演示信号、本地音频文件、麦克风输入、机内声音，以及视角拖动、缩放和灵敏度、平滑度、音量调节。

分析内核有两版，可在面板里切换。v1 用 f32 模仿 Web Audio `AnalyserNode`，可以在 `AudioWorklet` 里运行；v2 是全 f64 的异步管线，读数换成真实的 dBFS / Hz，并把显示曲线调得更强烈。v2 是可选的——没有编译就自动禁用，页面继续使用 v1。

## Screenshots

![v1](./.screenshots/core_v1.png)

---

![v2](./.screenshots/core_v2.png)

---

![rgb](./.screenshots/effect_rgb.png)

## Run locally

需要 Node（`^20.19` 或 `>=22.12`）、pnpm、Rust 1.85+（edition 2024），以及 `rustup target add wasm32-unknown-unknown`。

```bash
pnpm install
pnpm run build:wasm        # v1 内核（必需）
pnpm run build:wasm:v2     # v2 内核（可选，不编就没有 v2 按钮）
pnpm run dev               # http://127.0.0.1:4174
```

生产构建用 `pnpm build`（会先编译 v1 内核）配 `pnpm preview`。两版内核的数学都可以在 Node 里离线验证，不需要浏览器：

```bash
pnpm run check:kernel
pnpm run check:kernel:v2
```

## Notes

This is an audiovisual interaction experiment. It is not a physical simulation of a reactor, and audio is processed locally in the browser.

v2 只有编译过才会出现：它放在 `public/` 而不是走静态 import，页面在运行时抓取并校验 wasm 魔数，`scripts/copy-wasm.mjs` 会删掉过期的旧副本，所以「文件存在」严格等于「编译过」。

两版内核的柱高不可直接比较——v2 的显示曲线按设计更激进。v2 目前只在主线程运行；`wasm32-unknown-unknown` 上没有时钟也没有可 park 的线程，因此 Tokio 只用到 `sync` 原语，执行器由 JS 逐帧驱动（推导见 `wasm_v2/src/lib.rs` 头部）。

## License

MIT. See [LICENSE](LICENSE).
