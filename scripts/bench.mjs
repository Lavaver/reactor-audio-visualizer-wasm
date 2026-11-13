// Per-frame analysis cost of each kernel, measured in Node with no browser and no
// rendering. Writes docs/bench.md so the numbers in the docs are reproducible rather
// than transcribed.
//
//   node scripts/bench.mjs [frames]
//
// What this does and does not measure: it times `push` + `step` per frame, which is the
// whole analysis chain including the block allocation v2/v3 make for the channel. It does
// not include WebGL/WebGPU submission, canvas compositing, or the audio graph — a browser
// frame is this number plus those.
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const FRAMES = Number(process.argv[2] ?? 600);
const WARMUP = 60;
const FFT = 2048;
const RATE = 48000;
const BUDGET_MS = 1000 / 60;

const KERNELS = {
  v1: 'wasm/target/wasm32-unknown-unknown/release/reactor_kernel.wasm',
  v2: 'wasm_v2/target/wasm32-unknown-unknown/release/reactor_kernel_v2.wasm',
  v3: 'wasm_v3/target/wasm32-unknown-unknown/release/reactor_kernel_v3.wasm',
};

async function load(relative) {
  const bytes = await readFile(join(root, relative));
  return (await WebAssembly.instantiate(await WebAssembly.compile(bytes), {})).exports;
}

/** A deterministic mix: a tone plus a noise floor, so no band is trivially empty. */
function source() {
  let seed = 20260113;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff * 2 - 1;
  };
  return i => 0.35 * Math.sin((2 * Math.PI * 220 * i) / RATE)
    + 0.12 * Math.sin((2 * Math.PI * 3100 * i) / RATE)
    + 0.05 * random();
}

/**
 * The block every kernel is fed, generated once.
 *
 * Generating it inside the timed call would dominate the measurement: 2048 iterations with
 * a closure call each is ~0.1 ms, which is the same order as the whole analysis chain --
 * the first version of this script reported v1 and v2 as equally fast because the generator,
 * not the kernel, was being timed. `Float32Array.set` is one memcpy.
 */
const generator = source();
const BLOCK = new Float32Array(FFT);
for (let i = 0; i < FFT; i++) BLOCK[i] = generator(i);

const percentile = (sorted, fraction) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];

/**
 * One measurement: hand `FRAMES` blocks to a kernel and time each push+step pair.
 *
 * `label` goes in the markdown table, `short` in the chart's label column where there is
 * no room for the full description.
 */
function measure(key, short, label, call) {
  for (let i = 0; i < WARMUP; i++) call();
  const samples = new Float64Array(FRAMES);
  for (let i = 0; i < FRAMES; i++) {
    const started = process.hrtime.bigint();
    call();
    samples[i] = Number(process.hrtime.bigint() - started) / 1e6;
  }
  const sorted = Array.from(samples).sort((a, b) => a - b);
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  return {
    key,
    short,
    label,
    mean,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    share: (mean / BUDGET_MS) * 100,
  };
}

const rows = [];

// --- v1: f32, Hann, byte spectrum -------------------------------------------
{
  const v1 = await load(KERNELS.v1);
  v1.rk_init();
  v1.rk_set_fft(FFT);
  v1.rk_set_channels(32);
  const wave = new Float32Array(v1.memory.buffer, v1.rk_wave_ptr(), FFT);
  rows.push(measure('v1', 'v1 · 32 bands', 'v1 · 32 bands, FFT 2048', () => {
    wave.set(BLOCK);
    v1.rk_analyse();
    v1.rk_step(1, RATE, 1.4, 0.75);
  }));
}

// --- v2: f64 FFT only --------------------------------------------------------
{
  const v2 = await load(KERNELS.v2);
  v2.rk2_init();
  v2.rk2_configure(FFT, RATE);
  v2.rk2_set_channels(32);
  const wave = () => new Float32Array(v2.memory.buffer, v2.rk2_wave_ptr(), FFT);
  rows.push(measure('v2', 'v2 · 32 bands', 'v2 · 32 bands, f64 FFT', () => {
    wave().set(BLOCK);
    v2.rk2_push();
    v2.rk2_step(1 / 60, 1.4, 0.75);
  }));
}

// --- v3: FFT + CQT -----------------------------------------------------------
// `channels = 4` moves the crossover (`Q fs / N_fft`) below the lowest band centre, so the
// CQT runs zero bins: the difference between the two rows is the CQT's cost, measured
// rather than estimated.
const v3 = await load(KERNELS.v3);
const v3Push = channels => {
  v3.rk3_init();
  v3.rk3_configure(FFT, RATE);
  v3.rk3_set_channels(channels);
};
{
  v3Push(30); // 30 bands is the smallest count where the CQT still runs; see below
  const wave = () => new Float32Array(v3.memory.buffer, v3.rk3_wave_ptr(), FFT);
  const call = () => {
    wave().set(BLOCK);
    v3.rk3_push();
    v3.rk3_step(1 / 60, 1.4, 0.75);
  };
  v3Push(4);
  const bands = v3.rk3_cqt_bins();
  const withoutCqt = measure('v3-off', 'v3 · CQT off', `v3 · 4 bands (CQT ${bands} bins)`, call);
  v3Push(32);
  const withCqt = measure('v3', 'v3 · 32 bands', `v3 · 32 bands (CQT ${v3.rk3_cqt_bins()} bins)`, call);
  rows.push(withoutCqt, withCqt);

  // Cost per CQT sample, the number that decides whether a band count is affordable.
  const bins = v3.rk3_cqt_bins();
  const freqs = new Float64Array(v3.memory.buffer, v3.rk3_cqt_freq_ptr(), 64);
  const rho = Math.pow(320, 1 / 32);
  const q = 4 / (rho - 1);
  let samples = 0;
  for (let bin = 0; bin < bins; bin++) samples += Math.round((q * RATE) / freqs[bin]);
  const cqtMs = withCqt.mean - withoutCqt.mean;
  const nsPerSample = (cqtMs / Math.max(samples, 1)) * 1e6;
  rows.push({
    key: 'cqt',
    short: 'CQT share',
    detail: `${samples} samples · ${nsPerSample.toFixed(1)} ns/sample`,
    label: `  ↳ CQT share: ${samples} samples/frame, ${nsPerSample.toFixed(1)} ns/sample`,
    mean: cqtMs,
    // A difference of two means has no distribution, so neither the table nor the chart
    // gets a p50/p95 for it -- an invented tail would be read as measured.
    p50: null,
    p95: null,
    share: (cqtMs / BUDGET_MS) * 100,
    samples,
    nsPerSample,
    bins: v3.rk3_cqt_bins(),
  });
  withCqt.cqtShare = cqtMs;
}

// --- sizes -------------------------------------------------------------------
const sizes = [];
for (const [name, path] of Object.entries(KERNELS)) {
  const bytes = await readFile(join(root, path));
  sizes.push({ name, kB: bytes.byteLength / 1024 });
}

// --- chart -------------------------------------------------------------------
// Hand-written SVG instead of a charting dependency: two fixed charts do not justify one,
// and generating the picture in the same run is what keeps it from drifting away from the
// numbers it claims to show.
const ACCENT = { v1: '#7f8ea3', v2: '#b98cff', v3: '#46e6ff', fft: '#2f7284', cqt: '#46e6ff' };
const INK = '#dbe6f0';
const MUTED = '#6b7c8f';
const GRID = '#1b2530';

const W = 1000;
const M = 24;
const LABEL_X = 236; // label column, right-aligned
const PLOT_X = 252;
const PLOT_W = 470;
const VALUE_X = 734;
const PCT_X = W - M;
const ROW_H = 46;
const BAR_H = 13;

const esc = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const round = (value, digits) => Number(value.toFixed(digits));
const barWidth = (value, axis) => Math.max(0, (value / axis.max) * PLOT_W);

/** Axis top and ticks: the smallest step that holds every value in at most six intervals. */
function axisFor(values, steps) {
  const peak = Math.max(...values.filter(Number.isFinite));
  const step = steps.find(candidate => Math.ceil(peak / candidate) <= 6) ?? steps[steps.length - 1];
  const max = Math.ceil(peak / step) * step;
  const ticks = [];
  for (let i = 0; i * step <= max + step / 2; i++) ticks.push(round(i * step, 4));
  return { max, ticks };
}

/**
 * One section: a title, a row per bar, an axis under it and an optional right-hand column.
 *
 * `bars` are `{ short, detail, value, p95, segments }`; `segments` is what splits a bar
 * into named parts (the v3 bar is FFT + CQT at the same total, so the split is the same
 * number the table reports, not a second estimate). `p95` is optional: a module size has no
 * tail, and drawing one would invent a distribution.
 */
function section(title, note, bars, axis, format, startY, right) {
  const height = bars.length * ROW_H;
  const bottom = startY + height;
  const out = [];

  out.push(`<text x="${M}" y="${startY - 36}" class="h">${esc(title)}</text>`);
  out.push(`<text x="${M}" y="${startY - 17}" class="note">${esc(note)}</text>`);
  if (right) out.push(`<text x="${PCT_X}" y="${startY - 17}" class="note" text-anchor="end">${esc(right.header)}</text>`);

  for (const tick of axis.ticks) {
    const x = round(PLOT_X + barWidth(tick, axis), 1);
    out.push(`<line x1="${x}" y1="${startY - 4}" x2="${x}" y2="${bottom}" stroke="${GRID}" stroke-width="1"/>`);
    out.push(`<text x="${x}" y="${bottom + 16}" class="tick" text-anchor="middle">${tick}</text>`);
  }
  out.push(`<line x1="${PLOT_X}" y1="${startY - 4}" x2="${PLOT_X}" y2="${bottom}" stroke="#2a3a4a" stroke-width="1"/>`);

  bars.forEach((bar, index) => {
    const centre = startY + index * ROW_H + ROW_H / 2;
    const labelY = bar.detail ? centre - 2 : centre + 4;
    out.push(`<text x="${LABEL_X}" y="${labelY}" class="label" text-anchor="end">${esc(bar.short)}</text>`);
    if (bar.detail) out.push(`<text x="${LABEL_X}" y="${labelY + 13}" class="tick" text-anchor="end">${esc(bar.detail)}</text>`);

    // Envelope first: the p95 is the tail of the same measurement, not a second bar.
    if (bar.p95 !== undefined) {
      out.push(`<rect x="${PLOT_X}" y="${round(centre - BAR_H / 2 - 3, 1)}" width="${round(barWidth(bar.p95, axis), 1)}" height="${BAR_H + 6}" rx="3" fill="${bar.color}" fill-opacity="0.10" stroke="${bar.color}" stroke-opacity="0.35" stroke-width="1"/>`);
    }

    let x = PLOT_X;
    for (const segment of bar.segments ?? [{ value: bar.value, color: bar.color }]) {
      const width = barWidth(segment.value, axis);
      if (width > 0.5) {
        out.push(`<rect x="${round(x, 1)}" y="${round(centre - BAR_H / 2, 1)}" width="${round(Math.max(width - 1, 1), 1)}" height="${BAR_H}" rx="2" fill="${segment.color}"/>`);
      }
      x += width;
    }

    out.push(`<text x="${VALUE_X}" y="${labelY}" class="value">${esc(format(bar.value))}</text>`);
    if (bar.p95 !== undefined) out.push(`<text x="${VALUE_X}" y="${labelY + 13}" class="tick">p95 ${esc(format(bar.p95))}</text>`);
    if (right) out.push(`<text x="${PCT_X}" y="${labelY}" class="pct" text-anchor="end">${right.format(bar.value)}</text>`);
  });

  return { svg: out.join('\n'), y: bottom };
}

function chart() {
  const out = [];
  let y = M + 26;
  out.push(`<text x="${M}" y="${y}" class="title">reactor kernel · analysis cost</text>`);
  y += 20;
  out.push(`<text x="${M}" y="${y}" class="note">node scripts/bench.mjs · ${FRAMES} frames after ${WARMUP} warm-up · ${FFT} samples/frame at ${RATE} Hz · analysis only, no rendering</text>`);

  const byKey = Object.fromEntries(rows.map(row => [row.key, row]));
  const costBars = rows.map(row => {
    const bar = { short: row.short, detail: row.detail, value: row.mean, p95: row.p95 ?? undefined, color: ACCENT[row.key] ?? ACCENT.v1 };
    if (row.key === 'v3') {
      // Split at the measured CQT share: base + share sums to this row's mean exactly.
      const base = row.mean - byKey.cqt.mean;
      bar.segments = [
        { value: base, color: ACCENT.fft },
        { value: byKey.cqt.mean, color: ACCENT.cqt },
      ];
      bar.detail = `FFT ${base.toFixed(3)} + CQT ${byKey.cqt.mean.toFixed(3)}`;
    }
    return bar;
  });
  const costAxis = axisFor(costBars.map(bar => bar.p95), [0.1, 0.2, 0.25, 0.5, 1, 2, 5]);
  y += 58;
  const cost = section(
    'analysis per frame (ms)',
    'bar = mean, outline = p95; the v3 bar is split into FFT and CQT at the same total',
    costBars, costAxis, value => `${value.toFixed(3)} ms`, y,
    { header: '% of 16.67 ms frame', format: value => `${((value / BUDGET_MS) * 100).toFixed(1)}%` },
  );
  out.push(cost.svg);
  y = cost.y + 86;

  const sizeAxis = axisFor(sizes.map(size => size.kB), [5, 10, 25, 50, 100, 200, 500]);
  const size = section(
    '.wasm module size (kB)',
    'compiled wasm32-unknown-unknown release, as shipped by scripts/copy-wasm.mjs',
    sizes.map(entry => ({ short: `kernel ${entry.name}`, value: entry.kB, color: ACCENT[entry.name] ?? ACCENT.v1 })),
    sizeAxis, value => `${value.toFixed(1)} kB`, y, null,
  );
  out.push(size.svg);
  y = size.y + 40;

  out.push(`<text x="${M}" y="${y}" class="note">60 Hz frame budget = ${BUDGET_MS.toFixed(2)} ms in total; the % column is the analysis share of it. WebGL/WebGPU submission, compositing and the audio graph are not measured.</text>`);

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${round(y + M, 0)}" viewBox="0 0 ${W} ${round(y + M, 0)}" role="img" aria-label="Analysis cost per frame and wasm module size per kernel">
<style>
  text { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
  .title { font-size: 16px; font-weight: 700; fill: ${INK}; letter-spacing: 0.4px; }
  .h { font-size: 13px; font-weight: 700; fill: ${INK}; letter-spacing: 0.3px; }
  .note { font-size: 10.5px; fill: ${MUTED}; }
  .label { font-size: 12px; fill: ${INK}; }
  .value { font-size: 12px; font-weight: 700; fill: ${INK}; }
  .pct { font-size: 12px; fill: ${INK}; }
  .tick { font-size: 10px; fill: ${MUTED}; }
</style>
<rect width="100%" height="100%" fill="#0b1016"/>
${out.join('\n')}
</svg>
`;
}

// --- report ------------------------------------------------------------------
const cell = value => (value == null ? '—' : value.toFixed(3));

console.log(`\n${FRAMES} frames after ${WARMUP} warm-up, analysis only (no rendering), 60 Hz budget = ${BUDGET_MS.toFixed(2)} ms\n`);
console.log('| configuration | mean ms | p50 ms | p95 ms | % of 60 Hz frame |');
console.log('|---|---|---|---|---|');
for (const row of rows) {
  console.log(`| ${row.label} | ${cell(row.mean)} | ${cell(row.p50)} | ${cell(row.p95)} | ${row.share.toFixed(1)}% |`);
}
console.log('\n| kernel | .wasm |');
console.log('|---|---|');
for (const size of sizes) console.log(`| ${size.name} | ${size.kB.toFixed(1)} kB |`);

const doc = [
  '# Analysis cost per frame',
  '',
  '![Analysis cost per frame and wasm module size per kernel](bench.svg)',
  '',
  `Generated by \`node scripts/bench.mjs\` — ${FRAMES} frames after ${WARMUP} warm-up, one`,
  `block (${FFT} samples at ${RATE} Hz) per frame, no browser, no rendering.`,
  '',
  'The 60 Hz frame budget is 16.67 ms **in total**; the numbers below are the analysis',
  'share of it. Rendering (WebGL/WebGPU submission, compositing) is not included.',
  '',
  'These are single-run figures from one machine. At 0.07–0.36 ms per frame the run-to-run',
  'spread is 10–20% — enough to reorder v1, v2 and the CQT-free v3 row — so read the ratios',
  'and the order of magnitude, not the third digit. The CQT share is the stable number: it is',
  'a difference between two rows measured back to back in the same process.',
  '',
  '| configuration | mean ms | p50 ms | p95 ms | % of 60 Hz frame |',
  '|---|---|---|---|---|',
  ...rows.map(row => `| ${row.label} | ${cell(row.mean)} | ${cell(row.p50)} | ${cell(row.p95)} | ${row.share.toFixed(1)}% |`),
  '',
  'The `v3 · 4 bands` row is the same kernel with the CQT switched off by construction:',
  'at 4 bands the derived crossover (`Q fs / N_fft`, with `Q = 4/(rho-1)` and',
  '`rho = 320^(1/4)`) falls below the lowest band centre, so no bin fits and the constant-Q',
  'path does no work. The gap between the two v3 rows is therefore the CQT cost, measured',
  'rather than estimated — and the per-sample figure under it is what decides whether a',
  'wider CQT is affordable.',
  '',
  '| kernel | .wasm |',
  '|---|---|',
  ...sizes.map(size => `| ${size.name} | ${size.kB.toFixed(1)} kB |`),
  '',
  'Sizes are after the state was moved out of the data segment: `State::new()` is all-zero',
  'and the non-zero defaults are applied in `rk*_init`, so the large buffers are zero-filled',
  'memory rather than bytes in the file. v3 went from 515.1 kB to the figure above with no',
  'change in behaviour; the three kernel suites are what says so.',
  '',
  'The chart above is `docs/bench.svg`, written by the same run as this file, so the bars and',
  'the tables cannot disagree. It is plain SVG with no scripting and no external font: it',
  'opens in any browser and renders in a markdown preview.',
  '',
].join('\n');
await writeFile(join(root, 'docs', 'bench.md'), doc);
await writeFile(join(root, 'docs', 'bench.svg'), chart());

// --- README block ------------------------------------------------------------
// The README shows the chart, so its numbers have to come from this run too: a table typed
// in by hand next to a generated picture is the one arrangement where the two are guaranteed
// to disagree eventually. Only the marked block is touched; the prose around it is hand-written.
const README_LABELS = {
  v1: 'v1 · 32 带，f32 FFT 2048',
  v2: 'v2 · 32 带，f64 FFT',
  v3: 'v3 · 32 带，FFT + CQT',
};
const sizeOf = Object.fromEntries(sizes.map(entry => [entry.name, entry]));
const rowOf = Object.fromEntries(rows.map(row => [row.key, row]));
const README_START = '<!-- bench:start';
const README_END = '<!-- bench:end -->';

const block = [
  '<!-- bench:start — 由 node scripts/bench.mjs 生成，勿手改；数字与 docs/bench.svg 同源 -->',
  '',
  '![每帧分析开销与模块体积](docs/bench.svg)',
  '',
  '| 配置 | 均值 ms | p50 ms | p95 ms | 占 60 Hz 帧 | 模块体积 |',
  '|---|---|---|---|---|---|',
  ...['v1', 'v2', 'v3'].map(key => {
    const row = rowOf[key];
    return `| ${README_LABELS[key]} | ${cell(row.mean)} | ${cell(row.p50)} | ${cell(row.p95)} | ${row.share.toFixed(1)}% | ${sizeOf[key].kB.toFixed(1)} kB |`;
  }),
  '',
  `CQT 占 v3 的 ${rowOf.cqt.share.toFixed(1)}%：${rowOf.cqt.bins} 个 bin、每帧 ${rowOf.cqt.samples.toLocaleString('en-US')} 个样本、`
    + `约 ${rowOf.cqt.nsPerSample.toFixed(1)} ns/样本。关掉它之后同一份内核是 ${rowOf['v3-off'].mean.toFixed(3)} ms，与 v2 同档。`,
  '',
  README_END,
].join('\n');

const readmePath = join(root, 'README.md');
const readme = await readFile(readmePath, 'utf8');
const startAt = readme.indexOf(README_START);
const endAt = readme.indexOf(README_END);
if (startAt < 0 || endAt < startAt) {
  throw new Error(
    'README.md 缺少 <!-- bench:start … --> / <!-- bench:end --> 标记块。\n'
    + '这一块由本脚本生成，手写的数字会和图分家，所以拒绝猜测插入位置。',
  );
}
await writeFile(readmePath, readme.slice(0, startAt) + block + readme.slice(endAt + README_END.length));

console.log('\nwrote docs/bench.md, docs/bench.svg and the README block');
