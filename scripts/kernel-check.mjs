// Verifies the wasm kernel on its own, without a browser or React.
// Run with: node app/scripts/kernel-check.mjs
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(here, '..', 'wasm', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel.wasm');
const bytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;
const memory = e.memory;

let failed = 0;
function check(label, actual, expected) {
  const pass = Object.is(actual, expected) || (typeof actual === 'number' && Math.abs(actual - expected) < 1e-6);
  console.log(`${pass ? 'PASS' : 'FAIL'} ${label}: ${actual}${pass ? '' : ` != ${expected}`}`);
  if (!pass) failed++;
}
function checkTrue(label, actual, detail = '') {
  console.log(`${actual ? 'PASS' : 'FAIL'} ${label}${detail ? `: ${detail}` : ''}`);
  if (!actual) failed++;
}

// --- shape of the ABI -------------------------------------------------------
const required = ['memory', 'rk_init', 'rk_step', 'rk_set_channels', 'rk_set_fft', 'rk_bins_ptr',
  'rk_heights_ptr', 'rk_rod_x_ptr', 'rk_rod_stride', 'rk_channel_amps_ptr', 'rk_rod_count'];
for (const name of required) checkTrue(`export ${name}`, typeof e[name] !== 'undefined');

const rods = e.rk_init();
check('rod count', rods, 709);
check('reported rod count', e.rk_rod_count(), 709);
check('rod stride in f32 words', e.rk_rod_stride(), 5);
check('default channels', e.rk_channels ? e.rk_channels() : 32, 32);

const stride = e.rk_rod_stride();
const rodView = new Float32Array(memory.buffer, e.rk_rod_x_ptr(), 709 * stride);
const heights = () => new Float32Array(memory.buffer, e.rk_heights_ptr(), 709);
const amps = () => new Float32Array(memory.buffer, e.rk_channel_amps_ptr(), 64);

const chOf = i => rodView[i * stride + 4];
const rOf = i => rodView[i * stride + 2];
// The lattice walks x from -15 upward, so rod 0 is (-15, 0) on the rim and the
// core centre sits somewhere in the middle.
check('first rod is on the rim', rOf(0), 15);
checkTrue('a centre rod exists', Array.from({ length: 709 }, (_, i) => rOf(i)).some(r => r === 0));
checkTrue('all rod channels in range', Array.from({ length: 709 }, (_, i) => chOf(i)).every(c => c >= 0 && c <= 31));

// --- channel mapping --------------------------------------------------------
e.rk_set_channels(8);
check('channels after set', e.rk_channels(), 8);
const mapped = Array.from({ length: 709 }, (_, i) => chOf(i));
check('max channel index', Math.max(...mapped), 7);
checkTrue('channel matches radius band',
  mapped.every((c, i) => c === Math.min(7, Math.floor(rOf(i) / 15 * 8))),
  'floor(r/15*n) clamped to n-1');

// --- spectrum -> channel energy --------------------------------------------
e.rk_set_fft(2048);
check('bin count', e.rk_bin_count(), 1024);
const bins = new Uint8Array(memory.buffer, e.rk_bins_ptr(), 1024);
// A single strong 1 kHz tone: bin width is 48000/2048 = 23.4375 Hz.
const toneBin = Math.round(1000 / (48000 / 2048));
bins.fill(0);
bins[toneBin] = 255;

for (let i = 0; i < 40; i++) e.rk_step(1, 48000, 1.4, 0.75);

const channelAmps = amps();
// Band 4 covers 805-1656 Hz, i.e. 38 bins, so a single full-scale tone averages
// to ~1/38. What matters is that it is the loudest band and roughly that size.
const toneBand = Math.floor(8 * Math.log(1000 / 45) / Math.log(320));
const loudest = Array.from(channelAmps).slice(0, 8).indexOf(Math.max(...Array.from(channelAmps).slice(0, 8)));
check('tone lands in the expected band', loudest, toneBand);
checkTrue('single-bin tone averages over its band', channelAmps[toneBand] > 0.02 && channelAmps[toneBand] < 0.05,
  channelAmps[toneBand].toFixed(4));
checkTrue('a narrow tone lifts the rods only slightly', Math.max(...heights()) < 0.2,
  Math.max(...heights()).toFixed(3));

// A broad spectrum (what music looks like) should drive the rods much higher.
bins.fill(180, 1, 600);
for (let i = 0; i < 40; i++) e.rk_step(1, 48000, 1.4, 0.75);
checkTrue('broad spectrum lifts the rods', Math.max(...heights()) > 1.0, Math.max(...heights()).toFixed(3));

// --- determinism ------------------------------------------------------------
const snapshot = Array.from(heights());
e.rk_init();
e.rk_set_channels(8);
e.rk_set_fft(2048);
const bins2 = new Uint8Array(memory.buffer, e.rk_bins_ptr(), 1024);
bins2[toneBin] = 255;
for (let i = 0; i < 40; i++) e.rk_step(1, 48000, 1.4, 0.75);
bins2.fill(180, 1, 600);
for (let i = 0; i < 40; i++) e.rk_step(1, 48000, 1.4, 0.75);
const replay = Array.from(heights());
checkTrue('same input reproduces same heights', replay.every((v, i) => Math.abs(v - snapshot[i]) < 1e-6));

// The representative-rod statistics were removed with the on-canvas labels;
// the ABI must no longer carry them.
checkTrue('stats export is gone', e.rk_stats_ptr === undefined && e.rk_recompute_stats === undefined);

// --- FFT inside wasm -------------------------------------------------------
// The browser now only supplies time-domain samples; everything below is Rust.
e.rk_init();
e.rk_set_fft(2048);
check('transform size', e.rk_fft_size(), 2048);
check('bin count from fft size', e.rk_bin_count(), 1024);

const wave = new Float32Array(memory.buffer, e.rk_wave_ptr(), 2048);
const binHz = 48000 / 2048;
const argmax = view => {
  let best = 0;
  let bestIndex = 0;
  for (let i = 0; i < view.length; i++) if (view[i] > best) { best = view[i]; bestIndex = i; }
  return { best, bestIndex };
};

wave.fill(0);
for (let i = 0; i < 40; i++) e.rk_analyse();
checkTrue('silence gives an empty spectrum', bins.every(value => value === 0), 'all zero after 40 frames');

for (let i = 0; i < 2048; i++) wave[i] = 0.01 * Math.sin((2 * Math.PI * 1000 * i) / 48000);
for (let i = 0; i < 80; i++) e.rk_analyse();
const tone = argmax(bins);
// 1000 Hz / 23.4375 Hz = 42.67, so it must land on bin 42 or 43.
checkTrue('1 kHz tone peaks on the right bin', Math.abs(tone.bestIndex - 1000 / binHz) <= 1,
  `bin ${tone.bestIndex} (expected ~${(1000 / binHz).toFixed(2)})`);
checkTrue('tone magnitude is in a sane byte range', tone.best > 150 && tone.best < 240, `byte ${tone.best}`);

// Re-run the analysis on the same samples: smoothing must converge, not drift.
const converged = bins[tone.bestIndex];
for (let i = 0; i < 10; i++) e.rk_analyse();
check('smoothing converges', bins[tone.bestIndex], converged);

// A different frequency must move the peak, proving the transform is not glue.
wave.fill(0);
for (let i = 0; i < 2048; i++) wave[i] = 0.01 * Math.sin((2 * Math.PI * 4000 * i) / 48000);
for (let i = 0; i < 80; i++) e.rk_analyse();
const higher = argmax(bins);
checkTrue('4 kHz tone peaks near bin 171', Math.abs(higher.bestIndex - 4000 / binHz) <= 1,
  `bin ${higher.bestIndex} (expected ~${(4000 / binHz).toFixed(2)})`);

// A constant offset is not spectral content. The kernel subtracts the window
// mean before the FFT, so a biased source leaves the low bins empty instead of
// parking a "0 Hz" peak there — which is exactly what the readouts used to show.
for (let i = 0; i < 2048; i++) wave[i] = 0.5;
for (let i = 0; i < 80; i++) e.rk_analyse();
checkTrue('a DC offset leaves the spectrum empty', bins.every(value => value === 0),
  `max byte ${Math.max(...bins)}`);

// And the offset must not hide a real tone riding on top of it.
for (let i = 0; i < 2048; i++) wave[i] = 0.5 + 0.01 * Math.sin((2 * Math.PI * 1000 * i) / 48000);
for (let i = 0; i < 80; i++) e.rk_analyse();
const biased = argmax(bins);
checkTrue('a tone under a DC offset still peaks on its own bin',
  Math.abs(biased.bestIndex - 1000 / binHz) <= 1,
  `bin ${biased.bestIndex} (expected ~${(1000 / binHz).toFixed(2)})`);

// Halving the transform size halves the bins and keeps the peak frequency.
e.rk_set_fft(1024);
check('fft resized', e.rk_fft_size(), 1024);
check('bins follow fft size', e.rk_bin_count(), 512);
const smallWave = new Float32Array(memory.buffer, e.rk_wave_ptr(), 1024);
for (let i = 0; i < 1024; i++) smallWave[i] = 0.01 * Math.sin((2 * Math.PI * 1000 * i) / 48000);
for (let i = 0; i < 80; i++) e.rk_analyse();
const resized = argmax(bins.subarray(0, 512));
checkTrue('same frequency, half the bins', Math.abs(resized.bestIndex - 1000 / (48000 / 1024)) <= 1,
  `bin ${resized.bestIndex} (expected ~${(1000 / (48000 / 1024)).toFixed(2)})`);
e.rk_set_fft(2048);

// Determinism: identical samples must produce identical bytes.
wave.fill(0);
for (let i = 0; i < 2048; i++) wave[i] = 0.02 * Math.sin((2 * Math.PI * 440 * i) / 48000);
for (let i = 0; i < 40; i++) e.rk_analyse();
const firstPass = Array.from(bins.subarray(0, 1024));
e.rk_set_fft(2048);
for (let i = 0; i < 40; i++) e.rk_analyse();
const secondPass = Array.from(bins.subarray(0, 1024));
checkTrue('analysis is deterministic', firstPass.every((value, i) => value === secondPass[i]));

console.log(failed === 0 ? '\nALL KERNEL CHECKS PASSED' : `\n${failed} KERNEL CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
