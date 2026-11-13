// Dump the rod lattice and the heights one kernel produces for one fixture, as JSON.
//
//   node scripts/height-dump.mjs <kernel.wasm> <out.json> [music|flat]
//
// The renderer is shared by every kernel, so the only thing that differs between two
// versions on screen is this table: x/z (where a rod stands), channel (which band it reads)
// and height. Drawing it offline is the only way to compare looks without a browser.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

const [, , modulePath, outPath, fixtureName = 'music'] = process.argv;
if (!modulePath || !outPath) {
  console.error('usage: node scripts/height-dump.mjs <kernel.wasm> <out.json> [music|flat]');
  process.exit(2);
}

const FFT = 2048;
const RATE = 48000;
const CHANNELS = 32;
const RODS = 709;
const BLOCKS = 60;
const GAIN = 1.4;
const SMOOTHING = 0.75;

/**
 * A mixture that behaves like mastered music: a bass-heavy tilt, two held tones and a
 * tremolo. Built once per call -- a factory, because the caller wants one closure over one
 * buffer, not a new 131k-sample buffer per sample it reads.
 */
function music(seed = 4242) {
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return (state / 0x7fffffff) * 2 - 1;
  };
  const samples = new Float32Array(1 << 17);
  let low = 0;
  let low2 = 0;
  for (let i = 0; i < samples.length; i++) {
    const white = random();
    low += 0.25 * (white - low);
    low2 += 0.05 * (low - low2);
    const t = i / RATE;
    const tremolo = 1 + 0.35 * Math.sin(2 * Math.PI * 1.7 * t);
    const tone = 0.5 * Math.sin(2 * Math.PI * 220 * t) + 0.3 * Math.sin(2 * Math.PI * 660 * t);
    samples[i] = (low * 1.4 + low2 * 2.2) * tremolo + tone;
  }
  let sumSq = 0;
  for (const value of samples) sumSq += value * value;
  const scale = 0.09 / Math.sqrt(sumSq / samples.length);
  for (let i = 0; i < samples.length; i++) samples[i] *= scale;
  return i => samples[i % samples.length];
}

const fixtures = {
  music: music(),
  /** The same mix one octave hotter: pushes bands towards the ceiling, where rings flatten. */
  flat: music(999),
};
const source = fixtures[fixtureName];
if (!source) throw new Error(`unknown fixture ${fixtureName}`);

const module = (await WebAssembly.instantiate(await readFile(modulePath), {})).instance.exports;
module.rk3_init();
module.rk3_configure(FFT, RATE);
module.rk3_set_channels(CHANNELS);

let offset = 0;
for (let block = 0; block < BLOCKS; block++) {
  const wave = new Float32Array(module.memory.buffer, module.rk3_wave_ptr(), FFT);
  for (let i = 0; i < FFT; i++) wave[i] = source(offset + i);
  offset += FFT;
  module.rk3_push();
  module.rk3_step(1 / 60, GAIN, SMOOTHING);
}

const stride = module.rk3_rod_stride();
const rodWords = new Float32Array(module.memory.buffer, module.rk3_rod_ptr(), RODS * stride);
const heights = new Float32Array(module.memory.buffer, module.rk3_heights_ptr(), RODS);

const rods = [];
for (let index = 0; index < RODS; index++) {
  rods.push({
    x: rodWords[index * stride],
    z: rodWords[index * stride + 1],
    radius: rodWords[index * stride + 2],
    channel: rodWords[index * stride + 4],
    height: heights[index],
  });
}

await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify({ module: modulePath, fixture: fixtureName, gain: GAIN, rods }));
const tall = Math.max(...rods.map(r => r.height));
const short = Math.min(...rods.map(r => r.height));
console.log(`${outPath}: ${rods.length} rods, heights ${short.toFixed(2)}..${tall.toFixed(2)}`);
