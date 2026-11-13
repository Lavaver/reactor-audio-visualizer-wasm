// Verifies the v2 kernel on its own, without a browser or React.
// Run with: node scripts/kernel-check-v2.mjs
//
// These assertions are about *physics and precision*: absolute levels in dBFS,
// Parseval to within a part in 1e12, and a peak frequency that must not be
// quantised to the bin grid. Everything JS reads as data is an `f64` view over the
// module's memory — the same bytes the kernel computed, with no conversion step in
// between — so the tolerances below are limited by the maths, not by the transport.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(here, '..', 'wasm_v2', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel_v2.wasm');
const bytes = await readFile(wasmPath);
const module = await WebAssembly.compile(bytes);
const instance = await WebAssembly.instantiate(module, {});
const e = instance.exports;

let failed = 0;
function check(label, actual, expected) {
  const pass = Object.is(actual, expected);
  console.log(`${pass ? 'PASS' : 'FAIL'} ${label}: ${actual}${pass ? '' : ` != ${expected}`}`);
  if (!pass) failed++;
}
function near(label, actual, expected, tolerance) {
  const pass = Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance;
  console.log(
    `${pass ? 'PASS' : 'FAIL'} ${label}: ${Number(actual).toPrecision(17)} (want ${Number(expected).toPrecision(17)} ±${tolerance})`,
  );
  if (!pass) failed++;
}
/** Relative error, for assertions that should scale with the magnitude. */
function nearRelative(label, actual, expected, relative) {
  const tolerance = Math.abs(expected) * relative;
  const pass = Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance;
  const error = expected === 0 ? Math.abs(actual) : Math.abs((actual - expected) / expected);
  console.log(`${pass ? 'PASS' : 'FAIL'} ${label}: rel err ${error.toExponential(2)} (want ≤ ${relative.toExponential(1)})`);
  if (!pass) failed++;
}
function checkTrue(label, actual, detail = '') {
  console.log(`${actual ? 'PASS' : 'FAIL'} ${label}${detail ? `: ${detail}` : ''}`);
  if (!actual) failed++;
}

// Every view is rebuilt on demand. `rk2_push` hands a block to the channel, which
// allocates, and a wasm memory growth detaches every existing view — a cached
// typed array would silently start reporting nothing.
const memory = () => e.memory;
const waveView = () => new Float32Array(memory().buffer, e.rk2_wave_ptr(), e.rk2_fft_size());
/** The raw measurement: linear power per bin, no log, no narrowing. */
const powerView = () => new Float64Array(memory().buffer, e.rk2_bin_power_ptr(), e.rk2_bin_count());
const windowView = () => new Float64Array(memory().buffer, e.rk2_window_ptr(), e.rk2_fft_size());
const dbView = () => new Float64Array(memory().buffer, e.rk2_bin_db_ptr(), e.rk2_bin_count());
const barsView = () => new Float64Array(memory().buffer, e.rk2_bars_ptr(), e.rk2_bin_count());
const bandRmsView = () => new Float64Array(memory().buffer, e.rk2_band_rms_ptr(), 64);
const ampView = () => new Float64Array(memory().buffer, e.rk2_channel_amps_ptr(), 64);
/** After the auto-gain and attack/release: what the rods actually follow. */
const driveView = () => new Float64Array(memory().buffer, e.rk2_band_drive_ptr(), 64);
const report = () => new Float64Array(memory().buffer, e.rk2_report_ptr(), 4);
// The two arrays that stay f32, because the GPU is the consumer.
const heightView = () => new Float32Array(memory().buffer, e.rk2_heights_ptr(), e.rk2_rod_count());
const rodsView = () => new Float32Array(memory().buffer, e.rk2_rod_ptr(), e.rk2_rod_count() * e.rk2_rod_stride());
const [PEAK_HZ, CENTROID_HZ, RMS_DB, PEAK_DB] = [0, 1, 2, 3];
const SAMPLE_RATE = 48000;
const binHz = SAMPLE_RATE / 2048;
/** Bin 42: exactly 42 cycles fit in the block, so the block has no DC bias. */
const ON_BIN_HZ = 42 * binHz;

const stride = 5;
const radiusOf = index => rodsView()[index * stride + 2];
const channelOf = index => rodsView()[index * stride + 4];
const seedOf = index => rodsView()[index * stride + 3];

/** Fill the staging block with a sine and hand it to the asynchronous stage. */
function pushTone(frequency, amplitude) {
  const wave = waveView();
  for (let i = 0; i < wave.length; i++) {
    wave[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / SAMPLE_RATE);
  }
  return e.rk2_push();
}
function pushSilence() {
  waveView().fill(0);
  return e.rk2_push();
}
/** Σ bin power, straight out of the f64 view. By Parseval this is the mean square. */
function totalPower() {
  let sum = 0;
  for (const value of powerView()) sum += value;
  return sum;
}
/** Deterministic pseudo-random source, so noise tests are reproducible. */
function noise(amplitude, seed = 22222) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return amplitude * ((state / 0x7fffffff) * 2 - 1);
  };
}
/** Fill the staging block from a generator and push it. */
function pushGenerator(generator) {
  const wave = waveView();
  for (let i = 0; i < wave.length; i++) wave[i] = generator();
  return e.rk2_push();
}
const stepFrames = (count, dt = 1 / 60) => {
  for (let i = 0; i < count; i++) e.rk2_step(dt, 1.4, 0.75);
};
/** The 0..1 band amount for a tone of amplitude A in band `channel` of 8. */
const bandAmount = amplitude => (20 * Math.log10(amplitude / Math.SQRT2) + 60) / 60;
/** Rod height the easing converges to for a given band amount. */
const rodTarget = (amplitude, gain = 1.4) => 0.07 + Math.pow(bandAmount(amplitude), 1.7) * gain * 4.8;

// --- ABI shape --------------------------------------------------------------
const required = ['memory', 'rk2_version', 'rk2_init', 'rk2_configure', 'rk2_push', 'rk2_step',
  'rk2_reset', 'rk2_clear_spectrum', 'rk2_set_channels', 'rk2_channels', 'rk2_bin_count',
  'rk2_fft_size', 'rk2_rod_count', 'rk2_rod_stride', 'rk2_rod_ptr', 'rk2_heights_ptr',
  'rk2_channel_amps_ptr', 'rk2_band_rms_ptr', 'rk2_band_drive_ptr', 'rk2_wave_ptr',
  'rk2_bin_power_ptr',
  'rk2_window_ptr', 'rk2_bin_db_ptr', 'rk2_bars_ptr', 'rk2_report_ptr', 'rk2_sine_ref',
  'rk2_processed', 'rk2_dropped'];
for (const name of required) checkTrue(`export ${name}`, typeof e[name] !== 'undefined');
check('version', e.rk2_version(), 2);

// --- geometry matches v1 ----------------------------------------------------
check('rod count', e.rk2_init(), 709);
check('rod stride in f32 words', e.rk2_rod_stride(), 5);
near('first rod sits on the rim', radiusOf(0), 15, 0);
checkTrue('a centre rod exists', Array.from({ length: 709 }, (_, i) => radiusOf(i)).includes(0));
// The seed hash is the v1 expression verbatim, so colours do not shift on a swap.
near('seed hash matches v1', seedOf(0), Math.abs(Math.sin(-15 * 127.1 + 0 * 311.7)), 1e-6);

// --- the switch really is free ----------------------------------------------
// The renderer uploads the rod lattice to the GPU once, at creation, and both
// backends are supposed to produce it identically — that is what lets the front
// end swap kernels without rebuilding any buffer. Verified against the real v1
// module rather than against a comment.
try {
  const v1Bytes = await readFile(join(here, '..', 'wasm', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel.wasm'));
  // `instantiate` on a compiled Module returns the instance itself.
  const v1 = (await WebAssembly.instantiate(await WebAssembly.compile(v1Bytes), {})).exports;
  v1.rk_init();
  v1.rk_set_channels(32);
  e.rk2_set_channels(32);
  const v1Rods = new Float32Array(v1.memory.buffer, v1.rk_rod_x_ptr(), v1.rk_rod_count() * v1.rk_rod_stride());
  const v2Rods = rodsView();
  check('v1 and v2 agree on the rod count', v1.rk_rod_count(), e.rk2_rod_count());
  check('v1 and v2 agree on the stride', v1.rk_rod_stride(), e.rk2_rod_stride());
  let mismatches = 0;
  for (let i = 0; i < v2Rods.length; i++) if (v1Rods[i] !== v2Rods[i]) mismatches++;
  check('v1 and v2 build a bit-identical rod lattice', mismatches, 0);
} catch (error) {
  console.log(`SKIP rod-lattice cross-check (v1 kernel not built): ${error.message}`);
}

// Before configuration there is no pipeline, so a push must be a no-op.
check('push before configure is rejected', pushSilence(), 0);

// --- configuration ----------------------------------------------------------
e.rk2_configure(2048, SAMPLE_RATE);
check('transform size', e.rk2_fft_size(), 2048);
check('bin count', e.rk2_bin_count(), 1024);
checkTrue('sine reference is a plausible peak bin', e.rk2_sine_ref() > 0.2 && e.rk2_sine_ref() < 0.3,
  e.rk2_sine_ref().toPrecision(17));

// --- the f64 memory contract ------------------------------------------------
// A Float64Array view demands an 8-byte base offset; wasm would throw on a
// misaligned pointer, but asserting it here says *why* it is safe rather than
// relying on the throw.
for (const [name, ptr, length] of [
  ['bin_power', e.rk2_bin_power_ptr(), e.rk2_bin_count()],
  ['bin_db', e.rk2_bin_db_ptr(), e.rk2_bin_count()],
  ['bars', e.rk2_bars_ptr(), e.rk2_bin_count()],
  ['band_rms', e.rk2_band_rms_ptr(), 64],
  ['channel_amps', e.rk2_channel_amps_ptr(), 64],
  ['report', e.rk2_report_ptr(), 4],
]) {
  check(`${name} pointer is 8-byte aligned`, ptr % 8, 0);
  checkTrue(`${name} view fits inside wasm memory`, ptr + length * 8 <= memory().buffer.byteLength,
    `${ptr} + ${length * 8} ≤ ${memory().buffer.byteLength}`);
}
// JS numbers are f64, so a scalar export carries the full width too.
const sineRef = e.rk2_sine_ref();
checkTrue('sine_ref is a full-precision f64', Number.isFinite(sineRef) && sineRef !== Math.fround(sineRef),
  sineRef.toPrecision(17));

// A view over wasm memory must be live: no copy step, so the numbers change under
// the same array object as the kernel writes them.
const livePower = powerView();
pushSilence();
const silentSum = livePower.reduce((sum, value) => sum + value, 0);
pushTone(ON_BIN_HZ, 0.5);
const liveSum = livePower.reduce((sum, value) => sum + value, 0);
checkTrue('the f64 view is live, not a snapshot', silentSum === 0 && liveSum > 0.1,
  `${silentSum} -> ${liveSum.toPrecision(17)}`);

// --- the asynchronous stage actually runs -----------------------------------
// This is the one check that proves the Tokio plumbing: the block is handed to the
// channel, and the counter only advances if the consumer task was polled, took the
// block and ran the analysis.
const processedBefore = e.rk2_processed();
check('push moves exactly one block', pushSilence(), 1);
check('processed counter follows', e.rk2_processed(), processedBefore + 1);
check('nothing was dropped', e.rk2_dropped(), 0);
check('the queue is drained every call', pushSilence(), 1);
check('configure is idempotent', e.rk2_configure(2048, SAMPLE_RATE), undefined);
check('a re-configure does not lose blocks', pushSilence(), 1);

// --- silence ----------------------------------------------------------------
e.rk2_reset();
for (let i = 0; i < 8; i++) pushSilence();
stepFrames(30);
checkTrue('silence reports the silence sentinel', report()[RMS_DB] <= -143,
  report()[RMS_DB].toPrecision(17));
check('silence has no peak frequency', report()[PEAK_HZ], 0);
checkTrue('silence leaves every bar empty', Array.from(barsView()).every(v => v === 0));

// --- absolute levels --------------------------------------------------------
// A sine of amplitude A has peak 20·log10(A) and RMS 10·log10(A²/2).
//
// Level checks use an on-bin tone (42 cycles in the block) so the measurement is
// not disturbed by the block boundary: an off-bin tone ends mid-cycle, its mean is
// then genuinely non-zero, and removing that DC shifts the peak by ~0.05 dB.
//
// The tolerance is 1e-6 dB, not 1e-12, and the reason is *not* the analysis: the
// staging buffer is `f32` because `AnalyserNode.getFloatTimeDomainData` has no
// other precision to offer, so every sample arrives already rounded to 2^-24
// relative. f32-in / f64-out means the source sets the floor; the arithmetic
// applied to it is exact to about 1e-13 and would show as a much smaller error on
// an exact input.
const amplitudeOf = decibels => Math.pow(10, decibels / 20);
e.rk2_reset();
pushTone(ON_BIN_HZ, 0.5);
near('0.5 amplitude peak level', report()[PEAK_DB], 20 * Math.log10(0.5), 1e-6);
near('0.5 amplitude RMS level', report()[RMS_DB], 10 * Math.log10(0.25 / 2), 1e-6);
near('measured peak amplitude round-trips', amplitudeOf(report()[PEAK_DB]), 0.5, 1e-6);

// Parseval, checked against the *exact* identity the kernel implements:
//   Σ bin_power = Σ w²y² / Σ w²
// where y is the DC-removed block and w the window actually in use. Recomputed
// here from the f64 views, so the residual measures the arithmetic rather than a
// modelling assumption. This is the assertion an f32 output path could not pass.
{
  const power = powerView();
  const window = windowView();
  const samples = waveView();
  const n = samples.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += samples[i];
  mean /= n;
  let weighted = 0;
  let windowSq = 0;
  for (let i = 0; i < n; i++) {
    const w = window[i];
    const centred = samples[i] - mean;
    weighted += w * w * centred * centred;
    windowSq += w * w;
  }
  let measured = 0;
  for (const value of power) measured += value;
  nearRelative('bin powers satisfy Parseval for the real window', measured, weighted / windowSq, 1e-12);
}

// The output arrays must genuinely carry f64: a value that survives as f32 would
// be representable at 24 bits of mantissa, and these are not.
{
  const db = dbView();
  const wide = Array.from(db).filter(value => Math.abs(value) > 1 && value !== Math.fround(value));
  checkTrue('bin_db carries more precision than an f32 could hold', wide.length > 100,
    `${wide.length} of ${db.length} bins differ from their f32 rounding`);
  const peak = report()[PEAK_HZ];
  checkTrue('the peak frequency is not f32-rounded', peak !== Math.fround(peak), peak.toPrecision(17));
}

// The dB array must be the exact logarithm of the power array, with the same
// floor the kernel applies so `log10(0)` never reaches a display.
{
  const reference = e.rk2_sine_ref();
  const power = powerView();
  const db = dbView();
  const EPS = 1e-20;
  let worst = 0;
  for (let bin = 0; bin < power.length; bin++) {
    const expected = 10 * Math.log10(Math.max(power[bin] / reference, EPS));
    worst = Math.max(worst, Math.abs(db[bin] - expected));
  }
  checkTrue('bin_db is the exact logarithm of bin_power', worst < 1e-12, `max Δ ${worst.toExponential(2)} dB`);
}

e.rk2_reset();
pushTone(ON_BIN_HZ, 1.0);
near('full-scale peak level is 0 dBFS', report()[PEAK_DB], 0, 1e-6);
near('full-scale RMS level', report()[RMS_DB], 10 * Math.log10(1 / 2), 1e-6);

// Off the bin grid the meter is still within a tenth of a dB; the residual is the
// partial cycle at the block edge, which DC removal cannot invent back.
e.rk2_reset();
pushTone(1000, 0.5);
near('an off-bin tone meters within 0.1 dB', report()[PEAK_DB], 20 * Math.log10(0.5), 0.1);

// --- frequency precision ----------------------------------------------------
// The bin grid is 23.4375 Hz, so a raw argmax can only be that accurate. These
// bounds are a fraction of a bin, which is only reachable with interpolation.
for (const [frequency, label] of [[1000, '1 kHz'], [1005, '1005 Hz'], [4000, '4 kHz'], [440, '440 Hz'],
  [60, '60 Hz']]) {
  e.rk2_reset();
  pushTone(frequency, 0.5);
  const measured = report()[PEAK_HZ];
  near(`${label} peak frequency`, measured, frequency, 3);
  checkTrue(`${label} peak frequency is not an integer-bin value`,
    Number.isFinite(measured) && measured % binHz !== 0, `${measured.toFixed(6)} Hz`);
  const naive = Math.round(frequency / binHz) * binHz;
  if (Math.abs(naive - frequency) > 1) {
    checkTrue(`${label} beats the raw bin grid`,
      Math.abs(measured - frequency) < Math.abs(naive - frequency),
      `interpolated ${measured.toFixed(6)} Hz vs bin ${naive.toFixed(2)} Hz`);
  }
}

// Centroid of a single tone is that tone.
e.rk2_reset();
pushTone(1000, 0.5);
near('centroid of a lone tone', report()[CENTROID_HZ], 1000, 10);

// A high tone must move the centroid far above a low one.
e.rk2_reset();
pushTone(300, 0.5);
const lowCentroid = report()[CENTROID_HZ];
e.rk2_reset();
pushTone(6000, 0.5);
checkTrue('centroid follows the tone upward', report()[CENTROID_HZ] > lowCentroid * 4,
  `${lowCentroid.toFixed(0)} Hz -> ${report()[CENTROID_HZ].toFixed(0)} Hz`);

// --- band integration -------------------------------------------------------
e.rk2_init();
e.rk2_configure(2048, SAMPLE_RATE);
e.rk2_set_channels(8);
check('channels after set', e.rk2_channels(), 8);
const mapped = Array.from({ length: 709 }, (_, i) => channelOf(i));
check('max channel index', Math.max(...mapped), 7);
checkTrue('channel matches radius band',
  mapped.every((c, i) => c === Math.min(7, Math.floor(radiusOf(i) / 15 * 8))),
  'floor(r/15*n) clamped to n-1');

e.rk2_reset();
pushTone(1000, 0.5);
const amps = Array.from(ampView()).slice(0, 8);
// band(k) = 45 Hz · 320^(k/8); 1 kHz falls in band 4, which spans 805–1740 Hz.
check('tone lands in the expected band', amps.indexOf(Math.max(...amps)), 4);

// The measurement is what the readouts and the tests trust, so it is pinned
// exactly: band RMS is the tone's own RMS, and a 4× quieter tone is 12.04 dB down.
near('band RMS equals the tone RMS', bandRmsView()[4], 0.5 / Math.SQRT2, 1e-6);
e.rk2_reset();
pushTone(1000, 0.125);
near('band RMS tracks a 4× quieter tone', bandRmsView()[4], 0.125 / Math.SQRT2, 1e-6);

// `channel_amps` is the *display* amount, deliberately shaped (floor, tilt, gamma,
// auto-gain). Pinning its exact value would mean mirroring tuning constants into
// the test, so it is checked for the properties the visuals depend on instead.
const amountFor = amplitude => {
  e.rk2_reset();
  pushTone(1000, amplitude);
  return ampView()[4];
};
const driveFor = amplitude => {
  e.rk2_reset();
  pushTone(1000, amplitude);
  stepFrames(200);
  return driveView()[4];
};

e.rk2_reset();
pushTone(1000, 0.5);
const toneAmps = Array.from(ampView()).slice(0, 8);
checkTrue('the tone dominates its neighbours', toneAmps[3] < toneAmps[4] * 0.6 && toneAmps[5] < toneAmps[4] * 0.6,
  `${toneAmps[3].toFixed(3)} / ${toneAmps[4].toFixed(3)} / ${toneAmps[5].toFixed(3)}`);

const levels = [0.0025, 0.01, 0.05, 0.2, 0.8];
const amounts = levels.map(amountFor);
checkTrue('the display amount rises monotonically with level',
  amounts.every((value, index) => index === 0 || value >= amounts[index - 1]),
  amounts.map(v => v.toFixed(3)).join(' < '));
// `channel_amps` is the curve *before* the auto-gain, so a block 50 dB down is
// expected to read low here. What makes it visible is the auto-gain, and that is
// asserted below on `band_drive` — the array the rods actually follow.
checkTrue('before the auto-gain a very quiet band reads low', amounts[0] < 0.1,
  amounts[0].toFixed(3));
checkTrue('a loud band reaches the top', amounts.at(-1) > 0.9, amounts.at(-1).toFixed(3));

// The tilt is the reason the outer rings move at all: same level, higher band,
// more display amount. Without it a 9 kHz band reads ~20 dB below the bass, which
// the 5 dB tilt cannot fully recover — but it lifts it far enough to be visible.
// Band edges at 8 channels: … 3533, 7310, 15127 Hz, so 9 kHz is band 7.
const tiltLifts = (() => {
  e.rk2_reset();
  pushTone(1000, 0.05);
  const low = ampView()[4];
  e.rk2_reset();
  pushTone(9000, 0.05);
  const high = ampView()[7];
  return { low, high };
})();
checkTrue('the tilt lets a high band reach a useful height', tiltLifts.high > 0.5,
  `1 kHz ${tiltLifts.low.toFixed(3)} vs 9 kHz ${tiltLifts.high.toFixed(3)}`);

// The auto-gain is the tension lever: a block 30 dB down must still drive the
// lattice hard, or the reactor sits dead through quiet passages.
const quietDrive = driveFor(0.0025);
const loudDrive = driveFor(0.8);
const silenceDrive = (() => {
  e.rk2_reset();
  pushSilence();
  stepFrames(200);
  return driveView()[4];
})();
checkTrue('the auto-gain lifts a quiet block', quietDrive > 0.5 * loudDrive,
  `quiet ${quietDrive.toFixed(3)} vs loud ${loudDrive.toFixed(3)}`);
checkTrue('the auto-gain leaves silence dark', silenceDrive === 0, silenceDrive.toFixed(3));

// --- DC rejection -----------------------------------------------------------
e.rk2_reset();
waveView().fill(0.5);
e.rk2_push();
checkTrue('a DC offset leaves the spectrum empty', totalPower() < 1e-9,
  `Σ power ${totalPower().toExponential(2)}`);
checkTrue('a DC offset is not reported as level', report()[RMS_DB] <= -143,
  report()[RMS_DB].toPrecision(17));
checkTrue('a DC offset leaves the bands empty', Array.from(ampView()).every(v => v === 0));
// ...while a tone riding on that offset survives.
e.rk2_reset();
{
  const wave = waveView();
  for (let i = 0; i < wave.length; i++) {
    wave[i] = 0.5 + 0.01 * Math.sin((2 * Math.PI * 1000 * i) / SAMPLE_RATE);
  }
  e.rk2_push();
}
near('a tone under a DC offset keeps its pitch', report()[PEAK_HZ], 1000, 3);

// --- frame-rate independent smoothing ---------------------------------------
// The same wall-clock time must ease the bars to the same place whether it
// arrived as 60 or 120 frames.
const settleBars = (frames, dt) => {
  e.rk2_reset();
  pushTone(ON_BIN_HZ, 0.5);
  stepFrames(frames, dt);
  return Array.from(barsView());
};
const at60 = settleBars(30, 1 / 60);
const at120 = settleBars(60, 1 / 120);
const worstDelta = Math.max(...at60.map((value, i) => Math.abs(value - at120[i])));
checkTrue('bar easing is frame-rate independent', worstDelta < 0.01, `max Δ ${worstDelta.toFixed(4)}`);

// Repeated identical frames must converge, not drift or overshoot.
e.rk2_reset();
pushTone(ON_BIN_HZ, 0.5);
stepFrames(90);
const convergedBars = Array.from(barsView());
pushTone(ON_BIN_HZ, 0.5);
stepFrames(90);
const drift = Math.max(...convergedBars.map((value, i) => Math.abs(value - barsView()[i])));
checkTrue('bars converge on a steady tone', drift < 1e-9, `max Δ ${drift.toExponential(2)}`);
// The bar value is the calibrated bin level mapped onto the display window:
// `(db - min) / (max - min)`. Both constants must match wasm_v2/src/lib.rs.
const DISPLAY_MIN_DB = -100;
const DISPLAY_MAX_DB = 0;
const barFor = decibels => (decibels - DISPLAY_MIN_DB) / (DISPLAY_MAX_DB - DISPLAY_MIN_DB);
near('a half-scale tone fills its bars to −6 dB', Math.max(...convergedBars),
  barFor(20 * Math.log10(0.5)), 1e-6);
e.rk2_reset();
pushTone(ON_BIN_HZ, 1.0);
stepFrames(90);
near('a full-scale tone fills the bars completely', Math.max(...barsView()), 1, 1e-6);
checkTrue('bars stay inside 0..1', convergedBars.every(v => v >= 0 && v <= 1));

// The 48-bar spectrum must stay readable at ordinary levels.
//
// This is the regression the display window caused once already: a full-scale sine
// puts its whole energy in one bin, but real per-bin content is spread out and sits
// 40–60 dB lower. A −60 dB floor left the loudest bar of a −30 dBFS block at 0.24
// and of a −40 dBFS block at 0.08 — a flat stub where a spectrum used to be.
for (const [amplitude, rmsDb, minimum] of [[0.173, -20, 0.35], [0.052, -30, 0.3], [0.017, -40, 0.25]]) {
  e.rk2_reset();
  pushGenerator(noise(amplitude));
  stepFrames(90);
  const bars = Array.from(barsView());
  const loudest = Math.max(...bars);
  const visible = bars.filter(v => v > 0.02).length;
  checkTrue(`a ${rmsDb} dBFS broadband block still has a spectrum`, loudest > minimum,
    `loudest bar ${loudest.toFixed(3)}`);
  checkTrue(`a ${rmsDb} dBFS broadband block lights most bins`, visible > 800,
    `${visible} of ${bars.length} bars visible`);
}
// Silence must still be silent: the wider window must not lift the floor.
e.rk2_reset();
pushSilence();
stepFrames(60);
checkTrue('the wider window does not lift silence', Array.from(barsView()).every(v => v === 0));

// --- rods -------------------------------------------------------------------
// The rods follow `band_drive`, which is the display curve plus the auto-gain, so
// their absolute height is a tuning choice rather than a measurement. What must
// hold is the *shape* of the response, and the claim the visuals rest on: the whole
// lattice answers, not just the rings near the loud bands.
e.rk2_reset();
pushTone(1000, 0.9);
stepFrames(180);
const loudHeights = Array.from(heightView());
checkTrue('a loud tone lifts the rods high', Math.max(...loudHeights) > 5,
  Math.max(...loudHeights).toFixed(2));

e.rk2_reset();
pushTone(1000, 0.02);
stepFrames(180);
const quietHeights = Array.from(heightView());
// 33 dB quieter, and thanks to the auto-gain it still reaches a useful height —
// that is the point, and it is why a quiet passage no longer leaves the core dead.
checkTrue('a quiet tone still drives the rods', Math.max(...quietHeights) > 2,
  Math.max(...quietHeights).toFixed(2));
checkTrue('a louder tone is at least as tall', Math.max(...loudHeights) >= Math.max(...quietHeights),
  `${Math.max(...quietHeights).toFixed(2)} vs ${Math.max(...loudHeights).toFixed(2)}`);
checkTrue('rod heights are finite and non-negative', quietHeights.every(v => Number.isFinite(v) && v >= 0));
checkTrue('rods rest above the baseline', Math.min(...quietHeights) >= 0);

// A pure tone occupies one band, so only the rods inside that band's rings move —
// the rest hold the baseline. This is the honest structure the tilt must not
// destroy by flattening everything.
checkTrue('a lone tone moves one ring, not the whole disc',
  quietHeights.filter(v => v > 1).length < quietHeights.length / 2,
  `${quietHeights.filter(v => v > 1).length} of ${quietHeights.length} rods raised`);

// A broadband block is what music looks like: most bands light up at once.
e.rk2_reset();
{
  const wave = waveView();
  for (let i = 0; i < wave.length; i++) {
    wave[i] = 0.02 * (Math.sin((2 * Math.PI * 120 * i) / SAMPLE_RATE)
      + Math.sin((2 * Math.PI * 700 * i) / SAMPLE_RATE)
      + Math.sin((2 * Math.PI * 2500 * i) / SAMPLE_RATE)
      + Math.sin((2 * Math.PI * 7000 * i) / SAMPLE_RATE));
  }
  e.rk2_push();
}
stepFrames(180);
const broadHeights = Array.from(heightView());
const raised = (heights, floor) => heights.filter(value => value > floor).length;
checkTrue('a broadband block raises far more rods', raised(broadHeights, 1.0) > raised(quietHeights, 1.0) * 2,
  `${raised(quietHeights, 1.0)} -> ${raised(broadHeights, 1.0)} rods`);
checkTrue('the broadband block activates several bands',
  Array.from(ampView()).slice(0, 8).filter(v => v > 0.1).length >= 3,
  Array.from(ampView()).slice(0, 8).map(v => v.toFixed(2)).join(' '));

// --- v2 must out-tension v1 -------------------------------------------------
// This is the requirement the display curve exists to meet, so it is asserted
// rather than described: on identical samples, with the gain and smoothing the app
// uses, v2's rods must reach higher and move in greater numbers than v1's.
try {
  const v1Bytes = await readFile(join(here, '..', 'wasm', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel.wasm'));
  const v1 = (await WebAssembly.instantiate(await WebAssembly.compile(v1Bytes), {})).exports;
  v1.rk_init();
  v1.rk_set_channels(32);
  v1.rk_set_fft(2048);
  const v1Wave = new Float32Array(v1.memory.buffer, v1.rk_wave_ptr(), 2048);
  const v1Heights = () => new Float32Array(v1.memory.buffer, v1.rk_heights_ptr(), v1.rk_rod_count());

  e.rk2_init();
  e.rk2_configure(2048, SAMPLE_RATE);
  e.rk2_set_channels(32);
  e.rk2_reset();

  // Pseudo-music: noise low-passed twice, which gives the 1/f slope real music has
  // once you stop looking at individual notes.
  const blockLength = 2048;
  const samples = new Float32Array(blockLength * 60);
  let state = 777;
  const next = () => {
    state = (state * 1664525 + 1013904223) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  let low = 0;
  let low2 = 0;
  for (let i = 0; i < samples.length; i++) {
    const white = next() * 2 - 1;
    low += 0.25 * (white - low);
    low2 += 0.05 * (low - low2);
    samples[i] = low * 1.4 + low2 * 2.2;
  }
  // Normalise to −20 dBFS, a loud but ordinary master.
  let sumSq = 0;
  for (const value of samples) sumSq += value * value;
  const scale = 0.1 / Math.sqrt(sumSq / samples.length);
  for (let i = 0; i < samples.length; i++) samples[i] *= scale;

  const wave = waveView();
  for (let block = 0; block < 60; block++) {
    const offset = block * blockLength;
    for (let i = 0; i < blockLength; i++) {
      v1Wave[i] = samples[offset + i];
      wave[i] = samples[offset + i];
    }
    v1.rk_analyse();
    v1.rk_step(1, SAMPLE_RATE, 1.4, 0.75);
    e.rk2_push();
    e.rk2_step(1 / 60, 1.4, 0.75);
  }

  const v1Values = Array.from(v1Heights());
  const v2Values = Array.from(heightView());
  const mean = values => values.reduce((total, value) => total + value, 0) / values.length;
  const moving = values => values.filter(value => value > 1).length;

  checkTrue('v2 reaches at least as high as v1', Math.max(...v2Values) >= Math.max(...v1Values),
    `v1 ${Math.max(...v1Values).toFixed(2)} vs v2 ${Math.max(...v2Values).toFixed(2)}`);
  checkTrue('v2 drives the lattice harder on average', mean(v2Values) > mean(v1Values) * 1.2,
    `v1 ${mean(v1Values).toFixed(2)} vs v2 ${mean(v2Values).toFixed(2)}`);
  checkTrue('v2 moves at least as many rods', moving(v2Values) >= moving(v1Values),
    `v1 ${moving(v1Values)} vs v2 ${moving(v2Values)} of ${v2Values.length}`);
} catch (error) {
  console.log(`SKIP v1 tension comparison (v1 kernel not built): ${error.message}`);
}

// --- a kernel switch interpolates, it does not cut ---------------------------
// The engine seeds this kernel's heights with the shape the previous version had
// reached, then eases at 0.94 for 0.6 s. Both halves are checked here: the seeded
// field must survive, and the easing must travel the whole way.
e.rk2_reset();
pushTone(1000, 0.5);
heightView().fill(3.0);
e.rk2_step(1 / 60, 1.4, 0.94);
const afterOneFrame = Array.from(heightView());
checkTrue('a seeded rod field survives the first frame',
  Math.max(...afterOneFrame) < 3.0 * 1.1 && Math.min(...afterOneFrame) > 3.0 * 0.85,
  `${Math.min(...afterOneFrame).toFixed(3)} … ${Math.max(...afterOneFrame).toFixed(3)} from 3.000`);

// The seed is what makes the difference. The same single step taken from the floor
// barely lifts anything, because the band's own 35 ms attack has only just started.
e.rk2_reset();
pushTone(1000, 0.5);
e.rk2_step(1 / 60, 1.4, 0.94);
const fromFloor = Math.max(...heightView());
checkTrue('the seeded field starts where the old kernel left off, not at the floor',
  Math.max(...afterOneFrame) > fromFloor * 5,
  `seeded ${Math.max(...afterOneFrame).toFixed(3)} vs from floor ${fromFloor.toFixed(3)}`);

// One frame of the slowed easing covers ~5% of the distance, where the user's
// default (~15%) would read as a cut.
heightView().fill(3.0);
const ceiling = (() => {
  for (let i = 0; i < 200; i++) e.rk2_step(1 / 60, 1.4, 0.94);
  return Math.max(...heightView());
})();
const firstStep = Math.abs(Math.max(...afterOneFrame) - 3.0);
checkTrue('one frame of the settle is a small fraction of the journey',
  firstStep < (ceiling - 3.0) * 0.15,
  `first frame ${firstStep.toFixed(3)} of ${(ceiling - 3.0).toFixed(2)}`);
checkTrue('the interpolation reaches the new kernel after the settle window', ceiling > 5,
  ceiling.toFixed(2));

// --- determinism ------------------------------------------------------------
const replay = () => {
  e.rk2_reset();
  pushTone(ON_BIN_HZ, 0.37);
  stepFrames(20);
  return Array.from(barsView());
};
const first = replay();
const second = replay();
checkTrue('identical input reproduces identical bars', first.every((v, i) => v === second[i]));

// --- reset ------------------------------------------------------------------
e.rk2_reset();
check('reset clears the peak frequency', report()[PEAK_HZ], 0);
checkTrue('reset returns the levels to silence', report()[RMS_DB] <= -143 && report()[PEAK_DB] <= -143);
checkTrue('reset clears the bars', Array.from(barsView()).every(v => v === 0));
checkTrue('reset clears the rods', Array.from(heightView()).every(v => v === 0));
checkTrue('reset clears the bands', Array.from(ampView()).every(v => v === 0));

// --- clear_spectrum keeps the rods, and they collapse ------------------------
// Stopping playback clears the spectrum. If that also cleared the heights, all 709
// rods would drop to zero in one frame; the reactor is supposed to ease them down.
// This is the contract the engine's `stop()` depends on.
e.rk2_reset();
pushTone(ON_BIN_HZ, 0.9);
stepFrames(120);
const rodsBeforeClear = Math.max(...heightView());
e.rk2_clear_spectrum();
checkTrue('clear_spectrum keeps the rod state', Math.max(...heightView()) === rodsBeforeClear,
  Math.max(...heightView()).toFixed(4));
checkTrue('clear_spectrum empties the spectrum', totalPower() === 0 && Array.from(barsView()).every(v => v === 0));

e.rk2_step(1 / 60, 1.4, 0.75);
const afterOneFrameOfCollapse = Math.max(...heightView());
checkTrue('the first frame of the collapse is a step down, not a cut',
  afterOneFrameOfCollapse < rodsBeforeClear && afterOneFrameOfCollapse > rodsBeforeClear * 0.7,
  `${rodsBeforeClear.toFixed(2)} -> ${afterOneFrameOfCollapse.toFixed(2)}`);
stepFrames(60);
checkTrue('the rods settle at the baseline after the collapse', Math.max(...heightView()) < 0.2,
  Math.max(...heightView()).toFixed(3));

console.log(failed === 0 ? '\nALL V2 KERNEL CHECKS PASSED' : `\n${failed} V2 KERNEL CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
