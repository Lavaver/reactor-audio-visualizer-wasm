//! v2 audio kernel: an asynchronous, f64 analysis pipeline with no Web Audio
//! semantics anywhere in it.
//!
//! # What "v2" changes
//!
//! v1 (`wasm/src/lib.rs`) reproduces `AnalyserNode`: a Hann window, a 0.75
//! per-frame exponential smoother, a −100…−30 dB window quantised to one byte
//! per bin, band edges rounded to whole bins, and `f32` throughout. That was the
//! point of v1 — it had to look like the thing it replaced.
//!
//! v2 drops the imitation and reports physics instead:
//!
//! * `f64` window, transform, power accumulation **and outputs**. Every array JS
//!   reads as data is `f64`.
//! * A 4-term Blackman-Harris window (sidelobes ≈ −92 dB) normalised by its
//!   coherent gain and its energy, so bin powers are calibrated and Parseval
//!   holds: the bin powers of a block sum to that block's mean square.
//! * Band energy integrated over *fractional* bin edges, so a band never jumps
//!   by a whole bin when it is re-tuned.
//! * Peak frequency refined by parabolic interpolation, giving sub-bin
//!   resolution instead of quantising to `sample_rate / fft_size`.
//! * Units are real: `dBFS`, `Hz`, linear RMS. Nothing is mapped into a byte.
//! * Display smoothing is an attack/release filter driven by the *measured*
//!   frame time, not a fixed per-frame coefficient.
//!
//! # The `f64` memory contract
//!
//! JS numbers *are* IEEE-754 binary64, and wasm has native `f64`, so handing out
//! `*const f64` pointers means a `Float64Array` view over this module's memory is
//! simultaneously zero-copy and bit-exact: no conversion step exists in which a
//! digit could be lost.
//!
//! * Out to JS: [`rk2_bin_power_ptr`] (linear power, the raw measurement),
//!   [`rk2_bin_db_ptr`], [`rk2_bars_ptr`], [`rk2_band_rms_ptr`],
//!   [`rk2_channel_amps_ptr`], [`rk2_report_ptr`], [`rk2_sine_ref`]. All `f64`,
//!   all 8-byte aligned by construction, because Rust aligns a `[f64; N]` field to
//!   8 and every one of these arrays is one.
//! * In from JS: `rk2_configure` and `rk2_step` take `f64`, since JS passes
//!   numbers at that width anyway; narrowing at the door would be the only place
//!   precision could leak out of the analysis.
//!
//! Exactly two `f32` boundaries remain, and both are forced from outside:
//!
//! * [`rk2_wave_ptr`]: `AnalyserNode.getFloatTimeDomainData` can only produce
//!   `f32` samples, so that is the precision of the *source*. Everything applied
//!   to those samples is `f64`.
//! * [`rk2_heights_ptr`] and [`rk2_rod_ptr`]: rod geometry and rod heights are
//!   uploaded to the GPU as `f32`, where the 7th decimal of a height is not
//!   visible. The cast happens on the last line of [`advance_heights`] and
//!   nowhere else.
//!
//! # Where Tokio stops
//!
//! A Tokio reactor cannot exist on `wasm32-unknown-unknown`. This is not a
//! packaging limitation, it is std: the wasm PAL is `sys/time/unsupported.rs`,
//! whose `Instant::now` is `panic!("time not implemented on this platform")`, and
//! there are no threads to park. Tokio's own JSPI parker only applies to
//! `wasm32-unknown-emscripten`, a different toolchain. So:
//!
//! * `tokio::sync::mpsc` is used for the buffer stream — it is runtime-agnostic
//!   and only needs a waker.
//! * The consumer is a real `async fn`: `recv().await` genuinely suspends between
//!   frames and resumes when JS pushes the next block.
//! * The executor is [`pump`]: one `poll` per JS call, with a waker that does
//!   nothing, because the browser's frame loop is already the thing that decides
//!   when to poll again. Nothing ever waits on a clock, so nothing can park.
//!
//! The clock lives in JS (`AudioContext.currentTime`, `performance.now()`), which
//! is both available and more accurate than `Instant` would have been.
//!
//! # Concurrency
//!
//! Single-threaded by construction: one wasm instance, one JS thread, no
//! re-entrant calls, and no allocation on the output path.

use core::cell::UnsafeCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Waker};

use tokio::sync::mpsc;

/// Rods in the hexagonal lattice, matching v1 so a backend switch keeps the
/// exact same geometry (and therefore the same GPU buffers).
pub const ROD_COUNT: usize = 709;
/// Largest transform, and the bound on the staging block.
pub const MAX_FFT: usize = 4096;
/// Bin count for the largest transform (the positive half of the spectrum).
pub const MAX_BINS: usize = MAX_FFT / 2;
pub const MIN_CHANNELS: usize = 4;
pub const MAX_CHANNELS: usize = 64;

/// Band layout: `band(k) = 45 Hz · 320^(k / channels)`, top edge 14.4 kHz. Kept
/// from v1 because it is a display choice, not a Web Audio behaviour.
const BAND_LOW_HZ: f64 = 45.0;
const BAND_SPAN: f64 = 320.0;

// --- display curve ----------------------------------------------------------
//
// Everything below shapes a *measurement* into a *drawing amount*. It runs after
// `band_rms` has been computed and normalised, and never writes back into
// `bin_power`, `bin_db`, `band_rms` or `report` — those stay calibrated and are
// what the readouts and the tests read. So the reactor can be as dramatic as we
// like without the numbers telling a lie.

/// A band this far below full scale reads as zero.
///
/// Deliberately narrow now that the auto-gain owns sensitivity: this constant no
/// longer decides *whether* quiet material is visible (the auto-gain does), it
/// decides how much contrast survives between bands. A wide floor flattens the
/// lattice into a uniform disc; this one keeps the bass tall and the outer rings
/// shorter while still leaving them alive.
const BAND_FLOOR_DB: f64 = -54.0;
/// Gamma applied to the 0..1 band amount. Just under 1: a touch of lift on the
/// middle of the range without squashing the top, which is where the shape lives.
const BAND_COMPAND: f64 = 0.95;
/// Spectral tilt, in dB per octave above [`SLOPE_PIVOT_HZ`].
///
/// Musical content falls off roughly as 1/f, so by 8 kHz a band can be 20 dB below
/// the bass band that carries the same perceived loudness. Without a tilt only the
/// innermost two rings of the core ever move; with it the whole lattice answers.
/// This is the single biggest contributor to the visual difference, and it is
/// unapologetically a drawing curve.
const SLOPE_DB_PER_OCTAVE: f64 = 1.2;
const SLOPE_PIVOT_HZ: f64 = 300.0;
/// The tilt stops here, so the top bands do not simply saturate. Kept deliberately
/// modest: this lifts the outer rings into view, it does not flatten the spectrum
/// into a uniform disc — the bass should still be the tallest thing in the room.
const SLOPE_MAX_DB: f64 = 5.0;

// --- display auto-gain ------------------------------------------------------
//
// A real spectrum analyser is read with a fixed reference; a reactor looked at for
// its drama is not. Without this, a quiet passage leaves the whole core below the
// baseline and only a loud master ever fills it — which is exactly what the honest
// dBFS curve does. So the *display* tracks the loudest band and boosts it to a
// fixed target, the way a visualiser always has.
//
// It only ever boosts, never attenuates, and it acts on the drawing curve alone:
// `band_rms`, `bin_power`, `bin_db` and `report` stay absolute, so every readout
// and every test still sees unmodified dBFS.

/// Where the loudest band is placed by the auto-gain.
///
/// Slightly *above* full scale, so the loudest band is always pegged and the rest
/// of the lattice is measured against that. v1 reaches the same thing by accident:
/// its byte average in a bass band sits at 0.96 because the byte floor is so high.
/// Here it is deliberate, and it is what makes the core read as driven rather than
/// polite.
const AGC_TARGET_DB: f64 = 2.0;
/// Never boost more than this, or a silent room would be rendered as a full core.
const AGC_MAX_BOOST_DB: f64 = 40.0;
/// How fast the tracked peak falls back, in dB per second. Slow enough that a gap
/// between notes does not pump the whole lattice up and down.
const AGC_RELEASE_DB_PER_S: f64 = 6.0;
/// Per-band attack/release, in milliseconds. Fast attack makes transients hit;
/// the slower release is what makes the core keep the hit for a moment instead of
/// flickering at 60 Hz.
const BAND_ATTACK_MS: f64 = 35.0;
const BAND_RELEASE_MS: f64 = 260.0;

/// Display range for one bin, in dB relative to a full-scale sine sitting on that
/// bin. 0 dB is that reference; the floor is 100 dB down, not 60.
///
/// The wide window is not a taste choice. A full-scale sine concentrates its whole
/// energy into one bin, so its peak bin reads 0 dB — but real per-bin content is
/// spread over the spectrum and sits 40–60 dB below that. Measured on white noise:
/// at −30 dBFS RMS the loudest bin reads about −45 dB, and at −40 dBFS about −55 dB.
/// With the floor at −60 dB those blocks lit 0 and 25 of 1024 bins, i.e. the 48-bar
/// spectrum collapsed to a flat stub. A 100 dB window keeps the display honest —
/// nothing is lifted, the mapping is still `(dB - min) / (0 - min)` — while leaving
/// the spectrum readable at normal listening levels.
const DISPLAY_MIN_DB: f64 = -100.0;
const DISPLAY_MAX_DB: f64 = 0.0;
/// Attack/release time constants for the bar display, in milliseconds. Driven by
/// the measured frame time, so a 120 Hz screen and a 60 Hz screen ease alike.
const ATTACK_MS: f64 = 12.0;
const RELEASE_MS: f64 = 140.0;
/// Reported instead of `-inf` once a block is silent all the way down.
const SILENCE_DB: f64 = -144.0;
/// Buffers that may wait in the channel; a deeper queue only adds latency.
const PIPELINE_DEPTH: usize = 4;
/// Keeps `log10` finite on an exactly-zero bin.
const EPS: f64 = 1e-20;

/// Words in the report array behind [`rk2_report_ptr`], in this order.
pub const REPORT_PEAK_HZ: usize = 0;
pub const REPORT_CENTROID_HZ: usize = 1;
pub const REPORT_RMS_DB: usize = 2;
pub const REPORT_PEAK_DB: usize = 3;
pub const REPORT_WORDS: usize = 4;

/// 4-term Blackman-Harris constants: −92 dB sidelobes, at the cost of a wide
/// main lobe. Leakage, not resolution, was the error that dominated v1.
const BH_A0: f64 = 0.35875;
const BH_A1: f64 = 0.48829;
const BH_A2: f64 = 0.14128;
const BH_A3: f64 = 0.01168;

#[repr(C)]
#[derive(Clone, Copy)]
struct Rod {
    x: f32,
    z: f32,
    radius: f32,
    seed: f32,
    channel: f32,
}

const ROD_ZERO: Rod = Rod {
    x: 0.0,
    z: 0.0,
    radius: 0.0,
    seed: 0.0,
    channel: 0.0,
};

/// One block of time-domain samples travelling through the channel.
///
/// Owning the samples is the point: a `push` must be able to hand a block over
/// and forget it, and the consumer must be able to `await` one without aliasing
/// the caller's staging buffer.
struct Block {
    samples: Vec<f32>,
}

struct State {
    rods: [Rod; ROD_COUNT],
    heights: [f32; ROD_COUNT],
    channels: usize,
    /// Transform size and the rate it is measured at.
    fft_size: usize,
    sample_rate: f64,
    bin_count: usize,
    /// Window, and the two normalisations derived from it: coherent gain and
    /// mean square. Rebuilt only when the transform size changes.
    window: [f64; MAX_FFT],
    window_cg: f64,
    window_s2: f64,
    /// Peak bin value of a full-scale sine parked on its own bin. Bin powers are
    /// reported relative to this, so 0 dB is "full scale at this bin".
    sine_ref: f64,
    bitrev: [u16; MAX_FFT],
    tables_for: usize,
    /// Staging block JS writes before [`rk2_push`]. The one `f32` on the way in,
    /// because `AnalyserNode` has no other precision to offer.
    wave: [f32; MAX_FFT],
    /// f64 transform scratch space.
    re: [f64; MAX_FFT],
    im: [f64; MAX_FFT],
    /// Instantaneous bin powers, one-sided (the mirrored half is already folded
    /// in), so summing them yields the block's mean square. This is the raw
    /// physical quantity JS reads: no log, no narrowing.
    bin_power: [f64; MAX_BINS],
    /// The same, in dB relative to [`State::sine_ref`]. Derived for display.
    bin_db: [f64; MAX_BINS],
    /// Smoothed 0..1 display values the bars read.
    bars: [f64; MAX_BINS],
    /// Linear RMS per band, `f64`, straight out of the fractional integration.
    /// This is the measurement: calibrated, and never shaped for the screen.
    band_rms: [f64; MAX_CHANNELS],
    /// Band RMS mapped onto 0..1 through the display curve (floor, slope, gamma).
    /// A drawing amount, not a measurement.
    channel_amps: [f64; MAX_CHANNELS],
    /// [`State::channel_amps`] after attack/release smoothing; what the rods follow.
    band_drive: [f64; MAX_CHANNELS],
    /// Peak-hold of the loudest band, in dBFS, that the display auto-gain tracks.
    band_peak_db: f64,
    /// `[peak_hz, centroid_hz, rms_db, peak_db]`, all `f64`.
    report: [f64; REPORT_WORDS],
    /// Sender end of the buffer channel; `None` until configured.
    tx: Option<mpsc::Sender<Block>>,
    processed: u32,
    dropped: u32,
    ready: bool,
}

impl State {
    /// `const` so the whole kernel can live in a `static`; every field that
    /// needs an allocator starts empty and is filled by [`rk2_configure`].
    /// The non-zero defaults, so `new()` can stay all-zero and keep the state out of the
    /// wasm file's data segment.
    fn defaults(&mut self) {
        self.channels = 32;
        self.fft_size = 2048;
        self.sample_rate = 48_000.0;
        self.bin_count = 1024;
        self.band_peak_db = f64::NEG_INFINITY;
        self.report = [0.0, 0.0, SILENCE_DB, SILENCE_DB];
    }

    const fn new() -> Self {
        Self {
            rods: [ROD_ZERO; ROD_COUNT],
            heights: [0.0; ROD_COUNT],
            channel_amps: [0.0; MAX_CHANNELS],
            // All zero on purpose: one non-zero initialiser makes the whole `State` an
            // initialized data image, which copies the transform scratch and the bin arrays
            // into the .wasm file instead of leaving them zero-filled. Non-zero defaults are
            // in `defaults()`, called from `rk2_init`.
            channels: 0,
            fft_size: 0,
            sample_rate: 0.0,
            bin_count: 0,
            window: [0.0; MAX_FFT],
            window_cg: 0.0,
            window_s2: 0.0,
            sine_ref: 0.0,
            bitrev: [0; MAX_FFT],
            tables_for: 0,
            wave: [0.0; MAX_FFT],
            re: [0.0; MAX_FFT],
            im: [0.0; MAX_FFT],
            bin_power: [0.0; MAX_BINS],
            bin_db: [0.0; MAX_BINS],
            bars: [0.0; MAX_BINS],
            band_rms: [0.0; MAX_CHANNELS],
            band_drive: [0.0; MAX_CHANNELS],
            band_peak_db: 0.0,
            report: [0.0; REPORT_WORDS],
            tx: None,
            processed: 0,
            dropped: 0,
            ready: false,
        }
    }
}

struct Kernel(UnsafeCell<State>);

// SAFETY: one wasm instance, one JS thread. Every export below is called from
// that thread, and none of them re-enters the kernel while holding a borrow: the
// pipeline stage runs inside [`pump`], which holds no `State` borrow.
unsafe impl Sync for Kernel {}

static KERNEL: Kernel = Kernel(UnsafeCell::new(State::new()));

/// The one mutable handle the exports and the pipeline go through.
fn kernel() -> &'static mut State {
    // SAFETY: see the `unsafe impl Sync` note above.
    unsafe { &mut *KERNEL.0.get() }
}

// --- executor ---------------------------------------------------------------

/// The pipeline task lives outside `State` on purpose.
///
/// Polling it must not hold a `&mut State`, because the task calls [`kernel`]
/// internally; keeping the two in separate cells is what makes the aliasing
/// question go away entirely instead of being argued about.
struct Executor(UnsafeCell<Option<Pin<Box<dyn Future<Output = ()>>>>>);

// SAFETY: same argument as `Kernel` — polling happens on the single JS thread.
unsafe impl Sync for Executor {}

static EXECUTOR: Executor = Executor(UnsafeCell::new(None));

/// Give the pipeline one scheduling turn: run it until it blocks on `recv`.
///
/// Returns after a single `poll`: either the stage drained every queued block and
/// parked on an empty channel, or it finished. It can never spin, because the
/// only thing it waits for is a message that this same JS call already sent.
///
/// The waker is [`Waker::noop`] on purpose. Every wake in this design means "data
/// is ready", and data readiness is already known at the next push, which sends
/// and then polls in the same call — storing a callback would never be read.
fn pump() {
    // SAFETY: single JS thread, and no `State` borrow is held across this call.
    let slot = unsafe { &mut *EXECUTOR.0.get() };
    if let Some(task) = slot.as_mut() {
        let mut cx = Context::from_waker(Waker::noop());
        let _ = task.as_mut().poll(&mut cx);
    }
}

/// Rebuild the channel and the consumer task, dropping any queued blocks.
fn restart_pipeline(state: &mut State) {
    let (tx, rx) = mpsc::channel::<Block>(PIPELINE_DEPTH);
    // Safe as above; `EXECUTOR` and `KERNEL` are distinct allocations, so holding
    // `state` while replacing the task does not alias.
    let slot = unsafe { &mut *EXECUTOR.0.get() };
    *slot = Some(Box::pin(stage(rx)));
    state.tx = Some(tx);
}

/// The asynchronous stage: consume blocks for as long as the channel lives.
///
/// `recv().await` is where the "fully asynchronous" part actually lives: with no
/// block queued the future returns `Pending`, control goes back to JS, and the
/// next [`rk2_push`] resumes it exactly where it stopped.
async fn stage(mut rx: mpsc::Receiver<Block>) {
    while let Some(block) = rx.recv().await {
        analyse_block(kernel(), &block.samples);
    }
}

// --- tables and transform ---------------------------------------------------

fn build_tables(state: &mut State) {
    let size = state.fft_size;
    let denominator = (size - 1).max(1) as f64;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for index in 0..size {
        let phase = core::f64::consts::TAU * index as f64 / denominator;
        let value =
            BH_A0 - BH_A1 * phase.cos() + BH_A2 * (2.0 * phase).cos() - BH_A3 * (3.0 * phase).cos();
        state.window[index] = value;
        sum += value;
        sum_sq += value * value;
    }
    let n = size as f64;
    state.window_cg = sum / n;
    state.window_s2 = sum_sq / n;
    // One-sided peak bin of a full-scale on-bin sine: |X| = A·N·CG/2, doubled
    // because the negative-frequency half is folded in.
    state.sine_ref = if state.window_s2 > 0.0 {
        state.window_cg * state.window_cg / (2.0 * state.window_s2)
    } else {
        1.0
    };

    let shift = 32 - size.trailing_zeros();
    for (index, slot) in state.bitrev[..size].iter_mut().enumerate() {
        *slot = ((index as u32).reverse_bits() >> shift) as u16;
    }
    state.tables_for = size;
}

/// In-place iterative radix-2 Cooley-Tukey FFT in `f64`.
fn fft(re: &mut [f64], im: &mut [f64], bitrev: &[u16]) {
    let size = re.len();
    for (index, reversed) in bitrev.iter().enumerate().take(size) {
        let reversed = *reversed as usize;
        if reversed > index {
            re.swap(index, reversed);
            im.swap(index, reversed);
        }
    }
    let mut span = 2;
    while span <= size {
        let angle = -core::f64::consts::TAU / span as f64;
        let (step_re, step_im) = (angle.cos(), angle.sin());
        let half = span / 2;
        let mut start = 0;
        while start < size {
            let mut twiddle_re = 1.0f64;
            let mut twiddle_im = 0.0f64;
            for offset in 0..half {
                let low = start + offset;
                let high = low + half;
                let rotated_re = re[high] * twiddle_re - im[high] * twiddle_im;
                let rotated_im = re[high] * twiddle_im + im[high] * twiddle_re;
                re[high] = re[low] - rotated_re;
                im[high] = im[low] - rotated_im;
                re[low] += rotated_re;
                im[low] += rotated_im;
                let next_re = twiddle_re * step_re - twiddle_im * step_im;
                twiddle_im = twiddle_re * step_im + twiddle_im * step_re;
                twiddle_re = next_re;
            }
            start += span;
        }
        span <<= 1;
    }
}

/// Window, transform and band integration for one block.
fn analyse_block(state: &mut State, samples: &[f32]) {
    let size = state.fft_size;
    if samples.len() < size {
        return;
    }
    if state.tables_for != size {
        build_tables(state);
    }
    let n = size as f64;
    state.processed = state.processed.wrapping_add(1);

    // DC removal, then the meter. A constant offset is not audible content; left
    // in, it would dominate the RMS figure and tilt the whole low end.
    let mut mean = 0.0f64;
    for sample in &samples[..size] {
        mean += *sample as f64;
    }
    mean /= n;

    let mut sum_sq = 0.0f64;
    let mut peak = 0.0f64;
    for (index, sample) in samples[..size].iter().enumerate() {
        let centred = *sample as f64 - mean;
        sum_sq += centred * centred;
        let magnitude = centred.abs();
        if magnitude > peak {
            peak = magnitude;
        }
        state.re[index] = centred * state.window[index];
        state.im[index] = 0.0;
    }

    {
        let State { re, im, bitrev, .. } = &mut *state;
        fft(&mut re[..size], &mut im[..size], &bitrev[..size]);
    }

    let norm = 1.0 / (n * n * state.window_s2);
    let bins = state.bin_count.min(size / 2);
    let mut total = 0.0f64;
    let mut weighted = 0.0f64;
    for bin in 0..bins {
        let re = state.re[bin];
        let im = state.im[bin];
        // Fold the negative-frequency mirror in, except at DC and Nyquist where
        // the two halves are the same bin.
        let one_sided = if bin == 0 || bin == size / 2 {
            1.0
        } else {
            2.0
        };
        let power = (re * re + im * im) * norm * one_sided;
        state.bin_power[bin] = power;
        state.bin_db[bin] = 10.0 * (power / state.sine_ref).max(EPS).log10();
        total += power;
        weighted += power * (bin as f64 * state.sample_rate / n);
    }

    state.band_rms = [0.0; MAX_CHANNELS];
    integrate_bands(state);
    // Band RMS -> the 0..1 amount the rods react to. Done here rather than in
    // `step` because it is a measurement of this block, not a frame animation:
    // the value must be readable as soon as the block has been analysed.
    map_bands(state);

    // Computed before the report is borrowed mutably, because the peak search
    // only needs to read the bins.
    let peak_hz = interpolated_peak(state);
    let rms_db = if sum_sq > 0.0 {
        10.0 * (sum_sq / n).log10()
    } else {
        SILENCE_DB
    };
    let peak_db = if peak > 0.0 {
        20.0 * peak.log10()
    } else {
        SILENCE_DB
    };
    let centroid_hz = if total > 0.0 { weighted / total } else { 0.0 };

    let report = &mut state.report;
    report[REPORT_PEAK_HZ] = peak_hz;
    report[REPORT_CENTROID_HZ] = centroid_hz;
    report[REPORT_RMS_DB] = rms_db;
    report[REPORT_PEAK_DB] = peak_db;
}

/// Refine the strongest bin with a parabola fitted through its neighbours in
/// dB, which is what turns a 23 Hz bin grid into a frequency estimate that is
/// good to a fraction of a bin.
fn interpolated_peak(state: &State) -> f64 {
    let bins = state.bin_count;
    if bins < 4 {
        return 0.0;
    }
    let mut best = 1usize;
    let mut best_power = state.bin_power[1];
    for bin in 2..bins - 1 {
        if state.bin_power[bin] > best_power {
            best_power = state.bin_power[bin];
            best = bin;
        }
    }
    if best_power <= EPS {
        return 0.0;
    }
    let left = state.bin_db[best - 1];
    let centre = state.bin_db[best];
    let right = state.bin_db[best + 1];
    let denominator = left - 2.0 * centre + right;
    let offset = if denominator.abs() > 1e-12 {
        (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let bin_hz = state.sample_rate / state.fft_size as f64;
    (best as f64 + offset) * bin_hz
}

/// Integrate bin power over each band, weighting partial bins by how much of
/// them the band covers.
///
/// Integrating instead of rounding is what keeps `band_rms` continuous: a band
/// whose edge sits at bin 37.4 takes 40% of bin 37 and 60% of bin 38, so a
/// sample-rate change moves the reading smoothly instead of stepping it.
fn integrate_bands(state: &mut State) {
    let count = state.channels;
    let n = state.fft_size as f64;
    let fs = state.sample_rate;
    let top = (state.bin_count as f64 - 1.0).max(1.0);

    for channel in 0..count {
        let low_hz = BAND_LOW_HZ * BAND_SPAN.powf(channel as f64 / count as f64);
        let high_hz = BAND_LOW_HZ * BAND_SPAN.powf((channel + 1) as f64 / count as f64);
        let mut low = (low_hz * n / fs).clamp(1.0, top);
        let mut high = (high_hz * n / fs).clamp(1.0, top);
        if high < low {
            core::mem::swap(&mut low, &mut high);
        }

        let mut power = 0.0f64;
        if high - low < 1.0 {
            // Narrower than a bin: read the bin the band centre falls in.
            let bin = (0.5 * (low + high)).round().clamp(1.0, top) as usize;
            power = state.bin_power[bin];
        } else {
            let mut bin = low.floor() as usize;
            let end = high.ceil() as usize;
            while bin <= end && bin < state.bin_count {
                let overlap = (high.min(bin as f64 + 1.0) - low.max(bin as f64)).max(0.0);
                if overlap > 0.0 {
                    power += overlap * state.bin_power[bin];
                }
                bin += 1;
            }
        }
        state.band_rms[channel] = power.sqrt();
    }
}

// --- exports ----------------------------------------------------------------

/// Fill the rod lattice. Identical arithmetic to v1, including the seed hash, so
/// both backends hand the renderers the same geometry and a switch costs nothing.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_init() -> u32 {
    let state = kernel();
    state.defaults();
    let mut index = 0usize;
    let mut x = -15i32;
    while x <= 15 {
        let mut z = -15i32;
        while z <= 15 {
            let radius = ((x * x + z * z) as f32).sqrt();
            if radius <= 15.0 {
                if index < ROD_COUNT {
                    state.rods[index] = Rod {
                        x: x as f32,
                        z: z as f32,
                        radius,
                        seed: ((x as f32) * 127.1f32 + (z as f32) * 311.7f32).sin().abs(),
                        channel: 0.0,
                    };
                }
                index += 1;
            }
            z += 1;
        }
        x += 1;
    }
    state.ready = index == ROD_COUNT;
    state.channels = 32;
    state.heights = [0.0; ROD_COUNT];
    apply_channels(state);
    reset_analysis(state);
    index as u32
}

/// Version marker, so the JS side (and its tests) can tell the two modules apart.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_version() -> u32 {
    2
}

/// Choose the transform size (rounded down to a power of two), the sample rate
/// the bands are laid out against, and rebuild the pipeline.
///
/// `sample_rate` is `f64`: it divides every band edge, and a rate that arrives
/// already rounded would put a bias in every frequency the kernel reports.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_configure(fft_size: u32, sample_rate: f64) {
    let state = kernel();
    let requested = (fft_size as usize).clamp(64, MAX_FFT);
    // `requested` is a u32; using `usize::BITS` here shifted 32 bits too far for any
    // fft_size below 2048.
    let size = 1usize << (u32::BITS - 1 - requested.leading_zeros()) as usize;
    let rate = if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    };

    let retuned = size != state.fft_size || (rate - state.sample_rate).abs() > 0.5;
    state.fft_size = size;
    state.sample_rate = rate;
    state.bin_count = size / 2;
    if state.tables_for != size {
        build_tables(state);
    }
    // Idempotent: a second call with the same numbers must not throw away a
    // pipeline that is already running, or every frame would lose its queue.
    if retuned || state.tx.is_none() {
        reset_analysis(state);
        restart_pipeline(state);
    }
}

/// Peak bin power of a full-scale sine, i.e. the reference the bin dB values are
/// expressed against. Exposed so the JS side can rebuild absolute power from the
/// dB array when it needs to (the test harness does).
#[unsafe(no_mangle)]
pub extern "C" fn rk2_sine_ref() -> f64 {
    kernel().sine_ref
}

/// Drop every history buffer and start a new session, with an empty channel.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_reset() {
    let state = kernel();
    reset_analysis(state);
    restart_pipeline(state);
}

fn reset_analysis(state: &mut State) {
    state.bin_power = [0.0; MAX_BINS];
    state.bin_db = [0.0; MAX_BINS];
    state.bars = [0.0; MAX_BINS];
    state.band_rms = [0.0; MAX_CHANNELS];
    state.channel_amps = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
    state.band_peak_db = f64::NEG_INFINITY;
    state.report = [0.0, 0.0, SILENCE_DB, SILENCE_DB];
    state.heights = [0.0; ROD_COUNT];
}

/// Drop the spectrum but keep the rod state.
///
/// This is what starting (or seeking) playback needs: the previous session's
/// spectrum must not flash for a frame, but the rods should keep easing from
/// where they are instead of snapping to the floor.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_clear_spectrum() {
    let state = kernel();
    state.bin_power = [0.0; MAX_BINS];
    state.bin_db = [0.0; MAX_BINS];
    state.bars = [0.0; MAX_BINS];
    state.band_rms = [0.0; MAX_CHANNELS];
    state.channel_amps = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
    state.band_peak_db = f64::NEG_INFINITY;
    state.report = [0.0, 0.0, SILENCE_DB, SILENCE_DB];
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_fft_size() -> u32 {
    kernel().fft_size as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_bin_count() -> u32 {
    kernel().bin_count as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_rod_count() -> u32 {
    ROD_COUNT as u32
}

/// Stride between rods in `f32` words: `[x, z, radius, seed, channel]`.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_rod_stride() -> u32 {
    (core::mem::size_of::<Rod>() / core::mem::size_of::<f32>()) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_rod_ptr() -> *const f32 {
    kernel().rods.as_ptr().cast::<f32>()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_heights_ptr() -> *mut f32 {
    kernel().heights.as_mut_ptr()
}

/// Per-band 0..1 amounts after the display curve, `f64`, `MAX_CHANNELS` long.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_channel_amps_ptr() -> *const f64 {
    kernel().channel_amps.as_ptr()
}

/// Per-band amounts after attack/release smoothing, `f64` — what the rods follow.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_band_drive_ptr() -> *const f64 {
    kernel().band_drive.as_ptr()
}

/// Linear RMS per band, `f64`, `MAX_CHANNELS` long: the raw band energy before it
/// is mapped onto a display range.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_band_rms_ptr() -> *const f64 {
    kernel().band_rms.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_set_channels(channels: u32) {
    let state = kernel();
    state.channels = (channels as usize).clamp(MIN_CHANNELS, MAX_CHANNELS);
    apply_channels(state);
    state.band_rms = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_channels() -> u32 {
    kernel().channels as u32
}

fn apply_channels(state: &mut State) {
    let count = state.channels;
    for rod in state.rods.iter_mut() {
        let band = (rod.radius / 15.0 * count as f32).floor() as usize;
        rod.channel = band.min(count - 1) as f32;
    }
}

/// Staging buffer for one block of time-domain samples, `f32`, `fft_size` long.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_wave_ptr() -> *mut f32 {
    kernel().wave.as_mut_ptr()
}

/// Hand the staged block to the asynchronous stage and give it one turn.
///
/// Returns the number of blocks the stage consumed during this call, which is
/// normally 1: the consumer is polled immediately after the send, so the queue
/// only grows if a caller pushes without ever letting the stage run.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_push() -> u32 {
    let accepted = {
        // Borrow `state` only for the send, and drop the borrow before polling:
        // the stage calls `kernel()` itself, so holding it across `pump` would
        // create two `&mut State` for the same object.
        let state = kernel();
        if !state.ready {
            return 0;
        }
        let size = state.fft_size;
        let block = Block {
            samples: state.wave[..size].to_vec(),
        };
        let before = state.processed;
        match state.tx.as_ref() {
            Some(tx) => match tx.try_send(block) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Backpressure: the stage has not run for a while. Dropping
                    // the newest block is the right loss for a visualiser — an
                    // old frame displayed late is worse than a skipped one.
                    state.dropped = state.dropped.wrapping_add(1);
                    return 0;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return 0;
                }
            },
            None => return 0,
        }
        before
    };
    pump();
    kernel().processed.wrapping_sub(accepted)
}

/// Instantaneous bin power in dB relative to a full-scale on-bin sine.
/// Linear power per bin, `f64`, `bin_count` long.
///
/// This is the raw measurement: one-sided power, already normalised so that
/// summing the array yields `Σ w²y² / Σ w²` for the block. JS reads it in place
/// with a `Float64Array` — no conversion, no copy, no narrowing.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_bin_power_ptr() -> *const f64 {
    kernel().bin_power.as_ptr()
}

/// The analysis window in use, `f64`, `fft_size` long.
///
/// Exposed so the Parseval identity can be checked against the exact coefficients
/// the kernel used, instead of against a re-derived approximation of them.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_window_ptr() -> *const f64 {
    kernel().window.as_ptr()
}

/// Bin power in dB relative to a full-scale on-bin sine, `f64`, for display.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_bin_db_ptr() -> *const f64 {
    kernel().bin_db.as_ptr()
}

/// Smoothed 0..1 values for the bar display, one per bin, `f64`.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_bars_ptr() -> *const f64 {
    kernel().bars.as_ptr()
}

/// `[peak_hz, centroid_hz, rms_db, peak_db]`, all `f64`.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_report_ptr() -> *const f64 {
    kernel().report.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_processed() -> u32 {
    kernel().processed
}

#[unsafe(no_mangle)]
pub extern "C" fn rk2_dropped() -> u32 {
    kernel().dropped
}

/// One frame of audio reaction.
///
/// - `dt_seconds`: measured frame time. Display smoothing and rod easing are both
///   exponentials of it, so the motion is frame-rate independent.
/// - `gain`: visual sensitivity multiplier.
/// - `smoothing`: 0..0.95; higher means the rods travel more slowly.
///
/// All three are `f64`: the browser hands JS `f64` numbers, and truncating them
/// here would undo the precision the analysis just kept.
#[unsafe(no_mangle)]
pub extern "C" fn rk2_step(dt_seconds: f64, gain: f64, smoothing: f64) {
    let state = kernel();
    if !state.ready {
        return;
    }
    let dt = if dt_seconds.is_finite() {
        dt_seconds.clamp(0.001, 0.25)
    } else {
        1.0 / 60.0
    };
    let gain = if gain.is_finite() {
        gain.clamp(0.0, 8.0)
    } else {
        1.0
    };
    let smoothing = if smoothing.is_finite() {
        smoothing.clamp(0.0, 0.95)
    } else {
        0.75
    };

    smooth_bars(state, dt);
    smooth_bands(state, dt);
    advance_heights(state, dt * 60.0, gain, smoothing);
}

/// Attack/release smoothing of the bar display in the dB domain.
fn smooth_bars(state: &mut State, dt: f64) {
    let bins = state.bin_count;
    let range = DISPLAY_MAX_DB - DISPLAY_MIN_DB;
    let dt_ms = dt * 1000.0;
    for bin in 0..bins {
        let decibels = state.bin_db[bin];
        let target = ((decibels - DISPLAY_MIN_DB) / range).clamp(0.0, 1.0);
        let current = state.bars[bin];
        let tau = if target > current {
            ATTACK_MS
        } else {
            RELEASE_MS
        };
        let coefficient = 1.0 - (-dt_ms / tau).exp();
        state.bars[bin] = current + (target - current) * coefficient;
    }
}

/// Band RMS -> the 0..1 amount the reactor reacts to, with no auto-gain applied.
///
/// Three shaping steps, each one a display decision:
///
/// 1. **tilt** by [`SLOPE_DB_PER_OCTAVE`] above [`SLOPE_PIVOT_HZ`], compensating the
///    1/f slope of music so the outer rings move too;
/// 2. **floor** at [`BAND_FLOOR_DB`], so quiet bands still register;
/// 3. **compand** by [`BAND_COMPAND`], lifting the middle of the range.
///
/// The calibrated value in [`State::band_rms`] is left alone; anything that needs
/// the real number (readouts, tests) reads that instead.
fn map_bands(state: &mut State) {
    let count = state.channels;
    for channel in 0..count {
        state.channel_amps[channel] = band_amount(&state.band_rms, channel, count, 0.0);
    }
}

/// The display curve for one band, given a boost from the auto-gain.
///
/// Silence is special-cased: a band with no energy returns 0 *before* the boost is
/// applied. Both the tilt and the auto-gain are lifts, and a lift applied to a
/// floor that is only just below the display window would otherwise make an empty
/// room glow — the exact failure the `DC offset` check exists to catch.
fn band_amount(rms: &[f64; MAX_CHANNELS], channel: usize, count: usize, boost_db: f64) -> f64 {
    let value = rms[channel];
    if value <= 0.0 {
        return 0.0;
    }
    let decibels = 20.0 * value.log10() + boost_db + tilt_db(channel, count);
    let base = ((decibels - BAND_FLOOR_DB) / -BAND_FLOOR_DB).clamp(0.0, 1.0);
    base.powf(BAND_COMPAND)
}

/// Tilt for one band, in dB: zero at and below the pivot, then
/// [`SLOPE_DB_PER_OCTAVE`] per octave, capped at [`SLOPE_MAX_DB`].
///
/// Added in the dB domain rather than applied as a gain, so that it shifts the
/// spectrum up without rescaling the top of the range: a band already near full
/// scale has nothing to gain from more, while a quiet outer band does.
fn tilt_db(channel: usize, count: usize) -> f64 {
    let octaves = (band_centre_hz(channel, count) / SLOPE_PIVOT_HZ)
        .max(1.0)
        .log2();
    (SLOPE_DB_PER_OCTAVE * octaves).min(SLOPE_MAX_DB)
}

/// Geometric centre of a band, which is where its tilt is measured.
fn band_centre_hz(channel: usize, count: usize) -> f64 {
    let low = BAND_LOW_HZ * BAND_SPAN.powf(channel as f64 / count as f64);
    let high = BAND_LOW_HZ * BAND_SPAN.powf((channel + 1) as f64 / count as f64);
    (low * high).sqrt()
}

/// Auto-gain + attack/release: the step that turns the display curve into motion.
///
/// Fast attack makes transients hit; the slower release is what makes the core keep
/// the hit for a moment instead of flickering at 60 Hz.
fn smooth_bands(state: &mut State, dt: f64) {
    let count = state.channels;

    // Track the loudest band in this block, then let the hold decay slowly.
    let mut block_max = f64::NEG_INFINITY;
    for channel in 0..count {
        let value = state.band_rms[channel];
        if value > 0.0 {
            let decibels = 20.0 * value.log10();
            if decibels > block_max {
                block_max = decibels;
            }
        }
    }
    let fallen = state.band_peak_db - AGC_RELEASE_DB_PER_S * dt;
    state.band_peak_db = if block_max.is_finite() {
        fallen.max(block_max)
    } else {
        fallen
    };
    let boost = (AGC_TARGET_DB - state.band_peak_db).clamp(0.0, AGC_MAX_BOOST_DB);

    let dt_ms = dt * 1000.0;
    for channel in 0..count {
        let target = band_amount(&state.band_rms, channel, count, boost);
        let current = state.band_drive[channel];
        let tau = if target > current {
            BAND_ATTACK_MS
        } else {
            BAND_RELEASE_MS
        };
        let coefficient = 1.0 - (-dt_ms / tau).exp();
        state.band_drive[channel] = current + (target - current) * coefficient;
    }
}

/// Same easing curve as v1, kept so switching backends does not change the
/// reactor's personality — only the numbers feeding it.
fn advance_heights(state: &mut State, dt_frames: f64, gain: f64, smoothing: f64) {
    let damping = 1.0 - (0.45 + smoothing * 0.53).powf(dt_frames);
    for index in 0..ROD_COUNT {
        let channel = state.rods[index].channel as usize;
        let amp = state.band_drive[channel].powf(1.7) * gain;
        let target = 0.07 + amp * 4.8;
        let height = &mut state.heights[index];
        // The one narrowing cast left on the way out: the renderers upload rod
        // heights to the GPU as `f32`, and a 7th decimal of a rod height is not
        // visible. Every measurement above this line stays `f64`.
        *height = (*height as f64 + (target - *height as f64) * damping) as f32;
    }
}
