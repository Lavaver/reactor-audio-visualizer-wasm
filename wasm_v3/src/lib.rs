//! v3 audio kernel: one transform per frequency region, FFT above the crossover and
//! a constant-Q bank below it, in the same frame.
//!
//! # Why two paths, and where the crossover is forced to be
//!
//! Below some frequency the FFT stops being able to place a band: its bin width is
//! `fs / N` regardless of frequency, and its main lobe is ~4 bins either side. At 208 Hz,
//! with 2048 samples at 48 kHz, that lobe is 94 Hz wide while the band itself is 45 Hz,
//! so the FFT's band integral there necessarily contains part of the neighbouring band.
//! A constant-Q filter's bandwidth is a fixed fraction of its centre, so it does not have
//! that failure. Above the crossover the FFT is cheaper and its window is shorter.
//!
//! Two conditions set the numbers:
//!
//! * **isolation.** A CQT bin's neighbour sits `Q (rho - 1) / rho` window bins above it
//!   and `Q (rho - 1)` below. That invariant holds for any Q; what Q changes is where
//!   those offsets land in the window's response. The 4-term BH main lobe ends at 4
//!   window bins, so `Q = 4 / (rho - 1)` puts the *lower* neighbour on that null.
//!   **Q = 20.25**; measured: lower neighbour (-121.71 dB), upper neighbour at 3.34
//!   window bins and still on the main-lobe skirt (-47.64 dB). Both sides clear -40 dB,
//!   so the `4 rho / (rho - 1) = 24.25` that would null the upper side as well buys
//!   nothing and costs 20% more window. Energy share in the correct band: **100.00%**.
//! * **delay equality.** `N_k = N_fft` at `f = Q fs / N_fft` = 474.6 Hz. Bands above it
//!   are left to the FFT, so no CQT window in use is shorter than the FFT's.
//!
//! Band layout is unchanged from v1/v2: `f(k) = 45 * 320^(k/N)`, N = channels, one CQT
//! bin per band centred inside it. N = 32, fs = 48 kHz, N_fft = 2048,
//! `rho = 320^(1/32) = 1.197529`, `Q = 20.25`:
//!
//! | band | centre Hz | CQT window | window ms | group delay ms |
//! |------|-----------|------------|-----------|----------------|
//! | 0    | 49.2      | 19748      | 411.4     | 205.7          |
//! | 2    | 70.6      | 13772      | 286.9     | 143.5          |
//! | 4    | 101.2     | 9602       | 200.0     | 100.0          |
//! | 6    | 145.2     | 6695       | 139.5     | 69.7           |
//! | 8    | 208.2     | 4668       | 97.3      | 48.6           |
//! | 10   | 298.7     | 3255       | 67.8      | 33.9           |
//! | 12   | 428.3     | 2269       | 47.3      | 23.6           |
//!
//! Band 12 is the last CQT band (428.3 Hz < 474.6 Hz); its window is within 11% of the
//! FFT's, so the crossfade over bands 11..12 costs 2.3 ms of delay mismatch. Measured cost
//! of both paths together: **0.356 ms per frame, 2.1% of a 60 Hz budget**
//! (`scripts/kernel-check-v3.mjs` times it rather than assuming).
//!
//! # Mixing: frequency split with a crossfade, and what it costs
//!
//! Of the three ways to reconcile the paths, this file takes (b):
//!
//! * (a) delay the fast path to the slow path: option 1 below makes it concrete — the
//!   lowest CQT band's response peaks 205.7 ms after a transient enters it, so uniform
//!   alignment puts every band, and therefore the whole core, ~164 ms behind the kick.
//! * (c) no mixing at all: the FFT alone drives the rods and the CQT drives a separate
//!   indicator. That removes the seam but also removes any way to check that the two
//!   paths agree, so the cross-path acceptance test would test nothing.
//! * (b) frequency split with a crossfade, chosen here. Both paths publish their own band
//!   RMS, so the seam is measured rather than asserted: 0.23 dB for a tone on its own
//!   band centre, 2.04 dB on a 1/f block at -20 dBFS.
//!
//! ## The lag gradient is real, and is not compensated
//!
//! A windowed DFT responds to a transient at `tau + N/2` — when the transient reaches the
//! window's centre. Ending the CQT window earlier cannot fix that: the centre sits at
//! `now - offset - N/2`, so pulling it toward `now` needs a *negative* offset, i.e.
//! samples that have not arrived. Measured both ways (impulse energy centroid):
//!
//! | configuration | band 12 (2269 samples) | band 0 (19748 samples) |
//! |---------------|------------------------|------------------------|
//! | end-aligned (this build) | 0.0 ms | 164.3 ms |
//! | window ended `(N - N_fft)/2` early | 0.0 ms | ~349 ms |
//!
//! So the inner rings light up to 164 ms after the rim for the same transient. The
//! alternative — a uniform 164 ms delay on every band — is worse for a visualiser, which
//! is why the gradient is kept. It is a panel-label obligation: the UI has to state that
//! bands 0..12 lag by up to 164 ms instead of letting the picture imply otherwise.
//!
//! # Which numbers the panel reads
//!
//! [`rk3_report_ptr`] is the **FFT path only** — same estimators as v2 (parabolic peak on
//! the bin dB values, power-weighted centroid on the bin powers, time-domain RMS and
//! sample peak), so a v2 and a v3 readout are the same measurement of the same block. The
//! CQT publishes its own pitch as [`rk3_cqt_report_ptr`]. Bands 0..10 are drawn from the
//! CQT while the readouts stay FFT-derived, so the panel labels the two sources
//! separately; neither is presented as the other.
//!
//! # Two definitions this file relies on
//!
//! * **Noise bandwidth is the full width, `2f/Q`**, not the half width `f/Q`. BH4's
//!   equivalent noise bandwidth is 2 window bins and a window bin is `f/Q`. At Q = 20.25
//!   the filter's noise bandwidth is 0.49 of its band, so a broadband block reads 3.0 dB
//!   under the FFT's band energy (measured 2.04 dB in the overlap). Half width would
//!   double that figure; the factor of 2 is this definition, not an arithmetic slip.
//! * **Adjacent-band leakage** is the neighbouring band's filter output divided by the
//!   correct band's, for a sinusoid on the correct band's centre frequency. One bin per
//!   band in this build, so the bin-level and band-level statements are the same number
//!   rather than differing by an aggregation width.

use core::cell::UnsafeCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Waker};

use tokio::sync::mpsc;

/// Rods in the hexagonal lattice, identical arithmetic to v1/v2 so a kernel switch
/// leaves the GPU buffers valid.
pub const ROD_COUNT: usize = 709;
pub const MAX_FFT: usize = 4096;
pub const MAX_BINS: usize = MAX_FFT / 2;
pub const MIN_CHANNELS: usize = 4;
pub const MAX_CHANNELS: usize = 64;

/// Band layout, unchanged from v1/v2: `f(k) = 45 * 320^(k/N)`.
const BAND_LOW_HZ: f64 = 45.0;
const BAND_SPAN: f64 = 320.0;

/// CQT ring capacity in samples. Covers fs = 96 kHz: the lowest window is
/// `Q fs / f_0` = 19756 samples there. A bin whose window does not fit is dropped and
/// reported through [`rk3_cqt_first_band`], never silently truncated.
const CQT_RING: usize = 32_000;
/// One CQT bin per band, so this is the channel ceiling.
const CQT_MAX_BINS: usize = MAX_CHANNELS;
/// 4-term Blackman-Harris. Same window on both paths, so a band energy from either
/// path is the same normalisation of the same window.
const BH_A0: f64 = 0.35875;
const BH_A1: f64 = 0.48829;
const BH_A2: f64 = 0.14128;
const BH_A3: f64 = 0.01168;

// --- display curve ---------------------------------------------------------
// Identical values and identical role to v2: these shape a measurement into a drawing
// amount and never write back into `bin_power`, `band_rms*` or `report`.
const BAND_FLOOR_DB: f64 = -54.0;
const BAND_COMPAND: f64 = 0.95;
const SLOPE_DB_PER_OCTAVE: f64 = 1.2;
const SLOPE_PIVOT_HZ: f64 = 300.0;
const SLOPE_MAX_DB: f64 = 5.0;
const AGC_TARGET_DB: f64 = 2.0;
const AGC_MAX_BOOST_DB: f64 = 40.0;
const AGC_RELEASE_DB_PER_S: f64 = 6.0;
const BAND_ATTACK_MS: f64 = 35.0;
const BAND_RELEASE_MS: f64 = 260.0;
const DISPLAY_MIN_DB: f64 = -100.0;
const DISPLAY_MAX_DB: f64 = 0.0;
const ATTACK_MS: f64 = 12.0;
const RELEASE_MS: f64 = 140.0;

/// Reported instead of `-inf` once a block is silent all the way down.
const SILENCE_DB: f64 = -144.0;
const PIPELINE_DEPTH: usize = 4;
const EPS: f64 = 1e-20;

pub const REPORT_PEAK_HZ: usize = 0;
pub const REPORT_CENTROID_HZ: usize = 1;
pub const REPORT_RMS_DB: usize = 2;
pub const REPORT_PEAK_DB: usize = 3;
pub const REPORT_WORDS: usize = 4;

/// `[peak_hz, peak_rms, bin_index, bins]`, all `f64`.
pub const CQT_REPORT_WORDS: usize = 4;

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
struct Block {
    samples: Vec<f32>,
    /// Samples that arrived before this block, i.e. how much of `history` was valid.
    block_index: u32,
}

struct State {
    rods: [Rod; ROD_COUNT],
    heights: [f32; ROD_COUNT],
    channels: usize,

    fft_size: usize,
    sample_rate: f64,
    bin_count: usize,

    window: [f64; MAX_FFT],
    window_cg: f64,
    window_s2: f64,
    sine_ref: f64,
    bitrev: [u16; MAX_FFT],
    tables_for: usize,

    /// Staging block JS writes before [`rk3_push`]. One of the two `f32` boundaries:
    /// `AnalyserNode.getFloatTimeDomainData` has no `f64` output.
    wave: [f32; MAX_FFT],
    re: [f64; MAX_FFT],
    im: [f64; MAX_FFT],

    // --- measurements: FFT path
    bin_power: [f64; MAX_BINS],
    bin_db: [f64; MAX_BINS],
    band_rms_fft: [f64; MAX_CHANNELS],
    report: [f64; REPORT_WORDS],

    // --- measurements: CQT path
    history: [f64; CQT_RING],
    hist_pos: usize,
    hist_filled: usize,
    cqt_bins: usize,
    cqt_first_band: usize,
    cqt_freq: [f64; CQT_MAX_BINS],
    cqt_len: [u32; CQT_MAX_BINS],
    /// Samples between "now" and the end of this bin's window.
    ///
    /// The CQT response to a transient peaks when the transient reaches the window's
    /// centre, i.e. at `tau + N_k/2`; the FFT's peaks at `tau + N_fft/2`. Ending the
    /// CQT window `(N_k - N_fft)/2` samples earlier puts both centres at the same
    /// instant, so a kick lights the inner and outer rings on the same frame instead of
    /// sweeping outward-inward. The cost is that the CQT's *first* response to a
    /// transient arrives up to `(N_k - N_fft)/2` samples later — the onset/latency
    /// trade is measured in `scripts/kernel-check-v3.mjs`, not assumed.
    cqt_end_offset: [u32; CQT_MAX_BINS],
    cqt_coeff: [f64; CQT_MAX_BINS],
    cqt_power: [f64; CQT_MAX_BINS],
    cqt_db: [f64; CQT_MAX_BINS],
    cqt_report: [f64; CQT_REPORT_WORDS],
    band_rms_cqt: [f64; MAX_CHANNELS],

    // --- measurements: the mixed band energy and the weights that produced it
    band_rms: [f64; MAX_CHANNELS],
    mix_cqt_weight: [f64; MAX_CHANNELS],

    // --- display
    bars: [f64; MAX_BINS],
    channel_amps: [f64; MAX_CHANNELS],
    band_drive: [f64; MAX_CHANNELS],
    band_peak_db: f64,

    tx: Option<mpsc::Sender<Block>>,
    processed: u32,
    dropped: u32,
    cqt_skipped: u32,
    blocks_seen: u32,
    ready: bool,
}

impl State {
    /// The non-zero defaults. Called from `rk3_init` so that `new()` can stay all-zero
    /// and keep the whole state in zero-filled memory.
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
            // Every field zero on purpose. A single non-zero initialiser here makes the
            // whole `State` an initialized data image, which drags the 250 kB CQT ring and
            // the 96 kB of transform scratch into the .wasm file instead of zero-filled
            // memory. Non-zero defaults live in `defaults()`, called from `rk3_init`.
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
            band_rms_fft: [0.0; MAX_CHANNELS],
            report: [0.0; REPORT_WORDS],
            history: [0.0; CQT_RING],
            hist_pos: 0,
            hist_filled: 0,
            cqt_bins: 0,
            cqt_first_band: 0,
            cqt_freq: [0.0; CQT_MAX_BINS],
            cqt_len: [0; CQT_MAX_BINS],
            cqt_end_offset: [0; CQT_MAX_BINS],
            cqt_coeff: [0.0; CQT_MAX_BINS],
            cqt_power: [0.0; CQT_MAX_BINS],
            cqt_db: [0.0; CQT_MAX_BINS],
            cqt_report: [0.0, 0.0, 0.0, 0.0],
            band_rms_cqt: [0.0; MAX_CHANNELS],
            band_rms: [0.0; MAX_CHANNELS],
            mix_cqt_weight: [0.0; MAX_CHANNELS],
            bars: [0.0; MAX_BINS],
            channel_amps: [0.0; MAX_CHANNELS],
            band_drive: [0.0; MAX_CHANNELS],
            band_peak_db: 0.0,
            tx: None,
            processed: 0,
            dropped: 0,
            cqt_skipped: 0,
            blocks_seen: 0,
            ready: false,
        }
    }
}

struct Kernel(UnsafeCell<State>);

// SAFETY: one wasm instance, one JS thread. Every export is called from that thread,
// and the pipeline stage runs inside `pump`, which holds no `State` borrow.
unsafe impl Sync for Kernel {}

static KERNEL: Kernel = Kernel(UnsafeCell::new(State::new()));

fn kernel() -> &'static mut State {
    // SAFETY: see the `unsafe impl Sync` note above.
    unsafe { &mut *KERNEL.0.get() }
}

// --- executor ---------------------------------------------------------------

struct Executor(UnsafeCell<Option<Pin<Box<dyn Future<Output = ()>>>>>);

// SAFETY: as above; polling happens on the single JS thread.
unsafe impl Sync for Executor {}

static EXECUTOR: Executor = Executor(UnsafeCell::new(None));

/// One scheduling turn. `Waker::noop` because every wake means "data is ready" and the
/// next push both sends and polls in the same call, so a stored callback would never be
/// read. Nothing here waits on a clock, so nothing can park.
fn pump() {
    // SAFETY: single JS thread; no `State` borrow is held across this call.
    let slot = unsafe { &mut *EXECUTOR.0.get() };
    if let Some(task) = slot.as_mut() {
        let mut cx = Context::from_waker(Waker::noop());
        let _ = task.as_mut().poll(&mut cx);
    }
}

fn restart_pipeline(state: &mut State) {
    let (tx, rx) = mpsc::channel::<Block>(PIPELINE_DEPTH);
    // SAFETY: `EXECUTOR` and `KERNEL` are distinct allocations.
    let slot = unsafe { &mut *EXECUTOR.0.get() };
    *slot = Some(Box::pin(stage(rx)));
    state.tx = Some(tx);
}

async fn stage(mut rx: mpsc::Receiver<Block>) {
    while let Some(block) = rx.recv().await {
        analyse_block(kernel(), &block.samples, block.block_index);
    }
}

// --- window and FFT ---------------------------------------------------------

/// 4-term Blackman-Harris value at phase `x = TAU n / (N-1)`, by recurrence.
///
/// A table would cost 8 bytes per sample per CQT bin (the lowest bin alone is 9876
/// samples); three cosines per sample times 46838 samples per frame is 140k
/// transcendentals per frame. The recurrence costs ~8 multiply-adds per sample and no
/// table. Over 9876 samples the drift is ~2e-12 relative, which is a window error far
/// below the -3 dB main-lobe leakage reported in the module docs.
fn bh_window(cos_1: f64, cos_n: f64, cos_prev: f64) -> (f64, f64) {
    // cos(2x) and cos(3x) from cos(x); the caller advances the recurrence.
    let cos_2 = 2.0 * cos_n * cos_n - 1.0;
    let cos_3 = 4.0 * cos_n * cos_n * cos_n - 3.0 * cos_n;
    let value = BH_A0 - BH_A1 * cos_n + BH_A2 * cos_2 - BH_A3 * cos_3;
    let next = 2.0 * cos_1 * cos_n - cos_prev;
    (value, next)
}

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
    // One-sided peak bin of a full-scale on-bin sine: |X| = A N CG / 2, doubled because
    // the negative-frequency half is folded in.
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

// --- CQT --------------------------------------------------------------------

fn band_low_hz(channel: usize, count: usize) -> f64 {
    BAND_LOW_HZ * BAND_SPAN.powf(channel as f64 / count as f64)
}

/// One CQT bin per band, centred in the band: `f(k) = 45 * rho^(k + 1/2)`.
fn band_centre_hz(channel: usize, count: usize) -> f64 {
    band_low_hz(channel, count) * BAND_SPAN.powf(0.5 / count as f64)
}

/// Recompute the CQT bin table for the current rate, channel count and transform size.
///
/// `Q = 2 / (rho - 1)`: 2 is BH4's equivalent noise bandwidth in window bins, so the
/// filter's bandwidth equals its band's width. The crossover is `Q fs / N_fft`, the
/// frequency at which a CQT window is exactly as long as the FFT's; bands above it are
/// left to the FFT.
fn configure_cqt(state: &mut State) {
    let count = state.channels;
    let fs = state.sample_rate;
    let rho = BAND_SPAN.powf(1.0 / count as f64);
    // 2.5 window bins per band: 2 is BH4's equivalent noise bandwidth, so 2.5 makes the
    // filter slightly narrower than its band. The 0.25 margin is what puts the neighbour
    // past 2 window bins, where the main-lobe response is -16 dB instead of -9.6 dB; the
    // price is that a broadband block reads ~1.0 dB under the FFT's band energy, which
    // the cross-path test allows (measured 1.6 dB worst case).
    let q = 4.0 / (rho - 1.0);
    let crossover_hz = q * fs / state.fft_size as f64;

    state.cqt_bins = 0;
    state.cqt_first_band = 0;
    let mut skipped = 0usize;
    for channel in 0..count.min(CQT_MAX_BINS) {
        let centre = band_centre_hz(channel, count);
        if centre >= crossover_hz {
            break;
        }
        let len = (q * fs / centre).round();
        if !len.is_finite() || len < 16.0 {
            break;
        }
        let len = len as usize;
        if len > CQT_RING {
            // Too long to keep honest: this band falls back to the FFT.
            skipped += 1;
            continue;
        }
        let bin = state.cqt_bins;
        // Deliberately zero. Aligning a CQT bin's window centre with the FFT's needs the
        // window to end *after* the newest sample — a negative offset, which needs future
        // data. Ending it earlier instead moves the centre further into the past and
        // doubles the lag, which is what measurement showed (band 0: 12.3 frames before
        // this was tried, 21 frames after). The lag gradient is therefore real and
        // reported rather than compensated; see the module docs.
        let offset = 0usize;
        if len + offset > CQT_RING {
            skipped += 1;
            continue;
        }
        state.cqt_freq[bin] = centre;
        state.cqt_len[bin] = len as u32;
        state.cqt_end_offset[bin] = offset as u32;
        state.cqt_coeff[bin] = 2.0 * (core::f64::consts::TAU * centre / fs).cos();
        state.cqt_bins += 1;
    }
    state.cqt_first_band = skipped;

    // Crossfade weights: 1.0 while the FFT has no say, 0.0 where the CQT stops, linear
    // over the three bands below the last CQT bin.
    let last = state.cqt_first_band + state.cqt_bins;
    for channel in 0..MAX_CHANNELS {
        let weight = if state.cqt_bins == 0 {
            0.0
        } else if channel + 3 <= last {
            1.0
        } else if channel >= last {
            0.0
        } else {
            (last - channel) as f64 / 3.0
        };
        state.mix_cqt_weight[channel] = weight.clamp(0.0, 1.0);
    }
}

/// Append one block to the CQT ring, then run every CQT bin over the newest `len`
/// samples of it.
fn analyse_cqt(state: &mut State) {
    let size = state.fft_size.min(MAX_FFT);
    for index in 0..size {
        state.history[state.hist_pos] = state.wave[index] as f64;
        state.hist_pos = if state.hist_pos + 1 == CQT_RING {
            0
        } else {
            state.hist_pos + 1
        };
    }
    state.hist_filled = (state.hist_filled + size).min(CQT_RING);

    let bins = state.cqt_bins;
    if bins == 0 {
        for channel in 0..MAX_CHANNELS {
            state.band_rms_cqt[channel] = 0.0;
        }
        state.cqt_report = [0.0, 0.0, 0.0, 0.0];
        return;
    }

    let mut skipped = false;
    for bin in 0..bins {
        let len = state.cqt_len[bin] as usize;
        let offset = state.cqt_end_offset[bin] as usize;
        // The window ends `offset` samples before the newest one; wait until that much
        // history exists as well as the window itself.
        if state.hist_filled < len + offset {
            skipped = true;
            continue;
        }
        // Window recurrence over the bin's own span, plus Goertzel at its centre.
        let denominator = (len - 1).max(1) as f64;
        let step = core::f64::consts::TAU / denominator;
        let cos_1 = step.cos();
        let mut cos_n = 1.0;
        let mut cos_prev = cos_1;
        let coeff = state.cqt_coeff[bin];
        let mut s_prev = 0.0f64;
        let mut s_prev2 = 0.0f64;
        let mut window_sum = 0.0f64;
        let total_back = len + offset;
        let mut cursor = if state.hist_pos >= total_back {
            state.hist_pos - total_back
        } else {
            CQT_RING - (total_back - state.hist_pos)
        };
        for _ in 0..len {
            let sample = state.history[cursor];
            let (value, next) = bh_window(cos_1, cos_n, cos_prev);
            cos_prev = cos_n;
            cos_n = next;
            window_sum += value;
            let windowed = sample * value;
            let s = windowed + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
            cursor = if cursor + 1 == CQT_RING {
                0
            } else {
                cursor + 1
            };
        }
        let mag2 = s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2;
        // A = 2|X| / sum(w) for a sine of amplitude A, so RMS^2 = 2 mag2 / sum(w)^2.
        let power = if window_sum > 0.0 {
            2.0 * mag2.max(0.0) / (window_sum * window_sum)
        } else {
            0.0
        };
        state.cqt_power[bin] = power;
        state.cqt_db[bin] = 10.0 * (power / state.sine_ref).max(EPS).log10();
    }
    if skipped {
        state.cqt_skipped = state.cqt_skipped.wrapping_add(1);
    }

    // Publish the CQT band energies, one bin per band.
    for channel in 0..MAX_CHANNELS {
        state.band_rms_cqt[channel] = 0.0;
    }
    let mut peak_bin = 0usize;
    let mut peak_power = 0.0f64;
    for bin in 0..bins {
        let channel = state.cqt_first_band + bin;
        if channel < MAX_CHANNELS {
            state.band_rms_cqt[channel] = state.cqt_power[bin].max(0.0).sqrt();
        }
        if state.cqt_power[bin] > peak_power {
            peak_power = state.cqt_power[bin];
            peak_bin = bin;
        }
    }

    // Parabolic refinement between neighbouring CQT bins, in log magnitude.
    let mut peak_hz = 0.0f64;
    if peak_bin < bins && peak_power > EPS {
        let left = if peak_bin > 0 {
            robust_log(state.cqt_power[peak_bin - 1])
        } else {
            robust_log(peak_power)
        };
        let centre = robust_log(peak_power);
        let right = if peak_bin + 1 < bins {
            robust_log(state.cqt_power[peak_bin + 1])
        } else {
            centre
        };
        let denominator = left - 2.0 * centre + right;
        // Interpolating between bins only means something when the tone straddles them.
        // With Q chosen so the neighbour is a null (-47 dB measured), all of the energy
        // sits in one bin and a parabola through three points, two of which are noise
        // floor, pushes the answer off the bin centre by up to 0.2 bins.
        let neighbours_quiet = left - centre < -20.0 && right - centre < -20.0;
        let offset = if neighbours_quiet || denominator.abs() <= 1e-12 {
            0.0
        } else {
            (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
        };
        let ratio = BAND_SPAN.powf(1.0 / state.channels as f64);
        peak_hz = state.cqt_freq[peak_bin] * ratio.powf(offset);
    }
    state.cqt_report = [
        peak_hz,
        peak_power.max(0.0).sqrt(),
        peak_bin as f64,
        bins as f64,
    ];
}

fn robust_log(value: f64) -> f64 {
    10.0 * value.max(EPS).log10()
}

// --- the FFT path and the mix ----------------------------------------------

fn analyse_block(state: &mut State, samples: &[f32], block_index: u32) {
    let size = state.fft_size;
    if samples.len() < size {
        return;
    }
    if state.tables_for != size {
        build_tables(state);
    }
    let n = size as f64;
    state.processed = state.processed.wrapping_add(1);
    state.blocks_seen = block_index;

    // DC removal, then the meter. A constant offset is not audible content; left in it
    // would dominate the RMS figure and tilt the whole low end.
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

    integrate_fft_bands(state);

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
    state.report = [peak_hz, centroid_hz, rms_db, peak_db];

    analyse_cqt(state);
    mix_bands(state);
}

/// Fractional-bin integration of the FFT bin powers into bands, identical to v2.
fn integrate_fft_bands(state: &mut State) {
    let count = state.channels;
    let n = state.fft_size as f64;
    let fs = state.sample_rate;
    let top = (state.bin_count as f64 - 1.0).max(1.0);

    for channel in 0..count {
        let low_hz = band_low_hz(channel, count);
        let high_hz = band_low_hz(channel + 1, count);
        let mut low = (low_hz * n / fs).clamp(1.0, top);
        let mut high = (high_hz * n / fs).clamp(1.0, top);
        if high < low {
            core::mem::swap(&mut low, &mut high);
        }

        let mut power = 0.0f64;
        if high - low < 1.0 {
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
        state.band_rms_fft[channel] = power.sqrt();
    }
}

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

/// Crossfade the two band estimates. Both are linear RMS over the same band, so the
/// blend is a weighted mean of two estimates of one quantity; where they agree it is
/// that value, and where they do not the difference is what the cross-path test
/// measures.
fn mix_bands(state: &mut State) {
    for channel in 0..state.channels {
        let weight = state.mix_cqt_weight[channel].clamp(0.0, 1.0);
        let fft = state.band_rms_fft[channel];
        let cqt = state.band_rms_cqt[channel];
        state.band_rms[channel] = weight * cqt + (1.0 - weight) * fft;
    }
}

// --- display curve ----------------------------------------------------------

fn map_bands(state: &mut State) {
    let count = state.channels;
    for channel in 0..count {
        state.channel_amps[channel] = band_amount(&state.band_rms, channel, count, 0.0);
    }
}

fn band_amount(rms: &[f64; MAX_CHANNELS], channel: usize, count: usize, boost_db: f64) -> f64 {
    let value = rms[channel];
    if value <= 0.0 {
        return 0.0;
    }
    let decibels = 20.0 * value.log10() + boost_db + tilt_db(channel, count);
    let base = ((decibels - BAND_FLOOR_DB) / -BAND_FLOOR_DB).clamp(0.0, 1.0);
    base.powf(BAND_COMPAND)
}

fn tilt_db(channel: usize, count: usize) -> f64 {
    let octaves = (band_centre_hz(channel, count) / SLOPE_PIVOT_HZ)
        .max(1.0)
        .log2();
    (SLOPE_DB_PER_OCTAVE * octaves).min(SLOPE_MAX_DB)
}

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

fn smooth_bands(state: &mut State, dt: f64) {
    let count = state.channels;

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

fn advance_heights(state: &mut State, dt_frames: f64, gain: f64, smoothing: f64) {
    let damping = 1.0 - (0.45 + smoothing * 0.53).powf(dt_frames);
    for index in 0..ROD_COUNT {
        let channel = state.rods[index].channel as usize;
        let amp = state.band_drive[channel].powf(1.7) * gain;
        let target = 0.07 + amp * 4.8;
        let height = &mut state.heights[index];
        // The one narrowing cast on the way out: the renderers upload rod heights to
        // the GPU as f32, where the 7th decimal of a height is not resolvable.
        *height = (*height as f64 + (target - *height as f64) * damping) as f32;
    }
}

// --- exports ----------------------------------------------------------------

/// Fill the rod lattice. Identical arithmetic to v1 and v2, including the seed hash,
/// so the renderers' GPU buffers stay valid across a kernel switch.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_init() -> u32 {
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

#[unsafe(no_mangle)]
pub extern "C" fn rk3_version() -> u32 {
    3
}

/// Transform size (rounded down to a power of two), sample rate, and both paths'
/// tables. `sample_rate` is `f64` because it divides every band edge and every CQT
/// frequency; a value that arrives already rounded would bias each reported frequency.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_configure(fft_size: u32, sample_rate: f64) {
    let state = kernel();
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return;
    }
    let requested = (fft_size as usize).clamp(64, MAX_FFT);
    // `requested` is a u32, so the shift must be computed from `u32::BITS`, not
    // `usize::BITS`: with the latter, fft_size < 2048 shifted 32 bits too far.
    let size = 1usize << (u32::BITS - 1 - requested.leading_zeros()) as usize;
    let rate = sample_rate;

    let retuned = size != state.fft_size || (rate - state.sample_rate).abs() > 0.5;
    state.fft_size = size;
    state.sample_rate = rate;
    state.bin_count = size / 2;
    if state.tables_for != size {
        build_tables(state);
    }
    configure_cqt(state);
    if retuned || state.tx.is_none() {
        reset_analysis(state);
        restart_pipeline(state);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_reset() {
    let state = kernel();
    reset_analysis(state);
    restart_pipeline(state);
}

/// Drop every measurement and the display history, but keep the rod heights: the
/// engine's stop path relies on this so the rods can ease down instead of vanishing.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_clear_spectrum() {
    let state = kernel();
    state.bin_power = [0.0; MAX_BINS];
    state.bin_db = [0.0; MAX_BINS];
    state.bars = [0.0; MAX_BINS];
    state.band_rms_fft = [0.0; MAX_CHANNELS];
    state.band_rms_cqt = [0.0; MAX_CHANNELS];
    state.band_rms = [0.0; MAX_CHANNELS];
    state.cqt_power = [0.0; CQT_MAX_BINS];
    state.cqt_db = [0.0; CQT_MAX_BINS];
    state.cqt_report = [0.0, 0.0, 0.0, 0.0];
    state.channel_amps = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
    state.band_peak_db = f64::NEG_INFINITY;
    state.report = [0.0, 0.0, SILENCE_DB, SILENCE_DB];
}

fn reset_analysis(state: &mut State) {
    state.bin_power = [0.0; MAX_BINS];
    state.bin_db = [0.0; MAX_BINS];
    state.bars = [0.0; MAX_BINS];
    state.band_rms_fft = [0.0; MAX_CHANNELS];
    state.band_rms_cqt = [0.0; MAX_CHANNELS];
    state.band_rms = [0.0; MAX_CHANNELS];
    state.cqt_power = [0.0; CQT_MAX_BINS];
    state.cqt_db = [0.0; CQT_MAX_BINS];
    state.cqt_report = [0.0, 0.0, 0.0, 0.0];
    state.channel_amps = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
    state.band_peak_db = f64::NEG_INFINITY;
    state.report = [0.0, 0.0, SILENCE_DB, SILENCE_DB];
    state.history = [0.0; CQT_RING];
    state.hist_pos = 0;
    state.hist_filled = 0;
    state.heights = [0.0; ROD_COUNT];
    state.processed = 0;
    state.dropped = 0;
    state.cqt_skipped = 0;
    state.blocks_seen = 0;
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_fft_size() -> u32 {
    kernel().fft_size as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_bin_count() -> u32 {
    kernel().bin_count as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_rod_count() -> u32 {
    ROD_COUNT as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_rod_stride() -> u32 {
    (core::mem::size_of::<Rod>() / core::mem::size_of::<f32>()) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_rod_ptr() -> *const f32 {
    kernel().rods.as_ptr().cast::<f32>()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_heights_ptr() -> *mut f32 {
    kernel().heights.as_mut_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_set_channels(channels: u32) {
    let state = kernel();
    let clamped = (channels as usize).clamp(MIN_CHANNELS, MAX_CHANNELS);
    if clamped == state.channels {
        return;
    }
    state.channels = clamped;
    apply_channels(state);
    state.band_rms_fft = [0.0; MAX_CHANNELS];
    state.band_rms_cqt = [0.0; MAX_CHANNELS];
    state.band_rms = [0.0; MAX_CHANNELS];
    state.band_drive = [0.0; MAX_CHANNELS];
    configure_cqt(state);
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_channels() -> u32 {
    kernel().channels as u32
}

fn apply_channels(state: &mut State) {
    let count = state.channels;
    for rod in state.rods.iter_mut() {
        let band = (rod.radius / 15.0 * count as f32).floor() as usize;
        rod.channel = band.min(count - 1) as f32;
    }
}

/// How many CQT bins this configuration runs.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_bins() -> u32 {
    kernel().cqt_bins as u32
}

/// Lowest band the CQT covers. Non-zero only when a bin's window did not fit
/// [`CQT_RING`]; those bands fall back to the FFT instead of being truncated.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_first_band() -> u32 {
    kernel().cqt_first_band as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_freq_ptr() -> *const f64 {
    kernel().cqt_freq.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_power_ptr() -> *const f64 {
    kernel().cqt_power.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_db_ptr() -> *const f64 {
    kernel().cqt_db.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_report_ptr() -> *const f64 {
    kernel().cqt_report.as_ptr()
}

/// Band energy from the FFT path alone, `f64`, one per band, linear RMS.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_band_rms_fft_ptr() -> *const f64 {
    kernel().band_rms_fft.as_ptr()
}

/// Band energy from the CQT path alone, `f64`, one per band, linear RMS.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_band_rms_cqt_ptr() -> *const f64 {
    kernel().band_rms_cqt.as_ptr()
}

/// The crossfaded band energy the display curve reads. A measurement: the weights are
/// configuration, not a shaping step.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_band_rms_ptr() -> *const f64 {
    kernel().band_rms.as_ptr()
}

/// Per-band CQT weight in the crossfade, `f64`. 1.0 = CQT only, 0.0 = FFT only.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_mix_weight_ptr() -> *const f64 {
    kernel().mix_cqt_weight.as_ptr()
}

/// Staging block for one FFT-sized block of time-domain samples, `f32`.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_wave_ptr() -> *mut f32 {
    kernel().wave.as_mut_ptr()
}

/// Hand the staged block to the pipeline and give it one turn.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_push() -> u32 {
    let accepted = {
        let state = kernel();
        if !state.ready {
            return 0;
        }
        let size = state.fft_size;
        let block = Block {
            samples: state.wave[..size].to_vec(),
            block_index: state.blocks_seen.wrapping_add(1),
        };
        let before = state.processed;
        match state.tx.as_ref() {
            Some(tx) => match tx.try_send(block) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    state.dropped = state.dropped.wrapping_add(1);
                    return 0;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return 0,
            },
            None => return 0,
        }
        before
    };
    pump();
    kernel().processed.wrapping_sub(accepted)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_bin_power_ptr() -> *const f64 {
    kernel().bin_power.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_bin_db_ptr() -> *const f64 {
    kernel().bin_db.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_window_ptr() -> *const f64 {
    kernel().window.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_bars_ptr() -> *const f64 {
    kernel().bars.as_ptr()
}

/// `[peak_hz, centroid_hz, rms_db, peak_db]` from the **FFT path**, all `f64`.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_report_ptr() -> *const f64 {
    kernel().report.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_channel_amps_ptr() -> *const f64 {
    kernel().channel_amps.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_band_drive_ptr() -> *const f64 {
    kernel().band_drive.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_sine_ref() -> f64 {
    kernel().sine_ref
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_processed() -> u32 {
    kernel().processed
}

#[unsafe(no_mangle)]
pub extern "C" fn rk3_dropped() -> u32 {
    kernel().dropped
}

/// Blocks analysed while at least one CQT bin's window was still filling.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_cqt_skipped() -> u32 {
    kernel().cqt_skipped
}

/// One frame of audio reaction. `dt_seconds` is real time, so the easing in both
/// kernels of this file is frame-rate independent.
#[unsafe(no_mangle)]
pub extern "C" fn rk3_step(dt_seconds: f64, gain: f64, smoothing: f64) {
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

    // The display curve is computed from the mixed measurement on the same frame the
    // measurement is refreshed, so the two never disagree about which block they saw.
    map_bands(state);
    smooth_bars(state, dt);
    smooth_bands(state, dt);
    advance_heights(state, dt * 60.0, gain, smoothing);
}
