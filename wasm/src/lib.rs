//! Time-domain audio -> FFT -> spectrum -> channel energy -> rod heights.
//!
//! This crate is compiled to `wasm32-unknown-unknown` as a plain C-ABI cdylib.
//! The browser is only responsible for *capturing* samples: it copies one block
//! of time-domain data into [`rk_wave_ptr`], calls [`rk_analyse`] and then
//! [`rk_step`] once per animation frame, and reads rod heights, channel energies
//! and the byte spectrum back as typed-array views over this module's memory.
//!
//! Device access (`getUserMedia` / `getDisplayMedia`, OS permission prompts)
//! cannot live here: wasm has no syscalls and no device handles, so Web Audio
//! always hands the samples over. Everything after that point is in Rust.
//!
//! Everything here is single-threaded by construction: one wasm instance, one
//! JS main thread, no re-entrant calls.

use core::cell::UnsafeCell;

/// Number of rods in the hexagonal-ish core, matching the `|x|虏 + |z|虏 <= 15虏`
/// lattice the original renderer used.
pub const ROD_COUNT: usize = 709;
/// Largest transform the kernel supports; also bounds the time-domain block the
/// browser copies in.
pub const MAX_FFT: usize = 4096;
/// One byte per bin for the lower half of the spectrum.
pub const MAX_BINS: usize = MAX_FFT / 2;
pub const MIN_CHANNELS: usize = 4;
pub const MAX_CHANNELS: usize = 64;

/// Lowest band edge and total span of the logarithmic channel layout:
/// `band(k) = 45 Hz * 320^(k / channels)`, so the top edge is 14.4 kHz.
const BAND_LOW_HZ: f32 = 45.0;
const BAND_SPAN: f32 = 320.0;
/// Same decibel window the page used to get from `AnalyserNode`, so the byte
/// bins keep the meaning the renderer expects.
const MIN_DB: f32 = -100.0;
const MAX_DB: f32 = -30.0;
/// Analyser-style exponential smoothing on the linear magnitude.
const BIN_SMOOTHING: f32 = 0.75;

/// Geometry of one rod, laid out as five `f32` so JS can read every rod from a
/// single `Float32Array` with a stride of 5 and no per-field crossings.
///
/// `channel` is stored as `f32` because it only ever holds `0..=63`, which is
/// exactly representable; the cast back is one instruction per rod per frame.
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

struct State {
    rods: [Rod; ROD_COUNT],
    heights: [f32; ROD_COUNT],
    bins: [u8; MAX_BINS],
    bin_count: usize,
    /// Time-domain block the browser copies in before [`rk_analyse`].
    wave: [f32; MAX_FFT],
    /// Hann window, rebuilt whenever the transform size changes.
    window: [f32; MAX_FFT],
    fft_re: [f32; MAX_FFT],
    fft_im: [f32; MAX_FFT],
    /// Bit-reversal permutation for the current transform size.
    bitrev: [u16; MAX_FFT],
    /// Smoothed linear magnitudes, one per output bin.
    smooth: [f32; MAX_BINS],
    fft_size: usize,
    /// Which size the window/bitrev tables were built for.
    tables_for: usize,
    channel_amps: [f32; MAX_CHANNELS],
    channels: usize,
    ready: bool,
}

impl State {
    const fn new() -> Self {
        Self {
            rods: [ROD_ZERO; ROD_COUNT],
            heights: [0.0; ROD_COUNT],
            bins: [0; MAX_BINS],
            bin_count: MAX_BINS,
            wave: [0.0; MAX_FFT],
            window: [0.0; MAX_FFT],
            fft_re: [0.0; MAX_FFT],
            fft_im: [0.0; MAX_FFT],
            bitrev: [0; MAX_FFT],
            smooth: [0.0; MAX_BINS],
            fft_size: 2048,
            tables_for: 0,
            channel_amps: [0.0; MAX_CHANNELS],
            channels: 32,
            ready: false,
        }
    }
}

struct Kernel(UnsafeCell<State>);

// SAFETY: wasm32-unknown-unknown has no threads in this build, and every export
// below is called from the same JS thread without re-entrancy.
unsafe impl Sync for Kernel {}

static KERNEL: Kernel = Kernel(UnsafeCell::new(State::new()));

/// The single mutable handle every export goes through.
fn kernel() -> &'static mut State {
    // SAFETY: see the `unsafe impl Sync` note above.
    unsafe { &mut *KERNEL.0.get() }
}

/// Fill the rod lattice. Idempotent so a hot reload can call it twice.
#[unsafe(no_mangle)]
pub extern "C" fn rk_init() -> u32 {
    let state = kernel();
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
                        // Same hash as the original JS so colors stay identical.
                        seed: (x as f32 * 127.1 + z as f32 * 311.7).sin().abs(),
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
    index as u32
}

/// Map every rod onto one of `channels` bands by its radius: rods in the same
/// band move together, which is what makes the ring read as "one channel".
#[unsafe(no_mangle)]
pub extern "C" fn rk_set_channels(channels: u32) {
    let state = kernel();
    state.channels = (channels as usize).clamp(MIN_CHANNELS, MAX_CHANNELS);
    apply_channels(state);
}

fn apply_channels(state: &mut State) {
    let count = state.channels;
    for rod in state.rods.iter_mut() {
        let band = (rod.radius / 15.0 * count as f32).floor() as usize;
        rod.channel = band.min(count - 1) as f32;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_channels() -> u32 {
    kernel().channels as u32
}

/// Choose the transform size (rounded down to a power of two) and rebuild the
/// window / bit-reversal tables. Also clears the spectrum.
#[unsafe(no_mangle)]
pub extern "C" fn rk_set_fft(fft_size: u32) {
    let state = kernel();
    let requested = (fft_size as usize).clamp(64, MAX_FFT);
    // Largest power of two that still fits the request.
    let size = 1usize << (usize::BITS - 1 - requested.leading_zeros());
    state.fft_size = size;
    state.bin_count = size / 2;
    state.bins[..].fill(0);
    build_tables(state);
}

/// Current transform size in samples (twice the bin count).
#[unsafe(no_mangle)]
pub extern "C" fn rk_fft_size() -> u32 {
    kernel().fft_size as u32
}

/// Copy one block of time-domain samples here before calling [`rk_analyse`].
#[unsafe(no_mangle)]
pub extern "C" fn rk_wave_ptr() -> *mut f32 {
    kernel().wave.as_mut_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_bin_count() -> u32 {
    kernel().bin_count as u32
}

/// Byte spectrum, one entry per bin, 0..255.
#[unsafe(no_mangle)]
pub extern "C" fn rk_bins_ptr() -> *mut u8 {
    kernel().bins.as_mut_ptr()
}

/// Build the Hann window, the bit-reversal permutation and reset the smoothing
/// state. Cheap enough to run on a size change; never runs per frame.
fn build_tables(state: &mut State) {
    let size = state.fft_size;
    let denominator = (size - 1).max(1) as f32;
    for index in 0..size {
        // Hann: 0.5 - 0.5 cos(2蟺n/(N-1)), the window Chromium uses for analysis.
        state.window[index] =
            0.5 - 0.5 * (core::f32::consts::TAU * index as f32 / denominator).cos();
    }
    let shift = 32 - size.trailing_zeros();
    for index in 0..size {
        state.bitrev[index] = ((index as u32).reverse_bits() >> shift) as u16;
    }
    state.smooth[..size / 2].fill(0.0);
    state.tables_for = size;
}

/// In-place iterative radix-2 Cooley-Tukey FFT over `re`/`im`.
///
/// `bitrev` must hold the bit-reversal permutation for `re.len()`; the inputs
/// are expected to be already windowed. No allocation, no branches on NaN.
fn fft(re: &mut [f32], im: &mut [f32], bitrev: &[u16]) {
    let size = re.len();
    for index in 0..size {
        let reversed = bitrev[index] as usize;
        if reversed > index {
            re.swap(index, reversed);
            im.swap(index, reversed);
        }
    }
    let mut span = 2;
    while span <= size {
        let angle = -core::f32::consts::TAU / span as f32;
        let (step_re, step_im) = (angle.cos(), angle.sin());
        let half = span / 2;
        let mut start = 0;
        while start < size {
            let mut twiddle_re = 1.0f32;
            let mut twiddle_im = 0.0f32;
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

/// Analyse the samples behind [`rk_wave_ptr`] and fill the byte spectrum.
///
/// A sine of amplitude `A` lands at about `A / 2` after the `2 / N`
/// normalisation (the Hann window halves the coherent sum), which is the same
/// scale `AnalyserNode` reported to the old build. That keeps the decibel
/// window below meaningful, so the rods move as far as they used to.
#[unsafe(no_mangle)]
pub extern "C" fn rk_analyse() {
    let state = kernel();
    if !state.ready {
        return;
    }
    let size = state.fft_size;
    if state.tables_for != size {
        build_tables(state);
    }

    // Remove the DC offset before windowing. A source with a constant bias (a
    // microphone bias, a badly mastered file) parks its energy in bin 0, and the
    // Hann window smears it into bin 1 as well; the readouts then reported that
    // as a "0 Hz" peak frequency. Subtracting the window mean removes the bias
    // everywhere instead of hiding bin 0 downstream.
    let mut mean = 0.0;
    for index in 0..size {
        mean += state.wave[index];
    }
    mean /= size as f32;

    for index in 0..size {
        state.fft_re[index] = (state.wave[index] - mean) * state.window[index];
        state.fft_im[index] = 0.0;
    }
    {
        let State {
            fft_re,
            fft_im,
            bitrev,
            ..
        } = &mut *state;
        fft(&mut fft_re[..size], &mut fft_im[..size], &bitrev[..size]);
    }

    let norm = 2.0 / size as f32;
    let bins = state.bin_count.min(size / 2);
    for bin in 0..bins {
        let re = state.fft_re[bin];
        let im = state.fft_im[bin];
        let magnitude = (re * re + im * im).sqrt() * norm;
        let smoothed = state.smooth[bin] * BIN_SMOOTHING + magnitude * (1.0 - BIN_SMOOTHING);
        state.smooth[bin] = smoothed;
        let decibels = 20.0 * smoothed.max(1.0e-7).log10();
        let normalized = (decibels - MIN_DB) / (MAX_DB - MIN_DB);
        state.bins[bin] = (normalized.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_rod_count() -> u32 {
    ROD_COUNT as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_heights_ptr() -> *const f32 {
    kernel().heights.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_rod_x_ptr() -> *const f32 {
    kernel().rods.as_ptr().cast::<f32>()
}

/// Stride between two rods in the array behind [`rk_rod_x_ptr`], counted in
/// `f32` words: `[x, z, radius, seed, channel]` per rod.
#[unsafe(no_mangle)]
pub extern "C" fn rk_rod_stride() -> u32 {
    (size_of::<Rod>() / size_of::<f32>()) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_channel_amps_ptr() -> *const f32 {
    kernel().channel_amps.as_ptr()
}

/// One frame of audio reaction.
///
/// - `dt_frames`: elapsed time in 16.67 ms frames (already clamped by JS).
/// - `gain`: visual sensitivity multiplier.
/// - `smoothing`: 0..0.95; higher means the rods travel more slowly.
#[unsafe(no_mangle)]
pub extern "C" fn rk_step(dt_frames: f32, sample_rate: f32, gain: f32, smoothing: f32) {
    let state = kernel();
    if !state.ready {
        return;
    }
    let dt = if dt_frames.is_finite() {
        dt_frames.clamp(0.1, 3.0)
    } else {
        1.0
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
    let sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    };

    compute_channels(state, sample_rate);
    advance_heights(state, dt, gain, smoothing);
}

/// Average the spectrum inside each logarithmic band, normalized to 0..1.
fn compute_channels(state: &mut State, sample_rate: f32) {
    let channels = state.channels;
    let bins = state.bin_count;
    let bin_hz = (sample_rate / (bins as f32 * 2.0)).max(0.1);
    let last = bins.saturating_sub(2).max(1);

    for channel in 0..channels {
        let low = BAND_LOW_HZ * BAND_SPAN.powf(channel as f32 / channels as f32);
        let high = BAND_LOW_HZ * BAND_SPAN.powf((channel + 1) as f32 / channels as f32);
        let start = ((low / bin_hz).round() as usize).clamp(1, last);
        let end = ((high / bin_hz).round() as usize).clamp(1, last);
        let (start, end) = if end >= start {
            (start, end)
        } else {
            (end, start)
        };

        let mut sum = 0u32;
        for bin in start..=end {
            sum += state.bins[bin] as u32;
        }
        let count = (end - start + 1) as f32;
        state.channel_amps[channel] = sum as f32 / (count * 255.0);
    }
}

fn advance_heights(state: &mut State, dt: f32, gain: f32, smoothing: f32) {
    // Same easing curve as the JS version: one scalar, applied to every rod.
    let damping = 1.0 - (0.45 + smoothing * 0.53).powf(dt);
    for index in 0..ROD_COUNT {
        let channel = state.rods[index].channel as usize;
        let amp = state.channel_amps[channel].powf(1.7) * gain;
        let target = 0.07 + amp * 4.8;
        let height = &mut state.heights[index];
        *height += (target - *height) * damping;
    }
}
