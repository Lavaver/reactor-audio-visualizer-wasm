// Analysis on the audio rendering thread.
//
// `AudioWorkletGlobalScope` has no `fetch`, no timers and no DOM, so the
// compiled `WebAssembly.Module` is handed over through `processorOptions` and
// all work is driven by `process()` calls. This keeps the window + FFT + dB
// mapping off the main thread entirely; only the resulting byte bins cross back.
//
// The kernel is the same wasm build the main thread uses, but this is a second
// instance with its own linear memory: the main thread keeps the rod model, this
// one owns the transform.

/** Analyse every hop samples; 50 % overlap halves the latency vs a full window. */
const HOP_RATIO = 0.5;

class ReactorAnalysisProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module, fftSize } = options.processorOptions ?? {};
    this.fftSize = fftSize || 2048;
    this.hop = Math.max(1, Math.round(this.fftSize * HOP_RATIO));
    this.window = new Float32Array(this.fftSize);
    this.filled = 0;
    this.exports = null;
    this.wave = null;
    this.bins = null;

    this.port.onmessage = event => {
      // Dropping the partial window avoids analysing audio from before a pause.
      if (event.data?.type === 'reset') this.filled = 0;
    };

    try {
      // Linking only: compilation already happened on the main thread.
      const instance = new WebAssembly.Instance(module);
      const exports = instance.exports;
      exports.rk_init();
      exports.rk_set_fft(this.fftSize);
      this.wave = new Float32Array(exports.memory.buffer, exports.rk_wave_ptr(), this.fftSize);
      this.bins = new Uint8Array(exports.memory.buffer, exports.rk_bins_ptr(), exports.rk_bin_count());
      this.exports = exports;
      this.port.postMessage({ type: 'ready', bins: this.bins.length });
    } catch (error) {
      this.port.postMessage({ type: 'error', message: String(error?.message ?? error) });
    }
  }

  process(inputs) {
    if (!this.exports) return true;
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let index = 0; index < channel.length; index++) {
      this.window[this.filled++] = channel[index];
      if (this.filled < this.fftSize) continue;

      this.wave.set(this.window);
      this.exports.rk_analyse();
      // `slice()` copies out of wasm memory, then the buffer is transferred.
      const copy = this.bins.slice();
      this.port.postMessage({ type: 'bins', bins: copy }, [copy.buffer]);

      this.window.copyWithin(0, this.hop);
      this.filled = this.fftSize - this.hop;
    }
    return true;
  }
}

registerProcessor('reactor-analysis', ReactorAnalysisProcessor);
