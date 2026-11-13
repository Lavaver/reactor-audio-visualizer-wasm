// Exercises the local-file path end to end: decode, duration, playback, seek.
//
//   node app/scripts/file-check.mjs [cdpPort] [pageUrl] [wavPath] [highWavPath]
//
// The fixtures are synthesized here unless paths are given, so the check needs
// nothing checked in; Chromium decodes plain PCM WAV natively.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const [port = '9345', pageUrl = 'http://127.0.0.1:4174/', wavPath, highWavPath] = process.argv.slice(2);
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

/** A mono 16-bit PCM WAV: one sine with a 2 Hz tremolo, plus an optional offset. */
function toneWav(frequency, { seconds = 5, sampleRate = 44100, amplitude = 0.6, offset = 0 } = {}) {
  const frames = Math.round(seconds * sampleRate);
  const samples = Buffer.alloc(frames * 2);
  for (let i = 0; i < frames; i++) {
    const t = i / sampleRate;
    const envelope = 0.45 + 0.35 * Math.sin(2 * Math.PI * 2 * t);
    const value = Math.sin(2 * Math.PI * frequency * t) * envelope * amplitude + offset;
    samples.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(value * 32767))), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(36 + samples.length, 4);
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(samples.length, 40);
  return Buffer.concat([header, samples]);
}

const fixtures = join(dirname(fileURLToPath(import.meta.url)), '..', '.fixtures');
const lowTone = wavPath ?? join(fixtures, 'tone-440.wav');
const highTone = highWavPath ?? join(fixtures, 'tone-2k.wav');
/** 1 kHz riding on a large DC offset: DC must never be reported as a pitch. */
const offsetTone = join(fixtures, 'tone-dc.wav');
await mkdir(fixtures, { recursive: true });
if (!wavPath) await writeFile(lowTone, toneWav(440));
if (!highWavPath) await writeFile(highTone, toneWav(2000));
await writeFile(offsetTone, toneWav(1000, { amplitude: 0.35, offset: 0.5 }));

let failed = 0;
const check = (label, actual, expected) => {
  const pass = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`${pass ? 'PASS' : 'FAIL'} ${label}: ${JSON.stringify(actual)}${pass ? '' : ` != ${JSON.stringify(expected)}`}`);
  if (!pass) failed++;
};
const checkTrue = (label, actual, detail = '') => {
  console.log(`${actual ? 'PASS' : 'FAIL'} ${label}${detail ? `: ${detail}` : ''}`);
  if (!actual) failed++;
};

const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = targets.find(entry => entry.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});
let seq = 0;
const pending = new Map();
const errors = [];
ws.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const entry = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) entry.reject(new Error(message.error.message));
    else entry.resolve(message.result);
    return;
  }
  if (message.method === 'Runtime.exceptionThrown') {
    const details = message.params.exceptionDetails;
    if ((details.url ?? '').includes('127.0.0.1')) errors.push(details.exception?.description ?? details.text);
  }
};
const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression, userGesture = false) => {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true, userGesture });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  return result.result.value;
};
const state = () =>
  evaluate(`JSON.stringify({
    track: document.getElementById('trackname')?.textContent,
    duration: document.getElementById('duration')?.textContent,
    elapsed: document.getElementById('elapsed')?.textContent,
    status: document.getElementById('status')?.textContent,
    seek: document.getElementById('seek')?.value,
    seekEnabled: !document.getElementById('seek')?.disabled,
    level: document.getElementById('level')?.textContent,
    peaklevel: document.getElementById('peaklevel')?.textContent,
    freq: document.getElementById('freq')?.textContent,
    avgfreq: document.getElementById('avgfreq')?.textContent,
    message: document.getElementById('message')?.textContent
  })`).then(JSON.parse);

await send('Page.enable');
await send('Runtime.enable');
await send('DOM.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: pageUrl });
await wait(1800);

// Feed the fixture straight into the hidden <input type=file>.
const document = await send('DOM.getDocument', { depth: -1 });
const { nodeId } = await send('DOM.querySelector', {
  nodeId: document.root.nodeId,
  selector: '#file',
});
checkTrue('file input found', nodeId > 0, String(nodeId));
await send('DOM.setFileInputFiles', { nodeId, files: [lowTone] });
await wait(1500);

const loaded = await state();
console.log('LOADED', JSON.stringify(loaded));
check('track name shows the file', loaded.track, lowTone.split(/[\\/]/).pop());
check('duration comes from decoded audio', loaded.duration, '00:05');
check('file playback started', loaded.status, '运行中 · 音频分析');
check('seek enabled for files', loaded.seekEnabled, true);

/** "444Hz" -> 444, "2.01kHz" -> 2010, anything else -> NaN. */
const readHertz = text => {
  const match = /^([\d.]+)(kHz|Hz)$/.exec(String(text ?? '').trim());
  if (!match) return Number.NaN;
  return Number(match[1]) * (match[2] === 'kHz' ? 1000 : 1);
};

// The fixture is a pure 440 Hz sine, so the power-weighted centroid must land on
// the tone itself. A byte-weighted centroid used to drift to ~8 kHz here because
// the empty high bins still carry weight.
const centroid = readHertz(loaded.avgfreq);
checkTrue('average frequency follows the 440 Hz tone', Math.abs(centroid - 440) <= 60, `${loaded.avgfreq}`);
checkTrue('440 Hz stays in Hz', /^\d+Hz$/.test(String(loaded.avgfreq ?? '').trim()), loaded.avgfreq);
const peakLevel = Number(String(loaded.peaklevel ?? '').replace(/[^\d-]/g, ''));
const levelNow = Number(String(loaded.level ?? '').replace(/[^\d-]/g, ''));
checkTrue('peak hold is not below the current level', peakLevel >= levelNow, `${loaded.peaklevel} vs ${loaded.level}`);

// The fixture is tremolo-modulated (2 Hz, ~11 dB swing), so a healthy peak hold
// must visibly move. The first implementation held the session maximum and
// looked frozen.
const peakSamples = [loaded.peaklevel];
for (let i = 0; i < 4; i++) {
  await wait(250);
  peakSamples.push((await state()).peaklevel);
}
checkTrue('peak hold moves on a modulated source', new Set(peakSamples).size > 1, peakSamples.join(' '));

await wait(900);
const advanced = await state();
checkTrue('elapsed advances while playing', advanced.elapsed !== loaded.elapsed, `${loaded.elapsed} -> ${advanced.elapsed}`);
check('still playing mid-clip', advanced.status, '运行中 · 音频分析');

// Jump to the middle; the elapsed label must follow the seek.
const beforeSeek = await state();
console.log('BEFORE SEEK', JSON.stringify({ elapsed: beforeSeek.elapsed, seek: beforeSeek.seek }));
// Drive the slider the way a user does: press, move, release. The component only
// previews while dragging and commits on release, so a lone `input` event would
// legitimately do nothing.
await evaluate(`(() => {
  const slider = document.getElementById('seek');
  const rect = slider.getBoundingClientRect();
  const pointer = (type, target, x, y) => target.dispatchEvent(new PointerEvent(type, {
    bubbles: true, pointerId: 7, pointerType: 'mouse', clientX: x, clientY: y,
  }));
  pointer('pointerdown', slider, rect.left + rect.width * 0.2, rect.top + rect.height / 2);
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  setter.call(slider, '50');
  slider.dispatchEvent(new Event('input', { bubbles: true }));
  pointer('pointerup', window, rect.left + rect.width / 2, rect.top + rect.height / 2);
  return 'ok';
})()`, true);
await wait(200);
const justAfter = await state();
console.log('AFTER SEEK', JSON.stringify({ elapsed: justAfter.elapsed, seek: justAfter.seek, status: justAfter.status }));
await wait(500);
const seeked = await state();
console.log('SEEKED', JSON.stringify(seeked));
// Percentages are timing-independent; an exact "00:02" label depends on how long
// the harness waited after the release.
const percent = text => Number(String(text ?? '').replace(/[^\d.]/g, ''));
checkTrue(
  'seek landed near the middle',
  percent(justAfter.seek) >= 45 && percent(justAfter.seek) <= 60,
  justAfter.seek,
);
checkTrue(
  'seek rewound playback',
  percent(justAfter.seek) < percent(beforeSeek.seek),
  `${beforeSeek.seek} -> ${justAfter.seek}`,
);
check('playback continues after seek', seeked.status, '运行中 · 音频分析');

// 1 kHz and up is shown in kHz with two decimals ("2.00kHz"). Only a source up
// there can demonstrate the switch; the 440 Hz fixture must stay in Hz.
await send('DOM.setFileInputFiles', { nodeId, files: [highTone] });
await wait(1600);
const high = await state();
console.log('HIGH TONE', JSON.stringify({ track: high.track, avgfreq: high.avgfreq, freq: high.freq }));
check('high fixture loaded', high.track, highTone.split(/[\\/]/).pop());
checkTrue(
  'average frequency switches to kHz above 1000 Hz',
  /^\d+\.\d{2}kHz$/.test(String(high.avgfreq ?? '').trim()),
  high.avgfreq,
);
checkTrue(
  'the kHz value still tracks the 2 kHz tone',
  Math.abs(readHertz(high.avgfreq) - 2000) <= 150,
  high.avgfreq,
);

// A large DC offset used to win the peak search outright and show "0 Hz", and
// it dragged the centroid towards zero too. Both readouts now start at bin 1.
await send('DOM.setFileInputFiles', { nodeId, files: [offsetTone] });
await wait(1600);
const offset = await state();
console.log('DC OFFSET', JSON.stringify({ freq: offset.freq, avgfreq: offset.avgfreq }));
checkTrue(
  'a DC offset is never reported as the peak frequency',
  readHertz(offset.freq) > 0,
  offset.freq,
);
checkTrue(
  'the peak frequency still finds the 1 kHz tone under the offset',
  Math.abs(readHertz(offset.freq) - 1000) <= 80,
  offset.freq,
);
checkTrue(
  'the centroid is not dragged towards DC',
  Math.abs(readHertz(offset.avgfreq) - 1000) <= 200,
  offset.avgfreq,
);

check('no runtime errors', errors, []);
await send('Emulation.clearDeviceMetricsOverride');
ws.close();
console.log(failed === 0 ? '\nFILE CHECK PASSED' : `\n${failed} FILE CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
