// Black-box browser check against the built bundle.
//
// Usage: start a static server for `app/dist`, launch a Chromium or Edge with
// --remote-debugging-port=9345 --autoplay-policy=no-user-gesture-required, then:
//   node app/scripts/browser-check.mjs [port]
//
// It asserts user-visible behaviour only: DOM readouts, rendered pixels, and the
// absence of console errors. The kernel maths is covered by kernel-check.mjs.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const shotDir = join(here, '..', '.screenshots');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const port = process.argv[2] ?? '9345';
const pageUrl = process.argv[3] ?? 'http://127.0.0.1:4174/';

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

let targets;
for (let attempt = 0; attempt < 40; attempt++) {
  try {
    targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    break;
  } catch {
    await wait(300);
  }
}
if (!targets) throw new Error(`no CDP endpoint on ${port}`);
const target = targets.find(entry => entry.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
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
    const url = details.url ?? '';
    if (!url || url.includes('127.0.0.1')) errors.push(details.exception?.description ?? details.text);
  }
  if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
    const entry = message.params.entry;
    const text = entry.text ?? '';
    if ((entry.url ?? '').includes('127.0.0.1') && !/favicon|404/i.test(text)) errors.push(text);
  }
};

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });

const evaluate = async (expression, userGesture = false) => {
  const result = await send('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
    userGesture,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  }
  return result.result.value;
};

const readState = () =>
  evaluate(`JSON.stringify({
    rods: document.getElementById('count')?.textContent,
    channelsLabel: document.getElementById('channelsLabel')?.textContent,
    channelsOut: document.getElementById('channelsval')?.textContent,
    channelsSlider: document.getElementById('channels')?.value,
    autoPressed: document.getElementById('autoch')?.getAttribute('aria-pressed'),
    gpu: document.getElementById('gpu')?.textContent,
    status: document.getElementById('status')?.textContent,
    level: document.getElementById('level')?.textContent,
    peaklevel: document.getElementById('peaklevel')?.textContent,
    freq: document.getElementById('freq')?.textContent,
    avgfreq: document.getElementById('avgfreq')?.textContent,
    cpu: document.getElementById('frametime')?.textContent,
    fps: document.getElementById('fps')?.textContent,
    p99: document.getElementById('p99')?.textContent,
    track: document.getElementById('trackname')?.textContent,
    play: document.getElementById('play')?.textContent,
    elapsed: document.getElementById('elapsed')?.textContent,
    analysis: document.getElementById('analysis')?.textContent,
    message: document.getElementById('message')?.textContent
  })`).then(JSON.parse);

// The core canvas is WebGL now, so pixels come out of the drawing buffer via
// readPixels. That only returns content because the app is loaded with
// `?preserve=1`, which asks for `preserveDrawingBuffer`. Rows come back
// bottom-up, unlike the old getImageData path.
const webglSample = () =>
  evaluate(`(() => {
    const canvas = document.getElementById('core');
    const gl = canvas.getContext('webgl2');
    if (!gl) return { error: 'no webgl2 context' };
    const width = canvas.width;
    const height = canvas.height;
    const data = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let lit = 0, alpha = 0, saturated = 0;
    for (let i = 0; i < data.length; i += 16) {
      if (data[i + 3] > 8) lit++;
      alpha += data[i + 3];
      const max = Math.max(data[i], data[i + 1], data[i + 2]);
      const min = Math.min(data[i], data[i + 1], data[i + 2]);
      if (max > 40 && max - min > 55) saturated++;
    }
    return { width, height, lit, alpha, saturated };
  })()`);

const canvasPixels = async () => {
  const sample = await webglSample();
  if (sample.error) throw new Error(sample.error);
  return { width: sample.width, height: sample.height, lit: sample.lit, alpha: sample.alpha };
};

const colorfulness = async () => {
  const sample = await webglSample();
  if (sample.error) throw new Error(sample.error);
  return sample.saturated;
};

/**
 * Circular mean of the strongly saturated pixels inside two disc-space annuli:
 * the rod field and the halo rim. Both come from one frame, so the wipe cannot
 * advance between the two measurements.
 */
const wipeBearings = () =>
  evaluate(`(() => {
  const canvas = document.getElementById('core');
  const gl = canvas.getContext('webgl2');
  if (!gl) return { error: 'no webgl2 context' };
  const width = canvas.width;
  const height = canvas.height;
  const data = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);

  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  const dpr = width / cw;
  const scale = Math.min(cw / 45, ch / 37);
  const cx = cw / 2;
  const cy = ch * 0.55;
  const tilt = 0.58; // the reset view; the drag above is horizontal only

  const meanAngle = (inner, outer) => {
    let sx = 0, sy = 0, count = 0;
    for (let yUp = 0; yUp < height; yUp++) {
      const dy = ((height - 1 - yUp + 0.5) / dpr - cy) / (scale * tilt);
      if (Math.abs(dy) > outer * 1.2) continue;
      for (let x = 0; x < width; x++) {
        const dx = ((x + 0.5) / dpr - cx) / scale;
        const radius = Math.sqrt(dx * dx + dy * dy);
        if (radius < inner || radius > outer) continue;
        const i = (yUp * width + x) * 4;
        const max = Math.max(data[i], data[i + 1], data[i + 2]);
        const min = Math.min(data[i], data[i + 1], data[i + 2]);
        if (max < 40 || max - min < 55) continue;
        const angle = Math.atan2(dy, dx);
        sx += Math.cos(angle);
        sy += Math.sin(angle);
        count++;
      }
    }
    if (!count) return null;
    return { degrees: Math.round((Math.atan2(sy, sx) * 180) / Math.PI), count };
  };

  return { rods: meanAngle(8, 14), rim: meanAngle(17.3, 17.85) };
})()`);

/** Shortest signed distance between two bearings, in degrees. */
const bearingGap = (a, b) => Math.abs((((a - b) % 360) + 540) % 360 - 180);

/** `?preserve=1` is what makes the drawing buffer readable from outside a frame. */
const withPreserve = url => `${url}${url.includes('?') ? '&' : '?'}preserve=1`;
const liveUrl = withPreserve(pageUrl);

await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440,
  height: 1000,
  deviceScaleFactor: 1,
  mobile: false,
});
await send('Page.navigate', { url: liveUrl });
await wait(2000);

// --- first paint -----------------------------------------------------------
const boot = await readState();
console.log('BOOT', JSON.stringify(boot));
check('709 rods rendered', boot.rods, '709');
checkTrue('gpu name detected', Boolean(boot.gpu) && boot.gpu !== '检测中…', boot.gpu);
check('idle status', boot.status, '待机 · 演示预览');

// The custom variable font is part of the design; verify the browser really
// resolved it instead of silently falling back to Arial / PingFang.
const fontState = await evaluate(`(async () => {
  await document.fonts.ready;
  return {
    loaded: document.fonts.check('16px "MiSans VF"'),
    family: getComputedStyle(document.body).fontFamily
  };
})()`);
console.log('FONT', JSON.stringify(fontState));
checkTrue('MiSans VF is loaded', fontState.loaded, fontState.family);
checkTrue('body text resolves to MiSans', fontState.family.includes('MiSans VF'), fontState.family);

// The meter readouts and axis labels used to be `monospace`. They now share the
// MiSans family, so their digits must still keep one advance width — that is
// what `font-variant-numeric: tabular-nums` buys, and it is measured rather
// than read off the stylesheet.
const numerals = await evaluate(`(() => {
  const selectors = ['.meters strong', '.timeline>div', '.slider-title output', '.track strong', '.axis', '#analysis'];
  const read = selector => {
    const element = document.querySelector(selector);
    if (!element) return { selector, missing: true };
    const style = getComputedStyle(element);
    return { selector, family: style.fontFamily, numeric: style.fontVariantNumeric };
  };
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;font:22px "MiSans VF"';
  document.body.appendChild(probe);
  const width = (text, numeric) => {
    probe.style.fontVariantNumeric = numeric;
    probe.textContent = text;
    return probe.getBoundingClientRect().width;
  };
  const tabular = [width('1111111111', 'tabular-nums'), width('8888888888', 'tabular-nums')];
  const normal = [width('1111111111', 'normal'), width('8888888888', 'normal')];
  probe.remove();
  return { rules: selectors.map(read), tabular, normal };
})()`);
console.log('NUMERALS', JSON.stringify(numerals));
for (const rule of numerals.rules) {
  checkTrue(`${rule.selector} exists`, !rule.missing, rule.selector);
  checkTrue(`${rule.selector} uses MiSans`, (rule.family ?? '').includes('MiSans VF'), rule.family);
  checkTrue(`${rule.selector} asks for tabular figures`, (rule.numeric ?? '').includes('tabular-nums'), rule.numeric);
}
checkTrue(
  'every digit keeps one advance width under tnum',
  Math.abs(numerals.tabular[0] - numerals.tabular[1]) < 0.01,
  `${numerals.tabular[0]} vs ${numerals.tabular[1]} px (normal: ${numerals.normal[0]} / ${numerals.normal[1]})`,
);

const idle = await canvasPixels();
checkTrue('canvas is painted', idle.lit > 100, `${idle.lit} sampled pixels`);
await mkdir(shotDir, { recursive: true });
let shot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
await writeFile(join(shotDir, 'react-idle.png'), Buffer.from(shot.data, 'base64'));

// --- stage toggles: auto-rotate + RGB --------------------------------------
// Rotation is measured on the rim ticks only. Rods reach 15.62 rod units at
// their widest (centre <= 15 plus the +-0.44 corner), so sampling the annulus
// between 15.75 and 16.13 sees nothing but the 80 ticks: their sample changes
// if and only if the view turned, whatever the rod heights are doing.
const rimSignature = () =>
  evaluate(`(() => {
  const canvas = document.getElementById('core');
  const gl = canvas.getContext('webgl2');
  if (!gl) return { error: 'no webgl2 context' };
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  const dpr = canvas.width / cw;
  const scale = Math.min(cw / 45, ch / 37);
  const cx = cw / 2;
  const cy = ch * 0.55;
  const tilt = 0.58; // the view the engine starts from (zoom 1)
  const rx = 16.2 * scale;
  const ry = 16.2 * scale * tilt;
  const x0 = Math.max(0, Math.floor((cx - rx) * dpr));
  const y0 = Math.max(0, Math.floor((cy - ry) * dpr));
  const w = Math.min(canvas.width - x0, Math.ceil(2 * rx * dpr));
  const h = Math.min(canvas.height - y0, Math.ceil(2 * ry * dpr));
  // readPixels hands back rows bottom-up, so the sample rect is mirrored in y.
  const buffer = new Uint8Array(canvas.width * canvas.height * 4);
  gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
  let signature = 0;
  let hits = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = ((x0 + x + 0.5) / dpr - cx) / scale;
      const dy = ((y0 + y + 0.5) / dpr - cy) / (scale * tilt);
      // Front half only: a raised rod top shifts towards the centre here, while
      // behind the centre it shifts outward and would leak into the annulus.
      if (dy < 1) continue;
      const radius = Math.sqrt(dx * dx + dy * dy);
      if (radius < 15.75 || radius > 16.13) continue;
      const i = ((canvas.height - 1 - (y0 + y)) * canvas.width + (x0 + x)) * 4;
      signature = (signature * 31 + buffer[i] * 3 + buffer[i + 1] * 5 + buffer[i + 2] * 7 + buffer[i + 3]) % 1000000007;
      hits++;
    }
  }
  return { signature, hits };
})()`);

const pressed = id => evaluate(`document.getElementById('${id}').getAttribute('aria-pressed')`);
check('auto-rotate defaults on', await pressed('autorotate'), 'true');
check('rgb defaults off', await pressed('rgb'), 'false');

// A switch nobody can see is not a switch: the stage keeps its own height, so
// the bottom bar can drift below the fold as the panel grows.
const toggleSpot = await evaluate(`(() => {
  const rect = document.getElementById('autorotate').getBoundingClientRect();
  return {
    top: Math.round(rect.top),
    bottom: Math.round(rect.bottom),
    viewport: window.innerHeight,
    scrollY: Math.round(window.scrollY)
  };
})()`);
checkTrue(
  'stage switches sit inside the first screen',
  toggleSpot.bottom <= toggleSpot.viewport,
  JSON.stringify(toggleSpot),
);

const rimA = await rimSignature();
await wait(900);
const rimB = await rimSignature();
checkTrue('rim ticks are sampled', rimA.hits > 50, `${rimA.hits} pixels`);
checkTrue(
  'auto-rotate turns the core',
  rimA.signature !== rimB.signature,
  `signature ${rimA.signature} -> ${rimB.signature}`,
);

await evaluate(`document.getElementById('autorotate').click()`, true);
await wait(250);
check('auto-rotate released', await pressed('autorotate'), 'false');
const rimC = await rimSignature();
await wait(900);
const rimD = await rimSignature();
checkTrue(
  'a still core holds still',
  rimC.signature === rimD.signature,
  `signature ${rimC.signature} -> ${rimD.signature}`,
);
await evaluate(`document.getElementById('autorotate').click()`, true);
check('auto-rotate back on', await pressed('autorotate'), 'true');

// RGB is verified by counting strongly saturated pixels, which the muted olive
// palette of the default look does not produce.
const mutedPixels = await colorfulness();
// Turn the view well away from its starting angle first. The wipe front is
// supposed to be anchored to the disc, and a front that is anchored to the
// screen instead only becomes obvious once the reactor is rotated.
const turned = await evaluate(`(() => {
  const canvas = document.getElementById('core');
  const rect = canvas.getBoundingClientRect();
  const fire = (type, x, y) => canvas.dispatchEvent(new PointerEvent(type, {
    clientX: x, clientY: y, pointerId: 1, bubbles: true, pointerType: 'mouse'
  }));
  fire('pointerdown', rect.left + 200, rect.top + 300);
  fire('pointermove', rect.left + 560, rect.top + 300);
  fire('pointerup', rect.left + 560, rect.top + 300);
  return 'turned';
})()`, true);
check('view turned before the wipe check', turned, 'turned');

await evaluate(`document.getElementById('rgb').click()`, true);
check('rgb engaged', await pressed('rgb'), 'true');
// The palette is supposed to arrive as a clockwise wipe, not as a flip. Sample
// twice mid-flight: both must be between "off" and "fully on", and the second
// must be further along than the first.
await wait(250);
const wipeA = await colorfulness();
const bearings = await wipeBearings();
await wait(250);
const wipeB = await colorfulness();
await wait(950);
const litPixels = await colorfulness();
console.log('WIPE IN', JSON.stringify({ mutedPixels, wipeA, wipeB, litPixels }));
console.log('WIPE BEARINGS', JSON.stringify(bearings));
checkTrue(
  'the wipe front stays anchored to the disc',
  bearings.rods !== null && bearings.rim !== null
    && bearingGap(bearings.rods.degrees, bearings.rim.degrees) <= 15,
  `${bearings.rods?.degrees}° on the rods vs ${bearings.rim?.degrees}° on the rim`,
);
checkTrue(
  'RGB floods the stage with saturated light',
  litPixels > mutedPixels * 2 + 200,
  `${mutedPixels} -> ${litPixels} saturated samples`,
);
checkTrue(
  'RGB wipes in instead of snapping on',
  wipeA > mutedPixels && wipeA < litPixels * 0.7,
  `quarter way: ${wipeA} of ${litPixels}`,
);
checkTrue('the wipe front advances', wipeB > wipeA, `${wipeA} -> ${wipeB}`);

await evaluate(`document.getElementById('rgb').click()`, true);
check('rgb released', await pressed('rgb'), 'false');
await wait(250);
const clearA = await colorfulness();
await wait(1000);
const clearedPixels = await colorfulness();
console.log('WIPE OUT', JSON.stringify({ clearA, clearedPixels }));
checkTrue(
  'turning it off sweeps too, rather than snapping back',
  clearA > litPixels * 0.5,
  `${clearA} of ${litPixels} still lit a quarter of the way through`,
);
checkTrue(
  'the swept palette is gone when the front finishes',
  clearedPixels < litPixels * 0.35,
  `${clearedPixels} of ${litPixels}`,
);

// Leave the stage lit for the playback screenshot below.
await evaluate(`document.getElementById('rgb').click()`, true);
await wait(1300);

// --- playback --------------------------------------------------------------
await evaluate(`document.getElementById('play').click()`, true);
await wait(1600);
const playing = await readState();
console.log('PLAYING', JSON.stringify(playing));
check('playing status', playing.status, '运行中 · 演示信号');
check('play icon switched to pause', playing.play, 'Ⅱ');
checkTrue('analysis runs on the audio thread', (playing.analysis ?? '').includes('AUDIO THREAD'), playing.analysis);
checkTrue('level meter shows dB', /-?\d+dB/.test(playing.level ?? ''), playing.level);
checkTrue('peak frequency shows Hz', /\d+Hz/.test(playing.freq ?? ''), playing.freq);
checkTrue('elapsed clock advanced', playing.elapsed !== '00:00', playing.elapsed);

// --- new readouts: peak hold + spectral centroid ---------------------------
const asNumber = text => Number(String(text ?? '').replace(/[^\d-]/g, ''));
/** "656Hz" -> 656, "1.35kHz" -> 1350, anything else -> NaN. */
const readHertz = text => {
  const match = /^([\d.]+)(kHz|Hz)$/.exec(String(text ?? '').trim());
  if (!match) return Number.NaN;
  return Number(match[1]) * (match[2] === 'kHz' ? 1000 : 1);
};
console.log('READOUTS', JSON.stringify({
  level: playing.level,
  peaklevel: playing.peaklevel,
  freq: playing.freq,
  avgfreq: playing.avgfreq,
}));
checkTrue('peak level shows dB', /-?\d+dB/.test(playing.peaklevel ?? ''), playing.peaklevel);
// The unit is chosen from the magnitude: two decimals of kHz at 1 kHz and up,
// whole Hz below. Asserted as an invariant so it holds whatever the demo does.
const averageText = String(playing.avgfreq ?? '').trim();
const averageHz = readHertz(averageText);
checkTrue(
  'average frequency picks its unit from its magnitude',
  averageHz >= 1000 ? /^\d+\.\d{2}kHz$/.test(averageText) : /^\d+Hz$/.test(averageText),
  averageText,
);
checkTrue(
  'peak hold is at least the current level',
  asNumber(playing.peaklevel) >= asNumber(playing.level),
  `${playing.peaklevel} vs ${playing.level}`,
);
// The demo signal is eight tones between 55 Hz and 659 Hz, so the centroid has
// to sit inside that range rather than at either extreme.
checkTrue(
  'average frequency sits inside the demo band',
  averageHz >= 40 && averageHz <= 2000,
  `${averageText} (${averageHz} Hz)`,
);

// The mini spectrum is a 2D canvas that now lives inside the readouts grid. A
// half-width cell leaves 136px for 48 bars, and the old fixed 3px gap turned
// every bar into a hairline there; keep an eye on the room each bar gets.
const spectrum = await evaluate(`(() => {
  const canvas = document.getElementById('spectrum');
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return { width, height, slot: 0, ink: 0 };
  const data = canvas.getContext('2d').getImageData(0, 0, width, height).data;
  let ink = 0;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      if (data[(y * width + x) * 4 + 3] > 8) { ink++; break; }
    }
  }
  return { width, height, slot: Number((width / 48).toFixed(2)), ink };
})()`);
console.log('SPECTRUM', JSON.stringify(spectrum));
checkTrue(
  'mini spectrum has room for 48 bars',
  spectrum.slot >= 4,
  `${spectrum.width}px / 48 = ${spectrum.slot}px per bar`,
);
checkTrue('mini spectrum draws bars', spectrum.ink > 30, `${spectrum.ink} columns with ink`);

// --- performance readouts ---------------------------------------------------
const perfNumber = text => {
  const match = /^-?\d+(?:\.\d+)?/.exec(String(text ?? '').trim());
  return match ? Number(match[0]) : Number.NaN;
};
const perf = {
  frameMs: perfNumber(playing.cpu),
  fps: perfNumber(playing.fps),
  p99: perfNumber(playing.p99),
};
console.log('PERF', JSON.stringify({ frameMs: playing.cpu, fps: playing.fps, p99: playing.p99 }));
checkTrue(
  'main-thread cost reads in milliseconds',
  /^\d+\.\dms$/.test(String(playing.cpu ?? '').trim()),
  playing.cpu,
);
checkTrue(
  'main-thread cost is under a frame budget',
  perf.frameMs >= 0 && perf.frameMs < 50,
  playing.cpu,
);
checkTrue('presented FPS is plausible', perf.fps >= 20 && perf.fps <= 240, playing.fps);
// The 99th percentile frame time is the slow end, so its FPS can only be the
// same as or below the average.
checkTrue('1% low never exceeds the average FPS', perf.p99 <= perf.fps, `${playing.p99} vs ${playing.fps}`);

// --- renderer switch --------------------------------------------------------
const backendState = () => evaluate(`JSON.stringify({
  label: document.getElementById('renderer')?.textContent,
  pressed: ['canvas2d', 'webgl', 'webgpu'].map(k => document.getElementById('backend-' + k)?.getAttribute('aria-pressed')),
  disabled: ['canvas2d', 'webgl', 'webgpu'].map(k => document.getElementById('backend-' + k)?.disabled),
  titles: ['canvas2d', 'webgl', 'webgpu'].map(k => document.getElementById('backend-' + k)?.title),
  note: document.getElementById('rendernote')?.textContent ?? null
})`).then(JSON.parse);

/** Painted-pixel count from whichever context the current backend owns. */
const coreInk = () => evaluate(`(() => {
  const canvas = document.getElementById('core');
  const count = data => {
    let lit = 0;
    for (let i = 3; i < data.length; i += 16) if (data[i] > 8) lit++;
    return lit;
  };
  const gl = canvas.getContext('webgl2');
  if (gl) {
    const data = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return { api: 'webgl2', lit: count(data) };
  }
  const ctx = canvas.getContext('2d');
  if (ctx) return { api: '2d', lit: count(ctx.getImageData(0, 0, canvas.width, canvas.height).data) };
  return { api: 'webgpu', lit: -1 };
})()`);

const initialBackend = await backendState();
console.log('BACKENDS', JSON.stringify(initialBackend));
check('default backend is WebGL', initialBackend.label, 'WebGL');
check('only WebGL starts selected', initialBackend.pressed, ['false', 'true', 'false']);
// Each switch has to explain how that backend draws, who can run it and what it
// costs; three identical tooltips would carry no information.
checkTrue(
  'every backend has its own tooltip',
  initialBackend.titles.every(title => typeof title === 'string' && title.length > 20)
    && new Set(initialBackend.titles).size === 3,
  JSON.stringify(initialBackend.titles),
);

await evaluate(`document.getElementById('backend-canvas2d').click()`, true);
await wait(600);
const legacy = await backendState();
const legacyPixels = await coreInk();
check('switching to Canvas 2D updates the label', legacy.label, '传统');
check('Canvas 2D is selected', legacy.pressed, ['true', 'false', 'false']);
checkTrue('Canvas 2D backend paints', legacyPixels.api === '2d' && legacyPixels.lit > 100,
  `${legacyPixels.api}, ${legacyPixels.lit} sampled pixels`);

await evaluate(`document.getElementById('backend-webgl').click()`, true);
await wait(600);
const back = await backendState();
const backPixels = await coreInk();
check('switching back to WebGL updates the label', back.label, 'WebGL');
checkTrue('WebGL backend paints again', backPixels.api === 'webgl2' && backPixels.lit > 100,
  `${backPixels.api}, ${backPixels.lit} sampled pixels`);

// WebGPU depends on the machine: either the button is offered and works, or it
// is disabled so nobody can select a backend that cannot run.
const webgpuEnabled = initialBackend.disabled[2] === false;
if (webgpuEnabled) {
  await evaluate(`document.getElementById('backend-webgpu').click()`, true);
  await wait(900);
  const gpu = await backendState();
  console.log('WEBGPU', JSON.stringify(gpu));
  check('WebGPU can be selected when offered', gpu.label, 'WebGPU');
  checkTrue('WebGPU draws without a fallback note', gpu.note === null, gpu.note ?? 'no note');
  await evaluate(`document.getElementById('backend-webgl').click()`, true);
  await wait(600);
} else {
  check('WebGPU is disabled when unavailable', initialBackend.disabled, [false, false, true]);
}

// Screenshot bytes depend on whether the compositor decided to repaint, so the
// "is it animating" assertion samples the canvas itself instead.
const frameA = await canvasPixels();
await wait(420);
const frameB = await canvasPixels();
checkTrue(
  'animation is running (canvas changes)',
  frameA.alpha !== frameB.alpha || frameA.lit !== frameB.lit,
  `alpha ${frameA.alpha} -> ${frameB.alpha}, lit ${frameA.lit} -> ${frameB.lit}`,
);

shot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
await mkdir(shotDir, { recursive: true });
await writeFile(join(shotDir, 'react-playing.png'), Buffer.from(shot.data, 'base64'));

// --- adaptive channels -----------------------------------------------------
await evaluate(`document.getElementById('autoch').click()`, true);
await wait(400);
const auto = await readState();
console.log('AUTO', JSON.stringify(auto));
check('auto pressed', auto.autoPressed, 'true');
checkTrue('auto label prefix', (auto.channelsOut ?? '').startsWith('AUTO · '), auto.channelsOut);
checkTrue('auto slider follows channels', auto.channelsSlider === auto.channelsLabel, `${auto.channelsSlider} / ${auto.channelsLabel}`);

// Dragging the slider must drop back to manual mode.
await evaluate(`(() => {
  const slider = document.getElementById('channels');
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  setter.call(slider, '12');
  slider.dispatchEvent(new Event('input', { bubbles: true }));
  return 'ok';
})()`, true);
await wait(300);
const manual = await readState();
console.log('MANUAL', JSON.stringify(manual));
check('auto released by slider', manual.autoPressed, 'false');
check('manual channel count', manual.channelsLabel, '12');
check('manual output label', manual.channelsOut, '12 CH');

// --- interaction -----------------------------------------------------------
const drag = await evaluate(`(() => {
  const canvas = document.getElementById('core');
  const rect = canvas.getBoundingClientRect();
  const fire = (type, x, y) => canvas.dispatchEvent(new PointerEvent(type, {
    clientX: x, clientY: y, pointerId: 1, bubbles: true, pointerType: 'mouse'
  }));
  fire('pointerdown', rect.left + 200, rect.top + 200);
  fire('pointermove', rect.left + 320, rect.top + 240);
  fire('pointerup', rect.left + 320, rect.top + 240);
  return 'dragged';
})()`, true);
check('pointer drag accepted', drag, 'dragged');
await wait(300);
const afterDrag = await canvasPixels();
checkTrue('canvas still painted after drag', afterDrag.lit > 100, `${afterDrag.lit} sampled pixels`);

await evaluate(`document.getElementById('reset').click()`, true);
await evaluate(`document.getElementById('play').click()`, true);
await wait(300);
const paused = await readState();
// Pausing the demo lands on the same "preview" label the original page used.
check('pause returns to idle status', paused.status, '待机 · 演示预览');
check('play icon back to triangle', paused.play, '▶');
check(
  'all four readouts clear on pause',
  [paused.level, paused.peaklevel, paused.freq, paused.avgfreq],
  ['—dB', '—dB', '—Hz', '—Hz'],
);

// Mobile layout keeps the same panel and canvas, so a screenshot is enough.
await send('Emulation.setDeviceMetricsOverride', {
  width: 390,
  height: 844,
  deviceScaleFactor: 1,
  mobile: true,
  screenWidth: 390,
  screenHeight: 844,
});
await wait(600);
// The switches drop to the stage's bottom edge here, next to the drag hint.
shot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
await writeFile(join(shotDir, 'react-mobile-stage.png'), Buffer.from(shot.data, 'base64'));
const mobileToggles = await evaluate(`(() => {
  const switches = document.querySelector('.stage-toggles').getBoundingClientRect();
  const hint = document.querySelector('.stage-bottom span').getBoundingClientRect();
  return {
    overlap: switches.left < hint.right && hint.left < switches.right
      && switches.top < hint.bottom && hint.top < switches.bottom,
    switches: [Math.round(switches.left), Math.round(switches.top)],
    hintRight: Math.round(hint.right),
    onScreen: switches.bottom <= window.innerHeight + window.scrollY
  };
})()`);
checkTrue('stage switches clear the drag hint on mobile', !mobileToggles.overlap, JSON.stringify(mobileToggles));
checkTrue('stage switches stay on screen on mobile', mobileToggles.onScreen, JSON.stringify(mobileToggles));
// Scroll the readouts into view so the mobile artefact shows the four meters,
// not just the stage.
await evaluate(`document.getElementById('level').closest('.readings')?.scrollIntoView({ block: 'center' })`);
await wait(300);
shot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
await writeFile(join(shotDir, 'react-mobile.png'), Buffer.from(shot.data, 'base64'));
const overflow = await evaluate(`JSON.stringify({
  scrollW: document.documentElement.scrollWidth,
  innerW: window.innerWidth
})`).then(JSON.parse);
checkTrue('no horizontal overflow on mobile', overflow.scrollW <= overflow.innerW + 1, JSON.stringify(overflow));

// --- main-thread fallback --------------------------------------------------
// `?noworklet=1` disables the audio-thread analyser; the page must still work
// through the wasm-on-main-thread path.
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
const fallbackUrl = `${liveUrl}&noworklet=1`;
await send('Page.navigate', { url: fallbackUrl });
await wait(1800);
const fallbackBoot = await readState();
checkTrue('fallback boots with the main-thread label', (fallbackBoot.analysis ?? '').includes('MAIN THREAD'), fallbackBoot.analysis);
await evaluate(`document.getElementById('play').click()`, true);
await wait(1500);
const fallback = await readState();
console.log('FALLBACK', JSON.stringify(fallback));
check('fallback plays', fallback.status, '运行中 · 演示信号');
checkTrue('fallback still measures level', /-?\d+dB/.test(fallback.level ?? ''), fallback.level);
checkTrue('fallback still reports a peak', /\d+Hz/.test(fallback.freq ?? ''), fallback.freq);
const fallbackFrameA = await canvasPixels();
await wait(420);
const fallbackFrameB = await canvasPixels();
checkTrue('fallback animates', fallbackFrameA.alpha !== fallbackFrameB.alpha,
  `alpha ${fallbackFrameA.alpha} -> ${fallbackFrameB.alpha}`);

check('no runtime errors', errors, []);
await send('Emulation.clearDeviceMetricsOverride');
ws.close();

console.log(failed === 0 ? '\nALL BROWSER CHECKS PASSED' : `\n${failed} BROWSER CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
