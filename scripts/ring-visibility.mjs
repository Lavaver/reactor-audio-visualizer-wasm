// How visible are the rings in a dumped height field?
//
//   node scripts/ring-visibility.mjs .fixtures/heights-before.json [more.json ...]
//   node scripts/ring-visibility.mjs --assert .fixtures/heights-v3.json
//
// With `--assert` the two bounds below are checked and the exit code is 1 on failure, so
// this is the contract that keeps the rings from creeping back in.
//
// Four numbers, each aimed at one of the cues the visual system uses to group the rods
// into a circle:
//
//   equal   mean |Δheight| between neighbours *along* one ring, as a fraction of that
//           ring's height. Equal heights along a closed curve *is* the ring percept.
//   radial  the same between neighbours across rings. A hoop shows up as radial >> along.
//   ratio   spread inside a ring divided by spread between ring means. Near 1 means a
//           ring can no longer be told from its neighbours; near 0 means hoops.
//   steps   distinct heights over the 709 rods. Few values = the field is a staircase.
import { readFile } from 'node:fs/promises';

/** Neighbours along a ring must differ by at least this fraction of the ring's height. */
const MIN_ALONG = 0.03;
/** A radial step may not stand out from an along-ring step by more than this factor. */
const MAX_RADIAL_RATIO = 2.0;
/** Spread inside a ring, over spread between ring means. */
const MAX_WITHIN_RATIO = 0.8;

const argv = process.argv.slice(2);
const asserting = argv.includes('--assert');
const files = argv.filter(arg => arg !== '--assert');
if (files.length === 0) {
  console.error('usage: node scripts/ring-visibility.mjs [--assert] <dump.json> ...');
  process.exit(2);
}

const angleOf = rod => Math.atan2(rod.z, rod.x);

function measure(dump) {
  const rods = dump.rods;
  const byRing = new Map();
  for (const rod of rods) {
    const key = Math.round(rod.channel);
    if (!byRing.has(key)) byRing.set(key, []);
    byRing.get(key).push(rod);
  }
  for (const ring of byRing.values()) ring.sort((a, b) => angleOf(a) - angleOf(b));

  // Spread inside each ring, and the spread of ring means: the two terms of `ratio`.
  let withinSum = 0;
  const means = [];
  for (const ring of byRing.values()) {
    const heights = ring.map(rod => rod.height);
    withinSum += Math.max(...heights) - Math.min(...heights);
    means.push(heights.reduce((a, b) => a + b, 0) / heights.length);
  }
  const within = withinSum / byRing.size;
  const between = Math.max(...means) - Math.min(...means);

  // Neighbour differences: consecutive rods along a ring, and the nearest rod in the next
  // ring out along the same bearing.
  let alongSum = 0;
  let alongCount = 0;
  for (const ring of byRing.values()) {
    const scale = Math.max(...ring.map(rod => rod.height));
    for (let i = 0; i < ring.length; i++) {
      const next = ring[(i + 1) % ring.length];
      alongSum += Math.abs(ring[i].height - next.height) / Math.max(scale, 1e-6);
      alongCount++;
    }
  }
  const along = alongSum / Math.max(alongCount, 1);

  let radialSum = 0;
  let radialCount = 0;
  const rings = [...byRing.entries()].sort((a, b) => a[0] - b[0]);
  for (let r = 0; r + 1 < rings.length; r++) {
    const inner = rings[r][1];
    const outer = rings[r + 1][1];
    for (const rod of outer) {
      const bearing = angleOf(rod);
      let best = inner[0];
      let bestDelta = Infinity;
      for (const candidate of inner) {
        let delta = Math.abs(angleOf(candidate) - bearing);
        if (delta > Math.PI) delta = 2 * Math.PI - delta;
        if (delta < bestDelta) {
          bestDelta = delta;
          best = candidate;
        }
      }
      const scale = Math.max(best.height, rod.height, 1e-6);
      radialSum += Math.abs(best.height - rod.height) / scale;
      radialCount++;
    }
  }
  const radial = radialSum / Math.max(radialCount, 1);

  const steps = new Set(rods.map(rod => rod.height.toFixed(2))).size;
  return { rings: byRing.size, within, between, ratio: within / between, along, radial, steps };
}

console.log('file'.padEnd(34), 'rings', 'along', 'radial', 'rad/along', 'within', 'between', 'ratio', 'steps');
let failures = 0;
const verdict = (ok, name, value, bound) => {
  if (ok) return;
  failures++;
  console.log(`  FAIL ${name}: ${value} (bound ${bound})`);
};
for (const file of files) {
  const dump = JSON.parse(await readFile(file, 'utf8'));
  const m = measure(dump);
  const row = [
    file.padEnd(34),
    String(m.rings).padStart(5),
    m.along.toFixed(3).padStart(6),
    m.radial.toFixed(3).padStart(6),
    (m.radial / m.along).toFixed(2).padStart(9),
    m.within.toFixed(2).padStart(6),
    m.between.toFixed(2).padStart(7),
    m.ratio.toFixed(2).padStart(6),
    String(m.steps).padStart(5),
  ];
  console.log(row.join(' '));
  if (!asserting) continue;
  verdict(m.along >= MIN_ALONG, 'neighbours along a ring differ', m.along.toFixed(3), `>= ${MIN_ALONG}`);
  verdict(
    m.radial / m.along <= MAX_RADIAL_RATIO,
    'a radial step is not special',
    (m.radial / m.along).toFixed(2),
    `<= ${MAX_RADIAL_RATIO}`,
  );
  verdict(m.ratio <= MAX_WITHIN_RATIO, 'a ring is not told from its neighbours', m.ratio.toFixed(2), `<= ${MAX_WITHIN_RATIO}`);
}
if (asserting) {
  console.log(failures === 0 ? 'RINGS ARE NOT READABLE' : `${failures} RING VISIBILITY CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}
