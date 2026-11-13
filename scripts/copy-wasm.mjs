// Copies the compiled kernel next to the app sources so Vite can emit it as an
// asset. Keeping the copy step explicit means `vite build` never needs cargo.
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', 'wasm', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel.wasm');
const targetDir = join(here, '..', 'src', 'generated');
const target = join(targetDir, 'reactor_kernel.wasm');

await mkdir(targetDir, { recursive: true });
await copyFile(source, target);

const { size } = await stat(target);
console.log(`copied reactor_kernel.wasm (${(size / 1024).toFixed(1)} kB) -> src/generated/`);
