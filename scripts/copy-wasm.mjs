// Copies the compiled kernels next to the app sources so Vite can emit them.
// Keeping the copy step explicit means `vite build` never needs cargo.
//
// v1 is required: without it the page has nothing to analyse with. v2 is
// optional by design — `pnpm run build:wasm:v2` decides whether it exists, and a
// stale copy from an earlier build is removed when the source is missing, so the
// button can never offer a kernel that is not there.
import { copyFile, mkdir, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const targetDir = join(root, 'src', 'generated');
const publicDir = join(root, 'public');

const KERNELS = [
  {
    name: 'reactor_kernel.wasm',
    // v1 is imported by the app as a hashed asset.
    source: join(root, 'wasm', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel.wasm'),
    target: join(targetDir, 'reactor_kernel.wasm'),
    required: true,
  },
  {
    name: 'reactor_kernel_v2.wasm',
    // v2 must be optional, so it is served verbatim from `public/` and probed at
    // runtime instead of being imported (a missing import would fail the build).
    source: join(root, 'wasm_v2', 'target', 'wasm32-unknown-unknown', 'release', 'reactor_kernel_v2.wasm'),
    target: join(publicDir, 'reactor_kernel_v2.wasm'),
    required: false,
  },
];

await mkdir(targetDir, { recursive: true });
await mkdir(publicDir, { recursive: true });

let missing = 0;
for (const kernel of KERNELS) {
  try {
    await copyFile(kernel.source, kernel.target);
    const { size } = await stat(kernel.target);
    console.log(`copied ${kernel.name} (${(size / 1024).toFixed(1)} kB) -> ${kernel.target.replace(root, '.')}`);
  } catch (error) {
    if (kernel.required) throw error;
    // Remove whatever is left over, so "the file exists" keeps meaning "it was
    // built": the frontend probe trusts exactly that.
    await rm(kernel.target, { force: true });
    console.log(`skipped ${kernel.name}: not built (run: pnpm run build:wasm:v2)`);
    missing++;
  }
}

if (missing > 0) {
  console.log('\nv2 内核未编译，前端 v2 按钮将保持禁用并继续使用 v1。');
}
