// WebGPU backend: the same scene, expressed as WGSL instead of GLSL.
//
// The maths is deliberately a transliteration of `gl.ts` so the two backends can
// be compared pixel for pixel: same oblique projection, same instanced boxes,
// same flowing palette and clockwise wipe. What differs is the plumbing —
// pipelines instead of programs, uniform buffers instead of loose uniforms, and
// an explicit multisampled render target because WebGPU has no `antialias: true`.

import type { Scene, View } from './draw';
import type { CoreRenderer, RodSource } from './renderer';

const TAU = Math.PI * 2;
const FLOW_RATE = 0.0009;
const ROD_BASE = 0.07;
const HEIGHT_SCALE = 4.8;
const ROD_HALF = 0.44;
const RING_SEGMENTS = 72;
const TICK_COUNT = 80;
const TICK_INNER = 15.7;
const TICK_OUTER = 16.15;
const TICK_SPAN = 0.034;
const HALO_RADIUS = 17.9;
const HALO_SEGMENTS = 96;
/** pos3, angle, radius, mode, baked3. */
const RIM_FLOATS = 9;
const BOX_FLOATS = 7;
const BOX_VERTICES = 30;
const MODE_FLOW = 1;
const MODE_BAKED = 2;
const SAMPLE_COUNT = 4;

interface RingSpec {
  radius: number;
  widthPx: number;
  hueOffset: number;
  lightness: number;
  saturation: number;
  plain: [number, number, number];
}

const RINGS: RingSpec[] = [
  { radius: 16.4, widthPx: 20, hueOffset: NaN, lightness: 0, saturation: 0, plain: [0.024, 0.043, 0.027] },
  { radius: 16.5, widthPx: 2, hueOffset: 0, lightness: 0.55, saturation: 0.95, plain: [0.314, 0.349, 0.282] },
  { radius: 16.0, widthPx: 1, hueOffset: 120, lightness: 0.62, saturation: 0.95, plain: [0.529, 0.537, 0.412] },
  { radius: 17.3, widthPx: 1, hueOffset: 240, lightness: 0.4, saturation: 0.95, plain: [0.188, 0.227, 0.176] },
];

/** Shared WGSL: uniforms, projection and the flowing palette. */
const COMMON = `
struct Uniforms {
  view: vec4f,     // resolution.xy, cos, sin
  shape: vec4f,    // scale, tilt, flow, sweep
  palette: vec4f,  // rgbOn, rodBase, heightScale, unused
};
@group(0) @binding(0) var<uniform> u: Uniforms;

const TAU: f32 = 6.28318530718;
const DEG: f32 = 57.29577951308232;
const FRONT_WIDTH: f32 = 0.045;

fn project(world: vec3f, rz: ptr<function, f32>) -> vec4f {
  let rotated = vec2f(world.x * u.view.z - world.z * u.view.w, world.x * u.view.w + world.z * u.view.z);
  let sx = u.view.x * 0.5 + rotated.x * u.shape.x;
  let sy = u.view.y * 0.55 + rotated.y * u.shape.x * u.shape.y - world.y * u.shape.x;
  *rz = rotated.y;
  // WebGPU clip space is z ∈ [0, w] (D3D/Metal convention), not OpenGL's
  // [-w, w]: a negative z is clipped away entirely. Map the rotated depth into
  // [0, 1] with larger z (nearer) staying smaller, so depthCompare 'less'
  // keeps the same ordering the WebGL backend gets from its [-1, 1] range.
  let depth = clamp(0.5 - rotated.y * 0.025, 0.0, 1.0);
  return vec4f(
    sx / (u.view.x * 0.5) - 1.0,
    1.0 - sy / (u.view.y * 0.5),
    depth,
    1.0);
}

fn hslToRgb(h: f32, s: f32, l: f32) -> vec3f {
  let hn = fract(h / 360.0);
  // WGSL has no float mod, and fract(x) * 6 is not mod(x, 6): wrap with
  // fract(x / 6) * 6 instead, which is what the GLSL version's mod() does.
  let wrapped = fract((hn * 6.0 + vec3f(0.0, 4.0, 2.0)) / 6.0) * 6.0;
  let rgb = clamp(abs(wrapped - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
  return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
}

fn sectorLit(sector: f32, on: f32, sweep: f32) -> bool {
  if (on > 0.5) { return sector <= sweep; }
  return sector > sweep;
}

fn frontStrength(sector: f32, sweep: f32) -> f32 {
  if (sweep >= 1.0) { return 0.0; }
  let d = abs(sector - sweep);
  let dist = min(d, 1.0 - d);
  return max(0.0, 1.0 - dist / FRONT_WIDTH);
}

fn flowHue(angle: f32, radius: f32, flow: f32) -> f32 {
  return (angle * 2.0 + radius * 0.21 - flow * 2.0) * DEG;
}

fn flowComet(angle: f32, flow: f32) -> f32 {
  return 0.5 + 0.5 * sin(angle * 3.0 - flow * 3.0);
}

fn plainColor(seed: f32, amp: f32) -> vec3f {
  var base = vec3f(139.0 + seed * 29.0, 148.0 + seed * 22.0, 119.0 + seed * 21.0);
  if (seed > 0.92) { base = vec3f(143.0, 109.0, 81.0); }
  else if (seed > 0.85) { base = vec3f(97.0, 118.0, 104.0); }
  return clamp((base + amp * vec3f(43.0, 40.0, 24.0)) / 255.0, vec3f(0.0), vec3f(1.0));
}
`;

const ROD_SHADER = `${COMMON}
struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) uv: vec2f,
  @location(2) top: f32,
};

@vertex fn vs(
  @location(0) corner: vec3f,
  @location(1) shade: f32,
  @location(2) top: f32,
  @location(3) uv: vec2f,
  @location(4) rod: vec2f,
  @location(5) seed: f32,
  @location(6) radius: f32,
  @location(7) height: f32
) -> VsOut {
  let amp = max(0.0, (height - u.palette.y) / u.palette.z);
  let world = vec3f(rod.x + corner.x, u.palette.y + corner.y * height, rod.y + corner.z);

  var rz: f32;
  var out: VsOut;
  out.position = project(world, &rz);

  // Same world-anchored wipe as the WebGL backend: the sector comes from the
  // rod's own angle, never from a vertex, so a box is never split in half.
  let rodAngle = atan2(rod.y, rod.x);
  let sector = fract(rodAngle / TAU);
  var color: vec3f;
  if (sectorLit(sector, u.palette.x, u.shape.w)) {
    let front = frontStrength(sector, u.shape.w);
    let lightness = (30.0 + flowComet(rodAngle, u.shape.z) * 34.0 + min(18.0, amp * 30.0) + front * 26.0) / 100.0;
    color = hslToRgb(flowHue(rodAngle, radius, u.shape.z), max(0.0, 0.95 - front * 0.62), lightness);
  } else {
    color = plainColor(seed, amp);
  }

  out.color = color * shade;
  out.uv = uv;
  out.top = top;
  return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  var color = in.color;
  if (in.top > 0.5) {
    if (max(abs(in.uv.x - 0.5), abs(in.uv.y - 0.5)) < 0.30) { color *= 0.80; }
  }
  return vec4f(color, 1.0);
}
`;

const RIM_SHADER = `${COMMON}
struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@vertex fn vs(
  @location(0) pos: vec3f,
  @location(1) angle: f32,
  @location(2) radius: f32,
  @location(3) mode: f32,
  @location(4) baked: vec3f
) -> VsOut {
  var rz: f32;
  var out: VsOut;
  out.position = project(pos, &rz);

  var color = baked;
  if (mode < 1.5) {
    let sector = fract(angle / TAU);
    if (sectorLit(sector, u.palette.x, u.shape.w)) {
      let front = frontStrength(sector, u.shape.w);
      let lightness = (22.0 + flowComet(angle, u.shape.z) * 48.0 + front * 26.0) / 100.0;
      let saturation = max(0.0, 0.95 - front * 0.55);
      color = hslToRgb(flowHue(angle, radius, u.shape.z), saturation, lightness);
    }
  }
  out.color = color;
  return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

const HALO_SHADER = `${COMMON}
struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) disc: vec2f,
};

@vertex fn vs(@location(0) pos: vec3f) -> VsOut {
  var rz: f32;
  var out: VsOut;
  out.position = project(pos, &rz);
  out.disc = pos.xz;
  return out;
}

// Was this fragment part of a triangle that should be drawn? The wipe is a
// per-fragment discard, exactly like the WebGL wedge.
@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  let radius = length(in.disc);
  let angle = atan2(in.disc.y, in.disc.x);
  if (!sectorLit(fract(angle / TAU), u.palette.x, u.shape.w)) { discard; }

  let energy = u.palette.w;
  let rainbow = hslToRgb(u.shape.z * DEG + angle * DEG + 90.0, 1.0, 0.58);
  let body = mix(0.10, 0.40, smoothstep(12.0, 17.9, radius)) * (0.35 + energy * 0.65);
  let squeeze = sqrt(cos(angle) * cos(angle) + sin(angle) * sin(angle) * u.shape.y * u.shape.y);
  let rimPx = abs(radius - 17.9) * u.shape.x * squeeze;
  let rim = (1.0 - smoothstep(0.0, 1.4, rimPx)) * (0.22 + energy * 0.6);

  var cometColor = vec3f(0.0);
  var cometAlpha = 0.0;
  for (var i = 0; i < 3; i = i + 1) {
    let cometAngle = u.shape.z + f32(i) * TAU / 3.0;
    let spot = vec2f(cos(cometAngle), sin(cometAngle)) * 16.6;
    let d = length(in.disc - spot) / 3.8;
    let glow = exp(-d * d * 1.8);
    cometColor += hslToRgb(u.shape.z * DEG + cometAngle * DEG, 1.0, 0.82) * glow;
    cometAlpha += glow;
  }
  let cometGain = 0.32 + energy * 0.55;
  cometColor *= cometGain;
  cometAlpha *= cometGain;

  let color = rainbow * (body + rim) + cometColor;
  return vec4f(color, min(1.0, body + rim + cometAlpha));
}
`;

function buildBox(): Float32Array {
  const corners: Array<[number, number]> = [
    [-ROD_HALF, -ROD_HALF],
    [ROD_HALF, -ROD_HALF],
    [ROD_HALF, ROD_HALF],
    [-ROD_HALF, ROD_HALF],
  ];
  const sides = [0.44, 0.38, 0.36, 0.4];
  const data: number[] = [];
  const push = (x: number, up: number, z: number, shade: number, top: number, u: number, v: number) =>
    data.push(x, up, z, shade, top, u, v);
  const uv: Array<[number, number]> = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const index of [0, 1, 2, 0, 2, 3]) {
    const [x, z] = corners[index];
    const [u, v] = uv[index];
    push(x, 1, z, 1, 1, u, v);
  }
  for (let i = 0; i < 4; i++) {
    const [ax, az] = corners[i];
    const [bx, bz] = corners[(i + 1) % 4];
    const shade = sides[i];
    push(ax, 0, az, shade, 0, 0, 0);
    push(bx, 0, bz, shade, 0, 0, 0);
    push(bx, 1, bz, shade, 0, 0, 0);
    push(ax, 0, az, shade, 0, 0, 0);
    push(bx, 1, bz, shade, 0, 0, 0);
    push(ax, 1, az, shade, 0, 0, 0);
  }
  return new Float32Array(data);
}

export class WebgpuRenderer implements CoreRenderer {
  readonly kind = 'webgpu' as const;

  static async create(canvas: HTMLCanvasElement, source: RodSource): Promise<WebgpuRenderer> {
    if (!('gpu' in navigator) || !navigator.gpu) throw new Error('这个浏览器没有 WebGPU。');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU 没有可用的适配器。');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('无法获取 WebGPU 画布上下文。');
    return new WebgpuRenderer(canvas, source, device, context);
  }

  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly source: RodSource;

  private readonly uniformBuffer: GPUBuffer;
  private readonly uniforms = new Float32Array(12);
  private readonly bindGroup: GPUBindGroup;

  private readonly rodPipeline: GPURenderPipeline;
  private readonly rimPipeline: GPURenderPipeline;
  private readonly haloPipeline: GPURenderPipeline;

  private readonly boxBuffer: GPUBuffer;
  private readonly instanceBuffer: GPUBuffer;
  private readonly heightBuffer: GPUBuffer;
  private readonly tickBuffer: GPUBuffer;
  private readonly ringBuffer: GPUBuffer;
  private readonly haloBuffer: GPUBuffer;
  private readonly ringData: Float32Array;
  private readonly tickVertices: number;
  private readonly haloVertices: number;

  private depth: GPUTexture | null = null;
  private msaa: GPUTexture | null = null;
  private width = 0;
  private height = 0;

  private constructor(
    canvas: HTMLCanvasElement,
    source: RodSource,
    device: GPUDevice,
    context: GPUCanvasContext,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.source = source;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.format, alphaMode: 'premultiplied' });

    this.uniformBuffer = device.createBuffer({
      size: this.uniforms.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const layout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.bindGroup = device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    const rodModule = device.createShaderModule({ code: ROD_SHADER });
    const rimModule = device.createShaderModule({ code: RIM_SHADER });
    const haloModule = device.createShaderModule({ code: HALO_SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const multisample = { count: SAMPLE_COUNT };
    const targets: GPUColorTargetState[] = [{ format: this.format }];

    this.rodPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: rodModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: BOX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32' },
              { shaderLocation: 2, offset: 16, format: 'float32' },
              { shaderLocation: 3, offset: 20, format: 'float32x2' },
            ],
          },
          {
            arrayStride: 16,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'float32x2' },
              { shaderLocation: 5, offset: 8, format: 'float32' },
              { shaderLocation: 6, offset: 12, format: 'float32' },
            ],
          },
          {
            arrayStride: 4,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 7, offset: 0, format: 'float32' }],
          },
        ],
      },
      fragment: { module: rodModule, entryPoint: 'fs', targets },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample,
    });

    this.rimPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: rimModule, entryPoint: 'vs', buffers: [this.rimLayout(RIM_FLOATS * 4)] },
      fragment: { module: rimModule, entryPoint: 'fs', targets },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      multisample,
    });

    this.haloPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: haloModule, entryPoint: 'vs', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      fragment: {
        module: haloModule,
        entryPoint: 'fs',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      multisample,
    });

    this.boxBuffer = this.staticBuffer(buildBox());
    const instance = new Float32Array(source.count * 4);
    for (let i = 0; i < source.count; i++) {
      const offset = i * source.stride;
      instance[i * 4] = source.rods[offset];
      instance[i * 4 + 1] = source.rods[offset + 1];
      instance[i * 4 + 2] = source.rods[offset + 3];
      instance[i * 4 + 3] = source.rods[offset + 2];
    }
    this.instanceBuffer = this.staticBuffer(instance);
    this.heightBuffer = device.createBuffer({
      size: source.count * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const ticks = this.buildTicks();
    this.tickVertices = ticks.length / RIM_FLOATS;
    this.tickBuffer = this.staticBuffer(ticks);
    this.ringData = new Float32Array(RINGS.length * RING_SEGMENTS * 6 * RIM_FLOATS);
    this.ringBuffer = device.createBuffer({
      size: this.ringData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const halo = new Float32Array((HALO_SEGMENTS + 2) * 3);
    for (let i = 0; i <= HALO_SEGMENTS; i++) {
      const angle = (i / HALO_SEGMENTS) * TAU;
      halo[(i + 1) * 3] = Math.cos(angle) * HALO_RADIUS;
      halo[(i + 1) * 3 + 2] = Math.sin(angle) * HALO_RADIUS;
    }
    this.haloVertices = HALO_SEGMENTS + 2;
    this.haloBuffer = this.staticBuffer(halo);
  }


  private rimLayout(stride: number): GPUVertexBufferLayout {
    return {
      arrayStride: stride,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32' },
        { shaderLocation: 2, offset: 16, format: 'float32' },
        { shaderLocation: 3, offset: 20, format: 'float32' },
        { shaderLocation: 4, offset: 24, format: 'float32x3' },
      ],
    };
  }

  private staticBuffer(data: Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return buffer;
  }

  private buildTicks(): Float32Array {
    const data = new Float32Array(TICK_COUNT * 6 * RIM_FLOATS);
    let cursor = 0;
    const write = (x: number, z: number, angle: number, baked: [number, number, number]) => {
      data[cursor++] = x;
      data[cursor++] = 0;
      data[cursor++] = z;
      data[cursor++] = angle;
      data[cursor++] = TICK_INNER;
      data[cursor++] = MODE_FLOW;
      data[cursor++] = baked[0];
      data[cursor++] = baked[1];
      data[cursor++] = baked[2];
    };
    for (let i = 0; i < TICK_COUNT; i++) {
      const start = (i / TICK_COUNT) * TAU;
      const end = start + TICK_SPAN;
      const baked: [number, number, number] = i % 2 ? [0.282, 0.302, 0.192] : [0.678, 0.675, 0.392];
      const inner0 = [Math.cos(start) * TICK_INNER, Math.sin(start) * TICK_INNER];
      const inner1 = [Math.cos(end) * TICK_INNER, Math.sin(end) * TICK_INNER];
      const outer1 = [Math.cos(end) * TICK_OUTER, Math.sin(end) * TICK_OUTER];
      const outer0 = [Math.cos(start) * TICK_OUTER, Math.sin(start) * TICK_OUTER];
      for (const point of [inner0, inner1, outer1, inner0, outer1, outer0]) write(point[0], point[1], start, baked);
    }
    return data;
  }

  resize(): void {
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.depth?.destroy();
    this.msaa?.destroy();
    this.depth = this.device.createTexture({
      size: [width, height],
      format: 'depth24plus',
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaa = this.device.createTexture({
      size: [width, height],
      format: this.format,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  drawCore(width: number, height: number, view: View, scene: Scene): void {
    // The layout can settle after the ResizeObserver already fired (the engine had
    // no renderer yet while WebGPU was still being created), which left the render
    // targets at the old size and clipped the scene. Cheap to re-check every frame.
    this.resize();
    if (width <= 0 || height <= 0 || !this.depth || !this.msaa) return;
    const scale = Math.min(width / 45, height / 37) * view.zoom;
    const flow = scene.phase * FLOW_RATE;

    const u = this.uniforms;
    u[0] = width;
    u[1] = height;
    u[2] = Math.cos(view.angle);
    u[3] = Math.sin(view.angle);
    u[4] = scale;
    u[5] = view.tilt;
    u[6] = flow;
    u[7] = scene.rgbSweep;
    u[8] = scene.rgbOn ? 1 : 0;
    u[9] = ROD_BASE;
    u[10] = HEIGHT_SCALE;
    u[11] = scene.rgbOn || scene.rgbSweep < 1 ? scene.energy : 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u.buffer as ArrayBuffer, u.byteOffset, u.byteLength);

    const ringVertices = this.buildRings(scale, view.tilt, scene.rgbOn && scene.rgbSweep >= 1, flow * (180 / Math.PI));
    this.device.queue.writeBuffer(this.ringBuffer, 0, this.ringData.buffer as ArrayBuffer, 0, ringVertices * RIM_FLOATS * 4);
    this.device.queue.writeBuffer(this.heightBuffer, 0, scene.heights.buffer as ArrayBuffer, scene.heights.byteOffset, scene.heights.byteLength);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaa.createView(),
        resolveTarget: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this.depth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });

    // Same order as the WebGL backend: glow first, then rim, then the rods.
    if (scene.rgbOn || scene.rgbSweep < 1) {
      pass.setPipeline(this.haloPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.haloBuffer);
      pass.draw(this.haloVertices);
    }
    pass.setPipeline(this.rimPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.tickBuffer);
    pass.draw(this.tickVertices);
    pass.setVertexBuffer(0, this.ringBuffer);
    pass.draw(ringVertices);

    pass.setPipeline(this.rodPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.boxBuffer);
    pass.setVertexBuffer(1, this.instanceBuffer);
    pass.setVertexBuffer(2, this.heightBuffer);
    pass.draw(BOX_VERTICES, this.source.count);

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private buildRings(scale: number, tilt: number, settled: boolean, hue: number): number {
    const data = this.ringData;
    let cursor = 0;
    for (const ring of RINGS) {
      const baked = Number.isNaN(ring.hueOffset) ? ring.plain : settled ? hueRgb(hue + ring.hueOffset, ring.saturation, ring.lightness) : ring.plain;
      const halfWidth = ring.widthPx / 2;
      const vertex = (angle: number, sign: number) => {
        const squeeze = Math.hypot(Math.cos(angle), Math.sin(angle) * tilt);
        const radius = ring.radius + (sign * halfWidth) / (scale * squeeze);
        data[cursor++] = Math.cos(angle) * radius;
        data[cursor++] = 0;
        data[cursor++] = Math.sin(angle) * radius;
        data[cursor++] = angle;
        data[cursor++] = ring.radius;
        data[cursor++] = MODE_BAKED;
        data[cursor++] = baked[0];
        data[cursor++] = baked[1];
        data[cursor++] = baked[2];
      };
      for (let segment = 0; segment < RING_SEGMENTS; segment++) {
        const start = (segment / RING_SEGMENTS) * TAU;
        const end = ((segment + 1) / RING_SEGMENTS) * TAU;
        vertex(start, -1);
        vertex(end, -1);
        vertex(end, 1);
        vertex(start, -1);
        vertex(end, 1);
        vertex(start, 1);
      }
    }
    return cursor / RIM_FLOATS;
  }

  dispose(): void {
    this.depth?.destroy();
    this.msaa?.destroy();
    this.boxBuffer.destroy();
    this.instanceBuffer.destroy();
    this.heightBuffer.destroy();
    this.tickBuffer.destroy();
    this.ringBuffer.destroy();
    this.haloBuffer.destroy();
    this.uniformBuffer.destroy();
    void this.canvas;
  }
}

function hueRgb(hue: number, saturation: number, lightness: number): [number, number, number] {
  const h = (((hue % 360) + 360) % 360) / 360;
  const q = lightness < 0.5 ? lightness * (1 + saturation) : lightness + saturation - lightness * saturation;
  const p = 2 * lightness - q;
  const channel = (t: number) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  return [channel(h + 1 / 3), channel(h), channel(h - 1 / 3)];
}
