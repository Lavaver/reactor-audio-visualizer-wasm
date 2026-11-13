// WebGL2 backend for the core scene.
//
// The projection is the same oblique one the canvas renderer used: rotate
// around Y, squash the z axis by the camera tilt, then lift each point by its
// height. There is no perspective divide, so everything is a handful of
// multiply-adds in the vertex shaders instead of a matrix stack.
//
// What moved to the GPU, and why: the 709 rods are instanced boxes, so the
// per-frame painter's sort is gone (the depth buffer orders them), and the
// flowing RGB pattern — polar angle, spiral hue, comets, wipe front — is
// evaluated in the shaders. The CPU now uploads 709 heights and a few uniforms.
//
// One property the canvas renderer had for free is transparency: the dish
// interior shows the page's CSS gradient through it. The context keeps
// `alpha: true` and every shader writes premultiplied colour, so the additive
// halo accumulates alpha instead of contributing light only.

import type { Scene, View } from './draw';
import type { CoreRenderer, RodSource } from './renderer';

const TAU = Math.PI * 2;
const RAD = 180 / Math.PI;
/** How fast the RGB pattern travels, in radians per millisecond. */
const FLOW_RATE = 0.0009;
/** Rod base offset and height scale, mirroring the wasm kernel. */
const ROD_BASE = 0.07;
const HEIGHT_SCALE = 4.8;
/** Half-width of a rod in the disc plane. */
const ROD_HALF = 0.44;
const RING_SEGMENTS = 72;
const TICK_COUNT = 80;
const TICK_INNER = 15.7;
const TICK_OUTER = 16.15;
const TICK_SPAN = 0.034;
const HALO_RADIUS = 17.9;
const HALO_SEGMENTS = 96;
/** Floats per vertex for the rim/ring program: pos3, angle, radius, mode, plain3. */
const RIM_FLOATS = 9;
/** Floats per vertex for the static rod box: corner3, shade, top, uv2. */
const BOX_FLOATS = 7;
const BOX_VERTICES = 30;

/** `aMode` for the rim/ring program: pick the flow colour or a baked one. */
const MODE_FLOW = 1;
const MODE_BAKED = 2;

interface RingSpec {
  radius: number;
  /** Stroke width in CSS pixels: these are UI lines, not world geometry. */
  widthPx: number;
  /** Hue offset once the RGB palette has settled; NaN keeps the ring dark. */
  hueOffset: number;
  lightness: number;
  saturation: number;
  plain: [number, number, number];
}

const RINGS: RingSpec[] = [
  // The dark collar of the dish. It stays dark in both palettes.
  { radius: 16.4, widthPx: 20, hueOffset: NaN, lightness: 0, saturation: 0, plain: [0.024, 0.043, 0.027] },
  { radius: 16.5, widthPx: 2, hueOffset: 0, lightness: 0.55, saturation: 0.95, plain: [0.314, 0.349, 0.282] },
  { radius: 16.0, widthPx: 1, hueOffset: 120, lightness: 0.62, saturation: 0.95, plain: [0.529, 0.537, 0.412] },
  { radius: 17.3, widthPx: 1, hueOffset: 240, lightness: 0.40, saturation: 0.95, plain: [0.188, 0.227, 0.176] },
];

// --- shader sources ---------------------------------------------------------

/** Rotation, tilt, vertical lift and clip-space mapping, shared by every pass. */
const PROJECT = `
uniform vec2 uResolution;
uniform float uCos;
uniform float uSin;
uniform float uScale;
uniform float uTilt;

vec4 project(vec3 world) {
    vec2 rotated = vec2(world.x * uCos - world.z * uSin, world.x * uSin + world.z * uCos);
    float sx = uResolution.x * 0.5 + rotated.x * uScale;
    float sy = uResolution.y * 0.55 + rotated.y * uScale * uTilt - world.y * uScale;
    return vec4(
        sx / (uResolution.x * 0.5) - 1.0,
        1.0 - sy / (uResolution.y * 0.5),
        -rotated.y * 0.05,
        1.0);
}
`;

/** The flowing palette: hue arms, comets and the clockwise wipe front. */
const FLOW = `
const float TAU = 6.28318530718;
const float DEG = 57.29577951308232;
const float FRONT_WIDTH = 0.045;

vec3 hslToRgb(float h, float s, float l) {
    h = mod(h, 360.0) / 360.0;
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
}

/** Has the wipe front passed this sector? Both directions run the same front. */
bool sectorLit(float sector, bool on, float sweep) {
    return on ? sector <= sweep : sector > sweep;
}

/** How close a sector is to the leading edge, 0..1. */
float frontStrength(float sector, float sweep) {
    if (sweep >= 1.0) return 0.0;
    float d = abs(sector - sweep);
    float dist = min(d, 1.0 - d);
    return max(0.0, 1.0 - dist / FRONT_WIDTH);
}

/** Hue of the flowing arms: two spirals travelling outwards. */
float flowHue(float angle, float radius, float flow) {
    return (angle * 2.0 + radius * 0.21 - flow * 2.0) * DEG;
}

/** Three brightness comets chasing around the disc. */
float flowComet(float angle, float flow) {
    return 0.5 + 0.5 * sin(angle * 3.0 - flow * 3.0);
}

/** Base colour of a rod in the plain palette. */
vec3 plainColor(float seed, float amp) {
    vec3 base = seed > 0.92 ? vec3(143.0, 109.0, 81.0)
              : seed > 0.85 ? vec3(97.0, 118.0, 104.0)
              : vec3(139.0 + seed * 29.0, 148.0 + seed * 22.0, 119.0 + seed * 21.0);
    return clamp((base + amp * vec3(43.0, 40.0, 24.0)) / 255.0, 0.0, 1.0);
}
`;

const ROD_VS = `#version 300 es
precision highp float;

in vec3 aCorner;    // x/z corner offsets, y = 0 at the base and 1 at the top
in float aShade;    // per-face brightness, baked on the CPU
in float aTop;
in vec2 aUv;
in vec2 aRod;       // rod centre in the disc plane (per instance)
in float aSeed;
in float aRadius;
in float aHeight;   // per instance, re-uploaded every frame

uniform float uFlow;
uniform bool uRgbOn;
uniform float uSweep;
uniform float uRodBase;
uniform float uHeightScale;

out vec3 vColor;
out vec2 vUv;
out float vTop;

${PROJECT}
${FLOW}

void main() {
    float amp = max(0.0, (aHeight - uRodBase) / uHeightScale);
    vec3 world = vec3(aRod.x + aCorner.x, uRodBase + aCorner.y * aHeight, aRod.y + aCorner.z);

    gl_Position = project(world);

    // The wipe front is anchored to the disc, not to the screen: it has to use
    // the rod's world angle so the palette turns with the reactor exactly like
    // the halo underneath it. Deriving it from this vertex's own position would
    // also let one box straddle the front across its four corners.
    float rodAngle = atan(aRod.y, aRod.x);
    float sector = fract(rodAngle / TAU);
    vec3 color;
    if (sectorLit(sector, uRgbOn, uSweep)) {
        float front = frontStrength(sector, uSweep);
        float lightness = (30.0 + flowComet(rodAngle, uFlow) * 34.0
            + min(18.0, amp * 30.0) + front * 26.0) / 100.0;
        color = hslToRgb(flowHue(rodAngle, aRadius, uFlow), max(0.0, 0.95 - front * 0.62), lightness);
    } else {
        color = plainColor(aSeed, amp);
    }

    vColor = color * aShade;
    vUv = aUv;
    vTop = aTop;
}
`;

const ROD_FS = `#version 300 es
precision highp float;

in vec3 vColor;
in vec2 vUv;
in float vTop;
out vec4 outColor;

void main() {
    vec3 color = vColor;
    // The small dark cap the 2D renderer stamped on every rod top.
    if (vTop > 0.5) {
        if (max(abs(vUv.x - 0.5), abs(vUv.y - 0.5)) < 0.30) color *= 0.80;
    }
    outColor = vec4(color, 1.0);
}
`;

const RIM_VS = `#version 300 es
precision highp float;

in vec3 aPos;
in float aAngle;
in float aRadius;
in float aMode;
in vec3 aBaked;

uniform float uFlow;
uniform bool uRgbOn;
uniform float uSweep;

out vec3 vColor;

${PROJECT}
${FLOW}

void main() {
    gl_Position = project(aPos);

    vec3 color = aBaked;
    if (aMode < 1.5) {
        // Same world-anchored front as the rods, and the tick's own angle rather
        // than each vertex, so a single tick is never split in half.
        float sector = fract(aAngle / TAU);
        if (sectorLit(sector, uRgbOn, uSweep)) {
            float front = frontStrength(sector, uSweep);
            float lightness = (22.0 + flowComet(aAngle, uFlow) * 48.0 + front * 26.0) / 100.0;
            float saturation = max(0.0, 0.95 - front * 0.55);
            color = hslToRgb(flowHue(aAngle, aRadius, uFlow), saturation, lightness);
        }
    }
    vColor = color;
}
`;

const RIM_FS = `#version 300 es
precision highp float;

in vec3 vColor;
out vec4 outColor;

void main() {
    outColor = vec4(vColor, 1.0);
}
`;

const HALO_VS = `#version 300 es
precision highp float;

in vec3 aPos;
out vec2 vDisc;

${PROJECT}

void main() {
    gl_Position = project(aPos);
    // Disc-space coordinates: the halo's front therefore uses the same world
    // angle as the rods, so both sweep together under the view rotation.
    vDisc = aPos.xz;
}
`;

const HALO_FS = `#version 300 es
precision highp float;

in vec2 vDisc;

uniform float uFlow;
uniform float uTilt;
uniform float uScale;
uniform float uEnergy;
uniform float uSweep;
uniform bool uRgbOn;

out vec4 outColor;

${FLOW}

void main() {
    float radius = length(vDisc);
    float angle = atan(vDisc.y, vDisc.x);
    float sector = fract(angle / TAU);
    if (!sectorLit(sector, uRgbOn, uSweep)) discard;

    vec3 rainbow = hslToRgb(uFlow * DEG + angle * DEG + 90.0, 1.0, 0.58);
    float body = mix(0.10, 0.40, smoothstep(12.0, 17.9, radius)) * (0.35 + uEnergy * 0.65);

    // The rim is a screen-space band; without the squeeze correction it would
    // thin out at the top and bottom of the ellipse.
    float squeeze = sqrt(cos(angle) * cos(angle) + sin(angle) * sin(angle) * uTilt * uTilt);
    float rimPx = abs(radius - 17.9) * uScale * squeeze;
    float rim = (1.0 - smoothstep(0.0, 1.4, rimPx)) * (0.22 + uEnergy * 0.6);

    // Three comets orbiting the edge, matching the lobes on the rods.
    vec3 cometColor = vec3(0.0);
    float cometAlpha = 0.0;
    for (int i = 0; i < 3; i++) {
        float cometAngle = uFlow + float(i) * TAU / 3.0;
        vec2 spot = vec2(cos(cometAngle), sin(cometAngle)) * 16.6;
        float d = length(vDisc - spot) / 3.8;
        float glow = exp(-d * d * 1.8);
        cometColor += hslToRgb(uFlow * DEG + cometAngle * DEG, 1.0, 0.82) * glow;
        cometAlpha += glow;
    }
    float cometGain = 0.32 + uEnergy * 0.55;
    cometColor *= cometGain;
    cometAlpha *= cometGain;

    vec3 color = rainbow * (body + rim) + cometColor;
    outColor = vec4(color, min(1.0, body + rim + cometAlpha));
}
`;

// --- helpers ----------------------------------------------------------------

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('无法创建着色器对象');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '未知原因';
    gl.deleteShader(shader);
    throw new Error(`着色器编译失败：${log}`);
  }
  return shader;
}

function link(gl: WebGL2RenderingContext, vertex: string, fragment: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('无法创建着色器程序');
  const vs = compile(gl, gl.VERTEX_SHADER, vertex);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.detachShader(program, vs);
  gl.detachShader(program, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? '未知原因';
    gl.deleteProgram(program);
    throw new Error(`着色器程序链接失败：${log}`);
  }
  return program;
}

type UniformMap = Record<string, WebGLUniformLocation | null>;

function uniforms(gl: WebGL2RenderingContext, program: WebGLProgram, names: string[]): UniformMap {
  const map: UniformMap = {};
  for (const name of names) map[name] = gl.getUniformLocation(program, name);
  return map;
}

/** Standard HSL -> RGB, so the rings can be coloured on the CPU. */
function hslToRgb(hue: number, saturation: number, lightness: number): [number, number, number] {
  const h = (((hue % 360) + 360) % 360) / 360;
  const q = lightness < 0.5
    ? lightness * (1 + saturation)
    : lightness + saturation - lightness * saturation;
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

/** The 30 vertices of one rod box: a top face plus four shaded sides. */
function buildBox(): Float32Array {
  const corners: Array<[number, number]> = [
    [-ROD_HALF, -ROD_HALF],
    [ROD_HALF, -ROD_HALF],
    [ROD_HALF, ROD_HALF],
    [-ROD_HALF, ROD_HALF],
  ];
  // Brightness of each side, taken from the flat colours the 2D renderer used.
  const sides = [0.44, 0.38, 0.36, 0.40];
  const data: number[] = [];
  // Attribute order is (x, up, z), matching `vec3 aCorner` in the shader.
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

// --- renderer ---------------------------------------------------------------

export class WebglRenderer implements CoreRenderer {
  readonly kind = 'webgl' as const;
  private readonly gl: WebGL2RenderingContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly count: number;

  private readonly rodProgram: WebGLProgram;
  private readonly rimProgram: WebGLProgram;
  private readonly haloProgram: WebGLProgram;
  private readonly rodU: UniformMap;
  private readonly rimU: UniformMap;
  private readonly haloU: UniformMap;

  private readonly rodVao: WebGLVertexArrayObject;
  private readonly ringVao: WebGLVertexArrayObject;
  private readonly tickVao: WebGLVertexArrayObject;
  private readonly haloVao: WebGLVertexArrayObject;
  private readonly heightBuffer: WebGLBuffer;
  private readonly ringBuffer: WebGLBuffer;
  private readonly ringData: Float32Array;
  private readonly tickVertices: number;
  private readonly haloVertices: number;
  private readonly owned: Array<WebGLBuffer | WebGLVertexArrayObject> = [];

  constructor(canvas: HTMLCanvasElement, source: RodSource, preserveDrawingBuffer: boolean) {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: true,
      depth: true,
      // Only the regression harness needs to read pixels back after compositing.
      preserveDrawingBuffer,
    });
    if (!gl) throw new Error('这个浏览器没有可用的 WebGL2，无法绘制反应堆本体。');
    this.gl = gl;
    this.canvas = canvas;
    this.count = source.count;

    this.rodProgram = link(gl, ROD_VS, ROD_FS);
    this.rimProgram = link(gl, RIM_VS, RIM_FS);
    this.haloProgram = link(gl, HALO_VS, HALO_FS);

    this.rodU = uniforms(gl, this.rodProgram, [
      'uResolution', 'uCos', 'uSin', 'uScale', 'uTilt',
      'uFlow', 'uRgbOn', 'uSweep', 'uRodBase', 'uHeightScale',
    ]);
    this.rimU = uniforms(gl, this.rimProgram, [
      'uResolution', 'uCos', 'uSin', 'uScale', 'uTilt', 'uFlow', 'uRgbOn', 'uSweep',
    ]);
    this.haloU = uniforms(gl, this.haloProgram, [
      'uResolution', 'uCos', 'uSin', 'uScale', 'uTilt',
      'uFlow', 'uRgbOn', 'uSweep', 'uEnergy',
    ]);

    const rod = this.createRodVao(source);
    this.rodVao = rod.vao;
    this.heightBuffer = rod.heights;
    const ticks = this.buildTicks();
    this.tickVertices = ticks.length / RIM_FLOATS;
    this.tickVao = this.createRimVao(ticks, false);

    this.ringData = new Float32Array(RINGS.length * RING_SEGMENTS * 6 * RIM_FLOATS);
    this.ringBuffer = this.track(this.require(gl.createBuffer()));
    this.ringVao = this.createRimVao(this.ringData, true);

    const halo = this.buildHalo();
    this.haloVertices = halo.length / 3;
    this.haloVao = this.createHaloVao(halo);

    this.resize();
  }

  private require<T>(value: T | null): T {
    if (value === null) throw new Error('WebGL 资源创建失败');
    return value;
  }

  private track<T extends WebGLBuffer | WebGLVertexArrayObject>(value: T): T {
    this.owned.push(value);
    return value;
  }

  private attributeOf(program: WebGLProgram, name: string): number {
    const location = this.gl.getAttribLocation(program, name);
    if (location < 0) throw new Error(`着色器里没有找到属性 ${name}`);
    return location;
  }

  private createRodVao(source: RodSource): { vao: WebGLVertexArrayObject; heights: WebGLBuffer } {
    const gl = this.gl;
    const vao = this.track(this.require(gl.createVertexArray()));
    gl.bindVertexArray(vao);

    // Box geometry, shared by every instance.
    const box = this.track(this.require(gl.createBuffer()));
    gl.bindBuffer(gl.ARRAY_BUFFER, box);
    gl.bufferData(gl.ARRAY_BUFFER, buildBox(), gl.STATIC_DRAW);
    const boxStride = BOX_FLOATS * 4;
    const bindBox = (name: string, size: number, offset: number) => {
      const location = this.attributeOf(this.rodProgram, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, boxStride, offset);
    };
    bindBox('aCorner', 3, 0);
    bindBox('aShade', 1, 12);
    bindBox('aTop', 1, 16);
    bindBox('aUv', 2, 20);

    // Per-instance constants: where the rod stands, its seed and its radius.
    const instance = new Float32Array(source.count * 4);
    for (let i = 0; i < source.count; i++) {
      const offset = i * source.stride;
      instance[i * 4] = source.rods[offset];
      instance[i * 4 + 1] = source.rods[offset + 1];
      instance[i * 4 + 2] = source.rods[offset + 3];
      instance[i * 4 + 3] = source.rods[offset + 2];
    }
    const constants = this.track(this.require(gl.createBuffer()));
    gl.bindBuffer(gl.ARRAY_BUFFER, constants);
    gl.bufferData(gl.ARRAY_BUFFER, instance, gl.STATIC_DRAW);
    const instanceStride = 4 * 4;
    const bindInstance = (name: string, size: number, offset: number) => {
      const location = this.attributeOf(this.rodProgram, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, instanceStride, offset);
      gl.vertexAttribDivisor(location, 1);
    };
    bindInstance('aRod', 2, 0);
    bindInstance('aSeed', 1, 8);
    bindInstance('aRadius', 1, 12);

    // Per-instance heights, rewritten every frame straight from wasm memory.
    const heights = this.track(this.require(gl.createBuffer()));
    gl.bindBuffer(gl.ARRAY_BUFFER, heights);
    gl.bufferData(gl.ARRAY_BUFFER, source.count * 4, gl.DYNAMIC_DRAW);
    const heightLocation = this.attributeOf(this.rodProgram, 'aHeight');
    gl.enableVertexAttribArray(heightLocation);
    gl.vertexAttribPointer(heightLocation, 1, gl.FLOAT, false, 4, 0);
    gl.vertexAttribDivisor(heightLocation, 1);

    gl.bindVertexArray(null);
    return { vao, heights };
  }

  private createRimVao(data: Float32Array, reuseRingBuffer: boolean): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.track(this.require(gl.createVertexArray()));
    gl.bindVertexArray(vao);
    const buffer = reuseRingBuffer ? this.ringBuffer : this.track(this.require(gl.createBuffer()));
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      data,
      reuseRingBuffer ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW,
    );
    const stride = RIM_FLOATS * 4;
    const bind = (name: string, size: number, offset: number) => {
      const location = this.attributeOf(this.rimProgram, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
    };
    bind('aPos', 3, 0);
    bind('aAngle', 1, 12);
    bind('aRadius', 1, 16);
    bind('aMode', 1, 20);
    bind('aBaked', 3, 24);
    gl.bindVertexArray(null);
    return vao;
  }

  private createHaloVao(halo: Float32Array): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.track(this.require(gl.createVertexArray()));
    gl.bindVertexArray(vao);
    const buffer = this.track(this.require(gl.createBuffer()));
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, halo, gl.STATIC_DRAW);
    const location = this.attributeOf(this.haloProgram, 'aPos');
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 12, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  /** 80 rim ticks: static geometry whose colour is decided in the shader. */
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
      // The two alternating rim colours of the original page.
      const baked: [number, number, number] = i % 2
        ? [0.282, 0.302, 0.192]
        : [0.678, 0.675, 0.392];
      const inner0 = [Math.cos(start) * TICK_INNER, Math.sin(start) * TICK_INNER];
      const inner1 = [Math.cos(end) * TICK_INNER, Math.sin(end) * TICK_INNER];
      const outer1 = [Math.cos(end) * TICK_OUTER, Math.sin(end) * TICK_OUTER];
      const outer0 = [Math.cos(start) * TICK_OUTER, Math.sin(start) * TICK_OUTER];
      for (const point of [inner0, inner1, outer1, inner0, outer1, outer0]) {
        write(point[0], point[1], start, baked);
      }
    }
    return data;
  }

  private buildHalo(): Float32Array {
    const data = new Float32Array((HALO_SEGMENTS + 2) * 3);
    for (let i = 0; i <= HALO_SEGMENTS; i++) {
      const angle = (i / HALO_SEGMENTS) * TAU;
      data[(i + 1) * 3] = Math.cos(angle) * HALO_RADIUS;
      data[(i + 1) * 3 + 1] = 0;
      data[(i + 1) * 3 + 2] = Math.sin(angle) * HALO_RADIUS;
    }
    return data;
  }

  /**
   * Rebuild the four rings for this frame.
   *
   * Their thickness is a CSS pixel count, so the world-space radius offset has
   * to be divided by how much a radial step is squashed at that angle — without
   * it the stroke would thin out at the top and bottom of the ellipse.
   */
  private buildRings(scale: number, tilt: number, settled: boolean, hue: number): number {
    const data = this.ringData;
    let cursor = 0;
    for (const ring of RINGS) {
      const baked = Number.isNaN(ring.hueOffset)
        ? ring.plain
        : settled
          ? hslToRgb(hue + ring.hueOffset, ring.saturation, ring.lightness)
          : ring.plain;
      const halfWidth = ring.widthPx / 2;
      const radiusAt = (angle: number, sign: number) => {
        const squeeze = Math.hypot(Math.cos(angle), Math.sin(angle) * tilt);
        return ring.radius + (sign * halfWidth) / (scale * squeeze);
      };
      const vertex = (angle: number, sign: number) => {
        const radius = radiusAt(angle, sign);
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

  private applyProjection(map: UniformMap, width: number, height: number, view: View, scale: number): void {
    const gl = this.gl;
    gl.uniform2f(map.uResolution, width, height);
    gl.uniform1f(map.uCos, Math.cos(view.angle));
    gl.uniform1f(map.uSin, Math.sin(view.angle));
    gl.uniform1f(map.uScale, scale);
    gl.uniform1f(map.uTilt, view.tilt);
  }

  resize(): void {
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  drawCore(width: number, height: number, view: View, scene: Scene): void {
    const gl = this.gl;
    if (width <= 0 || height <= 0) return;

    const scale = Math.min(width / 45, height / 37) * view.zoom;
    const flow = scene.phase * FLOW_RATE;
    const hue = flow * RAD;
    const settled = scene.rgbOn && scene.rgbSweep >= 1;
    const rgbOn = scene.rgbOn ? 1 : 0;

    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.SCISSOR_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 1. The light pollution, behind everything and purely additive.
    if (scene.rgbOn || scene.rgbSweep < 1) {
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.haloProgram);
      this.applyProjection(this.haloU, width, height, view, scale);
      gl.uniform1f(this.haloU.uFlow, flow);
      gl.uniform1f(this.haloU.uSweep, scene.rgbSweep);
      gl.uniform1f(this.haloU.uEnergy, scene.energy);
      gl.uniform1i(this.haloU.uRgbOn, rgbOn);
      gl.bindVertexArray(this.haloVao);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, this.haloVertices);
    }

    // 2. Rings and rim ticks, opaque and flat.
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.useProgram(this.rimProgram);
    this.applyProjection(this.rimU, width, height, view, scale);
    gl.uniform1f(this.rimU.uFlow, flow);
    gl.uniform1f(this.rimU.uSweep, scene.rgbSweep);
    gl.uniform1i(this.rimU.uRgbOn, rgbOn);

    gl.bindVertexArray(this.tickVao);
    gl.drawArrays(gl.TRIANGLES, 0, this.tickVertices);

    const ringVertices = this.buildRings(scale, view.tilt, settled, hue);
    gl.bindVertexArray(this.ringVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.ringData, 0, ringVertices * RIM_FLOATS);
    gl.drawArrays(gl.TRIANGLES, 0, ringVertices);

    // 3. The rods themselves: instanced boxes, ordered by the depth buffer.
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.useProgram(this.rodProgram);
    this.applyProjection(this.rodU, width, height, view, scale);
    gl.uniform1f(this.rodU.uFlow, flow);
    gl.uniform1f(this.rodU.uSweep, scene.rgbSweep);
    gl.uniform1i(this.rodU.uRgbOn, rgbOn);
    gl.uniform1f(this.rodU.uRodBase, ROD_BASE);
    gl.uniform1f(this.rodU.uHeightScale, HEIGHT_SCALE);

    gl.bindVertexArray(this.rodVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.heightBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, scene.heights, 0, this.count);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, BOX_VERTICES, this.count);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const gl = this.gl;
    for (const resource of this.owned) {
      if (resource instanceof WebGLBuffer) gl.deleteBuffer(resource);
      else gl.deleteVertexArray(resource);
    }
    this.owned.length = 0;
    gl.deleteProgram(this.rodProgram);
    gl.deleteProgram(this.rimProgram);
    gl.deleteProgram(this.haloProgram);
  }
}
