import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type ProceduralModelOptions = {
  wireframe?: boolean;
  castShadow?: boolean;
  receiveShadow?: boolean;
  textureSize?: number;
  textureAnisotropy?: number;
  qualityPriority?: 'reference-fidelity' | 'balanced';
};

export type ProceduralModelRuntime = {
  nodes: Record<string, THREE.Object3D>;
  meshes: Record<string, THREE.Mesh>;
  sockets: Record<string, THREE.Object3D>;
  colliders: Record<string, unknown>;
  destructionGroups: Record<string, THREE.Object3D[]>;
};

type SculptMaterialSpec = Record<string, any>;

function buildLatheGeometry(profile: { points: [number, number][]; segments?: number }): THREE.LatheGeometry {
  const points = profile.points.map(([x, y]) => new THREE.Vector2(Math.max(0.0001, x), y));
  return new THREE.LatheGeometry(points, profile.segments ?? 24);
}

function buildTubeGeometry(
  path: { points: [number, number, number][]; radius?: number; radialSegments?: number; closed?: boolean },
): THREE.TubeGeometry {
  const vectors = path.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const curve = new THREE.CatmullRomCurve3(vectors, path.closed ?? false);
  const tubularSegments = Math.max(8, path.points.length * 6);
  return new THREE.TubeGeometry(curve, tubularSegments, path.radius ?? 0.05, path.radialSegments ?? 8, path.closed ?? false);
}

type TaperedStation = { position: [number, number, number]; rx: number; rz: number; twist?: number };

// Frames come from PARALLEL TRANSPORT, not from a Frenet frame. A Frenet frame is defined by
// the curve's normal, which flips sign wherever the path has an inflection or straightens out,
// and every flip twists the surface 180 degrees within one segment. Carrying the previous frame
// forward and removing only its along-path component keeps the twist continuous. THREE's own
// extrudePath and TubeGeometry do not expose this, which is why this is hand-built.
function buildTaperedSweepGeometry(
  sweep: { stations: TaperedStation[]; radialSegments?: number; capEnds?: boolean },
): THREE.BufferGeometry {
  const stations = sweep.stations;
  if (stations.length < 2) throw new Error('tapered-sweep needs at least two stations');
  const radial = Math.max(3, sweep.radialSegments ?? 10);
  const centres = stations.map((s) => new THREE.Vector3(...s.position));

  const tangents = centres.map((_, i) => {
    const prev = centres[Math.max(0, i - 1)];
    const next = centres[Math.min(centres.length - 1, i + 1)];
    const t = next.clone().sub(prev);
    // Coincident neighbours would normalise to NaN and poison every downstream vertex.
    return t.lengthSq() < 1e-12 ? new THREE.Vector3(0, 1, 0) : t.normalize();
  });

  // Seed a reference axis that is not parallel to the first tangent, or the first cross
  // product is degenerate and the whole sweep collapses to a line.
  let ref = new THREE.Vector3(0, 0, 1);
  if (Math.abs(tangents[0].dot(ref)) > 0.9) ref = new THREE.Vector3(1, 0, 0);

  const normals: THREE.Vector3[] = [];
  const binormals: THREE.Vector3[] = [];
  let carried = ref.clone().sub(tangents[0].clone().multiplyScalar(ref.dot(tangents[0]))).normalize();
  for (let i = 0; i < tangents.length; i += 1) {
    const t = tangents[i];
    // Project the carried frame back onto the plane perpendicular to this tangent.
    const n = carried.clone().sub(t.clone().multiplyScalar(carried.dot(t)));
    if (n.lengthSq() < 1e-12) {
      const fallback = Math.abs(t.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      n.copy(fallback.sub(t.clone().multiplyScalar(fallback.dot(t))));
    }
    n.normalize();
    normals.push(n);
    binormals.push(new THREE.Vector3().crossVectors(t, n).normalize());
    carried = n;
  }

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const ringStart: number[] = [];
  const isPoint: boolean[] = [];

  for (let i = 0; i < stations.length; i += 1) {
    const st = stations[i];
    const v = i / (stations.length - 1);
    ringStart.push(positions.length / 3);
    // A station whose section has collapsed emits ONE vertex, not a ring of radius zero.
    // A degenerate ring still carries `radial` coincident vertices and `radial` zero-area
    // triangles, so the lock ends in a blunt cap the width of the floating-point noise
    // rather than at a point -- and a hair lock, a horn or a blade tip has to reach a point.
    if (st.rx <= 1e-6 && st.rz <= 1e-6) {
      isPoint.push(true);
      positions.push(centres[i].x, centres[i].y, centres[i].z);
      uvs.push(0.5, v);
      continue;
    }
    isPoint.push(false);
    const twist = ((st.twist ?? 0) * Math.PI) / 180;
    for (let j = 0; j <= radial; j += 1) {
      const theta = (j / radial) * Math.PI * 2 + twist;
      const offset = normals[i].clone().multiplyScalar(Math.cos(theta) * st.rx)
        .add(binormals[i].clone().multiplyScalar(Math.sin(theta) * st.rz));
      const p = centres[i].clone().add(offset);
      positions.push(p.x, p.y, p.z);
      uvs.push(j / radial, v);
    }
  }

  for (let i = 0; i < stations.length - 1; i += 1) {
    const a0 = ringStart[i];
    const b0 = ringStart[i + 1];
    if (isPoint[i] && isPoint[i + 1]) continue;   // two collapsed stations bound nothing
    for (let j = 0; j < radial; j += 1) {
      // Wound so the face normal points radially OUTWARD.
      //
      // Ring vertices advance from `normal` toward `binormal`, and binormal is
      // tangent x normal, so increasing theta runs counter-clockwise seen from the
      // far end of the segment. Taking the ring-to-ring edge first therefore puts
      // the cross product on the inside. Measured as signed volume on the built
      // mesh: every tapered-sweep came out negative -- a torso at -0.0674 and a
      // tail at -0.0044 against a positive ellipsoid head -- so every sweep this
      // generator has ever emitted rendered its back faces, with normals pointing
      // into the solid and every lighting judgement made on the wrong surface.
      if (isPoint[i]) indices.push(a0, b0 + j + 1, b0 + j);
      else if (isPoint[i + 1]) indices.push(a0 + j, a0 + j + 1, b0);
      else indices.push(a0 + j, a0 + j + 1, b0 + j, a0 + j + 1, b0 + j + 1, b0 + j);
    }
  }

  if (sweep.capEnds ?? true) {
    for (const end of [0, stations.length - 1]) {
      if (isPoint[end]) continue;   // a point end is already closed
      const centreIndex = positions.length / 3;
      positions.push(centres[end].x, centres[end].y, centres[end].z);
      uvs.push(0.5, end === 0 ? 0 : 1);
      const base = ringStart[end];
      for (let j = 0; j < radial; j += 1) {
        if (end === 0) indices.push(centreIndex, base + j + 1, base + j);
        else indices.push(centreIndex, base + j, base + j + 1);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function readLayerNumber(value: unknown, keys: string[], fallback: number): number {
  if (typeof value === 'number') return value;
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of keys) {
      if (typeof record[key] === 'number') return record[key] as number;
    }
  }
  return fallback;
}

function hexToRgb(hex: string): [number, number, number] {
  const normalized = /^#[0-9a-f]{3}$/i.test(hex)
    ? '#' + hex.slice(1).split('').map((part) => part + part).join('')
    : hex;
  const value = /^#[0-9a-f]{6}$/i.test(normalized) ? Number.parseInt(normalized.slice(1), 16) : 0x8a7a5f;
  return [clampAlbedoChannel((value >> 16) & 255), clampAlbedoChannel((value >> 8) & 255), clampAlbedoChannel(value & 255)];
}

function materialPalette(spec: SculptMaterialSpec): string[] {
  const palette = spec.colorVariation?.palette;
  if (Array.isArray(palette) && palette.length > 0) return palette.filter((value) => typeof value === 'string');
  const secondary = spec.albedo?.secondary;
  const colors = [spec.baseColor ?? spec.color ?? spec.albedo?.dominant, ...(Array.isArray(secondary) ? secondary : [])];
  return colors.filter((value): value is string => typeof value === 'string' && value.startsWith('#'));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clampAlbedoChannel(value: number): number {
  return Math.max(30, Math.min(240, Math.round(value)));
}

function clampPbrF0(value: number): number {
  return Math.max(0.02, Math.min(1, value));
}

function clampPbrIor(value: number): number {
  return Math.max(1, Math.min(2.5, value));
}

function clampPbrMetalness(value: number): number {
  return value >= 0.5 ? 1 : 0;
}

function clampedAlbedoColor(spec: SculptMaterialSpec): THREE.Color {
  const source = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  // setStyle with an explicit SRGBColorSpace, NOT the numeric constructor.
  //
  // `new THREE.Color(r, g, b)` treats its arguments as LINEAR working-space components,
  // while an authored `baseColor` hex is sRGB. Feeding one to the other skipped the
  // transfer function and lifted every dark albedo: #2e2a28, authored as a near-black
  // vinyl, rendered at roughly sRGB 0.46 — a mid grey. The error is largest exactly where
  // it matters most, because the transfer curve is steepest near black.
  return new THREE.Color().setStyle(source, THREE.SRGBColorSpace);
}

function smoothCurve(value: number): number {
  return value * value * (3 - 2 * value);
}

function periodicHash(x: number, y: number, seed: number, periodX: number, periodY: number): number {
  const wrappedX = ((x % periodX) + periodX) % periodX;
  const wrappedY = ((y % periodY) + periodY) % periodY;
  let value = Math.imul(wrappedX + seed * 17, 374761393) ^ Math.imul(wrappedY + seed * 31, 668265263);
  value = Math.imul(value ^ (value >>> 13), 1274126177);
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295;
}

function periodicValueNoise(u: number, v: number, seed: number, periodX: number, periodY: number): number {
  const x = u * periodX;
  const y = v * periodY;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smoothCurve(x - x0);
  const ty = smoothCurve(y - y0);
  const a = periodicHash(x0, y0, seed, periodX, periodY);
  const b = periodicHash(x0 + 1, y0, seed, periodX, periodY);
  const c = periodicHash(x0, y0 + 1, seed, periodX, periodY);
  const d = periodicHash(x0 + 1, y0 + 1, seed, periodX, periodY);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(a, b, tx), THREE.MathUtils.lerp(c, d, tx), ty);
}

type SurfaceBand = {
  frequency: number;
  amplitude: number;
  stretchX: number;
  stretchY: number;
  ridge: boolean;
};

function surfaceBands(spec: SculptMaterialSpec): SurfaceBand[] {
  const source = Array.isArray(spec.surfaceFrequencyBands) ? spec.surfaceFrequencyBands : [];
  const parsed = source.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const band = item as Record<string, unknown>;
    const frequency = typeof band.frequency === 'number' ? band.frequency : 0;
    const amplitude = typeof band.amplitude === 'number' ? band.amplitude : 0;
    if (frequency <= 0 || amplitude <= 0) return [];
    const stretch = Array.isArray(band.stretch) ? band.stretch : [1, 1];
    const description = `${String(band.pattern ?? '')} ${String(band.role ?? '')}`.toLowerCase();
    return [{
      frequency,
      amplitude,
      stretchX: typeof stretch[0] === 'number' ? Math.max(0.1, stretch[0]) : 1,
      stretchY: typeof stretch[1] === 'number' ? Math.max(0.1, stretch[1]) : 1,
      ridge: /(ridge|groove|grain|fiber|striated|crack)/.test(description),
    }];
  });
  return parsed.length > 0 ? parsed : [
    { frequency: 2, amplitude: 0.42, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 12, amplitude: 0.22, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 56, amplitude: 0.08, stretchX: 1, stretchY: 1, ridge: false },
  ];
}

function sampleSurface(u: number, v: number, bands: SurfaceBand[], seed: number): number {
  let value = 0;
  let weight = 0;
  for (let index = 0; index < bands.length; index += 1) {
    const band = bands[index];
    const periodX = Math.max(1, Math.round(band.frequency * band.stretchX));
    const periodY = Math.max(1, Math.round(band.frequency * band.stretchY));
    let sample = periodicValueNoise(u, v, seed + index * 1013, periodX, periodY);
    if (band.ridge) sample = 1 - Math.abs(sample * 2 - 1);
    value += sample * band.amplitude;
    weight += band.amplitude;
  }
  return weight > 0 ? clamp01(value / weight) : 0.5;
}

function mixPalette(colors: [number, number, number][], value: number): [number, number, number] {
  if (colors.length === 1) return colors[0];
  const scaled = clamp01(value) * (colors.length - 1);
  const index = Math.min(colors.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const a = colors[index];
  const b = colors[index + 1];
  return [
    Math.round(THREE.MathUtils.lerp(a[0], b[0], mix)),
    Math.round(THREE.MathUtils.lerp(a[1], b[1], mix)),
    Math.round(THREE.MathUtils.lerp(a[2], b[2], mix)),
  ];
}

type ColorGradientStop = { offset: number; color: string };
type ColorGradientSpec = {
  type: 'linear' | 'radial';
  axis: [number, number];
  stops: ColorGradientStop[];
};

function parseRgba(value: string): [number, number, number] {
  const match = /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(value);
  if (!match) return [138, 122, 95];
  return [clampAlbedoChannel(Number(match[1])), clampAlbedoChannel(Number(match[2])), clampAlbedoChannel(Number(match[3]))];
}

// Analytical per-pixel gradient sample. The extraction schema's colorGradient carries
// exact rgba(...) stop colors (see extract_part_color_recipe.py), so this samples the
// same trend directly in JS math rather than round-tripping through a Canvas 2D
// createLinearGradient/createRadialGradient object — same visual result, and it composes
// directly with the existing noise/height-correlated colorVariation blend below.
function sampleColorGradient(gradient: ColorGradientSpec, u: number, v: number): [number, number, number] {
  const stops = gradient.stops.length >= 2 ? gradient.stops : [{ offset: 0, color: 'rgba(138,122,95,1)' }, { offset: 1, color: 'rgba(138,122,95,1)' }];
  let t: number;
  if (gradient.type === 'radial') {
    const [cx, cy] = gradient.axis;
    const dx = u - cx;
    const dy = v - cy;
    const maxRadius = Math.max(0.001, Math.hypot(Math.max(cx, 1 - cx), Math.max(cy, 1 - cy)));
    t = clamp01(Math.hypot(dx, dy) / maxRadius);
  } else {
    const [ax, ay] = gradient.axis;
    const projection = (u - 0.5) * ax + (v - 0.5) * ay;
    const maxProjection = 0.5 * (Math.abs(ax) + Math.abs(ay)) || 0.5;
    t = clamp01(projection / maxProjection + 0.5);
  }
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.max(0, Math.floor(scaled)));
  const mix = scaled - index;
  const a = parseRgba(stops[index].color);
  const b = parseRgba(stops[index + 1].color);
  return [
    THREE.MathUtils.lerp(a[0], b[0], mix),
    THREE.MathUtils.lerp(a[1], b[1], mix),
    THREE.MathUtils.lerp(a[2], b[2], mix),
  ];
}

function writePixel(data: Uint8ClampedArray, offset: number, red: number, green: number, blue: number): void {
  data[offset] = Math.max(0, Math.min(255, Math.round(red)));
  data[offset + 1] = Math.max(0, Math.min(255, Math.round(green)));
  data[offset + 2] = Math.max(0, Math.min(255, Math.round(blue)));
  data[offset + 3] = 255;
}

function makeCanvas(size: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

function createMapTexture(
  canvas: HTMLCanvasElement,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [2, 2];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 2,
    typeof repeat[1] === 'number' ? repeat[1] : 2,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

type ProceduralTextureSet = {
  albedo: THREE.Texture;
  roughness: THREE.Texture;
  height: THREE.Texture;
  normal: THREE.Texture;
  ao: THREE.Texture;
  source: 'reference-pixel-extraction' | 'procedural';
};

function referenceMapUrl(spec: SculptMaterialSpec, channel: string): string | null {
  const reference = spec.referencePbr;
  if (!reference || typeof reference !== 'object') return null;
  if (reference.usable === false) return null;
  const confidence = typeof reference.confidence === 'number'
    ? reference.confidence
    : (typeof reference.estimatedFidelity === 'number' ? reference.estimatedFidelity : 0);
  const threshold = typeof reference.targetThreshold === 'number' ? reference.targetThreshold : 0.7;
  if (confidence < threshold) return null;
  const maps = reference.maps;
  if (!maps || typeof maps !== 'object') return null;
  const map = (maps as Record<string, unknown>)[channel];
  if (!map || typeof map !== 'object') return null;
  const record = map as Record<string, unknown>;
  const url = typeof record.url === 'string' && record.url.trim() ? record.url : record.path;
  return typeof url === 'string' && url.trim() ? url : null;
}

function createLoadedMapTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.Texture {
  const texture = new THREE.TextureLoader().load(url);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [1, 1];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 1,
    typeof repeat[1] === 'number' ? repeat[1] : 1,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

function makeReferenceTextureSet(spec: SculptMaterialSpec, options: ProceduralModelOptions): ProceduralTextureSet | null {
  const albedo = referenceMapUrl(spec, 'albedo');
  const roughness = referenceMapUrl(spec, 'roughness');
  const height = referenceMapUrl(spec, 'height');
  const normal = referenceMapUrl(spec, 'normal');
  const ao = referenceMapUrl(spec, 'ao');
  if (!albedo || !roughness || !height || !normal || !ao) return null;
  return {
    albedo: createLoadedMapTexture(albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createLoadedMapTexture(roughness, THREE.NoColorSpace, spec, options),
    height: createLoadedMapTexture(height, THREE.NoColorSpace, spec, options),
    normal: createLoadedMapTexture(normal, THREE.NoColorSpace, spec, options),
    ao: createLoadedMapTexture(ao, THREE.NoColorSpace, spec, options),
    source: 'reference-pixel-extraction',
  };
}

function makeProceduralTextureSet(
  id: string,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): ProceduralTextureSet | null {
  if (typeof document === 'undefined') return null;
  const qualityFirst = (options.qualityPriority ?? 'reference-fidelity') === 'reference-fidelity';
  const requested = options.textureSize ?? spec.textureResolution;
  const requestedSize = typeof requested === 'number' && Number.isFinite(requested)
    ? requested
    : (qualityFirst ? 1024 : 512);
  const size = Math.max(256, Math.min(2048, 2 ** Math.round(Math.log2(requestedSize))));
  const canvases = {
    albedo: makeCanvas(size),
    roughness: makeCanvas(size),
    height: makeCanvas(size),
    normal: makeCanvas(size),
    ao: makeCanvas(size),
  };
  const contexts = {
    albedo: canvases.albedo.getContext('2d'),
    roughness: canvases.roughness.getContext('2d'),
    height: canvases.height.getContext('2d'),
    normal: canvases.normal.getContext('2d'),
    ao: canvases.ao.getContext('2d'),
  };
  if (!contexts.albedo || !contexts.roughness || !contexts.height || !contexts.normal || !contexts.ao) return null;
  const images = {
    albedo: contexts.albedo.createImageData(size, size),
    roughness: contexts.roughness.createImageData(size, size),
    height: contexts.height.createImageData(size, size),
    normal: contexts.normal.createImageData(size, size),
    ao: contexts.ao.createImageData(size, size),
  };
  const seed = hashString(id);
  const bands = surfaceBands(spec);
  const heightField = new Float32Array(size * size);
  const roughnessField = new Float32Array(size * size);
  const palette = materialPalette(spec);
  const fallback = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  const colors = (palette.length >= 2 ? palette : [fallback, '#6E614B', '#A08F70']).map(hexToRgb);
  const baseRoughness = clamp01(readLayerNumber(spec.roughness, ['base'], 0.76));
  const roughnessVariation = clamp01(readLayerNumber(spec.roughness, ['variation'], 0.18));
  const colorAmplitude = clamp01(readLayerNumber(spec.colorVariation, ['amplitude', 'variation'], 0.18));
  const heightCorrelation = clamp01(readLayerNumber(spec.colorVariation, ['heightCorrelation'], 0.3));
  const colorGradient: ColorGradientSpec | undefined = spec.colorGradient;
  for (let y = 0; y < size; y += 1) {
    const v = y / size;
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const index = y * size + x;
      const height = sampleSurface(u, v, bands, seed + 101);
      const roughNoise = sampleSurface(u, v, bands, seed + 7001);
      const colorNoise = sampleSurface(u, v, bands, seed + 15013);
      heightField[index] = height;
      roughnessField[index] = clamp01(baseRoughness + (roughNoise - 0.5) * roughnessVariation * 2);
      let color: [number, number, number];
      if (colorGradient) {
        // Evidence-derived spatial gradient (Plan 1.3 Workstream C) takes priority
        // over the noise-based palette blend below — it is a measured trend, not a guess.
        color = sampleColorGradient(colorGradient, u, v);
      } else {
        const paletteValue = clamp01(
          0.5 + (colorNoise - 0.5) * colorAmplitude * 2 + (height - 0.5) * heightCorrelation
        );
        color = mixPalette(colors, paletteValue);
      }
      writePixel(images.albedo.data, index * 4, color[0], color[1], color[2]);
    }
  }
  const normalStrength = Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35));
  const aoStrength = clamp01(readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35));
  for (let y = 0; y < size; y += 1) {
    const up = ((y - 1 + size) % size) * size;
    const down = ((y + 1) % size) * size;
    for (let x = 0; x < size; x += 1) {
      const left = (x - 1 + size) % size;
      const right = (x + 1) % size;
      const index = y * size + x;
      const center = heightField[index];
      const dx = (heightField[y * size + right] - heightField[y * size + left]) * normalStrength * 6;
      const dy = (heightField[down + x] - heightField[up + x]) * normalStrength * 6;
      const inverseLength = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const normalX = -dx * inverseLength;
      const normalY = -dy * inverseLength;
      const normalZ = inverseLength;
      const neighborAverage = (
        heightField[y * size + left] + heightField[y * size + right]
        + heightField[up + x] + heightField[down + x]
      ) * 0.25;
      const cavity = Math.max(0, neighborAverage - center);
      const ao = clamp01(1 - aoStrength * (cavity * 12 + (1 - center) * 0.16));
      const offset = index * 4;
      const heightByte = center * 255;
      const roughnessByte = roughnessField[index] * 255;
      writePixel(images.height.data, offset, heightByte, heightByte, heightByte);
      writePixel(images.roughness.data, offset, roughnessByte, roughnessByte, roughnessByte);
      writePixel(
        images.normal.data, offset,
        (normalX * 0.5 + 0.5) * 255,
        (normalY * 0.5 + 0.5) * 255,
        (normalZ * 0.5 + 0.5) * 255,
      );
      writePixel(images.ao.data, offset, ao * 255, ao * 255, ao * 255);
    }
  }
  contexts.albedo.putImageData(images.albedo, 0, 0);
  contexts.roughness.putImageData(images.roughness, 0, 0);
  contexts.height.putImageData(images.height, 0, 0);
  contexts.normal.putImageData(images.normal, 0, 0);
  contexts.ao.putImageData(images.ao, 0, 0);
  return {
    albedo: createMapTexture(canvases.albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createMapTexture(canvases.roughness, THREE.NoColorSpace, spec, options),
    height: createMapTexture(canvases.height, THREE.NoColorSpace, spec, options),
    normal: createMapTexture(canvases.normal, THREE.NoColorSpace, spec, options),
    ao: createMapTexture(canvases.ao, THREE.NoColorSpace, spec, options),
    source: 'procedural',
  };
}

function createSculptMaterial(id: string, spec: SculptMaterialSpec, options: ProceduralModelOptions, denseComponent = false): THREE.MeshPhysicalMaterial {
  // A material that declares -- with evidence -- that its subject carries no texture
  // detail gets NO texture set. Synthesising one anyway is not a harmless default: the
  // branch below then forces color to white and roughness to 1 and reads both from the
  // generated maps, so the authored albedo and the reference-derived roughness are both
  // discarded, and the model gains mottling the reference does not have. Measured on the
  // tuxedo cat, whose black fur rendered as speckled grey-and-white from a palette that
  // only ever described two flat regions.
  const textureless = (spec.textureless as { declared?: boolean } | undefined)?.declared === true;
  const textures = textureless
    ? null
    : makeReferenceTextureSet(spec, options) ?? makeProceduralTextureSet(id, spec, options);
  const material = new THREE.MeshPhysicalMaterial({
    color: textures ? 0xffffff : clampedAlbedoColor(spec),
    roughness: textures ? 1 : clamp01(readLayerNumber(spec.roughness, ['base'], 0.76)),
    metalness: clampPbrMetalness(readLayerNumber(spec.metalness, ['base'], 0.0)),
    clearcoat: clamp01(readLayerNumber(spec.clearcoat, ['base', 'amount'], 0)),
    clearcoatRoughness: clamp01(readLayerNumber(spec.clearcoatRoughness, ['base'], 0.25)),
    transmission: clamp01(readLayerNumber(spec.transmission, ['base', 'amount'], 0)),
    ior: clampPbrIor(readLayerNumber(spec.ior, ['base', 'value'], 1.5)),
    thickness: Math.max(0, readLayerNumber(spec.thickness, ['base', 'amount'], 0)),
    attenuationDistance: Math.max(0.001, readLayerNumber(spec.attenuationDistance, ['base', 'value'], Infinity)),
    attenuationColor: new THREE.Color(typeof spec.attenuationColor === 'string' ? spec.attenuationColor : '#ffffff'),
    sheen: clamp01(readLayerNumber(spec.sheen, ['base', 'amount'], 0)),
    sheenColor: new THREE.Color(typeof spec.sheenColor === 'string' ? spec.sheenColor : '#ffffff'),
    sheenRoughness: clamp01(readLayerNumber(spec.sheenRoughness, ['base'], 1.0)),
    iridescence: clamp01(readLayerNumber(spec.iridescence, ['base', 'amount'], 0)),
    iridescenceIOR: clampPbrIor(readLayerNumber(spec.iridescenceIOR, ['base', 'value'], 1.3)),
    anisotropy: clamp01(readLayerNumber(spec.anisotropy, ['base', 'amount'], 0)),
    anisotropyRotation: readLayerNumber(spec.anisotropy, ['rotation'], 0),
    specularIntensity: clampPbrF0(readLayerNumber(spec.specularF0 ?? spec.f0 ?? spec.specularIntensity, ['base', 'value'], 1.0)),
    specularColor: new THREE.Color(typeof spec.specularColor === 'string' ? spec.specularColor : '#ffffff'),
    emissive: new THREE.Color(typeof spec.emissive === 'string' ? spec.emissive : '#000000'),
    emissiveIntensity: Math.max(0, readLayerNumber(spec.emissiveIntensity, ['base'], 1.0)),
    opacity: clamp01(readLayerNumber(spec.opacity, ['base'], 1)),
    transparent: readLayerNumber(spec.transmission, ['base', 'amount'], 0) > 0 || readLayerNumber(spec.opacity, ['base'], 1) < 1,
    alphaTest: Math.max(0, readLayerNumber(spec.alpha, ['cutoff', 'alphaTest'], 0)),
    wireframe: options.wireframe ?? false,
    side: spec.doubleSided === true ? THREE.DoubleSide : THREE.FrontSide,
    flatShading: spec.flatShading === true,
  });
  if (textures) {
    material.map = textures.albedo;
    material.roughnessMap = textures.roughness;
    material.normalMap = textures.normal;
    material.normalScale.setScalar(Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35)));
    material.aoMap = textures.ao;
    material.aoMap.channel = 0;
    material.aoMapIntensity = readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35);
    const denseMesh = denseComponent || spec.denseMesh === true || spec.geometryDensity === 'dense' || spec.topologyClass === 'dense';
    const bumpScale = Math.max(0, readLayerNumber(spec.bump, ['amplitude', 'strength'], 0));
    const effectiveBumpScale = denseMesh ? Math.max(0.05, bumpScale) : bumpScale;
    if (effectiveBumpScale > 0) {
      material.bumpMap = textures.height;
      material.bumpScale = effectiveBumpScale;
    }
    const displacementScale = Math.max(0, readLayerNumber(spec.displacement, ['amplitude', 'strength'], 0));
    const effectiveDisplacementScale = denseMesh ? Math.max(0.005, displacementScale) : displacementScale;
    if (effectiveDisplacementScale > 0) {
      material.displacementMap = textures.height;
      material.displacementScale = effectiveDisplacementScale;
      material.displacementBias = -effectiveDisplacementScale * 0.5;
    }
  }
  material.envMapIntensity = readLayerNumber(spec, ['envMapIntensity'], 0.8);
  material.userData.sculptMaterial = spec;
  material.userData.proceduralMapsIndependent = true;
  material.userData.pbrConstraints = { albedoRange: [30, 240], binaryMetalness: true, f0Range: [0.02, 1], iorRange: [1, 2.5] };
  material.userData.pbrTextureSource = textures?.source ?? 'flat-fallback';
  material.userData.referencePbr = spec.referencePbr ?? null;
  material.userData.referenceMaterialId = spec.referenceMaterialId ?? spec.materialReference?.profileId ?? null;
  material.userData.materialEvidence = spec.materialEvidence ?? null;
  material.userData.validationViews = spec.materialReference?.validationViews ?? [];
  material.needsUpdate = true;
  return material;
}

type AttachmentEndpoint = {
  start: THREE.Vector3;
  midpoint: THREE.Vector3;
  quaternion: THREE.Quaternion;
  length: number;
  baseRadius: number;
  endRadius: number;
};

function readVector3(value: unknown, fallback: [number, number, number]): THREE.Vector3 {
  if (Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === 'number')) {
    return new THREE.Vector3(value[0], value[1], value[2]);
  }
  return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function makeAttachmentEndpoint(attachment: unknown): AttachmentEndpoint | null {
  if (!attachment || typeof attachment !== 'object') return null;
  const record = attachment as Record<string, unknown>;
  const start = readVector3(record.localStart, [0, 0, 0]);
  const end = readVector3(record.localEnd, [0, 1, 0]);
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 0.0001) return null;
  const direction = delta.clone().normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  const baseRadius = Math.max(0.005, readNumber(record.baseRadius, 0.06));
  const endRadius = Math.max(0.003, readNumber(record.endRadius, baseRadius * 0.55));
  return {
    start,
    midpoint: delta.multiplyScalar(0.5),
    quaternion,
    length,
    baseRadius,
    endRadius,
  };
}

// Generated from ObjectSculptSpec target: Blue Tulip
// Sculpt build pass: optimization-pass
// This factory is intentionally pass-gated. Finish browser screenshot review before unlocking deeper passes.
export function createBlueTulipModel(options: ProceduralModelOptions = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = "Blue Tulip";
  root.userData.reconstructionEvidence = {"itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": {"solved": false, "fovDegrees": 40.0, "aspect": 0.488, "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, "positionHint": [0.0, -0.4, 4.2], "note": "Orthographic-like frontal view; ref.png 488x1000, unit=bloom height 321px; camera solving skipped (no likeness/projection route)."}, "approximationNotes": []};
  root.userData.materialPipeline = {};
  root.userData.materialReferenceRegistry = null;

  const materialMap: Record<string, THREE.Material> = {};
  materialMap["petal-blue"] = createSculptMaterial(
    "petal-blue",
    {"id": "petal-blue", "name": "Tepal (matte-satin velvet)", "type": "physical", "shaderModel": "MeshPhysicalMaterial (sheen, metalness 0)", "baseColor": "#8799C0", "color": "#8799C0", "albedo": {"dominant": "#8296C3", "secondary": ["#BDCFEA", "#63729F", "#9AAFD8"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights."}, "colorVariation": {"palette": ["#8296C3", "#BDCFEA", "#63729F", "#9AAFD8", "#373F5E"], "pattern": "reference-derived pixel palette", "amplitude": 0.212, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.457, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.27, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.124, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.58, "variation": 0.05, "localResponse": "reference-derived roughness estimate; cavities and textured zones trend rougher, bright highlights trend smoother", "map": {"path": "evidence/pbr-petal-blue/petal-blue_roughness.png", "url": "/tulip-maps/petal-blue_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction (extract_pbr_evidence.py, this run)"}}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.19, "heightSource": {"path": "/home/claude/tulip/evidence/pbr-petal-blue/petal-blue_height.png", "url": "petal-blue_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.013}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.38, "contactShadowBias": 0.35, "notes": "Reference-derived cavity estimate from local height minima; verify against grazing-light screenshot."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "petal-crease-dark", "region": "lower bloom overlap seams Y 0..0.4", "effect": "albedo toward #505A81, roughness +0.1", "kind": "seam"}, {"id": "petal-ridge-gloss", "region": "front tepal central ridge", "effect": "roughness 0.5, sheen 0.3", "kind": "stain"}, {"id": "petal-top-light", "region": "upper tepal shoulders", "effect": "albedo toward #C2D3EC", "kind": "stain"}, {"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}, {"id": "surface-venation", "region": "tepal outer surface v 0.1-0.9", "effect": "fine longitudinal veins (17 cycles across, cubed) + broad streaks (5 cycles); albedo -7%/-3.5%", "implementation": "tulipRefinement.ts makeVeinTexture(): code-generated 256x512 CanvasTexture of 46 wavy longitudinal veins on white, UV-mapped (u across, v base->tip), multiplies vertex-colour gradient. Vertex grid (14 across) too coarse to carry veins."}, {"id": "overlap-seam-shadow", "region": "inner tepal edges |u|>0.55 under outer tepals", "effect": "albedo -38% to read the dark seams between tepals", "implementation": "tulipRefinement.ts vertex colour"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Albedo mean RGB(135,154,192); lit p90 (194,211,236); crease shadow p10 (80,90,129). Matte-satin dielectric with faint sheen. | Botanical tepal: non-metallic dielectric, satin/velvety (sheen lobe), thin rims slightly translucent (approximated with lighter rim vertex colour + sheen, no transmission pass for runtime cost).", "sheen": {"amount": 0.65, "color": "#C9D8FF", "roughness": 0.5}, "referencePbr": {"version": "1.0", "sourceImage": "/home/kinso/img2threejs-blue-tulip/evidence/crops/petal.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.816, "estimatedFidelity": 0.816, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "diagnostics": {"sourceWidth": 205, "sourceHeight": 270, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 205, "height": 270}, "mask": {"backgroundColor": "#ECECEC", "backgroundNoise": 139.743, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.926}, "mapStats": {"valueRange": 0.5051, "heightP90Gradient": 0.02862, "roughnessBase": 0.7, "roughnessVariation": 0.05, "normalStrength": 0.19, "blurRadius": 21}, "palette": ["#8296C3", "#BDCFEA", "#63729F", "#9AAFD8", "#373F5E"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "evidenceReport": "evidence/pbr-petal-blue.json", "staleMapsRemoved": ["albedo", "ao", "height", "normal", "roughness"], "maps": {"albedo": {"path": "evidence/pbr-petal-blue/petal-blue_albedo.png", "url": "/tulip-maps/petal-blue_albedo.png", "channel": "albedo", "note": "128px web copy of this run extract_pbr_evidence output"}, "roughness": {"path": "evidence/pbr-petal-blue/petal-blue_roughness.png", "url": "/tulip-maps/petal-blue_roughness.png", "channel": "roughness", "note": "128px web copy of this run extract_pbr_evidence output"}, "height": {"path": "evidence/pbr-petal-blue/petal-blue_height.png", "url": "/tulip-maps/petal-blue_height.png", "channel": "height", "note": "128px web copy of this run extract_pbr_evidence output"}, "normal": {"path": "evidence/pbr-petal-blue/petal-blue_normal.png", "url": "/tulip-maps/petal-blue_normal.png", "channel": "normal", "note": "128px web copy of this run extract_pbr_evidence output"}, "ao": {"path": "evidence/pbr-petal-blue/petal-blue_ao.png", "url": "/tulip-maps/petal-blue_ao.png", "channel": "ao", "note": "128px web copy of this run extract_pbr_evidence output"}}}, "finishClass": "plastic", "texturePalette": ["#A2B7DA", "#A5B7D9", "#9DB0D6", "#8697C3", "#5F6C97"], "proceduralTexture": "vertex-gradient", "clearcoat": {"base": 0.0, "variation": 0.0}, "clearcoatRoughness": {"base": 0.05, "variation": 0.0}, "transmission": {"base": 0.0, "variation": 0.0}, "ior": {"base": 1.5, "value": 1.5}, "envMapIntensity": 1.0, "finishOverride": {"reason": "analyze_texture.py classified smooth botanical crops as painted-metal (clearcoat 1.0); overridden by agent vision: botanical dielectric", "botanicalFinish": true}, "calibratedBaseColor": {"value": "#B4C6F4", "method": "material-pass region-mean match vs ref crops (tools/regionstats.py); multiplies reference-derived vertex-colour gradient", "pass": "material-pass"}},
    options
  );
  materialMap["stem-green"] = createSculptMaterial(
    "stem-green",
    {"id": "stem-green", "name": "Stem (satin yellow-green)", "type": "physical", "shaderModel": "MeshPhysicalMaterial (sheen, metalness 0)", "baseColor": "#7F903A", "color": "#7F903A", "albedo": {"dominant": "#5E6E0F", "secondary": ["#A8BA71", "#809418", "#91A351"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights."}, "colorVariation": {"palette": ["#5E6E0F", "#A8BA71", "#809418", "#91A351", "#3D4A0A"], "pattern": "reference-derived pixel palette", "amplitude": 0.204, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.45, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.244, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.108, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.42, "variation": 0.05, "localResponse": "reference-derived roughness estimate; cavities and textured zones trend rougher, bright highlights trend smoother", "map": {"path": "evidence/pbr-stem-green/stem-green_roughness.png", "url": "/tulip-maps/stem-green_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction (extract_pbr_evidence.py, this run)"}}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.182, "heightSource": {"path": "/home/claude/tulip/evidence/pbr-stem-green/stem-green_height.png", "url": "stem-green_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.38, "contactShadowBias": 0.35, "notes": "Reference-derived cavity estimate from local height minima; verify against grazing-light screenshot."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "stem-lit-side", "region": "stem lit side", "effect": "roughness 0.45, albedo +10%", "kind": "stain"}, {"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "RGB(127,144,58) mean; p10 (72,86,5); p90 (175,192,122). | Waxy cuticle stem: dielectric, low-mid roughness, faint clearcoat-like cuticle highlight.", "referencePbr": {"version": "1.0", "sourceImage": "/home/kinso/img2threejs-blue-tulip/evidence/crops/stem.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.749, "estimatedFidelity": 0.749, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "diagnostics": {"sourceWidth": 32, "sourceHeight": 410, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 32, "height": 410}, "mask": {"backgroundColor": "#909F7E", "backgroundNoise": 109.704, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.9587}, "mapStats": {"valueRange": 0.4854, "heightP90Gradient": 0.02229, "roughnessBase": 0.693, "roughnessVariation": 0.05, "normalStrength": 0.182, "blurRadius": 21}, "palette": ["#5E6E0F", "#A8BA71", "#809418", "#91A351", "#3D4A0A"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "evidenceReport": "evidence/pbr-stem-green.json", "staleMapsRemoved": ["albedo", "ao", "height", "normal", "roughness"], "maps": {"albedo": {"path": "evidence/pbr-stem-green/stem-green_albedo.png", "url": "/tulip-maps/stem-green_albedo.png", "channel": "albedo", "note": "128px web copy of this run extract_pbr_evidence output"}, "roughness": {"path": "evidence/pbr-stem-green/stem-green_roughness.png", "url": "/tulip-maps/stem-green_roughness.png", "channel": "roughness", "note": "128px web copy of this run extract_pbr_evidence output"}, "height": {"path": "evidence/pbr-stem-green/stem-green_height.png", "url": "/tulip-maps/stem-green_height.png", "channel": "height", "note": "128px web copy of this run extract_pbr_evidence output"}, "normal": {"path": "evidence/pbr-stem-green/stem-green_normal.png", "url": "/tulip-maps/stem-green_normal.png", "channel": "normal", "note": "128px web copy of this run extract_pbr_evidence output"}, "ao": {"path": "evidence/pbr-stem-green/stem-green_ao.png", "url": "/tulip-maps/stem-green_ao.png", "channel": "ao", "note": "128px web copy of this run extract_pbr_evidence output"}}}, "colorGradient": {"type": "radial", "axis": [0.5035005091649695, 0.5037268047069473], "stops": [{"offset": 0.0, "color": "rgba(145, 164, 36, 1.0)"}, {"offset": 0.55, "color": "rgba(95, 118, 2, 1.0)"}, {"offset": 1.0, "color": "rgba(100, 112, 76, 1.0)"}], "confidence": 1.0}, "finishClass": "plastic", "texturePalette": ["#839246", "#9BAC4F", "#8BA026", "#6E820B", "#546410"], "proceduralTexture": "vertex-gradient", "clearcoat": {"base": 0.12, "variation": 0.0}, "clearcoatRoughness": {"base": 0.05, "variation": 0.0}, "transmission": {"base": 0.0, "variation": 0.0}, "ior": {"base": 1.5, "value": 1.5}, "envMapIntensity": 1.0, "finishOverride": {"reason": "analyze_texture.py classified smooth botanical crops as painted-metal (clearcoat 1.0); overridden by agent vision: botanical dielectric", "botanicalFinish": true}, "sheen": {"amount": 0.2, "color": "#D8E8A8", "roughness": 0.6}, "calibratedBaseColor": {"value": "#9DB32E", "method": "material-pass region-mean match vs ref crops (tools/regionstats.py); multiplies reference-derived vertex-colour gradient", "pass": "material-pass"}},
    options
  );
  materialMap["leaf-green"] = createSculptMaterial(
    "leaf-green",
    {"id": "leaf-green", "name": "Leaf (matte glaucous grey-green)", "type": "physical", "shaderModel": "MeshPhysicalMaterial (sheen, metalness 0)", "baseColor": "#697748", "color": "#697748", "albedo": {"dominant": "#616E2D", "secondary": ["#54611D", "#46540B", "#9DB28E"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights."}, "colorVariation": {"palette": ["#616E2D", "#54611D", "#46540B", "#9DB28E", "#727F43"], "pattern": "reference-derived pixel palette", "amplitude": 0.172, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.424, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.262, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.119, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.64, "variation": 0.05, "localResponse": "reference-derived roughness estimate; cavities and textured zones trend rougher, bright highlights trend smoother", "map": {"path": "evidence/pbr-leaf-green/leaf-green_roughness.png", "url": "/tulip-maps/leaf-green_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction (extract_pbr_evidence.py, this run)"}}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.187, "heightSource": {"path": "/home/claude/tulip/evidence/pbr-leaf-green/leaf-green_height.png", "url": "leaf-green_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.012}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.38, "contactShadowBias": 0.35, "notes": "Reference-derived cavity estimate from local height minima; verify against grazing-light screenshot."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "leaf-margin-light", "region": "leaf margin band", "effect": "albedo toward #A3B89A, roughness 0.75", "kind": "stain"}, {"id": "leaf-midrib-line", "region": "leaf midrib", "effect": "albedo +8%", "kind": "linework"}, {"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "RGB(105,119,72) mean; margin up to (163,184,154). | Glaucous leaf: dielectric with matte waxy bloom read as a grey-blue sheen lobe; lighter margins and midrib.", "referencePbr": {"version": "1.0", "sourceImage": "/home/kinso/img2threejs-blue-tulip/evidence/crops/leaf.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.847, "estimatedFidelity": 0.847, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "diagnostics": {"sourceWidth": 140, "sourceHeight": 310, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 140, "height": 310}, "mask": {"backgroundColor": "#FFFFFF", "backgroundNoise": 166.006, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.4461}, "mapStats": {"valueRange": 0.4102, "heightP90Gradient": 0.02664, "roughnessBase": 0.683, "roughnessVariation": 0.05, "normalStrength": 0.187, "blurRadius": 21}, "palette": ["#616E2D", "#54611D", "#46540B", "#9DB28E", "#727F43"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "evidenceReport": "evidence/pbr-leaf-green.json", "staleMapsRemoved": ["albedo", "ao", "height", "normal", "roughness"], "maps": {"albedo": {"path": "evidence/pbr-leaf-green/leaf-green_albedo.png", "url": "/tulip-maps/leaf-green_albedo.png", "channel": "albedo", "note": "128px web copy of this run extract_pbr_evidence output"}, "roughness": {"path": "evidence/pbr-leaf-green/leaf-green_roughness.png", "url": "/tulip-maps/leaf-green_roughness.png", "channel": "roughness", "note": "128px web copy of this run extract_pbr_evidence output"}, "height": {"path": "evidence/pbr-leaf-green/leaf-green_height.png", "url": "/tulip-maps/leaf-green_height.png", "channel": "height", "note": "128px web copy of this run extract_pbr_evidence output"}, "normal": {"path": "evidence/pbr-leaf-green/leaf-green_normal.png", "url": "/tulip-maps/leaf-green_normal.png", "channel": "normal", "note": "128px web copy of this run extract_pbr_evidence output"}, "ao": {"path": "evidence/pbr-leaf-green/leaf-green_ao.png", "url": "/tulip-maps/leaf-green_ao.png", "channel": "ao", "note": "128px web copy of this run extract_pbr_evidence output"}}}, "finishClass": "plastic", "texturePalette": ["#B9C1AB", "#919D7D", "#65744B", "#6A7753", "#848D72"], "proceduralTexture": "vertex-gradient", "clearcoat": {"base": 0.0, "variation": 0.0}, "clearcoatRoughness": {"base": 0.05, "variation": 0.0}, "transmission": {"base": 0.0, "variation": 0.0}, "ior": {"base": 1.5, "value": 1.5}, "envMapIntensity": 1.0, "finishOverride": {"reason": "analyze_texture.py classified smooth botanical crops as painted-metal (clearcoat 1.0); overridden by agent vision: botanical dielectric", "botanicalFinish": true}, "sheen": {"amount": 0.45, "color": "#B7C9B0", "roughness": 0.7}, "calibratedBaseColor": {"value": "#737E45", "method": "material-pass region-mean match vs ref crops (tools/regionstats.py); multiplies reference-derived vertex-colour gradient", "pass": "material-pass"}},
    options
  );
  materialMap["crease-blue"] = createSculptMaterial(
    "crease-blue",
    {"id": "crease-blue", "name": "Tepal seam shadow", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#505A81", "color": "#505A81", "albedo": {"dominant": "#8296C3", "secondary": ["#BDCFEA", "#63729F", "#9AAFD8"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights."}, "colorVariation": {"palette": ["#8296C3", "#BDCFEA", "#63729F", "#9AAFD8", "#373F5E"], "pattern": "reference-derived pixel palette", "amplitude": 0.212, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.457, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.27, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.124, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.7, "variation": 0.05, "localResponse": "reference-derived roughness estimate; cavities and textured zones trend rougher, bright highlights trend smoother", "map": {"path": "evidence/pbr-petal-blue/petal-blue_roughness.png", "url": "/tulip-maps/petal-blue_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction (extract_pbr_evidence.py, this run)"}}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.19, "heightSource": {"path": "/home/claude/tulip/evidence/pbr-crease-blue/crease-blue_height.png", "url": "crease-blue_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.013}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.38, "contactShadowBias": 0.35, "notes": "Reference-derived cavity estimate from local height minima; verify against grazing-light screenshot."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Dark blue seam where tepals overlap.", "referencePbr": {"version": "1.0", "sourceImage": "/home/claude/tulip/evidence/crops/petal.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.829, "estimatedFidelity": 0.829, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "diagnostics": {"sourceWidth": 205, "sourceHeight": 270, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 205, "height": 270}, "mask": {"backgroundColor": "#ECECEC", "backgroundNoise": 139.743, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.926}, "mapStats": {"valueRange": 0.5051, "heightP90Gradient": 0.02862, "roughnessBase": 0.7, "roughnessVariation": 0.05, "normalStrength": 0.19, "blurRadius": 21}, "palette": ["#8296C3", "#BDCFEA", "#63729F", "#9AAFD8", "#373F5E"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "staleMapsRemoved": ["albedo", "ao", "height", "normal", "roughness"], "maps": {"albedo": {"path": "evidence/pbr-petal-blue/petal-blue_albedo.png", "url": "/tulip-maps/petal-blue_albedo.png", "channel": "albedo", "note": "128px web copy of this run extract_pbr_evidence output"}, "roughness": {"path": "evidence/pbr-petal-blue/petal-blue_roughness.png", "url": "/tulip-maps/petal-blue_roughness.png", "channel": "roughness", "note": "128px web copy of this run extract_pbr_evidence output"}, "height": {"path": "evidence/pbr-petal-blue/petal-blue_height.png", "url": "/tulip-maps/petal-blue_height.png", "channel": "height", "note": "128px web copy of this run extract_pbr_evidence output"}, "normal": {"path": "evidence/pbr-petal-blue/petal-blue_normal.png", "url": "/tulip-maps/petal-blue_normal.png", "channel": "normal", "note": "128px web copy of this run extract_pbr_evidence output"}, "ao": {"path": "evidence/pbr-petal-blue/petal-blue_ao.png", "url": "/tulip-maps/petal-blue_ao.png", "channel": "ao", "note": "128px web copy of this run extract_pbr_evidence output"}}}},
    options
  );
  materialMap["leaf-midrib"] = createSculptMaterial(
    "leaf-midrib",
    {"id": "leaf-midrib", "name": "Leaf midrib", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#8A9A66", "color": "#8A9A66", "albedo": {"dominant": "#76875B", "secondary": ["#485530", "#55603E", "#65754B"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights."}, "colorVariation": {"palette": ["#76875B", "#485530", "#55603E", "#65754B", "#99A57D"], "pattern": "reference-derived pixel palette", "amplitude": 0.108, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.37, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.282, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.13, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.684, "variation": 0.065, "localResponse": "reference-derived roughness estimate; cavities and textured zones trend rougher, bright highlights trend smoother", "map": {"path": "evidence/pbr-leaf-green/leaf-green_roughness.png", "url": "/tulip-maps/leaf-green_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction (extract_pbr_evidence.py, this run)"}}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.193, "heightSource": {"path": "/home/claude/tulip/evidence/pbr-leaf-midrib/leaf-midrib_height.png", "url": "leaf-midrib_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.014}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.38, "contactShadowBias": 0.35, "notes": "Reference-derived cavity estimate from local height minima; verify against grazing-light screenshot."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Lighter midrib line.", "referencePbr": {"version": "1.0", "sourceImage": "/home/claude/tulip/evidence/crops/leaf_right.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.854, "estimatedFidelity": 0.854, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "diagnostics": {"sourceWidth": 105, "sourceHeight": 390, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 105, "height": 390}, "mask": {"backgroundColor": "#EEEEEE", "backgroundNoise": 259.673, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.4578}, "mapStats": {"valueRange": 0.258, "heightP90Gradient": 0.03144, "roughnessBase": 0.684, "roughnessVariation": 0.065, "normalStrength": 0.193, "blurRadius": 21}, "palette": ["#76875B", "#485530", "#55603E", "#65754B", "#99A57D"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "staleMapsRemoved": ["albedo", "ao", "height", "normal", "roughness"], "maps": {"albedo": {"path": "evidence/pbr-leaf-green/leaf-green_albedo.png", "url": "/tulip-maps/leaf-green_albedo.png", "channel": "albedo", "note": "128px web copy of this run extract_pbr_evidence output"}, "roughness": {"path": "evidence/pbr-leaf-green/leaf-green_roughness.png", "url": "/tulip-maps/leaf-green_roughness.png", "channel": "roughness", "note": "128px web copy of this run extract_pbr_evidence output"}, "height": {"path": "evidence/pbr-leaf-green/leaf-green_height.png", "url": "/tulip-maps/leaf-green_height.png", "channel": "height", "note": "128px web copy of this run extract_pbr_evidence output"}, "normal": {"path": "evidence/pbr-leaf-green/leaf-green_normal.png", "url": "/tulip-maps/leaf-green_normal.png", "channel": "normal", "note": "128px web copy of this run extract_pbr_evidence output"}, "ao": {"path": "evidence/pbr-leaf-green/leaf-green_ao.png", "url": "/tulip-maps/leaf-green_ao.png", "channel": "ao", "note": "128px web copy of this run extract_pbr_evidence output"}}}},
    options
  );

  const nodes: Record<string, THREE.Object3D> = { root };
  const meshes: Record<string, THREE.Mesh> = {};
  const sockets: Record<string, THREE.Object3D> = {};
  const colliders: Record<string, unknown> = {};
  const destructionGroups: Record<string, THREE.Object3D[]> = {};

  const attachment_root_0 = null;
  const endpoint_root_0 = makeAttachmentEndpoint(attachment_root_0);
  const node_root_0 = new THREE.Group();
  node_root_0.name = "Stem__pivot";
  node_root_0.scale.set(1, 1, 1);
  if (endpoint_root_0) {
    node_root_0.position.copy(endpoint_root_0.start);
    node_root_0.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_root_0.position.set(0.0, 0.0, 0.0);
    node_root_0.rotation.set(0.0, 0.0, 0.0);
  }
  node_root_0.userData.sculptComponent = {"id": "root", "name": "Stem", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Long thin path-following cylinder: fiber-strand -> tube (surface_topology decision tree step 2).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.05, 0.0], [0.004, -0.5, 0.0], [0.02, -1.0, 0.0], [0.05, -1.5, 0.0], [0.085, -1.95, 0.0]], "radius": 0.045, "radialSegments": 16, "closed": false}}, "parent": null, "attachment": null, "dimensions": {"width": 0.12, "height": 2.0, "depth": 0.12, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}}, "material": "stem-green", "materialLayers": ["stem-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "stem-sheen", "kind": "stain", "description": "Satin sheen band along the lit side of the stem, roughness ~0.5", "region": "stem full length"}], "surfaceDetail": {"macroRoughness": 0.42, "microRoughness": 0.45, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12)", "displacementPattern": "none", "occlusionPattern": "contact shadow under receptacle from shadow-casting key light", "edgeWearPattern": "none", "notes": "waxy cuticle sheen band along lit side"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "root", "dominantAlbedo": "rgba(128, 148, 31, 1.0)", "secondaryAlbedo": "rgba(76, 90, 20, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.257, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/stem.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.401}, "colorGradient": {"type": "radial", "axis": [0.5035005091649695, 0.5037268047069473], "stops": [{"offset": 0.0, "color": "rgba(145, 164, 36, 1.0)"}, {"offset": 0.55, "color": "rgba(95, 118, 2, 1.0)"}, {"offset": 1.0, "color": "rgba(100, 112, 76, 1.0)"}], "confidence": 1.0}}};
  node_root_0.userData.actionProfile = {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}};
  (nodes["root"] ?? root).add(node_root_0);
  nodes["root"] = node_root_0;
  const mesh_root_0Geometry = endpoint_root_0
    ? new THREE.CylinderGeometry(endpoint_root_0.endRadius, endpoint_root_0.baseRadius, endpoint_root_0.length, 16, 6)
    : buildTubeGeometry({"points": [[0.0, 0.05, 0.0], [0.004, -0.5, 0.0], [0.02, -1.0, 0.0], [0.05, -1.5, 0.0], [0.085, -1.95, 0.0]], "radius": 0.045, "radialSegments": 16, "closed": false});
  if (!endpoint_root_0) {
    mesh_root_0Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_root_0 = new THREE.Mesh(
    mesh_root_0Geometry,
    materialMap["stem-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_root_0.name = "Stem";
  if (endpoint_root_0) {
    mesh_root_0.position.copy(endpoint_root_0.midpoint);
    mesh_root_0.quaternion.copy(endpoint_root_0.quaternion);
  }
  mesh_root_0.castShadow = options.castShadow ?? true;
  mesh_root_0.receiveShadow = options.receiveShadow ?? true;
  mesh_root_0.userData.sculptComponent = {"id": "root", "name": "Stem", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Long thin path-following cylinder: fiber-strand -> tube (surface_topology decision tree step 2).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.05, 0.0], [0.004, -0.5, 0.0], [0.02, -1.0, 0.0], [0.05, -1.5, 0.0], [0.085, -1.95, 0.0]], "radius": 0.045, "radialSegments": 16, "closed": false}}, "parent": null, "attachment": null, "dimensions": {"width": 0.12, "height": 2.0, "depth": 0.12, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}}, "material": "stem-green", "materialLayers": ["stem-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "stem-sheen", "kind": "stain", "description": "Satin sheen band along the lit side of the stem, roughness ~0.5", "region": "stem full length"}], "surfaceDetail": {"macroRoughness": 0.42, "microRoughness": 0.45, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12)", "displacementPattern": "none", "occlusionPattern": "contact shadow under receptacle from shadow-casting key light", "edgeWearPattern": "none", "notes": "waxy cuticle sheen band along lit side"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "root", "dominantAlbedo": "rgba(128, 148, 31, 1.0)", "secondaryAlbedo": "rgba(76, 90, 20, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.257, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/stem.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.401}, "colorGradient": {"type": "radial", "axis": [0.5035005091649695, 0.5037268047069473], "stops": [{"offset": 0.0, "color": "rgba(145, 164, 36, 1.0)"}, {"offset": 0.55, "color": "rgba(95, 118, 2, 1.0)"}, {"offset": 1.0, "color": "rgba(100, 112, 76, 1.0)"}], "confidence": 1.0}}};
  node_root_0.add(mesh_root_0);
  meshes["root"] = mesh_root_0;
  colliders["root"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_root_0);

  const endpoint_bloom_1 = makeAttachmentEndpoint(null);
  const node_bloom_1 = new THREE.Group();
  node_bloom_1.name = "Bloom cup core__pivot";
  node_bloom_1.scale.set(1, 1, 1);
  if (endpoint_bloom_1) {
    node_bloom_1.position.copy(endpoint_bloom_1.start);
    node_bloom_1.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_bloom_1.position.set(-0.06, -0.035, 0.0);
    node_bloom_1.rotation.set(0.0, 0.0, 0.0);
  }
  node_bloom_1.userData.sculptComponent = {"id": "bloom", "name": "Bloom cup core", "level": "macro", "role": "body", "importance": 0.95, "confidence": 0.7, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Single continuous ovoid volume revolved about the stem axis (surface_topology: rotationally symmetric continuous-sculpt -> lathe). Hidden interior/back tepals are inferred from this volume.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0328, 0.0], [0.1484, 0.06], [0.2834, 0.13], [0.3173, 0.2], [0.344, 0.255], [0.3627, 0.32], [0.38, 0.38], [0.3879, 0.44], [0.3964, 0.505], [0.4105, 0.57], [0.4234, 0.63], [0.4246, 0.69], [0.4259, 0.754], [0.4063, 0.82], [0.389, 0.878], [0.308, 0.93], [0.2457, 0.97]], "segments": 40}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-top", "localStart": [-0.06, -0.035, 0.0], "localEnd": [-0.06, 0.165, 0.0], "contactType": "embed", "embedDepth": 0.05, "gapTolerance": 0.01, "baseRadius": 0.059, "endRadius": 0.3, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.04, "height": 1.0, "depth": 1.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.06, -0.035, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "whole-object", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "bloom", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "bloom-ovoid", "kind": "contour", "description": "Ovoid silhouette: 1.04 wide x 1.0 tall, widest at Y=0.75 (upper third), tapering to 0.35 wide at Y=0.13", "region": "bloom bbox x68-403 y51-372"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "bloom", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_bloom_1.userData.actionProfile = {"animationRole": "whole-object", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "bloom", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["root"] ?? root).add(node_bloom_1);
  nodes["bloom"] = node_bloom_1;
  const mesh_bloom_1Geometry = endpoint_bloom_1
    ? new THREE.CylinderGeometry(endpoint_bloom_1.endRadius, endpoint_bloom_1.baseRadius, endpoint_bloom_1.length, 16, 6)
    : buildLatheGeometry({"points": [[0.0328, 0.0], [0.1484, 0.06], [0.2834, 0.13], [0.3173, 0.2], [0.344, 0.255], [0.3627, 0.32], [0.38, 0.38], [0.3879, 0.44], [0.3964, 0.505], [0.4105, 0.57], [0.4234, 0.63], [0.4246, 0.69], [0.4259, 0.754], [0.4063, 0.82], [0.389, 0.878], [0.308, 0.93], [0.2457, 0.97]], "segments": 40});
  if (!endpoint_bloom_1) {
    mesh_bloom_1Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_bloom_1 = new THREE.Mesh(
    mesh_bloom_1Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bloom_1.name = "Bloom cup core";
  if (endpoint_bloom_1) {
    mesh_bloom_1.position.copy(endpoint_bloom_1.midpoint);
    mesh_bloom_1.quaternion.copy(endpoint_bloom_1.quaternion);
  }
  mesh_bloom_1.castShadow = options.castShadow ?? true;
  mesh_bloom_1.receiveShadow = options.receiveShadow ?? true;
  mesh_bloom_1.userData.sculptComponent = {"id": "bloom", "name": "Bloom cup core", "level": "macro", "role": "body", "importance": 0.95, "confidence": 0.7, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Single continuous ovoid volume revolved about the stem axis (surface_topology: rotationally symmetric continuous-sculpt -> lathe). Hidden interior/back tepals are inferred from this volume.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0328, 0.0], [0.1484, 0.06], [0.2834, 0.13], [0.3173, 0.2], [0.344, 0.255], [0.3627, 0.32], [0.38, 0.38], [0.3879, 0.44], [0.3964, 0.505], [0.4105, 0.57], [0.4234, 0.63], [0.4246, 0.69], [0.4259, 0.754], [0.4063, 0.82], [0.389, 0.878], [0.308, 0.93], [0.2457, 0.97]], "segments": 40}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-top", "localStart": [-0.06, -0.035, 0.0], "localEnd": [-0.06, 0.165, 0.0], "contactType": "embed", "embedDepth": 0.05, "gapTolerance": 0.01, "baseRadius": 0.059, "endRadius": 0.3, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.04, "height": 1.0, "depth": 1.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.06, -0.035, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "whole-object", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "bloom", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "bloom-ovoid", "kind": "contour", "description": "Ovoid silhouette: 1.04 wide x 1.0 tall, widest at Y=0.75 (upper third), tapering to 0.35 wide at Y=0.13", "region": "bloom bbox x68-403 y51-372"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "bloom", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_bloom_1.add(mesh_bloom_1);
  meshes["bloom"] = mesh_bloom_1;
  colliders["bloom"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["bloom"] ??= [];
  destructionGroups["bloom"].push(node_bloom_1);

  const endpoint_leaf_left_2 = makeAttachmentEndpoint(null);
  const node_leaf_left_2 = new THREE.Group();
  node_leaf_left_2.name = "Leaf (left)__pivot";
  node_leaf_left_2.scale.set(1, 1, 1);
  if (endpoint_leaf_left_2) {
    node_leaf_left_2.position.copy(endpoint_leaf_left_2.start);
    node_leaf_left_2.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_leaf_left_2.position.set(0.0, 0.0, 0.0);
    node_leaf_left_2.rotation.set(0.0, 0.0, 0.0);
  }
  node_leaf_left_2.userData.sculptComponent = {"id": "leaf-left", "name": "Leaf (left)", "level": "macro", "role": "body", "importance": 0.8, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Lanceolate blade whose section narrows to a point: tapered-sweep (a tube would read as a noodle).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0218, -1.4579, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [-0.0501, -1.2564, 0.0464], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.1226, -1.0544, 0.1131], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.1966, -0.8519, 0.1921], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.2804, -0.6518, 0.2826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.3753, -0.4547, 0.383], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.4741, -0.2582, 0.4897], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.5763, -0.0623, 0.6], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildFoldedLeaf", "reason": "Lanceolate tulip leaf is a channelled (V-folded) thin blade that clasps the stem; an elliptic tapered-sweep reads as a flat strip.", "rule": "Spine = taperedSweep station positions (Catmull-Rom); half-width = station.rz; V-fold lift 0.35*|u|*w; slight twist 0.25 rad base-to-tip; thickness 0.012; lighter margin and midrib vertex colours. Form pass: base wrap (fold +0.8 fading by v=0.22) clasps the stem; spine arches in depth (left +0.2z, right -0.12z at tip)."}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-left", "localStart": [0.0218, -1.4579, 0.0], "localEnd": [-0.5763, -0.0623, 0.1], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.015, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.7, "height": 1.5, "depth": 0.1, "units": "relative", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "leaf-left-fold", "kind": "ridge", "description": "Midrib fold: V-section along spine, +0.02 relief, lighter margin", "region": "leaf left"}, {"id": "leaf-left-margin", "kind": "stain", "description": "Lighter grey-green margin (163,184,154) vs blade (105,119,72)", "region": "leaf left edge"}], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "leaf-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_leaf_left_2.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}};
  (nodes["root"] ?? root).add(node_leaf_left_2);
  nodes["leaf-left"] = node_leaf_left_2;
  const mesh_leaf_left_2Geometry = endpoint_leaf_left_2
    ? new THREE.CylinderGeometry(endpoint_leaf_left_2.endRadius, endpoint_leaf_left_2.baseRadius, endpoint_leaf_left_2.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0218, -1.4579, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [-0.0501, -1.2564, 0.0464], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.1226, -1.0544, 0.1131], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.1966, -0.8519, 0.1921], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.2804, -0.6518, 0.2826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.3753, -0.4547, 0.383], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.4741, -0.2582, 0.4897], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.5763, -0.0623, 0.6], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true});
  if (!endpoint_leaf_left_2) {
    mesh_leaf_left_2Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_leaf_left_2 = new THREE.Mesh(
    mesh_leaf_left_2Geometry,
    materialMap["leaf-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_leaf_left_2.name = "Leaf (left)";
  if (endpoint_leaf_left_2) {
    mesh_leaf_left_2.position.copy(endpoint_leaf_left_2.midpoint);
    mesh_leaf_left_2.quaternion.copy(endpoint_leaf_left_2.quaternion);
  }
  mesh_leaf_left_2.castShadow = options.castShadow ?? true;
  mesh_leaf_left_2.receiveShadow = options.receiveShadow ?? true;
  mesh_leaf_left_2.userData.sculptComponent = {"id": "leaf-left", "name": "Leaf (left)", "level": "macro", "role": "body", "importance": 0.8, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Lanceolate blade whose section narrows to a point: tapered-sweep (a tube would read as a noodle).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0218, -1.4579, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [-0.0501, -1.2564, 0.0464], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.1226, -1.0544, 0.1131], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.1966, -0.8519, 0.1921], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.2804, -0.6518, 0.2826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [-0.3753, -0.4547, 0.383], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [-0.4741, -0.2582, 0.4897], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [-0.5763, -0.0623, 0.6], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildFoldedLeaf", "reason": "Lanceolate tulip leaf is a channelled (V-folded) thin blade that clasps the stem; an elliptic tapered-sweep reads as a flat strip.", "rule": "Spine = taperedSweep station positions (Catmull-Rom); half-width = station.rz; V-fold lift 0.35*|u|*w; slight twist 0.25 rad base-to-tip; thickness 0.012; lighter margin and midrib vertex colours. Form pass: base wrap (fold +0.8 fading by v=0.22) clasps the stem; spine arches in depth (left +0.2z, right -0.12z at tip)."}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-left", "localStart": [0.0218, -1.4579, 0.0], "localEnd": [-0.5763, -0.0623, 0.1], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.015, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.7, "height": 1.5, "depth": 0.1, "units": "relative", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "leaf-left-fold", "kind": "ridge", "description": "Midrib fold: V-section along spine, +0.02 relief, lighter margin", "region": "leaf left"}, {"id": "leaf-left-margin", "kind": "stain", "description": "Lighter grey-green margin (163,184,154) vs blade (105,119,72)", "region": "leaf left edge"}], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "leaf-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_leaf_left_2.add(mesh_leaf_left_2);
  meshes["leaf-left"] = mesh_leaf_left_2;
  colliders["leaf-left"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["leaf-left"] ??= [];
  destructionGroups["leaf-left"].push(node_leaf_left_2);

  const endpoint_leaf_right_3 = makeAttachmentEndpoint(null);
  const node_leaf_right_3 = new THREE.Group();
  node_leaf_right_3.name = "Leaf (right)__pivot";
  node_leaf_right_3.scale.set(1, 1, 1);
  if (endpoint_leaf_right_3) {
    node_leaf_right_3.position.copy(endpoint_leaf_right_3.start);
    node_leaf_right_3.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_leaf_right_3.position.set(0.0, 0.0, 0.0);
    node_leaf_right_3.rotation.set(0.0, 0.0, 0.0);
  }
  node_leaf_right_3.userData.sculptComponent = {"id": "leaf-right", "name": "Leaf (right)", "level": "macro", "role": "body", "importance": 0.8, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Lanceolate blade whose section narrows to a point: tapered-sweep (a tube would read as a noodle).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0935, -1.4735, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [0.1436, -1.2527, -0.0383], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.1937, -1.031, -0.1006], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.2394, -0.8067, -0.1826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.2969, -0.5838, -0.2753], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.3651, -0.3618, -0.3776], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.4355, -0.139, -0.4811], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.5078, 0.0841, -0.59], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildFoldedLeaf", "reason": "Lanceolate tulip leaf is a channelled (V-folded) thin blade that clasps the stem; an elliptic tapered-sweep reads as a flat strip.", "rule": "Spine = taperedSweep station positions (Catmull-Rom); half-width = station.rz; V-fold lift 0.35*|u|*w; slight twist 0.25 rad base-to-tip; thickness 0.012; lighter margin and midrib vertex colours. Form pass: base wrap (fold +0.8 fading by v=0.22) clasps the stem; spine arches in depth (left +0.2z, right -0.12z at tip)."}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-right", "localStart": [0.0935, -1.4735, 0.0], "localEnd": [0.5078, 0.0841, -0.09], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.015, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.7, "height": 1.5, "depth": 0.1, "units": "relative", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "leaf-right-fold", "kind": "ridge", "description": "Midrib fold: V-section along spine, +0.02 relief, lighter margin", "region": "leaf right"}, {"id": "leaf-right-margin", "kind": "stain", "description": "Lighter grey-green margin (163,184,154) vs blade (105,119,72)", "region": "leaf right edge"}], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "leaf-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_leaf_right_3.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}};
  (nodes["root"] ?? root).add(node_leaf_right_3);
  nodes["leaf-right"] = node_leaf_right_3;
  const mesh_leaf_right_3Geometry = endpoint_leaf_right_3
    ? new THREE.CylinderGeometry(endpoint_leaf_right_3.endRadius, endpoint_leaf_right_3.baseRadius, endpoint_leaf_right_3.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0935, -1.4735, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [0.1436, -1.2527, -0.0383], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.1937, -1.031, -0.1006], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.2394, -0.8067, -0.1826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.2969, -0.5838, -0.2753], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.3651, -0.3618, -0.3776], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.4355, -0.139, -0.4811], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.5078, 0.0841, -0.59], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true});
  if (!endpoint_leaf_right_3) {
    mesh_leaf_right_3Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_leaf_right_3 = new THREE.Mesh(
    mesh_leaf_right_3Geometry,
    materialMap["leaf-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_leaf_right_3.name = "Leaf (right)";
  if (endpoint_leaf_right_3) {
    mesh_leaf_right_3.position.copy(endpoint_leaf_right_3.midpoint);
    mesh_leaf_right_3.quaternion.copy(endpoint_leaf_right_3.quaternion);
  }
  mesh_leaf_right_3.castShadow = options.castShadow ?? true;
  mesh_leaf_right_3.receiveShadow = options.receiveShadow ?? true;
  mesh_leaf_right_3.userData.sculptComponent = {"id": "leaf-right", "name": "Leaf (right)", "level": "macro", "role": "body", "importance": 0.8, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Lanceolate blade whose section narrows to a point: tapered-sweep (a tube would read as a noodle).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0935, -1.4735, 0.0], "rx": 0.012, "rz": 0.024, "twist": 0.0}, {"position": [0.1436, -1.2527, -0.0383], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.1937, -1.031, -0.1006], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.2394, -0.8067, -0.1826], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.2969, -0.5838, -0.2753], "rx": 0.012, "rz": 0.1181, "twist": 0.0}, {"position": [0.3651, -0.3618, -0.3776], "rx": 0.012, "rz": 0.103, "twist": 0.0}, {"position": [0.4355, -0.139, -0.4811], "rx": 0.012, "rz": 0.0744, "twist": 0.0}, {"position": [0.5078, 0.0841, -0.59], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 12, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildFoldedLeaf", "reason": "Lanceolate tulip leaf is a channelled (V-folded) thin blade that clasps the stem; an elliptic tapered-sweep reads as a flat strip.", "rule": "Spine = taperedSweep station positions (Catmull-Rom); half-width = station.rz; V-fold lift 0.35*|u|*w; slight twist 0.25 rad base-to-tip; thickness 0.012; lighter margin and midrib vertex colours. Form pass: base wrap (fold +0.8 fading by v=0.22) clasps the stem; spine arches in depth (left +0.2z, right -0.12z at tip)."}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-right", "localStart": [0.0935, -1.4735, 0.0], "localEnd": [0.5078, 0.0841, -0.09], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.015, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.7, "height": 1.5, "depth": 0.1, "units": "relative", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "leaf-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "leaf-right-fold", "kind": "ridge", "description": "Midrib fold: V-section along spine, +0.02 relief, lighter margin", "region": "leaf right"}, {"id": "leaf-right-margin", "kind": "stain", "description": "Lighter grey-green margin (163,184,154) vs blade (105,119,72)", "region": "leaf right edge"}], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "leaf-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_leaf_right_3.add(mesh_leaf_right_3);
  meshes["leaf-right"] = mesh_leaf_right_3;
  colliders["leaf-right"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["leaf-right"] ??= [];
  destructionGroups["leaf-right"].push(node_leaf_right_3);

  const endpoint_tepal_front_4 = makeAttachmentEndpoint(null);
  const node_tepal_front_4 = new THREE.Group();
  node_tepal_front_4.name = "Front tepal (outer whorl)__pivot";
  node_tepal_front_4.scale.set(1, 1, 1);
  if (endpoint_tepal_front_4) {
    node_tepal_front_4.position.copy(endpoint_tepal_front_4.start);
    node_tepal_front_4.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_tepal_front_4.position.set(0.0, 0.0, 0.0);
    node_tepal_front_4.rotation.set(0.0, 0.0, 0.0);
  }
  node_tepal_front_4.userData.sculptComponent = {"id": "tepal-front", "name": "Front tepal (outer whorl)", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.95, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0443, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.16, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2446, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2762, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2838, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2755, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2294, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1032, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 0}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.66, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.95}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-front", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-front-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-front", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_front_4.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-front", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_front_4);
  nodes["tepal-front"] = node_tepal_front_4;
  const mesh_tepal_front_4Geometry = endpoint_tepal_front_4
    ? new THREE.CylinderGeometry(endpoint_tepal_front_4.endRadius, endpoint_tepal_front_4.baseRadius, endpoint_tepal_front_4.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0443, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.16, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2446, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2762, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2838, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2755, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2294, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1032, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_front_4) {
    mesh_tepal_front_4Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_front_4 = new THREE.Mesh(
    mesh_tepal_front_4Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_front_4.name = "Front tepal (outer whorl)";
  if (endpoint_tepal_front_4) {
    mesh_tepal_front_4.position.copy(endpoint_tepal_front_4.midpoint);
    mesh_tepal_front_4.quaternion.copy(endpoint_tepal_front_4.quaternion);
  }
  mesh_tepal_front_4.castShadow = options.castShadow ?? true;
  mesh_tepal_front_4.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_front_4.userData.sculptComponent = {"id": "tepal-front", "name": "Front tepal (outer whorl)", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.95, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0443, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.16, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2446, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2762, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2838, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2755, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2294, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1032, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 0}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.66, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.95}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-front", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-front-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-front", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_front_4.add(mesh_tepal_front_4);
  meshes["tepal-front"] = mesh_tepal_front_4;
  colliders["tepal-front"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-front"] ??= [];
  destructionGroups["tepal-front"].push(node_tepal_front_4);

  const endpoint_tepal_outer_a_5 = makeAttachmentEndpoint(null);
  const node_tepal_outer_a_5 = new THREE.Group();
  node_tepal_outer_a_5.name = "Outer tepal A__pivot";
  node_tepal_outer_a_5.scale.set(1, 1, 1);
  if (endpoint_tepal_outer_a_5) {
    node_tepal_outer_a_5.position.copy(endpoint_tepal_outer_a_5.start);
    node_tepal_outer_a_5.rotation.set(0.0, 2.0944, 0.0);
  } else {
    node_tepal_outer_a_5.position.set(0.0, 0.0, 0.0);
    node_tepal_outer_a_5.rotation.set(0.0, 2.0944, 0.0);
  }
  node_tepal_outer_a_5.userData.sculptComponent = {"id": "tepal-outer-a", "name": "Outer tepal A", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.85, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4328], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3036], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1581], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 120}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6072000000000001, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 2.0944, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-outer-a-tip", "kind": "contour", "description": "Pointed tip, flared outward ~0.1 unit", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-outer-a", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_outer_a_5.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_outer_a_5);
  nodes["tepal-outer-a"] = node_tepal_outer_a_5;
  const mesh_tepal_outer_a_5Geometry = endpoint_tepal_outer_a_5
    ? new THREE.CylinderGeometry(endpoint_tepal_outer_a_5.endRadius, endpoint_tepal_outer_a_5.baseRadius, endpoint_tepal_outer_a_5.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4328], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3036], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1581], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_outer_a_5) {
    mesh_tepal_outer_a_5Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_outer_a_5 = new THREE.Mesh(
    mesh_tepal_outer_a_5Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_outer_a_5.name = "Outer tepal A";
  if (endpoint_tepal_outer_a_5) {
    mesh_tepal_outer_a_5.position.copy(endpoint_tepal_outer_a_5.midpoint);
    mesh_tepal_outer_a_5.quaternion.copy(endpoint_tepal_outer_a_5.quaternion);
  }
  mesh_tepal_outer_a_5.castShadow = options.castShadow ?? true;
  mesh_tepal_outer_a_5.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_outer_a_5.userData.sculptComponent = {"id": "tepal-outer-a", "name": "Outer tepal A", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.85, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4328], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3036], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1581], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 120}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6072000000000001, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 2.0944, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-outer-a-tip", "kind": "contour", "description": "Pointed tip, flared outward ~0.1 unit", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-outer-a", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_outer_a_5.add(mesh_tepal_outer_a_5);
  meshes["tepal-outer-a"] = mesh_tepal_outer_a_5;
  colliders["tepal-outer-a"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-outer-a"] ??= [];
  destructionGroups["tepal-outer-a"].push(node_tepal_outer_a_5);

  const endpoint_tepal_outer_b_6 = makeAttachmentEndpoint(null);
  const node_tepal_outer_b_6 = new THREE.Group();
  node_tepal_outer_b_6.name = "Outer tepal B__pivot";
  node_tepal_outer_b_6.scale.set(1, 1, 1);
  if (endpoint_tepal_outer_b_6) {
    node_tepal_outer_b_6.position.copy(endpoint_tepal_outer_b_6.start);
    node_tepal_outer_b_6.rotation.set(0.0, -2.0944, 0.0);
  } else {
    node_tepal_outer_b_6.position.set(0.0, 0.0, 0.0);
    node_tepal_outer_b_6.rotation.set(0.0, -2.0944, 0.0);
  }
  node_tepal_outer_b_6.userData.sculptComponent = {"id": "tepal-outer-b", "name": "Outer tepal B", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.85, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4362], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3169], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1781], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 240}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6072000000000001, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, -2.0944, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-outer-b-tip", "kind": "contour", "description": "Pointed tip, flared outward ~0.1 unit", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-outer-b", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_outer_b_6.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_outer_b_6);
  nodes["tepal-outer-b"] = node_tepal_outer_b_6;
  const mesh_tepal_outer_b_6Geometry = endpoint_tepal_outer_b_6
    ? new THREE.CylinderGeometry(endpoint_tepal_outer_b_6.endRadius, endpoint_tepal_outer_b_6.baseRadius, endpoint_tepal_outer_b_6.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4362], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3169], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1781], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_outer_b_6) {
    mesh_tepal_outer_b_6Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_outer_b_6 = new THREE.Mesh(
    mesh_tepal_outer_b_6Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_outer_b_6.name = "Outer tepal B";
  if (endpoint_tepal_outer_b_6) {
    mesh_tepal_outer_b_6.position.copy(endpoint_tepal_outer_b_6.midpoint);
    mesh_tepal_outer_b_6.quaternion.copy(endpoint_tepal_outer_b_6.quaternion);
  }
  mesh_tepal_outer_b_6.castShadow = options.castShadow ?? true;
  mesh_tepal_outer_b_6.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_outer_b_6.userData.sculptComponent = {"id": "tepal-outer-b", "name": "Outer tepal B", "level": "meso", "role": "petal", "importance": 0.9, "confidence": 0.85, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.0408, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1472, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.225, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2541, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2611, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2534, "twist": 0.0}, {"position": [0.0, 0.88, 0.4362], "rx": 0.018, "rz": 0.211, "twist": 0.0}, {"position": [0.0, 0.97, 0.3169], "rx": 0.018, "rz": 0.0949, "twist": 0.0}, {"position": [0.0, 1.03, 0.1781], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip).", "angleDegrees": 240}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6072000000000001, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.85}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, -2.0944, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-outer-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-outer-b-tip", "kind": "contour", "description": "Pointed tip, flared outward ~0.1 unit", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-outer-b", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_outer_b_6.add(mesh_tepal_outer_b_6);
  meshes["tepal-outer-b"] = mesh_tepal_outer_b_6;
  colliders["tepal-outer-b"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-outer-b"] ??= [];
  destructionGroups["tepal-outer-b"].push(node_tepal_outer_b_6);

  const endpoint_tepal_inner_a_7 = makeAttachmentEndpoint(null);
  const node_tepal_inner_a_7 = new THREE.Group();
  node_tepal_inner_a_7.name = "Inner tepal A (hidden, inferred)__pivot";
  node_tepal_inner_a_7.scale.set(1, 1, 1);
  if (endpoint_tepal_inner_a_7) {
    node_tepal_inner_a_7.position.copy(endpoint_tepal_inner_a_7.start);
    node_tepal_inner_a_7.rotation.set(0.0, 1.0472, 0.0);
  } else {
    node_tepal_inner_a_7.position.set(0.0, 0.0, 0.0);
    node_tepal_inner_a_7.rotation.set(0.0, 1.0472, 0.0);
  }
  node_tepal_inner_a_7.userData.sculptComponent = {"id": "tepal-inner-a", "name": "Inner tepal A (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 60}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 1.0472, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-a-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-a", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_a_7.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_inner_a_7);
  nodes["tepal-inner-a"] = node_tepal_inner_a_7;
  const mesh_tepal_inner_a_7Geometry = endpoint_tepal_inner_a_7
    ? new THREE.CylinderGeometry(endpoint_tepal_inner_a_7.endRadius, endpoint_tepal_inner_a_7.baseRadius, endpoint_tepal_inner_a_7.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_inner_a_7) {
    mesh_tepal_inner_a_7Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_inner_a_7 = new THREE.Mesh(
    mesh_tepal_inner_a_7Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_inner_a_7.name = "Inner tepal A (hidden, inferred)";
  if (endpoint_tepal_inner_a_7) {
    mesh_tepal_inner_a_7.position.copy(endpoint_tepal_inner_a_7.midpoint);
    mesh_tepal_inner_a_7.quaternion.copy(endpoint_tepal_inner_a_7.quaternion);
  }
  mesh_tepal_inner_a_7.castShadow = options.castShadow ?? true;
  mesh_tepal_inner_a_7.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_inner_a_7.userData.sculptComponent = {"id": "tepal-inner-a", "name": "Inner tepal A (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 60}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 1.0472, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-a", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-a-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-a", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_a_7.add(mesh_tepal_inner_a_7);
  meshes["tepal-inner-a"] = mesh_tepal_inner_a_7;
  colliders["tepal-inner-a"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-inner-a"] ??= [];
  destructionGroups["tepal-inner-a"].push(node_tepal_inner_a_7);

  const endpoint_tepal_inner_b_8 = makeAttachmentEndpoint(null);
  const node_tepal_inner_b_8 = new THREE.Group();
  node_tepal_inner_b_8.name = "Inner tepal B (hidden, inferred)__pivot";
  node_tepal_inner_b_8.scale.set(1, 1, 1);
  if (endpoint_tepal_inner_b_8) {
    node_tepal_inner_b_8.position.copy(endpoint_tepal_inner_b_8.start);
    node_tepal_inner_b_8.rotation.set(0.0, -1.0472, 0.0);
  } else {
    node_tepal_inner_b_8.position.set(0.0, 0.0, 0.0);
    node_tepal_inner_b_8.rotation.set(0.0, -1.0472, 0.0);
  }
  node_tepal_inner_b_8.userData.sculptComponent = {"id": "tepal-inner-b", "name": "Inner tepal B (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 180}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, -1.0472, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-b-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-b", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_b_8.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_inner_b_8);
  nodes["tepal-inner-b"] = node_tepal_inner_b_8;
  const mesh_tepal_inner_b_8Geometry = endpoint_tepal_inner_b_8
    ? new THREE.CylinderGeometry(endpoint_tepal_inner_b_8.endRadius, endpoint_tepal_inner_b_8.baseRadius, endpoint_tepal_inner_b_8.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_inner_b_8) {
    mesh_tepal_inner_b_8Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_inner_b_8 = new THREE.Mesh(
    mesh_tepal_inner_b_8Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_inner_b_8.name = "Inner tepal B (hidden, inferred)";
  if (endpoint_tepal_inner_b_8) {
    mesh_tepal_inner_b_8.position.copy(endpoint_tepal_inner_b_8.midpoint);
    mesh_tepal_inner_b_8.quaternion.copy(endpoint_tepal_inner_b_8.quaternion);
  }
  mesh_tepal_inner_b_8.castShadow = options.castShadow ?? true;
  mesh_tepal_inner_b_8.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_inner_b_8.userData.sculptComponent = {"id": "tepal-inner-b", "name": "Inner tepal B (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 180}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, -1.0472, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-b", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-b-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-b", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_b_8.add(mesh_tepal_inner_b_8);
  meshes["tepal-inner-b"] = mesh_tepal_inner_b_8;
  colliders["tepal-inner-b"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-inner-b"] ??= [];
  destructionGroups["tepal-inner-b"].push(node_tepal_inner_b_8);

  const endpoint_tepal_inner_c_9 = makeAttachmentEndpoint(null);
  const node_tepal_inner_c_9 = new THREE.Group();
  node_tepal_inner_c_9.name = "Inner tepal C (hidden, inferred)__pivot";
  node_tepal_inner_c_9.scale.set(1, 1, 1);
  if (endpoint_tepal_inner_c_9) {
    node_tepal_inner_c_9.position.copy(endpoint_tepal_inner_c_9.start);
    node_tepal_inner_c_9.rotation.set(0.0, 3.1416, 0.0);
  } else {
    node_tepal_inner_c_9.position.set(0.0, 0.0, 0.0);
    node_tepal_inner_c_9.rotation.set(0.0, 3.1416, 0.0);
  }
  node_tepal_inner_c_9.userData.sculptComponent = {"id": "tepal-inner-c", "name": "Inner tepal C (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 300}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 3.1416, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-c", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-c-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-c", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_c_9.userData.actionProfile = {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-c", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["bloom"] ?? root).add(node_tepal_inner_c_9);
  nodes["tepal-inner-c"] = node_tepal_inner_c_9;
  const mesh_tepal_inner_c_9Geometry = endpoint_tepal_inner_c_9
    ? new THREE.CylinderGeometry(endpoint_tepal_inner_c_9.endRadius, endpoint_tepal_inner_c_9.baseRadius, endpoint_tepal_inner_c_9.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true});
  if (!endpoint_tepal_inner_c_9) {
    mesh_tepal_inner_c_9Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tepal_inner_c_9 = new THREE.Mesh(
    mesh_tepal_inner_c_9Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tepal_inner_c_9.name = "Inner tepal C (hidden, inferred)";
  if (endpoint_tepal_inner_c_9) {
    mesh_tepal_inner_c_9.position.copy(endpoint_tepal_inner_c_9.midpoint);
    mesh_tepal_inner_c_9.quaternion.copy(endpoint_tepal_inner_c_9.quaternion);
  }
  mesh_tepal_inner_c_9.castShadow = options.castShadow ?? true;
  mesh_tepal_inner_c_9.receiveShadow = options.receiveShadow ?? true;
  mesh_tepal_inner_c_9.userData.sculptComponent = {"id": "tepal-inner-c", "name": "Inner tepal C (hidden, inferred)", "level": "meso", "role": "petal", "importance": 0.4, "confidence": 0.45, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Thin curved tepal blade, width tapering to a pointed tip: tapered-sweep hugging the cup surface; arranged on a 3-fold whorl.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.02, 0.0784], "rx": 0.018, "rz": 0.043, "twist": 0.0}, {"position": [0.0, 0.15, 0.3224], "rx": 0.018, "rz": 0.1552, "twist": 0.0}, {"position": [0.0, 0.3, 0.3926], "rx": 0.018, "rz": 0.2372, "twist": 0.0}, {"position": [0.0, 0.45, 0.4281], "rx": 0.018, "rz": 0.2679, "twist": 0.0}, {"position": [0.0, 0.6, 0.4586], "rx": 0.018, "rz": 0.2753, "twist": 0.0}, {"position": [0.0, 0.75, 0.4684], "rx": 0.018, "rz": 0.2672, "twist": 0.0}, {"position": [0.0, 0.88, 0.4245], "rx": 0.018, "rz": 0.2225, "twist": 0.0}, {"position": [0.0, 0.97, 0.2703], "rx": 0.018, "rz": 0.1001, "twist": 0.0}, {"position": [0.0, 1.03, 0.1081], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 10, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "buildCuppedTepal", "reason": "Generator has no tapered cupped-shell primitive: tapered-sweep has an elliptic (lens) section and curve-sweep a constant section; with an attachment endpoint the generator emits a connector cylinder. A tepal needs a thin shell that is convex across its width, tapers to a point and flares at the tip.", "rule": "Stations of taperedSweep are reused as the profile: station.position.z = cup radius r(v), station.position.y = height, station.rz = half-width (arc length). Section = circular arc about the bloom axis with outward bulge 0.10*(1-u^2), edges tucked inward; tip flare 0.05; thickness 0.016 at base tapering to 0.006 at the rim; closed solid (outer + inner surface + rim strip). Form pass: inner whorl height x0.975 so tips sit under the outer crown.", "angleDegrees": 300}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "cup-surface", "localStart": [0.0, 0.02, 0.0792], "localEnd": [0.0, 1.0, 0.273], "contactType": "overlap", "embedDepth": 0.02, "gapTolerance": 0.01, "baseRadius": 0.03, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.6402, "height": 1.0, "depth": 0.1, "units": "relative", "confidence": 0.45}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 3.1416, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "bendable", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "tepal-inner-c", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tepal-inner-c-tip", "kind": "contour", "description": "Pointed tip closing the cup", "region": "tepal tip"}], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "tepal-inner-c", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_tepal_inner_c_9.add(mesh_tepal_inner_c_9);
  meshes["tepal-inner-c"] = mesh_tepal_inner_c_9;
  colliders["tepal-inner-c"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["tepal-inner-c"] ??= [];
  destructionGroups["tepal-inner-c"].push(node_tepal_inner_c_9);

  const endpoint_receptacle_10 = makeAttachmentEndpoint(null);
  const node_receptacle_10 = new THREE.Group();
  node_receptacle_10.name = "Receptacle collar__pivot";
  node_receptacle_10.scale.set(1, 1, 1);
  if (endpoint_receptacle_10) {
    node_receptacle_10.position.copy(endpoint_receptacle_10.start);
    node_receptacle_10.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_receptacle_10.position.set(0.06, 0.0, 0.0);
    node_receptacle_10.rotation.set(0.0, 0.0, 0.0);
  }
  node_receptacle_10.userData.sculptComponent = {"id": "receptacle", "name": "Receptacle collar", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.5, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Short revolved neck where tepals converge on the stem.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.075, -0.08], [0.085, 0.0], [0.11, 0.08], [0.07, 0.14]], "segments": 20}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "receptacle", "localStart": [0.06, -0.08, 0.0], "localEnd": [0.06, 0.14, 0.0], "contactType": "embed", "embedDepth": 0.04, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.07, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.2, "height": 0.22, "depth": 0.2, "units": "relative", "confidence": 0.5}, "transform": {"position": [0.06, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.8, 0.8, 0.8]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "receptacle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}}, "material": "stem-green", "materialLayers": ["stem-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.42, "microRoughness": 0.45, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12)", "displacementPattern": "none", "occlusionPattern": "contact shadow under receptacle from shadow-casting key light", "edgeWearPattern": "none", "notes": "waxy cuticle sheen band along lit side"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "receptacle", "dominantAlbedo": "rgba(128, 148, 31, 1.0)", "secondaryAlbedo": "rgba(76, 90, 20, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.257, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/stem.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.401}, "colorGradient": {"type": "radial", "axis": [0.5035005091649695, 0.5037268047069473], "stops": [{"offset": 0.0, "color": "rgba(145, 164, 36, 1.0)"}, {"offset": 0.55, "color": "rgba(95, 118, 2, 1.0)"}, {"offset": 1.0, "color": "rgba(100, 112, 76, 1.0)"}], "confidence": 1.0}}};
  node_receptacle_10.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "receptacle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}};
  (nodes["bloom"] ?? root).add(node_receptacle_10);
  nodes["receptacle"] = node_receptacle_10;
  const mesh_receptacle_10Geometry = endpoint_receptacle_10
    ? new THREE.CylinderGeometry(endpoint_receptacle_10.endRadius, endpoint_receptacle_10.baseRadius, endpoint_receptacle_10.length, 16, 6)
    : buildLatheGeometry({"points": [[0.075, -0.08], [0.085, 0.0], [0.11, 0.08], [0.07, 0.14]], "segments": 20});
  if (!endpoint_receptacle_10) {
    mesh_receptacle_10Geometry.scale(0.8, 0.8, 0.8);
  }
  const mesh_receptacle_10 = new THREE.Mesh(
    mesh_receptacle_10Geometry,
    materialMap["stem-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_receptacle_10.name = "Receptacle collar";
  if (endpoint_receptacle_10) {
    mesh_receptacle_10.position.copy(endpoint_receptacle_10.midpoint);
    mesh_receptacle_10.quaternion.copy(endpoint_receptacle_10.quaternion);
  }
  mesh_receptacle_10.castShadow = options.castShadow ?? true;
  mesh_receptacle_10.receiveShadow = options.receiveShadow ?? true;
  mesh_receptacle_10.userData.sculptComponent = {"id": "receptacle", "name": "Receptacle collar", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.5, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Short revolved neck where tepals converge on the stem.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.075, -0.08], [0.085, 0.0], [0.11, 0.08], [0.07, 0.14]], "segments": 20}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "receptacle", "localStart": [0.06, -0.08, 0.0], "localEnd": [0.06, 0.14, 0.0], "contactType": "embed", "embedDepth": 0.04, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.07, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.2, "height": 0.22, "depth": 0.2, "units": "relative", "confidence": 0.5}, "transform": {"position": [0.06, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.8, 0.8, 0.8]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "receptacle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stem-green"}}, "material": "stem-green", "materialLayers": ["stem-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.42, "microRoughness": 0.45, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12)", "displacementPattern": "none", "occlusionPattern": "contact shadow under receptacle from shadow-casting key light", "edgeWearPattern": "none", "notes": "waxy cuticle sheen band along lit side"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "receptacle", "dominantAlbedo": "rgba(128, 148, 31, 1.0)", "secondaryAlbedo": "rgba(76, 90, 20, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.257, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/stem.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.401}, "colorGradient": {"type": "radial", "axis": [0.5035005091649695, 0.5037268047069473], "stops": [{"offset": 0.0, "color": "rgba(145, 164, 36, 1.0)"}, {"offset": 0.55, "color": "rgba(95, 118, 2, 1.0)"}, {"offset": 1.0, "color": "rgba(100, 112, 76, 1.0)"}], "confidence": 1.0}}};
  node_receptacle_10.add(mesh_receptacle_10);
  meshes["receptacle"] = mesh_receptacle_10;
  colliders["receptacle"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["receptacle"] ??= [];
  destructionGroups["receptacle"].push(node_receptacle_10);

  const endpoint_sheath_left_11 = makeAttachmentEndpoint(null);
  const node_sheath_left_11 = new THREE.Group();
  node_sheath_left_11.name = "Leaf sheath (left)__pivot";
  node_sheath_left_11.scale.set(1, 1, 1);
  if (endpoint_sheath_left_11) {
    node_sheath_left_11.position.copy(endpoint_sheath_left_11.start);
    node_sheath_left_11.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_sheath_left_11.position.set(0.0, 0.0, 0.0);
    node_sheath_left_11.rotation.set(0.0, 0.0, 0.0);
  }
  node_sheath_left_11.userData.sculptComponent = {"id": "sheath-left", "name": "Leaf sheath (left)", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.4, "primitive": "tapered-sweep", "topologyClass": "conforming-shell", "topologyRationale": "Leaf base wraps the stem: shallow shell conforming to the stem surface.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-left", "localStart": [0.06, -1.55, 0.0], "localEnd": [0.055, -1.33, 0.0], "contactType": "overlap", "embedDepth": 0.03, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.15, "height": 0.22, "depth": 0.15, "units": "relative", "confidence": 0.4}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "sheath-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_sheath_left_11.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}};
  (nodes["root"] ?? root).add(node_sheath_left_11);
  nodes["sheath-left"] = node_sheath_left_11;
  const mesh_sheath_left_11Geometry = endpoint_sheath_left_11
    ? new THREE.CylinderGeometry(endpoint_sheath_left_11.endRadius, endpoint_sheath_left_11.baseRadius, endpoint_sheath_left_11.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true});
  if (!endpoint_sheath_left_11) {
    mesh_sheath_left_11Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_sheath_left_11 = new THREE.Mesh(
    mesh_sheath_left_11Geometry,
    materialMap["leaf-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_sheath_left_11.name = "Leaf sheath (left)";
  if (endpoint_sheath_left_11) {
    mesh_sheath_left_11.position.copy(endpoint_sheath_left_11.midpoint);
    mesh_sheath_left_11.quaternion.copy(endpoint_sheath_left_11.quaternion);
  }
  mesh_sheath_left_11.castShadow = options.castShadow ?? true;
  mesh_sheath_left_11.receiveShadow = options.receiveShadow ?? true;
  mesh_sheath_left_11.userData.sculptComponent = {"id": "sheath-left", "name": "Leaf sheath (left)", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.4, "primitive": "tapered-sweep", "topologyClass": "conforming-shell", "topologyRationale": "Leaf base wraps the stem: shallow shell conforming to the stem surface.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-left", "localStart": [0.06, -1.55, 0.0], "localEnd": [0.055, -1.33, 0.0], "contactType": "overlap", "embedDepth": 0.03, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.15, "height": 0.22, "depth": 0.15, "units": "relative", "confidence": 0.4}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "sheath-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_sheath_left_11.add(mesh_sheath_left_11);
  meshes["sheath-left"] = mesh_sheath_left_11;
  colliders["sheath-left"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["sheath-left"] ??= [];
  destructionGroups["sheath-left"].push(node_sheath_left_11);

  const endpoint_sheath_right_12 = makeAttachmentEndpoint(null);
  const node_sheath_right_12 = new THREE.Group();
  node_sheath_right_12.name = "Leaf sheath (right)__pivot";
  node_sheath_right_12.scale.set(1, 1, 1);
  if (endpoint_sheath_right_12) {
    node_sheath_right_12.position.copy(endpoint_sheath_right_12.start);
    node_sheath_right_12.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_sheath_right_12.position.set(0.0, 0.0, 0.0);
    node_sheath_right_12.rotation.set(0.0, 0.0, 0.0);
  }
  node_sheath_right_12.userData.sculptComponent = {"id": "sheath-right", "name": "Leaf sheath (right)", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.4, "primitive": "tapered-sweep", "topologyClass": "conforming-shell", "topologyRationale": "Leaf base wraps the stem: shallow shell conforming to the stem surface.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-right", "localStart": [0.06, -1.55, 0.0], "localEnd": [0.055, -1.33, 0.0], "contactType": "overlap", "embedDepth": 0.03, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.15, "height": 0.22, "depth": 0.15, "units": "relative", "confidence": 0.4}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "sheath-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_sheath_right_12.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}};
  (nodes["root"] ?? root).add(node_sheath_right_12);
  nodes["sheath-right"] = node_sheath_right_12;
  const mesh_sheath_right_12Geometry = endpoint_sheath_right_12
    ? new THREE.CylinderGeometry(endpoint_sheath_right_12.endRadius, endpoint_sheath_right_12.baseRadius, endpoint_sheath_right_12.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true});
  if (!endpoint_sheath_right_12) {
    mesh_sheath_right_12Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_sheath_right_12 = new THREE.Mesh(
    mesh_sheath_right_12Geometry,
    materialMap["leaf-green"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_sheath_right_12.name = "Leaf sheath (right)";
  if (endpoint_sheath_right_12) {
    mesh_sheath_right_12.position.copy(endpoint_sheath_right_12.midpoint);
    mesh_sheath_right_12.quaternion.copy(endpoint_sheath_right_12.quaternion);
  }
  mesh_sheath_right_12.castShadow = options.castShadow ?? true;
  mesh_sheath_right_12.receiveShadow = options.receiveShadow ?? true;
  mesh_sheath_right_12.userData.sculptComponent = {"id": "sheath-right", "name": "Leaf sheath (right)", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.4, "primitive": "tapered-sweep", "topologyClass": "conforming-shell", "topologyRationale": "Leaf base wraps the stem: shallow shell conforming to the stem surface.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.02, -1.55, 0.0], "rx": 0.066, "rz": 0.066, "twist": 0.0}, {"position": [0.02, -1.42, 0.0], "rx": 0.064, "rz": 0.064, "twist": 0.0}, {"position": [0.02, -1.33, 0.0], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 14, "capEnds": true}}, "parent": "root", "attachment": {"parentId": "root", "parentSocket": "stem-leaf-right", "localStart": [0.06, -1.55, 0.0], "localEnd": [0.055, -1.33, 0.0], "contactType": "overlap", "embedDepth": 0.03, "gapTolerance": 0.01, "baseRadius": 0.075, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.15, "height": 0.22, "depth": 0.15, "units": "relative", "confidence": 0.4}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "sheath-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-green"}}, "material": "leaf-green", "materialLayers": ["leaf-green"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "sheath-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_sheath_right_12.add(mesh_sheath_right_12);
  meshes["sheath-right"] = mesh_sheath_right_12;
  colliders["sheath-right"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["sheath-right"] ??= [];
  destructionGroups["sheath-right"].push(node_sheath_right_12);

  const endpoint_crease_left_13 = makeAttachmentEndpoint(null);
  const node_crease_left_13 = new THREE.Group();
  node_crease_left_13.name = "Overlap crease (left)__pivot";
  node_crease_left_13.scale.set(1, 1, 1);
  if (endpoint_crease_left_13) {
    node_crease_left_13.position.copy(endpoint_crease_left_13.start);
    node_crease_left_13.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_crease_left_13.position.set(0.0, 0.0, 0.0);
    node_crease_left_13.rotation.set(0.0, 0.0, 0.0);
  }
  node_crease_left_13.userData.sculptComponent = {"id": "crease-left", "name": "Overlap crease (left)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Dark seam where front tepal edge overlaps the neighbouring tepal, Y 0..0.4 (reference y 280-372).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [-0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [-0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "surface", "localStart": [-0.0391, 0.02, 0.111], "localEnd": [-0.1358, 0.4, 0.4108], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}}, "material": "crease-blue", "materialLayers": ["crease-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "crease-left", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_crease_left_13.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}};
  (nodes["bloom"] ?? root).add(node_crease_left_13);
  nodes["crease-left"] = node_crease_left_13;
  const mesh_crease_left_13Geometry = endpoint_crease_left_13
    ? new THREE.CylinderGeometry(endpoint_crease_left_13.endRadius, endpoint_crease_left_13.baseRadius, endpoint_crease_left_13.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [-0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [-0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true});
  if (!endpoint_crease_left_13) {
    mesh_crease_left_13Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_crease_left_13 = new THREE.Mesh(
    mesh_crease_left_13Geometry,
    materialMap["crease-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_crease_left_13.name = "Overlap crease (left)";
  if (endpoint_crease_left_13) {
    mesh_crease_left_13.position.copy(endpoint_crease_left_13.midpoint);
    mesh_crease_left_13.quaternion.copy(endpoint_crease_left_13.quaternion);
  }
  mesh_crease_left_13.castShadow = options.castShadow ?? true;
  mesh_crease_left_13.receiveShadow = options.receiveShadow ?? true;
  mesh_crease_left_13.userData.sculptComponent = {"id": "crease-left", "name": "Overlap crease (left)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Dark seam where front tepal edge overlaps the neighbouring tepal, Y 0..0.4 (reference y 280-372).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [-0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [-0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [-0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [-0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "surface", "localStart": [-0.0391, 0.02, 0.111], "localEnd": [-0.1358, 0.4, 0.4108], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}}, "material": "crease-blue", "materialLayers": ["crease-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "crease-left", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_crease_left_13.add(mesh_crease_left_13);
  meshes["crease-left"] = mesh_crease_left_13;
  colliders["crease-left"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["crease-left"] ??= [];
  destructionGroups["crease-left"].push(node_crease_left_13);

  const endpoint_midrib_left_14 = makeAttachmentEndpoint(null);
  const node_midrib_left_14 = new THREE.Group();
  node_midrib_left_14.name = "Leaf midrib (left)__pivot";
  node_midrib_left_14.scale.set(1, 1, 1);
  if (endpoint_midrib_left_14) {
    node_midrib_left_14.position.copy(endpoint_midrib_left_14.start);
    node_midrib_left_14.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_midrib_left_14.position.set(0.0, 0.0, 0.0);
    node_midrib_left_14.rotation.set(0.0, 0.0, 0.0);
  }
  node_midrib_left_14.userData.sculptComponent = {"id": "midrib-left", "name": "Leaf midrib (left)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Raised midrib crease along the leaf spine.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [-0.0588, -1.2564, 0.0334], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [-0.1382, -1.0544, 0.0508], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.2161, -0.8519, 0.0658], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.2999, -0.6518, 0.0806], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.3909, -0.4547, 0.0952], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.4828, -0.2582, 0.1069], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "leaf-left", "attachment": {"parentId": "leaf-left", "parentSocket": "surface", "localStart": [-0.0588, -1.2564, 0.0334], "localEnd": [-0.4828, -0.2582, 0.1069], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}}, "material": "leaf-midrib", "materialLayers": ["leaf-midrib"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "midrib-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_midrib_left_14.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}};
  (nodes["leaf-left"] ?? root).add(node_midrib_left_14);
  nodes["midrib-left"] = node_midrib_left_14;
  const mesh_midrib_left_14Geometry = endpoint_midrib_left_14
    ? new THREE.CylinderGeometry(endpoint_midrib_left_14.endRadius, endpoint_midrib_left_14.baseRadius, endpoint_midrib_left_14.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [-0.0588, -1.2564, 0.0334], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [-0.1382, -1.0544, 0.0508], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.2161, -0.8519, 0.0658], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.2999, -0.6518, 0.0806], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.3909, -0.4547, 0.0952], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.4828, -0.2582, 0.1069], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true});
  if (!endpoint_midrib_left_14) {
    mesh_midrib_left_14Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_midrib_left_14 = new THREE.Mesh(
    mesh_midrib_left_14Geometry,
    materialMap["leaf-midrib"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_midrib_left_14.name = "Leaf midrib (left)";
  if (endpoint_midrib_left_14) {
    mesh_midrib_left_14.position.copy(endpoint_midrib_left_14.midpoint);
    mesh_midrib_left_14.quaternion.copy(endpoint_midrib_left_14.quaternion);
  }
  mesh_midrib_left_14.castShadow = options.castShadow ?? true;
  mesh_midrib_left_14.receiveShadow = options.receiveShadow ?? true;
  mesh_midrib_left_14.userData.sculptComponent = {"id": "midrib-left", "name": "Leaf midrib (left)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Raised midrib crease along the leaf spine.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [-0.0588, -1.2564, 0.0334], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [-0.1382, -1.0544, 0.0508], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.2161, -0.8519, 0.0658], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.2999, -0.6518, 0.0806], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [-0.3909, -0.4547, 0.0952], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [-0.4828, -0.2582, 0.1069], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "leaf-left", "attachment": {"parentId": "leaf-left", "parentSocket": "surface", "localStart": [-0.0588, -1.2564, 0.0334], "localEnd": [-0.4828, -0.2582, 0.1069], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-left", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}}, "material": "leaf-midrib", "materialLayers": ["leaf-midrib"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "midrib-left", "dominantAlbedo": "rgba(101, 113, 49, 1.0)", "secondaryAlbedo": "rgba(78, 91, 22, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.139, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_left.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.447}}};
  node_midrib_left_14.add(mesh_midrib_left_14);
  meshes["midrib-left"] = mesh_midrib_left_14;
  colliders["midrib-left"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["midrib-left"] ??= [];
  destructionGroups["midrib-left"].push(node_midrib_left_14);

  const endpoint_crease_right_15 = makeAttachmentEndpoint(null);
  const node_crease_right_15 = new THREE.Group();
  node_crease_right_15.name = "Overlap crease (right)__pivot";
  node_crease_right_15.scale.set(1, 1, 1);
  if (endpoint_crease_right_15) {
    node_crease_right_15.position.copy(endpoint_crease_right_15.start);
    node_crease_right_15.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_crease_right_15.position.set(0.0, 0.0, 0.0);
    node_crease_right_15.rotation.set(0.0, 0.0, 0.0);
  }
  node_crease_right_15.userData.sculptComponent = {"id": "crease-right", "name": "Overlap crease (right)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Dark seam where front tepal edge overlaps the neighbouring tepal, Y 0..0.4 (reference y 280-372).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "surface", "localStart": [0.0391, 0.02, 0.111], "localEnd": [0.1358, 0.4, 0.4108], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}}, "material": "crease-blue", "materialLayers": ["crease-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "crease-right", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_crease_right_15.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}};
  (nodes["bloom"] ?? root).add(node_crease_right_15);
  nodes["crease-right"] = node_crease_right_15;
  const mesh_crease_right_15Geometry = endpoint_crease_right_15
    ? new THREE.CylinderGeometry(endpoint_crease_right_15.endRadius, endpoint_crease_right_15.baseRadius, endpoint_crease_right_15.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true});
  if (!endpoint_crease_right_15) {
    mesh_crease_right_15Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_crease_right_15 = new THREE.Mesh(
    mesh_crease_right_15Geometry,
    materialMap["crease-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_crease_right_15.name = "Overlap crease (right)";
  if (endpoint_crease_right_15) {
    mesh_crease_right_15.position.copy(endpoint_crease_right_15.midpoint);
    mesh_crease_right_15.quaternion.copy(endpoint_crease_right_15.quaternion);
  }
  mesh_crease_right_15.castShadow = options.castShadow ?? true;
  mesh_crease_right_15.receiveShadow = options.receiveShadow ?? true;
  mesh_crease_right_15.userData.sculptComponent = {"id": "crease-right", "name": "Overlap crease (right)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Dark seam where front tepal edge overlaps the neighbouring tepal, Y 0..0.4 (reference y 280-372).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0391, 0.02, 0.111], "rx": 0.006, "rz": 0.0042, "twist": 0.0}, {"position": [0.0709, 0.1, 0.25], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.0977, 0.18, 0.3363], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1204, 0.26, 0.3728], "rx": 0.006, "rz": 0.0116, "twist": 0.0}, {"position": [0.1299, 0.34, 0.3964], "rx": 0.006, "rz": 0.0088, "twist": 0.0}, {"position": [0.1358, 0.4, 0.4108], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "bloom", "attachment": {"parentId": "bloom", "parentSocket": "surface", "localStart": [0.0391, 0.02, 0.111], "localEnd": [0.1358, 0.4, 0.4108], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "crease-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "crease-blue"}}, "material": "crease-blue", "materialLayers": ["crease-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "crease-right", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_crease_right_15.add(mesh_crease_right_15);
  meshes["crease-right"] = mesh_crease_right_15;
  colliders["crease-right"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["crease-right"] ??= [];
  destructionGroups["crease-right"].push(node_crease_right_15);

  const endpoint_midrib_right_16 = makeAttachmentEndpoint(null);
  const node_midrib_right_16 = new THREE.Group();
  node_midrib_right_16.name = "Leaf midrib (right)__pivot";
  node_midrib_right_16.scale.set(1, 1, 1);
  if (endpoint_midrib_right_16) {
    node_midrib_right_16.position.copy(endpoint_midrib_right_16.start);
    node_midrib_right_16.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_midrib_right_16.position.set(0.0, 0.0, 0.0);
    node_midrib_right_16.rotation.set(0.0, 0.0, 0.0);
  }
  node_midrib_right_16.userData.sculptComponent = {"id": "midrib-right", "name": "Leaf midrib (right)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Raised midrib crease along the leaf spine.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.1588, -1.2527, 0.0027], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [0.2211, -1.031, -0.0103], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.2735, -0.8067, -0.0283], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.331, -0.5838, -0.0454], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.3925, -0.3618, -0.0618], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.4507, -0.139, -0.0703], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "leaf-right", "attachment": {"parentId": "leaf-right", "parentSocket": "surface", "localStart": [0.1588, -1.2527, 0.0027], "localEnd": [0.4507, -0.139, -0.0703], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}}, "material": "leaf-midrib", "materialLayers": ["leaf-midrib"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "midrib-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_midrib_right_16.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}};
  (nodes["leaf-right"] ?? root).add(node_midrib_right_16);
  nodes["midrib-right"] = node_midrib_right_16;
  const mesh_midrib_right_16Geometry = endpoint_midrib_right_16
    ? new THREE.CylinderGeometry(endpoint_midrib_right_16.endRadius, endpoint_midrib_right_16.baseRadius, endpoint_midrib_right_16.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.1588, -1.2527, 0.0027], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [0.2211, -1.031, -0.0103], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.2735, -0.8067, -0.0283], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.331, -0.5838, -0.0454], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.3925, -0.3618, -0.0618], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.4507, -0.139, -0.0703], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true});
  if (!endpoint_midrib_right_16) {
    mesh_midrib_right_16Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_midrib_right_16 = new THREE.Mesh(
    mesh_midrib_right_16Geometry,
    materialMap["leaf-midrib"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_midrib_right_16.name = "Leaf midrib (right)";
  if (endpoint_midrib_right_16) {
    mesh_midrib_right_16.position.copy(endpoint_midrib_right_16.midpoint);
    mesh_midrib_right_16.quaternion.copy(endpoint_midrib_right_16.quaternion);
  }
  mesh_midrib_right_16.castShadow = options.castShadow ?? true;
  mesh_midrib_right_16.receiveShadow = options.receiveShadow ?? true;
  mesh_midrib_right_16.userData.sculptComponent = {"id": "midrib-right", "name": "Leaf midrib (right)", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.55, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Raised midrib crease along the leaf spine.", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.1588, -1.2527, 0.0027], "rx": 0.006, "rz": 0.0028, "twist": 0.0}, {"position": [0.2211, -1.031, -0.0103], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.2735, -0.8067, -0.0283], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.331, -0.5838, -0.0454], "rx": 0.006, "rz": 0.0077, "twist": 0.0}, {"position": [0.3925, -0.3618, -0.0618], "rx": 0.006, "rz": 0.0059, "twist": 0.0}, {"position": [0.4507, -0.139, -0.0703], "rx": 0.0, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "leaf-right", "attachment": {"parentId": "leaf-right", "parentSocket": "surface", "localStart": [0.1588, -1.2527, 0.0027], "localEnd": [0.4507, -0.139, -0.0703], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.06, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.55}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "midrib-right", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "leaf-midrib"}}, "material": "leaf-midrib", "materialLayers": ["leaf-midrib"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.64, "microRoughness": 0.66, "bumpAmplitude": 0.0, "normalPattern": "faint procedural normal (scale 0.12) for waxy grain", "displacementPattern": "none", "occlusionPattern": "channel fold darkens towards midrib under rim light", "edgeWearPattern": "pale glaucous margin |u|>0.82", "notes": "midrib light stripe; underside 8% paler"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "midrib-right", "dominantAlbedo": "rgba(79, 91, 56, 1.0)", "secondaryAlbedo": "rgba(106, 123, 79, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.126, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/home/claude/tulip/evidence/crops/leaf_right.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.449}}};
  node_midrib_right_16.add(mesh_midrib_right_16);
  meshes["midrib-right"] = mesh_midrib_right_16;
  colliders["midrib-right"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["midrib-right"] ??= [];
  destructionGroups["midrib-right"].push(node_midrib_right_16);

  const endpoint_front_ridge_17 = makeAttachmentEndpoint(null);
  const node_front_ridge_17 = new THREE.Group();
  node_front_ridge_17.name = "Front tepal soft ridge__pivot";
  node_front_ridge_17.scale.set(1, 1, 1);
  if (endpoint_front_ridge_17) {
    node_front_ridge_17.position.copy(endpoint_front_ridge_17.start);
    node_front_ridge_17.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_front_ridge_17.position.set(0.0, 0.0, 0.0);
    node_front_ridge_17.rotation.set(0.0, 0.0, 0.0);
  }
  node_front_ridge_17.userData.sculptComponent = {"id": "front-ridge", "name": "Front tepal soft ridge", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.5, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Soft vertical ridge shading on the large front tepal (reference x~200-260).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.08, 0.2177], "rx": 0.006, "rz": 0.0, "twist": 0.0}, {"position": [0.0, 0.3, 0.4046], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.5, 0.4473], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.7, 0.4793], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.88, 0.4365], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.96, 0.2994], "rx": 0.006, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "tepal-front", "attachment": {"parentId": "tepal-front", "parentSocket": "surface", "localStart": [0.0, 0.08, 0.2078], "localEnd": [0.0, 0.96, 0.2903], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.04, "height": 0.9, "depth": 0.03, "units": "relative", "confidence": 0.5}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "front-ridge", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "front-ridge", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_front_ridge_17.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "front-ridge", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}};
  (nodes["tepal-front"] ?? root).add(node_front_ridge_17);
  nodes["front-ridge"] = node_front_ridge_17;
  const mesh_front_ridge_17Geometry = endpoint_front_ridge_17
    ? new THREE.CylinderGeometry(endpoint_front_ridge_17.endRadius, endpoint_front_ridge_17.baseRadius, endpoint_front_ridge_17.length, 16, 6)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.08, 0.2177], "rx": 0.006, "rz": 0.0, "twist": 0.0}, {"position": [0.0, 0.3, 0.4046], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.5, 0.4473], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.7, 0.4793], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.88, 0.4365], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.96, 0.2994], "rx": 0.006, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true});
  if (!endpoint_front_ridge_17) {
    mesh_front_ridge_17Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_front_ridge_17 = new THREE.Mesh(
    mesh_front_ridge_17Geometry,
    materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_front_ridge_17.name = "Front tepal soft ridge";
  if (endpoint_front_ridge_17) {
    mesh_front_ridge_17.position.copy(endpoint_front_ridge_17.midpoint);
    mesh_front_ridge_17.quaternion.copy(endpoint_front_ridge_17.quaternion);
  }
  mesh_front_ridge_17.castShadow = options.castShadow ?? true;
  mesh_front_ridge_17.receiveShadow = options.receiveShadow ?? true;
  mesh_front_ridge_17.userData.sculptComponent = {"id": "front-ridge", "name": "Front tepal soft ridge", "level": "micro", "role": "detail", "importance": 0.6, "confidence": 0.5, "primitive": "tapered-sweep", "topologyClass": "surface-relief", "topologyRationale": "Soft vertical ridge shading on the large front tepal (reference x~200-260).", "geometryDescriptor": {"topologyIntent": "organic deformed mesh", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.08, 0.2177], "rx": 0.006, "rz": 0.0, "twist": 0.0}, {"position": [0.0, 0.3, 0.4046], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.5, 0.4473], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.7, 0.4793], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.88, 0.4365], "rx": 0.006, "rz": 0.02, "twist": 0.0}, {"position": [0.0, 0.96, 0.2994], "rx": 0.006, "rz": 0.0, "twist": 0.0}], "radialSegments": 6, "capEnds": true}, "handRefinement": {"module": "src/tulip-model/tulipRefinement.ts", "function": "applyRefinement", "rule": "Surface relief is baked into the refined shell geometry/vertex colour (overlap gap shading, central ridge, midrib) so separate relief meshes are hidden to avoid z-fighting."}}, "parent": "tepal-front", "attachment": {"parentId": "tepal-front", "parentSocket": "surface", "localStart": [0.0, 0.08, 0.2078], "localEnd": [0.0, 0.96, 0.2903], "contactType": "overlap", "embedDepth": 0.005, "gapTolerance": 0.01, "baseRadius": 0.0, "endRadius": 0.0, "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.04, "height": 0.9, "depth": 0.03, "units": "relative", "confidence": 0.5}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "front-ridge", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "petal-blue"}}, "material": "petal-blue", "materialLayers": ["petal-blue"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.58, "microRoughness": 0.62, "bumpAmplitude": 0.0, "normalPattern": "none (tiled normal map removed: produced star highlight); satin read comes from sheen lobe", "displacementPattern": "none", "occlusionPattern": "overlap seam shadow baked into inner-tepal edge vertex colour; inside-cup -32%", "edgeWearPattern": "lighter translucent rim band |u|>0.72", "notes": "venation = fine longitudinal albedo veins (17/5 cycles) in vertex colour"}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "front-ridge", "dominantAlbedo": "rgba(131, 152, 197, 1.0)", "secondaryAlbedo": "rgba(185, 203, 232, 1.0)", "materialClass": "unknown", "materialClassConfidence": 0.6, "roughnessEstimate": 0.367, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/home/claude/tulip/evidence/crops/petal.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.53}}};
  node_front_ridge_17.add(mesh_front_ridge_17);
  meshes["front-ridge"] = mesh_front_ridge_17;
  colliders["front-ridge"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "proxy bounds"};
  destructionGroups["front-ridge"] ??= [];
  destructionGroups["front-ridge"].push(node_front_ridge_17);

  // repetition system: tepal-whorls (InstancedMesh, radial, count=6, level=meso)
  {
    const parent = nodes["root"] ?? root;
    const geo = new THREE.BoxGeometry(1, 1, 1, 4, 4, 4);
    const mat = materialMap["petal-blue"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 });
    // Contract (PLAN_1.5 WS-E): instanceScale is ABSOLUTE, in the parent pivot's
    // local units -- it is never multiplied by the parent component's own declared
    // dimensional scale. This falls out of the same fix as componentTree: the pivot
    // Group this cluster is parented to always carries identity scale (dimensions are
    // baked into that component's OWN geometry, not exposed on the Group), so an
    // instanced fastener/tooth/spoke sized [0.05, 0.05, 0.05] renders at exactly that
    // size regardless of how non-uniformly its host component is shaped, and a
    // `radial` ring's placement stays circular instead of being squashed into an
    // ellipse by a non-uniform host.
    const scl = [0.1, 0.1, 0.1];
    const axis = new THREE.Vector3(0.0, 0.0, 1.0).normalize();
    const radius = 0.0;
    const seed = Math.abs(axis.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const perp = new THREE.Vector3().crossVectors(axis, seed).normalize();
    // One InstancedMesh = one draw call for all repeated parts (teeth/fasteners/spokes),
    // replacing the former per-instance Mesh clone loop (real-time perf principle).
    const cluster = new THREE.InstancedMesh(geo, mat, 6);
    const _m = new THREE.Matrix4();
    const _p = new THREE.Vector3();
    const _q = new THREE.Quaternion();
    const _s = new THREE.Vector3(scl[0], scl[1], scl[2]);
    for (let i = 0; i < 6; i++) {
      const ang = ((0.0) + (i * 360) / 6) * Math.PI / 180;
      const dir = perp.clone().applyQuaternion(new THREE.Quaternion().setFromAxisAngle(axis, ang));
      _p.copy(radius > 0 ? dir.clone().multiplyScalar(radius * 0.5) : new THREE.Vector3());
      _q.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);
      _m.compose(_p, _q, _s);
      cluster.setMatrixAt(i, _m);
    }
    cluster.instanceMatrix.needsUpdate = true;
    cluster.castShadow = options.castShadow ?? true;
    cluster.receiveShadow = options.receiveShadow ?? true;
    cluster.name = "tepal-whorls";
    parent.add(cluster);
  }

  root.userData.sculptRuntime = { nodes, meshes, sockets, colliders, destructionGroups } satisfies ProceduralModelRuntime;
  root.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  root.userData.actionReadiness = {
    note: 'Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets.',
  };
  return root;
}

export function createBlueTulipLookDevLights(
  mode: 'neutral' | 'grazing' | 'reference' = 'neutral',
): THREE.Group {
  const lights = new THREE.Group();
  lights.name = "Blue Tulip look-dev lights";
  const hemi = new THREE.HemisphereLight(
    mode === 'reference' ? 0xfff0d6 : 0xf2f4ff,
    0x363b42,
    mode === 'grazing' ? 0.28 : mode === 'reference' ? 0.72 : 0.85,
  );
  lights.add(hemi);
  const key = new THREE.DirectionalLight(
    mode === 'reference' ? 0xffcf8a : 0xfff4e8,
    mode === 'grazing' ? 4.2 : mode === 'reference' ? 2.6 : 2.15,
  );
  if (mode === 'grazing') key.position.set(7.5, 1.1, 4.0);
  else if (mode === 'reference') key.position.set(-4.5, 7.5, 5.0);
  else key.position.set(-4.0, 6.0, 5.5);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  key.shadow.bias = -0.00025;
  key.shadow.normalBias = 0.018;
  key.shadow.radius = 7;
  key.shadow.blurSamples = 24;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 30;
  key.shadow.camera.left = -2.6;
  key.shadow.camera.right = 2.6;
  key.shadow.camera.top = 2.6;
  key.shadow.camera.bottom = -2.6;
  key.shadow.camera.updateProjectionMatrix();
  lights.add(key);
  const fill = new THREE.DirectionalLight(0xa8c4ff, mode === 'grazing' ? 0.12 : 0.42);
  fill.position.set(4.0, 3.0, 3.5);
  lights.add(fill);
  const rim = new THREE.DirectionalLight(0xfff1c4, mode === 'grazing' ? 0.28 : 0.85);
  rim.position.set(0.5, 4.5, -6.0);
  lights.add(rim);
  lights.userData.reviewMode = mode;
  lights.userData.lightingFromPhoto = [{"id": "key", "type": "directional", "direction": [-0.4, 0.7, 0.6], "intensity": 1.6, "color": "#FFFFFF", "evidence": "brighter upper-left tepal shoulders vs darker right/lower seams in ref.png"}, {"id": "fill", "type": "hemisphere", "skyColor": "#EAF0FA", "groundColor": "#C8CDB8", "intensity": 0.7, "evidence": "shadow lift: seam p10 luma 66 not black"}, {"id": "rim", "type": "directional", "direction": [0.5, 0.3, -0.8], "intensity": 0.5, "color": "#DDE8FF", "evidence": "soft edge separation of right tepal from bg"}, {"id": "environment", "type": "environment", "map": "neutral-studio", "intensity": 0.6, "exposure": 1.0, "toneMapping": "ACESFilmic", "contactShadow": "soft ground contact shadow / ambient occlusion in tepal seams", "evidence": "flat studio-style cut-out background"}];
  lights.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  return lights;
}

// PBR materials (clearcoat/iridescence/transmission/anisotropy) need an environment
// map to visually behave as intended — call this once per renderer and assign the
// result to scene.environment before rendering. No external HDR asset required.
export function createBlueTulipEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return texture;
}

// Plan 1.3 §3.2 — auto-framing by bounding box. The Divine Eye can only compare a
// render to the reference if the object is FRAMED consistently (an object framed
// differently scores as wrong even when its shape is right). This positions the camera
// deterministically from the object's bounding box so it fills the frame at a stable
// margin, and sets near/far to the object scale. Call after adding the model to the
// scene, and again on resize (after updating camera.aspect).
export function frameBlueTulipCamera(
  camera: THREE.PerspectiveCamera,
  object: THREE.Object3D,
  options: { margin?: number; azimuthDeg?: number; elevationDeg?: number } = {},
): void {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const margin = options.margin ?? 1.15;
  const maxDim = Math.max(size.x, size.y, size.z) * margin;
  const fov = (camera.fov * Math.PI) / 180;
  // distance so the largest object dimension fits vertically in the frame
  const distance = (maxDim / 2) / Math.tan(fov / 2);
  const az = ((options.azimuthDeg ?? 0) * Math.PI) / 180;
  const el = ((options.elevationDeg ?? 0) * Math.PI) / 180;
  const dir = new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    Math.cos(az) * Math.cos(el),
  );
  camera.position.copy(center).addScaledVector(dir, distance);
  camera.near = Math.max(0.01, distance - maxDim);
  camera.far = distance + maxDim * 2;
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}

// Plan 1.3 §3.2c — PRESENTATION composer (DOF + bloom). CRITICAL (R-POSTFX): this is
// for the showcase/hero render ONLY. The Divine Eye's EVALUATION render MUST use a
// plain renderer with NO composer — bloom blows highlights and DOF blurs edges, which
// would corrupt the deterministic IoU/DCD/edge/blowout signals. Enable dof/bloom ONLY
// when the reference photo actually exhibits them (detect_reference_effects.py authorizes).
export function createBlueTulipPresentationComposer(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  options: { dof?: boolean; bloom?: boolean; bloomStrength?: number; dofFocus?: number; dofAperture?: number } = {},
): EffectComposer {
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  if (options.dof) {
    composer.addPass(new BokehPass(scene, camera, {
      focus: options.dofFocus ?? 10.0,
      aperture: options.dofAperture ?? 0.0002,
      maxblur: 0.01,
    }));
  }
  if (options.bloom) {
    const size = new THREE.Vector2();
    renderer.getSize(size);
    composer.addPass(new UnrealBloomPass(size, options.bloomStrength ?? 0.4, 0.4, 0.85));
  }
  return composer;
}

export function configureBlueTulipRenderer(renderer: THREE.WebGLRenderer): void {
  // Load-bearing for view-dependent finishes (anodized / Doppler): without ACES + sRGB
  // the environment reflection reads flat/washed instead of a believable metal response.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

export function createBlueTulipInspectControls(
  camera: THREE.Camera,
  domElement: HTMLElement,
): OrbitControls {
  // View-dependent finishes only read correctly once the user orbits — their color
  // comes from the environment reflection, not albedo, so free rotation matters here.
  const controls = new OrbitControls(camera, domElement);
  controls.enableDamping = true;
  controls.minDistance = 1.0;
  controls.maxDistance = 8.0;
  controls.autoRotate = false;
  return controls;
}
