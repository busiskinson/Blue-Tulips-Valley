/* Hand refinement for the img2threejs-generated blue tulip (refine-code).

   Why this file exists: the generated factory (createObjectModel.ts) owns the hierarchy, pivots,
   sockets, materials and the stem/receptacle/bloom-core geometry. Its primitive set has no tapered,
   cupped thin shell, so tepals and leaves come out as lens-section sweeps or connector cylinders.
   This module rebuilds ONLY those surfaces from the same spec data the generator embedded on each
   mesh (userData.sculptComponent.geometryDescriptor.taperedSweep + handRefinement), so the spec
   stays the single source of truth. The rules are mirrored in object-sculpt-spec.json
   (geometryDescriptor.handRefinement) — keep both in sync. */

import * as THREE from 'three';

type Station = { position: [number, number, number]; rx: number; rz: number };
type Comp = { id: string; geometryDescriptor?: { taperedSweep?: { stations: Station[] }; handRefinement?: { angleDegrees?: number } } };

export type TulipRig = {
  root: THREE.Object3D;
  bloom: THREE.Object3D;
  stem: THREE.Object3D | null;
  tepals: { pivot: THREE.Object3D; axis: THREE.Vector3; outer: boolean; phase: number }[];
  leaves: { pivot: THREE.Object3D; axis: THREE.Vector3; phase: number }[];
  materials: THREE.Material[];
};

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/* Interpolates station data by arc parameter v in [0,1] (Catmull-Rom over station index). */
function stationSampler(stations: Station[]) {
  const pts = stations.map((s) => new THREE.Vector3(...s.position));
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const n = stations.length - 1;
  const lerpField = (v: number, f: (s: Station) => number) => {
    const x = Math.min(n, Math.max(0, v * n)), i = Math.min(n - 1, Math.floor(x)), t = x - i;
    const a = f(stations[Math.max(0, i - 1)]), b = f(stations[i]), c = f(stations[i + 1]), d = f(stations[Math.min(n, i + 2)]);
    // Catmull-Rom scalar spline keeps the width profile smooth instead of piecewise linear
    const t2 = t * t, t3 = t2 * t;
    return Math.max(0, 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3));
  };
  return { curve, hw: (v: number) => lerpField(v, (s) => s.rz) };
}

/* Closed thin shell: outer grid + inner grid + side strips on all four borders. */
function shellGeometry(
  NU: number, NV: number,
  outer: (u: number, v: number, out: THREE.Vector3) => void,
  inner: ((u: number, v: number, out: THREE.Vector3) => void) | null, // null = single surface (field LOD)
  shade: (u: number, v: number, back: boolean) => [number, number, number],
): THREE.BufferGeometry {
  const pos: number[] = [], col: number[] = [], uv: number[] = [], idx: number[] = [];
  const p = new THREE.Vector3();
  const grid = (fn: typeof outer, back: boolean) => {
    const start = pos.length / 3;
    for (let j = 0; j <= NV; j++) {
      const v = j / NV;
      for (let i = 0; i <= NU; i++) {
        const u = (2 * i) / NU - 1;
        fn(u, v, p);
        pos.push(p.x, p.y, p.z);
        col.push(...shade(u, v, back));
        uv.push(i / NU, v);
      }
    }
    for (let j = 0; j < NV; j++) {
      for (let i = 0; i < NU; i++) {
        const a = start + j * (NU + 1) + i, c = a + NU + 1;
        if (back) idx.push(a, a + 1, c, a + 1, c + 1, c);
        else idx.push(a, c, a + 1, a + 1, c, c + 1);
      }
    }
    return start;
  };
  const so = grid(outer, false);
  if (!inner) return finish();
  const si = grid(inner, true);
  // Rim strips. Duplicated vertices keep a crisp, lit edge (reads as petal thickness).
  const rim = (ring: number[]) => {
    const start = pos.length / 3;
    for (const k of ring) {
      for (const base of [so, si]) {
        pos.push(pos[(base + k) * 3], pos[(base + k) * 3 + 1], pos[(base + k) * 3 + 2]);
        col.push(col[(base + k) * 3] * 1.08, col[(base + k) * 3 + 1] * 1.08, col[(base + k) * 3 + 2] * 1.06);
        uv.push(uv[(base + k) * 2], uv[(base + k) * 2 + 1]);
      }
    }
    for (let m = 0; m < ring.length - 1; m++) {
      const a = start + m * 2, b = a + 2;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
  };
  const left: number[] = [], right: number[] = [], top: number[] = [], bottom: number[] = [];
  for (let j = 0; j <= NV; j++) { left.push(j * (NU + 1)); right.push(j * (NU + 1) + NU); }
  for (let i = 0; i <= NU; i++) { top.push(NV * (NU + 1) + i); bottom.push(i); }
  rim(left.reverse()); rim(right); rim(top.reverse()); rim(bottom);
  return finish();
  function finish() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }
}

/* Petal style. Defaults reproduce the reference (closed, rounded bud). The website's field tulips
   use ROMANTIC: pointed ogival tips, a visible gap between the tips (inner tepals show through),
   a soft outward curl at the top and a slimmer base, so the bloom reads as a tulip, not a ball. */
export type TepalStyle = { tip: number; drop: number; flare: number; base: number; open: number; upright: number; lift: number };
export const REFERENCE_STYLE: TepalStyle = { tip: 0, drop: 0.03, flare: 0.03, base: 0, open: 0, upright: 0, lift: 0 };
export const ROMANTIC_STYLE: TepalStyle = { tip: 0.84, drop: 0.07, flare: 0.04, base: 0.2, open: 0.03, upright: 1, lift: 0.1 };

/* One tepal in bloom space. phi = 0 faces the reference camera (+Z). */
export function buildCuppedTepal(stations: Station[], phiDeg: number, outer: boolean, quality = 1, single = false, style: TepalStyle = REFERENCE_STYLE): THREE.BufferGeometry {
  const { curve, hw } = stationSampler(stations);
  const phi = THREE.MathUtils.degToRad(phiDeg);
  const widthGain = outer ? 1.28 : 1.12;
  const radial = outer ? 1.08 : 1.0;
  const heightGain = outer ? 1 : 0.975;        // form pass: inner tips stay just under the outer crown
  const lift = 0;                              // review blockout#1: inner tips poked above the rounded top
  const c = new THREE.Vector3();
  // Shoulder = widest point of the cup. style.upright keeps the upper tepal near that radius
  // (the reference bud folds over the top into a dome), style.lift lengthens the tips upward.
  let shoulder = 0;
  for (let k = 0; k <= 40; k++) shoulder = Math.max(shoulder, curve.getPoint(k / 40, c).z);
  const geo = (u: number, v: number, inset: number, out: THREE.Vector3) => {
    curve.getPoint(v, c);                      // c.z = cup radius, c.y = height
    const up = style.upright * smooth(0.55, 0.9, v);
    c.z += (Math.max(c.z, shoulder * (1 - 0.16 * smooth(0.6, 1, v))) - c.z) * up;
    c.y += style.lift * smooth(0.6, 1, v);
    const r = Math.max(0.02, c.z * radial);
    // style.tip narrows the upper tepal to a point; style.base pinches the cup's foot
    const taper = 1 - style.tip * Math.pow(smooth(0.45, 1, v), 0.9);
    const a = Math.min(1.05, (hw(v) * widthGain) / r) * Math.max(0.06, taper);
    const th = phi + u * a;
    const bulge = 1 + 0.1 * (1 - u * u) * Math.pow(Math.sin(Math.PI * Math.min(1, v * 1.1)), 0.6);
    const tuck = 1 - 0.035 * u * u * smooth(0.15, 0.7, v);       // edges slip under the neighbour
    const flare = style.flare * smooth(0.72, 1, v) * (1 - 0.6 * u * u) * (outer ? 1 : 0.7);  // outward curl at the tip
    const pinch = 1 - style.base * (1 - smooth(0, 0.35, v));                                // slimmer foot
    const opening = 1 + style.open * smooth(0.4, 1, v) * (outer ? 1 : 0.8);                // crown a little open
    const rr = r * bulge * tuck * pinch * opening + flare - inset;
    const y = c.y * (1 + lift) * heightGain - style.drop * u * u * smooth(0.45, 1, v); // pointed, ogival tip
    out.set(Math.sin(th) * rr, y, Math.cos(th) * rr);
  };
  const thick = (v: number) => 0.016 * (1 - v) + 0.006 * v;
  const venSeed = phiDeg * 0.07;
  return shellGeometry(
    Math.max(4, Math.round(14 * quality)), Math.max(6, Math.round(22 * quality)),
    (u, v, o) => geo(u, v, 0, o),
    single ? null : (u, v, o) => geo(u, v, thick(v), o),
    (u, v, back) => {
      let b = 0.72 + 0.28 * smooth(0, 0.5, v) + 0.12 * smooth(0.55, 1, v); // darker base -> light tip
      b += 0.1 * smooth(0.72, 1, Math.abs(u));                              // thin, translucent rims
      // broad streaks only; fine veins live in the generated vein texture (vertex grid is too coarse)
      const streak = 0.5 + 0.5 * Math.cos(u * Math.PI * 5 + v * 2.2 + venSeed * 1.7);
      b *= 1 - 0.035 * streak * smooth(0.1, 0.45, v);
      b += 0.035 * Math.exp(-u * u * 60) * smooth(0.2, 0.8, v);             // central keel highlight
      if (!outer) b *= 0.9 * (1 - 0.38 * smooth(0.55, 0.95, Math.abs(u)) * smooth(0.05, 0.3, v)); // shadowed seams under outer edges
      if (back) b *= 0.68;                                                  // inside of the cup
      return [b * 0.96, b * 0.985, b * 1.03];
    },
  );
}

/* Channelled lanceolate leaf in the leaf pivot's parent space. */
export function buildFoldedLeaf(stations: Station[], side: 1 | -1, quality = 1, single = false): THREE.BufferGeometry {
  const { curve, hw } = stationSampler(stations);
  const T = new THREE.Vector3(), C = new THREE.Vector3(), S = new THREE.Vector3(), N = new THREE.Vector3();
  const Z = new THREE.Vector3(0, 0, 1);
  const geo = (u: number, v: number, inset: number, out: THREE.Vector3) => {
    curve.getPoint(v, C);
    curve.getTangent(Math.min(0.999, Math.max(0.001, v)), T);
    S.crossVectors(T, Z).normalize();
    S.applyAxisAngle(T, side * 0.25 * v);              // gentle twist toward the tip
    N.crossVectors(S, T).normalize();
    if (N.z < 0) N.negate();
    const w = hw(v);
    // V channel, edges lifted toward the viewer; near the base the blade wraps around the stem
    const fold = (0.38 + 0.8 * (1 - smooth(0, 0.22, v))) * Math.abs(u) * w;
    const curl = 0.05 * Math.pow(Math.abs(u), 3) * w * smooth(0.2, 0.9, v);
    out.copy(C).addScaledVector(S, u * w).addScaledVector(N, fold + curl - inset);
  };
  return shellGeometry(
    Math.max(2, Math.round(8 * quality)), Math.max(8, Math.round(26 * quality)),
    (u, v, o) => geo(u, v, 0, o),
    single ? null : (u, v, o) => geo(u, v, 0.012, o),
    (u, v, back) => {
      let b = 0.92 + 0.12 * (1 - v) * smooth(0.0, 0.25, 1 - v) * 0.6;
      b += 0.28 * smooth(0.82, 1, Math.abs(u));              // pale glaucous margin
      b += 0.12 * Math.exp(-u * u * 80);                      // midrib
      b *= 1 - 0.08 * smooth(0.75, 1, v);                     // tip slightly darker
      if (back) b *= 1.08;                                     // underside is paler, waxier
      return [b * 0.98, b, b * 0.97];
    },
  );
}

/* Procedural vein texture (code-generated, no image asset): fine longitudinal veins on a white
   ground, UV u across the tepal and v from base to tip. Multiplies the vertex-colour gradient. */
function makeVeinTexture(): THREE.CanvasTexture | null {
  if (typeof document === 'undefined') return null;
  const W = 256, H = 512, c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  if (!g) return null;
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, W, H);
  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let k = 0; k < 46; k++) {
    const x0 = (k + 0.5) / 46 * W + (rnd() - 0.5) * 3, wob = 1.5 + rnd() * 3, ph = rnd() * 6;
    const dark = 0.86 + rnd() * 0.08, w = 1.4 + rnd() * 1.6;
    g.strokeStyle = `rgba(${Math.round(150 * dark)},${Math.round(160 * dark)},${Math.round(200 * dark)},${0.1 + rnd() * 0.1})`;
    g.lineWidth = w;
    g.beginPath();
    for (let y = H * 0.06; y <= H * 0.97; y += 8) {
      const x = x0 + Math.sin(y / 70 + ph) * wob;
      if (y === H * 0.06) g.moveTo(x, H - y); else g.lineTo(x, H - y);
    }
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function comp(o: THREE.Object3D): Comp | undefined {
  return o.userData.sculptComponent as Comp | undefined;
}

/* Applies the refinement in place and returns a rig for animation. */
export function applyRefinement(model: THREE.Group, opts: { quality?: number } = {}): TulipRig {
  const q = opts.quality ?? 1;
  const rt = model.userData.sculptRuntime as { nodes: Record<string, THREE.Object3D>; meshes: Record<string, THREE.Mesh> };
  const { nodes, meshes } = rt;
  const rig: TulipRig = { root: model, bloom: nodes['bloom'], stem: nodes['root'] ?? null, tepals: [], leaves: [], materials: [] };
  const mats = new Set<THREE.Material>();

  const reseat = (id: string, geometry: THREE.BufferGeometry, anchor: THREE.Vector3) => {
    const mesh = meshes[id], pivot = nodes[id];
    mesh.geometry.dispose();
    mesh.geometry = geometry;
    // Pivot sits on the organ's base so rotations hinge naturally; mesh stays in parent space.
    pivot.position.copy(anchor);
    pivot.rotation.set(0, 0, 0);
    pivot.scale.set(1, 1, 1);
    mesh.position.copy(anchor).negate();
    mesh.quaternion.identity();
    mesh.scale.set(1, 1, 1);
    (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((m) => mats.add(m));
  };

  for (const id of Object.keys(meshes)) {
    const c = comp(meshes[id]);
    const st = c?.geometryDescriptor?.taperedSweep?.stations;
    if (id.startsWith('tepal-') && st) {
      const outer = !id.startsWith('tepal-inner');
      const ang = c?.geometryDescriptor?.handRefinement?.angleDegrees ?? 0;
      const g = buildCuppedTepal(st, ang, outer, q);
      const s0 = st[0].position, phi = THREE.MathUtils.degToRad(ang);
      const anchor = new THREE.Vector3(Math.sin(phi) * s0[2], s0[1], Math.cos(phi) * s0[2]);
      reseat(id, g, anchor);
      rig.tepals.push({ pivot: nodes[id], axis: new THREE.Vector3(Math.cos(phi), 0, -Math.sin(phi)), outer, phase: ang * 0.05 });
    } else if ((id === 'leaf-left' || id === 'leaf-right') && st) {
      const g = buildFoldedLeaf(st, id === 'leaf-left' ? 1 : -1, q);
      reseat(id, g, new THREE.Vector3(...st[0].position));
      rig.leaves.push({ pivot: nodes[id], axis: new THREE.Vector3(0, 0, id === 'leaf-left' ? 1 : -1), phase: id === 'leaf-left' ? 0 : 1.7 });
    } else if (/^(crease-|midrib-|front-ridge)/.test(id)) {
      meshes[id].visible = false; // relief is baked into the refined shells
    }
  }

  // Bloom core: an inner volume that only shows through the tepal gaps -> darker, smaller.
  const core = meshes['bloom'];
  if (core) {
    core.scale.multiplyScalar(0.86);
    const m = (core.material as THREE.MeshStandardMaterial).clone();
    m.color.set('#2c3a66');
    m.roughness = 0.8;
    core.material = m;
    mats.add(m);
  }

  // The generator realises repetitionSystems "tepal-whorls" as a radius-0 InstancedMesh of six
  // 0.1-scale copies inside the cup. The six explicit tepal components already realise that whorl,
  // so the duplicate is removed (it was invisible, cost a draw call and was an unnamed part).
  const dup = model.getObjectByName('tepal-whorls') as THREE.InstancedMesh | undefined;
  if (dup) { dup.removeFromParent(); dup.dispose(); }

  // Every visible mesh's material (stem, receptacle and sheaths too, not only re-seated organs)
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) (Array.isArray(m.material) ? m.material : [m.material]).forEach((x) => mats.add(x));
  });
  // Sheaths clasp the stem: slimmer than the generated connector so they read as a wrapped leaf base.
  for (const id of ['sheath-left', 'sheath-right']) if (meshes[id]) meshes[id].scale.set(0.62, 1, 0.62);

  // Materials: botanical dielectric, vertex-colour gradients carry the reference shading.
  const veins = makeVeinTexture();
  for (const m of mats) {
    const sm = m as THREE.MeshPhysicalMaterial;
    const id: string = (sm.userData?.sculptMaterial?.id as string) ?? '';
    // The factory's procedural texture set forces colour to white and multiplies a synthetic albedo
    // map, plus AO/displacement maps that would warp the refined shells. Vertex-colour gradients
    // (reference-derived) carry albedo here; a faint normal map is kept for micro surface breakup.
    for (const k of ['map', 'aoMap', 'displacementMap', 'bumpMap', 'roughnessMap'] as const) {
      const tex = (sm as any)[k] as THREE.Texture | null;
      if (tex) { tex.dispose(); (sm as any)[k] = null; }
    }
    if (sm.normalMap) sm.normalScale.setScalar(0.12);
    sm.metalness = 0;
    if ('clearcoat' in sm) sm.clearcoat = id === 'stem-green' ? 0.12 : 0;
    if (id.includes('petal') || id.includes('crease')) {
      sm.vertexColors = true;
      sm.color.set('#b4c6f4'); // material pass: calibrated to reference petal mean (146,164,203)
      sm.roughness = 0.58;
      if ('sheen' in sm) { sm.sheen = 0.65; sm.sheenRoughness = 0.5; sm.sheenColor.set('#c9d8ff'); }
      sm.side = THREE.FrontSide;
    } else if (id.includes('leaf')) {
      sm.vertexColors = true;
      sm.color.set('#6e7a3e'); // material pass: reference leaf mean (100,115,73)
      sm.roughness = 0.64;
      if ('sheen' in sm) { sm.sheen = 0.45; sm.sheenRoughness = 0.7; sm.sheenColor.set('#c4cba6'); }
      sm.side = THREE.FrontSide;
    } else if (id.includes('stem')) {
      sm.color.set('#98b015'); if ('sheen' in sm) sm.sheen = 0; // material pass: reference stem mean (140,161,43)
      sm.roughness = 0.42;
    }
    if (id.includes('petal') && sm.normalMap) { sm.normalMap.dispose(); sm.normalMap = null; } // tiled map drew a star highlight on the satin
    if (id === 'petal-blue') sm.map = veins;
    sm.needsUpdate = true;
    rig.materials.push(sm);
  }
  // Generated connector meshes (sheaths, receptacle, bloom core) carry no colour attribute; with a
  // vertex-coloured material that renders black. Give them a neutral one.
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || m.geometry.getAttribute('color')) return;
    const vc = (Array.isArray(m.material) ? m.material : [m.material]).some((x) => (x as THREE.MeshStandardMaterial).vertexColors);
    if (!vc) return;
    const n = m.geometry.getAttribute('position').count;
    m.geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(n * 3).fill(0.92), 3));
  });
  return rig;
}

/* Breeze: whole-flower sway from the stem, tepals breathe on their base hinge, leaves flutter. */
export function animateTulip(rig: TulipRig, t: number, strength = 1) {
  const s = strength;
  if (rig.bloom) {
    rig.bloom.rotation.z = Math.sin(t * 0.9) * 0.035 * s + Math.sin(t * 2.3) * 0.008 * s;
    rig.bloom.rotation.x = Math.sin(t * 0.7 + 1.1) * 0.025 * s;
  }
  for (const tp of rig.tepals) {
    const a = (Math.sin(t * 1.3 + tp.phase) * 0.5 + 0.5) * (tp.outer ? 0.05 : 0.03) * s;
    tp.pivot.quaternion.setFromAxisAngle(tp.axis, a);
  }
  for (const lf of rig.leaves) {
    lf.pivot.quaternion.setFromAxisAngle(lf.axis, Math.sin(t * 1.1 + lf.phase) * 0.03 * s);
  }
}

export function disposeTulip(model: THREE.Object3D) {
  const done = new Set<unknown>();
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) {
      if (!done.has(m.geometry)) { done.add(m.geometry); m.geometry.dispose(); }
      (Array.isArray(m.material) ? m.material : [m.material]).forEach((mat) => {
        if (done.has(mat)) return;
        done.add(mat);
        for (const v of Object.values(mat)) if (v instanceof THREE.Texture && !done.has(v)) { done.add(v); v.dispose(); }
        mat.dispose();
      });
    }
  });
}

/* Field LOD of the same reconstructed tulip, for the instanced flowers around the camera.
   Built from the SAME spec station data the hero uses (tepal/leaf taperedSweep stations, stem
   tubePath, bloom anchor), as single-surface shells, then normalised to the field's unit tulip
   (base at y = 0, top of the bloom at y = 1). Returns two geometries: the bloom (vertex colour =
   shading multiplier; the instance colour supplies the blue) and the greens (baked green colour). */
export function buildFieldTulip(model: THREE.Group, quality = 0.5, style: TepalStyle = REFERENCE_STYLE): { head: THREE.BufferGeometry; greens: THREE.BufferGeometry } {
  const rt = model.userData.sculptRuntime as { nodes: Record<string, THREE.Object3D>; meshes: Record<string, THREE.Mesh> };
  const spec = (id: string) => (rt.meshes[id]?.userData.sculptComponent ?? {}) as Comp & { geometryDescriptor?: { tubePath?: { points: [number, number, number][]; radius?: number } } };
  const bloomAt = rt.nodes['bloom'] ? rt.nodes['bloom'].position.clone() : new THREE.Vector3();
  const heads: THREE.BufferGeometry[] = [], greens: THREE.BufferGeometry[] = [];

  for (const id of ['tepal-front', 'tepal-outer-a', 'tepal-outer-b', 'tepal-inner-a', 'tepal-inner-b', 'tepal-inner-c']) {
    const c = spec(id), st = c.geometryDescriptor?.taperedSweep?.stations;
    if (!st) continue;
    const g = buildCuppedTepal(st, c.geometryDescriptor?.handRefinement?.angleDegrees ?? 0, !id.startsWith('tepal-inner'), quality, true, style);
    g.translate(bloomAt.x, bloomAt.y, bloomAt.z);
    heads.push(g);
  }

  const lin = (hex: string) => new THREE.Color(hex); // THREE.Color stores linear values
  const leafC = lin('#6e7a3e'), stemC = lin('#98b015'); // reference leaf / stem colours
  for (const [id, side] of [['leaf-left', 1], ['leaf-right', -1]] as const) {
    const st = spec(id).geometryDescriptor?.taperedSweep?.stations;
    if (!st) continue;
    const g = buildFoldedLeaf(st, side, quality, true);
    const col = g.getAttribute('color');
    for (let i = 0; i < col.count; i++) col.setXYZ(i, col.getX(i) * leafC.r, col.getY(i) * leafC.g, col.getZ(i) * leafC.b);
    g.deleteAttribute('uv');
    greens.push(g);
  }

  // Stem: the spec's tube path as a slim 6-sided tube, slightly darker at the foot.
  const tp = spec('root').geometryDescriptor?.tubePath;
  if (tp) {
    const curve = new THREE.CatmullRomCurve3(tp.points.map((p) => new THREE.Vector3(...p)));
    const g = new THREE.TubeGeometry(curve, 10, (tp.radius ?? 0.045) * 0.9, 6, false);
    g.deleteAttribute('uv');
    const pos = g.getAttribute('position'), col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const k = 0.7 + 0.3 * smooth(-1.95, 0, pos.getY(i));
      col.set([stemC.r * k, stemC.g * k, stemC.b * k], i * 3);
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    greens.push(g);
  }

  // Merge (indexed, same attribute set) and normalise to the field unit.
  const merge = (list: THREE.BufferGeometry[]) => {
    const pos: number[] = [], nor: number[] = [], col: number[] = [], idx: number[] = [];
    for (const g of list) {
      const base = pos.length / 3, p = g.getAttribute('position'), n = g.getAttribute('normal'), c = g.getAttribute('color');
      for (let i = 0; i < p.count; i++) { pos.push(p.getX(i), p.getY(i), p.getZ(i)); nor.push(n.getX(i), n.getY(i), n.getZ(i)); col.push(c.getX(i), c.getY(i), c.getZ(i)); }
      const ix = g.getIndex();
      if (ix) for (let i = 0; i < ix.count; i++) idx.push(base + ix.getX(i));
      else for (let i = 0; i < p.count; i++) idx.push(base + i);
      g.dispose();
    }
    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    out.setIndex(idx);
    return out;
  };
  const head = merge(heads), green = merge(greens);
  head.computeBoundingBox(); green.computeBoundingBox();
  const minY = Math.min(head.boundingBox!.min.y, green.boundingBox!.min.y), maxY = head.boundingBox!.max.y;
  const s = 1 / (maxY - minY);
  for (const g of [head, green]) {
    g.translate(-(tp ? tp.points[tp.points.length - 1][0] : 0), -minY, 0); // stem foot on the instance origin
    g.scale(s, s, s);
    g.computeBoundingSphere();
  }
  return { head, greens: green };
}
