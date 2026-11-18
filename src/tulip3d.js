/* Real 3D blue tulips for the flowers closest to the camera.

   Built from the img2threejs analysis (image-analysis.md, detail inventory):
   6 overlapping tepals (3 outer, 3 inner) with a pointed, slightly flared tip,
   soft longitudinal creases, darker petal base, a gently curved tapering stem,
   and two long arched leaves with a midrib groove.

   Unit tulip: height 1. Stem 0 to 0.58, cup 0.55 to 0.97. Each instance is scaled
   to the same height the old flat flower had, and placed at the same spot, so the
   field looks the same, only closer flowers become real 3D shapes.

   Far flowers stay as the cheap point sprites. A pool of about 1000 real tulips follows
   the camera and takes over the nearest sprites (those sprites are hidden while swapped). */

import * as THREE from "three";

const TAU = Math.PI * 2;

/* ---------- small geometry helpers ---------- */
const newB = () => ({ pos: [], col: [], idx: [] });
const vert = (b, x, y, z, r, g, bl) => { b.pos.push(x, y, z); b.col.push(r, g, bl); return b.pos.length / 3 - 1; };
function quads(b, start, rows, cols) {
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = start + j * (cols + 1) + i, c = a + cols + 1;
      b.idx.push(a, c, a + 1, a + 1, c, c + 1);
    }
  }
}
function toGeo(b) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(b.pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(b.col, 3));
  g.setIndex(b.idx);
  g.computeVertexNormals();
  return g;
}

/* The bloom. open: 0 = tight cup, 1 = slightly more open (natural variation). */
function buildHead(open) {
  const b = newB();
  const NP = 6, NU = 8, NV = 10, R = 0.105, H0 = 0.55, HH = 0.42;
  for (let k = 0; k < NP; k++) {
    const outer = k % 2 === 0;
    const base = (k / NP) * TAU + (outer ? 0 : 0.05);
    const start = b.pos.length / 3;
    for (let j = 0; j <= NV; j++) {
      const v = j / NV;
      const prof = 0.3 + 0.7 * Math.pow(Math.sin(Math.PI * 0.85 * v), 0.8);          // cup outline
      const flare = (0.16 + 0.55 * open) * Math.pow(Math.max(0, v - 0.7) / 0.3, 2);   // tips curl outward
      const width = Math.pow(1 - Math.pow(v, 2.6), 0.5) * 0.9 + 0.1;                   // pointed tip
      for (let i = 0; i <= NU; i++) {
        const c = (2 * i) / NU - 1;
        const a = base + c * (Math.PI / NP) * 1.35 * width;
        const bulge = 1 + 0.14 * (1 - c * c);                  // each petal is convex across
        const crease = 1 + 0.01 * Math.cos(c * Math.PI * 2);   // soft longitudinal creases
        const r = R * (prof + flare) * bulge * crease * (outer ? 1.05 : 0.92) * (1 + open * 0.18 * v);
        const y = H0 + v * HH * (outer ? 1 : 0.97);
        // shade: darker at the base, lighter toward the tip and edges, inner petals slightly darker
        let s = 0.6 + 0.4 * Math.pow(v, 0.7) + 0.08 * Math.pow(Math.abs(c), 3) + 0.035 * Math.cos(c * Math.PI * 2);
        if (!outer) s *= 0.9;
        vert(b, r * Math.cos(a), y, r * Math.sin(a), s, s, s);
      }
    }
    quads(b, start, NV, NU);
  }
  return toGeo(b);
}

/* Stem and two long leaves (all green, vertex colored). */
function buildGreens() {
  const b = newB();
  // Stem: gently curved, tapering, darker at the foot
  const RAD = 6, RINGS = 12;
  for (let j = 0; j <= RINGS; j++) {
    const t = j / RINGS, y = t * 0.585, r = 0.015 - 0.007 * t, cx = 0.018 * Math.sin(t * Math.PI * 0.9);
    for (let i = 0; i < RAD; i++) {
      const a = (i / RAD) * TAU;
      vert(b, cx + r * Math.cos(a), y, r * Math.sin(a), 0.1 + 0.1 * t, 0.22 + 0.14 * t, 0.09 + 0.05 * t);
    }
  }
  for (let j = 0; j < RINGS; j++) {
    for (let i = 0; i < RAD; i++) {
      const a = j * RAD + i, n = j * RAD + ((i + 1) % RAD);
      b.idx.push(a, a + RAD, n, n, a + RAD, n + RAD);
    }
  }
  // Leaves: arch outward, pointed tip, midrib groove, lighter along the midrib
  const NL = 14, NC = 4;
  [{ phi: 0.5, L: 1.0 }, { phi: 0.5 + Math.PI - 0.35, L: 0.84 }].forEach(({ phi, L }) => {
    const dx = Math.cos(phi), dz = Math.sin(phi), px = -dz, pz = dx;
    const start = b.pos.length / 3;
    for (let j = 0; j <= NL; j++) {
      const t = j / NL;
      const out = 0.34 * L * Math.pow(t, 1.35) + 0.05 * Math.pow(t, 4);
      const yc = 0.02 + 0.8 * L * t - 0.28 * L * t * t;
      const hw = 0.04 * Math.pow(Math.sin(Math.PI * (0.1 + 0.9 * t)), 0.8);
      for (let i = 0; i <= NC; i++) {
        const c = (2 * i) / NC - 1;
        const y = yc + hw * (0.35 * c * c - 0.18 * Math.exp(-c * c * 18));
        const rib = 0.1 * Math.exp(-c * c * 30);
        vert(b, dx * out + px * hw * c + 0.012, y, dz * out + pz * hw * c, 0.08 + 0.14 * t + rib, 0.2 + 0.22 * t + rib, 0.07 + 0.09 * t + rib * 0.6);
      }
    }
    quads(b, start, NL, NC);
  });
  return toGeo(b);
}

/* Wind: bends each vertex more the higher it is. Same sway as the flat flowers. */
function addWind(material, uTime, flutter, key) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uTime;
    shader.vertexShader =
      "uniform float uTime;\n" +
      shader.vertexShader.replace(
        "#include <begin_vertex>",
        `vec3 transformed = vec3(position);
        vec3 ip = instanceMatrix[3].xyz;
        float h = max(position.y, 0.0);
        float seed = fract(sin(ip.x * 12.9898 + ip.z * 78.233) * 43758.5453);
        float sway = sin(uTime * 1.1 + seed * 6.283 + ip.x * 0.05 + ip.z * 0.04) * 0.6 + sin(uTime * 0.4 + ip.x * 0.02 + ip.z * 0.03) * 0.5;
        float bend = h * h * 0.07 * sway;
        transformed.x += bend;
        transformed.z += bend * 0.45;
        transformed.x += sin(uTime * 2.2 + position.y * 6.0 + ip.z) * ${flutter.toFixed(3)} * h;`
      );
  };
  material.customProgramCacheKey = () => key;
  return material;
}

export class Tulips3D {
  /* positions: tulip positions (Float32Array xyz). rnd: the 4 random numbers per flower.
     hideAttr: BufferAttribute (aHide) on the sprite points, set to 1 while a flower is real 3D. */
  constructor({ scene, uTime, positions, rnd, hideAttr, mobile }) {
    this.scene = scene;
    this.pos = positions;
    this.rnd = rnd;
    this.hide = hideAttr;
    this.n = positions.length / 3;
    this.K = mobile ? 350 : 1000;
    this.range = mobile ? 17 : 24;
    this.active = true;
    this.sinceUpdate = 99;
    this.lastCam = new THREE.Vector3(1e9, 0, 0);
    this.state = new Uint8Array(this.n);
    this.current = [];
    this.dummy = new THREE.Object3D();
    this.col = new THREE.Color();
    this.fwd = new THREE.Vector3();
    this.disposables = [];

    // Lights are only used by these flowers. Everything else in the scene is unlit shader material.
    this.lights = [new THREE.HemisphereLight(0x8fb2ff, 0x16263a, 1.5), new THREE.DirectionalLight(0xb4d0ff, 2.0)];
    this.lights[1].position.set(-0.3, 0.42, -0.85).multiplyScalar(50);
    this.lights.forEach((l) => scene.add(l));

    // Petals: satin / velvety, no metal. Main blue #8296C3 is applied per instance.
    this.headMat = new THREE.MeshPhysicalMaterial({
      vertexColors: true, metalness: 0, roughness: 0.62, side: THREE.DoubleSide,
      specularIntensity: 0.35, sheen: 0.7, sheenRoughness: 0.55, sheenColor: new THREE.Color(0xbfd2ff),
    });
    addWind(this.headMat, uTime, 0, "tulip-head");
    this.greenMat = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0, roughness: 0.7, side: THREE.DoubleSide });
    addWind(this.greenMat, uTime, 0.012, "tulip-green");

    const gGreens = buildGreens();
    this.disposables.push(this.headMat, this.greenMat, gGreens);
    this.variants = [0, 0.45].map((open) => {
      const gh = buildHead(open);
      this.disposables.push(gh);
      const heads = new THREE.InstancedMesh(gh, this.headMat, this.K);
      const greens = new THREE.InstancedMesh(gGreens, this.greenMat, this.K);
      heads.frustumCulled = greens.frustumCulled = false;   // instances move around, so no culling by the first bounds
      heads.count = greens.count = 0;
      scene.add(heads, greens);
      return { heads, greens };
    });

    // Flowers sorted by their angle around the valley, so nearby ones are found quickly
    this.ang = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let a = Math.atan2(positions[i * 3 + 2], positions[i * 3]);
      if (a < 0) a += TAU;
      this.ang[i] = a;
    }
    this.order = Uint32Array.from({ length: this.n }, (_, i) => i).sort((a, b) => this.ang[a] - this.ang[b]);
  }

  setTint(night) { this.headMat.color.setScalar(1 - 0.45 * night); } // deep blue look when the letter is open

  setActive(on) {
    if (on === this.active) return;
    this.active = on;
    this.variants.forEach((v) => { v.heads.visible = v.greens.visible = on; });
    if (!on) this._release(); else this.lastCam.set(1e9, 0, 0);
  }

  _release() {
    for (const i of this.current) { this.state[i] = 0; this.hide.setX(i, 0); }
    this.current = [];
    this.variants.forEach((v) => { v.heads.count = v.greens.count = 0; });
    this.hide.needsUpdate = true;
  }

  /* Called every frame; does real work about 3 times a second, or when the camera moved a few units. */
  update(camera, dt) {
    if (!this.active) return;
    this.sinceUpdate += dt;
    const C = camera.position;
    if (this.sinceUpdate < 0.35 && C.distanceToSquared(this.lastCam) < 9) return;
    this.sinceUpdate = 0;
    this.lastCam.copy(C);
    camera.getWorldDirection(this.fwd);

    // 1. candidates: flowers within a narrow angle window around the camera, close and not far behind
    let th = Math.atan2(C.z, C.x);
    if (th < 0) th += TAU;
    const win = Math.min(Math.PI, (this.range + 6) / Math.max(40, Math.hypot(C.x, C.z)));
    const cand = [];
    const scan = (lo, hi) => {
      let a = 0, b = this.n;
      while (a < b) { const m = (a + b) >> 1; if (this.ang[this.order[m]] < lo) a = m + 1; else b = m; }
      for (let m = a; m < this.n && this.ang[this.order[m]] <= hi; m++) {
        const i = this.order[m], x = this.pos[i * 3] - C.x, z = this.pos[i * 3 + 2] - C.z;
        const d2 = x * x + z * z;
        if (d2 > this.range * this.range) continue;
        if (x * this.fwd.x + z * this.fwd.z < -4) continue;
        cand.push(i, d2);
      }
    };
    const lo = th - win, hi = th + win;
    if (lo < 0) { scan(lo + TAU, TAU); scan(0, hi); } else if (hi > TAU) { scan(lo, TAU); scan(0, hi - TAU); } else scan(lo, hi);

    // 2. keep the nearest K
    const idx = [];
    for (let m = 0; m < cand.length; m += 2) idx.push(m);
    idx.sort((a, b) => cand[a + 1] - cand[b + 1]);
    const chosen = idx.slice(0, this.K).map((m) => cand[m]);

    // 3. flip sprite visibility for flowers that changed
    const nowSet = new Set(chosen);
    for (const i of this.current) if (!nowSet.has(i)) { this.state[i] = 0; this.hide.setX(i, 0); }
    for (const i of chosen) if (!this.state[i]) { this.state[i] = 1; this.hide.setX(i, 1); }
    this.hide.needsUpdate = true;
    this.current = chosen;

    // 4. write one instance per chosen flower (same position and height as the sprite)
    const counts = [0, 0], d = this.dummy;
    for (const i of chosen) {
      const r0 = this.rnd[i * 4], r1 = this.rnd[i * 4 + 1], r2 = this.rnd[i * 4 + 2], r3 = this.rnd[i * 4 + 3];
      const v = r1 < 0.5 ? 0 : 1, slot = counts[v]++;
      const H = 1.3 + r0 * 1.1, wide = 0.88 + r3 * 0.3;
      d.position.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      d.rotation.set((r3 - 0.5) * 0.14, r2 * TAU, (r0 - 0.5) * 0.14);
      d.scale.set(H * wide, H, H * wide);
      d.updateMatrix();
      const { heads, greens } = this.variants[v];
      heads.setMatrixAt(slot, d.matrix);
      greens.setMatrixAt(slot, d.matrix);
      // main blue #8296C3 with small natural variation
      this.col.set(0x8296c3).offsetHSL((r2 - 0.5) * 0.02, (r0 - 0.5) * 0.08, (r3 - 0.5) * 0.08);
      heads.setColorAt(slot, this.col);
    }
    this.variants.forEach((vr, v) => {
      vr.heads.count = vr.greens.count = counts[v];
      vr.heads.instanceMatrix.needsUpdate = vr.greens.instanceMatrix.needsUpdate = true;
      if (vr.heads.instanceColor) vr.heads.instanceColor.needsUpdate = true;
    });
  }

  dispose() {
    this._release();
    this.lights.forEach((l) => this.scene.remove(l));
    this.variants.forEach((v) => { this.scene.remove(v.heads, v.greens); v.heads.dispose(); v.greens.dispose(); });
    this.disposables.forEach((d) => d.dispose());
  }
}
