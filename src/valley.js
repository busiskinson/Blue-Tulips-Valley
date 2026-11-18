import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { spriteVert, spriteFrag } from "./shaders.js";
import { audio } from "./audio.js";

const TAU = Math.PI * 2;
// Two looks: DAY = soft powder blue (before the click), NIGHT = the original deep blue (letter open)
const DAY = { fog: new THREE.Color(0x1b3558), horizon: new THREE.Color(0x2a4a75), top: new THREE.Color(0x061428) };
const NIGHT = { fog: new THREE.Color(0x07203d), horizon: new THREE.Color(0x0a2a52), top: new THREE.Color(0x010612) };
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/* ---------- Terrain: a ring-shaped valley. The camera path runs along its floor ---------- */
const rPath = (a) => 110 + 16 * Math.sin(3 * a + 0.5) + 9 * Math.sin(5 * a + 1.7);
const noise = (x, z) => Math.sin(x * 0.05 + 1.3) * Math.cos(z * 0.043) * 0.5 + Math.sin(x * 0.11 + z * 0.09) * 0.25 + Math.sin(x * 0.23 - z * 0.19 + 2) * 0.12;
function groundHeight(x, z) {
  const d = Math.hypot(x, z) - rPath(Math.atan2(z, x));  // distance from the valley floor
  const side = smooth(16, 58, Math.abs(d));              // 0 on the floor, 1 on the hills
  return side * (9 + 6 * noise(x * 0.6, z * 0.6)) + noise(x, z) * (0.35 + 0.5 * side);
}

let seed = 7;
const rand = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

/* ---------- Post effect: split RGB, film noise, vignette that tightens with speed ---------- */
const FX = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uSpeed: { value: 0 } },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uTime; uniform float uSpeed; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 c = vUv - 0.5; float r = length(c);
      float s = 0.0006 + 0.004 * r * r + 0.0015 * uSpeed;
      vec3 col = vec3(texture2D(tDiffuse, vUv + c * s).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - c * s).b);
      col += (hash(vUv * vec2(1920.0, 1080.0) + fract(uTime)) - 0.5) * 0.025;
      float vig = 1.0 - smoothstep(mix(0.45, 0.25, uSpeed), 0.95, r * 1.2);
      col *= mix(0.3, 1.0, vig);
      gl_FragColor = vec4(col, 1.0);
    }`,
};

export class Valley {
  /* container: the #valley element. onLoaded: first frame is drawn.
     onIntroDone: called every frame after the opening move has finished. */
  constructor(container, onLoaded, onIntroDone) {
    this.onLoaded = onLoaded;
    this.onIntroDone = onIntroDone;
    const mobile = matchMedia("(max-width: 820px)").matches || /Mobi|Android/i.test(navigator.userAgent);
    this.reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const q = mobile ? 0.4 : 1;
    this.dpr = Math.min(window.devicePixelRatio || 1, mobile ? 1.25 : 1.5);

    const mount = container.querySelector(".js-canvas-container") || container;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(this.dpr);
    mount.appendChild(this.renderer.domElement);

    const FOG = new THREE.Color(0x1b3558);
    this.FOG = FOG;
    this.scene = new THREE.Scene();
    this.scene.background = FOG;
    this.scene.fog = new THREE.FogExp2(FOG, 0.0075);
    this.camera = new THREE.PerspectiveCamera(58, 1, 0.1, 700);

    // Values shared by all shaders
    this.U = {
      uTime: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) },
      uMouse: { value: new THREE.Vector2(9, 9) }, uMouseOn: { value: 0 },
      uDisperse: { value: 0 }, uNight: { value: 0 }, uFog: { value: FOG }, uFogD: { value: 0.0075 },
    };
    this.disposables = [];
    const track = (o) => { this.disposables.push(o); return o; };

    /* Camera path: a closed loop along the valley floor, so you can wander forever */
    const pts = [];
    for (let i = 0; i < 96; i++) {
      const a = (i / 96) * TAU, r = rPath(a), x = r * Math.cos(a), z = r * Math.sin(a);
      pts.push(new THREE.Vector3(x, groundHeight(x, z) + 2.0, z));
    }
    this.curve = new THREE.CatmullRomCurve3(pts, true, "centripetal");
    this.curve.arcLengthDivisions = 1200;
    this.len = this.curve.getLength();

    /* Sky with a moon */
    const moonDir = new THREE.Vector3(-0.3, 0.42, -0.85).normalize();
    this.dome = new THREE.Mesh(
      track(new THREE.SphereGeometry(500, 24, 12)),
      track(new THREE.ShaderMaterial({
        side: THREE.BackSide, depthWrite: false, fog: false,
        uniforms: { uHorizon: { value: new THREE.Color(0x2a4a75) }, uTop: { value: new THREE.Color(0x061428) }, uMoon: { value: moonDir }, uN: this.U.uNight },
        vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: `uniform vec3 uHorizon; uniform vec3 uTop; uniform vec3 uMoon; uniform float uN; varying vec3 vP;
          void main(){ vec3 v = normalize(vP);
            vec3 c = mix(uHorizon, uTop, smoothstep(-0.05, 0.7, v.y));
            float m = max(dot(v, uMoon), 0.0);
            c += vec3(0.35, 0.55, 0.95) * (pow(m, 1200.0) * (2.0 + 2.0 * uN) + pow(m, 40.0) * (0.12 + 0.18 * uN));
            gl_FragColor = vec4(c, 1.0); }`,
      }))
    );
    this.scene.add(this.dome);

    /* Ground: dark navy-green terrain */
    const gg = track(new THREE.PlaneGeometry(520, 520, 160, 160));
    gg.rotateX(-Math.PI / 2);
    const gp = gg.attributes.position, colors = new Float32Array(gp.count * 3), c1 = new THREE.Color(0x0a1c18), c2 = new THREE.Color(0x12301c), tmp = new THREE.Color();
    for (let i = 0; i < gp.count; i++) {
      const x = gp.getX(i), z = gp.getZ(i), h = groundHeight(x, z);
      gp.setY(i, h);
      tmp.copy(c1).lerp(c2, Math.min(1, Math.max(0, 0.5 + noise(x * 1.7, z * 1.7)))).toArray(colors, i * 3);
    }
    gg.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.groundMat = track(new THREE.MeshBasicMaterial({ vertexColors: true }));
    this.scene.add(new THREE.Mesh(gg, this.groundMat));

    /* Tulips and leaves: point sprites scattered along the valley */
    const bluePalette = [0x8fb0e0, 0xa7c0e8, 0x7f9fd6, 0xb8cdee, 0x9ab4e4, 0x6f8fcb].map((h) => new THREE.Color(h));
    const deepBlue = [0x0755bd, 0x0874e9, 0x082d63, 0x0a3f8f, 0x0755bd, 0x54aaff].map((h) => new THREE.Color(h));
    const deepGreen = [0x0d3b26, 0x0f4a2c, 0x0a3322, 0x14553a].map((h) => new THREE.Color(h));
    const greenPalette = [0x2f6b35, 0x3a7d3c, 0x2a5f33, 0x437f3f].map((h) => new THREE.Color(h));
    const makeSprites = (count, leaf) => {
      const pos = new Float32Array(count * 3), rnd = new Float32Array(count * 4), col = new Float32Array(count * 3), col2 = new Float32Array(count * 3);
      const spread = leaf ? 58 : 46;
      for (let i = 0; i < count; i++) {
        const a = rand() * TAU, r = rPath(a) + (rand() + rand() + rand() - 1.5) * spread;
        const x = r * Math.cos(a), z = r * Math.sin(a);
        pos.set([x, groundHeight(x, z), z], i * 3);
        rnd.set([rand(), rand(), rand(), rand()], i * 4);
        const pi = rand(), k = rand();
        const base = leaf ? greenPalette[(pi * greenPalette.length) | 0] : bluePalette[(pi * bluePalette.length) | 0];
        tmp.copy(base).multiplyScalar(0.85 + k * 0.3).toArray(col, i * 3);
        const base2 = leaf ? deepGreen[(pi * deepGreen.length) | 0] : deepBlue[(pi * deepBlue.length) | 0];
        tmp.copy(base2).multiplyScalar((leaf ? 1 : 1.5) * (0.75 + k * 0.5)).toArray(col2, i * 3);
      }
      const g = track(new THREE.BufferGeometry());
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("aRand", new THREE.BufferAttribute(rnd, 4));
      g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
      g.setAttribute("aColor2", new THREE.BufferAttribute(col2, 3));
      const m = track(new THREE.ShaderMaterial({ uniforms: this.U, vertexShader: spriteVert, fragmentShader: spriteFrag, defines: leaf ? { LEAF: 1 } : {} }));
      const p = new THREE.Points(g, m);
      p.frustumCulled = false;
      this.scene.add(p);
    };
    makeSprites(Math.round(34000 * q), true);   // leaves first
    makeSprites(Math.round(46000 * q), false);  // blue tulips

    /* Dust and bokeh that follow the camera */
    const dn = mobile ? 350 : 900;
    const dpos = new Float32Array(dn * 3), dsz = new Float32Array(dn), dwarm = new Float32Array(dn);
    for (let i = 0; i < dn; i++) {
      dpos.set([rand(), rand(), rand()], i * 3);
      dsz[i] = rand() < 0.1 ? 22 + rand() * 24 : 2 + rand() * 4;
      dwarm[i] = rand() < 0.15 ? 1 : 0;
    }
    const dg = track(new THREE.BufferGeometry());
    dg.setAttribute("position", new THREE.BufferAttribute(dpos, 3));
    dg.setAttribute("aSize", new THREE.BufferAttribute(dsz, 1));
    dg.setAttribute("aWarm", new THREE.BufferAttribute(dwarm, 1));
    this.dustMat = track(new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: this.U.uTime, uCam: { value: new THREE.Vector3() }, uPR: { value: this.dpr } },
      vertexShader: `uniform float uTime; uniform vec3 uCam; uniform float uPR; attribute float aSize; attribute float aWarm;
        varying float vA; varying float vWarm; varying float vBig;
        void main(){
          vec3 box = vec3(70.0, 22.0, 70.0);
          vec3 base = position * box + vec3(sin(uTime * 0.2 + position.z * 20.0), uTime * 0.12, cos(uTime * 0.17 + position.x * 20.0)) * 1.5;
          vec3 q = mod(base - uCam, box) - box * 0.5;
          vec3 w = uCam + vec3(q.x, q.y * 0.35 - 0.2, q.z);
          vec4 mv = viewMatrix * vec4(w, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(aSize * uPR * (26.0 / -mv.z), 1.0, 80.0 * uPR);
          vA = (0.5 + 0.5 * sin(uTime * 0.9 + position.x * 60.0)) * smoothstep(1.5, 5.0, -mv.z) * (1.0 - smoothstep(25.0, 35.0, -mv.z));
          vWarm = aWarm; vBig = step(15.0, aSize);
        }`,
      fragmentShader: `varying float vA; varying float vWarm; varying float vBig;
        void main(){
          float d = length(gl_PointCoord - 0.5);
          float soft = smoothstep(0.5, 0.0, d);
          float ring = smoothstep(0.5, 0.42, d) * smoothstep(0.3, 0.46, d);
          float a = mix(soft * soft, soft * 0.2 + ring * 0.3, vBig) * vA;
          vec3 c = mix(vec3(0.33, 0.67, 1.0), vec3(1.0, 0.7, 0.35), vWarm);
          gl_FragColor = vec4(c * a * (vBig > 0.5 ? 0.22 : 0.55), a);
        }`,
    }));
    const dust = new THREE.Points(dg, this.dustMat);
    dust.frustumCulled = false;
    this.scene.add(dust);

    /* Light rays: soft vertical beams that always face the camera */
    const rayGeo = track(new THREE.PlaneGeometry(7, 60).translate(0, 30, 0));
    this.rays = [];
    for (let i = 0; i < (mobile ? 10 : 22); i++) {
      const a = rand() * TAU, r = rPath(a) + (rand() - 0.5) * 80, x = r * Math.cos(a), z = r * Math.sin(a);
      const mat = track(new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        uniforms: { uA: { value: 0 } },
        vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: "uniform float uA; varying vec2 vUv; void main(){ float h = 1.0 - abs(vUv.x - 0.5) * 2.0; h *= h; float v = pow(vUv.y, 1.2); gl_FragColor = vec4(vec3(0.25, 0.5, 1.0) * h * v * uA, 1.0); }",
      }));
      const m = new THREE.Mesh(rayGeo, mat);
      m.position.set(x, groundHeight(x, z), z);
      this.scene.add(m);
      this.rays.push(m);
    }

    /* Post-processing: bloom, then RGB split + noise + vignette */
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.65, 0.5, 0.7);
    this.composer.addPass(this.bloom);
    this.fx = new ShaderPass(FX);
    this.composer.addPass(this.fx);
    this.composer.addPass(new OutputPass());

    /* Motion state */
    this.dist = 0;                 // how far along the path we are (world units)
    this.vel = 0;                  // current speed
    this.impulse = 0;              // extra speed from wheel or touch
    this.idle = this.reduce ? 1.2 : 3.2; // slow drift when nobody scrolls
    this.hover = 0; this.hoverTarget = 0;
    this.night = 0; this.nightTarget = 0;
    this.mouse = new THREE.Vector2(0, 0); this.mouseT = new THREE.Vector2(0, 0);
    this.look = new THREE.Vector3();
    this.introT = 0;
    this.frames = 0;
    this.clock = new THREE.Clock();
    this.P = new THREE.Vector3(); this.Q = new THREE.Vector3(); this.T = new THREE.Vector3(); this.R = new THREE.Vector3();

    this._bind();
    this._resize();
    this._frame = this._frame.bind(this);
    this._frame();
  }

  // Letter open: switch to the deep blue glowing look. Letter closed: back to the soft look.
  setNight(on) {
    this.nightTarget = on ? 1 : 0;
    if (on) { this.hoverTarget = 0; audio.setMuffled(false); }
  }
  startHoverTransition() { if (this.nightTarget) return; this.hoverTarget = 1; audio.setMuffled(true); }
  stopHoverTransition() { this.hoverTarget = 0; audio.setMuffled(false); }

  _bind() {
    this._onResize = () => this._resize();
    this._onMove = (e) => { this.mouseT.set((e.clientX / innerWidth) * 2 - 1, -((e.clientY / innerHeight) * 2 - 1)); this.U.uMouseOn.value = e.pointerType === "touch" ? 0 : 1; };
    this._onLeave = () => { this.U.uMouseOn.value = 0; };
    this._onWheel = (e) => { this.impulse += e.deltaY * 0.04; };
    this._onTouchStart = (e) => { this._ty = e.touches[0].clientY; };
    this._onTouchMove = (e) => { const y = e.touches[0].clientY; this.impulse += (this._ty - y) * 0.25; this._ty = y; };
    this._onVis = () => { if (document.hidden) cancelAnimationFrame(this.raf); else { this.clock.getDelta(); this._frame(); } };
    addEventListener("resize", this._onResize);
    addEventListener("pointermove", this._onMove);
    document.addEventListener("pointerleave", this._onLeave);
    addEventListener("wheel", this._onWheel, { passive: true });
    addEventListener("touchstart", this._onTouchStart, { passive: true });
    addEventListener("touchmove", this._onTouchMove, { passive: true });
    document.addEventListener("visibilitychange", this._onVis);
  }

  _resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w / 2, h / 2);
    this.camera.aspect = w / h;
    this.camera.fov = w / h < 1 ? 72 : 58;
    this.camera.updateProjectionMatrix();
    this.U.uRes.value.set(w * this.dpr, h * this.dpr);
  }

  _frame() {
    this.raf = requestAnimationFrame(this._frame);
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const U = this.U;
    U.uTime.value += dt * (this.reduce ? 0.3 : 1);

    // Speed: drift when idle, wheel/touch adds speed, hover eases everything to a stop
    this.vel += (this.idle - this.vel) * Math.min(1, dt * 1.2);
    this.vel = THREE.MathUtils.clamp(this.vel + this.impulse, -40, 40);
    this.impulse = 0;
    this.hover += (this.hoverTarget - this.hover) * Math.min(1, dt * 1.5);
    this.dist += this.vel * (1 - this.hover) * dt;
    U.uDisperse.value = this.hover;
    this.night += (this.nightTarget - this.night) * Math.min(1, dt * 1.2);
    if (Math.abs(this.nightTarget - this.night) < 0.003) this.night = this.nightTarget; // settle exactly
    const n = this.night;
    U.uNight.value = n;
    this.FOG.copy(DAY.fog).lerp(NIGHT.fog, n);
    this.scene.fog.color.copy(this.FOG);
    const du = this.dome.material.uniforms;
    du.uHorizon.value.copy(DAY.horizon).lerp(NIGHT.horizon, n);
    du.uTop.value.copy(DAY.top).lerp(NIGHT.top, n);
    this.groundMat.color.setScalar(1 - 0.5 * n);
    this.bloom.strength = 0.65 + 0.25 * n;
    this.bloom.radius = 0.5 + 0.2 * n;
    this.bloom.threshold = 0.7 - 0.5 * n;
    const speed = Math.min(1, Math.abs(this.vel) / 14);
    this.fx.uniforms.uSpeed.value += (speed - this.fx.uniforms.uSpeed.value) * Math.min(1, dt * 2);
    this.fx.uniforms.uTime.value = U.uTime.value;

    // Mouse (smoothed)
    this.mouse.lerp(this.mouseT, Math.min(1, dt * 4));
    U.uMouse.value.copy(this.mouse);

    // Intro: the camera starts high above the field and settles into the flowers
    this.introT += dt;
    const ip = Math.min(1, this.introT / (this.reduce ? 0.5 : 3.5));
    const lift = Math.pow(1 - ip, 3) * 32;

    // Place the camera on the path
    const u = ((this.dist / this.len) % 1 + 1) % 1;
    this.curve.getPointAt(u, this.P);
    this.curve.getPointAt((u + 9 / this.len) % 1, this.Q);
    this.T.copy(this.Q).sub(this.P).normalize();
    this.R.crossVectors(this.T, this.camera.up).normalize();
    const bob = Math.sin(U.uTime.value * 0.6) * 0.06;
    this.camera.position.set(this.P.x, this.P.y + bob + lift, this.P.z);
    this.Q.y -= 0.7;
    this.Q.addScaledVector(this.R, this.mouse.x * 5).y += this.mouse.y * 2.5; // camera tilts toward the mouse
    this.look.lerp(this.Q, ip < 1 || this.frames < 2 ? 1 : Math.min(1, dt * 5));
    this.camera.lookAt(this.look);
    this.dome.position.copy(this.camera.position);
    this.dustMat.uniforms.uCam.value.copy(this.camera.position);

    for (const m of this.rays) {
      const d = m.position.distanceTo(this.camera.position);
      m.material.uniforms.uA.value = 0.1 * smooth(10, 30, d) * (1 - smooth(60, 140, d));
      m.lookAt(this.camera.position.x, m.position.y, this.camera.position.z);
    }

    this.composer.render();
    this.frames++;
    if (this.frames === 2) this.onLoaded && this.onLoaded();
    if (ip >= 1) this.onIntroDone && this.onIntroDone();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    removeEventListener("resize", this._onResize);
    removeEventListener("pointermove", this._onMove);
    document.removeEventListener("pointerleave", this._onLeave);
    removeEventListener("wheel", this._onWheel);
    removeEventListener("touchstart", this._onTouchStart);
    removeEventListener("touchmove", this._onTouchMove);
    document.removeEventListener("visibilitychange", this._onVis);
    this.disposables.forEach((d) => d.dispose());
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
