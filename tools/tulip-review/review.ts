// img2threejs review harness: renders the real factory + refinement in the browser for captures.
import * as THREE from 'three';
import { createBlueTulipModel } from '../../src/tulip-model/createObjectModel';
import { applyRefinement, animateTulip, buildFieldTulip } from '../../src/tulip-model/tulipRefinement';

const params = new URLSearchParams(location.search);
const raw = params.has('raw');          // raw = generator output only (no refinement)
const W = innerWidth, H = innerHeight;
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#efefef');
// Lighting pass: rig follows spec.lightingFromPhoto (key from upper-left-front, hemisphere fill, cool rim)
scene.add(new THREE.HemisphereLight(0xeaf0fa, 0xc8cdb8, 1.7));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-0.4, 0.7, 0.6).multiplyScalar(8); key.castShadow = true;
key.shadow.radius = 4; key.shadow.normalBias = 0.02;
const sc = key.shadow.camera as THREE.OrthographicCamera; sc.left = -1.5; sc.right = 1.5; sc.top = 1.6; sc.bottom = -2.4;
key.shadow.mapSize.set(1024, 1024); key.shadow.bias = -0.0005;
const fill = new THREE.DirectionalLight(0xdfe8ff, 0.9); fill.position.set(3, 1, 3);
const rim = new THREE.DirectionalLight(0xcfe0ff, 1.2); rim.position.set(0, 2, -4);
scene.add(key, fill, rim);

const model = createBlueTulipModel({ castShadow: true, receiveShadow: true, qualityPriority: 'balanced' });
const rig = raw ? null : applyRefinement(model);
// Interaction pass: pose the rig mid-breeze (?anim=<time>) to prove pivots move without tearing
if (rig && params.has('anim')) animateTulip(rig, Number(params.get('anim')), 1.6);
if (params.has('nomaps')) {
  // Map-stripped render: proves the form reads without any texture maps (Tier 1 requirement).
  model.traverse((o: any) => {
    if (!o.isMesh) return;
    for (const m of [].concat(o.material)) {
      for (const k of ['map', 'roughnessMap', 'normalMap', 'bumpMap', 'aoMap', 'metalnessMap', 'displacementMap']) if ((m as any)[k]) (m as any)[k] = null;
      (m as any).needsUpdate = true;
    }
  });
}
if (params.has('field')) {
  // ?field = the website's field tulip (romantic petal style), shown in the same framing
  const { head, greens } = buildFieldTulip(model, 1);
  const g = new THREE.Group();
  g.add(new THREE.Mesh(head, new THREE.MeshPhysicalMaterial({ vertexColors: true, color: '#9fb6ee', roughness: 0.55, sheen: 0.8, sheenColor: new THREE.Color('#d2e0ff'), side: THREE.DoubleSide })));
  g.add(new THREE.Mesh(greens, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.62, side: THREE.DoubleSide })));
  g.scale.setScalar(3.4); g.position.set(-0.035, -2.4, 0);
  model.visible = false;
  scene.add(g);
}
scene.add(model);

// Reference framing: bloom height = 1 unit = 315 px of the 1000 px reference.
const camera = new THREE.PerspectiveCamera(20, W / H, 0.1, 100);
const target = new THREE.Vector3(-0.035, -0.41, 0);
const dist = (3.17 / 2) / Math.tan(THREE.MathUtils.degToRad(10)) * (1000 / H) * (H / 1000);
function setCamera(spec: { azimuthDegrees: number; elevationDegrees: number; target?: number[] }) {
  const t = spec.target ? new THREE.Vector3(...(spec.target as [number, number, number])) : target;
  const az = THREE.MathUtils.degToRad(spec.azimuthDegrees), el = THREE.MathUtils.degToRad(spec.elevationDegrees);
  camera.position.set(t.x + Math.sin(az) * Math.cos(el) * dist, t.y + Math.sin(el) * dist, t.z + Math.cos(az) * Math.cos(el) * dist);
  camera.lookAt(t);
  renderer.render(scene, camera);
}
let tris = 0; model.traverse((o: any) => { if (o.isMesh && o.visible) tris += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3; });
(window as any).__TULIP_STATS__ = { triangles: Math.round(tris) };
// Runtime part-tree dump for check_part_coverage.py (assembly gate) and action-ready evidence
{
  const rt = model.userData.sculptRuntime as any;
  const triOf = (m: any) => Math.round((m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3);
  const parts = Object.entries(rt.meshes).map(([id, m]: [string, any]) => ({ name: id, kind: 'part', module: id, triangles: triOf(m), visible: m.visible, note: m.visible ? undefined : 'relief baked into refined shell (spec handRefinement)' }));
  let unnamed = 0; const named = new Set(Object.values(rt.meshes));
  model.traverse((o: any) => { if (o.isMesh && !named.has(o)) unnamed++; });
  (window as any).__TULIP_PARTS__ = {
    model: 'blue-tulip', parts, unnamedMeshes: unnamed, integralMeshes: parts.length,
    pivots: Object.keys(rt.nodes), sockets: Object.keys(rt.sockets), colliders: Object.keys(rt.colliders),
    rig: rig ? { tepals: rig.tepals.length, leaves: rig.leaves.length, bloom: !!rig.bloom } : null,
  };
}
(window as any).__IMG2THREEJS_CAPTURE__ = { setCamera: async (s: any) => setCamera(s) };
setCamera({ azimuthDegrees: Number(params.get('az') ?? 0), elevationDegrees: Number(params.get('el') ?? 0) });
(window as any).__IMG2THREEJS_READY__ = true;
