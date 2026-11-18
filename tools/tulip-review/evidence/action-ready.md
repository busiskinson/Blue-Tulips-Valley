# Action-ready verification (runtime dump: evidence/parts.json)
- root.userData.sculptRuntime present: nodes (18 pivots, one per spec component), meshes (18 named), colliders (18 proxy boxes), destructionGroups.
- Every mesh is a named part (unnamedMeshes 0 after removing the duplicate tepal-whorls InstancedMesh); part coverage 0 errors / 0 warnings.
- Pivots used for animation (TulipRig in src/tulip-model/tulipRefinement.ts): bloom pivot (sway), 6 tepal pivots re-seated on each tepal base with a tangent hinge axis (breathing / open on reveal), 2 leaf pivots on the stem clasp (flutter). Posed stress render: evidence/review/interaction-pass-posed (turntable clean).
- KNOWN GAP: runtime `sockets` map is empty. Spec attachments name parentSockets (stem-top, receptacle, stem-leaf-left/right) but the generated factory resolves them into pivot positions instead of emitting socket Object3Ds. Nothing in the website needs sockets; reported, not fixed.
- Picking: the website picks the whole flower through a proxy cylinder (drag-to-rotate); per-part picking/explode is available through the named meshes but not exposed in the site UI.
