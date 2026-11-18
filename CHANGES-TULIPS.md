# Tulip upgrade: what changed

Only the blue tulip flowers were touched.

New file
- src/tulip3d.js : real 3D tulip (6 overlapping petals, curved tapering stem, two arched leaves with a midrib groove),
  satin material (metalness 0, main blue #8296C3), wind sway, instancing (about 1000 on desktop, 350 on phones).

Edited files (originals saved in backup-original-flowers/)
- src/valley.js  : imports tulip3d.js; makeSprites now returns the tulip positions and adds a hide flag; creates the 3D tulips;
                   3 small calls in the frame loop (tint, on/off while flowers scatter, update); dispose.
- src/shaders.js : one added attribute (aHide) and a 2 line early exit so a flat flower hides while its 3D version is drawn.

Not touched: terrain, sky, fog, camera path and scrolling, text, letter, music, effects, page layout.

How it works: far flowers stay as cheap flat sprites. The closest ~1000 flowers (within about 24 units of the camera)
are swapped for real 3D tulips at the same position and height, so the field layout is unchanged.

To go back: copy the two files from backup-original-flowers/ over src/valley.js and src/shaders.js and delete src/tulip3d.js.
