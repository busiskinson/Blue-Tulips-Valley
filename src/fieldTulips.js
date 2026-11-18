/* Builds the field version of the img2threejs-reconstructed blue tulip.
   The full model is created only to read its spec data, turned into the light single-surface
   field geometry, and then freed again: nothing extra is drawn on screen. Loaded lazily. */

import { createBlueTulipModel } from "./tulip-model/createObjectModel";
import { applyRefinement, buildFieldTulip, disposeTulip } from "./tulip-model/tulipRefinement";

export function makeFieldTulipGeometry(quality = 0.5) {
  const model = createBlueTulipModel({ castShadow: false, receiveShadow: false, qualityPriority: "balanced" });
  applyRefinement(model, { quality: 0.3 });
  const geo = buildFieldTulip(model, quality);
  disposeTulip(model);
  return geo;
}
