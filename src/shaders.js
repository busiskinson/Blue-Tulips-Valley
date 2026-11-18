/* Shaders for the tulip and leaf point sprites.
   Every flower is ONE point on the screen. The fragment shader draws a tulip
   (head, stem, leaves) inside that point, so tens of thousands are cheap. */

export const spriteVert = /* glsl */ `
uniform float uTime;
uniform vec2 uRes;
uniform vec2 uMouse;
uniform float uMouseOn;
uniform float uDisperse;
uniform float uFogD;
attribute vec4 aRand;
attribute vec3 aColor;
attribute vec3 aColor2;
varying vec3 vColor;
varying vec3 vColor2;
varying float vSway;
varying float vDisp;
varying float vFog;
varying float vGlow;
varying vec2 vSeed;

void main() {
  #ifdef LEAF
    float dsp = 0.0;
  #else
    float dsp = uDisperse;
  #endif

  // Hover transition: flowers drift outward and become floating particles
  vec3 wp = position;
  vec3 dir = normalize(vec3(aRand.z - 0.5, aRand.w * 0.7 + 0.2, aRand.y - 0.5) + 1e-4);
  wp += dir * dsp * (4.0 + 12.0 * aRand.x);
  wp.y += dsp * (1.5 + sin(uTime * 0.5 + aRand.y * 20.0) * 0.8);

  vec4 mv = modelViewMatrix * vec4(wp, 1.0);
  float dist = max(-mv.z, 0.01);

  // Flowers bloom from small to full size as the camera gets close.
  // Leaves grow in earlier, further away.
  #ifdef LEAF
    float grow = smoothstep(170.0, 60.0, dist);
  #else
    float grow = mix(0.3, 1.0, 1.0 - smoothstep(25.0, 130.0, dist));
  #endif

  float height = 1.3 + aRand.x * 1.1;
  float s = height * grow * mix(1.0, 0.08, smoothstep(0.0, 0.5, dsp));
  s *= projectionMatrix[1][1] * uRes.y * 0.5 / dist;
  s *= smoothstep(1.0, 3.5, dist);          // dissolve right in front of the lens
  s = min(s, 256.0);
  gl_PointSize = s;

  gl_Position = projectionMatrix * mv;
  vec2 ndc = gl_Position.xy / gl_Position.w;

  // Mouse: push points away in screen space, strongest close to the cursor
  vec2 asp = vec2(uRes.x / uRes.y, 1.0);
  vec2 d = (ndc - uMouse) * asp;
  float l = length(d);
  float k = exp(-l * l * 7.0) * uMouseOn;
  ndc += normalize(d + 1e-5) / asp * k * 0.16;

  // The point sits at the flower's base, so lift the sprite up by half its size
  ndc.y += s / uRes.y * (1.0 - smoothstep(0.0, 0.4, dsp));
  gl_Position.xy = ndc * gl_Position.w;

  vSway = sin(uTime * 1.1 + aRand.z * 6.283 + position.x * 0.05 + position.z * 0.04) * 0.6
        + sin(uTime * 0.4 + position.x * 0.02 + position.z * 0.03) * 0.5;
  vColor = aColor;
  vColor2 = aColor2;
  vGlow = k;
  vDisp = dsp;
  vSeed = aRand.zw;
  float f = dist * uFogD;
  vFog = 1.0 - exp(-f * f);
}
`;

export const spriteFrag = /* glsl */ `
uniform float uTime;
uniform float uNight;
uniform vec3 uFog;
varying vec3 vColor;
varying vec3 vColor2;
varying float vSway;
varying float vDisp;
varying float vFog;
varying float vGlow;
varying vec2 vSeed;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// One long pointed leaf. p.y is height (0 bottom, 1 top), sd = lean direction.
float blade(vec2 p, float sd, float L, float W, out float sh) {
  float t = p.y / L;
  if (t <= 0.0 || t >= 1.0) { sh = 0.0; return 0.0; }
  float c = sd * (0.012 + 0.2 * pow(t, 1.6));
  float hw = W * sin(3.14159 * pow(t, 0.7));
  float dx = abs(p.x - c);
  sh = 1.0 - dx / max(hw, 1e-4);
  return step(dx, hw);
}

vec4 leafShape(vec2 p, vec3 vColor) {
  p.x -= vSway * 0.05 * p.y * p.y;
  float sh;
  float sd = vSeed.x > 0.5 ? 1.0 : -1.0;
  if (blade(p, sd, 0.9, 0.022, sh) > 0.5) return vec4(vColor * (0.5 + 0.9 * sh) * (0.5 + 0.8 * p.y), 1.0);
  if (blade(p, -sd, 0.6, 0.02, sh) > 0.5) return vec4(vColor * (0.4 + 0.8 * sh), 1.0);
  return vec4(0.0);
}

vec4 tulipShape(vec2 p) {
  p.x -= vSway * 0.07 * p.y * p.y;                       // wind lean
  p.x -= (vSeed.y - 0.5) * 0.05 * p.y * p.y;             // each flower leans differently
  float x = p.x, y = p.y;

  // Head: a closed, egg-shaped cup. Three overlapping petals (left, front, right)
  float t = (y - 0.48) / 0.44;
  if (t > 0.0 && t < 1.0) {
    float open = mix(0.92, 1.12, vSeed.x);
    float w = (0.115 * pow(sin(3.14159 * pow(t, 0.8) * 0.9), 0.6) + 0.014) * open;
    float xn = x / w;
    if (abs(xn) < 1.0) {
      float ci = abs(xn) < 0.38 ? 0.0 : (xn < 0.0 ? -0.69 : 0.69);
      float lu = (xn - ci) / (ci == 0.0 ? 0.38 : 0.31);
      float top = (ci == 0.0 ? 1.0 : 0.95) - 0.07 * abs(lu);     // softly pointed tips
      if (t < top) {
        vec3 c = vColor * (0.7 + 0.35 * t);                       // powder blue, a bit deeper low down
        c *= 0.72 + 0.34 * (1.0 - lu * lu);                       // rounded, soft shading
        c *= ci == 0.0 ? 1.06 : 0.9;                              // front petal sits on top
        float seam = smoothstep(0.8, 1.0, abs(lu));
        c *= mix(vec3(1.0), vec3(0.72, 0.78, 0.95), seam);        // creases lean violet-blue
        c *= 1.0 + 0.03 * sin(lu * 14.0 + vSeed.x * 6.0);         // fine petal lines
        float spec = exp(-pow((lu + 0.3) / 0.22, 2.0)) * smoothstep(0.15, 0.6, t) * (1.0 - smoothstep(0.8, 1.0, t));
        c += vec3(0.28, 0.34, 0.45) * spec * 0.55;                // soft satin sheen

        // Dew drops: tiny lenses with a bright catch-light
        vec2 gc = gl_PointCoord * 26.0;
        vec2 cell = floor(gc);
        vec2 f = fract(gc) - 0.5;
        if (hash(cell + vSeed * 7.0) > 0.72) {
          vec2 off = (vec2(hash(cell + 3.1), hash(cell + 7.7)) - 0.5) * 0.4;
          float rad = 0.12 + 0.14 * hash(cell + 1.3);
          float d = length(f - off);
          if (d < rad) {
            c = mix(c, c * 1.15 + 0.06, 0.8);
            c = mix(c, c * 0.85, smoothstep(rad * 0.7, rad, d));   // darker drop edge
            if (length(f - off - vec2(-rad * 0.35, rad * 0.35)) < rad * 0.28) c = vec3(0.95, 0.97, 1.0);
          }
        }
        return vec4(c, 1.0);
      }
    }
  }
  // Stem: pale grey-green, slightly curved
  float sx = x - 0.014 * sin(y * 3.0 + vSeed.x * 6.0);
  if (abs(sx) < 0.0125 - 0.003 * y && y > 0.02 && y < 0.52) {
    float lit = 0.8 + 0.4 * smoothstep(0.01, -0.01, sx);
    return vec4(vec3(0.17, 0.3, 0.15) * lit * (0.7 + 0.5 * y), 1.0);
  }
  // Two broad, glossy leaves with a lighter middle rib
  float sh;
  if (blade(p, -1.0, 0.5, 0.06, sh) > 0.5) return vec4(vec3(0.07, 0.3, 0.09) * (0.55 + 0.7 * sh) + vec3(0.06, 0.1, 0.04) * smoothstep(0.88, 1.0, sh), 1.0);
  if (blade(p, 1.0, 0.4, 0.055, sh) > 0.5) return vec4(vec3(0.07, 0.3, 0.09) * (0.55 + 0.7 * sh) + vec3(0.06, 0.1, 0.04) * smoothstep(0.88, 1.0, sh), 1.0);
  return vec4(0.0);
}

// The original deep-blue tulip (used when the letter is open)
vec4 tulipOld(vec2 p, vec3 vColor) {
  p.x -= vSway * 0.07 * p.y * p.y;
  float x = p.x, y = p.y;
  float t = (y - 0.55) / 0.4;
  if (t > 0.0 && t < 1.0) {
    float w = 0.022 + 0.08 * sin(3.14159 * pow(t, 0.65) * 0.8);
    float u = abs(x) / w;
    float tips = abs(cos(x / w * 2.356));
    if (u < 1.0 && t < 0.88 + 0.12 * tips) {
      vec3 c = vColor * (0.55 + 0.55 * t + 0.4 * (1.0 - u));
      c *= 1.0 - 0.4 * exp(-pow((u - 0.62) / 0.07, 2.0));
      c += vec3(0.05, 0.25, 0.6) * 0.5 * smoothstep(0.7, 1.0, u) * t;
      float h2 = hash(floor(gl_PointCoord * 30.0) + vSeed * 13.0);
      c += vec3(0.6, 0.85, 1.0) * step(0.993, h2) * (0.6 + 0.4 * sin(uTime * 2.0 + vSeed.x * 30.0)) * 2.5;
      return vec4(c, 1.0);
    }
  }
  if (abs(x) < 0.011 && y > 0.02 && y < 0.6) return vec4(vec3(0.03, 0.2, 0.1) * (0.6 + 0.8 * y), 1.0);
  float sh;
  if (blade(p, -1.0, 0.44, 0.03, sh) > 0.5) return vec4(vec3(0.02, 0.17, 0.09) * (0.5 + 0.8 * sh), 1.0);
  if (blade(p, 1.0, 0.34, 0.03, sh) > 0.5) return vec4(vec3(0.02, 0.17, 0.09) * (0.5 + 0.8 * sh), 1.0);
  return vec4(0.0);
}

vec4 orbOld() {
  vec2 q = gl_PointCoord - 0.5;
  float r = length(q) * 2.0;
  if (r > 1.0) return vec4(0.0);
  return vec4(vColor2 * 1.6 + vec3(0.2, 0.5, 1.0) * 1.4 * (1.0 - r), 1.0);
}

// Dispersed flower = a small glowing orb
vec4 orb() {
  vec2 q = gl_PointCoord - 0.5;
  float r = length(q) * 2.0;
  if (r > 1.0) return vec4(0.0);
  return vec4(vColor * 1.1 + vec3(0.2, 0.45, 0.9) * 0.6 * (1.0 - r), 1.0);
}

vec4 shapeSoft(vec2 p) {
  #ifdef LEAF
    return leafShape(p, vColor);
  #else
    return vDisp > 0.35 ? orb() : tulipShape(p);
  #endif
}
vec4 shapeDeep(vec2 p) {
  #ifdef LEAF
    return leafShape(p, vColor2);
  #else
    return vDisp > 0.35 ? orbOld() : tulipOld(p, vColor2);
  #endif
}

void main() {
  vec2 p = vec2(gl_PointCoord.x - 0.5, 1.0 - gl_PointCoord.y);
  // uNight: 0 = soft powder blue, 1 = deep blue with glow. Only one look is drawn
  // at the two ends; in between the colors blend smoothly (no pattern).
  float n = uNight;
  vec4 r;
  if (n < 0.01) {
    r = shapeSoft(p);
  } else if (n > 0.99) {
    r = shapeDeep(p);
  } else {
    vec4 a = shapeSoft(p);
    vec4 b = shapeDeep(p);
    r = n < 0.5 ? a : b;
    if (a.a > 0.5 && b.a > 0.5) r.rgb = mix(a.rgb, b.rgb, smoothstep(0.0, 1.0, n));
  }
  if (r.a < 0.5) discard;
  vec3 glow = mix(vec3(0.1, 0.3, 0.7) * 0.35, vec3(0.15, 0.5, 1.0) * 1.3, n);
  vec3 col = r.rgb + glow * vGlow;   // points near the mouse glow
  col = mix(col, uFog, vFog);
  gl_FragColor = vec4(col, 1.0);
}
`;
