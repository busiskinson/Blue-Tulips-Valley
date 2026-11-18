import "./style.css";
import { Valley } from "./valley.js";
import { audio } from "./audio.js";

const ui = document.getElementById("ui");
const loader = document.getElementById("loader");
const sound = document.getElementById("sound");
const go = document.getElementById("go");
const goLabel = document.getElementById("goLabel");
const letter = document.getElementById("letter");
const closeBtn = document.getElementById("lclose");

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/* ---------- The song ("Loose" file in the public folder) ---------- */
const song = new Audio(import.meta.env.BASE_URL + "loose.mp3"); // works on GitHub Pages sub-paths too
song.loop = true;
song.preload = "auto";
song.volume = 0;
let songActive = false;
let fadeRaf = 0;

function fadeSong(to, ms) {
  cancelAnimationFrame(fadeRaf);
  const from = song.volume, t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    song.volume = Math.min(1, Math.max(0, from + (to - from) * k));
    if (k < 1) fadeRaf = requestAnimationFrame(step);
    else if (to === 0 && !songActive) song.pause();
  };
  fadeRaf = requestAnimationFrame(step);
}

/* ---------- Open / close the message ---------- */
function setLetter(open) {
  letter.classList.toggle("on", open);
  letter.setAttribute("aria-hidden", String(!open));
  ui.classList.toggle("reading", open);
  go.setAttribute("aria-expanded", String(open));
  goLabel.textContent = open ? "Close the letter" : "Open the letter";
  songActive = open;
  if (valley) valley.setNight(open);
  if (open) setTimeout(() => closeBtn.focus({ preventScroll: true }), 300);
  else go.focus({ preventScroll: true });
  if (open) {
    if (audio.isInitialized && !audio.isMuted) audio.toggleMute(); // the song replaces the ambient pad
    song.muted = false;
    song.play().then(() => fadeSong(0.85, 2500)).catch(() => {});
    sound.setAttribute("aria-pressed", "true");
  } else {
    fadeSong(0, 1200);
    sound.setAttribute("aria-pressed", "false");
  }
}
go.addEventListener("click", () => setLetter(true));
closeBtn.addEventListener("click", () => setLetter(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && songActive) setLetter(false); });

let valley = null;

if (!hasWebGL()) {
  // No WebGL: show the page with a plain dark gradient behind the text
  document.documentElement.classList.add("no-webgl");
  loader.classList.add("gone");
  ui.classList.add("on");
  sound.onclick = () => {
    if (!songActive) return;
    song.muted = !song.muted;
    sound.setAttribute("aria-pressed", String(!song.muted));
  };
} else {
  let introShown = false;
  valley = new Valley(
    document.getElementById("valley"),
    () => loader.classList.add("gone"),
    () => { if (!introShown) { introShown = true; ui.classList.add("on"); } }
  );

  if (import.meta.env.DEV) window.__valley = valley; // handy for debugging in the browser console
  audio.initAudio();
  // Sound button: controls the song while it is playing, otherwise the soft ambient sound
  sound.onclick = () => {
    if (songActive) {
      song.muted = !song.muted;
      sound.setAttribute("aria-pressed", String(!song.muted));
      return;
    }
    if (!audio.isInitialized) return;
    const muted = audio.toggleMute();
    sound.setAttribute("aria-pressed", String(!muted));
  };

  // Hovering the button scatters the flowers into light and softens the music
  const start = () => { valley.startHoverTransition(); if (songActive) fadeSong(0.35, 600); };
  const stop = () => { valley.stopHoverTransition(); if (songActive) fadeSong(0.85, 800); };
  // Mouse: hover. Touch/pen: press and hold (there is no hover). Keyboard: focus.
  const isMouse = (e) => e.pointerType === "mouse";
  go.addEventListener("pointerenter", (e) => { if (isMouse(e)) start(); });
  go.addEventListener("pointerleave", (e) => { if (isMouse(e)) stop(); });
  go.addEventListener("pointerdown", (e) => { if (!isMouse(e)) start(); });
  go.addEventListener("pointerup", (e) => { if (!isMouse(e)) stop(); });
  go.addEventListener("pointercancel", (e) => { if (!isMouse(e)) stop(); });
  go.addEventListener("contextmenu", (e) => { if (e.pointerType && !isMouse(e)) e.preventDefault(); });
  go.addEventListener("focus", () => { if (go.matches(":focus-visible")) start(); });
  go.addEventListener("blur", stop);
}

if (import.meta.hot) import.meta.hot.dispose(() => { song.pause(); valley && valley.dispose(); });
