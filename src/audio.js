/* Ambient sound made live with the Web Audio API (no audio file needed).
   A soft pad of slow sine notes plus gentle wind noise.
   It starts muted. Browsers only allow sound after a click, so the
   audio context is resumed when the person presses the sound button. */

let ctx, master, filter;
let inited = false;
let muted = true;

export const audio = {
  get isInitialized() { return inited; },
  get isMuted() { return muted; },

  initAudio() {
    if (inited) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();

    master = ctx.createGain();
    master.gain.value = 0;
    filter = ctx.createBiquadFilter();           // hover muffles the sound through this
    filter.type = "lowpass";
    filter.frequency.value = 9000;
    filter.connect(master);
    master.connect(ctx.destination);

    // Pad: A major-ish chord, each note doubled and slightly detuned
    [110, 164.81, 220, 277.18, 329.63].forEach((f, i) => {
      for (const detune of [-5, 5]) {
        const o = ctx.createOscillator();
        o.type = "sine";
        o.frequency.value = f;
        o.detune.value = detune;
        const g = ctx.createGain();
        g.gain.value = 0.035;
        const lfo = ctx.createOscillator();      // slow breathing
        lfo.frequency.value = 0.04 + i * 0.017;
        const lg = ctx.createGain();
        lg.gain.value = 0.025;
        lfo.connect(lg).connect(g.gain);
        o.connect(g).connect(filter);
        o.start();
        lfo.start();
      }
    });

    // Wind: looping noise through a band-pass filter
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    const noise = ctx.createBufferSource();
    noise.buffer = buf;
    noise.loop = true;
    const band = ctx.createBiquadFilter();
    band.type = "bandpass";
    band.frequency.value = 450;
    band.Q.value = 0.6;
    const wg = ctx.createGain();
    wg.gain.value = 0.05;
    const wl = ctx.createOscillator();
    wl.frequency.value = 0.08;
    const wlg = ctx.createGain();
    wlg.gain.value = 0.035;
    wl.connect(wlg).connect(wg.gain);
    noise.connect(band).connect(wg).connect(filter);
    noise.start();
    wl.start();

    inited = true;
  },

  // Returns true when the sound is now muted
  toggleMute() {
    if (!inited) return true;
    if (ctx.state === "suspended") ctx.resume();
    muted = !muted;
    const t = ctx.currentTime;
    master.gain.cancelScheduledValues(t);
    master.gain.setValueAtTime(master.gain.value, t);
    master.gain.linearRampToValueAtTime(muted ? 0 : 0.8, t + 1.2);
    return muted;
  },

  setMuffled(on) {
    if (!inited) return;
    filter.frequency.setTargetAtTime(on ? 450 : 9000, ctx.currentTime, 0.5);
  },
};
