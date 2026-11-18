BLUE TULIP VALLEY HERO
======================

Run it
  1. Unzip, open the folder in VS Code.
  2. In the terminal:  npm install
  3. Then:             npm run dev
  4. Open the link it prints (usually http://localhost:5173)

Build for hosting:  npm run build   (output goes to the dist folder)

Files
  index.html      page structure (header, headline, arrow button, sound button, loader)
  src/style.css   interface styles
  src/main.js     starts the valley, sound button, hover on the arrow
  src/valley.js   the 3D engine: terrain, camera path, tulips, effects
  src/shaders.js  draws each tulip and leaf inside one point sprite
  src/audio.js    ambient sound made live (no audio file)

Change the look (all in src/valley.js)
  Flower count:  makeSprites(46000 ...) for tulips, makeSprites(34000 ...) for leaves
  Tulip colors:  the bluePalette list
  Fog:           FOG color and 0.0075 (appears twice)
  Idle speed:    this.idle (units per second)
  Glow:          UnrealBloomPass(..., 0.65, 0.5, 0.7) = strength, radius, threshold (lower strength = less glow)

Change the words: index.html (brand name "Fernhollow", headline, caption).

The letter and the song
  public/loose.mp3   the song file (keep this name, or change the path in src/main.js)
  index.html         the message text is inside <aside class="letter">
  The "Open the letter" button shows the message on the left and starts the song.
  Clicking it again closes the message and fades the song out.

When the letter is open
  The headline, caption and button fade away. The flowers dissolve from the soft powder blue
  into the original deep blue with glow, and the sky and fog turn darker. The Close button
  (or the Esc key) brings the soft look and the headline back.
  Deep-blue flower colors: "deepBlue" list in src/valley.js. Glow while open: the 0.25, 0.2
  and 0.5 numbers next to "bloom" in src/valley.js.


