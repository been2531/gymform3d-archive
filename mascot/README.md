# Vinyl muscle mascot (Concept 3 — "If your muscles could talk")

A single procedural character for @gymform.3d. The Python script is the
source of truth: it builds the body from metaballs, a 6-bone armature, and
a library of swappable face pieces. No paid assets. No image generation.

## Pass criteria

1. Reads like `concept3_sheet.jpg`: glossy pink-red vinyl, chubby, cute,
   figure-like mascot, big thighs, short arms, little feet, dark charcoal studio.
2. The face is fixed. Three expression kits (smug / scream / faint) are
   swappable mesh pieces on the same sockets — they do not regenerate.
3. Readable at 1080×1920 on a phone: the expression is clear at a glance.
4. Simple: one smooth body and a few motions (bounce, shake, fall). No
   complex human rig.

## Render

From the repo root, with Blender 4.2.3:

```bash
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  -P mascot/build_mascot.py -- \
  --out output/mascot/2026-11-17 --mode stills

python3 mascot/compose.py --out output/mascot/2026-11-17 --mode stills
```

`--mode preview` renders 540×960 contact stills. `--mode reel` (or `all`)
builds a 13 s, 24 fps silent test reel; then:

```bash
python3 mascot/compose.py --out output/mascot/2026-11-17 --mode reel --fps 24
```

Renders land in `output/` and are not committed. Typography is one English
label of at most three words, white uppercase on a black pill, top-centre,
with the bottom 15% of the frame left clear.
