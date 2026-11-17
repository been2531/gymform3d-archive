# stickfig

Procedural, code-drawn stick-figure renderer for @gymform.3d. One shared 2D
joint skeleton (head, neck, shoulders, elbows, hands, hips, knees, feet) and
two fixed styles: a neon stickman and a paper cut-out. No generative images
in the pipeline, so the character cannot drift.

From the `daily_info_reels` repo root (nice 19 so other box work stays
responsive):

```bash
nice -n 19 python3 -m stickfig.render round1
nice -n 19 python3 -m stickfig.render round2
nice -n 19 python3 -m stickfig.render round3
```

Stills and silent test clips are written to
`output/concepts/YYYY-MM-DD/evolve/` and are not committed.
