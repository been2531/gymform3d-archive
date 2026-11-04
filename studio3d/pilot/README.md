# studio3d/pilot — Day-4-look Blender lat pulldown

Deterministic headless-Blender pilot for **Wide-Grip Lat Pulldown**. See `docs/mission/PILOT_BLENDER_LATPULLDOWN.md`.

Quick start (after `python3 studio3d/fetch_assets.py` and system `blender`):

```bash
chmod +x studio3d/pilot/blender_cli.sh
python3 studio3d/pilot/render_keyposes.py
python3 studio3d/pilot/make_ab_sheet.py
python3 studio3d/pilot/qc_pilot.py
```

Artifacts: `output/pilot/lat/`.
