# Pilot diagnosis (Round 4)

Read-only scripts and metrics. See `docs/mission/BLENDER_FEASIBILITY_DIAGNOSIS.md`.

```bash
# Isolated écorché + pilot shader (no MH pipeline)
studio3d/pilot/blender_cli.sh studio3d/pilot/diagnosis/render_feasibility_ecorche.py

# Why QC passed broken v3 frames
python3 studio3d/pilot/diagnosis/qc_gap_analysis.py
```

Outputs under `output/pilot/diagnosis/`.
