# Exercise standards

Every exercise should have a spec **before** any beauty render. The human template
is [`_TEMPLATE.md`](_TEMPLATE.md). The file QC actually reads is
`exercises/<id>.spec.yaml` (schema documented in the template).

Loader: `tools/rig3d/spec_load.py`. Consumer: `tools/rig3d/qc3d.py` (group `spec`).

Do not commit third-party reference photos — link them with licence and credit.

Look post-mortem for the clay mannequin: [`ROOTCAUSE_LOOK.md`](ROOTCAUSE_LOOK.md) (drawers, glow cracks, paler glow, and the reusable base asset kit).
Implemented kit: [`KIT.md`](KIT.md) (`modules/kit.yaml`, `tools/rig3d/kit/`).
Grab3D char1 was declared a failure; replacement search (none adopted): [`BASE_MODEL_SEARCH.md`](BASE_MODEL_SEARCH.md).

## Index

| id | name | spec | status |
|---|---|---|---|
| `db_row` | Bench one-arm dumbbell row | [`exercises/db_row.spec.yaml`](../../exercises/db_row.spec.yaml) | filled |
| `bb_back_squat` | Barbell Back Squat | — | pending |
| `bb_bench_press` | Barbell Bench Press | — | pending |
| `bb_deadlift` | Barbell Deadlift | — | pending |
| `db_goblet_squat` | Dumbbell Goblet Squat | — | pending |
| `db_lunge` | Dumbbell Lunge | — | pending |
| `db_shoulder_press` | Dumbbell Shoulder Press | — | pending |
| `hammer_curl` | Hammer Curls | — | pending |
| `push_up` | Push-Up | — | pending |
