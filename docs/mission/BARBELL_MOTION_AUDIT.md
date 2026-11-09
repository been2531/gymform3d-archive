# Barbell motion audit — free sources only (t730u)

Date: 2026-11-10 (Asia/Seoul). Access-date for YMove free page: 2026-11-10 via local mirror `_scratch/motion_src/ymove.html` (canonical https://ymove.app/free-exercise-videos). 0 generations. No paid assets.

## YMove free set (25) — barbell-relevant

| Candidate | Free motion? | Evidence | Decision |
|---|---|---|---|
| Barbell Back Squat | **YES** | YMove free #1 `fd0eaa34-…` | playlist `bb_back_squat` ready |
| Barbell Bench Press | **YES** | YMove free #2 `dd7e706c-…` | playlist `bb_bench_press` ready |
| Barbell Deadlift (conventional) | **YES** | YMove free #3 `b59952ca-…` | playlist `bb_deadlift` ready |
| Romanian Deadlift (barbell) | **NO** | Only **Kettlebell** RDL free (`e7175536-…`) | blocked / optional KB proxy |
| Bent-over barbell row | **NO** | Not in free 25. Local `_scratch/motion_src/ymove_bb_row.mp4` is a **market** clip per `REUSABLE_ASSET_PLAN.md` — do not use | blocked |
| Overhead press (barbell) | **NO** | Only Kettlebell Push Press free | blocked (KB proxy only if asked) |
| Barbell curl | **NO** | Hammer Curls / cable curl only | blocked |
| Hip thrust | **NO** | Not in free 25 | blocked |

**Licence (FACT):** YMove free page states royalty-free / commercial use allowed for the 25 HD clips; no resale/redistribution of the videos. Pipeline uses extracted MediaPipe pose only; clips stay gitignored under `_scratch/`.

## Other free/CC0 motion sources checked

| Source | Licence | Gym barbell coverage | Use? |
|---|---|---|---|
| CMU mocap | free incl. commercial; no raw resale | Few gym lifts | No (not coach-grade barbell) |
| Mixamo | free commercial; Adobe login | Push-up / air squat / curl-ish | No without owner sign-in |
| OpenGym3D | MIT but **$19** | Has bent-over row, RDL, etc. | Blocked until purchase approval |
| Fit3D | non-commercial | 47 gym exercises | **Rejected** for monetised IG |
| SMPL/AMASS family | non-commercial / SMPL-bound | — | Rejected |

## Prop module

- Builder: `tools/rig3d/props/barbell_olympic.py` (procedural Olympic bar ~2.2 m, cobalt matte plates, thin brushed gold ring, brushed steel knurl bump).
- Spec: `modules/equipment/barbell_olympic.yaml` — empties `grip.L` / `grip.R` (+ wide) for two-hand reuse.
- Bench: `tools/rig3d/props/bench_flat.py` + `modules/equipment/bench_flat.yaml` (warm grey pad).
- Still: `docs/mission/barbell_olympic_prop_still.png`.
- **Blocker:** shared `retarget_render_v2.py` still parents a **dumbbell** only; barbell/bench grip wiring is deferred so the concurrent quality-upgrade worker is not conflicted.

## Playlist change

Dropped `db_lunge` and `push_up` (no free motion; owner will not film). Added the three YMove-free barbell lifts. See repo-root `playlist.yaml`.
