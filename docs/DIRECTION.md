# @gymform.3d — content direction (t554u)

**What it is:** a global 3D gym encyclopedia. Every reel shows **the same exercise in wrong vs right form** so a
viewer sees at a glance **where the body takes the load**.

- **One exercise, one common mistake per day** (the exercise's `mistakes[0]` in `configs/motion_standards.json`).
- **Mistake type decides the anatomy layer** (`mistake_type`):
  - **joint** mistake (spine rounding, knee valgus, shoulder impingement…): steps 1–2 bone X-Ray showing only the
    exercise's main bones (`target_bones`, like muscle fibre), steps 3–4 muscle.
  - **muscle** mistake (target muscle not getting the load): all 4 steps muscle, wrong vs right, crisper angle callout.
- **Fixed step/camera plan:** 1 wrong/front, 2 right/front, 3 wrong/side, 4 right/side. 2 motions × 2 fixed native 3D
  cameras per exercise; 4 × 5 s = 20 s.
- **Look = Day 4** (GOLD #1): skin tone, backdrop grade, shorts, framing, title/preview, angle callout, tempo.
  Mannequin colour never changes (no tint/darkening around bones).
- **Credible motion and tempo:** ranges and tempo from `configs/motion_standards.json` (sourced: ACSM/NSCA/ACE/EMG).
- **Goals:** saves and comments (clear wrong/right contrast; spoken + caption experience question at the end).
- **Zero spend** unless the owner approves a specific cost. Reuse gold assets first (`gold.json`).
- **Confirm sample frames before rendering:** preflight sheet → owner look → one full render → QC → approval.
