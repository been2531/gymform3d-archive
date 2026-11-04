#!/usr/bin/env python3
"""Auto-QC for a rig3d run (Routinizer v2): rig checks (checks.json from retarget_render_v2.py), Day 4 look checks
(look.json from grade3d.py) and the encoded reel. Every check = {name, value, limit, pass, fix} where `fix` names the
rule the pipeline may apply automatically (1 retry), or null = needs a bot (AI queue).
usage: python3 tools/rig3d/qc3d.py RUN_DIR CFG.json  -> RUN_DIR/qc.json"""
import json, os, subprocess, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY4 = os.path.join(ROOT, "output/day4_romanian_deadlift_4step_master.mp4")

def day4_sharp_ref(cache):
    if os.path.exists(cache):
        return json.load(open(cache))["lapvar"]
    vals = []
    for t in (13.5, 18.5):
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", DAY4, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        f = np.frombuffer(p, np.uint8).reshape(1920, 1080, 3).astype(np.float32)
        body = (np.abs(f - np.array([35, 39, 43], np.float32)).max(-1) > 22); body[:600] = False; body[:, :300] = False; body[:, 780:] = False   # figure only (no plates, callouts, UI)
        body = cv2.erode(body.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        lap = cv2.Laplacian(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F); vals.append(float(lap[body].var()))
    v = float(np.median(vals)); json.dump({"lapvar": v, "src": DAY4}, open(cache, "w")); return v

def probe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", mp4], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")

def ck(name, value, limit, ok, fix=None, group="rig"):
    return {"name": name, "group": group, "value": value, "limit": limit, "pass": bool(ok), "fix": None if ok else fix}

def run(run_dir, cfg):
    Q = cfg["qc"]; out = []
    R = json.load(open(os.path.join(run_dir, "render/checks.json"))); F = R["frames"]
    t = np.array([f["torso_deg"] for f in F]); med = float(np.median(t))
    out.append(ck("arm_crossing_frames", int(sum(f["arm_cross"] for f in F)), 0, sum(f["arm_cross"] for f in F) == 0))
    out.append(ck("torso_median_deg", round(med, 1), Q["torso_band_deg"], Q["torso_band_deg"][0] <= med <= Q["torso_band_deg"][1], None))
    dev = float(np.abs(t - med).max())
    out.append(ck("torso_steady_maxdev_deg", round(dev, 2), Q["torso_tol_deg"], dev <= Q["torso_tol_deg"], "torso_smooth"))
    g = max(f["db_grip_m"] for f in F); tm = [f for f in F if "grip_mesh_m" in f]; gm = max(f["grip_mesh_m"] for f in tm)
    out.append(ck("dumbbell_grip_bone_m", round(g, 4), Q["grip_max_m"], g <= Q["grip_max_m"], "grip_offset"))
    out.append(ck("dumbbell_to_hand_mesh_m", round(gm, 4), Q["grip_mesh_max_m"], gm <= Q["grip_mesh_max_m"], "grip_offset"))
    tf = max(f["tear_frac2x"] for f in tm); worst = max(tm, key=lambda f: f["tear_frac2x"])
    out.append(ck("mesh_tear_frac_edges_gt2x", round(tf, 5), Q["tear_frac2x_max"], tf <= Q["tear_frac2x_max"], "spine_spread"))
    out[-1]["where"] = {"frame": worst["f"], "bones": worst.get("tear_where"), "max_ratio": worst["tear_max_ratio"]}
    el = np.array([f["elbow_deg"] for f in F]); rom = float(el.max() - el.min())
    out.append(ck("working_elbow_rom_deg", round(rom, 1), f">= {Q['elbow_rom_min_deg']}", rom >= Q["elbow_rom_min_deg"]))
    # Day 4 look
    L = json.load(open(os.path.join(run_dir, "grade/look.json")))["frames"]
    def rng(key, lim, fix, name=None):
        v = [f[key] for f in L if f.get(key) is not None]
        lo_, hi_ = (min(v), max(v)) if v else (None, None)
        ok = bool(v) and lim[0] <= lo_ and hi_ <= lim[1]
        out.append(ck(name or key, [round(lo_, 3), round(hi_, 3)] if v else None, lim, ok, fix, "look"))
    rng("skin_p75", Q["skin_p75"], "skin_lock"); rng("skin_p90", Q["skin_p90"], "skin_lock")
    mx = max(f["skin_max"] for f in L if f.get("skin_max") is not None)
    out.append(ck("skin_max", round(mx, 1), Q["skin_max"], mx <= Q["skin_max"], "skin_lock", "look"))
    bd = [f["backdrop_rgb"] for f in L if f.get("backdrop_rgb")]; tgt = np.array(cfg["grade"]["backdrop_rgb"], float)
    bdev = float(np.abs(np.array(bd) - tgt).max()) if bd else 99
    out.append(ck("backdrop_rgb_maxdev", round(bdev, 1), Q["backdrop_tol"], bdev <= Q["backdrop_tol"], None, "look"))
    rng("glow_frac", Q["glow_frac"], None, "green_glow_frac_of_body")
    gi = min(f["glow_in_mask"] for f in L)
    out.append(ck("green_only_on_lats_rhomboids_midtrap", round(gi, 3), f">= {Q['glow_in_mask_min']}", gi >= Q["glow_in_mask_min"], None, "look"))
    rp = max(f["red_px"] for f in L)
    out.append(ck("no_red_glow_px", rp, 0, rp == 0, None, "look"))
    ref = day4_sharp_ref(os.path.join(run_dir, "day4_sharp_ref.json")); sh = float(np.median([f["sharp_lapvar"] for f in L]))
    out.append(ck("sharpness_vs_day4", round(sh / ref, 3), f">= {Q['sharp_rel_min']}", sh / ref >= Q["sharp_rel_min"], None, "look"))
    W, H = cfg["grade"]["out_res"]; bb = np.array([f["bbox"] for f in L if f.get("bbox")])
    inside = bool((bb[:, 0] > 4).all() and (bb[:, 2] < W - 4).all() and (bb[:, 1] > 4).all() and (bb[:, 3] < H - 4).all())
    fillw = float(np.median((bb[:, 2] - bb[:, 0]) / W)); fillh = float(np.median((bb[:, 3] - bb[:, 1]) / H))
    out.append(ck("figure_fully_in_frame", inside, True, inside, "camera_margin", "look"))
    out.append(ck("figure_fill_w_h", [round(fillw, 3), round(fillh, 3)], "w >= 0.60 or h >= 0.55", fillw >= 0.60 or fillh >= 0.55, "camera_tighter", "look"))
    # reel
    mp4 = os.path.join(run_dir, "reel", f"{cfg['slug']}_rig3d_v2.mp4"); pr = probe(mp4) if os.path.exists(mp4) else {}
    st = pr.get("streams", []); v = [s for s in st if s.get("codec_type") == "video"]; a = [s for s in st if s.get("codec_type") == "audio"]
    dur = float(pr.get("format", {}).get("duration", 0))
    out.append(ck("reel_1080x1920_30fps", [v[0]["width"], v[0]["height"], v[0]["r_frame_rate"]] if v else None, [W, H, "30/1"],
                  bool(v) and v[0]["width"] == W and v[0]["height"] == H and v[0]["r_frame_rate"] == "30/1", None, "reel"))
    out.append(ck("reel_has_audio", bool(a), True, bool(a), None, "reel"))
    out.append(ck("reel_duration_s", round(dur, 2), Q["dur_s"], Q["dur_s"][0] <= dur <= Q["dur_s"][1], None, "reel"))
    tj = os.path.join(run_dir, "reel/tts.json")
    if os.path.exists(tj):
        T = json.load(open(tj)); out.append(ck("tts_ends_before_reel_end", T["end_s"], f"<= {round(dur - 0.2, 2)}", T["end_s"] <= dur - 0.2, None, "reel"))
    cj = os.path.join(run_dir, "reel/caption_lint.json")
    if os.path.exists(cj):
        C = json.load(open(cj)); out.append(ck("caption_lint", C["errors"] or "ok", "no errors", not C["errors"], None, "reel"))
    res = {"pass": all(c["pass"] for c in out), "n_pass": sum(c["pass"] for c in out), "n": len(out), "checks": out,
           "auto_fixable": sorted({c["fix"] for c in out if not c["pass"] and c["fix"]}),
           "needs_ai": [c["name"] for c in out if not c["pass"] and not c["fix"]]}
    json.dump(res, open(os.path.join(run_dir, "qc.json"), "w"), indent=1)
    return res

if __name__ == "__main__":
    r = run(sys.argv[1], json.load(open(sys.argv[2])))
    for c in r["checks"]: print(("PASS " if c["pass"] else "FAIL ") + c["name"], c["value"], "limit", c["limit"])
    print("QC", "PASS" if r["pass"] else "FAIL", f"{r['n_pass']}/{r['n']}")
