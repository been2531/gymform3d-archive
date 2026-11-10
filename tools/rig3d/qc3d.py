#!/usr/bin/env python3
"""Auto-QC for a rig3d run (Routinizer v2): rig checks (checks.json from retarget_render_v2.py), Day 4 look checks
(look.json from grade3d.py) and the encoded reel. Every check = {name, value, limit, pass, fix} where `fix` names the
rule the pipeline may apply automatically (1 retry), or null = needs a bot (AI queue).
usage: python3 tools/rig3d/qc3d.py RUN_DIR CFG.json  -> RUN_DIR/qc.json"""
import json, os, subprocess, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY4 = os.path.join(ROOT, "output/day4_romanian_deadlift_4step_master.mp4")

def day4_sharp_ref(cache):
    """Day 4 muscle-step frames (7.5 s, 18.5 s), figure only: neutral-grey body pixels below the title, barbell band and
    plates cut out, glow/laser/callouts excluded by sharp.neutral_lapvar (same metric as our frames)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sharp import neutral_lapvar
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("metric") == "neutral_lapvar_v2": return c["lapvar"]
    vals = []
    for t in (7.5, 18.5):
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", DAY4, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        f = np.frombuffer(p, np.uint8).reshape(1920, 1080, 3)
        body = np.abs(f.astype(np.float32) - np.array([35, 39, 43], np.float32)).max(-1) > 10
        body[:600] = False; body[:, :280] = False; body[:, 800:] = False
        vals.append(neutral_lapvar(f, body, [(930, 1130)])[0])
    v = float(np.median(vals)); json.dump({"lapvar": v, "per_frame": vals, "metric": "neutral_lapvar_v2", "src": DAY4}, open(cache, "w")); return v

def probe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", mp4], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")

# v6: every check has an auto-fix rule (pipeline.RULES) or is explicitly "human" (needs a person / AI). A check missing
# here is reported as "unmapped" in qc.json (treated as human) so new checks cannot silently go without a policy.
RULE_MAP = {
    "arm_crossing_frames": "human", "torso_median_deg": "torso_target", "torso_steady_maxdev_deg": "torso_smooth", "dumbbell_grip_bone_m": "grip_offset",
    "dumbbell_to_hand_mesh_m": "grip_offset", "mesh_tear_frac_edges_gt2x": "spine_spread", "working_elbow_rom_deg": "human",
    "neck_in_line_with_spine_deg": "neck_follow_spine", "neck_flex_deg_min_max": "neck_follow_spine", "head_roll_deg": "neck_follow_spine", "head_yaw_deg": "neck_follow_spine",
    "neck_steady_maxdev_deg": "neck_follow_spine", "both_feet_planted": "plant_legs", "foot_drift_m": "plant_legs", "knee_angles_deg": "plant_legs",
    "knee_steady_maxdev_deg": "plant_legs", "knees_soft_and_balanced": "balance_knees", "dumbbell_handle_orientation": "grip_palm", "dumbbell_neutral_grip_deg": "grip_palm",
    "support_hand_on_thigh_m": "support_hand", "thigh_split_deg": "plant_legs", "posture_bones_present": "human",
    "joint_ranges_no_hyperextension": "joint_range", "self_intersection_capsule_m": "limb_clearance", "center_of_mass_over_feet_m": "com_balance",
    "proportions_symmetric_and_rigid": "human", "grip_present": "grip_wrap", "grip_finger_wrap_deg": "grip_wrap", "grip_fingertip_gap_mm": "grip_wrap",
    "grip_thumb_opposing": "grip_wrap", "grip_no_hand_handle_intersection": "grip_wrap", "mesh_no_spikes_edge_ratio": "spine_spread",
    "prop_dumbbell_heads_readable": "camera_yaw", "extremities_in_frame_margin_px": "camera_margin",
    "skin_p50": "human", "skin_p75": "human", "skin_p90": "human", "skin_max": "human", "backdrop_rgb_maxdev": "human", "floor_shadow_present": "human", "floor_horizon_edge": "human", "floor_luma_vs_backdrop": "human", "glow_fibre_contrast": "human", "glow_hue_sat_band": "human", "green_glow_frac_of_body": "human",
    "green_only_on_lats_rhomboids_midtrap": "human", "no_red_glow_px": "human", "sharpness_vs_day4": "human", "figure_fully_in_frame": "camera_margin",
    "figure_fill_w_h": "camera_tighter", "figure_size_not_below_target": "camera_bigger", "green_area_ratio_cap": "human",
    "overlay_in_safe_zone": "human", "overlay_text_no_overlap": "human", "overlay_clear_of_figure_px": "camera_top_margin", "callout_text_clear_of_figure": "human",
    "callout_angle_matches_measured": "human", "briefs_present": "human", "briefs_body_touch_frac": "human", "briefs_no_tearing_frac2x": "human",
    "briefs_no_visible_poke_through": "human", "briefs_coverage_every_frame": "human", "reel_1080x1920_30fps": "human", "reel_has_audio": "human",
    "reel_duration_s": "human", "tts_ends_before_reel_end": "human", "caption_lint": "human",
}

def apply_rule_map(out):
    for c in out:
        m = RULE_MAP.get(c["name"]); c["rule_map"] = m or "unmapped"
        if not c["pass"]:
            c["fix"] = None if m in (None, "human") else m
    return out

def ck(name, value, limit, ok, fix=None, group="rig"):
    return {"name": name, "group": group, "value": value, "limit": limit, "pass": bool(ok), "fix": None if ok else fix}

def rig_checks(rdir, cfg):
    """Every check that needs only the rig pass (retarget --checks-only): rig, posture, anatomy, grip, prop. Used by the
    full QC and by the pipeline's cheap rig pre-check loop (no render)."""
    Q = cfg["qc"]; out = []
    R = json.load(open(os.path.join(rdir, "checks.json"))); F = R["frames"]
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
    # posture (generic, tools/rig3d/posture_qc.py): neck/head in line with the spine, feet planted, knees, stance
    if Q.get("posture"):
        bz = os.path.join(rdir, "bones.npz")
        if not os.path.exists(bz): out.append(ck("posture_bones_present", False, True, False, None, "posture"))
        else:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import posture_qc
            pc, series = posture_qc.checks(bz, Q["posture"]); out += pc
            json.dump(series, open(os.path.join(rdir, "posture_series.json"), "w"))
    if Q.get("anatomy"):
        import anatomy_qc; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        out += anatomy_qc.run(rdir, Q["anatomy"], cfg)
    return out, F, med

def look_limits(cfg):
    """Skin bands follow grade.look (charcoal / clay). Missing preset keys fall back to qc.*."""
    Q = cfg["qc"]; name = str((cfg.get("grade") or {}).get("look", "charcoal")).lower()
    p = (Q.get("look_presets") or {}).get(name) or {}
    return {"skin_p50": p.get("skin_p50", Q.get("skin_p50")), "skin_p75": p.get("skin_p75", Q["skin_p75"]),
            "skin_p90": p.get("skin_p90", Q["skin_p90"]), "skin_max": p.get("skin_max", Q["skin_max"]), "look": name}

def run(run_dir, cfg):
    Q = cfg["qc"]; out, F, med = rig_checks(os.path.join(run_dir, "render"), cfg)
    # Day 4 look (skin bands depend on grade.look)
    L = json.load(open(os.path.join(run_dir, "grade/look.json")))["frames"]
    lim = look_limits(cfg)
    def rng(key, lim_, fix, name=None):
        v = [f[key] for f in L if f.get(key) is not None]
        lo_, hi_ = (min(v), max(v)) if v else (None, None)
        ok = bool(v) and lim_ is not None and lim_[0] <= lo_ and hi_ <= lim_[1]
        out.append(ck(name or key, [round(lo_, 3), round(hi_, 3)] if v else None, lim_, ok, fix, "look"))
    if lim.get("skin_p50"):
        rng("skin_p50", lim["skin_p50"], "skin_lock")
    rng("skin_p75", lim["skin_p75"], "skin_lock"); rng("skin_p90", lim["skin_p90"], "skin_lock")
    mx = max((f["skin_max"] for f in L if f.get("skin_max") is not None), default=None)
    out.append(ck("skin_max", None if mx is None else round(mx, 1), lim["skin_max"], mx is not None and mx <= lim["skin_max"], "skin_lock", "look"))
    bd = [f["backdrop_rgb"] for f in L if f.get("backdrop_rgb")]; tgt = np.array(cfg["grade"]["backdrop_rgb"], float)
    bdev = float(np.abs(np.array(bd) - tgt).max()) if bd else 99
    out.append(ck("backdrop_rgb_maxdev", round(bdev, 1), Q["backdrop_tol"], bdev <= Q["backdrop_tol"], None, "look"))
    if "floor_shadow_min" in Q:
        ds = [f.get("floor_shadow_delta") for f in L if f.get("floor_shadow_delta") is not None]
        mv = min(ds) if ds else None
        out.append(ck("floor_shadow_present", None if mv is None else round(float(mv), 2),
                      f">= {Q['floor_shadow_min']}", bool(ds) and mv >= Q["floor_shadow_min"], None, "look"))
    if "floor_horizon_jump_max" in Q:
        hv = [f.get("floor_horizon_jump") for f in L if f.get("floor_horizon_jump") is not None]
        mx = max(hv) if hv else None
        out.append(ck("floor_horizon_edge", None if mx is None else round(float(mx), 2),
                      f"<= {Q['floor_horizon_jump_max']}", bool(hv) and mx <= Q["floor_horizon_jump_max"], None, "look"))
    if "floor_luma_delta_max" in Q:
        dv = [f.get("floor_bg_delta") for f in L if f.get("floor_bg_delta") is not None]
        mx = max(dv) if dv else None
        out.append(ck("floor_luma_vs_backdrop", None if mx is None else round(float(mx), 2),
                      f"<= {Q['floor_luma_delta_max']}", bool(dv) and mx <= Q["floor_luma_delta_max"], None, "look"))
    if "glow_fibre_hp_min" in Q:
        fv = [f.get("glow_fibre_hp_std") for f in L if f.get("glow_fibre_hp_std") is not None]
        mn = min(fv) if fv else None
        out.append(ck("glow_fibre_contrast", None if mn is None else round(float(mn), 2),
                      f">= {Q['glow_fibre_hp_min']}", bool(fv) and mn >= Q["glow_fibre_hp_min"], None, "look"))
    if Q.get("glow_hue") or Q.get("glow_sat"):
        hues = [f.get("glow_hue_p50") for f in L if f.get("glow_hue_p50") is not None]
        sats = [f.get("glow_sat_p50") for f in L if f.get("glow_sat_p50") is not None]
        hlim = Q.get("glow_hue"); slim = Q.get("glow_sat")
        hok = bool(hues) and (not hlim or (min(hues) >= hlim[0] and max(hues) <= hlim[1]))
        sok = bool(sats) and (not slim or (min(sats) >= slim[0] and max(sats) <= slim[1]))
        out.append(ck("glow_hue_sat_band",
                      {"hue_p50": [round(min(hues), 1), round(max(hues), 1)] if hues else None,
                       "sat_p50": [round(min(sats), 1), round(max(sats), 1)] if sats else None},
                      {"hue": hlim, "sat": slim}, bool(hok and sok), None, "look"))
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
    if "figure_fill_h_min" in Q:          # v6: the figure must not shrink below the target (any camera/overlay fix)
        mh = float(np.min((bb[:, 3] - bb[:, 1]) / H)); out.append(ck("figure_size_not_below_target", round(mh, 3), f">= {Q['figure_fill_h_min']} (min over frames)", mh >= Q["figure_fill_h_min"], "camera_bigger", "look"))
    if "glow_frac_max" in Q:              # v6: green area cap (fraction of the body that glows), every frame
        gm_ = max(f["glow_frac"] for f in L); out.append(ck("green_area_ratio_cap", round(gm_, 4), f"<= {Q['glow_frac_max']}", gm_ <= Q["glow_frac_max"], "glow_cap", "look"))
    # Day 4 overlay: callout + title + preview pill
    if cfg.get("layout"):
        sz = Q["safe_zone"]; LF = json.load(open(os.path.join(run_dir, "grade/look.json")))
        def inside_sz(b): return b[0] >= sz[0] and b[1] >= sz[1] and b[2] <= sz[2] and b[3] <= sz[3]
        def inter(a, b): return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
        bad_sz, bad_ov, n_ui, figov, d2 = [], [], 0, [], []
        for f in L:
            u = f.get("ui") or {}; boxes = {k: u[k] for k in ("title", "pill") if k in u}
            if "callout" in u: boxes["callout_text"] = u["callout"]["text_box"]; figov.append(f.get("callout_fig_overlap", 1.0)); d2.append(u["callout"]["angle_2d_deg"])
            n_ui += len(boxes)
            bad_sz += [f"{f['f']}:{k}" for k, b in boxes.items() if not inside_sz(b)]
            ks = list(boxes); bad_ov += [f"{f['f']}:{a}/{b}" for i, a in enumerate(ks) for b in ks[i + 1:] if inter(boxes[a], boxes[b])]
        out.append(ck("overlay_in_safe_zone", bad_sz[:5] or f"{n_ui} boxes ok", sz, not bad_sz and n_ui > 0, None, "overlay"))
        out.append(ck("overlay_text_no_overlap", bad_ov[:5] or "ok", "title / pill / callout text disjoint", not bad_ov, None, "overlay"))
        if "overlay_figure_px_max" in Q:     # v5 generic: title / pill text boxes clear of the figure mask, every frame
            have = [f["overlay_fig_px"] for f in L if "overlay_fig_px" in f]
            worst = {k: max(h.get(k, 0) for h in have) for k in ("title", "pill")} if have else None
            okf = bool(have) and len(have) == len(L) and max(worst.values()) <= Q["overlay_figure_px_max"]
            out.append(ck("overlay_clear_of_figure_px", worst, f"<= {Q['overlay_figure_px_max']} figure px in title/pill box, every frame", okf, "camera_top_margin", "overlay"))
        if cfg["layout"].get("callout"):
            mo = max(figov) if figov else 1.0
            out.append(ck("callout_text_clear_of_figure", round(mo, 4), Q["callout_figure_overlap_max"], bool(figov) and mo <= Q["callout_figure_overlap_max"], None, "overlay"))
            shown = LF.get("callout_shown_deg"); meas = int(round(med)); dd = float(np.max(np.abs(np.array(d2) - shown))) if d2 else 99
            out.append(ck("callout_angle_matches_measured", {"shown": shown, "rig_median": round(med, 1), "max_2d_dev": round(dd, 1)},
                          f"shown == round(median); |2D - shown| <= {Q['callout_2d_tol_deg']}", shown == meas and dd <= Q["callout_2d_tol_deg"], None, "overlay"))
    # black boxer briefs (separate skinned garment mesh)
    BF = [f for f in F if "briefs_poke_frac" in f]
    if cfg.get("garment", {}).get("enabled"):
        if not BF:
            out.append(ck("briefs_present", False, True, False, None, "briefs"))
        else:
            tch = max(f["briefs_poke_frac"] for f in BF); tr = max(f["briefs_tear_frac2x"] for f in BF)
            out.append(ck("briefs_body_touch_frac", round(tch, 5), Q["briefs_touch_frac_max"], tch <= Q["briefs_touch_frac_max"], None, "briefs"))
            out.append(ck("briefs_no_tearing_frac2x", round(tr, 5), Q["briefs_tear_frac2x_max"], tr <= Q["briefs_tear_frac2x_max"], None, "briefs"))
            bp = [f["briefs_px"] for f in L if "briefs_px" in f]
            vp = max((f["briefs_poke_px"] / max(f["briefs_px"], 1)) for f in L if "briefs_px" in f) if bp else 1.0
            out.append(ck("briefs_no_visible_poke_through", round(vp, 5), Q["briefs_visible_poke_max"], bool(bp) and vp <= Q["briefs_visible_poke_max"], None, "briefs"))
            cov = (min(bp) / float(np.median(bp))) if bp else 0.0; gob = max((f.get("green_on_briefs_px", 0) for f in L), default=0)
            okc = len(bp) == len(L) and cov >= Q["briefs_cover_min_rel"] and gob == 0
            out.append(ck("briefs_coverage_every_frame", {"frames": f"{len(bp)}/{len(L)}", "min_rel": round(cov, 3), "green_on_briefs_px": gob},
                          f"all frames, >= {Q['briefs_cover_min_rel']} x median, 0 green", okc, None, "briefs"))
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
    out = apply_rule_map(out)
    res = {"pass": all(c["pass"] for c in out), "n_pass": sum(c["pass"] for c in out), "n": len(out), "checks": out,
           "auto_fixable": sorted({c["fix"] for c in out if not c["pass"] and c["fix"]}),
           "needs_ai": [c["name"] for c in out if not c["pass"] and not c["fix"]],
           "unmapped_checks": sorted(c["name"] for c in out if c.get("rule_map") == "unmapped")}
    json.dump(res, open(os.path.join(run_dir, "qc.json"), "w"), indent=1)
    return res

if __name__ == "__main__":
    r = run(sys.argv[1], json.load(open(sys.argv[2])))
    for c in r["checks"]: print(("PASS " if c["pass"] else "FAIL ") + c["name"], c["value"], "limit", c["limit"])
    print("QC", "PASS" if r["pass"] else "FAIL", f"{r['n_pass']}/{r['n']}")
