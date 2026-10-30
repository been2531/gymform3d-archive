"""Locked 3D set -> Day 4 look (t562u). Takes the passes written by studio3d/build_scene.py and applies the SAME
shared Day 4 pieces the clip pipeline uses: Day 4 backdrop grade (bg.studio_bg), Day 4 skin lock (spec
mannequin.skin_lock_all_steps) on the skin ID mask, Day 4 shorts tone, X-Ray bones (target bones only, on skin
pixels only, Day 1 v36 brightness), Day 4 callout (layout.draw_callout_day4) and title/preview.
usage: python3 studio3d/composite.py _tmp_3d/proof --day 6 --step 1 --out output/day6_3d_proof_t562.jpg"""
import argparse, json, os, sys
import cv2
import numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load(p):
    return np.array(Image.open(p).convert("RGBA")).astype(np.float32)


def id_masks(idp):
    r, g, b = idp[..., 0], idp[..., 1], idp[..., 2]
    hi, lo = 160, 90
    return {"skin": (r > hi) & (g < lo) & (b < lo), "shorts": (g > hi) & (r < lo) & (b < lo),
            "bench": (b > hi) & (r < lo) & (g < lo), "dumbbell": (r > hi) & (g > hi) & (b < lo)}


def skin_lock(rgb, skin, spec):
    """same contract as render.skin_tone_lock: monotonic luminance remap p75/p90/max -> Day 4, then Day 4 tint"""
    lk = spec["mannequin"]["skin_lock_all_steps"]
    L = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    core = cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    p50, p75, p90, pmx = (float(np.percentile(L[core], q)) for q in (50, 75, 90, 99.9))
    xs = [0.0, max(p50 * 0.35, 1.0), p50, p75, max(p90, p75 + 1), max(pmx, p90 + 2), 255.0]
    ys = [0.0, lk["p50"] * 0.30, lk["p50"], lk["p75"], lk["p90"], lk["max"], lk["max"]]
    Ln = np.interp(L, xs, ys)
    tr, tg, tb = lk.get("tint_rgb", [-3, 1, 3])
    return np.stack([Ln + tr, Ln + tg, Ln + tb], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix"); ap.add_argument("--day", type=int, default=6); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--no-ui", action="store_true")
    a = ap.parse_args()
    from gymform import ctx
    ctx.configure(json.load(open(os.path.join(ROOT, "configs", f"day{a.day:02d}.json"))))
    from gymform import bg, layout
    from gymform.ctx import SPEC, b4, LAST_MASKS
    meta = json.load(open(a.prefix + "_meta.json"))
    S = json.load(open(os.path.join(ROOT, "studio3d", "set_v1.json")))
    bea, idp, bon = load(a.prefix + "_beauty.png"), load(a.prefix + "_id.png"), load(a.prefix + "_bones.png")
    H, W = 1920, 1080
    if bea.shape[0] != H:
        rs = lambda x, it=cv2.INTER_LANCZOS4: cv2.resize(x, (W, H), interpolation=it)
        bea, bon, idp = rs(bea), rs(bon), rs(idp, cv2.INTER_NEAREST)
    M = id_masks(idp)
    rgb, al = bea[..., :3], bea[..., 3:] / 255.0
    # 1. Day 4 skin lock + shorts tone (own render pixels, graded - nothing painted)
    rgb = np.where(M["skin"][..., None], skin_lock(rgb, M["skin"], SPEC), rgb)
    lo, hi = SPEC["mannequin"]["shorts_lum_range"]
    Ls = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    sh_l = np.clip(lo + (Ls - np.percentile(Ls[M["shorts"]], 5)) * 0.5, lo, hi)
    rgb = np.where(M["shorts"][..., None], np.stack([sh_l] * 3, -1), rgb)
    # 2. X-Ray bones: target bones only (render already holds only them), shown on skin pixels only
    x = S["xray"]; tint = np.array(x["bone_tint"]["wrong" if meta["form"] == "wrong" else "right"], np.float32)
    ra = bon[..., 3] / 255.0                                   # 3D rim/facing term from the glass bone shader
    ba = np.clip(ra * 1.15, 0, 1) ** 0.85 * cv2.GaussianBlur(M["skin"].astype(np.float32), (0, 0), 1.0)
    lum = np.clip(0.42 + 0.62 * ra, 0, 1)                    # luminous rim, softer core: L50 >= 140 (QC (u))
    w2 = (x["core_white"] * lum ** 2)[..., None]
    col = tint[None, None] * 255 * lum[..., None] * (1 - w2) + 255 * w2
    col = np.clip(col, 0, 255)
    aa = (ba * x["opacity"])[..., None]
    rgb = rgb * (1 - aa) + col * aa
    glow = cv2.GaussianBlur(ba, (0, 0), 7.0)[..., None] * M["skin"][..., None]
    rgb = np.clip(rgb + glow * tint * 255 * 0.22, 0, 255)
    # 3. over the Day 4 backdrop (shadow catcher alpha carries the floor shadows)
    back = bg.studio_bg()
    out = rgb * al + back * (1 - al)
    out = np.clip(out, 0, 255).astype(np.uint8)
    kp = {k: tuple(v) for k, v in meta["kp"].items()}
    if not a.no_ui:
        ang = meta["angles_3d"]["R"]["arm_torso"]
        deg = f"{int(round(ang))}\u00b0"
        c = ctx.CFG["callouts"][str(a.step)]
        l2 = f"{int(round(c['l2'][1]))}% {c['l2'][0]}" if not isinstance(c["l2"], str) else c["l2"]
        shm = M["shorts"].astype(np.float32)
        LAST_MASKS["eq_front"] = cv2.GaussianBlur((M["bench"] | M["dumbbell"]).astype(np.float32), (0, 0), 1.0)
        SPEC["ui"]["callout_ty_min"] = int(max(kp["elb_l"][1], kp["elb_r"][1], kp["wr_l"][1], kp["wr_r"][1]) + 150)
        fr = layout.draw_callout_day4(out, shm, a.step, kp, meta["form"] == "wrong", deg, c["l1"], l2)
        icons = [b4._make_icon_pair(out, kp) for _ in range(4)]
        fg = (M["skin"] | M["shorts"]).astype(np.float32)
        out = layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(fg), icons, a.step - 1, meta["form"] == "wrong",
                                                    ctx.CFG["exercise_name"])
    Image.fromarray(out).save(a.out, quality=94)
    L = out.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    sk = cv2.erode(M["skin"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sk &= ~(cv2.dilate((ba > 0.05).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    print(json.dumps({"out": a.out, "skin_p50_p75_p90": [round(float(np.percentile(L[sk], q))) for q in (50, 75, 90)],
                      "bone_L50": round(float(np.percentile(L[ba > 0.5], 50))) if (ba > 0.5).any() else None,
                      "set_hash": meta["set_hash"], "angles_3d": meta["angles_3d"]}))


if __name__ == "__main__":
    main()
