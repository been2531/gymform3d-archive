"""Generic equipment module: zones (dumbbell/barbell/cable/bench/machine), plate-difference footprint, matte metal shading."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30


def equipment_zones(kp: dict):
    """Yields (x0, y0, x1, y1, zone_mask_uint8) per configured equipment zone."""
    for eq in CFG["equipment"]:
        z = np.zeros((H, W), np.uint8)
        if "anchor" in eq:
            ax, ay = kp[eq["anchor"]]
            cv2.circle(z, (int(ax), int(ay)), int(eq["radius"]), 1, -1)
        elif "segment" in eq:
            pa, pb = [kp[p] if isinstance(p, str) else tuple(p) for p in eq["segment"]]
            e = eq.get("ext", 0.0)
            pa2 = (pa[0] - (pb[0] - pa[0]) * e, pa[1] - (pb[1] - pa[1]) * e)
            pb2 = (pb[0] + (pb[0] - pa[0]) * e, pb[1] + (pb[1] - pa[1]) * e)
            cv2.line(z, (int(pa2[0]), int(pa2[1])), (int(pb2[0]), int(pb2[1])), 1, int(2 * eq["pad"]))
        elif "roi" in eq:
            x0, y0, x1, y1 = eq["roi"]
            z[y0:y1, x0:x1] = 1
        ys, xs = np.where(z > 0)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        yield x0, y0, x1, y1, z[y0:y1, x0:x1]


def equipment_soft_zone(kp: dict) -> np.ndarray:
    zz = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1, z in equipment_zones(kp):
        zz[y0:y1, x0:x1] = np.maximum(zz[y0:y1, x0:x1], z.astype(np.float32))
    # full weight ~25px inside the zone edge, fading to 0 at the edge
    return np.clip(cv2.distanceTransform((zz > 0).astype(np.uint8), cv2.DIST_L2, 3) / 25.0, 0.0, 1.0)


def footprint(rgb_1080, kp, F):
    """Generic equipment footprint (any type configured in CFG['equipment']) from the step's temporal backdrop
    plate: solid where the evidence is strong, proportional alpha for motion-blur smear / soft halo (no bloat)."""
    lum, act_dom = F["lum"], F["act_dom"]
    foot = np.zeros((H, W), dtype=np.float32)
    plate = CUR_PLATE.get("rgb")
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for x0, y0, x1, y1, zone in equipment_zones(kp):
        if plate is None:
            break
        d = np.abs(rgb_1080[y0:y1, x0:x1].astype(np.float32) - plate[y0:y1, x0:x1]).max(axis=-1)
        # plate must be backdrop there (dark): where the plate holds the body (arm rest positions) the difference
        # is a body ghost, not equipment
        plate_lum = plate[y0:y1, x0:x1].max(axis=-1)
        # the source backdrop also carries a soft, smooth light halo around the moving arms (+10..25 vs plate):
        # equipment = differs from the plate AND (strongly, or with native 3D surface texture)
        lz = lum[y0:y1, x0:x1]
        tex = cv2.GaussianBlur(np.abs(lz - cv2.GaussianBlur(lz, (0, 0), 2.0)), (0, 0), 1.2)
        # (matte outer plates can sit only ~8-10 above the plate in max channel but keep crisp 3D texture)
        # black metal is DARKER than the backdrop plate, while the light halo is brighter: darker-than-plate
        # pixels are equipment even where the plate face is smooth
        pl_l = 0.299 * plate[y0:y1, x0:x1, 0] + 0.587 * plate[y0:y1, x0:x1, 1] + 0.114 * plate[y0:y1, x0:x1, 2]
        # (activation light tints the metal in one channel, so count channels: >=2 of R,G,B darker than plate)
        dec = plate[y0:y1, x0:x1] - rgb_1080[y0:y1, x0:x1].astype(np.float32)
        # the source backdrop is a cool blue-grey (B-R ~ +10) while metal/rubber plates are neutral (B-R ~ +2):
        # the chroma drop vs the plate outlines the full plate discs even where their luminance equals the backdrop
        cz = rgb_1080[y0:y1, x0:x1].astype(np.float32)
        # (min of the B-R and B-G drops: a red glow only lowers B-R, a green glow only lowers B-G; neutral metal
        # lowers both)
        pz = plate[y0:y1, x0:x1]
        drop_br = (pz[:, :, 2] - pz[:, :, 0]) - (cz[:, :, 2] - cz[:, :, 0])
        drop_bg = (pz[:, :, 2] - pz[:, :, 1]) - (cz[:, :, 2] - cz[:, :, 1])
        chroma_drop = cv2.GaussianBlur(np.minimum(drop_br, drop_bg), (0, 0), 1.5)
        darker = ((pl_l - lz) > 7.0) | ((dec > 5.5).sum(axis=-1) >= 2)
        # textured differences are native 3D surface (metal or glow-lit forearm/hand), never the smooth glow
        # halo, so they carry no tint limit; the smooth neutral/chroma paths keep a strict tint limit
        m = ((((d > 34.0) & (tex > 2.0)) | ((d > 11.0) & (tex > 4.0))) & (zone > 0) & (plate_lum < 95.0)) | (
              (((d > 11.0) & (act_dom[y0:y1, x0:x1] < 7.0) & (plate_lum < 52.0))
               | ((chroma_drop > 4.5) & (d > 3.0) & (act_dom[y0:y1, x0:x1] < 12.0) & (plate_lum < 52.0))) & (zone > 0))
        # darker-than-backdrop pixels are never glow (glow only brightens): no tint limit, so glow-lit dark
        # metal and the glow-lit dark underside of the hand/forearm next to the equipment stay solid
        m = (m | ((d > 6.0) & darker & (plate_lum < 52.0) & (zone > 0))).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for ci in range(1, n_c):
            if st_c[ci, cv2.CC_STAT_AREA] >= 250:
                keep[lab_c == ci] = 1
        ff = np.pad(keep, 1).copy()
        cv2.floodFill(ff, None, (0, 0), 2)
        filled = (ff[1:-1, 1:-1] != 2)
        holes = (filled & (keep == 0)).astype(np.uint8)
        n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(holes)
        for hi in range(1, n_h):  # only small enclosed holes (plate centres / handle gaps), never body windows
            if st_h[hi, cv2.CC_STAT_AREA] > 2500:
                filled[lab_h == hi] = False
        # rigid-object completion: the dark metal parts (plates) of one piece of equipment are convex in
        # silhouette, so their joint convex hull restores plate rims that fade into the backdrop. Thin glow arcs
        # are removed by an opening first; the hull is only accepted when it stays compact (not body-sized).
        metal = (filled & (lum[y0:y1, x0:x1] < 100.0)).astype(np.uint8)
        metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        n_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(metal)
        big = [mi for mi in range(1, n_m) if st_m[mi, cv2.CC_STAT_AREA] >= 1200]
        if big and CFG.get("equipment_hull", False):
            pts = np.concatenate([np.column_stack(np.where(lab_m == mi))[:, ::-1] for mi in big]).astype(np.int32)
            hull = cv2.convexHull(pts)
            hm = np.zeros_like(metal)
            cv2.fillConvexPoly(hm, hull, 1)
            area_big = sum(st_m[mi, cv2.CC_STAT_AREA] for mi in big)
            if hm.sum() <= 2.2 * area_big and hm.sum() < 60000:
                filled = filled | (hm > 0)
        # plate completion: dark metal rings (C-shaped where a matte face meets the backdrop) are closed with a
        # plate-scale kernel and their enclosed faces filled; gaps wider than the kernel (handle span between
        # left/right plate stacks, arm-to-plate) stay open, so no backdrop slab is added
        ring = (filled & (lum[y0:y1, x0:x1] - 0.85 * act_dom[y0:y1, x0:x1] < 100.0)).astype(np.uint8)
        ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        rf = np.pad(ring, 1).copy()
        cv2.floodFill(rf, None, (0, 0), 2)
        ring_h = ((rf[1:-1, 1:-1] != 2) & (ring == 0)).astype(np.uint8)
        n_r, lab_r, st_r, _ = cv2.connectedComponentsWithStats(ring_h)
        ring_f = ring > 0
        for ri in range(1, n_r):
            if st_r[ri, cv2.CC_STAT_AREA] <= 9000:
                ring_f |= lab_r == ri
        # per-plate-stack convex completion: break thin handle links, then each plate stack (disc-like blob)
        # is replaced by its convex hull when it is compact and plate-sized (not a bridge across the hand)
        rs = cv2.morphologyEx(ring_f.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(rs)
        for si in range(1, n_s):
            a_s = st_s[si, cv2.CC_STAT_AREA]
            if a_s < 600:
                continue
            pts = np.column_stack(np.where(lab_s == si))[:, ::-1].astype(np.int32)
            hull = cv2.convexHull(pts)
            hm = np.zeros_like(rs)
            cv2.fillConvexPoly(hm, hull, 1)
            ha = int(hm.sum())
            if len(pts) >= 5:
                (_, _), (ea, eb), _ = cv2.fitEllipse(pts)
                aspect = max(ea, eb) / max(1.0, min(ea, eb))
            else:
                aspect = 9.0
            if ha <= 22000 and a_s >= 0.33 * ha and aspect <= 3.2:
                ring_f |= hm > 0
        # only keep completion additions that are dark (metal-like), never bright body/glow halo
        add = ring_f & ~filled & ((lum[y0:y1, x0:x1] - 0.85 * act_dom[y0:y1, x0:x1]) < 70.0)
        # completion pixels must also be neutral (metal), not the cool-blue backdrop beside the arm
        # (inside activation glow the chroma test is unreliable: there the convex plate completion is trusted)
        filled = filled | (add & ((chroma_drop > 2.5) | (act_dom[y0:y1, x0:x1] >= 12.0)))
        # difference-matte alpha (root-cause fix for bloated equipment): motion-blur smear and soft halos around a
        # moving object differ only weakly from the backdrop plate. Treating them as solid made a fast-moving
        # dumbbell render as an opaque, oversized blob. Only confident evidence (strong difference / native
        # texture / clearly darker than the backdrop, plus the faces it encloses) is solid; weak-difference
        # pixels keep a proportional alpha, so the composite reproduces the source silhouette and size.
        strong = ((d > 18.0) | ((tex > 4.0) & (d > 8.0)) | ((dec > 10.0).sum(axis=-1) >= 2)) & filled
        strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sf = np.pad(strong, 1).copy()
        cv2.floodFill(sf, None, (0, 0), 2)
        s_h = ((sf[1:-1, 1:-1] != 2) & (strong == 0)).astype(np.uint8)
        n_sh, lab_sh, st_sh, _ = cv2.connectedComponentsWithStats(s_h)
        strong_f = strong > 0
        for hi in range(1, n_sh):
            if st_sh[hi, cv2.CC_STAT_AREA] <= 9000:
                strong_f |= lab_sh == hi
        # neutral dark METAL that measurably differs from the backdrop plate (max-channel >= 6, or a neutral
        # chroma drop) is the object itself, not smear: always solid (matte plate faces have a small difference)
        lz_eq = lz - 0.85 * act_dom[y0:y1, x0:x1]
        metal_solid = filled & (lz_eq < 75.0) & ((d >= 6.0) | ((chroma_drop > 4.5) & (d > 3.0)))
        strong_f |= metal_solid
        # matte plate faces can match the backdrop locally (d ~ 4): close notches/holes inside the solid object,
        # dark pixels only (a plate-scale closing never bridges to bright body or across the open backdrop)
        closed = cv2.morphologyEx(strong_f.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        strong_f |= closed & (lz_eq < 75.0)
        # weak (proportional) alpha only as a thin soft edge of a solid object: weak-difference pixels away from
        # any solid equipment are backdrop-plate ghosts (arm positions held in the median plate), never equipment
        near_obj = cv2.dilate(strong_f.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        alpha = np.where(strong_f, 1.0, np.clip((d - 5.0) / 15.0, 0.0, 1.0) * near_obj).astype(np.float32)
        f = ((filled & (zone > 0)) * alpha).astype(np.float32)
        foot[y0:y1, x0:x1] = np.maximum(foot[y0:y1, x0:x1], f)
    return foot


def metal_shade(grey, F, eq_solid):
    """Matte gunmetal-black equipment that keeps the source's own tonal range (rims/highlights), so the composite
    never reads heavier than the source. Hand/forearm skin inside the footprint keeps the silver skin curve."""
    lum, act_dom, min_c = F["lum"], F["act_dom"], F["min_c"]
    lum_eq = np.maximum(lum - 0.85 * act_dom, min_c)
    skin_t = np.clip((lum_eq - 88.0) / 30.0, 0.0, 1.0)       # bright = hand skin, not metal
    metal = np.clip(lum_eq * 1.0 + 6.0, 12.0, 118.0)          # source tone kept (no flattening to a dark blob)
    eq_w = cv2.GaussianBlur(eq_solid, (0, 0), 1.5) * (1.0 - skin_t)
    return grey * (1.0 - eq_w) + metal * eq_w
