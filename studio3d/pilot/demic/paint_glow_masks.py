"""Project écorché muscle positions onto MH body verts → vertex color GLOW (RGBA weights)."""
from __future__ import annotations

import json
import os
import re

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ROOT = "/agent/repos/gymform3d-archive"
ECORCHE = os.path.join(ROOT, "assets3d", "ecorche.glb")
CACHE = os.path.join(ROOT, "assets3d", "demic_glow_masks.npz")

LAT_RE = re.compile(r"latissimus|teres major", re.I)
BI_RE = re.compile(r"biceps brachii|brachialis", re.I)
DELT_RE = re.compile(r"deltoid", re.I)
TRAP_RE = re.compile(r"trapezius", re.I)


def _enable_gltf():
    import addon_utils

    addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)


def _muscle_centroids():
    _enable_gltf()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=ECORCHE)
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    groups = {"lat": [], "bi": [], "delt": [], "trap": []}
    for o in imported:
        c = np.mean([o.matrix_world @ v.co for v in o.data.vertices], axis=0)
        n = o.name.lower()
        if LAT_RE.search(n):
            groups["lat"].append(c)
        elif BI_RE.search(n):
            groups["bi"].append(c)
        elif DELT_RE.search(n):
            groups["delt"].append(c)
        elif TRAP_RE.search(n):
            groups["trap"].append(c)
        bpy.data.objects.remove(o, do_unlink=True)
    return {k: np.array(v) if v else np.zeros((0, 3)) for k, v in groups.items()}


def _rig_weights(body_ob, prefixes):
    n = len(body_ob.data.vertices)
    out = np.zeros(n, dtype=np.float32)
    gi = {g.index: g.name for g in body_ob.vertex_groups}
    for i, v in enumerate(body_ob.data.vertices):
        for g in v.groups:
            nm = gi.get(g.group, "")
            if any(nm.startswith(p) or p in nm for p in prefixes):
                out[i] = max(out[i], g.weight)
    return out


def _nearest_weight(p, centers, sigma=0.09):
    if len(centers) == 0:
        return 0.0
    d = np.linalg.norm(centers - p, axis=1)
    return float(np.exp(-(d.min() ** 2) / (2 * sigma * sigma)))


def apply_vertex_glow(body_ob, arm, centroids=None):
    regen = os.environ.get("DEMIC_REGEN_MASKS") == "1"
    if centroids is None and os.path.isfile(CACHE) and not regen:
        data = np.load(CACHE)
        lat, bi, delt, trap = data["lat"], data["bi"], data["delt"], data["trap"]
    else:
        if centroids is None:
            centroids = _muscle_centroids()
        me = body_ob.data
        n = len(me.vertices)
        lat = np.zeros(n, dtype=np.float32)
        bi = np.zeros(n, dtype=np.float32)
        delt = np.zeros(n, dtype=np.float32)
        trap = np.zeros(n, dtype=np.float32)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body_ob.evaluated_get(dg)
        mw = ev.matrix_world
        chest_y = float(np.mean(centroids["lat"][:, 1])) if len(centroids["lat"]) else 0.0
        trap_z = float(np.mean(centroids["trap"][:, 2])) if len(centroids["trap"]) else 1.35
        for i, v in enumerate(me.vertices):
            p = np.array(mw @ v.co)
            n = np.array(v.normal)
            lat_w = _nearest_weight(p, centroids["lat"], sigma=0.11)
            bi_w = _nearest_weight(p, centroids["bi"], sigma=0.07)
            delt_w = _nearest_weight(p, centroids["delt"], sigma=0.07)
            trap_w = _nearest_weight(p, centroids["trap"], sigma=0.08)
            # Keep lats on back shell; biceps/delts on front/lateral; traps on upper back only.
            back = 0.25 + 0.75 * (1.0 if p[1] < chest_y - 0.01 else 0.0)
            front = 0.25 + 0.75 * (1.0 if p[1] > chest_y - 0.08 else 0.0)
            lat[i] = lat_w * back * (0.25 + 0.75 * max(0.0, -n[1]))
            bi[i] = bi_w * (0.35 + 0.65 * front) * (0.4 + 0.6 * max(0.0, n[1]))
            delt[i] = delt_w * (0.35 + 0.65 * front) * (0.35 + 0.65 * max(0.0, n[1]))
            trap[i] = trap_w * back * (1.0 if p[2] > trap_z - 0.15 else 0.12)
        bi_rig = _rig_weights(body_ob, ("upperarm02", "lowerarm01", "lowerarm02"))
        delt_rig = _rig_weights(body_ob, ("shoulder01", "upperarm01", "clavicle"))
        trap_rig = _rig_weights(body_ob, ("spine04", "spine05"))
        lat_rig = _rig_weights(body_ob, ("spine03", "spine04", "spine05"))
        lat = np.maximum(lat, lat_rig * 0.55)
        bi = np.maximum(bi, bi_rig * 0.9)
        delt = np.maximum(delt, delt_rig * 0.85)
        trap = np.maximum(trap, trap_rig * 0.75)
        np.savez(CACHE, lat=lat, bi=bi, delt=delt, trap=trap)
    layer = body_ob.data.color_attributes.new(name="GLOW", type="BYTE_COLOR", domain="POINT")
    me = body_ob.data
    for i, v in enumerate(me.vertices):
        if body_ob.name == "shorts":
            layer.data[i].color = (0, 0, 0, 1)
            continue
        layer.data[i].color = (
            min(1.0, lat[i]),
            min(1.0, bi[i]),
            min(1.0, delt[i]),
            min(1.0, trap[i]),
        )
    return lat, bi, delt, trap


def mask_for_form(form, lat, bi, delt, trap):
    if form == "right":
        return np.clip(lat * 1.05 + trap * 0.08, 0, 1)
    return np.clip(bi * 0.95 + delt * 0.9 + trap * 0.45, 0, 1)


def set_active_glow_channel(body_ob, form, lat, bi, delt, trap):
    """Write form-specific weight into vertex-color R (shader reads R only)."""
    w = mask_for_form(form, lat, bi, delt, trap)
    layer = body_ob.data.color_attributes.get("GLOW")
    if layer is None:
        layer = body_ob.data.color_attributes.new(name="GLOW", type="BYTE_COLOR", domain="POINT")
    for i in range(len(body_ob.data.vertices)):
        layer.data[i].color = (float(w[i]), float(bi[i]), float(delt[i]), float(trap[i]))
    return w
