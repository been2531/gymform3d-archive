"""Hand-authored muscle regions via MH vertex groups + facing tests (no écorché projection)."""
from __future__ import annotations

import os

import numpy as np

ROOT = "/agent/repos/gymform3d-archive"
CACHE = os.path.join(ROOT, "assets3d", "demic_discrete_glow.npz")

# Dominant-group prefixes per muscle (discrete patches).
LAT_GROUPS = ("spine03", "spine04", "spine05")
BI_GROUPS = ("upperarm02", "lowerarm01", "lowerarm02")
DELT_GROUPS = ("shoulder01", "clavicle")
TRAP_GROUPS = ("spine05", "neck01", "neck02")


def _dominant_weights(body_ob):
    n = len(body_ob.data.vertices)
    gi = {g.index: g.name for g in body_ob.vertex_groups}
    dom = [""] * n
    dom_w = np.zeros(n, dtype=np.float32)
    for i, v in enumerate(body_ob.data.vertices):
        if not v.groups:
            continue
        best = max(v.groups, key=lambda x: x.weight)
        dom[i] = gi.get(best.group, "")
        dom_w[i] = best.weight
    return dom, dom_w


def _match(name, prefixes):
    return any(name.startswith(p) for p in prefixes)


def _facing(body_ob, i, mode):
    n = body_ob.data.vertices[i].normal
    if mode == "back":
        return n.y > 0.35
    if mode == "front":
        return n.y < -0.2
    if mode == "lateral":
        return abs(n.x) > 0.45
    return True


def bake_discrete_masks(body_ob):
    dom, dom_w = _dominant_weights(body_ob)
    n = len(dom)
    lat = np.zeros(n, dtype=np.float32)
    bi = np.zeros(n, dtype=np.float32)
    delt = np.zeros(n, dtype=np.float32)
    trap = np.zeros(n, dtype=np.float32)
    import bpy
    from mathutils import Vector

    dg = bpy.context.evaluated_depsgraph_get()
    ev = body_ob.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world.to_3x3()
    for i in range(n):
        if i >= len(me.vertices):
            continue
        nw = (mw @ me.vertices[i].normal).normalized()
        ny, nz, nx = float(nw.y), float(nw.z), float(nw.x)
        nm = dom[i]
        if dom_w[i] < 0.45:
            continue
        # Lats: V-band on back, armpit to lower back; hidden from front.
        if _match(nm, LAT_GROUPS) and ny > 0.25 and nz > 0.05:
            lat[i] = 1.0
        elif _match(nm, LAT_GROUPS) and ny > 0.15 and abs(nx) > 0.35:
            lat[i] = 0.85
        # Biceps: front upper arm only.
        if _match(nm, BI_GROUPS) and ny < -0.05 and nz > 0.0:
            bi[i] = 1.0
        # Front delt cap.
        if _match(nm, DELT_GROUPS) and ny < -0.15 and nz > 0.25:
            delt[i] = 1.0
        # Upper trap: neck–shoulder slope, back/side only.
        if _match(nm, TRAP_GROUPS) and nz > 0.55 and ny > -0.05:
            trap[i] = 1.0
    ev.to_mesh_clear()
    np.savez(CACHE, lat=lat, bi=bi, delt=delt, trap=trap)
    return lat, bi, delt, trap


def load_or_bake(body_ob):
    if os.path.isfile(CACHE):
        d = np.load(CACHE)
        if len(d["lat"]) == len(body_ob.data.vertices):
            return d["lat"], d["bi"], d["delt"], d["trap"]
    return bake_discrete_masks(body_ob)


def mask_for_form(form, lat, bi, delt, trap):
    if form == "right":
        return np.clip(lat, 0, 1)
    return np.clip(np.maximum.reduce([bi, delt * 0.95, trap * 0.9]), 0, 1)


def apply_to_mesh(body_ob, form):
    lat, bi, delt, trap = load_or_bake(body_ob)
    w = mask_for_form(form, lat, bi, delt, trap)
    layer = body_ob.data.color_attributes.get("GLOW")
    if layer is None:
        layer = body_ob.data.color_attributes.new(name="GLOW", type="BYTE_COLOR", domain="POINT")
    for i in range(len(body_ob.data.vertices)):
        layer.data[i].color = (float(w[i]), float(bi[i]), float(delt[i]), float(trap[i]))
    return w, lat, bi, delt, trap
