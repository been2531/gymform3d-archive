"""Écorché muscles: per-region fit, bone parenting (BodyParts3D-derived, CC BY-SA — see ATTRIBUTION)."""
from __future__ import annotations

import re

import bpy
import numpy as np
from mathutils import Vector

from anatomy_ecorche import ECORCHE, EXCLUDE_RE, INCLUDE_RE, _enable_gltf, _pick_bone, _uv_fibre_axis


def _region_for(name):
    n = name.lower()
    if re.search(r"latissimus|trapezius|erector|oblique|rectus abdominis|pectoralis|serratus|rhomboid|splenius", n):
        return "torso"
    if re.search(r"quadriceps|vastus|rectus femoris|adductor|gracilis", n):
        return "leg"
    return "arm"


def _fit_empties(arm, kept):
    fit_root = bpy.data.objects.new("ecorche_fit", None)
    bpy.context.collection.objects.link(fit_root)
    groups = {"torso": [], "arm": [], "leg": []}
    for o in kept:
        groups[_region_for(o.name)].append(o)
        mw = o.matrix_world.copy()
        o.parent = fit_root
        o.matrix_parent_inverse = fit_root.matrix_world.inverted()
        o.matrix_world = mw
    bpy.context.view_layer.update()

    sh_l = arm.matrix_world @ arm.pose.bones["upperarm01.L"].head
    sh_r = arm.matrix_world @ arm.pose.bones["upperarm01.R"].head
    mh_span = (sh_l - sh_r).length
    chest = (sh_l + sh_r) / 2

    def span_x(objs, zc, band=0.25):
        xs = []
        for o in objs:
            for v in o.data.vertices:
                w = o.matrix_world @ v.co
                if abs(w.z - zc) < band:
                    xs.append(w.x)
        if not xs:
            for o in objs:
                for v in o.data.vertices:
                    xs.append((o.matrix_world @ v.co).x)
        return max(xs) - min(xs) if xs else 1.0

    zc = float(chest.z)
    scales = {}
    if groups["torso"]:
        scales["torso"] = mh_span / max(span_x(groups["torso"], zc), 1e-6) * 0.97
    if groups["arm"]:
        scales["arm"] = mh_span / max(span_x(groups["arm"], zc), 1e-6) * 0.94
    if groups["leg"]:
        hip_z = float((arm.matrix_world @ arm.pose.bones["upperleg01.L"].head).z)
        scales["leg"] = mh_span / max(span_x(groups["leg"], hip_z, 0.35), 1e-6) * 0.93

    for reg, objs in groups.items():
        if not objs or reg not in scales:
            continue
        sub = bpy.data.objects.new(f"fit_{reg}", None)
        bpy.context.collection.objects.link(sub)
        sub.parent = fit_root
        for o in objs:
            mw = o.matrix_world.copy()
            o.parent = sub
            o.matrix_parent_inverse = sub.matrix_world.inverted()
            o.matrix_world = mw
        sub.scale = (scales[reg], scales[reg], scales[reg])
    bpy.context.view_layer.update()

    coords = []
    for o in kept:
        for v in o.data.vertices:
            coords.append(o.matrix_world @ v.co)
    coords = np.array(coords)
    fit_root.location += Vector(chest) - Vector(coords.mean(0))
    bpy.context.view_layer.update()
    return kept


def import_and_bind_muscles(arm, _body):
    _enable_gltf()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=ECORCHE)
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    kept = []
    for o in imported:
        if EXCLUDE_RE.search(o.name):
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        if not INCLUDE_RE.search(o.name):
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        kept.append(o)
    if not kept:
        raise RuntimeError("no muscles imported from ecorche")
    bpy.context.view_layer.update()
    _fit_empties(arm, kept)
    for o in kept:
        _uv_fibre_axis(o)
        mw = o.matrix_world.copy()
        bn = _pick_bone(o.name, arm)
        o.parent = arm
        o.parent_type = "BONE"
        o.parent_bone = bn
        o.matrix_world = mw
    bpy.context.view_layer.update()
    return kept


def muscle_glow_pattern(form):
    if form == "wrong":
        return re.compile(r"(biceps brachii|deltoid|trapezius)", re.I)
    return re.compile(r"(latissimus|teres major|teres minor|infraspinatus)", re.I)
