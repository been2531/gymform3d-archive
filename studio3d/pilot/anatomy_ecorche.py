"""Load Charleeh/ecorche (CC BY-SA 4.0) upper-body muscles, fit to MakeHuman rig, UV for fibres."""
from __future__ import annotations

import os
import re

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import addon_utils

ROOT = "/agent/repos/gymform3d-archive"
ECORCHE = os.path.join(ROOT, "assets3d", "ecorche.glb")

# Upper-body muscles for lat-pulldown visibility (exclude leg/foot/hand bones by name)
INCLUDE_RE = re.compile(
    r"(latissimus|biceps brachii|deltoid|trapezius|pectoralis|serratus|infraspinatus|teres|"
    r"rhomboid|erector|oblique|rectus abdominis|brachialis|triceps brachii|coracobrachialis|"
    r"subscapularis|levator scapulae|splenius)",
    re.I,
)
EXCLUDE_RE = re.compile(r"(femoris|foot|toe|tibialis|gastrocnemius|soleus|peroneus|plantar|hip|gluteus|iliac)", re.I)

BONE_FOR = [
    (re.compile(r"latissimus", re.I), "spine04"),
    (re.compile(r"biceps brachii", re.I), "upperarm02"),
    (re.compile(r"deltoid", re.I), "upperarm01"),
    (re.compile(r"trapezius", re.I), "spine05"),
    (re.compile(r"pectoralis", re.I), "spine02"),
    (re.compile(r"brachialis|triceps brachii", re.I), "lowerarm01"),
]


def _enable_gltf():
    addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)


def _uv_fibre_axis(ob):
    me = ob.data
    if not me.loops:
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    coords = np.array([v.co for v in bm.verts])
    if len(coords) < 4:
        bm.free()
        return
    c = coords.mean(0)
    x = coords - c
    cov = x.T @ x / max(len(coords) - 1, 1)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj = (coords - c) @ axis
    pmin, pmax = proj.min(), proj.max()
    span = max(pmax - pmin, 1e-6)
    uvs = []
    layer = bm.loops.layers.uv.verify()
    for v in bm.verts:
        u = (float((v.co - Vector(c)) @ Vector(axis)) - pmin) / span
        uvs.append((u, 0.5))
    for face in bm.faces:
        for loop in face.loops:
            loop[layer].uv = uvs[loop.vert.index]
    bm.to_mesh(me)
    bm.free()
    me.update()


def _pick_bone(name, arm):
    side = ".L" if " left " in f" {name.lower()} " or name.lower().startswith("left ") else (
        ".R" if " right " in f" {name.lower()} " or name.lower().startswith("right ") else ""
    )
    for pat, bn in BONE_FOR:
        if pat.search(name):
            bone = bn + side if side and bn + side in arm.data.bones else bn
            if bone in arm.data.bones:
                return bone
    return "spine03"


def import_upper_muscles(arm, posed=False):
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
        raise RuntimeError("no ecorche muscles imported")
    bpy.context.view_layer.update()
    coords = []
    for o in kept:
        for v in o.data.vertices:
            coords.append(o.matrix_world @ v.co)
    coords = np.array(coords)
    chest = arm.matrix_world @ ((arm.pose.bones["upperarm01.L"].head + arm.pose.bones["upperarm01.R"].head) / 2)
    delta = Vector(chest) - Vector(coords.mean(0))
    for o in kept:
        o.location += delta
    bpy.context.view_layer.update()
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


def glow_targets(form):
    if form == "wrong":
        return re.compile(r"(biceps brachii|deltoid|trapezius)", re.I)
    return re.compile(r"latissimus", re.I)
