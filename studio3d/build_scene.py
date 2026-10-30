"""Locked 3D set (t561u/t562u): build the ONE fixed scene (mannequin + rig + anatomy + bench + dumbbells + lights +
camera presets) from studio3d/set_v1.json, pose it from an exercise config, render passes for one frame.
Run with the bpy venv:  /workspace/.bpyenv/bin/python studio3d/build_scene.py --exercise incline_dumbbell_press
   --form wrong --phase 0 --camera front --out _tmp_3d/proof   [--scale 0.5 --samples 16]
Outputs: <out>_beauty.png (RGBA, film transparent + shadow catcher), <out>_id.png (flat ID colours),
<out>_bones.png (RGBA X-Ray bone layer, target bones only), <out>_meta.json (projected joints, set hash)."""
import argparse, hashlib, json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ID_COL = {"skin": (1, 0, 0), "shorts": (0, 1, 0), "bench": (0, 0, 1), "dumbbell": (1, 1, 0), "floor": (0, 0, 0)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_hash(S):
    """hash of the locked set: set json (minus doc) + every asset file"""
    h = hashlib.sha256(json.dumps({k: v for k, v in S.items() if not k.startswith("_")}, sort_keys=True).encode())
    for a in S["assets"].values():
        h.update(sha(os.path.join(S["asset_dir"], a["file"])).encode())
    for k in ("build_scene.py", "composite.py"):
        h.update(sha(os.path.join(HERE, k)).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- mannequin
def load_mh(S):
    A, s = S["asset_dir"], S["mannequin"]["scale_from_mh"]
    V, F, grp = [], [], None
    for ln in open(os.path.join(A, "base.obj")):
        if ln.startswith("v "):
            x, y, z = map(float, ln.split()[1:4])
            V.append((x * s, -z * s, y * s))
        elif ln.startswith("g "):
            grp = ln.split()[1]
        elif ln.startswith("f ") and grp == "body":
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
    V = np.array(V)
    for tf, w in S["mannequin"].get("morph", {}).get("targets", {}).items():      # CC0 MakeHuman macro targets
        for ln in open(os.path.join(A, tf)):
            if ln[:1].isdigit():
                i, dx, dy, dz = ln.split()[:4]
                V[int(i)] += w * s * np.array((float(dx), -float(dz), float(dy)))
    sk = json.load(open(os.path.join(A, "default.mhskel")))
    J = {n: V[idx].mean(0) for n, idx in sk["joints"].items()}
    return V, F, sk, J


def make_mannequin(S):
    V, F, sk, J = load_mh(S)
    me = bpy.data.meshes.new("mannequin")
    me.from_pydata([tuple(v) for v in V], [], F)
    me.update()
    ob = bpy.data.objects.new("mannequin", me)
    bpy.context.collection.objects.link(ob)
    W = json.load(open(os.path.join(S["asset_dir"], "default_weights.mhw")))["weights"]
    for bn, lst in W.items():
        vg = ob.vertex_groups.new(name=bn)
        for i, w in lst:
            vg.add([i], w, "REPLACE")
    # armature
    ad = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", ad)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for bn, b in sk["bones"].items():
        eb = ad.edit_bones.new(bn)
        eb.head, eb.tail = Vector(J[b["head"]]), Vector(J[b["tail"]])
        if (eb.tail - eb.head).length < 1e-4:
            eb.tail = eb.head + Vector((0, 0, 0.01))
    for bn, b in sk["bones"].items():
        if b["parent"]:
            ad.edit_bones[bn].parent = ad.edit_bones[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    # drop helper geometry (loose verts) - groups follow the kept verts
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    fl = S["mannequin"].get("faceless")
    if fl:
        # faceless mannequin head: drop every face whose verts are dominated by head/face rig groups
        bm.verts.ensure_lookup_table()
        dl = bm.verts.layers.deform.verify()
        gname = {g.index: g.name for g in ob.vertex_groups}
        def headv(v):
            d = v[dl]
            if not d:
                return False
            g, w = max(d.items(), key=lambda t: t[1])
            return any(gname[g].startswith(p) for p in fl["bones_prefix"])
        hv = {v.index for v in bm.verts if headv(v)}
        HV = np.array([tuple(bm.verts[i].co) for i in hv])
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if sum(v.index in hv for v in f.verts) >= 2], context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        ob["head_box"] = [list(HV.min(0)), list(HV.max(0))]
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    mod = ob.modifiers.new("arm", "ARMATURE"); mod.object = arm
    ob.parent = arm
    sub = ob.modifiers.new("sub", "SUBSURF"); sub.levels = 1; sub.render_levels = 1
    fl = S["mannequin"].get("faceless")
    if fl:
        lo, hi = np.array(ob["head_box"][0]), np.array(ob["head_box"][1])
        c = (lo + hi) / 2; r = (hi - lo) / 2 * fl["egg_fit"]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, location=tuple(c))
        egg = bpy.context.object; egg.name = "head_egg"
        egg.scale = (r[0], r[1], r[2] * 0.98)
        for p in egg.data.polygons:
            p.use_smooth = True
        egg.parent = arm; egg.parent_type = "BONE"; egg.parent_bone = "head"
        pb = arm.pose.bones["head"]
        egg.matrix_parent_inverse = (arm.matrix_world @ pb.matrix @ Matrix.Translation((0, arm.data.bones["head"].length, 0))).inverted()
        ob["egg"] = egg.name
    return ob, arm, J


def make_shorts(S, body):
    """matte black shorts = a 4 mm shell of the body's own pelvis/upper-thigh faces (deforms with the same weights)"""
    sh = body.copy(); sh.data = body.data.copy(); sh.name = "shorts"
    bpy.context.collection.objects.link(sh)
    keep_groups = {"root", "pelvis.L", "pelvis.R", "upperleg01.L", "upperleg01.R", "spine05"}
    me = sh.data
    gi = {g.index: g.name for g in sh.vertex_groups}
    z = np.array([v.co.z for v in me.vertices])
    dom = []
    for v in me.vertices:
        ws = [(g.weight, gi[g.group]) for g in v.groups]
        dom.append(max(ws)[1] if ws else "")
    J = {n: sh.parent.data.bones[n].head_local for n in ("upperleg01.L", "lowerleg01.L", "spine04")}
    z_hi = J["spine04"].z + 0.01
    z_lo = J["upperleg01.L"].z - 0.55 * (J["upperleg01.L"].z - J["lowerleg01.L"].z)
    ok = np.array([(dom[i] in keep_groups or dom[i].startswith("upperleg02")) and z_lo < z[i] < z_hi for i in range(len(z))])
    bm = bmesh.new(); bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not all(ok[v.index] for v in f.verts)], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(me); bm.free()
    sol = sh.modifiers.new("off", "SOLIDIFY"); sol.thickness = S["mannequin"]["shorts"]["offset_m"]; sol.offset = 1.0
    sh.modifiers.move(sh.modifiers.find("off"), 0)
    return sh


# ---------------------------------------------------------------- anatomy (BodyParts3D, CC BY 4.0)
def read_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(t) for t in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
    return np.array(V) * 0.001, F          # mm -> m, already Z-up / front -Y / left +X


def ends(V, axis_lo=True):
    z = V[:, 2]; r = z.max() - z.min()
    return V[z > z.max() - 0.12 * r].mean(0), V[z < z.min() + 0.10 * r].mean(0)


def make_anatomy(S, arm, J):
    import zipfile
    zf = zipfile.ZipFile(os.path.join(S["asset_dir"], S["assets"]["anatomy_zip"]["file"]))
    cache = os.path.join(S["asset_dir"], "bp3d"); os.makedirs(cache, exist_ok=True)
    P = S["anatomy_parts"]

    def part(fj):
        p = os.path.join(cache, fj + ".obj")
        if not os.path.exists(p):
            open(p, "wb").write(zf.read(f"partof_BP3D_4.0_obj_99/{fj}.obj"))
        return read_obj(p)
    raw = {}
    for name, d in P.items():
        for side in ("L", "R", "C"):
            if side in d:
                vs, fs, off = [], [], 0
                for fj in d[side]:
                    v, f = part(fj); vs.append(v); fs += [[i + off for i in t] for t in f]; off += len(v)
                raw[(name, side)] = (np.vstack(vs), fs)
    B = arm.data.bones
    hl = lambda n: np.array(B[n].head_local)
    # torso fit: uniform scale + translation (axes already agree) on humeral heads + clavicle sternal ends
    src, dst = [], []
    for sd in "LR":
        hh, _ = ends(raw[("humerus", sd)][0]); src.append(hh); dst.append(hl(f"upperarm01.{sd}"))
        cv = raw[("clavicle", sd)][0]; med = cv[np.argsort(np.abs(cv[:, 0]))[: max(10, len(cv) // 12)]].mean(0)
        src.append(med); dst.append(hl(f"clavicle.{sd}"))
    src, dst = np.array(src), np.array(dst)
    cs, cd = src.mean(0), dst.mean(0)
    s = float(np.sqrt(((dst - cd) ** 2).sum() / ((src - cs) ** 2).sum()))
    T = lambda v: (v - cs) * s + cd
    objs = {}
    for (name, side), (v, f) in raw.items():
        v = T(v)
        if name in ("humerus", "radius_ulna"):
            if name == "humerus":
                a0, a1 = ends(v); b0, b1 = hl(f"upperarm01.{side}"), hl(f"lowerarm01.{side}")
            else:
                a0, a1 = ends(v); b0, b1 = hl(f"lowerarm01.{side}"), hl(f"wrist.{side}")
            q = Vector(a1 - a0).rotation_difference(Vector(b1 - b0))
            k = np.linalg.norm(b1 - b0) / np.linalg.norm(a1 - a0)
            R = np.array(q.to_matrix())
            v = ((v - a0) @ R.T) * k + b0
        me = bpy.data.meshes.new(f"{name}_{side}"); me.from_pydata([tuple(p) for p in v], [], f); me.update()
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(f"{name}_{side}", me); bpy.context.collection.objects.link(ob)
        bn = P[name]["bone"] + ("" if side == "C" else f".{side}")
        if bn not in arm.pose.bones:
            bn = P[name]["bone"]
        ob.parent = arm; ob.parent_type = "BONE"; ob.parent_bone = bn
        pb = arm.pose.bones[bn]
        ob.matrix_parent_inverse = (arm.matrix_world @ pb.matrix @ Matrix.Translation((0, B[bn].length, 0))).inverted()
        ob["anatomy"] = name; ob["kind"] = P[name].get("kind", "bone")
        objs[(name, side)] = ob
    return objs


# ---------------------------------------------------------------- equipment (procedural, ours)
def box(name, size, loc, rot_x=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object; o.name = name; o.scale = size; o.rotation_euler = (rot_x, 0, 0)
    bev = o.modifiers.new("bev", "BEVEL"); bev.width = 0.012; bev.segments = 3
    return o


def make_bench(S, incline_deg, seat_front_y):
    b = S["equipment"]["bench"]; parts = []
    h, L, BL, Wd, t = b["seat_h"], b["seat_len"], b["back_len"], b["pad_w"], b["pad_t"]
    y0 = seat_front_y + L / 2
    parts.append(box("bench_seat", (Wd, L, t), (0, y0, h)))
    a = math.radians(incline_deg)
    hy, hz = seat_front_y + L, h + t / 2
    cy, cz = hy + math.cos(a) * BL / 2, hz + math.sin(a) * BL / 2 - t / 2
    parts.append(box("bench_back", (Wd, BL, t), (0, cy, cz), rot_x=a))
    parts.append(box("bench_rail", (0.07, L + BL * math.cos(a) + 0.2, 0.06), (0, seat_front_y + (L + BL * math.cos(a)) / 2, 0.08)))
    parts.append(box("bench_post", (0.06, 0.06, h - 0.1), (0, y0, h / 2)))
    parts.append(box("bench_foot_f", (0.42, 0.07, 0.05), (0, seat_front_y - 0.05, 0.03)))
    parts.append(box("bench_foot_b", (0.42, 0.07, 0.05), (0, seat_front_y + L + BL * math.cos(a) + 0.05, 0.03)))
    parts.append(box("bench_strut", (0.05, 0.05, BL * 0.55), (0, hy + math.cos(a) * BL * 0.45, (hz + 0.08) / 2 + math.sin(a) * BL * 0.2), rot_x=-0.5))
    return parts


def make_dumbbell(S, name):
    d = S["equipment"]["dumbbell"]
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=d["handle_r"], depth=d["handle_len"] + 2 * d["head_len"], location=(0, 0, 0))
    hnd = bpy.context.object; hnd.name = name + "_handle"; parts.append(hnd)
    for sgn in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=d["head_r"], depth=d["head_len"], location=(0, 0, sgn * (d["handle_len"] / 2 + d["head_len"] / 2)))
        hd = bpy.context.object; hd.name = f"{name}_head{sgn}"
        bev = hd.modifiers.new("bev", "BEVEL"); bev.width = 0.006; bev.segments = 2
        parts.append(hd)
    root = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(root)
    for p in parts:
        p.parent = root
    root["parts"] = [p.name for p in parts]
    return root, parts


# ---------------------------------------------------------------- materials
def mat_principled(name, rgb, rough, spec=0.4, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*rgb, 1); p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if "Specular IOR Level" in p.inputs:
        p.inputs["Specular IOR Level"].default_value = spec
    return m


def mat_emit(name, rgb, strength=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial"); e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*rgb, 1); e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def mat_xray(name, tint, core_white, rim_power):
    """X-Ray bone (Day 1 v36 look): glassy - opacity and brightness rise toward the silhouette rim (layer-weight
    facing), the core stays see-through, so every bone keeps its 3D relief instead of reading as a flat decal"""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial"); e = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent"); mix = nt.nodes.new("ShaderNodeMixShader")
    lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.30
    pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 1.0 / rim_power
    mad = nt.nodes.new("ShaderNodeMath"); mad.operation = "MULTIPLY_ADD"; mad.use_clamp = True
    mad.inputs[1].default_value = 0.80; mad.inputs[2].default_value = 0.22
    nt.links.new(lw.outputs["Facing"], pw.inputs[0]); nt.links.new(pw.outputs[0], mad.inputs[0])
    e.inputs["Color"].default_value = (1, 1, 1, 1); e.inputs["Strength"].default_value = 1.0
    nt.links.new(mad.outputs[0], mix.inputs[0]); nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], o.inputs[0])
    return m


def assign(ob, m):
    ob.data.materials.clear(); ob.data.materials.append(m)


# ---------------------------------------------------------------- posing
def upd():
    bpy.context.view_layer.update()


def aim(arm, bn, d):
    pb = arm.pose.bones[bn]; upd()
    M = pb.matrix.copy()
    y = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = y.rotation_difference(Vector(d).normalized())
    R = q.to_matrix() @ M.to_3x3()
    N = R.to_4x4(); N.translation = M.translation
    pb.matrix = N; upd()


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def pose(arm, EX, form, phase):
    inc = math.radians(EX["setup"]["bench_incline_deg"])
    Tt = unit((0, math.cos(inc), math.sin(inc)))          # hips -> shoulders, along the bench back
    N = unit((0, -math.sin(inc), math.cos(inc)))           # chest normal (anterior)
    for bn in ("spine05", "spine04", "spine03", "spine02", "spine01"):
        aim(arm, bn, Tt)
    for bn in ("neck01", "neck02", "neck03", "head"):
        aim(arm, bn, unit(Tt * 0.85 + N * 0.35))
    kf = math.radians(EX.get("knee_flexion_deg", 95))
    for sd, sx in (("L", 1), ("R", -1)):
        th = unit((0.16 * sx, -1.0, 0.0))
        aim(arm, f"upperleg01.{sd}", th); aim(arm, f"upperleg02.{sd}", th)
        sh = unit((0.10 * sx, -math.cos(kf), -math.sin(kf)))
        sh = unit((0.08 * sx, math.cos(math.pi - kf) * -1, -1.0)) if False else unit((0.06 * sx, -math.cos(kf) * -1 * -1, -1.0))
        aim(arm, f"lowerleg01.{sd}", sh); aim(arm, f"lowerleg02.{sd}", sh)
        aim(arm, f"foot.{sd}", unit((0.05 * sx, -1.0, -0.25)))
    f0, f1 = EX["forms"][form]["bottom"], EX["forms"][form]["top"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    a, ef, dr = math.radians(lerp("arm_torso_deg")), math.radians(lerp("elbow_flexion_deg")), math.radians(lerp("elbow_drop_deg"))
    hands = {}
    for sd, sx in (("L", 1), ("R", -1)):
        X = np.array((sx, 0.0, 0.0))
        v = -Tt * math.cos(a) + X * math.sin(a)
        u = unit(v * math.cos(dr) - N * math.sin(dr))
        w = unit(N - u * np.dot(N, u))
        fa = unit(u * math.cos(ef) + w * math.sin(ef))
        aim(arm, f"upperarm01.{sd}", u); aim(arm, f"upperarm02.{sd}", u)
        aim(arm, f"lowerarm01.{sd}", fa); aim(arm, f"lowerarm02.{sd}", fa)
        aim(arm, f"wrist.{sd}", fa)
        for fn in range(2, 6):
            for seg in (1, 2, 3):
                bn = f"finger{fn}-{seg}.{sd}"
                if bn in arm.pose.bones:
                    pb = arm.pose.bones[bn]; pb.rotation_mode = "XYZ"; pb.rotation_euler = (math.radians(55), 0, 0)
        upd()
        wr = arm.pose.bones[f"wrist.{sd}"]
        hands[sd] = (np.array(arm.matrix_world @ wr.tail), fa)
    return hands, Tt, N


# ---------------------------------------------------------------- render
def setup_render(S, scale, samples):
    sc = bpy.context.scene; r = S["render"]
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = samples
    sc.cycles.use_denoising = r["denoise"]
    sc.render.threads_mode = "FIXED"; sc.render.threads = r["threads"]
    sc.render.resolution_x, sc.render.resolution_y = r["res"]; sc.render.resolution_percentage = int(scale * 100)
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*S["world_rgb_linear"], 1)
    return sc


def add_lights(S, anchor):
    for L in S["lights"]:
        ld = bpy.data.lights.new(L["name"], L["type"]); ld.energy = L["energy"]
        if L["type"] == "AREA":
            ld.size = L["size"]
        o = bpy.data.objects.new(L["name"], ld); bpy.context.collection.objects.link(o)
        o.location = Vector(L["loc"]) + Vector(anchor)
        d = (Vector(L["aim"]) + Vector(anchor)) - o.location
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_camera(S, preset, target):
    c = S["cameras"][preset]
    cd = bpy.data.cameras.new("cam"); cd.lens = c["lens_mm"]; cd.sensor_fit = "VERTICAL"; cd.sensor_height = 24
    cam = bpy.data.objects.new("cam", cd); bpy.context.collection.objects.link(cam)
    yaw, pit, dist = math.radians(c["yaw_deg"]), math.radians(c["pitch_deg"]), c["dist_m"]
    off = Vector((math.sin(yaw) * math.cos(pit) * -dist * -1 if False else -math.sin(yaw) * math.cos(pit) * dist,
                  -math.cos(yaw) * math.cos(pit) * dist, math.sin(pit) * dist))
    cam.location = Vector(target) + off
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def fit_bench(S, EX, body, arm, Tt, N):
    """locked contact rule: the body rests ON the bench. Height: the lowest buttock/thigh point sits on the seat
    top (bench height is fixed - it stands on the floor). Fore-aft: the bench slides so its back pad (same incline as
    the torso) touches the most posterior back point. Returns the bench seat_front_y."""
    b = S["equipment"]["bench"]; gap = S["mannequin"].get("contact", {}).get("gap_m", 0.004)
    a = math.radians(EX["setup"]["bench_incline_deg"])
    seat_top = b["seat_h"] + b["pad_t"] / 2
    n = np.array((0.0, -math.sin(a), math.cos(a)))
    def verts():
        dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg)
        V = np.array([ev.matrix_world @ v.co for v in ev.to_mesh().vertices]); ev.to_mesh_clear(); return V
    P = lambda: (np.array(arm.matrix_world @ arm.pose.bones["upperleg01.L"].head) + np.array(arm.matrix_world @ arm.pose.bones["upperleg01.R"].head)) / 2
    V, p = verts(), P()
    seat_sel = (np.abs(V[:, 0] - p[0]) < 0.22) & (V[:, 1] > p[1] - 0.30) & (V[:, 1] < p[1] + 0.12) & (V[:, 2] < p[2])
    zs = V[seat_sel, 2].min()
    arm.location = arm.location + Vector((0, 0, seat_top + gap - zs)); upd()
    V, p = verts(), P()
    rel = V - p; along = rel @ Tt
    back_sel = (np.abs(rel[:, 0]) < 0.16) & (along > 0.10) & (along < 0.55)
    dn = (V[back_sel] @ n).min() - gap
    hinge_y = (math.cos(a) * seat_top - dn) / math.sin(a)
    print(f"contact: seat z {zs:.3f}->{seat_top:.3f}, back plane {dn:.3f}, hinge y {hinge_y:.3f}")
    return hinge_y - b["seat_len"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", default="incline_dumbbell_press"); ap.add_argument("--form", default="wrong")
    ap.add_argument("--phase", type=float, default=0.0); ap.add_argument("--camera", default="front")
    ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0); ap.add_argument("--skip", default="")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    S = json.load(open(os.path.join(HERE, "set_v1.json")))
    EX = json.load(open(os.path.join(HERE, "exercises", a.exercise + ".json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = setup_render(S, a.scale, a.samples or S["render"]["samples"])
    body, arm, J = make_mannequin(S)
    shorts = make_shorts(S, body)
    anat = make_anatomy(S, arm, J)
    hands, Tt, N = pose(arm, EX, a.form, a.phase)
    # seat the posed mannequin: pelvis centred on x, then the bench is fitted to the body (contact rule)
    upd()
    pc = (arm.matrix_world @ arm.pose.bones["upperleg01.L"].head + arm.matrix_world @ arm.pose.bones["upperleg01.R"].head) / 2
    arm.location = Vector((-pc.x, -pc.y, 0)); upd()
    seat_front_y = fit_bench(S, EX, body, arm, Tt, N)
    bench = make_bench(S, EX["setup"]["bench_incline_deg"], seat_front_y)
    dbs = []
    for sd, (hp, fa) in hands.items():
        root, parts = make_dumbbell(S, f"db_{sd}")
        hp = np.array(arm.matrix_world @ arm.pose.bones[f"wrist.{sd}"].tail)
        fa = unit(fa)
        ax = unit(np.array((1.0, 0, 0)) - fa * fa[0])
        root.location = Vector(hp + fa * 0.045)
        root.rotation_euler = Vector(ax).to_track_quat("Z", "Y").to_euler()
        dbs += parts
    upd()
    # floor shadow catcher
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0)); floor = bpy.context.object; floor.name = "floor"
    floor.is_shadow_catcher = True
    # materials
    m = S["mannequin"]; e = S["equipment"]
    skin_m = mat_principled("skin", m["skin"]["base_rgb_linear"], m["skin"]["roughness"], m["skin"]["specular"])
    assign(body, skin_m)
    egg = bpy.data.objects.get("head_egg")
    if egg:
        assign(egg, skin_m)
    assign(shorts, mat_principled("shorts", m["shorts"]["rgb_linear"], m["shorts"]["roughness"], 0.15))
    mb = mat_principled("bench", e["bench"]["rgb_linear"], e["bench"]["roughness"], 0.3)
    for o in bench:
        assign(o, mb)
    mdb = mat_principled("db", e["dumbbell"]["rgb_linear"], 0.45, 0.4)
    mdh = mat_principled("db_metal", e["dumbbell"]["metal_rgb_linear"], 0.3, 0.5, metal=1.0)
    for o in dbs:
        assign(o, mdh if o.name.endswith("_handle") else mdb)
    # camera + lights
    chest = arm.matrix_world @ arm.pose.bones["spine02"].head
    to = S["cameras"][a.camera].get("target_offset_m", [0, 0, 0])
    tgt = (to[0], float(chest.y) + to[1], float(chest.z) + to[2])
    cam = add_camera(S, a.camera, tgt)
    add_lights(S, (0, float(chest.y) - 0.3, 0))
    tgt_bones = set(EX["target_bones"])
    xr = S["xray"]
    mx = mat_xray("xray", xr["bone_tint"][a.form], xr["core_white"], xr["rim_power"])
    for (name, sd), ob in anat.items():
        assign(ob, mx)
    out = os.path.abspath(a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    anat_objs = list(anat.values())
    scene_objs = [body, shorts, floor] + bench + dbs + ([egg] if egg else [])
    # pass 1: beauty (anatomy hidden)
    for o in anat_objs:
        o.hide_render = True
    sc.render.filepath = out + "_beauty.png"
    if "beauty" not in a.skip:
        bpy.ops.render.render(write_still=True)
    if "id" in a.skip:
        pass
    # pass 2: flat ID (1 sample, emission, no AA)
    sc.cycles.samples = 1; sc.cycles.use_denoising = False; sc.cycles.pixel_filter_type = "BOX"; sc.cycles.filter_width = 0.01
    ids = {"skin": [body] + ([egg] if egg else []), "shorts": [shorts], "bench": bench, "dumbbell": dbs}
    saved = {}
    for k, obs in ids.items():
        mi = mat_emit("id_" + k, ID_COL[k])
        for o in obs:
            saved[o.name] = list(o.data.materials); assign(o, mi)
    floor.hide_render = True
    sc.render.filepath = out + "_id.png"
    bpy.ops.render.render(write_still=True)
    # pass 3: X-Ray bones (target bones only, not occluded - composited onto skin pixels only)
    for o in scene_objs:
        o.hide_render = True
    for (name, sd), o in anat.items():
        o.hide_render = not (name in tgt_bones and o.get("kind") == "bone")
    sc.cycles.samples = max(4, (a.samples or S["render"]["samples"]) // 2); sc.cycles.use_denoising = True
    sc.cycles.pixel_filter_type = "BLACKMAN_HARRIS"; sc.cycles.filter_width = 1.5
    sc.render.filepath = out + "_bones.png"
    bpy.ops.render.render(write_still=True)
    # meta: projected joints in 1080x1920 px
    rx, ry = S["render"]["res"]
    def px(p):
        c = world_to_camera_view(sc, cam, Vector(p)); return [round(c.x * rx, 1), round((1 - c.y) * ry, 1)]
    P = lambda bn, tail=False: arm.matrix_world @ (arm.pose.bones[bn].tail if tail else arm.pose.bones[bn].head)
    kp = {"sh_l": px(P("upperarm01.L")), "sh_r": px(P("upperarm01.R")), "elb_l": px(P("lowerarm01.L")), "elb_r": px(P("lowerarm01.R")),
          "wr_l": px(P("wrist.L")), "wr_r": px(P("wrist.R")), "hip_l": px(P("upperleg01.L")), "hip_r": px(P("upperleg01.R")),
          "kn_l": px(P("lowerleg01.L")), "kn_r": px(P("lowerleg01.R")), "ank_l": px(P("foot.L")), "ank_r": px(P("foot.R"))}
    kp["sh"] = [(kp["sh_l"][0] + kp["sh_r"][0]) / 2, (kp["sh_l"][1] + kp["sh_r"][1]) / 2]
    # true 3D angles (QC: the motion standard is met in 3D, not a 2D estimate)
    def ang(a_, b_, c_):
        u, v = np.array(a_) - np.array(b_), np.array(c_) - np.array(b_)
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    hipc = (np.array(P("upperleg01.L")) + np.array(P("upperleg01.R"))) / 2
    shc = (np.array(P("upperarm01.L")) + np.array(P("upperarm01.R"))) / 2
    angles = {sd: {"elbow_flexion": round(180 - ang(P(f"upperarm01.{sd}"), P(f"lowerarm01.{sd}"), P(f"wrist.{sd}")), 1),
                   "arm_torso": round(ang(P(f"lowerarm01.{sd}"), P(f"upperarm01.{sd}"), np.array(P(f"upperarm01.{sd}")) + (hipc - shc)), 1)}
              for sd in "LR"}
    meta = {"set_version": S["version"], "set_hash": set_hash(S), "exercise": a.exercise, "form": a.form, "phase": a.phase,
            "camera": a.camera, "kp": kp, "angles_3d": angles, "id_colours": ID_COL,
            "attribution": S["assets"]["anatomy_zip"]["licence"]}
    json.dump(meta, open(out + "_meta.json", "w"), indent=1)
    print("META", json.dumps(meta)[:600])


if __name__ == "__main__":
    main()
