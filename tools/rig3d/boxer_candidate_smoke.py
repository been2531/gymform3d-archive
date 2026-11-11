#!/usr/bin/env python3
"""Tiny import + jab-retarget smoke test for a downloaded boxer candidate.

Research-only helper for docs/specs/BOXER_OPTIONS.md. Retargets the X Bot
alt04 Mixamo jab (frames guard 9 / full 14 / return 22) onto a candidate rig
by world-space bone *direction* (swing from the source, twist kept from the
target rest), scales the hips translation by leg length, then scores both
rigs with boxing_qc and measures mesh stretch (skin distortion).

  nice -n 19 blender -b --python tools/rig3d/boxer_candidate_smoke.py -- \
      --input output/_basemodels/boxer_v3/<slug>/scene.gltf --name <slug> \
      [--bonemap chempunk] [--samples 16]

Writes (gitignored) output/_basemodels/boxer_v3/smoke/<slug>_{rest,guard,full}.png
and <slug>_smoke.json. No production renders.
"""
import argparse, json, math, os, re, sys
import bpy, bmesh
from mathutils import Matrix, Vector, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_xbot_boxing as xb  # noqa: E402
import boxing_qc as qc  # noqa: E402
from pick_xbot_punches_v3 import analyze_rich, enrich_bones  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
JAB_FILE = os.path.join(xb.SRC_DIR, "alts/alt04_Jab Cross_BoxingJabCrossMedium.fbx")
JAB = dict(id="alt04_jab_L_f14", side="L", guard=9, full=14, ret=22, ptype="jab")
TARGET_H = 1.78

BONEMAPS = {
    "chempunk": {
        "ass": "Hips", "spine": "Spine", "chest": "Spine1", "chest.001": "Spine2",
        "head": "Head", "shoulder.L": "LeftShoulder", "upper_arm.L": "LeftArm",
        "lower_arm.L": "LeftForeArm", "hand.L": "LeftHand",
        "shoulder.R": "RightShoulder", "upper_arm.R": "RightArm",
        "lower_arm.R": "RightForeArm", "hand.R": "RightHand",
        "upper_leg.L": "LeftUpLeg", "lower_leg.L": "LeftLeg", "foot.L": "LeftFoot",
        "upper_leg.R": "RightUpLeg", "lower_leg.R": "RightLeg", "foot.R": "RightFoot",
    },
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bonemap", default="")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--xbot-only", action="store_true")
    ap.add_argument("--yaw", type=float, default=35.0)
    return ap.parse_args(argv)


def canon(name):
    n = re.sub(r"_\d+$", "", name)
    n = n.replace("mixamorig:", "").replace("mixamorig_", "")
    return n


def import_xbot():
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=JAB_FILE)
    new = [o for o in bpy.data.objects if o not in before]
    arm = [o for o in new if o.type == "ARMATURE"][0]
    meshes = [o for o in new if o.type == "MESH"]
    arm.name = "XBot.rig"
    bones = xb.bone_map(arm)
    f0, f1 = xb.frame_range(arm)
    bpy.context.scene.frame_set(f0)
    xb.scale_ground_center(arm, meshes, bones)
    bpy.context.view_layer.update()
    return arm, meshes, (f0, f1)


def import_candidate(path, bonemap):
    before = set(bpy.data.objects)
    if path.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    arm = [o for o in new if o.type == "ARMATURE"][0]
    meshes = []
    for o in new:
        if o.type != "MESH":
            continue
        skinned = any(m.type == "ARMATURE" for m in o.modifiers)
        if not skinned or o.name.startswith("Icosphere"):
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        meshes.append(o)
    if arm.animation_data:
        arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
        for c in list(pb.constraints):
            pb.constraints.remove(c)
    # canonical Mixamo names so xb.bone_map / analyze_rich work unchanged
    bm = BONEMAPS.get(bonemap, {})
    for b in arm.data.bones:
        c = canon(b.name)
        c = bm.get(c, c if not bm else None)
        if c:
            b.name = "mixamorig:" + c
    arm.name = "Cand.rig"
    # Collapse any parent chain above the armature into its world matrix.
    mw = arm.matrix_world.copy()
    top = arm
    while top.parent:
        top = top.parent
    arm.parent = None
    arm.matrix_world = mw
    bpy.context.view_layer.update()
    return arm, meshes


def eval_bbox(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        for v in me.vertices:
            w = oe.matrix_world @ v.co
            mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
        oe.to_mesh_clear()
    return mn, mx


def bw(arm, name):
    return arm.matrix_world @ arm.pose.bones[name].head


def normalize_candidate(arm, meshes, xarm):
    mn, mx = eval_bbox(meshes)
    h = mx.z - mn.z
    s = TARGET_H / h
    arm.matrix_world = Matrix.Scale(s, 4) @ arm.matrix_world
    bpy.context.view_layer.update()
    # face the same way as X Bot (character-left = same world side)
    def lr(a):  # REST pose (X Bot frame 1 is already a rotated stance)
        v = a.matrix_world @ a.data.bones["mixamorig:LeftUpLeg"].head_local - a.matrix_world @ a.data.bones["mixamorig:RightUpLeg"].head_local
        v.z = 0
        return v.normalized()
    bpy.context.scene.frame_set(1)
    xl = lr(xarm); cl = lr(arm)
    ang = math.atan2(xl.y, xl.x) - math.atan2(cl.y, cl.x)
    arm.matrix_world = Matrix.Rotation(ang, 4, "Z") @ arm.matrix_world
    bpy.context.view_layer.update()
    mn, mx = eval_bbox(meshes)
    c = (mn + mx) / 2
    arm.matrix_world = Matrix.Translation((-c.x, -c.y, -mn.z)) @ arm.matrix_world
    bpy.context.view_layer.update()
    return dict(raw_height=h, scale=s, yaw_fix_deg=math.degrees(ang))


def swing_twist_twist(q, axis=Vector((0, 1, 0))):
    p = Vector((q.x, q.y, q.z)).project(axis)
    t = Quaternion((q.w, p.x, p.y, p.z))
    if t.magnitude < 1e-8:
        return Quaternion()
    t.normalize()
    return t


def retarget(xarm, arm, f0, f1):
    sc = bpy.context.scene
    # rest world rotations
    sc.frame_set(f0)
    shared = [pb.name for pb in arm.pose.bones if pb.name in xarm.pose.bones]
    order = sorted(shared, key=lambda n: len(arm.pose.bones[n].parent_recursive))
    xrest = {n: (xarm.matrix_world.to_3x3().normalized() @ xarm.data.bones[n].matrix_local.to_3x3()).to_quaternion() for n in shared}
    trest = {n: (arm.matrix_world.to_3x3().normalized() @ arm.data.bones[n].matrix_local.to_3x3()).to_quaternion() for n in shared}
    twist = {n: swing_twist_twist(xrest[n].inverted() @ trest[n]) for n in shared}
    # hips translation scale by hip height
    hx = (xarm.matrix_world @ xarm.data.bones["mixamorig:Hips"].head_local)
    ht = (arm.matrix_world @ arm.data.bones["mixamorig:Hips"].head_local)
    k = ht.z / hx.z
    aw_inv = arm.matrix_world.to_3x3().normalized().inverted()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        for n in order:
            sq = (xarm.matrix_world.to_3x3().normalized() @ xarm.pose.bones[n].matrix.to_3x3().normalized()).to_quaternion()
            want = sq @ twist[n]
            pb = arm.pose.bones[n]
            bpy.context.view_layer.update()
            cur = pb.matrix.copy()
            rot_arm = (aw_inv @ want.to_matrix())
            m = rot_arm.to_4x4()
            if n == "mixamorig:Hips":
                sx = xarm.matrix_world @ xarm.pose.bones[n].head
                tw = ht + (sx - hx) * k
                m.translation = arm.matrix_world.inverted() @ tw
            else:
                m.translation = cur.translation
            # keep bone-local scale of the armature-space matrix
            pb.matrix = m
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if n == "mixamorig:Hips":
                pb.keyframe_insert("location", frame=f)
    sc.frame_set(f0)
    return dict(shared_bones=len(shared), hip_scale=k)


def qc_rig(arm, label):
    bones = enrich_bones(arm, xb.bone_map(arm))
    rows = analyze_rich(arm, bones, JAB["guard"] - 2, JAB["ret"] + 2)
    path = qc.punch_path_metrics(rows, dict(JAB), side=JAB["side"])
    win = [r for r in rows if JAB["guard"] <= r["f"] <= JAB["ret"]]

    def span(key):
        pts = [r[key] for r in win if r.get(key)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    g = [r for r in rows if r["f"] == JAB["guard"]][0]
    H = TARGET_H if label != "xbot" else None
    head = Vector(g["head"]); hl = Vector(g["hand_l"]); hr = Vector(g["hand_r"])
    shw = (Vector(g["arm_l"]) - Vector(g["arm_r"])).length
    height = (Vector(g["head"]).z)
    out = {k: path.get(k) for k in ("fist_max_dev_m", "fist_arc_height_m", "elbow_above_shoulder_max_m", "retract_max_dev_m", "start_to_chin_m") if k in path}
    out.update(foot_slide_l_m=span("foot_l"), foot_slide_r_m=span("foot_r"), root_drift_m=span("hips"),
               guard_lead_hand_to_head_rel=(hl - head).length / height,
               guard_rear_hand_to_head_rel=(hr - head).length / height,
               guard_hand_sep_over_shoulder=(hl - hr).length / max(shw, 1e-6),
               head_z=height)
    return out


def stretch(meshes, frames):
    dg = bpy.context.evaluated_depsgraph_get()
    res = {}
    rest = {}
    arm = None
    for o in meshes:
        me = o.data
        rest[o.name] = [(me.vertices[e.vertices[0]].co - me.vertices[e.vertices[1]].co).length for e in me.edges]
    for f in frames:
        bpy.context.scene.frame_set(f)
        ratios = []
        for o in meshes:
            oe = o.evaluated_get(dg)
            me = oe.to_mesh()
            s = o.matrix_world.to_scale().x
            rl = rest[o.name]
            for i, e in enumerate(me.edges):
                if rl[i] < 1e-7:
                    continue
                ratios.append((me.vertices[e.vertices[0]].co - me.vertices[e.vertices[1]].co).length / rl[i])
            oe.to_mesh_clear()
        ratios.sort()
        n = len(ratios)
        res[str(f)] = dict(p01=round(ratios[int(n * .01)], 3), p99=round(ratios[int(n * .99)], 3),
                           max=round(ratios[-1], 3), frac_over_1_5=round(sum(r > 1.5 for r in ratios) / n, 4))
    return res


def setup_render(samples, cover_objs, yaw):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("W"); sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.055, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam); sc.camera = cam
    cam.data.lens = 50
    for nm, loc, e in (("Key", (2.5, -3.0, 3.2), 600), ("Fill", (-3, -2, 1.8), 220), ("Rim", (0.5, 3.5, 2.8), 380)):
        L = bpy.data.objects.new(nm, bpy.data.lights.new(nm, "AREA"))
        L.data.energy = e; L.data.size = 2.0
        L.location = loc
        d = Vector((0, 0, 1.0)) - Vector(loc)
        L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(L)
    return cam


def frame_cam(cam, objs, yaw):
    mn, mx = eval_bbox(objs)
    c = (mn + mx) / 2
    h = max(mx.z - mn.z, 0.5)
    dist = h / 0.72 / (2 * math.tan(cam.data.angle_y / 2)) if hasattr(cam.data, "angle_y") else 4.5
    cam.data.sensor_fit = "VERTICAL"
    fov = 2 * math.atan(cam.data.sensor_height / (2 * cam.data.lens))
    dist = h / 0.72 / (2 * math.tan(fov / 2))
    a = math.radians(yaw)
    pos = Vector((c.x + math.sin(a) * dist, c.y - math.cos(a) * dist, c.z + 0.1))
    cam.location = pos
    cam.rotation_euler = (c - pos).to_track_quat("-Z", "Y").to_euler()


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = os.path.join(ROOT, "output/_basemodels/boxer_v3/smoke")
    os.makedirs(out, exist_ok=True)
    xarm, xmeshes, (f0, f1) = import_xbot()
    facts = {"name": a.name, "jab": JAB, "frames": [f0, f1]}
    cam = setup_render(a.samples, xmeshes, a.yaw)
    if a.xbot_only:
        facts["qc"] = qc_rig(xarm, "xbot")
        facts["stretch"] = stretch(xmeshes, [JAB["guard"], JAB["full"]])
        for tag, f in (("guard", JAB["guard"]), ("full", JAB["full"])):
            bpy.context.scene.frame_set(f); frame_cam(cam, xmeshes, a.yaw)
            render(os.path.join(out, f"{a.name}_{tag}.png"))
        act = xarm.animation_data.action
        xarm.animation_data.action = None
        for pb in xarm.pose.bones:
            pb.matrix_basis.identity()
        bpy.context.view_layer.update()
        frame_cam(cam, xmeshes, a.yaw)
        render(os.path.join(out, f"{a.name}_rest.png"))
        xarm.animation_data.action = act
    else:
        arm, meshes = import_candidate(a.input, a.bonemap)
        facts["norm"] = normalize_candidate(arm, meshes, xarm)
        for o in xmeshes:
            o.hide_render = True
        # rest still
        frame_cam(cam, meshes, a.yaw)
        render(os.path.join(out, f"{a.name}_rest.png"))
        facts["retarget"] = retarget(xarm, arm, f0, f1)
        facts["qc"] = qc_rig(arm, a.name)
        facts["stretch"] = stretch(meshes, [JAB["guard"], JAB["full"]])
        for tag, f in (("guard", JAB["guard"]), ("full", JAB["full"])):
            bpy.context.scene.frame_set(f); frame_cam(cam, meshes, a.yaw)
            render(os.path.join(out, f"{a.name}_{tag}.png"))
    with open(os.path.join(out, f"{a.name}_smoke.json"), "w") as fh:
        json.dump(facts, fh, indent=1)
    print("SMOKE", json.dumps(facts, indent=1))


main()
