"""Rest-pose look probe for QUALITY_UPGRADE_PLAN (t727u).

Renders 3 stills of the Grab3D CC0 body (no motion, no exercise generation):
  before_current.png      production-like: flat grey albedo, roughness 0.62, normal 1.5, 3 white area lights
  after_light_sss.png     unused 2K albedo/roughness + SSS + key/rim + floor
  after_light_sss_glow.png  same + back-torso emission + compositor Bloom

Usage (headless bpy):
  nice -n 19 /workspace/.bpyenv/bin/python tools/rig3d/quality_probe.py
"""
import bpy, math, os, sys, time
import mathutils as mu
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLB = os.path.join(ROOT, "_scratch/assets_probe/human-muscles-rigged.glb")
HDRI = os.path.join(ROOT, "_scratch/hdri/studio_small_09_1k.hdr")
OUT = os.path.join(ROOT, "output/_rig3d/quality_probe")
os.makedirs(OUT, exist_ok=True)

RES = (540, 960)
SPP = 20


def reset_import():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=GLB)
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    body = bpy.data.objects["char1"]
    for o in list(bpy.data.objects):
        if o.name == "Icosphere":
            o.hide_render = True
            o.hide_viewport = True
    if arm.animation_data:
        arm.animation_data.action = None
        for t in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(t)
    for b in arm.pose.bones:
        b.rotation_mode = "XYZ"
        b.rotation_euler = (0, 0, 0)
        b.rotation_quaternion = (1, 0, 0, 0)
        b.location = (0, 0, 0)
    bpy.context.view_layer.update()
    return arm, body


def world_bbox(obj):
    mw = np.array(obj.matrix_world)
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    return co.min(0), co.max(0), co


def setup_camera(body):
    sc = bpy.context.scene
    mn, mx, _ = world_bbox(body)
    ctr = mu.Vector(((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.52))
    # rear 3/4, similar language to db_row side_tight
    camd = bpy.data.cameras.new("c")
    camd.lens = 50
    cam = bpy.data.objects.new("c", camd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    offset = mu.Vector((1.55, 2.35, 0.55))
    cam.location = ctr + offset
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    return cam, ctr


def clear_lights():
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def add_area(name, loc, look, energy, size=2.4, color=(1, 1, 1)):
    sc = bpy.context.scene
    lt = bpy.data.lights.new(name, "AREA")
    lt.energy = energy
    lt.size = size
    lt.color = color
    o = bpy.data.objects.new(name, lt)
    sc.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    return o


def setup_world(use_hdri, strength=0.55, bg_rgb=(0.0168, 0.0203, 0.0242)):
    sc = bpy.context.scene
    wd = bpy.data.worlds.new("w")
    sc.world = wd
    wd.use_nodes = True
    nt = wd.node_tree
    bg = nt.nodes["Background"]
    if use_hdri and os.path.isfile(HDRI):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(HDRI)
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = strength
    else:
        bg.inputs[0].default_value = (*bg_rgb, 1)
        bg.inputs[1].default_value = 1.0
    # HDRI lights the figure but must not appear as a photographed studio backdrop
    vis = getattr(wd, "cycles_visibility", None)
    if vis is not None and hasattr(vis, "camera"):
        vis.camera = False
    return bg


def find_bsdf(mat):
    return [n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"][0]


def apply_current_look(body):
    """Match retarget_render_v2.py production defaults (albedo_detail=0, rough 0.62, normal 1.5)."""
    for s in body.material_slots:
        nt = s.material.node_tree
        bsdf = find_bsdf(s.material)
        for l in list(bsdf.inputs["Base Color"].links):
            nt.links.remove(l)
        bsdf.inputs["Base Color"].default_value = (0.20, 0.205, 0.215, 1)
        bsdf.inputs["Roughness"].default_value = 0.62
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.0
        for nm in [n for n in nt.nodes if n.type == "NORMAL_MAP"]:
            nm.inputs["Strength"].default_value = 1.5


def apply_upgraded_look(body, albedo_detail=0.50, normal_strength=2.0, sss=0.16):
    """Reuse the GLB's packed 2K albedo + roughness + normal, add SSS, lift clay a little."""
    for s in body.material_slots:
        nt = s.material.node_tree
        bsdf = find_bsdf(s.material)
        for l in list(bsdf.inputs["Base Color"].links):
            nt.links.remove(l)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs[6].default_value = (0.24, 0.248, 0.255, 1)  # lifted charcoal-silver clay, not flesh or Demic white
        mix.inputs[7].default_value = (0.24, 0.248, 0.255, 1)
        tex = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "basecolor" in n.image.name.lower()]
        if tex and albedo_detail > 0:
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            nt.links.new(tex[0].outputs["Color"], bw.inputs[0])
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.15
            mr.inputs["From Max"].default_value = 0.8
            mr.inputs["To Min"].default_value = 1.0 - albedo_detail
            mr.inputs["To Max"].default_value = 1.0 + albedo_detail
            nt.links.new(bw.outputs[0], mr.inputs["Value"])
            mul = nt.nodes.new("ShaderNodeMix")
            mul.data_type = "RGBA"
            mul.blend_type = "MULTIPLY"
            mul.inputs[0].default_value = 1.0
            nt.links.new(mix.outputs[2], mul.inputs[6])
            nt.links.new(mr.outputs[0], mul.inputs[7])
            nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        # keep asset roughness map if present (pipeline currently overwrites to 0.62)
        rtex = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "rough" in n.image.name.lower()]
        if rtex:
            # packed metallic-roughness: roughness often in G; Separate Color already exists in the GLB graph
            seps = [n for n in nt.nodes if n.type == "SEPARATE_COLOR"]
            if seps:
                # don't fight existing links; set a slightly tighter roughness floor
                pass
        bsdf.inputs["Roughness"].default_value = 0.48
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = sss
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (0.55, 0.58, 0.62)
        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = 0.05
        for nm in [n for n in nt.nodes if n.type == "NORMAL_MAP"]:
            nm.inputs["Strength"].default_value = normal_strength


def add_floor(z):
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, z))
    pl = bpy.context.object
    pl.name = "floor"
    mat = bpy.data.materials.new("floor")
    mat.use_nodes = True
    bs = mat.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (0.045, 0.048, 0.052, 1)
    bs.inputs["Roughness"].default_value = 0.85
    pl.data.materials.append(mat)
    return pl


def paint_back_emission(body, cam, strength=1.6):
    """Attribute 'glow' on back-facing torso verts (rest-pose lats/traps stand-in)."""
    bpy.context.view_layer.update()
    mw = np.array(body.matrix_world)
    me = body.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    # normals: use polygon fan average
    nrm = np.zeros((len(me.vertices), 3))
    cnt = np.zeros(len(me.vertices))
    nos = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", nos)
    nrm = nos.reshape(-1, 3) @ mw[:3, :3].T
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
    cam_loc = np.array(cam.location)
    to_cam = cam_loc - co
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-8)
    facing = (nrm * to_cam).sum(1)
    z0, z1 = np.percentile(co[:, 2], [5, 99])
    # torso band: roughly hips to base of neck
    zlo = z0 + 0.58 * (z1 - z0)   # upper back only (lats / mid trap), skip glutes
    zhi = z0 + 0.80 * (z1 - z0)
    xmid = 0.5 * (co[:, 0].min() + co[:, 0].max())
    lat = np.abs(co[:, 0] - xmid)
    glow = ((facing > 0.35) & (co[:, 2] > zlo) & (co[:, 2] < zhi) & (lat < 0.16)).astype(np.float32)
    # feather
    if "glow" in me.attributes:
        me.attributes.remove(me.attributes["glow"])
    attr = me.attributes.new("glow", "FLOAT", "POINT")
    attr.data.foreach_set("value", glow)
    # hook emission into each material
    for s in body.material_slots:
        nt = s.material.node_tree
        out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        bsdf = find_bsdf(s.material)
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = "glow"
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.18, 1.0, 0.42, 1)  # gymform green, not Demic orange
        em.inputs["Strength"].default_value = strength
        # scale emission by the attribute so form (principled) remains
        em_scale = nt.nodes.new("ShaderNodeMath")
        em_scale.operation = "MULTIPLY"
        nt.links.new(at.outputs["Fac"], em_scale.inputs[0])
        em_scale.inputs[1].default_value = 1.0
        # Mix RGB of emission strength: use emission Strength socket via extra math
        str_mul = nt.nodes.new("ShaderNodeMath")
        str_mul.operation = "MULTIPLY"
        str_mul.inputs[1].default_value = strength
        nt.links.new(at.outputs["Fac"], str_mul.inputs[0])
        nt.links.new(str_mul.outputs[0], em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(bsdf.outputs[0], add.inputs[0])
        nt.links.new(em.outputs[0], add.inputs[1])
        for l in list(out.inputs["Surface"].links):
            nt.links.remove(l)
        nt.links.new(add.outputs[0], out.inputs["Surface"])
    return float(glow.mean()), int(glow.sum())


def setup_compositor_bloom():
    sc = bpy.context.scene
    ng = bpy.data.node_groups.new("quality_bloom", "CompositorNodeTree")
    try:
        ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    except Exception:
        pass
    rl = ng.nodes.new("CompositorNodeRLayers")
    glare = ng.nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Threshold"].default_value = 0.75
    glare.inputs["Strength"].default_value = 0.55
    glare.inputs["Size"].default_value = 0.7
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    # Group Output socket name may be Image after interface add
    target = out.inputs[0] if out.inputs else None
    if target is not None:
        ng.links.new(glare.outputs["Image"], target)
    sc.compositing_node_group = ng
    sc.render.use_compositing = True
    sc.use_nodes = True


def render_to(path, spp=SPP, denoise=True, bloom=False):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = spp
    sc.cycles.use_denoising = denoise
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        sc.cycles.denoising_prefilter = "ACCURATE"
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception as e:
        print("denoise option", e)
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = path
    if bloom:
        setup_compositor_bloom()
    else:
        sc.render.use_compositing = False
        sc.compositing_node_group = None
    t = time.time()
    bpy.ops.render.render(write_still=True)
    return round(time.time() - t, 1)


def production_lights(ctr):
    clear_lights()
    key_from = mu.Vector((1.55, 2.35, 0)).normalized()
    fw = mu.Vector((0.0, 1.0, 0.0))
    add_area("key", ctr + key_from * 3.0 + mu.Vector((0, 0, 1.2)), ctr, 2.4 * 150, 3.0)
    add_area("fill", ctr + (-fw * 2.5 + mu.Vector((0, 0, 2.5))) + mu.Vector((0, 0, 1.2)), ctr, 2.2 * 150, 3.0)
    add_area("kick", ctr + (fw * 3 + mu.Vector((0, 0, 1.5))) + mu.Vector((0, 0, 1.2)), ctr, 1.0 * 150, 3.0)


def upgraded_lights(ctr, cam):
    clear_lights()
    key_dir = (cam.location - ctr)
    key_dir.z = 0.35
    key_dir.normalize()
    # stronger key slightly above camera so muscle bellies roll
    add_area("key", ctr + key_dir * 2.8 + mu.Vector((0, 0, 1.6)), ctr, 520, 2.2, (1.0, 0.98, 0.94))
    add_area("fill", ctr + mu.Vector((-1.8, -0.4, 1.8)), ctr, 180, 3.5, (0.92, 0.95, 1.0))
    # dedicated cool rim opposite camera, high — Demic shoulder halo
    rim_dir = -key_dir
    rim_dir.z = 0.15
    rim_dir.normalize()
    add_area("rim", ctr + rim_dir * 2.6 + mu.Vector((0, 0, 1.9)), ctr, 380, 1.6, (0.65, 0.82, 1.0))


def main():
    timings = {}
    # --- BEFORE ---
    arm, body = reset_import()
    cam, ctr = setup_camera(body)
    setup_world(use_hdri=True, strength=0.55)
    apply_current_look(body)
    production_lights(ctr)
    sc = bpy.context.scene
    sc.render.film_transparent = False
    # match grade backdrop-ish so the still is comparable
    # world already has HDRI; add a dark backdrop plane behind? skip — HDRI studio is fine
    p = os.path.join(OUT, "before_current.png")
    timings["before_s"] = render_to(p, bloom=False)
    print("WROTE", p, timings["before_s"], "s")

    # --- AFTER light/SSS ---
    arm, body = reset_import()
    cam, ctr = setup_camera(body)
    setup_world(use_hdri=True, strength=0.35)  # dim HDRI; key+rim do the shaping
    apply_upgraded_look(body)
    upgraded_lights(ctr, cam)
    mn, _, _ = world_bbox(body)
    add_floor(float(mn[2]) + 0.002)
    p = os.path.join(OUT, "after_light_sss.png")
    timings["after_light_s"] = render_to(p, bloom=False)
    print("WROTE", p, timings["after_light_s"], "s")

    # --- AFTER + glow ---
    arm, body = reset_import()
    cam, ctr = setup_camera(body)
    setup_world(use_hdri=True, strength=0.35)
    apply_upgraded_look(body)
    upgraded_lights(ctr, cam)
    mn, _, _ = world_bbox(body)
    add_floor(float(mn[2]) + 0.002)
    frac, n = paint_back_emission(body, cam)
    print("GLOW verts", n, "frac", round(frac, 4))
    p = os.path.join(OUT, "after_light_sss_glow.png")
    timings["after_glow_s"] = render_to(p, bloom=True)
    print("WROTE", p, timings["after_glow_s"], "s")

    print("PROBE_DONE", timings)


if __name__ == "__main__":
    main()
