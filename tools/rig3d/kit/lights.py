"""Studio light rig: low-strength Poly Haven HDRI + three area lights + contact floor."""
from __future__ import annotations


def apply_studio_world(sc, hdri, cfg, log=print):
    import bpy
    import os
    L = (cfg.get("kit") or {}).get("lights") or {}
    wd = sc.world or bpy.data.worlds.new("kit_world")
    sc.world = wd
    wd.use_nodes = True
    nt = wd.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.name = "KIT_HDRI_BG"
    hdri_s = float(L.get("hdri_strength", cfg.get("render", {}).get("hdri_strength", 0.18)))
    path = None
    if isinstance(hdri, dict):
        path = hdri.get("file")
    if path and os.path.isfile(path):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(path, check_existing=True)
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = hdri_s
        log(f"kit world HDRI={path} strength={hdri_s}")
    else:
        amb = float(L.get("ambient_strength", 0.10))
        bg.inputs["Color"].default_value = (0.82, 0.84, 0.87, 1)
        bg.inputs["Strength"].default_value = max(hdri_s, amb)
        log(f"kit world: no HDRI file, ambient {bg.inputs['Strength'].default_value}")
    # camera rays stay charcoal so the backdrop does not photograph the HDRI
    bg_cam = nt.nodes.new("ShaderNodeBackground")
    _cam = 0.0185
    bg_cam.inputs["Color"].default_value = (_cam, _cam, _cam + 0.001, 1)
    bg_cam.inputs["Strength"].default_value = 1.0
    lp = nt.nodes.new("ShaderNodeLightPath")
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs["Fac"])
    nt.links.new(bg.outputs["Background"], mx.inputs[1])
    nt.links.new(bg_cam.outputs["Background"], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    return wd


def add_three_point(col, ctr, cam_loc, cfg, log=print):
    import bpy
    import mathutils as mu
    L = (cfg.get("kit") or {}).get("lights") or {}
    RC = cfg.get("render") or {}
    key_from = (cam_loc - ctr)
    key_from.z = 0
    if key_from.length < 1e-4:
        key_from = mu.Vector((1.0, 0.4, 0.0))
    key_from.normalize()
    upv = mu.Vector((0.0, 0.0, 1.0))
    sidev = key_from.cross(upv)
    sidev = (sidev if sidev.length > 1e-4 else mu.Vector((1.0, 0.0, 0.0))).normalized()

    def add_area(name, loc, energy, size, color):
        if energy <= 1e-3:
            return None
        Lt = bpy.data.lights.new(name, "AREA")
        Lt.energy = float(energy)
        Lt.size = float(size)
        Lt.color = color
        o = bpy.data.objects.new(name, Lt)
        col.objects.link(o)
        o.location = loc
        o.rotation_euler = (ctr - o.location).to_track_quat("-Z", "Y").to_euler()
        return o

    key_e = float(L.get("key_energy", RC.get("key_energy", 42)))
    fill_e = float(L.get("fill_energy", RC.get("fill_energy", 28)))
    rim_e = float(L.get("rim_energy", RC.get("rim_energy", 16)))
    # Key nearly overhead-and-side (soft, not a grazing hot rim). A weak
    # camera-side fill evens the working forearm without blowing it out.
    add_area("key", ctr - key_from * 0.40 + upv * 3.90 + sidev * 2.80,
             key_e, float(L.get("key_size", 6.0)), (1.00, 0.99, 0.97))
    add_area("fill", ctr + key_from * 0.05 + upv * 1.90 - sidev * 3.20,
             fill_e, float(L.get("fill_size", 6.2)), (0.98, 0.99, 1.00))
    add_area("rim", ctr - key_from * 2.2 + upv * 1.55 + sidev * 0.4,
             rim_e, float(L.get("rim_size", 2.8)), (0.95, 0.97, 1.00))
    log(f"kit three-point key={key_e} fill={fill_e} rim={rim_e}")


def shade_contact_floor(floor, cfg, log=print):
    """Replace an emission cyclorama with a shadow-receiving dark BSDF."""
    import bpy
    if floor is None:
        return
    L = (cfg.get("kit") or {}).get("lights") or {}
    if not L.get("floor_receive_shadows", True):
        return
    fm = floor.data.materials[0] if floor.data.materials else bpy.data.materials.new("studio_floor")
    fm.use_nodes = True
    nt = fm.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    c = 0.022
    bs.inputs["Base Color"].default_value = (c, c, c + 0.002, 1)
    bs.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bs.inputs:
        bs.inputs["Specular IOR Level"].default_value = 0.18
    nt.links.new(bs.outputs[0], out.inputs["Surface"])
    if not floor.data.materials:
        floor.data.materials.append(fm)
    log("kit floor: principled charcoal (receives contact shadows)")


# back-compat alias
apply_studio_lights = apply_studio_world
