import os
import math
import time
import subprocess
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_DIR = "/workspace/daily_info_reels/frames_demic"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. BLENDER 5.2 SCENE & MATERIAL BUILDERS
# ---------------------------------------------------------

def clear_scene(res_x=540, res_y=960, samples=8):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = samples
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    
    # Use Standard view transform so neon emission colors (#FF2A00, #FFE600, #00FF66) stay ultra-vibrant!
    try:
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'Medium High Contrast'
    except Exception:
        pass
    
    world = bpy.data.worlds.new("DemicWorld")
    scene.world = world
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.014, 0.020, 0.028, 1.0)
        bg.inputs[1].default_value = 0.85
    return scene

def make_mat(name, base_color=(0.78, 0.81, 0.85, 1.0), metallic=0.04, roughness=0.38,
             subsurface=0.12, coat=0.18, emission_color=None, emission_strength=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Subsurface Weight"].default_value = subsurface
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.22
    if emission_color is not None and emission_strength > 0:
        bsdf.inputs["Emission Color"].default_value = emission_color
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        try:
            mat.surface_render_method = 'BLENDED'
        except Exception:
            pass
    return mat

def get_standard_materials():
    return {
        "skin": make_mat("DemicGreySkin", base_color=(0.77, 0.80, 0.84, 1.0), metallic=0.05, roughness=0.36, subsurface=0.15, coat=0.22),
        "skin_xray": make_mat("DemicXraySkin", base_color=(0.68, 0.73, 0.79, 1.0), metallic=0.02, roughness=0.25, subsurface=0.05, coat=0.1, alpha=0.26),
        "shorts": make_mat("DemicShorts", base_color=(0.045, 0.052, 0.065, 1.0), metallic=0.05, roughness=0.68, subsurface=0.0, coat=0.05),
        "bone_norm": make_mat("BoneNormal", base_color=(0.88, 0.86, 0.80, 1.0), metallic=0.0, roughness=0.35, emission_color=(0.85, 0.85, 0.80, 1.0), emission_strength=0.35),
        "red_glow": make_mat("NeonRed", base_color=(1.0, 0.12, 0.02, 1.0), roughness=0.2, emission_color=(1.0, 0.14, 0.0, 1.0), emission_strength=6.5),
        "orange_glow": make_mat("NeonOrange", base_color=(1.0, 0.38, 0.02, 1.0), roughness=0.2, emission_color=(1.0, 0.36, 0.0, 1.0), emission_strength=5.2),
        "yellow_glow": make_mat("NeonYellow", base_color=(1.0, 0.88, 0.05, 1.0), roughness=0.2, emission_color=(1.0, 0.88, 0.0, 1.0), emission_strength=5.5),
        "green_glow": make_mat("NeonGreen", base_color=(0.05, 1.0, 0.38, 1.0), roughness=0.2, emission_color=(0.0, 1.0, 0.38, 1.0), emission_strength=5.5),
        "magenta_glow": make_mat("NeonMagenta", base_color=(1.0, 0.05, 0.65, 1.0), roughness=0.2, emission_color=(1.0, 0.05, 0.65, 1.0), emission_strength=5.5),
    }

def add_ellipsoid(name, loc, scale, rot_deg=(0,0,0), mat=None, subdiv=1):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18, radius=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = tuple(math.radians(a) for a in rot_deg)
    bpy.ops.object.shade_smooth()
    if subdiv > 0:
        mod = obj.modifiers.new("Subd", type='SUBSURF')
        mod.levels = subdiv
        mod.render_levels = subdiv
    if mat:
        obj.data.materials.append(mat)
    return obj

def add_oriented_ellipsoid(name, p1, p2, rx, ry, mat=None, scale_z_mult=0.55, offset_local=(0,0,0)):
    v1, v2 = Vector(p1), Vector(p2)
    diff = v2 - v1
    length = diff.length
    if length < 1e-5:
        return None
    mid = (v1 + v2) * 0.5
    quat = Vector((0, 0, 1)).rotation_difference(diff.normalized())
    world_offset = quat @ Vector(offset_local)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18, radius=1.0, location=mid + world_offset)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (rx, ry, length * scale_z_mult)
    obj.rotation_euler = quat.to_euler()
    bpy.ops.object.shade_smooth()
    if mat:
        obj.data.materials.append(mat)
    return obj

def add_cylinder_between(name, p1, p2, r, mat=None):
    v1, v2 = Vector(p1), Vector(p2)
    diff = v2 - v1
    length = diff.length
    if length < 1e-5:
        return None
    mid = (v1 + v2) * 0.5
    quat = Vector((0, 0, 1)).rotation_difference(diff.normalized())
    bpy.ops.mesh.primitive_cylinder_add(vertices=22, radius=r, depth=length, location=mid)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = quat.to_euler()
    bpy.ops.object.shade_smooth()
    if mat:
        obj.data.materials.append(mat)
    return obj

def add_point_glow_light(name, loc, color=(1.0, 0.2, 0.05), energy=25.0, radius=0.15):
    ld = bpy.data.lights.new(name=name, type='POINT')
    ld.energy = energy
    ld.color = color
    ld.shadow_soft_size = radius
    lo = bpy.data.objects.new(name=name, object_data=ld)
    bpy.context.collection.objects.link(lo)
    lo.location = loc
    return lo

def setup_studio_environment(cam_loc, cam_target, focal_length=55):
    mat_floor = make_mat("StudioFloor", base_color=(0.024, 0.032, 0.044, 1.0), metallic=0.45, roughness=0.16, subsurface=0.0, coat=0.6)
    bpy.ops.mesh.primitive_plane_add(size=35.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.data.materials.append(mat_floor)

    # Subtle reflective floor grid strips for 3D depth perception
    mat_grid = make_mat("FloorGrid", base_color=(0.045, 0.060, 0.080, 1.0), metallic=0.6, roughness=0.2, subsurface=0.0)
    for gx in [-1.8, -0.9, 0.0, 0.9, 1.8]:
        add_cylinder_between(f"GridX_{gx}", (gx, -4.0, 0.002), (gx, 5.0, 0.002), 0.005, mat_grid)
    for gy in [-1.5, -0.5, 0.5, 1.5, 2.5]:
        add_cylinder_between(f"GridY_{gy}", (-4.0, gy, 0.002), (4.0, gy, 0.002), 0.005, mat_grid)

    mat_wall = make_mat("StudioWall", base_color=(0.018, 0.025, 0.035, 1.0), metallic=0.2, roughness=0.45, subsurface=0.0)
    bpy.ops.mesh.primitive_plane_add(size=35.0, location=(0, 4.2, 5.0), rotation=(math.radians(90), 0, 0))
    wall = bpy.context.active_object
    wall.data.materials.append(mat_wall)

    mat_metal = make_mat("DarkMetal", base_color=(0.055, 0.068, 0.085, 1.0), metallic=0.85, roughness=0.25, subsurface=0.0)
    for x_pos in [-2.2, 2.2]:
        add_cylinder_between(f"RackPillar_{x_pos}", (x_pos, 3.0, 0), (x_pos, 3.0, 3.5), 0.055, mat_metal)
    add_cylinder_between("RackCrossbar", (-2.2, 3.0, 2.5), (2.2, 3.0, 2.5), 0.04, mat_metal)
    add_cylinder_between("RackShelf1", (-2.2, 2.8, 0.95), (2.2, 2.8, 0.95), 0.035, mat_metal)
    # Background dumbbells on rack
    for dx in [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]:
        add_ellipsoid(f"DB_{dx}", (dx, 2.8, 1.03), (0.09, 0.14, 0.09), mat=mat_metal, subdiv=0)

    def add_area_light(name, loc, target, energy, color, size=2.4):
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = energy
        light_data.color = color
        light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc
        direction = Vector(target) - Vector(loc)
        light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        return light_obj

    # Dramatic Rim Lights + Studio Key & Fill
    add_area_light("RimLight_L", (-2.4, 2.0, 2.5), cam_target, 420.0, (0.85, 0.95, 1.0), size=2.0)
    add_area_light("RimLight_R", (2.4, 2.0, 2.5), cam_target, 420.0, (0.90, 0.97, 1.0), size=2.0)
    add_area_light("TopRim", (0.0, 1.2, 3.4), cam_target, 260.0, (0.92, 0.96, 1.0), size=2.2)
    add_area_light("KeyLight", (-1.8, -2.6, 2.4), cam_target, 220.0, (0.98, 0.98, 1.0), size=2.6)
    add_area_light("FillLight", (2.0, -2.2, 1.4), cam_target, 105.0, (0.65, 0.84, 0.98), size=2.8)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = cam_loc
    direction = Vector(cam_target) - Vector(cam_loc)
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return cam_obj

def add_gym_bench(center_x=0.0, center_y=0.0, top_z=0.48, length=0.95, width=0.38):
    mat_pad = make_mat(f"BenchPad_{center_x}", base_color=(0.05, 0.058, 0.072, 1.0), metallic=0.12, roughness=0.42, subsurface=0.0, coat=0.25)
    mat_frame = make_mat(f"BenchFrame_{center_x}", base_color=(0.13, 0.15, 0.18, 1.0), metallic=0.85, roughness=0.24, subsurface=0.0)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(center_x, center_y, top_z - 0.045))
    pad = bpy.context.active_object
    pad.scale = (width*0.5, length*0.5, 0.045)
    bv = pad.modifiers.new("Bevel", type='BEVEL')
    bv.width = 0.02
    bv.segments = 4
    bpy.ops.object.shade_smooth()
    pad.data.materials.append(mat_pad)
    for dy in [-length*0.36, length*0.36]:
        add_cylinder_between(f"BenchLeg_{center_x}_{dy}", (center_x, center_y+dy, 0.0), (center_x, center_y+dy, top_z-0.08), 0.034, mat_frame)
        add_cylinder_between(f"BenchFoot_{center_x}_{dy}", (center_x-0.20, center_y+dy, 0.02), (center_x+0.20, center_y+dy, 0.02), 0.026, mat_frame)
    add_cylinder_between(f"BenchSpine_{center_x}", (center_x, center_y-length*0.38, top_z-0.1), (center_x, center_y+length*0.38, top_z-0.1), 0.03, mat_frame)

def project_3d_point(scene, cam_obj, pt3d):
    co_2d = world_to_camera_view(scene, cam_obj, Vector(pt3d))
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    px = int(co_2d.x * w)
    py = int((1.0 - co_2d.y) * h)
    return (px, py)

print("Core module syntax verified")

# ---------------------------------------------------------
# 2. SCULPTED 3D ANATOMICAL MANNEQUIN BUILDERS
# ---------------------------------------------------------

def build_mannequin_sample_a(mats, t_correct=0.0):
    """
    Seated 3D muscular light-grey mannequin on gym bench (side-3/4 view).
    t_correct = 0.0 -> Forward Head Posture (Tech Neck):
      - Translucent neck skin showing glowing Red/Orange X-ray cervical spine vertebrae (C1-C7)
      - Glowing Red/Orange overloaded upper trapezius & levator scapulae
      - Forward craniovertebral angle
    t_correct = 1.0 -> Chin-Tuck Decompression Posture:
      - Aligned cervical spine (healthy glowing green/white alignment)
      - Glowing Neon Green deep cervical flexors + glowing Yellow/Green lower trapezius & rhomboids
    """
    skin = mats["skin"]
    skin_xray = mats["skin_xray"]
    shorts = mats["shorts"]

    # Interpolate posture parameters from Wrong (0.0) to Correct (1.0)
    # Forward direction of character is -X (facing left-front towards camera)
    hip = Vector((0.10, 0.0, 0.54))
    slouch_x = (1.0 - t_correct) * (-0.08)
    chest_lift_z = t_correct * 0.05
    lumbar = hip + Vector((0.02 + slouch_x*0.3, 0.0, 0.22))
    thoracic = lumbar + Vector((slouch_x*0.7, 0.0, 0.26 + chest_lift_z))
    c7 = thoracic + Vector((-0.02 + slouch_x*0.5, 0.0, 0.17))

    # Forward head shift when t_correct=0
    head_fwd_x = -0.22 * (1.0 - t_correct) - 0.03 * t_correct
    head_drop_z = 0.16 * (1.0 - t_correct) + 0.23 * t_correct
    c1 = c7 + Vector((head_fwd_x, 0.0, head_drop_z))
    head_center = c1 + Vector((-0.03*(1.0-t_correct), 0.0, 0.11))

    # 1. Bench
    add_gym_bench(center_x=0.12, center_y=0.0, top_z=0.48, length=0.85, width=0.42)

    # 2. Lower Body (Dark Charcoal Athletic Shorts + Muscular Quads & Calves)
    add_ellipsoid("PelvisShorts", hip, (0.15, 0.19, 0.12), mat=shorts)
    for side, sy in [("L", -0.11), ("R", 0.11)]:
        hip_j = hip + Vector((-0.02, sy, -0.02))
        knee_j = Vector((-0.34, sy*1.15, 0.49))
        ankle_j = Vector((-0.38, sy*1.15, 0.08))
        foot_tip = Vector((-0.54, sy*1.18, 0.04))
        # Shorts thigh sleeve
        mid_short = hip_j.lerp(knee_j, 0.48)
        add_oriented_ellipsoid(f"ShortLeg_{side}", hip_j, mid_short, 0.105, 0.10, mat=shorts, scale_z_mult=0.62)
        # Distal Quad & Knee
        add_oriented_ellipsoid(f"QuadDist_{side}", mid_short, knee_j, 0.088, 0.082, mat=skin, scale_z_mult=0.60)
        add_ellipsoid(f"Knee_{side}", knee_j, (0.068, 0.065, 0.068), mat=skin)
        # Calf (Gastrocnemius + Soleus)
        add_oriented_ellipsoid(f"Calf_{side}", knee_j, ankle_j, 0.072, 0.068, mat=skin, scale_z_mult=0.56, offset_local=(0.015, 0, 0.04))
        add_cylinder_between(f"Shin_{side}", knee_j, ankle_j, 0.048, mat=skin)
        add_ellipsoid(f"Ankle_{side}", ankle_j, (0.048, 0.048, 0.048), mat=skin)
        add_oriented_ellipsoid(f"Foot_{side}", ankle_j, foot_tip, 0.055, 0.048, mat=skin, scale_z_mult=0.62)

    # 3. Sculpted Torso (Lumbar, Obliques, Ribcage, Pecs, Lats)
    add_oriented_ellipsoid("Abdomen", hip, lumbar, 0.135, 0.165, mat=skin, scale_z_mult=0.68)
    add_oriented_ellipsoid("Ribcage", lumbar, thoracic, 0.16, 0.21, mat=skin, scale_z_mult=0.68)
    add_oriented_ellipsoid("UpperTorso", thoracic, c7, 0.15, 0.23, mat=skin, scale_z_mult=0.65)

    # Sculpted Pectorals (Front is -X)
    for side, sy in [("L", -0.085), ("R", 0.085)]:
        pec_loc = thoracic + Vector((-0.11, sy, 0.04))
        add_ellipsoid(f"Pec_{side}", pec_loc, (0.065, 0.085, 0.072), rot_deg=(0, 18, 0), mat=skin)
        lat_loc = thoracic + Vector((0.05, sy*1.45, 0.01))
        add_ellipsoid(f"Lat_{side}", lat_loc, (0.085, 0.065, 0.13), rot_deg=(0, -12, 0), mat=skin)

    # 4. Translucent Neck + Internal X-Ray Cervical Vertebrae (C1-C7) & Glowing Muscles!
    add_oriented_ellipsoid("NeckTranslucent", c7, c1, 0.072, 0.076, mat=skin_xray, scale_z_mult=0.58)

    # Decide shader for spine & muscles based on t_correct
    if t_correct < 0.45:
        mat_spine = mats["red_glow"]
        mat_upper_trap = mats["orange_glow"]
        mat_deep_neck = skin
        mat_mid_back = skin
        add_point_glow_light("CervicalStressLight", c7.lerp(c1, 0.5) + Vector((-0.05, -0.15, 0)), color=(1.0, 0.18, 0.02), energy=28.0)
    elif t_correct > 0.55:
        mat_spine = mats["green_glow"]
        mat_upper_trap = skin
        mat_deep_neck = mats["green_glow"]
        mat_mid_back = mats["yellow_glow"]
        add_point_glow_light("CervicalHealLight", c7.lerp(c1, 0.5) + Vector((-0.08, -0.15, 0)), color=(0.05, 1.0, 0.42), energy=28.0)
        add_point_glow_light("MidBackHealLight", thoracic + Vector((0.12, -0.15, 0.05)), color=(0.95, 0.90, 0.05), energy=24.0)
    else:
        mat_spine = mats["orange_glow"]
        mat_upper_trap = mats["yellow_glow"]
        mat_deep_neck = mats["yellow_glow"]
        mat_mid_back = mats["yellow_glow"]

    # 7 Sculpted Cervical Vertebrae + Spinous Processes inside the neck (X-Ray effect)
    for i in range(7):
        frac = (i + 0.5) / 7.0
        v_pos = c7.lerp(c1, frac)
        # Slight lordotic/kyphotic curve
        curve_x = -0.018 * math.sin(frac * math.pi) * (1.0 - t_correct)
        v_pos = v_pos + Vector((curve_x, 0.0, 0.0))
        add_ellipsoid(f"Vertebra_C{7-i}", v_pos, (0.032, 0.036, 0.014), mat=mat_spine, subdiv=0)
        # Spinous process protruding posteriorly (+X)
        add_ellipsoid(f"Spinous_C{7-i}", v_pos + Vector((0.032, 0.0, -0.004)), (0.024, 0.012, 0.009), rot_deg=(0, 20, 0), mat=mat_spine, subdiv=0)

    # Upper Trapezius / Levator Scapulae (glows RED/ORANGE in wrong posture)
    for side, sy in [("L", -0.09), ("R", 0.09)]:
        p_neck = c7.lerp(c1, 0.55) + Vector((0.03, sy*0.45, 0.0))
        p_shld = c7 + Vector((0.02, sy*1.9, -0.04))
        add_oriented_ellipsoid(f"UpperTrap_{side}", p_neck, p_shld, 0.048, 0.058, mat=mat_upper_trap, scale_z_mult=0.62)

    # Deep Cervical Flexors (Anterior neck — glows GREEN in correct posture)
    dcf_p1 = c7 + Vector((-0.042, -0.025, 0.02))
    dcf_p2 = c1 + Vector((-0.042, -0.020, -0.02))
    if t_correct > 0.45:
        add_oriented_ellipsoid("DeepNeckFlexor_L", dcf_p1, dcf_p2, 0.032, 0.038, mat=mat_deep_neck, scale_z_mult=0.58)
        # Lower Trapezius / Rhomboids (Mid-back — glows YELLOW/GREEN in correct posture)
        for side, sy in [("L", -0.08), ("R", 0.08)]:
            rh_p1 = c7 + Vector((0.07, sy*0.4, -0.03))
            rh_p2 = thoracic + Vector((0.09, sy*1.3, -0.08))
            add_oriented_ellipsoid(f"LowerTrap_{side}", rh_p1, rh_p2, 0.045, 0.055, mat=mat_mid_back, scale_z_mult=0.65)

    # 5. Faceless Smooth Sculpted Cranium & Jaw
    head_pitch = 18.0 * (1.0 - t_correct) - 2.0 * t_correct
    add_ellipsoid("Cranium", head_center, (0.118, 0.096, 0.132), rot_deg=(0, -head_pitch, 0), mat=skin, subdiv=2)
    jaw_loc = head_center + Vector((-0.045 - 0.02*(1.0-t_correct), 0.0, -0.065))
    add_ellipsoid("Jawline", jaw_loc, (0.082, 0.074, 0.068), rot_deg=(0, -head_pitch*0.8, 0), mat=skin, subdiv=1)

    # 6. Shoulders & Arms (Resting hands on thighs)
    for side, sy in [("L", -0.22), ("R", 0.22)]:
        shld = c7 + Vector((0.01 - 0.04*(1.0-t_correct), sy, -0.05))
        elbow = thoracic + Vector((0.04, sy*1.12, -0.10))
        wrist = Vector((-0.22, sy*0.82, 0.58))
        hand = Vector((-0.30, sy*0.75, 0.56))
        add_ellipsoid(f"Deltoid_{side}", shld, (0.082, 0.078, 0.088), mat=skin)
        add_oriented_ellipsoid(f"UpperArm_{side}", shld, elbow, 0.062, 0.058, mat=skin, scale_z_mult=0.58)
        add_ellipsoid(f"Elbow_{side}", elbow, (0.050, 0.050, 0.050), mat=skin)
        add_oriented_ellipsoid(f"Forearm_{side}", elbow, wrist, 0.052, 0.046, mat=skin, scale_z_mult=0.58)
        add_ellipsoid(f"Hand_{side}", hand, (0.058, 0.042, 0.028), rot_deg=(0, 15, 0), mat=skin)

    # Return key 3D landmarks for 2D/3D joint markers (`o—o—o`) and arrows
    ear_pt = head_center + Vector((-0.02, -0.095, -0.01))
    c7_pt = c7 + Vector((0.0, -0.10, 0.0))
    shld_pt = c7 + Vector((0.01 - 0.04*(1.0-t_correct), -0.22, -0.05))
    hip_pt = lumbar + Vector((0.0, -0.16, 0.0))
    return {
        "joint_chain": [ear_pt, c7_pt, shld_pt, hip_pt],
        "head_front": head_center + Vector((-0.14, -0.05, -0.02)),
        "head_back": head_center + Vector((0.12, -0.05, -0.02)),
        "mid_back": thoracic + Vector((0.14, -0.08, -0.02)),
    }


def build_mannequin_sample_b(mats, is_correct=False):
    """
    Sample B (UFC Fighter Lead Left Hook Biomechanics):
    Standing athletic 3D grey muscular fighter mannequin.
    - Wrong (is_correct=False): Arm-only wide swing, static flat feet, upright un-rotated torso,
      glowing RED/ORANGE anterior deltoid & shoulder joint capsule + elbow strain.
    - Right (is_correct=True): Kinetic chain lead hook — lead foot/hip pivot, glowing YELLOW/GREEN
      external obliques, serratus anterior, latissimus dorsi, and gluteus medius, compact 90-deg horizontal elbow.
    """
    skin = mats["skin"]
    skin_xray = mats["skin_xray"]
    shorts = mats["shorts"]

    rot_z = 28.0 if is_correct else 0.0
    rot_rad = math.radians(rot_z)
    def rot_vec(v):
        x, y, z = v
        return Vector((x*math.cos(rot_rad) - y*math.sin(rot_rad),
                       x*math.sin(rot_rad) + y*math.cos(rot_rad), z))

    pelvis = Vector((0.0, 0.0, 0.96))
    lumbar = pelvis + Vector((0.0, 0.0, 0.22))
    thoracic = lumbar + rot_vec((0.0, -0.02, 0.25))
    c7 = thoracic + rot_vec((0.0, -0.02, 0.18))
    head_c = c7 + Vector((0.02, -0.05, 0.24))

    # 1. Athletic MMA Shorts & Pelvis
    add_ellipsoid("FighterShorts", pelvis, (0.18, 0.15, 0.14), rot_deg=(0, 0, rot_z*0.7), mat=shorts)

    # 2. Legs (Orthodox/Lead stance)
    # Lead leg (Left = -X side closer to camera)
    l_hip = pelvis + rot_vec((-0.12, -0.04, -0.04))
    l_knee = Vector((-0.22, -0.22, 0.52))
    l_ankle = Vector((-0.24, -0.25, 0.09))
    l_foot = Vector((-0.18 if is_correct else -0.26, -0.42, 0.04)) # pivoted inward when correct!

    # Rear leg (Right = +X side)
    r_hip = pelvis + rot_vec((0.12, 0.05, -0.04))
    r_knee = Vector((0.20, 0.18, 0.52))
    r_ankle = Vector((0.24, 0.24, 0.09))
    r_foot = Vector((0.26, 0.10, 0.04))

    for side, hp, kn, ak, ft in [("L", l_hip, l_knee, l_ankle, l_foot), ("R", r_hip, r_knee, r_ankle, r_foot)]:
        mid_s = hp.lerp(kn, 0.50)
        add_oriented_ellipsoid(f"ShortThigh_{side}", hp, mid_s, 0.11, 0.105, mat=shorts, scale_z_mult=0.62)
        add_oriented_ellipsoid(f"Quad_{side}", mid_s, kn, 0.092, 0.088, mat=skin, scale_z_mult=0.60)
        add_ellipsoid(f"Knee_{side}", kn, (0.068, 0.068, 0.068), mat=skin)
        add_oriented_ellipsoid(f"Calf_{side}", kn, ak, 0.072, 0.068, mat=skin, scale_z_mult=0.56)
        add_ellipsoid(f"Ankle_{side}", ak, (0.050, 0.050, 0.050), mat=skin)
        add_oriented_ellipsoid(f"Foot_{side}", ak, ft, 0.055, 0.048, mat=skin, scale_z_mult=0.65)

    # 3. Muscular Torso + Glowing Kinetic Chain Muscles
    add_oriented_ellipsoid("TorsoLower", pelvis, lumbar, 0.155, 0.130, mat=skin, scale_z_mult=0.68)
    add_oriented_ellipsoid("TorsoMid", lumbar, thoracic, 0.195, 0.150, mat=skin, scale_z_mult=0.68)
    add_oriented_ellipsoid("TorsoUpper", thoracic, c7, 0.215, 0.145, mat=skin, scale_z_mult=0.65)

    # Pecs & Abs
    for side, sx in [("L", -0.09), ("R", 0.09)]:
        add_ellipsoid(f"Pec_{side}", thoracic + rot_vec((sx, -0.11, 0.05)), (0.085, 0.065, 0.075), rot_deg=(0, 0, rot_z), mat=skin)
    for row in range(3):
        for side, sx in [("L", -0.045), ("R", 0.045)]:
            ab_pos = lumbar + rot_vec((sx, -0.125, -0.04 + row*0.075))
            add_ellipsoid(f"Ab_{row}_{side}", ab_pos, (0.042, 0.032, 0.036), rot_deg=(0, 0, rot_z*0.6), mat=skin)

    # Glowing Core / Lat / Glute muscles when is_correct=True
    if is_correct:
        # Glowing Yellow/Green Obliques & Serratus & Lat on Lead Side
        add_ellipsoid("GlowOblique_L", lumbar + rot_vec((-0.135, -0.06, 0.04)), (0.055, 0.095, 0.125), rot_deg=(12, -15, rot_z), mat=mats["green_glow"])
        add_ellipsoid("GlowOblique_R", lumbar + rot_vec((0.125, -0.07, 0.04)), (0.050, 0.085, 0.115), rot_deg=(-10, 15, rot_z), mat=mats["yellow_glow"])
        add_ellipsoid("GlowLat_L", thoracic + rot_vec((-0.175, 0.02, 0.02)), (0.065, 0.095, 0.145), rot_deg=(0, -18, rot_z), mat=mats["yellow_glow"])
        add_ellipsoid("GlowGlute_L", pelvis + rot_vec((-0.155, 0.03, 0.01)), (0.068, 0.095, 0.095), rot_deg=(0, 0, rot_z*0.7), mat=mats["green_glow"])
        add_point_glow_light("CoreGlowLight", lumbar + Vector((-0.22, -0.25, 0.08)), color=(0.2, 1.0, 0.35), energy=32.0)
    else:
        # Normal non-glowing obliques & lats
        add_ellipsoid("Oblique_L", lumbar + Vector((-0.13, -0.04, 0.04)), (0.045, 0.085, 0.11), mat=skin)
        add_ellipsoid("Lat_L", thoracic + Vector((-0.165, 0.02, 0.02)), (0.055, 0.085, 0.135), mat=skin)

    # Neck & Head
    add_oriented_ellipsoid("Neck", c7, head_c - Vector((0,0,0.08)), 0.068, 0.068, mat=skin, scale_z_mult=0.65)
    add_ellipsoid("Head", head_c, (0.102, 0.118, 0.135), rot_deg=(8, 0, 10), mat=skin, subdiv=2)

    # Rear Right Guard Arm (+X)
    r_shld = c7 + rot_vec((0.22, 0.0, -0.04))
    r_elbow = r_shld + Vector((0.06, -0.14, -0.24))
    r_fist = head_c + Vector((0.10, -0.18, -0.08))
    add_ellipsoid("Deltoid_R", r_shld, (0.082, 0.082, 0.088), mat=skin)
    add_oriented_ellipsoid("UpperArm_R", r_shld, r_elbow, 0.062, 0.060, mat=skin, scale_z_mult=0.58)
    add_oriented_ellipsoid("Forearm_R", r_elbow, r_fist, 0.052, 0.048, mat=skin, scale_z_mult=0.58)
    add_ellipsoid("Fist_R", r_fist, (0.058, 0.065, 0.052), mat=skin)

    # Lead Left Hook Arm (-X)
    l_shld = c7 + rot_vec((-0.23, -0.02, -0.04))
    if not is_correct:
        # WRONG: Wide over-extended arm swing behind shoulder line, dropped elbow -> RED X-RAY SHOULDER OVERLOAD
        l_elbow = l_shld + Vector((-0.26, 0.08, -0.14))
        l_fist = l_elbow + Vector((-0.12, -0.32, 0.05))
        # Translucent shoulder capsule with glowing Red X-ray humeral head & overloaded anterior deltoid
        add_ellipsoid("Deltoid_L_Xray", l_shld, (0.095, 0.092, 0.100), mat=skin_xray)
        add_ellipsoid("HumeralHead_Red", l_shld, (0.062, 0.062, 0.062), mat=mats["red_glow"])
        add_oriented_ellipsoid("StrainBicep_L", l_shld, l_elbow, 0.068, 0.065, mat=mats["orange_glow"], scale_z_mult=0.58)
        add_ellipsoid("ElbowStrain_L", l_elbow, (0.058, 0.058, 0.058), mat=mats["red_glow"])
        add_oriented_ellipsoid("Forearm_L", l_elbow, l_fist, 0.054, 0.050, mat=skin, scale_z_mult=0.58)
        add_ellipsoid("Fist_L", l_fist, (0.062, 0.068, 0.055), mat=skin)
        add_point_glow_light("ShoulderOverloadLight", l_shld + Vector((-0.08, -0.15, 0.0)), color=(1.0, 0.15, 0.02), energy=34.0)
    else:
        # CORRECT: Compact 90-degree horizontal lead hook powered by core/hip rotation
        l_elbow = l_shld + Vector((-0.18, -0.18, -0.02))
        l_fist = Vector((0.02, -0.44, l_shld.z - 0.01))
        add_ellipsoid("Deltoid_L_Active", l_shld, (0.088, 0.086, 0.092), mat=mats["yellow_glow"])
        add_oriented_ellipsoid("UpperArm_L", l_shld, l_elbow, 0.066, 0.064, mat=skin, scale_z_mult=0.58)
        add_ellipsoid("Elbow_L", l_elbow, (0.054, 0.054, 0.054), mat=skin)
        add_oriented_ellipsoid("Forearm_L", l_elbow, l_fist, 0.056, 0.052, mat=skin, scale_z_mult=0.58)
        add_ellipsoid("Fist_L", l_fist, (0.064, 0.070, 0.056), mat=skin)

    return {
        "arm_chain": [l_shld, l_elbow, l_fist],
        "core_arc_start": lumbar + Vector((-0.20, -0.12, 0.0)),
        "core_arc_end": lumbar + Vector((0.08, -0.26, 0.04)),
        "foot_pivot": l_foot,
    }


def build_mannequin_sample_c(mats, is_correct=False):
    """
    Sample C (Daily Body Hack — Instant Calf Cramp Release & Tibia/Gastrocnemius X-Ray):
    Close-up 3D grey mannequin lower leg & foot on dark gym bench (directly echoing @demicstory's lower-leg X-ray thumbnail!).
    - Wrong (is_correct=False): Pointed toes (plantarflexion) -> X-ray glowing RED tibia/fibula & violently knotted RED/ORANGE gastrocnemius calf cramp.
    - Right (is_correct=True): Dorsiflexion (pulling toes back toward shin with hand/force arrow) -> glowing GREEN lengthened gastrocnemius & soleus + healthy bone alignment.
    """
    skin = mats["skin"]
    skin_xray = mats["skin_xray"]

    add_gym_bench(center_x=0.0, center_y=0.0, top_z=0.38, length=1.20, width=0.46)

    # Leg geometry in side profile (X = horizontal along bench, Z = vertical)
    hip_anchor = Vector((0.52, 0.0, 0.62))
    knee = Vector((0.24, 0.0, 0.78))
    ankle = Vector((-0.14, 0.0, 0.50))
    heel = Vector((-0.18, 0.0, 0.44))

    if not is_correct:
        # Plantarflexed (toes pointed tightly down/away -> severe calf cramp)
        toe_mid = Vector((-0.36, 0.0, 0.38))
        toe_tip = Vector((-0.44, 0.0, 0.35))
    else:
        # Dorsiflexed (toes pulled up/back toward knee -> instant reciprocal inhibition & calf stretch)
        toe_mid = Vector((-0.26, 0.0, 0.62))
        toe_tip = Vector((-0.24, 0.0, 0.71))

    # Distal thigh leading into knee
    add_oriented_ellipsoid("ThighSeg", hip_anchor, knee, 0.105, 0.098, mat=skin, scale_z_mult=0.60)
    add_ellipsoid("KneeJoint", knee, (0.078, 0.074, 0.078), mat=skin)

    # Translucent Lower Leg Outer Envelope so Internal X-Ray Tibia/Fibula & Glowing Calf Muscle shine through!
    add_oriented_ellipsoid("LowerLegEnvelope", knee, ankle, 0.078, 0.074, mat=skin_xray, scale_z_mult=0.58)

    # Internal Tibia & Fibula Bones
    mat_bone = mats["red_glow"] if not is_correct else mats["bone_norm"]
    add_cylinder_between("TibiaBone", knee + Vector((-0.01, -0.01, -0.03)), ankle + Vector((0.0, -0.01, 0.03)), 0.022, mat=mat_bone)
    add_cylinder_between("FibulaBone", knee + Vector((0.02, 0.02, -0.05)), ankle + Vector((0.02, 0.02, 0.02)), 0.013, mat=mat_bone)

    # Sculpted Gastrocnemius & Soleus Muscle Bellies (Posterior lower leg)
    calf_p1 = knee.lerp(ankle, 0.14) + Vector((0.035, 0.0, -0.04))
    calf_p2 = knee.lerp(ankle, 0.62) + Vector((0.042, 0.0, -0.05))
    soleus_p2 = knee.lerp(ankle, 0.84) + Vector((0.025, 0.0, -0.03))

    if not is_correct:
        # Contracted, bunched-up glowing RED/ORANGE cramping gastrocnemius
        add_oriented_ellipsoid("GastrocCramp_L", calf_p1, calf_p2, 0.068, 0.062, mat=mats["red_glow"], scale_z_mult=0.62, offset_local=(0.015, -0.02, 0.02))
        add_oriented_ellipsoid("GastrocCramp_R", calf_p1, calf_p2, 0.065, 0.058, mat=mats["orange_glow"], scale_z_mult=0.60, offset_local=(0.015, 0.025, 0.02))
        add_oriented_ellipsoid("SoleusStrain", calf_p2, soleus_p2, 0.045, 0.045, mat=mats["orange_glow"], scale_z_mult=0.60)
        add_cylinder_between("AchillesTendon", soleus_p2, heel, 0.018, mat=mats["red_glow"])
        add_point_glow_light("CalfCrampLight", calf_p1.lerp(calf_p2, 0.5) + Vector((0.08, -0.18, 0.0)), color=(1.0, 0.15, 0.02), energy=32.0)
    else:
        # Lengthened, released glowing GREEN/YELLOW gastrocnemius & soleus
        add_oriented_ellipsoid("GastrocRelease_L", calf_p1, soleus_p2, 0.054, 0.052, mat=mats["green_glow"], scale_z_mult=0.60, offset_local=(0.01, -0.02, 0.0))
        add_oriented_ellipsoid("GastrocRelease_R", calf_p1, soleus_p2, 0.052, 0.050, mat=mats["yellow_glow"], scale_z_mult=0.58, offset_local=(0.01, 0.02, 0.0))
        add_cylinder_between("AchillesStretch", soleus_p2, heel, 0.018, mat=mats["green_glow"])
        add_point_glow_light("CalfHealLight", calf_p1.lerp(soleus_p2, 0.5) + Vector((0.08, -0.18, 0.0)), color=(0.05, 1.0, 0.40), energy=32.0)

    # Foot & Toes
    add_ellipsoid("AnkleMalleolus", ankle, (0.048, 0.048, 0.048), mat=skin_xray)
    add_ellipsoid("HeelCalcaneus", heel, (0.045, 0.042, 0.045), mat=mat_bone if not is_correct else skin)
    add_oriented_ellipsoid("MidFoot", heel, toe_mid, 0.052, 0.044, mat=skin, scale_z_mult=0.62)
    add_oriented_ellipsoid("ForeFootToes", toe_mid, toe_tip, 0.044, 0.036, mat=mats["green_glow"] if is_correct else mats["orange_glow"], scale_z_mult=0.65)

    # Sculpted Mannequin Hand Reaching to Pull Toes Back in Correct Panel!
    if is_correct:
        wrist_pt = Vector((-0.02, -0.06, 0.88))
        palm_pt = Vector((-0.18, -0.04, 0.75))
        add_oriented_ellipsoid("ReachingForearm", Vector((0.18, -0.08, 1.05)), wrist_pt, 0.048, 0.044, mat=skin, scale_z_mult=0.58)
        add_oriented_ellipsoid("GripHand", wrist_pt, palm_pt, 0.046, 0.032, mat=skin, scale_z_mult=0.62)
        add_oriented_ellipsoid("GripFingers", palm_pt, toe_tip + Vector((-0.02, 0.0, 0.0)), 0.035, 0.028, mat=skin, scale_z_mult=0.65)

    return {
        "leg_chain": [knee, ankle, toe_mid],
        "toe_tip": toe_tip,
        "pull_target": knee + Vector((-0.12, 0.0, 0.04)),
        "calf_belly": calf_p1.lerp(calf_p2, 0.5) + Vector((0.06, 0.0, -0.02)),
    }

print("Mannequin builders appended")

# ---------------------------------------------------------
# 3. POST-PROCESSING: BLOOM, SYMBOLS ONLY (ZERO TEXT!)
# ---------------------------------------------------------

def apply_neon_bloom(img_pil, radius=18, intensity=0.85):
    """Extracts bright neon red/orange/yellow/green emission pixels and adds realistic studio optical bloom."""
    arr = np.array(img_pil, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / (max_c + 1e-5)
    mask = ((max_c > 145) & (sat > 0.42)).astype(np.float32)
    glow_arr = np.zeros_like(arr)
    for c in range(3):
        glow_arr[:,:,c] = arr[:,:,c] * mask
    glow_img = Image.fromarray(np.clip(glow_arr, 0, 255).astype(np.uint8))
    blur1 = glow_img.filter(ImageFilter.GaussianBlur(radius=radius))
    blur2 = glow_img.filter(ImageFilter.GaussianBlur(radius=radius*2.2))
    out = arr + np.array(blur1, dtype=np.float32) * intensity + np.array(blur2, dtype=np.float32) * (intensity * 0.6)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def draw_status_badge(img_pil, cx, cy, radius=54, is_correct=False):
    """Draws @demicstory's signature glossy Red X circle or Green Checkmark circle (zero text!)."""
    overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    color = (0, 230, 95) if is_correct else (255, 38, 25)
    glow_col = (0, 255, 110, 115) if is_correct else (255, 45, 20, 115)

    # Outer soft glow
    draw.ellipse([cx-radius-14, cy-radius-14, cx+radius+14, cy+radius+14], fill=glow_col)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    draw = ImageDraw.Draw(overlay)

    # Crisp dark rim + vibrant circle fill + subtle white border
    draw.ellipse([cx-radius-4, cy-radius-4, cx+radius+4, cy+radius+4], fill=(12, 16, 22, 230))
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=color + (255,), outline=(255, 255, 255, 235), width=4)

    # Crisp White X or White Checkmark (✓)
    w = max(8, int(radius * 0.22))
    if not is_correct:
        d = int(radius * 0.42)
        draw.line([(cx-d, cy-d), (cx+d, cy+d)], fill=(255, 255, 255, 255), width=w)
        draw.line([(cx-d, cy+d), (cx+d, cy-d)], fill=(255, 255, 255, 255), width=w)
    else:
        p1 = (cx - int(radius*0.44), cy + int(radius*0.04))
        p2 = (cx - int(radius*0.10), cy + int(radius*0.38))
        p3 = (cx + int(radius*0.48), cy - int(radius*0.34))
        draw.line([p1, p2], fill=(255, 255, 255, 255), width=w)
        draw.line([p2, p3], fill=(255, 255, 255, 255), width=w)
        draw.ellipse([p2[0]-w//2, p2[1]-w//2, p2[0]+w//2, p2[1]+w//2], fill=(255,255,255,255))

    return Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")

def draw_joint_chain_o_o_o(img_pil, pts_2d, line_color=(255, 255, 255), dot_fill=None):
    """Draws @demicstory's signature white `o—o—o` ball-and-stick anatomical joint markers."""
    overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    # Soft outer shadow/glow for visibility over bright mannequin
    for i in range(len(pts_2d) - 1):
        draw.line([pts_2d[i], pts_2d[i+1]], fill=(10, 15, 22, 180), width=11)
    for p in pts_2d:
        draw.ellipse([p[0]-15, p[1]-15, p[0]+15, p[1]+15], fill=(10, 15, 22, 180))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    draw = ImageDraw.Draw(overlay)

    # Crisp white line segments
    for i in range(len(pts_2d) - 1):
        draw.line([pts_2d[i], pts_2d[i+1]], fill=line_color + (245,), width=5)

    # Crisp hollow/filled joint rings (`o—o—o`)
    for idx, p in enumerate(pts_2d):
        r_out = 12
        r_in = 6
        fill_c = (dot_fill if dot_fill else (28, 34, 44)) + (255,)
        draw.ellipse([p[0]-r_out, p[1]-r_out, p[0]+r_out, p[1]+r_out], fill=(255, 255, 255, 255))
        draw.ellipse([p[0]-r_in, p[1]-r_in, p[0]+r_in, p[1]+r_in], fill=fill_c)

    return Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")

def draw_neon_arrow(img_pil, p_start, p_end, color=(0, 255, 102), width=12, head_len=32, curved=False):
    """Draws @demicstory's bold bright neon green (or red) directional force arrow."""
    overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    x1, y1 = p_start
    x2, y2 = p_end
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 5:
        return img_pil
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    # Stop shaft slightly before tip so arrowhead is sharp
    sx2 = x2 - ux * (head_len * 0.65)
    sy2 = y2 - uy * (head_len * 0.65)

    # Glow pass
    if curved:
        mid_x = (x1 + sx2)*0.5 + px * (length * 0.25)
        mid_y = (y1 + sy2)*0.5 + py * (length * 0.25)
        curve_pts = []
        for s in range(25):
            t = s / 24.0
            cx = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * sx2
            cy = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * sy2
            curve_pts.append((cx, cy))
        # Recompute tip tangent from last segment
        tx, ty = x2 - curve_pts[-2][0], y2 - curve_pts[-2][1]
        tlen = max(1e-4, math.hypot(tx, ty))
        ux, uy = tx / tlen, ty / tlen
        px, py = -uy, ux
        draw.line(curve_pts, fill=color + (130,), width=width+12)
    else:
        draw.line([(x1, y1), (sx2, sy2)], fill=color + (130,), width=width+12)

    head_w = head_len * 0.62
    bx = x2 - ux * head_len
    by = y2 - uy * head_len
    tri = [(x2, y2), (bx + px*head_w, by + py*head_w), (bx - px*head_w, by - py*head_w)]
    draw.polygon(tri, fill=color + (140,))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=7))
    draw = ImageDraw.Draw(overlay)

    # Crisp inner arrow pass
    if curved:
        draw.line(curve_pts, fill=color + (255,), width=width)
        draw.line(curve_pts, fill=(220, 255, 235, 210), width=max(3, width//3))
    else:
        draw.line([(x1, y1), (sx2, sy2)], fill=color + (255,), width=width)
        draw.line([(x1, y1), (sx2, sy2)], fill=(220, 255, 235, 210), width=max(3, width//3))
    draw.polygon(tri, fill=color + (255,), outline=(255, 255, 255, 220))

    return Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")

def draw_muscle_map_inset(img_pil, x, y, size=148, highlight_zone="neck", highlight_color=(0, 255, 102)):
    """Draws @demicstory's optional top-left rounded-square anatomical muscle-map silhouette inset (zero text!)."""
    overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    # Dark glass rounded square card
    draw.rounded_rectangle([x, y, x+size, y+size], radius=24, fill=(14, 20, 28, 215), outline=(120, 140, 165, 180), width=3)

    cx = x + size // 2
    cy = y + size // 2
    # Mini anatomical mannequin silhouette inside inset
    body_col = (175, 185, 198, 235)
    # Head
    draw.ellipse([cx-14, cy-54, cx+14, cy-24], fill=body_col)
    # Neck
    n_col = highlight_color + (255,) if highlight_zone == "neck" else body_col
    draw.rounded_rectangle([cx-8, cy-26, cx+8, cy-12], radius=4, fill=n_col)
    # Shoulders & Torso
    draw.polygon([(cx-34, cy-12), (cx+34, cy-12), (cx+24, cy+34), (cx-24, cy+34)], fill=body_col)
    # Highlighted muscle patches on silhouette
    if highlight_zone == "neck":
        draw.ellipse([cx-22, cy-24, cx+22, cy-4], fill=highlight_color + (245,))
    elif highlight_zone == "core":
        draw.ellipse([cx-24, cy-2, cx+24, cy+30], fill=highlight_color + (245,))
        draw.ellipse([cx-36, cy-14, cx-16, cy+6], fill=(255, 230, 0, 245))
    elif highlight_zone == "calf":
        # Draw lower legs in inset
        draw.rounded_rectangle([cx-22, cy-10, cx-6, cy+48], radius=7, fill=body_col)
        draw.rounded_rectangle([cx+6, cy-10, cx+22, cy+48], radius=7, fill=body_col)
        draw.ellipse([cx-24, cy+6, cx-4, cy+38], fill=highlight_color + (255,))
        draw.ellipse([cx+4, cy+6, cx+24, cy+38], fill=highlight_color + (255,))

    return Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")

print("Post-processing helpers appended")

# ---------------------------------------------------------
# 4. RENDER PIPELINE FOR SAMPLES A, B, C & 8-SECOND MP4 REEL
# ---------------------------------------------------------

def render_panel_a(t_correct=0.0, res_x=1080, res_y=960, samples=8, raw_path="/tmp/panel_a.png"):
    scene = clear_scene(res_x=res_x, res_y=res_y, samples=samples)
    mats = get_standard_materials()
    cam_obj = setup_studio_environment(cam_loc=(-0.55, -1.72, 1.05), cam_target=(-0.05, 0.0, 0.88), focal_length=58)
    landmarks = build_mannequin_sample_a(mats, t_correct=t_correct)

    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    img = Image.open(raw_path).convert("RGB")
    img = apply_neon_bloom(img, radius=18, intensity=0.95)

    # Project 3D joint chain (`o—o—o`) onto rendered 2D image
    pts_2d = [project_3d_point(scene, cam_obj, pt) for pt in landmarks["joint_chain"]]
    is_corr = (t_correct >= 0.5)
    dot_c = (0, 230, 95) if is_corr else (255, 50, 25)
    img = draw_joint_chain_o_o_o(img, pts_2d, line_color=(255, 255, 255), dot_fill=dot_c)

    # Directional Force Arrow (Green chin-tuck & thoracic extension arrow when correct, Red forward-strain arrow when wrong)
    p_head_f = project_3d_point(scene, cam_obj, landmarks["head_front"])
    p_head_b = project_3d_point(scene, cam_obj, landmarks["head_back"])
    p_mid_b = project_3d_point(scene, cam_obj, landmarks["mid_back"])
    if is_corr:
        # Bright Green Arrow showing chin-tuck retraction & upward crown decompression
        img = draw_neon_arrow(img, (p_head_f[0] - 85, p_head_f[1] + 8), (p_head_f[0] - 5, p_head_f[1] - 4), color=(0, 255, 102), width=11, head_len=30)
        img = draw_neon_arrow(img, (p_mid_b[0] + 25, p_mid_b[1] + 55), (p_mid_b[0] + 5, p_mid_b[1] - 45), color=(0, 255, 102), width=11, head_len=30)
    else:
        # Red downward/forward shear arrow on overloaded head
        img = draw_neon_arrow(img, (p_head_f[0] + 10, p_head_f[1] - 45), (p_head_f[0] - 75, p_head_f[1] + 25), color=(255, 45, 20), width=11, head_len=30)

    # Top-Right Status Badge (Red X or Green Check)
    img = draw_status_badge(img, res_x - 96, 96, radius=52, is_correct=is_corr)
    # Top-Left Muscle Map Inset
    img = draw_muscle_map_inset(img, 36, 36, size=136, highlight_zone="neck",
                                highlight_color=(0, 255, 102) if is_corr else (255, 45, 20))
    return img


def render_panel_b(is_correct=False, res_x=1080, res_y=960, samples=8, raw_path="/tmp/panel_b.png"):
    scene = clear_scene(res_x=res_x, res_y=res_y, samples=samples)
    mats = get_standard_materials()
    cam_obj = setup_studio_environment(cam_loc=(-0.48, -2.05, 1.28), cam_target=(-0.04, -0.05, 1.12), focal_length=55)
    landmarks = build_mannequin_sample_b(mats, is_correct=is_correct)

    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    img = Image.open(raw_path).convert("RGB")
    img = apply_neon_bloom(img, radius=18, intensity=0.95)

    # White `o—o—o` joint marker along Shoulder -> Elbow -> Fist
    pts_2d = [project_3d_point(scene, cam_obj, pt) for pt in landmarks["arm_chain"]]
    dot_c = (0, 230, 95) if is_correct else (255, 50, 25)
    img = draw_joint_chain_o_o_o(img, pts_2d, line_color=(255, 255, 255), dot_fill=dot_c)

    # Curved Rotational Arrow across Core / Lead Hook Path
    c_start = project_3d_point(scene, cam_obj, landmarks["core_arc_start"])
    c_end = project_3d_point(scene, cam_obj, landmarks["core_arc_end"])
    if is_correct:
        img = draw_neon_arrow(img, (c_start[0] - 40, c_start[1] + 20), (c_end[0] + 55, c_end[1] - 10),
                              color=(0, 255, 102), width=13, head_len=34, curved=True)
    else:
        shld_2d = pts_2d[0]
        elbow_2d = pts_2d[1]
        img = draw_neon_arrow(img, (elbow_2d[0] - 45, elbow_2d[1] + 40), (shld_2d[0] - 10, shld_2d[1] + 5),
                              color=(255, 45, 20), width=11, head_len=30)

    img = draw_status_badge(img, res_x - 96, 96, radius=52, is_correct=is_correct)
    img = draw_muscle_map_inset(img, 36, 36, size=136, highlight_zone="core",
                                highlight_color=(0, 255, 102) if is_correct else (255, 45, 20))
    return img


def render_panel_c(is_correct=False, res_x=1080, res_y=960, samples=8, raw_path="/tmp/panel_c.png"):
    scene = clear_scene(res_x=res_x, res_y=res_y, samples=samples)
    mats = get_standard_materials()
    cam_obj = setup_studio_environment(cam_loc=(-0.05, -1.42, 0.72), cam_target=(-0.05, 0.0, 0.56), focal_length=60)
    landmarks = build_mannequin_sample_c(mats, is_correct=is_correct)

    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    img = Image.open(raw_path).convert("RGB")
    img = apply_neon_bloom(img, radius=18, intensity=0.95)

    # White `o—o—o` joint marker along Knee -> Ankle -> Forefoot
    pts_2d = [project_3d_point(scene, cam_obj, pt) for pt in landmarks["leg_chain"]]
    dot_c = (0, 230, 95) if is_correct else (255, 50, 25)
    img = draw_joint_chain_o_o_o(img, pts_2d, line_color=(255, 255, 255), dot_fill=dot_c)

    p_toe = project_3d_point(scene, cam_obj, landmarks["toe_tip"])
    p_pull = project_3d_point(scene, cam_obj, landmarks["pull_target"])
    p_calf = project_3d_point(scene, cam_obj, landmarks["calf_belly"])

    if is_correct:
        # Bright Green Curved Dorsiflexion Arrow (pulling toes toward shin) + Calf release arrow
        img = draw_neon_arrow(img, (p_toe[0] - 45, p_toe[1] + 55), (p_pull[0] - 25, p_pull[1] - 15),
                              color=(0, 255, 102), width=13, head_len=34, curved=True)
        img = draw_neon_arrow(img, (p_calf[0] + 10, p_calf[1] - 30), (p_calf[0] - 35, p_calf[1] + 45),
                              color=(0, 255, 102), width=10, head_len=28)
    else:
        # Red plantarflexion cramp arrow
        img = draw_neon_arrow(img, (pts_2d[1][0] - 30, pts_2d[1][1] + 10), (p_toe[0] - 35, p_toe[1] + 25),
                              color=(255, 45, 20), width=12, head_len=30)

    img = draw_status_badge(img, res_x - 96, 96, radius=52, is_correct=is_correct)
    img = draw_muscle_map_inset(img, 36, 36, size=136, highlight_zone="calf",
                                highlight_color=(0, 255, 102) if is_correct else (255, 45, 20))
    return img


def stack_vertical_split(top_img, bot_img):
    """Combines Wrong (Top, 1080x960) and Right (Bottom, 1080x960) into a 1080x1920 vertical @demicstory comparison reel frame."""
    canvas = Image.new("RGB", (1080, 1920), (10, 14, 20))
    canvas.paste(top_img, (0, 0))
    canvas.paste(bot_img, (0, 960))
    draw = ImageDraw.Draw(canvas)
    # Clean minimalist divider line between Top and Bottom split
    draw.line([(0, 960), (1080, 960)], fill=(70, 85, 105), width=6)
    return canvas


def render_single_full_frame_a(t_correct=0.0, res_x=540, res_y=960, samples=4, raw_path="/tmp/anim_a.png"):
    """Renders a full-screen 9:16 frame (upscaled cleanly to 1080x1920) for the 8-second animated reel."""
    scene = clear_scene(res_x=res_x, res_y=res_y, samples=samples)
    mats = get_standard_materials()
    # Slight dynamic camera orbit across t_correct for authentic 3D CGI feel
    cam_x = -0.58 + 0.08 * t_correct
    cam_y = -1.85 + 0.05 * math.sin(t_correct * math.pi)
    cam_obj = setup_studio_environment(cam_loc=(cam_x, cam_y, 1.02), cam_target=(-0.05, 0.0, 0.82), focal_length=52)
    landmarks = build_mannequin_sample_a(mats, t_correct=t_correct)

    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    img_small = Image.open(raw_path).convert("RGB")
    img = img_small.resize((1080, 1920), Image.Resampling.LANCZOS)
    img = apply_neon_bloom(img, radius=22, intensity=0.95)

    # Scale projected coordinates by 2x (since 540x960 -> 1080x1920)
    pts_2d = []
    for pt in landmarks["joint_chain"]:
        p = project_3d_point(scene, cam_obj, pt)
        pts_2d.append((p[0]*2, p[1]*2))

    is_corr = (t_correct >= 0.5)
    dot_c = (0, 230, 95) if is_corr else (255, 50, 25)
    img = draw_joint_chain_o_o_o(img, pts_2d, line_color=(255, 255, 255), dot_fill=dot_c)

    p_head_f = project_3d_point(scene, cam_obj, landmarks["head_front"])
    p_head_f = (p_head_f[0]*2, p_head_f[1]*2)
    p_mid_b = project_3d_point(scene, cam_obj, landmarks["mid_back"])
    p_mid_b = (p_mid_b[0]*2, p_mid_b[1]*2)

    if t_correct < 0.42:
        img = draw_neon_arrow(img, (p_head_f[0] + 15, p_head_f[1] - 65), (p_head_f[0] - 95, p_head_f[1] + 35),
                              color=(255, 45, 20), width=14, head_len=38)
    elif t_correct > 0.58:
        img = draw_neon_arrow(img, (p_head_f[0] - 110, p_head_f[1] + 10), (p_head_f[0] - 8, p_head_f[1] - 6),
                              color=(0, 255, 102), width=14, head_len=38)
        img = draw_neon_arrow(img, (p_mid_b[0] + 35, p_mid_b[1] + 80), (p_mid_b[0] + 8, p_mid_b[1] - 65),
                              color=(0, 255, 102), width=14, head_len=38)

    img = draw_status_badge(img, 1080 - 120, 130, radius=64, is_correct=is_corr)
    img = draw_muscle_map_inset(img, 48, 64, size=168, highlight_zone="neck",
                                highlight_color=(0, 255, 102) if is_corr else (255, 45, 20))
    return img


def main():
    t0 = time.time()
    print("1. Rendering Sample A (Tech Neck X-Ray Cervical Spine vs Chin-Tuck Decompression)...")
    a_wrong = render_panel_a(t_correct=0.0, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_a_wrong.png")
    a_right = render_panel_a(t_correct=1.0, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_a_right.png")
    sample_a = stack_vertical_split(a_wrong, a_right)
    sample_a_path = f"{OUT_DIR}/demic_sample_a_neck.png"
    sample_a.save(sample_a_path)
    print(f"Saved {sample_a_path} ({time.time()-t0:.1f}s)")

    print("2. Rendering Sample B (UFC Lead Hook Shoulder Overload vs Core/Lat Rotation)...")
    b_wrong = render_panel_b(is_correct=False, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_b_wrong.png")
    b_right = render_panel_b(is_correct=True, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_b_right.png")
    sample_b = stack_vertical_split(b_wrong, b_right)
    sample_b_path = f"{OUT_DIR}/demic_sample_b_ufc.png"
    sample_b.save(sample_b_path)
    print(f"Saved {sample_b_path} ({time.time()-t0:.1f}s)")

    print("3. Rendering Sample C (Calf Cramp Tibia/Gastroc X-Ray vs Dorsiflexion Release)...")
    c_wrong = render_panel_c(is_correct=False, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_c_wrong.png")
    c_right = render_panel_c(is_correct=True, res_x=1080, res_y=960, samples=8, raw_path=f"{TMP_DIR}/raw_c_right.png")
    sample_c = stack_vertical_split(c_wrong, c_right)
    sample_c_path = f"{OUT_DIR}/demic_sample_c_calf.png"
    sample_c.save(sample_c_path)
    print(f"Saved {sample_c_path} ({time.time()-t0:.1f}s)")

    # Combine all 3 samples side-by-side into `/workspace/daily_info_reels/output/demic_clone_3tracks.png`
    grid_w = 1080 * 3 + 24 * 2
    grid_h = 1920
    grid_img = Image.new("RGB", (grid_w, grid_h), (8, 11, 16))
    grid_img.paste(sample_a, (0, 0))
    grid_img.paste(sample_b, (1080 + 24, 0))
    grid_img.paste(sample_c, ((1080 + 24) * 2, 0))
    tracks_path = f"{OUT_DIR}/demic_clone_3tracks.png"
    grid_img.save(tracks_path)
    print(f"Saved combined 3-track showcase: {tracks_path}")

    # 4. Render 8-second 30fps MP4 reel `/workspace/daily_info_reels/output/demic_clone_day1.mp4`
    # Render 9 key pose states across t_correct in [0.0 .. 1.0] + subtle breathing micro-motion and interpolate smoothly at 30fps (240 frames)
    print("4. Rendering 3D keyframe states for 8-second 30fps animated reel...")
    key_t_vals = [0.0, 0.06, 0.15, 0.30, 0.50, 0.70, 0.85, 0.94, 1.0]
    key_imgs = []
    for idx, kt in enumerate(key_t_vals):
        k_img = render_single_full_frame_a(t_correct=kt, res_x=540, res_y=960, samples=4, raw_path=f"{TMP_DIR}/key_{idx}.png")
        key_imgs.append(np.array(k_img, dtype=np.float32))

    # Generate 240 frames (8.0s @ 30fps):
    # - 0.0s to 3.2s (frames 0..95): Wrong posture with subtle breathing oscillation between key 0 and key 1
    # - 3.2s to 4.6s (frames 96..138): Smooth 3D transition from Wrong (Red X) to Right (Green ✓) across keys 1..7
    # - 4.6s to 8.0s (frames 139..239): Correct posture with active green muscle glow pulsing between key 7 and key 8
    total_frames = 240
    for f in range(total_frames):
        sec = f / 30.0
        if sec < 3.2:
            # Subtle breathing oscillation in wrong form
            w = 0.5 * (1.0 - math.cos(sec * math.pi * 1.5))
            arr = (1.0 - w) * key_imgs[0] + w * key_imgs[1]
        elif sec < 4.6:
            # Smooth transition across key_imgs[1..7]
            u = (sec - 3.2) / 1.4
            u_smooth = u * u * (3.0 - 2.0 * u)
            pos = 1.0 + u_smooth * 6.0
            i0 = int(math.floor(pos))
            i1 = min(8, i0 + 1)
            frac = pos - i0
            arr = (1.0 - frac) * key_imgs[i0] + frac * key_imgs[i1]
        else:
            # Subtle active hold oscillation in right form
            w = 0.5 * (1.0 - math.cos((sec - 4.6) * math.pi * 1.5))
            arr = (1.0 - w) * key_imgs[8] + w * key_imgs[7]

        frame_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        frame_pil.save(f"{TMP_DIR}/reel_frame_{f:04d}.jpg", quality=95)

    mp4_path = f"{OUT_DIR}/demic_clone_day1.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", f"{TMP_DIR}/reel_frame_%04d.jpg",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        mp4_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved 8-second 30fps reel: {mp4_path} (Total elapsed: {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
