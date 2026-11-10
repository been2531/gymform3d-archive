"""Rest-pose projection painting for rig3d glow.

Back-view first: draw clean anatomical shapes in 2D over an orthographic
render of the Grab3D mannequin, then sample them at render time via rest-pose
camera-projected UVs (the Grab3D atlas is split into hundreds of
vertex-split islands across the back, so an atlas bake seams). The
posed shader samples the 2D paint through UV layer "proj" (781b0de
colour recipe + outside halo). Landmark-polygon / Sobel paths are not used.

CLI:
  blender -b -P project_paint.py -- --ortho [--glb PATH] [--out PNG] [--camera JSON]
  python3 project_paint.py --draw [--camera JSON] [--ortho PNG] [--outdir DIR]
  blender -b -P project_paint.py -- --bake [--paint PNG] [--out PNG] [--camera JSON] [--glb PATH]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GLB = os.path.join(ROOT, "_scratch/assets_probe/human-muscles-rigged.glb")
DEFAULT_MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks")
DEFAULT_CAM = os.path.join(DEFAULT_MASK_DIR, "camera_back.json")

# Torso-framed back ortho (glTF / _read_glb coords: X laterality, Y height, Z depth,
# negative Z = back). Blender's glTF importer maps (x,y,z) -> (x, -z, y).
CAM = {
    "res": 2048,
    "ortho_scale": 1.05,
    "cam_x": 0.0,
    "cam_y": 1.15,   # height centre (glTF Y / Blender Z)
    "cam_z": -2.5,   # behind the back (glTF Z; Blender Y = +2.5)
    "view": "back",
}

# Mesh landmarks measured on this Grab3D rest pose (metres, glTF Y-up).
LM = {
    "C7_y": 1.5058,
    "T12_y": 1.1886,
    "iliac_y": 1.0086,
    "spine_y": 1.4419,
    "inf_angle_y": 1.2125,
    "axilla_y": 1.483,
    "axilla_ax": 0.1909,
    "scap_med_ax": 0.132,
    "rd_cx": 0.1548,
    "rd_cy": 1.4757,
}


def load_cam(path=None):
    p = path or DEFAULT_CAM
    if os.path.isfile(p):
        with open(p) as f:
            d = json.load(f)
        out = dict(CAM)
        out.update({k: d[k] for k in CAM if k in d})
        return out
    return dict(CAM)


def world_to_px(x, y, cam=None):
    cam = cam or CAM
    res = float(cam["res"])
    scale = float(cam["ortho_scale"])
    # Camera sits on +Y (Blender) looking at the back; right-handed look-at makes
    # camera-right = -X, so +X (character left) lands on the left of the image.
    px = -(np.asarray(x, np.float64) - cam["cam_x"]) / scale * res + res / 2.0
    py = -(np.asarray(y, np.float64) - cam["cam_y"]) / scale * res + res / 2.0
    return px, py


def poly_px(pts_xy, cam=None):
    pts = np.asarray(pts_xy, np.float64)
    px, py = world_to_px(pts[:, 0], pts[:, 1], cam)
    return np.stack([px, py], axis=1).astype(np.int32)


def _mirror_pts(pts):
    return [(-x, y) for x, y in pts]


# ---------------------------------------------------------------------------
# Anatomical 2D drawings in metres (back view, origin = midline).
# Lats stay as side wings — they must not fill the mid-scapular fossa (vest).
# ---------------------------------------------------------------------------

def chaikin(pts, iters=4, closed=True):
    """Corner-cutting to a dense smooth closed curve (Bezier-like)."""
    pts = np.asarray(pts, np.float64)
    for _ in range(int(iters)):
        ext = np.vstack([pts, pts[0]]) if closed else pts
        new = []
        for i in range(len(ext) - 1):
            p, q = ext[i], ext[i + 1]
            new.append(0.75 * p + 0.25 * q)
            new.append(0.25 * p + 0.75 * q)
        pts = np.asarray(new, np.float64)
    return [(float(x), float(y)) for x, y in pts]


def trap_rho():
    """One smooth purple mass: trap kite + rhomboid lobes, C7 to T12."""
    return [
        (0.000, 1.522),
        (-0.016, 1.504),
        (-0.036, 1.476),
        (-0.078, 1.450),
        (-0.124, 1.428),
        (-0.134, 1.352),
        (-0.122, 1.272),
        (-0.090, 1.222),
        (-0.046, 1.198),
        (-0.010, 1.186),
        (0.010, 1.186),
        (0.046, 1.198),
        (0.090, 1.222),
        (0.122, 1.272),
        (0.134, 1.352),
        (0.124, 1.428),
        (0.078, 1.450),
        (0.036, 1.476),
        (0.016, 1.504),
    ]


def trap_kite():
    return trap_rho()


def rhomboid_right():
    # Kept for compatibility; the live paint uses the combined trap_rho shape.
    return [
        (0.058, 1.438),
        (0.112, 1.426),
        (0.128, 1.355),
        (0.118, 1.275),
        (0.090, 1.218),
        (0.056, 1.230),
        (0.050, 1.335),
    ]


def lat_right():
    # One continuous wing: axilla/armpit down the flank to the TL fascia.
    return [
        (0.188, 1.500),
        (0.202, 1.478),
        (0.224, 1.438),
        (0.236, 1.378),
        (0.238, 1.300),
        (0.228, 1.210),
        (0.206, 1.118),
        (0.168, 1.048),
        (0.112, 1.016),
        (0.056, 1.012),
        (0.038, 1.058),
        (0.036, 1.122),
        (0.058, 1.178),
        (0.100, 1.218),
        (0.142, 1.268),
        (0.156, 1.348),
        (0.164, 1.422),
        (0.174, 1.472),
    ]


def rear_delt_right():
    return {"center": (0.178, 1.448), "rx": 0.046, "ry": 0.032, "angle": -22.0}


def _fill_poly(mask, pts, cam):
    import cv2
    cv2.fillConvexPoly(mask, poly_px(pts, cam), 255) if _is_convex(pts) else \
        cv2.fillPoly(mask, [poly_px(pts, cam)], 255)


def _is_convex(pts):
    return False


def _ellipse(mask, spec, cam, sign=1.0):
    import cv2
    cx, cy = spec["center"]
    px, py = world_to_px(sign * cx, cy, cam)
    res = float(cam["res"])
    scale = float(cam["ortho_scale"])
    rx = int(round(spec["rx"] / scale * res))
    ry = int(round(spec["ry"] / scale * res))
    ang = float(spec.get("angle", 0.0)) * (1.0 if sign > 0 else -1.0)
    cv2.ellipse(mask, (int(round(px)), int(round(py))), (max(rx, 1), max(ry, 1)),
                ang, 0, 360, 255, -1, lineType=cv2.LINE_AA)


def draw_id_masks(cam=None):
    """Return dict of uint8 HxW masks: lats, rhomboids, mid_trap, rear_delt."""
    import cv2
    cam = cam or CAM
    res = int(cam["res"])
    masks = {k: np.zeros((res, res), np.uint8) for k in
             ("lats", "rhomboids", "mid_trap", "rear_delt")}
    _fill = lambda m, pts: cv2.fillPoly(m, [poly_px(pts, cam)], 255, lineType=cv2.LINE_AA)
    purple = chaikin(trap_rho(), iters=4)
    _fill(masks["mid_trap"], purple)
    # Combined kite+rhomboid lives in mid_trap; rhomboids stays empty (same hue).
    _fill(masks["lats"], chaikin(lat_right(), iters=4))
    _fill(masks["lats"], chaikin(_mirror_pts(lat_right()), iters=4))
    _ellipse(masks["rear_delt"], rear_delt_right(), cam, +1.0)
    _ellipse(masks["rear_delt"], rear_delt_right(), cam, -1.0)
    # Priority: rear_delt > purple > lats so each pixel has one owner.
    rd, tr, lat = (masks[k] > 127 for k in ("rear_delt", "mid_trap", "lats"))
    masks["lats"][rd | tr] = 0
    masks["mid_trap"][rd] = 0
    # Pre-blur the 2D mask edge ~2 px before projection.
    for k in masks:
        masks[k] = cv2.GaussianBlur(masks[k], (0, 0), 2.0)
    return masks


def _hsv_srgb(h, s, v):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import muscle_ids as M
    lin = np.asarray(M.hsv_to_lin_rgb(h, s, v), np.float32)

    def _enc(c):
        a = c <= 0.0031308
        out = np.empty_like(c)
        out[a] = 12.92 * c[a]
        out[~a] = 1.055 * np.power(np.clip(c[~a], 0, None), 1.0 / 2.4) - 0.055
        return out

    return np.clip(_enc(lin), 0, 1)


HUES = {
    "lats": (8, 205, 236),
    "rhomboids": (141, 200, 240),
    "mid_trap": (141, 200, 240),
    "rear_delt": (162, 210, 235),
}


def _fibre_layer(mask, direction, period_px=18.0):
    """Subtle brighter ridges along `direction` (dx, dy) inside mask."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.asarray(direction, np.float32)
    d = d / (np.linalg.norm(d) + 1e-8)
    # Phase varies perpendicular to fibre direction.
    perp = np.array([-d[1], d[0]], np.float32)
    phase = (xx * perp[0] + yy * perp[1]) / period_px * 2.0 * math.pi
    wave = np.clip(np.sin(phase), 0, 1) ** 5
    return (wave * (mask.astype(np.float32) / 255.0)).astype(np.float32)


def compose_paint(masks, cam=None):
    """RGBA float 0-1: sRGB colour + coverage. Fibres and subtle seams included."""
    import cv2
    cam = cam or CAM
    res = int(cam["res"])
    rgb = np.zeros((res, res, 3), np.float32)
    alpha = np.zeros((res, res), np.float32)
    owner = np.zeros((res, res), np.int8)
    order = [("lats", 1), ("rhomboids", 2), ("mid_trap", 3), ("rear_delt", 4)]
    fibres = {
        "lats": (0.22, -0.97),       # image space: up-and-out along the wing
        "rhomboids": (1.0, 0.08),    # near-horizontal, spine -> scapula
        "mid_trap": (0.15, 1.0),     # fan-ish vertical down the kite
        "rear_delt": (1.0, 0.22),
    }
    cols = {k: _hsv_srgb(*HUES[k]) for k in HUES}
    for name, oid in order:
        m = masks[name].astype(np.float32) / 255.0
        cover = m > 0.04
        rgb[cover] = cols[name]
        alpha = np.maximum(alpha, m)
        owner[cover] = oid
        fib = _fibre_layer(masks[name], fibres[name], period_px=22.0 if name != "lats" else 26.0)
        # Subtle fibre ridges — a quiet lift toward a hotter colour, not a stripe overlay.
        hot = np.clip(cols[name] * 0.62 + np.array([1.0, 0.82, 0.70]) * 0.38, 0, 1)
        lift = (fib * 0.12)[..., None]
        rgb = rgb * (1.0 - lift) + hot * lift
    # Subtle dark-ish seams (2 px) only where two muscles meet — not an outer stroke.
    seam = np.zeros((res, res), np.float32)
    kernels = [np.ones((3, 3), np.uint8)]
    for i, (a, _) in enumerate(order):
        da = cv2.dilate((masks[a] > 80).astype(np.uint8), kernels[0], 1)
        for b, _ in order[i + 1:]:
            db = cv2.dilate((masks[b] > 80).astype(np.uint8), kernels[0], 1)
            seam = np.maximum(seam, (da & db).astype(np.float32))
    seam = cv2.GaussianBlur(seam, (0, 0), 0.6)
    dark = np.array([0.12, 0.10, 0.09], np.float32)
    sm = (seam * 0.28)[..., None]
    rgb = rgb * (1.0 - sm) + dark * sm
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    return rgba, owner


def overlay_drawing(ortho_bgr, rgba):
    """Composite paint over the rest ortho for the deliverable drawing image."""
    base = ortho_bgr.astype(np.float32) / 255.0
    if base.shape[2] == 4:
        base = base[:, :, :3]
    paint = rgba[..., :3]
    a = rgba[..., 3:4] * 0.88
    out = base * (1.0 - a) + paint * a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def fix_bake_alpha(path):
    """Cycles EMIT bake writes opaque black; rebuild alpha from colour so
    unpainted UV islands stay clay (not black emission)."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    mx = arr[:, :, :3].max(axis=2).astype(np.float32)
    arr[:, :, 3] = np.clip(mx / 40.0 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def save_png(path, rgba_or_rgb):
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = rgba_or_rgb
    if img.dtype != np.uint8:
        img = np.clip(np.asarray(img) * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    elif img.ndim == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)


def run_draw(camera_path=None, ortho_path=None, outdir=None, overlay_path=None):
    import cv2
    cam = load_cam(camera_path)
    outdir = outdir or DEFAULT_MASK_DIR
    os.makedirs(outdir, exist_ok=True)
    masks = draw_id_masks(cam)
    rgba, owner = compose_paint(masks, cam)
    paint_path = os.path.join(outdir, "db_row_back_paint.png")
    id_path = os.path.join(outdir, "db_row_back_id.png")
    save_png(paint_path, rgba)
    id_rgba = np.stack([
        masks["lats"], masks["rhomboids"], masks["mid_trap"], masks["rear_delt"]
    ], axis=2)
    save_png(id_path, id_rgba)
    if ortho_path and os.path.isfile(ortho_path):
        ortho = cv2.imread(ortho_path, cv2.IMREAD_COLOR)
        if ortho is not None:
            if ortho.shape[0] != rgba.shape[0]:
                ortho = cv2.resize(ortho, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_AREA)
            ortho = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)
            ov = overlay_drawing(ortho, rgba)
            overlay_path = overlay_path or os.path.join(outdir, "db_row_back_drawing.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    with open(os.path.join(outdir, "camera_back.json"), "w") as f:
        json.dump(cam, f, indent=2)
    return {"paint": paint_path, "id": id_path, "overlay": overlay_path, "cam": cam}


# ---------------------------------------------------------------------------
# Blender: rest-pose ortho render + camera-projection bake
# ---------------------------------------------------------------------------

def _blender_import_glb(glb):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh in " + glb)
    body = max(meshes, key=lambda o: len(o.data.vertices))
    return body


def _blender_setup_camera(cam):
    import bpy
    from mathutils import Vector
    sc = bpy.context.scene
    data = bpy.data.cameras.new("ortho_back")
    data.type = "ORTHO"
    data.ortho_scale = float(cam["ortho_scale"])
    obj = bpy.data.objects.new("ortho_back", data)
    sc.collection.objects.link(obj)
    # glTF (x,y,z) -> Blender (x, -z, y); camera behind the back at glTF z=-2.5
    loc = (
        float(cam["cam_x"]),
        -float(cam["cam_z"]),
        float(cam["cam_y"]),
    )
    obj.location = loc
    # Right-handed basis: look -Y (at the back), up +Z (head), right -X.
    from mathutils import Matrix
    x = (-1.0, 0.0, 0.0)
    y = (0.0, 0.0, 1.0)
    z = (0.0, 1.0, 0.0)
    obj.matrix_world = Matrix((
        (x[0], y[0], z[0], loc[0]),
        (x[1], y[1], z[1], loc[1]),
        (x[2], y[2], z[2], loc[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))
    sc.camera = obj
    return obj


def _clay_override(body):
    import bpy
    mat = body.active_material or (body.data.materials[0] if body.data.materials else None)
    if mat is None:
        mat = bpy.data.materials.new("clay")
        mat.use_nodes = True
        if body.data.materials:
            body.data.materials[0] = mat
        else:
            body.data.materials.append(mat)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    inp = bsdf.inputs.get("Base Color")
    if inp is None:
        return
    while inp.links:
        nt.links.remove(inp.links[0])
    inp.default_value = (0.62, 0.58, 0.55, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.55
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25


def blender_ortho(glb=None, out_png=None, camera_path=None):
    import bpy
    cam = load_cam(camera_path)
    glb = os.path.abspath(glb or DEFAULT_GLB)
    out_png = os.path.abspath(out_png or os.path.join(ROOT, "output/_rig3d/v4_authored/peach34_air/rest_back_2048.png"))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    body = _blender_import_glb(glb)
    _clay_override(body)
    cam_obj = _blender_setup_camera(cam)
    sc = bpy.context.scene
    res = int(cam["res"])
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.filepath = out_png
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except Exception:
        try:
            sc.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            sc.render.engine = "CYCLES"
            sc.cycles.samples = 16
    # studio lights so sculpt grooves read
    def _light(name, loc, energy, size=1.2):
        d = bpy.data.lights.new(name, "AREA")
        d.energy = energy
        d.size = size
        o = bpy.data.objects.new(name, d)
        o.location = loc
        sc.collection.objects.link(o)
        return o
    _light("key", (1.4, 2.2, 2.4), 250, 1.6)
    _light("fill", (-1.8, 1.6, 1.4), 80, 2.2)
    _light("rim", (0.2, 3.0, 1.8), 120, 1.4)
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.145, 0.145, 0.150, 1.0)
        bg.inputs[1].default_value = 1.0
    bpy.ops.render.render(write_still=True)
    os.makedirs(DEFAULT_MASK_DIR, exist_ok=True)
    with open(camera_path or DEFAULT_CAM, "w") as f:
        json.dump(cam, f, indent=2)
    print("ORTHO_DONE", out_png)
    return out_png


def blender_bake(paint_path, out_path=None, glb=None, camera_path=None):
    """Camera-project the 2D paint onto the mesh UV via a UV Project modifier."""
    import bpy
    cam = load_cam(camera_path)
    glb = os.path.abspath(glb or DEFAULT_GLB)
    paint_path = os.path.abspath(paint_path)
    out_path = os.path.abspath(out_path or os.path.join(DEFAULT_MASK_DIR, "db_row_glow_uv.png"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body = _blender_import_glb(glb)
    cam_obj = _blender_setup_camera(cam)
    me = body.data
    if not me.uv_layers:
        raise RuntimeError("mesh has no UV layer")
    base_uv = me.uv_layers[0].name
    if "proj" not in [u.name for u in me.uv_layers]:
        me.uv_layers.new(name="proj")
    mod = body.modifiers.new("uvproj", "UV_PROJECT")
    mod.uv_layer = "proj"
    try:
        mod.projector_count = 1
    except Exception:
        pass
    try:
        mod.projectors[0].object = cam_obj
    except Exception:
        try:
            mod.projectors[0] = cam_obj
        except Exception as e:
            print("uv project assign failed", e)
    # Keep UV Project unevaluated (do not apply — the armature modifier is first
    # and we want the same posed evaluation the ortho render used).
    for o in list(bpy.context.view_layer.objects):
        o.select_set(False)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    paint = bpy.data.images.load(paint_path)
    paint.colorspace_settings.name = "sRGB"
    w = h = 2048
    bake_img = bpy.data.images.new("glow_uv_bake", w, h, alpha=True, float_buffer=False)
    bake_img.generated_color = (0, 0, 0, 0)
    bake_img.colorspace_settings.name = "sRGB"

    mat = bpy.data.materials.new("proj_bake")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    img_n = nt.nodes.new("ShaderNodeTexImage")
    img_n.image = paint
    img_n.interpolation = "Cubic"
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "proj"
    nt.links.new(uvn.outputs[0], img_n.inputs[0])
    # Only the back (facing the ortho camera): Blender world +Y is toward camera.
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep.inputs[0])
    gtr = nt.nodes.new("ShaderNodeMath")
    gtr.operation = "GREATER_THAN"
    nt.links.new(sep.outputs[1], gtr.inputs[0])  # Y
    gtr.inputs[1].default_value = 0.18
    # Torso height window (Blender Z = glTF Y) so leg UV islands cannot receive paint.
    sep_p = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], sep_p.inputs[0])
    zlo = nt.nodes.new("ShaderNodeMath"); zlo.operation = "GREATER_THAN"
    nt.links.new(sep_p.outputs[2], zlo.inputs[0]); zlo.inputs[1].default_value = 0.96
    zhi = nt.nodes.new("ShaderNodeMath"); zhi.operation = "LESS_THAN"
    nt.links.new(sep_p.outputs[2], zhi.inputs[0]); zhi.inputs[1].default_value = 1.62
    zok = nt.nodes.new("ShaderNodeMath"); zok.operation = "MULTIPLY"
    nt.links.new(zlo.outputs[0], zok.inputs[0]); nt.links.new(zhi.outputs[0], zok.inputs[1])
    face = nt.nodes.new("ShaderNodeMath"); face.operation = "MULTIPLY"
    nt.links.new(gtr.outputs[0], face.inputs[0]); nt.links.new(zok.outputs[0], face.inputs[1])
    # emission colour = paint RGB, strength = paint A * facing * torso
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(img_n.outputs["Alpha"], mul.inputs[0])
    nt.links.new(face.outputs[0], mul.inputs[1])
    nt.links.new(img_n.outputs["Color"], em.inputs["Color"])
    nt.links.new(mul.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    # bake target node (selected Image Texture using original UV)
    tgt = nt.nodes.new("ShaderNodeTexImage")
    tgt.image = bake_img
    uv_base = nt.nodes.new("ShaderNodeUVMap")
    uv_base.uv_map = base_uv
    nt.links.new(uv_base.outputs[0], tgt.inputs[0])
    nt.nodes.active = tgt
    tgt.select = True
    if body.data.materials:
        body.data.materials[0] = mat
    else:
        body.data.materials.append(mat)
    body.active_material = mat

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 8
    sc.cycles.bake_type = "EMIT"
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.use_pass_emit = True
    sc.render.bake.margin = 2
    sc.render.bake.use_clear = True
    for o in list(bpy.context.view_layer.objects):
        o.select_set(o == body)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.bake(type="EMIT", margin=2)
    bake_img.filepath_raw = out_path
    bake_img.file_format = "PNG"
    bake_img.save()
    fix_bake_alpha(out_path)
    print("BAKE_DONE", out_path)
    return out_path


FOREARM_VGS = ("LeftForeArm", "RightForeArm", "LeftHand", "RightHand")


def write_rest_proj_uvs(me, cam=None, facing_ny=0.12, zlo=0.96, zhi=1.62,
                        exclude_vgs=None, exclude_mask=None):
    """Write rest-pose ortho-camera UVs into layer 'proj'.

    Uses bind-pose vertex coordinates (Blender: X laterality, Z height, +Y = back)
    so the UVs travel with skinning. Non-facing, out-of-window, and forearm/hand
    verts are sent outside [0, 1] for CLIP sampling (no atlas island seams).
    """
    cam = cam or load_cam()
    scale = float(cam["ortho_scale"])
    cx = float(cam["cam_x"])
    cy = float(cam["cam_y"])
    if "proj" not in me.uv_layers:
        me.uv_layers.new(name="proj")
    layer = me.uv_layers["proj"]
    nv = len(me.vertices)
    vu = np.full((nv, 2), -1.0, np.float32)
    ex = np.zeros(nv, dtype=bool)
    if exclude_mask is not None:
        ex |= np.asarray(exclude_mask).astype(bool)
    for name in (exclude_vgs if exclude_vgs is not None else FOREARM_VGS):
        vg = me.vertex_groups.get(name) if hasattr(me, "vertex_groups") else None
        # vertex_groups live on the object; caller may pass them via exclude_mask.
        # Fallback: skip here if the mesh datablock has no groups.
        if vg is None:
            continue
        gi = vg.index
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi and g.weight > 0.28:
                    ex[v.index] = True
    facing = 0
    for i, v in enumerate(me.vertices):
        if ex[i]:
            continue
        if v.normal.y < facing_ny:
            continue
        if v.co.z < zlo or v.co.z > zhi:
            continue
        vu[i, 0] = 0.5 - (v.co.x - cx) / scale
        vu[i, 1] = 0.5 + (v.co.z - cy) / scale
        facing += 1
    for li, loop in enumerate(me.loops):
        layer.data[li].uv = vu[loop.vertex_index]
    return {"proj": "rest_camera", "facing": facing, "excluded": int(ex.sum()), "nv": nv}


def write_rest_proj_uvs_obj(body, cam=None, facing_ny=0.12, zlo=0.96, zhi=1.62,
                            exclude_vgs=None, exclude_mask=None):
    """Write rest-pose camera UVs using the object's world matrix and vertex groups."""
    import mathutils as mu
    cam = cam or load_cam()
    me = body.data
    try:
        me.calc_normals()
    except Exception:
        pass
    scale = float(cam["ortho_scale"])
    cx = float(cam["cam_x"])
    cy = float(cam["cam_y"])
    if "proj" not in me.uv_layers:
        me.uv_layers.new(name="proj")
    layer = me.uv_layers["proj"]
    nv = len(me.vertices)
    vu = np.full((nv, 2), -1.0, np.float32)
    ex = np.zeros(nv, dtype=bool)
    if exclude_mask is not None:
        ex |= np.asarray(exclude_mask).astype(bool)
    for name in (exclude_vgs if exclude_vgs is not None else FOREARM_VGS):
        vg = body.vertex_groups.get(name)
        if vg is None:
            continue
        gi = vg.index
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi and g.weight > 0.28:
                    ex[v.index] = True
    mw = body.matrix_world
    xs = np.zeros(nv, np.float32)
    ys = np.zeros(nv, np.float32)
    zs = np.zeros(nv, np.float32)
    for i, v in enumerate(me.vertices):
        cw = mw @ v.co
        xs[i] = cw.x; ys[i] = cw.y; zs[i] = cw.z
    # Ortho camera sits at +Y looking -Y. The anatomical back is the +Y half of the torso.
    torso = (~ex) & (zs >= zlo) & (zs <= zhi)
    ycut = float(np.median(ys[torso])) if torso.any() else 0.0
    use = torso & (ys >= ycut)
    facing = 0
    for i in np.nonzero(use)[0]:
        vu[i, 0] = 0.5 - (xs[i] - cx) / scale
        vu[i, 1] = 0.5 + (zs[i] - cy) / scale
        facing += 1
    for li, loop in enumerate(me.loops):
        layer.data[li].uv = vu[loop.vertex_index]
    return {"proj": "rest_camera_world", "facing": facing, "excluded": int(ex.sum()),
            "nv": nv, "ycut": ycut, "z_rng": [float(zs.min()), float(zs.max())]}


def sample_uv_texture(me, image, shorts=None, exclude=None, uv_name="proj"):
    """Sample the (paint or baked) image at each vertex via a named UV layer."""
    import bpy
    nv = len(me.vertices)
    uv_layer = me.uv_layers.get(uv_name) or me.uv_layers.active or me.uv_layers[0]
    uvs = np.zeros((nv, 2), np.float32)
    loops = me.loops
    uv_data = uv_layer.data
    for li, loop in enumerate(loops):
        uvs[loop.vertex_index] = uv_data[li].uv
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
    h, w = pixels.shape[:2]
    inside = ((uvs[:, 0] >= 0.0) & (uvs[:, 0] <= 1.0) &
              (uvs[:, 1] >= 0.0) & (uvs[:, 1] <= 1.0))
    xx = np.clip(uvs[:, 0] * (w - 1), 0, w - 1)
    yy = np.clip(uvs[:, 1] * (h - 1), 0, h - 1)
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    samp = pixels[y0, x0]
    rgb = samp[:, :3].copy()
    gw = samp[:, 3].copy()
    gw[~inside] = 0.0
    if shorts is not None:
        sh = np.asarray(shorts)
        sh = sh if sh.dtype == bool else (sh > 0.5)
        gw[sh] = 0.0
    if exclude is not None:
        ex = np.asarray(exclude)
        ex = ex if ex.dtype == bool else (ex > 0.5)
        gw[ex] = 0.0
    return gw, rgb, {"src": "glow_proj", "uv": uv_layer.name,
                     "verts>0.5": int((gw > 0.5).sum())}


def _parse_argv(argv):
    args = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
                continue
            args[key] = True
        i += 1
    return args


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        # drop script name if present
        if argv and argv[0].endswith("project_paint.py"):
            argv = argv[1:]
    args = _parse_argv(argv)
    if "render-ortho" in args or args.get("ortho") is True:
        blender_ortho(args.get("glb"), args.get("out"), args.get("camera"))
        return
    if "bake" in args:
        paint = args.get("paint") or os.path.join(DEFAULT_MASK_DIR, "db_row_back_paint.png")
        blender_bake(paint, args.get("out"), args.get("glb"), args.get("camera"))
        return
    if "draw" in args or not args:
        info = run_draw(args.get("camera"), args.get("ortho"), args.get("outdir"), args.get("overlay"))
        print("DRAW_DONE", json.dumps({k: v for k, v in info.items() if k != "cam"}))
        return
    print("unknown args", args)


if __name__ == "__main__":
    main()
