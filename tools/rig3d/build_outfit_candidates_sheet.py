#!/usr/bin/env python3
"""Render a clay + textured contact sheet of outfit candidate assets.

v2 lighting (tools/rig3d conventions): AgX, exposure 0, key 78, camera-side
fill 36%, cool rim 55%, world 0.018, shadow-catcher floor, 50 mm vertical
24 mm sensor, yaw 35°, elev 10°, ~72% frame fill, 96 spp + OIDN, clay
(0.65, 0.656, 0.664) roughness 0.86.

Third-party meshes stay under gitignored output/_assets/. This script is a
helper and is not required in the repo commit.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None

ROOT = Path("/workspace/daily_info_reels")
ASSET_ROOT = ROOT / "output/_assets/outfit_candidates"
CELL_DIR = ASSET_ROOT / "_cells"
FACTS_PATH = ASSET_ROOT / "inspect_facts.json"
SHEET_PATH = ROOT / "output/outfit_candidates_v1.png"
SHEET_COPY = ASSET_ROOT / "outfit_candidates_v1.png"

CLAY = (0.65, 0.656, 0.664, 1.0)
CLAY_ROUGH = 0.86
KEY_ENERGY = 78.0
FILL_RATIO = 0.36
RIM_RATIO = 0.55
YAW = 35.0
ELEV = 10.0
FRAME_FILL = 0.72
SAMPLES = 96
EXPOSURE = 0.0
VIEW_TRANSFORM = "AgX"
CELL = 480
SENSOR_H = 24.0
LENS = 50.0

CANDIDATES = [
    dict(
        item="gloves",
        slug="artel_classic_red_glove",
        label="Classic Red Glove",
        sub="ARTEL_3D  ·  CC BY 4.0  ·  single glove",
        path=ASSET_ROOT / "gloves/artel_classic_red_glove/model.glb",
    ),
    dict(
        item="beanie",
        slug="arkify_black_knit_beanie",
        label="Black Knit Beanie",
        sub="Arkify 3D  ·  CC BY 4.0  ·  scan, high poly",
        path=ASSET_ROOT / "beanie/arkify_black_knit_beanie/model.glb",
    ),
    dict(
        item="beanie",
        slug="oluwaphemi_beanie",
        label="Rib Beanie",
        sub="oluwaphemi  ·  CC BY 4.0  ·  designed mesh",
        path=ASSET_ROOT / "beanie/oluwaphemi_beanie/model.glb",
    ),
    dict(
        item="trunks",
        slug="sutton_shorts_boxer",
        label="Shorts Boxer",
        sub="Ginny Sutton  ·  CC BY 4.0  ·  DUEL waist badge",
        path=ASSET_ROOT / "trunks/sutton_shorts_boxer/model.glb",
    ),
    dict(
        item="tank",
        slug="zyadelbrdan74_sleeveless_shirt",
        label="Sleeveless Shirt",
        sub="zyadelbrdan74  ·  CC BY 4.0  ·  charcoal tank",
        path=ASSET_ROOT / "tank/zyadelbrdan74_sleeveless_shirt/model.glb",
    ),
    dict(
        item="tank",
        slug="nycfashion_sleeveless_shirt",
        label="NYC Sleeveless Shirt",
        sub="mweisburgh  ·  CC BY 4.0  ·  white tank",
        path=ASSET_ROOT / "tank/nycfashion_sleeveless_shirt/model.glb",
    ),
]


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # factory empty still has a leftover camera/light in some builds; wipe
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for c in list(bpy.data.cameras):
        bpy.data.cameras.remove(c)
    for l in list(bpy.data.lights):
        bpy.data.lights.remove(l)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


def _import_glb(path: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    meshes = [o for o in new if o.type == "MESH"]
    return meshes, new


def _mesh_stats(meshes):
    import bmesh

    total_tris = 0
    total_verts = 0
    objects = []
    all_closed = True
    for o in meshes:
        me = o.data
        # tessellated tris
        me.calc_loop_triangles()
        tris = len(me.loop_triangles)
        verts = len(me.vertices)
        total_tris += tris
        total_verts += verts
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.edges.ensure_lookup_table()
        open_edges = sum(1 for e in bm.edges if len(e.link_faces) < 2)
        nonmanifold = sum(1 for e in bm.edges if not e.is_manifold)
        bm.free()
        closed = open_edges == 0
        if not closed:
            all_closed = False
        objects.append({
            "name": o.name,
            "tris": tris,
            "verts": verts,
            "open_edges": open_edges,
            "nonmanifold_edges": nonmanifold,
            "closed": closed,
            "materials": [s.material.name if s.material else None for s in o.material_slots],
        })
    images = []
    for img in bpy.data.images:
        if img.name in {"Render Result", "Viewer Node"}:
            continue
        images.append({
            "name": img.name,
            "size": list(img.size) if img.size else [0, 0],
            "colorspace": img.colorspace_settings.name,
            "filepath": img.filepath,
        })
    has_textures = any((im.get("size") or [0, 0])[0] > 0 for im in images)
    return {
        "mesh_count": len(meshes),
        "one_mesh": len(meshes) == 1,
        "all_closed": all_closed and len(meshes) > 0,
        "total_tris": total_tris,
        "total_verts": total_verts,
        "objects": objects,
        "has_textures": has_textures,
        "images": images,
    }


def _world_bounds(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            for v in me.vertices:
                pts.append(mw @ v.co)
        finally:
            ev.to_mesh_clear()
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx


def _roots(objs):
    roots = []
    seen = set()
    for o in objs:
        root = o
        while root.parent is not None:
            root = root.parent
        if root.name not in seen:
            roots.append(root)
            seen.add(root.name)
    return roots


TARGET_SPAN = 0.40  # metres, so lighting and clip stay in the v2 range


def _normalize_scale(meshes):
    """Uniform-scale the import so its longest side is TARGET_SPAN metres."""
    mn, mx = _world_bounds(meshes)
    span = max((mx - mn).x, (mx - mn).y, (mx - mn).z, 1e-8)
    factor = TARGET_SPAN / span
    for root in _roots(meshes):
        root.scale = root.scale * factor
    bpy.context.view_layer.update()
    return factor, span


def _place_on_floor(meshes):
    # move as a group so the bbox sits on z=0 and is centered in XY
    mn, mx = _world_bounds(meshes)
    mid = (mn + mx) * 0.5
    delta = Vector((-mid.x, -mid.y, -mn.z))
    for root in _roots(meshes):
        root.location += delta
    bpy.context.view_layer.update()
    return _world_bounds(meshes)


def _area_light(name, loc, size, energy, look_at, color=(1.0, 1.0, 1.0)):
    d = bpy.data.lights.new(name, "AREA")
    d.size = size
    d.energy = energy
    d.spread = 150
    d.color = color
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    direction = Vector(look_at) - Vector(loc)
    o.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return o


def setup_studio(cam, look, height):
    world = bpy.data.worlds.get("Charcoal") or bpy.data.worlds.new("Charcoal")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.018, 0.018, 0.020, 1)
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    if not any(l.to_node == out for l in nt.links):
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world

    # scale light sizes with subject height so small props are not over-wrapped
    s = max(0.25, min(2.5, height / 0.7))
    right = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0))
    view = (Vector(look) - cam.location).normalized()
    up = Vector((0, 0, 1))
    key_loc = cam.location + right * (0.90 * s) + up * (0.55 * s)
    fill_loc = cam.location + view * (0.25 * s) - right * (0.12 * s) + up * (0.12 * s)
    rim_loc = Vector(look) - view * (1.85 * s) + up * (1.15 * s) - right * (0.25 * s)
    # keep energy at v2 values; smaller area size for small subjects
    _area_light("Key", key_loc, 2.5 * s, KEY_ENERGY, look)
    _area_light("Fill", fill_loc, 3.8 * s, KEY_ENERGY * FILL_RATIO, look)
    _area_light("Rim", rim_loc, 2.1 * s, KEY_ENERGY * RIM_RATIO, look,
                color=(0.82, 0.90, 1.0))

    if "StudioFloor" not in bpy.data.objects:
        mesh = bpy.data.meshes.new("StudioFloor")
        verts = [(-8, -8, 0), (8, -8, 0), (8, 8, 0), (-8, 8, 0)]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()
        floor = bpy.data.objects.new("StudioFloor", mesh)
        bpy.context.scene.collection.objects.link(floor)
        fmat = bpy.data.materials.new("StudioFloorMat")
        fmat.use_nodes = True
        bsdf = fmat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.026, 0.026, 0.029, 1)
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.95
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.05
        floor.data.materials.append(fmat)
        floor.is_shadow_catcher = True
        floor.visible_glossy = False
        floor.visible_diffuse = True
        floor.visible_shadow = False


def setup_camera(meshes):
    mn, mx = _world_bounds(meshes)
    dim = mx - mn
    height = max(dim.z, 0.001)
    # for very flat/wide props, frame the larger of height or a projected size
    span = max(height, dim.x, dim.y)
    look = Vector(((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5))
    yaw_r = math.radians(YAW)
    elev_r = math.radians(ELEV)
    direction = Vector((
        math.sin(yaw_r) * math.cos(elev_r),
        -math.cos(yaw_r) * math.cos(elev_r),
        math.sin(elev_r),
    ))
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = LENS
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = SENSOR_H
    cam_data.clip_start = 0.01
    cam_data.clip_end = 50
    v_ang = 2 * math.atan((cam_data.sensor_height * 0.5) / cam_data.lens)
    dist = (span * 0.5) / (FRAME_FILL * math.tan(v_ang * 0.5))
    loc = look + direction.normalized() * dist
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    setup_studio(cam, look, height)
    return cam, height, dist


def apply_color_mgmt(sc):
    for candidate in (VIEW_TRANSFORM, "Filmic", "Standard"):
        try:
            sc.view_settings.view_transform = candidate
            break
        except Exception:
            continue
    sc.view_settings.look = "None"
    sc.view_settings.exposure = EXPOSURE
    sc.view_settings.gamma = 1.0
    try:
        sc.display_settings.display_device = "sRGB"
    except Exception:
        pass


def setup_cycles(sc):
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception:
        pass
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.render.resolution_x = CELL
    sc.render.resolution_y = CELL
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.film_transparent = False
    sc.render.threads_mode = "FIXED"
    sc.render.threads = max(1, (os.cpu_count() or 8) - 0)
    apply_color_mgmt(sc)
    # charcoal backdrop
    sc.world.color = (0.018, 0.018, 0.020) if sc.world else sc.world


def assign_clay(meshes):
    mat = bpy.data.materials.get("ClayV2") or bpy.data.materials.new("ClayV2")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = CLAY
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = CLAY_ROUGH
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.22
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        o.active_material = mat
    return mat


def render_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"  wrote {path}", flush=True)


def main():
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    facts = {}
    for cand in CANDIDATES:
        path = Path(cand["path"])
        if not path.is_file():
            print("MISSING", path, flush=True)
            facts[cand["slug"]] = {"ok": False, "error": "missing"}
            continue
        print(f"\n=== {cand['slug']} ===", flush=True)
        _reset()
        meshes, new = _import_glb(path)
        if not meshes:
            print("  no meshes", flush=True)
            facts[cand["slug"]] = {"ok": False, "error": "no_meshes"}
            continue
        stats = _mesh_stats(meshes)
        factor, orig_span = _normalize_scale(meshes)
        mn, mx = _place_on_floor(meshes)
        dim = mx - mn
        stats["bbox_m"] = [round(dim.x, 4), round(dim.y, 4), round(dim.z, 4)]
        stats["imported_span_m"] = round(orig_span, 4)
        stats["scale_factor"] = factor
        stats["ok"] = True
        facts[cand["slug"]] = stats
        print(
            f"  meshes={stats['mesh_count']} tris={stats['total_tris']} "
            f"closed={stats['all_closed']} textures={stats['has_textures']} "
            f"bbox={stats['bbox_m']}",
            flush=True,
        )
        cam, height, dist = setup_camera(meshes)
        setup_cycles(bpy.context.scene)
        # textured first (materials as imported)
        tex_path = CELL_DIR / f"{cand['slug']}_tex.png"
        render_to(tex_path)
        # then clay (replaces materials)
        assign_clay(meshes)
        clay_path = CELL_DIR / f"{cand['slug']}_clay.png"
        render_to(clay_path)

    FACTS_PATH.write_text(json.dumps(facts, indent=2))
    print("facts ->", FACTS_PATH, flush=True)

    # composite with PIL (available in system python; blender may also have it)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL missing inside Blender; writing facts only", flush=True)
        return 0

    compose_sheet(facts)
    return 0


def compose_sheet(facts=None):
    from PIL import Image, ImageDraw, ImageFont

    if facts is None:
        facts = json.loads(FACTS_PATH.read_text()) if FACTS_PATH.is_file() else {}

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        fontt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        fonts = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = fontb = fontt = fonts = ImageFont.load_default()

    # layout: 4 columns = two candidates × (clay|tex)
    # rows by category
    groups = [
        ("GLOVES", [c for c in CANDIDATES if c["item"] == "gloves"]),
        ("BEANIE", [c for c in CANDIDATES if c["item"] == "beanie"]),
        ("TRUNKS", [c for c in CANDIDATES if c["item"] == "trunks"]),
        ("TANK TOP", [c for c in CANDIDATES if c["item"] == "tank"]),
    ]
    pad = 16
    label_h = 72
    section_h = 36
    title_h = 110
    cols = 4  # clay tex clay tex
    W = pad + cols * (CELL + pad)
    # compute height
    H = title_h
    for _, items in groups:
        rows = max(1, (len(items) + 1) // 2)
        H += section_h + rows * (CELL + label_h + pad) + pad

    bg = (18, 18, 20)
    im = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(im)
    fg = (236, 236, 238)
    muted = (160, 162, 168)
    okc = (140, 200, 160)
    warn = (230, 180, 90)

    dr.text((pad, 18), "Outfit candidates v1   —   clay  |  textures", fill=fg, font=fontt)
    sub = ("v2 lighting: AgX, exposure 0, key 78, fill 36%, cool rim 55%, "
           "world 0.018, shadow-catcher floor, 50 mm vertical / 24 mm, "
           "yaw 35°, elev 10°, ~72% fill, 96 spp + OIDN, clay (0.65, 0.656, 0.664) r=0.86")
    dr.text((pad, 56), sub, fill=muted, font=fonts)
    dr.text((pad, 76),
            "Assets are third-party (gitignored). CC BY requires credit — see docs/specs/OUTFIT_ASSETS.md.",
            fill=muted, font=fonts)

    y = title_h
    missing = Image.new("RGB", (CELL, CELL), (32, 32, 36))
    md = ImageDraw.Draw(missing)
    md.text((24, CELL // 2), "not downloaded", fill=muted, font=font)

    def cell_img(slug, kind):
        p = CELL_DIR / f"{slug}_{kind}.png"
        if p.is_file():
            t = Image.open(p).convert("RGB")
            if t.size != (CELL, CELL):
                t = t.resize((CELL, CELL), Image.Resampling.LANCZOS)
            return t
        return missing

    for title, items in groups:
        dr.text((pad, y + 6), title, fill=fg, font=fontb)
        y += section_h
        for i, cand in enumerate(items):
            pair = i % 2  # 0 left pair, 1 right pair
            row = i // 2
            # if we start a new row after first of a group, y is already the row top
            x0 = pad + pair * 2 * (CELL + pad)
            y0 = y + row * (CELL + label_h + pad)
            im.paste(cell_img(cand["slug"], "clay"), (x0, y0))
            im.paste(cell_img(cand["slug"], "tex"), (x0 + CELL + pad, y0))
            # tiny captions over the cells
            dr.text((x0 + 8, y0 + 8), "CLAY", fill=(200, 200, 204), font=fonts)
            dr.text((x0 + CELL + pad + 8, y0 + 8), "TEXTURES", fill=(200, 200, 204), font=fonts)
            st = facts.get(cand["slug"]) or {}
            bits = []
            if st.get("ok"):
                bits.append(f"{st.get('total_tris', '?')} tris")
                bits.append(f"{st.get('mesh_count', '?')} mesh" + ("" if st.get('mesh_count')==1 else "es"))
                bits.append("closed" if st.get("all_closed") else "open / not watertight")
                bits.append("textures" if st.get("has_textures") else "no textures")
                if st.get("bbox_m"):
                    b = st["bbox_m"]
                    bits.append(f"bbox {b[0]:.2f}×{b[1]:.2f}×{b[2]:.2f} m")
            else:
                bits.append("render missing")
            cap = f"{cand['label']}\n{cand['sub']}\n" + " · ".join(bits)
            color = okc if st.get("ok") and st.get("has_textures") else (warn if st.get("ok") else (200, 120, 120))
            dr.text((x0, y0 + CELL + 8), cap, fill=color if False else fg, font=font)
        rows = max(1, (len(items) + 1) // 2)
        y += rows * (CELL + label_h + pad) + pad

    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    im.save(SHEET_PATH, "PNG")
    im.save(SHEET_COPY, "PNG")
    print(f"sheet {im.size} -> {SHEET_PATH}", flush=True)


if __name__ == "__main__":
    if "--compose-only" in sys.argv:
        compose_sheet()
        sys.exit(0)
    if bpy is None:
        raise SystemExit("This path needs Blender's Python (bpy). Use --compose-only to stitch cells.")
    sys.exit(main() or 0)
