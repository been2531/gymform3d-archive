"""Shared 3D glow shaders for rig3d (Blender 5.x).

Two backends, same mregion attribute (R lats, G rhomboids, B mid-trap|rear-delt):

  post     — simple emission (fac * contraction). Fibre striations + bloom stay in
             grade3d.py (the historical 2D path).
  blender  — per-muscle wave striations aligned to fibre direction, anisotropic
             tangent on the muscle BSDF, emission strength driven by contraction,
             and a compositor Fog-Glow whose input is Emission * glow_mask AOV
             so the bloom cannot leak onto skin, briefs, or the prop.

Node names are prefixed GF_ so a run can switch backends without rebuilding the
rest of the material (base color, SSS, asset normals).
"""
from __future__ import annotations

GF = "GF_"


def _purge(nt):
    for n in list(nt.nodes):
        if n.name.startswith(GF):
            nt.nodes.remove(n)


def _region_fac(nt):
    am = nt.nodes.new("ShaderNodeAttribute"); am.name = GF + "MREG"; am.attribute_name = "mregion"
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.name = GF + "SEP"
    nt.links.new(am.outputs["Color"], sep.inputs["Color"])
    mx1 = nt.nodes.new("ShaderNodeMath"); mx1.name = GF + "MX1"; mx1.operation = "MAXIMUM"
    nt.links.new(sep.outputs["Red"], mx1.inputs[0]); nt.links.new(sep.outputs["Green"], mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath"); mx2.name = GF + "MX2"; mx2.operation = "MAXIMUM"
    nt.links.new(mx1.outputs[0], mx2.inputs[0]); nt.links.new(sep.outputs["Blue"], mx2.inputs[1])
    ash = nt.nodes.new("ShaderNodeAttribute"); ash.name = GF + "SHORTS"; ash.attribute_name = "shorts"
    inv = nt.nodes.new("ShaderNodeMath"); inv.name = GF + "INV"; inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    nt.links.new(ash.outputs["Fac"], inv.inputs[1])
    fac = nt.nodes.new("ShaderNodeMath"); fac.name = GF + "FAC"; fac.operation = "MULTIPLY"
    nt.links.new(mx2.outputs[0], fac.inputs[0]); nt.links.new(inv.outputs[0], fac.inputs[1])
    return sep, fac


def _flex_strength(nt, fac_out, em_str, flex_min):
    """strength = fac * em_str * (flex_min + (1-flex_min) * flex)."""
    flex = nt.nodes.new("ShaderNodeValue"); flex.name = GF + "FLEX"; flex.label = "GF_FLEX"
    flex.outputs[0].default_value = 0.0
    one_m = nt.nodes.new("ShaderNodeMath"); one_m.name = GF + "F1"; one_m.operation = "SUBTRACT"
    one_m.inputs[0].default_value = 1.0; one_m.inputs[1].default_value = float(flex_min)
    mul_f = nt.nodes.new("ShaderNodeMath"); mul_f.name = GF + "F2"; mul_f.operation = "MULTIPLY"
    nt.links.new(one_m.outputs[0], mul_f.inputs[0]); nt.links.new(flex.outputs[0], mul_f.inputs[1])
    add_f = nt.nodes.new("ShaderNodeMath"); add_f.name = GF + "F3"; add_f.operation = "ADD"
    add_f.inputs[0].default_value = float(flex_min); nt.links.new(mul_f.outputs[0], add_f.inputs[1])
    stm = nt.nodes.new("ShaderNodeMath"); stm.name = GF + "STM"; stm.operation = "MULTIPLY"
    nt.links.new(fac_out, stm.inputs[0]); stm.inputs[1].default_value = float(em_str)
    stm2 = nt.nodes.new("ShaderNodeMath"); stm2.name = GF + "STM2"; stm2.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], stm2.inputs[0]); nt.links.new(add_f.outputs[0], stm2.inputs[1])
    return flex, stm2


def _hook_add(nt, bsdf, mout, em):
    add = nt.nodes.new("ShaderNodeAddShader"); add.name = GF + "ADD"
    for l in list(mout.inputs["Surface"].links):
        nt.links.remove(l)
    nt.links.new(bsdf.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], mout.inputs["Surface"])
    return add


def attach_simple_emission(nt, bsdf, mout, em_col, em_str, flex_min=0.58):
    """Historical 3D glow: flat emission, fibres/bloom left to grade3d."""
    _purge(nt)
    sep, fac = _region_fac(nt)
    flex, stm = _flex_strength(nt, fac.outputs[0], em_str, flex_min)
    em = nt.nodes.new("ShaderNodeEmission"); em.name = GF + "EM"
    em.inputs["Color"].default_value = tuple(em_col)
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    _hook_add(nt, bsdf, mout, em)
    return flex


def _wave(nt, name, vec, scale, distortion, bands, rot):
    mapn = nt.nodes.new("ShaderNodeMapping"); mapn.name = GF + "MAP_" + name
    mapn.inputs["Rotation"].default_value = rot
    mapn.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(vec, mapn.inputs["Vector"])
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.name = GF + "WV_" + name
    wv.wave_type = "BANDS"; wv.bands_direction = bands; wv.wave_profile = "SIN"
    wv.inputs["Scale"].default_value = float(scale)
    wv.inputs["Distortion"].default_value = float(distortion)
    wv.inputs["Detail"].default_value = 4.0
    wv.inputs["Detail Scale"].default_value = 1.6
    wv.inputs["Detail Roughness"].default_value = 0.35
    nt.links.new(mapn.outputs["Vector"], wv.inputs["Vector"])
    return wv.outputs["Fac"]


def attach_blender_glow(nt, bsdf, mout, em_col, em_str, flex_min=0.58, fibre_scale=42.0):
    """Per-muscle fibre waves + anisotropic tangent + AOV for compositor bloom."""
    _purge(nt)
    sep, fac = _region_fac(nt)
    tex = nt.nodes.new("ShaderNodeTexCoord"); tex.name = GF + "COORD"
    # fibre directions in object space (X laterality, Z up the spine at rest; hinge rotates the mesh):
    #   lats      — diagonal lumbar -> armpit (rotate around Y)
    #   rhomboids — medial-lateral (wave along Z so stripes run left-right)
    #   mid-trap  — along the spine (wave along X so stripes run up-down)
    w_lat = _wave(nt, "LAT", tex.outputs["Object"], fibre_scale, 1.8, "X", (0.0, 0.85, 0.0))
    w_rho = _wave(nt, "RHO", tex.outputs["Object"], fibre_scale * 1.15, 1.2, "Z", (0.0, 0.0, 0.0))
    w_trp = _wave(nt, "TRP", tex.outputs["Object"], fibre_scale * 0.95, 0.8, "X", (0.0, 0.0, 0.0))
    # weighted sum of the three waves by mregion channel, then a finer overlay
    m_lat = nt.nodes.new("ShaderNodeMath"); m_lat.name = GF + "WL"; m_lat.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Red"], m_lat.inputs[0]); nt.links.new(w_lat, m_lat.inputs[1])
    m_rho = nt.nodes.new("ShaderNodeMath"); m_rho.name = GF + "WR"; m_rho.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Green"], m_rho.inputs[0]); nt.links.new(w_rho, m_rho.inputs[1])
    m_trp = nt.nodes.new("ShaderNodeMath"); m_trp.name = GF + "WT"; m_trp.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Blue"], m_trp.inputs[0]); nt.links.new(w_trp, m_trp.inputs[1])
    a1 = nt.nodes.new("ShaderNodeMath"); a1.name = GF + "WA1"; a1.operation = "ADD"
    nt.links.new(m_lat.outputs[0], a1.inputs[0]); nt.links.new(m_rho.outputs[0], a1.inputs[1])
    a2 = nt.nodes.new("ShaderNodeMath"); a2.name = GF + "WA2"; a2.operation = "ADD"
    nt.links.new(a1.outputs[0], a2.inputs[0]); nt.links.new(m_trp.outputs[0], a2.inputs[1])
    # finer fascicle layer (shared direction, higher frequency)
    w_fine = _wave(nt, "FINE", tex.outputs["Object"], fibre_scale * 2.8, 2.4, "X", (0.0, 0.55, 0.0))
    mix_f = nt.nodes.new("ShaderNodeMix"); mix_f.name = GF + "FINE_MIX"; mix_f.data_type = "FLOAT"
    mix_f.inputs[0].default_value = 0.35
    nt.links.new(a2.outputs[0], mix_f.inputs[2]); nt.links.new(w_fine, mix_f.inputs[3])
    # remap to 0.42..1.0 so fibres modulate but never punch holes in the glow sheet
    remap = nt.nodes.new("ShaderNodeMapRange"); remap.name = GF + "REMAP"
    remap.inputs["From Min"].default_value = 0.0; remap.inputs["From Max"].default_value = 1.0
    remap.inputs["To Min"].default_value = 0.42; remap.inputs["To Max"].default_value = 1.0
    nt.links.new(mix_f.outputs[0], remap.inputs["Value"])
    fac_f = nt.nodes.new("ShaderNodeMath"); fac_f.name = GF + "FACF"; fac_f.operation = "MULTIPLY"
    nt.links.new(fac.outputs[0], fac_f.inputs[0]); nt.links.new(remap.outputs["Result"], fac_f.inputs[1])
    flex, stm = _flex_strength(nt, fac_f.outputs[0], em_str, flex_min)
    em = nt.nodes.new("ShaderNodeEmission"); em.name = GF + "EM"
    em.inputs["Color"].default_value = tuple(em_col)
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    _hook_add(nt, bsdf, mout, em)
    # anisotropic fibre highlight on the lit muscle surface (not the emission)
    if "Anisotropic" in bsdf.inputs:
        an = nt.nodes.new("ShaderNodeMath"); an.name = GF + "AN"; an.operation = "MULTIPLY"
        an.inputs[0].default_value = 0.55; nt.links.new(fac.outputs[0], an.inputs[1])
        nt.links.new(an.outputs[0], bsdf.inputs["Anisotropic"])
    if "Anisotropic Rotation" in bsdf.inputs:
        # blend rotation ~0.15 (lats diagonal) / 0.0 (trap vertical) by channels
        # leave the socket at 0.12 as a readable middle; per-channel rotation would
        # need a tangent field we do not have on this asset.
        bsdf.inputs["Anisotropic Rotation"].default_value = 0.12
    # AOV so the compositor can isolate glow pixels
    aov = nt.nodes.new("ShaderNodeOutputAOV"); aov.name = GF + "AOV"; aov.aov_name = "glow_mask"
    rgb = nt.nodes.new("ShaderNodeCombineColor"); rgb.name = GF + "AOVRGB"
    nt.links.new(fac.outputs[0], rgb.inputs["Red"]); nt.links.new(fac.outputs[0], rgb.inputs["Green"]); nt.links.new(fac.outputs[0], rgb.inputs["Blue"])
    nt.links.new(rgb.outputs["Color"], aov.inputs["Color"])
    return flex


def attach_surface_tint(nt, bsdf, mout, em_col, em_str, flex_min=0.58, tint_mix=0.58):
    """Method D: shift target-muscle base colour toward neon and add subtle lit emission."""
    _purge(nt)
    at = nt.nodes.new("ShaderNodeAttribute"); at.name = GF + "TGT"; at.attribute_name = "mid_tgt"
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.name = GF + "TSEP"
    nt.links.new(at.outputs["Color"], sep.inputs["Color"])
    ash = nt.nodes.new("ShaderNodeAttribute"); ash.name = GF + "SHORTS"; ash.attribute_name = "shorts"
    inv = nt.nodes.new("ShaderNodeMath"); inv.name = GF + "INV"; inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    nt.links.new(ash.outputs["Fac"], inv.inputs[1])
    fac = nt.nodes.new("ShaderNodeMath"); fac.name = GF + "FAC"; fac.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Red"], fac.inputs[0]); nt.links.new(inv.outputs[0], fac.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.name = GF + "TINT"
    facm = nt.nodes.new("ShaderNodeMath"); facm.name = GF + "TFAC"; facm.operation = "MULTIPLY"
    nt.links.new(fac.outputs[0], facm.inputs[0]); facm.inputs[1].default_value = float(tint_mix)
    nt.links.new(facm.outputs[0], mix.inputs[0])
    base_in = bsdf.inputs["Base Color"]
    if base_in.links:
        nt.links.new(base_in.links[0].from_socket, mix.inputs[6])
        nt.links.remove(base_in.links[0])
    else:
        mix.inputs[6].default_value = tuple(base_in.default_value)
    mix.inputs[7].default_value = (float(em_col[0]), float(em_col[1]), float(em_col[2]), 1.0)
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    flex, stm = _flex_strength(nt, fac.outputs[0], float(em_str) * 0.34, flex_min)
    em = nt.nodes.new("ShaderNodeEmission"); em.name = GF + "EM"
    em.inputs["Color"].default_value = tuple(em_col)
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    _hook_add(nt, bsdf, mout, em)
    return flex


def switch_materials(body_mats, backend, em_col, em_str, flex_min=0.58, fibre_scale=42.0):
    """(Re)attach GF_* glow nodes on every body material. Returns flex Value nodes."""
    flex_nodes = []
    for m_ in body_mats:
        nt = m_.node_tree
        bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
        mout = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        if backend == "blender":
            flex_nodes.append(attach_blender_glow(nt, bsdf, mout, em_col, em_str, flex_min, fibre_scale))
        elif backend == "tint":
            flex_nodes.append(attach_surface_tint(nt, bsdf, mout, em_col, em_str, flex_min))
        else:
            flex_nodes.append(attach_simple_emission(nt, bsdf, mout, em_col, em_str, flex_min))
    return flex_nodes


def set_flex(flex_nodes, value):
    v = float(value)
    for n in flex_nodes:
        n.outputs[0].default_value = v


def _ensure_aov(sc, name="glow_mask"):
    vl = sc.view_layers[0]
    if name not in vl.aovs:
        a = vl.aovs.add(); a.name = name
    try:
        vl.aovs[name].type = "COLOR"
    except Exception:
        pass
    vl.use_pass_emit = True


def setup_compositor(sc, bloom_strength=0.60):
    """Fog-glow the isolated emission and add it back. No bloom on skin / floor / prop."""
    _ensure_aov(sc, "glow_mask")
    ng = sc.compositing_node_group
    if ng is None:
        ng = __import__("bpy").data.node_groups.new("rig3d_cmp", "CompositorNodeTree")
        sc.compositing_node_group = ng
    nt = ng
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers"); rl.location = (0, 0)
    # Emission * glow_mask (extra clamp: only painted muscle verts bloom)
    mul = nt.nodes.new("ShaderNodeMixRGB"); mul.name = "GF_MUL"; mul.blend_type = "MULTIPLY"; mul.inputs[0].default_value = 1.0
    mul.location = (280, -80)
    emit = rl.outputs.get("Emission") or rl.outputs["Image"]
    mask = rl.outputs.get("glow_mask") or rl.outputs["Image"]
    nt.links.new(emit, mul.inputs[1]); nt.links.new(mask, mul.inputs[2])
    gl = nt.nodes.new("CompositorNodeGlare"); gl.location = (520, -80)
    try:
        gl.inputs["Type"].default_value = "Fog Glow"
    except Exception:
        pass
    try:
        gl.inputs["Quality"].default_value = "High"
    except Exception:
        pass
    gl.inputs["Threshold"].default_value = 0.12
    gl.inputs["Strength"].default_value = float(bloom_strength)
    if "Size" in gl.inputs:
        gl.inputs["Size"].default_value = 0.70
    if "Smoothness" in gl.inputs:
        gl.inputs["Smoothness"].default_value = 0.18
    nt.links.new(mul.outputs[0], gl.inputs["Image"])
    glow_out = gl.outputs.get("Glare") or gl.outputs["Image"]
    add = nt.nodes.new("ShaderNodeMixRGB"); add.name = "GF_ADD"; add.blend_type = "ADD"; add.inputs[0].default_value = 1.0
    add.location = (780, 40)
    nt.links.new(rl.outputs["Image"], add.inputs[1]); nt.links.new(glow_out, add.inputs[2])
    # Blender 5 compositor is a node group: the tree's Group Output Image socket is the write-out.
    if hasattr(nt, "interface"):
        for item in list(getattr(nt.interface, "items_tree", []) or []):
            try: nt.interface.remove(item)
            except Exception: pass
        nt.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    gout = nt.nodes.new("NodeGroupOutput"); gout.location = (1020, 40)
    nt.links.new(add.outputs[0], gout.inputs[0])
    sc.render.use_compositing = True
    try:
        sc.render.use_sequencer = False
    except Exception:
        pass


def teardown_compositor(sc):
    sc.render.use_compositing = False
