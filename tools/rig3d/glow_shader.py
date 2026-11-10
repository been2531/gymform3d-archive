"""Shared 3D glow shaders for rig3d (Blender 4.x / 5.x).

Backends:

  post     — simple emission (fac * contraction). Fibre striations + bloom stay in
             grade3d.py (the historical 2D path).
  blender  — per-muscle wave striations aligned to fibre direction (legacy).
  tint     — mid_tgt surface tint (method D).
  inscene  — Mix Shader of the clay Principled BSDF with a peach (or red) Emission
             driven by a blurred per-vertex `glow_w` weight. No procedural fibres;
             sculpt normals carry the remaining detail. Intended for view transform
             Standard so the emission colour displays as measured.

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



def attach_clay_emission(nt, bsdf, mout, em_col, em_str, flex_min=0.88, tint_mix=0.12,
                         fibre_bump=0.22, rim=0.22, fibre_scale=72.0, body_rim=0.22,
                         fibre_line=0.45, opacity=0.94):
    """Per-muscle glow that keeps clay groove shading.

    Colour = clay diffuse lighting without AO/cavity, times a peach (or
    purple) tint, plus a modest emission. A Mix Shader that replaced the
    clay with opaque emission flattened the muscle grain; we add emission
    on top of the still-lit Principled BSDF instead. UV-seam holes are
    filled in projection space (morph-close of the paint) and sampled via
    `glow_proj`; colour is never flooded across the mesh.
    """
    _purge(nt)
    at = nt.nodes.new("ShaderNodeAttribute"); at.name = GF + "GLOWW"; at.attribute_name = "glow_w"
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.name = GF + "GWSEP"
    nt.links.new(at.outputs["Color"], sep.inputs["Color"])
    ac = nt.nodes.new("ShaderNodeAttribute"); ac.name = GF + "GLOWRGB"; ac.attribute_name = "glow_rgb"
    ash = nt.nodes.new("ShaderNodeAttribute"); ash.name = GF + "SHORTS"; ash.attribute_name = "shorts"
    inv = nt.nodes.new("ShaderNodeMath"); inv.name = GF + "INV"; inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(ash.outputs["Fac"], inv.inputs[1])
    fac = nt.nodes.new("ShaderNodeMath"); fac.name = GF + "FAC"; fac.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Red"], fac.inputs[0])
    nt.links.new(inv.outputs["Value"], fac.inputs[1])
    _flex, stm = _flex_strength(nt, fac.outputs[0], 1.0, flex_min)

    # Tint = clay albedo times the projected muscle colour (not a replace).
    tf = nt.nodes.new("ShaderNodeMath"); tf.name = GF + "TINTF"; tf.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], tf.inputs[0]); tf.inputs[1].default_value = float(tint_mix)
    clp = nt.nodes.new("ShaderNodeClamp"); clp.name = GF + "TCLP"
    clp.inputs["Min"].default_value = 0.0
    clp.inputs["Max"].default_value = 1.0
    nt.links.new(tf.outputs[0], clp.inputs["Value"])
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"; mixc.name = GF + "TINT"
    mixc.blend_type = "MULTIPLY"
    nt.links.new(clp.outputs[0], mixc.inputs[0])
    base_in = bsdf.inputs["Base Color"]
    if base_in.links:
        nt.links.new(base_in.links[0].from_socket, mixc.inputs[6])
        nt.links.remove(base_in.links[0])
    else:
        mixc.inputs[6].default_value = tuple(base_in.default_value)
    nt.links.new(ac.outputs["Color"], mixc.inputs[7])
    nt.links.new(mixc.outputs[2], bsdf.inputs["Base Color"])
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.0

    # Fibre-direction waves (R lats, G rhomboids, B mid-trap/delt). Low distortion = crisp lines.
    am = nt.nodes.new("ShaderNodeAttribute"); am.name = GF + "MREG"; am.attribute_name = "mregion"
    msep = nt.nodes.new("ShaderNodeSeparateColor"); msep.name = GF + "MSEP"
    nt.links.new(am.outputs["Color"], msep.inputs["Color"])
    tex = nt.nodes.new("ShaderNodeTexCoord"); tex.name = GF + "FCOORD"
    w_lat = _wave(nt, "LAT", tex.outputs["Object"], float(fibre_scale), 0.45, "X", (0.0, 0.85, 0.0))
    w_rho = _wave(nt, "RHO", tex.outputs["Object"], float(fibre_scale) * 1.10, 0.35, "Z", (0.0, 0.0, 0.0))
    w_trp = _wave(nt, "TRP", tex.outputs["Object"], float(fibre_scale) * 0.95, 0.35, "X", (0.0, 0.0, 0.0))
    ml = nt.nodes.new("ShaderNodeMath"); ml.name = GF + "FL"; ml.operation = "MULTIPLY"
    nt.links.new(msep.outputs["Red"], ml.inputs[0]); nt.links.new(w_lat, ml.inputs[1])
    mr = nt.nodes.new("ShaderNodeMath"); mr.name = GF + "FR"; mr.operation = "MULTIPLY"
    nt.links.new(msep.outputs["Green"], mr.inputs[0]); nt.links.new(w_rho, mr.inputs[1])
    mt = nt.nodes.new("ShaderNodeMath"); mt.name = GF + "FT"; mt.operation = "MULTIPLY"
    nt.links.new(msep.outputs["Blue"], mt.inputs[0]); nt.links.new(w_trp, mt.inputs[1])
    a1 = nt.nodes.new("ShaderNodeMath"); a1.name = GF + "FA1"; a1.operation = "ADD"
    nt.links.new(ml.outputs[0], a1.inputs[0]); nt.links.new(mr.outputs[0], a1.inputs[1])
    a2 = nt.nodes.new("ShaderNodeMath"); a2.name = GF + "FA2"; a2.operation = "ADD"
    nt.links.new(a1.outputs[0], a2.inputs[0]); nt.links.new(mt.outputs[0], a2.inputs[1])
    # Thin bright ridges: wave^5 keeps only the peaks.
    ridge = nt.nodes.new("ShaderNodeMath"); ridge.name = GF + "RID"; ridge.operation = "POWER"
    nt.links.new(a2.outputs[0], ridge.inputs[0]); ridge.inputs[1].default_value = 5.0
    # emission strength = em_str * (1 + fibre_line * ridge)
    flm = nt.nodes.new("ShaderNodeMath"); flm.name = GF + "FLM"; flm.operation = "MULTIPLY"
    nt.links.new(ridge.outputs[0], flm.inputs[0]); flm.inputs[1].default_value = float(fibre_line)
    fla = nt.nodes.new("ShaderNodeMath"); fla.name = GF + "FLA"; fla.operation = "ADD"
    fla.inputs[0].default_value = 1.0
    nt.links.new(flm.outputs[0], fla.inputs[1])
    ems = nt.nodes.new("ShaderNodeMath"); ems.name = GF + "EMS"; ems.operation = "MULTIPLY"
    ems.inputs[0].default_value = float(em_str)
    nt.links.new(fla.outputs[0], ems.inputs[1])

    # Brighten fibre ridges toward a slightly hotter colour
    hot = nt.nodes.new("ShaderNodeMix"); hot.data_type = "RGBA"; hot.name = GF + "HOT"
    nt.links.new(ac.outputs["Color"], hot.inputs[6])
    hot.inputs[7].default_value = (1.0, 0.82, 0.70, 1.0)
    ht = nt.nodes.new("ShaderNodeMath"); ht.name = GF + "HT"; ht.operation = "MULTIPLY"
    nt.links.new(ridge.outputs[0], ht.inputs[0]); ht.inputs[1].default_value = 0.55
    nt.links.new(ht.outputs[0], hot.inputs[0])

    em = nt.nodes.new("ShaderNodeEmission"); em.name = GF + "EM"
    nt.links.new(hot.outputs[2], em.inputs["Color"])
    # Additive emission gated by glow_w. `opacity` now scales emission (it
    # used to be the Mix Shader factor that replaced the clay).
    emg = nt.nodes.new("ShaderNodeMath"); emg.name = GF + "EMG"; emg.operation = "MULTIPLY"
    nt.links.new(ems.outputs[0], emg.inputs[0])
    nt.links.new(stm.outputs[0], emg.inputs[1])
    opm = nt.nodes.new("ShaderNodeMath"); opm.name = GF + "OPM"; opm.operation = "MULTIPLY"
    nt.links.new(emg.outputs[0], opm.inputs[0]); opm.inputs[1].default_value = float(opacity)
    nt.links.new(opm.outputs[0], em.inputs["Strength"])

    mx = nt.nodes.new("ShaderNodeAddShader"); mx.name = GF + "MIXADD"
    nt.links.new(bsdf.outputs[0], mx.inputs[0])
    nt.links.new(em.outputs[0], mx.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.name = GF + "LW"
    lw.inputs["Blend"].default_value = 0.48
    fres = nt.nodes.new("ShaderNodeMath"); fres.name = GF + "FRES"; fres.operation = "SUBTRACT"
    fres.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], fres.inputs[1])
    invg = nt.nodes.new("ShaderNodeMath"); invg.name = GF + "INVG"; invg.operation = "SUBTRACT"
    invg.inputs[0].default_value = 1.0
    nt.links.new(stm.outputs[0], invg.inputs[1])
    # Cool Fresnel on clay only — never add energy onto an already-opaque glow core
    # (that clips R and washes S toward white).
    br0 = nt.nodes.new("ShaderNodeMath"); br0.name = GF + "BR0"; br0.operation = "MULTIPLY"
    nt.links.new(fres.outputs[0], br0.inputs[0]); br0.inputs[1].default_value = float(body_rim)
    br = nt.nodes.new("ShaderNodeMath"); br.name = GF + "BR"; br.operation = "MULTIPLY"
    nt.links.new(br0.outputs[0], br.inputs[0]); nt.links.new(invg.outputs[0], br.inputs[1])
    # Coloured rim only in the short feather (stm*(1-stm) peaks at the outline).
    edge = nt.nodes.new("ShaderNodeMath"); edge.name = GF + "EDGE"; edge.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], edge.inputs[0]); nt.links.new(invg.outputs[0], edge.inputs[1])
    gr = nt.nodes.new("ShaderNodeMath"); gr.name = GF + "GR"; gr.operation = "MULTIPLY"
    nt.links.new(fres.outputs[0], gr.inputs[0]); nt.links.new(edge.outputs[0], gr.inputs[1])
    gr2 = nt.nodes.new("ShaderNodeMath"); gr2.name = GF + "GR2"; gr2.operation = "MULTIPLY"
    nt.links.new(gr.outputs[0], gr2.inputs[0]); gr2.inputs[1].default_value = float(rim) * 4.0
    rsum = nt.nodes.new("ShaderNodeMath"); rsum.name = GF + "RSUM"; rsum.operation = "ADD"
    nt.links.new(br.outputs[0], rsum.inputs[0]); nt.links.new(gr2.outputs[0], rsum.inputs[1])
    rcol = nt.nodes.new("ShaderNodeMix"); rcol.data_type = "RGBA"; rcol.name = GF + "RCOL"
    rcol.inputs[6].default_value = (0.66, 0.80, 1.0, 1.0)
    nt.links.new(ac.outputs["Color"], rcol.inputs[7])
    nt.links.new(stm.outputs[0], rcol.inputs[0])
    rem = nt.nodes.new("ShaderNodeEmission"); rem.name = GF + "REM"
    nt.links.new(rcol.outputs[2], rem.inputs["Color"])
    nt.links.new(rsum.outputs[0], rem.inputs["Strength"])

    add = nt.nodes.new("ShaderNodeAddShader"); add.name = GF + "ADD"
    nt.links.new(mx.outputs[0], add.inputs[0])
    nt.links.new(rem.outputs[0], add.inputs[1])
    for l in list(mout.inputs["Surface"].links):
        nt.links.remove(l)
    nt.links.new(add.outputs[0], mout.inputs["Surface"])

    if float(fibre_bump) > 0:
        bstr = nt.nodes.new("ShaderNodeMath"); bstr.name = GF + "BSTR"; bstr.operation = "MULTIPLY"
        nt.links.new(stm.outputs[0], bstr.inputs[0]); bstr.inputs[1].default_value = float(fibre_bump)
        bump = nt.nodes.new("ShaderNodeBump"); bump.name = GF + "BUMP"
        bump.inputs["Distance"].default_value = 0.006
        nt.links.new(a2.outputs[0], bump.inputs["Height"])
        nt.links.new(bstr.outputs[0], bump.inputs["Strength"])
        nrm_in = bsdf.inputs.get("Normal")
        if nrm_in is not None and nrm_in.links:
            nt.links.new(nrm_in.links[0].from_socket, bump.inputs["Normal"])
            nt.links.remove(nrm_in.links[0])
        if nrm_in is not None:
            nt.links.new(bump.outputs["Normal"], nrm_in)
    aov = nt.nodes.new("ShaderNodeOutputAOV"); aov.name = GF + "AOV"; aov.aov_name = "glow_mask"
    rgb = nt.nodes.new("ShaderNodeCombineColor"); rgb.name = GF + "AOVRGB"
    nt.links.new(fac.outputs[0], rgb.inputs["Red"])
    nt.links.new(fac.outputs[0], rgb.inputs["Green"])
    nt.links.new(fac.outputs[0], rgb.inputs["Blue"])
    nt.links.new(rgb.outputs["Color"], aov.inputs["Color"])
    # Sample the (morph-closed) paint in projection space so UV-seam holes
    # in the mask fill without flooding colour across muscles.
    _bind_proj_texture(nt, ac, sep)
    _exclude_ao_cavity_in_glow(nt, fac.outputs[0])
    return _flex


def _exclude_ao_cavity_in_glow(nt, glow_fac_socket):
    """Skip AO and cavity in the glow region so seam/groove cracks go away.

    Normal-based shading stays on the Principled BSDF; only the extra
    darkening that punched black lines through the emission is removed.
    """
    inv = nt.nodes.new("ShaderNodeMath"); inv.name = GF + "AOINV"; inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(glow_fac_socket, inv.inputs[1])
    for name, tag in (("CLAY_AO_MIX", "AO"), ("CLAY_CAV_MIX", "CAV")):
        n = nt.nodes.get(name)
        if n is None:
            continue
        mul = nt.nodes.new("ShaderNodeMath"); mul.name = GF + tag + "FAC"; mul.operation = "MULTIPLY"
        sock = n.inputs[0]
        if sock.links:
            nt.links.new(sock.links[0].from_socket, mul.inputs[0])
            nt.links.remove(sock.links[0])
        else:
            mul.inputs[0].default_value = float(sock.default_value)
        nt.links.new(inv.outputs[0], mul.inputs[1])
        nt.links.new(mul.outputs[0], sock)


def _bind_proj_texture(nt, ac, sep):
    """If `glow_proj` is loaded, sample it by UV instead of vertex glow attrs."""
    try:
        import bpy
        img = bpy.data.images.get("glow_proj")
    except Exception:
        img = None
    if img is None:
        return
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.name = GF + "PROJUV"
    uv.uv_map = "proj"
    tim = nt.nodes.new("ShaderNodeTexImage"); tim.name = GF + "PROJTEX"
    tim.image = img
    try:
        tim.interpolation = "Cubic"
    except Exception:
        pass
    try:
        tim.extension = "CLIP"
    except Exception:
        pass
    nt.links.new(uv.outputs["UV"], tim.inputs["Vector"])
    for link in list(ac.outputs["Color"].links):
        nt.links.new(tim.outputs["Color"], link.to_socket)
    for link in list(sep.outputs["Red"].links):
        nt.links.new(tim.outputs["Alpha"], link.to_socket)


def attach_mesh_glow(nt, bsdf, mout, em_col, em_str, flex_min=0.88, tint_mix=0.22,
                     fibre_bump=0.42, rim=0.35, fibre_rot=(0.0, 0.85, 0.0),
                     fibre_scale=48.0):
    """Whole-mesh glow: this object IS the muscle, so fac=1 (no glow_w mask).

    Same luminous recipe as attach_clay_emission — tint the clay Principled
    toward `em_col`, add emission + Fresnel rim + fibre wave bump — but the
    silhouette is the mesh outline itself.
    """
    _purge(nt)
    gw = nt.nodes.new("ShaderNodeAttribute"); gw.name = GF + "GLOWW"; gw.attribute_name = "glow_w"
    gsep = nt.nodes.new("ShaderNodeSeparateColor"); gsep.name = GF + "GWSEP"
    nt.links.new(gw.outputs["Color"], gsep.inputs["Color"])
    # If the mesh has no glow_w, Fac is 0 — we always write the attribute.
    _flex, stm = _flex_strength(nt, gsep.outputs["Red"], 1.0, flex_min)
    tf = nt.nodes.new("ShaderNodeMath"); tf.name = GF + "TINTF"; tf.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], tf.inputs[0]); tf.inputs[1].default_value = float(tint_mix)
    clp = nt.nodes.new("ShaderNodeClamp"); clp.name = GF + "TCLP"
    clp.inputs["Min"].default_value = 0.0
    clp.inputs["Max"].default_value = 1.0
    nt.links.new(tf.outputs[0], clp.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.name = GF + "TINT"
    nt.links.new(clp.outputs[0], mix.inputs[0])
    base_in = bsdf.inputs["Base Color"]
    if base_in.links:
        nt.links.new(base_in.links[0].from_socket, mix.inputs[6])
        nt.links.remove(base_in.links[0])
    else:
        mix.inputs[6].default_value = tuple(base_in.default_value)
    mix.inputs[7].default_value = (float(em_col[0]), float(em_col[1]), float(em_col[2]), 1.0)
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion"); ao.name = GF + "AO"
    ao.inputs["Distance"].default_value = 0.10
    aom = nt.nodes.new("ShaderNodeMix"); aom.data_type = "RGBA"; aom.blend_type = "MULTIPLY"; aom.name = GF + "AOM"
    aom.inputs[0].default_value = 0.34
    aom.inputs[6].default_value = (float(em_col[0]), float(em_col[1]), float(em_col[2]), 1.0)
    nt.links.new(ao.outputs["Color"], aom.inputs[7])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(aom.outputs[2], bsdf.inputs["Emission Color"])
    lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.name = GF + "LW"
    lw.inputs["Blend"].default_value = 0.50
    # Hotter where the belly faces the camera, dimmer in creases — keeps form.
    cmap = nt.nodes.new("ShaderNodeMapRange"); cmap.name = GF + "CORE"
    cmap.inputs["From Min"].default_value = 0.0
    cmap.inputs["From Max"].default_value = 1.0
    cmap.inputs["To Min"].default_value = 0.42
    cmap.inputs["To Max"].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], cmap.inputs["Value"])
    stm_c = nt.nodes.new("ShaderNodeMath"); stm_c.name = GF + "COREM"; stm_c.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], stm_c.inputs[0]); nt.links.new(cmap.outputs["Result"], stm_c.inputs[1])
    ems = nt.nodes.new("ShaderNodeMath"); ems.name = GF + "EMS"; ems.operation = "MULTIPLY"
    nt.links.new(stm_c.outputs[0], ems.inputs[0]); ems.inputs[1].default_value = float(em_str)
    r1 = nt.nodes.new("ShaderNodeMath"); r1.name = GF + "R1"; r1.operation = "SUBTRACT"
    r1.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], r1.inputs[1])
    r2 = nt.nodes.new("ShaderNodeMath"); r2.name = GF + "R2"; r2.operation = "MULTIPLY"
    nt.links.new(r1.outputs[0], r2.inputs[0]); nt.links.new(stm.outputs[0], r2.inputs[1])
    r3 = nt.nodes.new("ShaderNodeMath"); r3.name = GF + "R3"; r3.operation = "MULTIPLY"
    nt.links.new(r2.outputs[0], r3.inputs[0]); r3.inputs[1].default_value = float(rim)
    ems2 = nt.nodes.new("ShaderNodeMath"); ems2.name = GF + "EMS2"; ems2.operation = "ADD"
    nt.links.new(ems.outputs[0], ems2.inputs[0]); nt.links.new(r3.outputs[0], ems2.inputs[1])
    if "Emission Strength" in bsdf.inputs:
        nt.links.new(ems2.outputs[0], bsdf.inputs["Emission Strength"])
    if float(fibre_bump) > 0:
        tex = nt.nodes.new("ShaderNodeTexCoord"); tex.name = GF + "FCOORD"
        mapn = nt.nodes.new("ShaderNodeMapping"); mapn.name = GF + "FMAP"
        mapn.inputs["Rotation"].default_value = tuple(float(x) for x in fibre_rot)
        nt.links.new(tex.outputs["Object"], mapn.inputs["Vector"])
        wv = nt.nodes.new("ShaderNodeTexWave"); wv.name = GF + "FIBRE"
        wv.wave_type = "BANDS"; wv.bands_direction = "X"; wv.wave_profile = "SIN"
        wv.inputs["Scale"].default_value = float(fibre_scale)
        wv.inputs["Distortion"].default_value = 1.8
        wv.inputs["Detail"].default_value = 4.0
        wv.inputs["Detail Scale"].default_value = 1.5
        wv.inputs["Detail Roughness"].default_value = 0.32
        nt.links.new(mapn.outputs["Vector"], wv.inputs["Vector"])
        bstr = nt.nodes.new("ShaderNodeMath"); bstr.name = GF + "BSTR"; bstr.operation = "MULTIPLY"
        nt.links.new(stm.outputs[0], bstr.inputs[0]); bstr.inputs[1].default_value = float(fibre_bump)
        bump = nt.nodes.new("ShaderNodeBump"); bump.name = GF + "BUMP"
        bump.inputs["Distance"].default_value = 0.012
        nt.links.new(wv.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bstr.outputs[0], bump.inputs["Strength"])
        nrm_in = bsdf.inputs.get("Normal")
        if nrm_in is not None and nrm_in.links:
            nt.links.new(nrm_in.links[0].from_socket, bump.inputs["Normal"])
            nt.links.remove(nrm_in.links[0])
        if nrm_in is not None:
            nt.links.new(bump.outputs["Normal"], nrm_in)
    aov = nt.nodes.new("ShaderNodeOutputAOV"); aov.name = GF + "AOV"; aov.aov_name = "glow_mask"
    rgb = nt.nodes.new("ShaderNodeCombineColor"); rgb.name = GF + "AOVRGB"
    nt.links.new(gsep.outputs["Red"], rgb.inputs["Red"])
    nt.links.new(gsep.outputs["Red"], rgb.inputs["Green"])
    nt.links.new(gsep.outputs["Red"], rgb.inputs["Blue"])
    nt.links.new(rgb.outputs["Color"], aov.inputs["Color"])
    return _flex



def attach_attr_sheets(nt, bsdf, mout, hues_rgb, em_str, flex_min=0.88,
                       tint_mix=0.18, rim=0.28):
    """Porcelain clay stays; add peach / purple / pink sheets from float attrs.

    No UVs, no projection, no copied shells. Soft edge = attribute gradient.
    hues_rgb keys: lats, mid_back, rear_delt -> linear RGB triples.
    """
    _purge(nt)

    def _afac(name, suffix):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.name = GF + suffix
        a.attribute_name = name
        try:
            a.attribute_type = "GEOMETRY"
        except Exception:
            pass
        return a.outputs["Fac"]

    wl = _afac("glow_lats", "WL")
    wm = _afac("glow_mid_back", "WM")
    wr = _afac("glow_rear_delt", "WR")

    s1 = nt.nodes.new("ShaderNodeMath"); s1.name = GF + "WS1"; s1.operation = "ADD"
    nt.links.new(wl, s1.inputs[0]); nt.links.new(wm, s1.inputs[1])
    s2 = nt.nodes.new("ShaderNodeMath"); s2.name = GF + "WS2"; s2.operation = "ADD"
    nt.links.new(s1.outputs[0], s2.inputs[0]); nt.links.new(wr, s2.inputs[1])
    wclp = nt.nodes.new("ShaderNodeClamp"); wclp.name = GF + "WCLP"
    wclp.inputs["Min"].default_value = 0.0
    wclp.inputs["Max"].default_value = 1.0
    nt.links.new(s2.outputs[0], wclp.inputs["Value"])

    def _tint(rgb, fac, suffix):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.name = GF + suffix
        mix.inputs[6].default_value = (0.0, 0.0, 0.0, 1.0)
        mix.inputs[7].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        nt.links.new(fac, mix.inputs[0])
        return mix.outputs[2]

    h_l = hues_rgb.get("lats") or (0.95, 0.32, 0.06)
    h_m = hues_rgb.get("mid_back") or (0.55, 0.12, 0.72)
    h_r = hues_rgb.get("rear_delt") or (0.90, 0.18, 0.55)
    c1 = _tint(h_l, wl, "C1")
    c2 = _tint(h_m, wm, "C2")
    c3 = _tint(h_r, wr, "C3")
    a12 = nt.nodes.new("ShaderNodeMix"); a12.data_type = "RGBA"; a12.blend_type = "ADD"
    a12.name = GF + "A12"; a12.inputs[0].default_value = 1.0
    nt.links.new(c1, a12.inputs[6]); nt.links.new(c2, a12.inputs[7])
    a123 = nt.nodes.new("ShaderNodeMix"); a123.data_type = "RGBA"; a123.blend_type = "ADD"
    a123.name = GF + "A123"; a123.inputs[0].default_value = 1.0
    nt.links.new(a12.outputs[2], a123.inputs[6]); nt.links.new(c3, a123.inputs[7])

    # slight clay tint so the sheet sits on the grooves, not instead of them
    tf = nt.nodes.new("ShaderNodeMath"); tf.name = GF + "TINTF"; tf.operation = "MULTIPLY"
    nt.links.new(wclp.outputs[0], tf.inputs[0]); tf.inputs[1].default_value = float(tint_mix)
    tclp = nt.nodes.new("ShaderNodeClamp"); tclp.name = GF + "TCLP"
    tclp.inputs["Min"].default_value = 0.0
    tclp.inputs["Max"].default_value = 1.0
    nt.links.new(tf.outputs[0], tclp.inputs["Value"])
    mixb = nt.nodes.new("ShaderNodeMix"); mixb.data_type = "RGBA"; mixb.name = GF + "TINT"
    nt.links.new(tclp.outputs[0], mixb.inputs[0])
    base_in = bsdf.inputs["Base Color"]
    if base_in.links:
        nt.links.new(base_in.links[0].from_socket, mixb.inputs[6])
        nt.links.remove(base_in.links[0])
    else:
        mixb.inputs[6].default_value = tuple(base_in.default_value)
    nt.links.new(a123.outputs[2], mixb.inputs[7])
    nt.links.new(mixb.outputs[2], bsdf.inputs["Base Color"])

    flex, stm = _flex_strength(nt, wclp.outputs[0], em_str, flex_min)
    lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.name = GF + "LW"
    lw.inputs["Blend"].default_value = 0.50
    r1 = nt.nodes.new("ShaderNodeMath"); r1.name = GF + "R1"; r1.operation = "SUBTRACT"
    r1.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], r1.inputs[1])
    r2 = nt.nodes.new("ShaderNodeMath"); r2.name = GF + "R2"; r2.operation = "MULTIPLY"
    nt.links.new(r1.outputs[0], r2.inputs[0]); nt.links.new(wclp.outputs[0], r2.inputs[1])
    r3 = nt.nodes.new("ShaderNodeMath"); r3.name = GF + "R3"; r3.operation = "MULTIPLY"
    nt.links.new(r2.outputs[0], r3.inputs[0]); r3.inputs[1].default_value = float(rim)
    ems = nt.nodes.new("ShaderNodeMath"); ems.name = GF + "EMS"; ems.operation = "ADD"
    nt.links.new(stm.outputs[0], ems.inputs[0]); nt.links.new(r3.outputs[0], ems.inputs[1])

    em = nt.nodes.new("ShaderNodeEmission"); em.name = GF + "EM"
    nt.links.new(a123.outputs[2], em.inputs["Color"])
    nt.links.new(ems.outputs[0], em.inputs["Strength"])
    _hook_add(nt, bsdf, mout, em)

    aov = nt.nodes.new("ShaderNodeOutputAOV"); aov.name = GF + "AOV"; aov.aov_name = "glow_mask"
    rgb = nt.nodes.new("ShaderNodeCombineColor"); rgb.name = GF + "AOVRGB"
    nt.links.new(wclp.outputs[0], rgb.inputs["Red"])
    nt.links.new(wclp.outputs[0], rgb.inputs["Green"])
    nt.links.new(wclp.outputs[0], rgb.inputs["Blue"])
    nt.links.new(rgb.outputs["Color"], aov.inputs["Color"])
    return flex


def switch_materials(body_mats, backend, em_col, em_str, flex_min=0.58, fibre_scale=42.0, tint_mix=0.28,
                     fibre_bump=0.32, rim=0.35, body_rim=0.18, fibre_line=0.32, opacity=0.94):
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
        elif backend in ("inscene", "native", "clay"):
            flex_nodes.append(attach_clay_emission(nt, bsdf, mout, em_col, em_str, flex_min,
                                                   tint_mix=tint_mix, fibre_bump=fibre_bump, rim=rim,
                                                   fibre_scale=fibre_scale, body_rim=body_rim,
                                                   fibre_line=fibre_line, opacity=opacity))
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


def _cmp_mix(nt, blend, fac=1.0):
    try:
        n = nt.nodes.new("CompositorNodeMixRGB")
    except Exception:
        n = nt.nodes.new("ShaderNodeMixRGB")
    n.blend_type = blend
    n.inputs[0].default_value = float(fac)
    return n


def setup_compositor(sc, bloom_strength=0.60):
    """Subtle Fog Glow on the glowing muscle only (beauty * glow_mask). Blender 4.2 + 5.x."""
    _ensure_aov(sc, "glow_mask")
    sc.use_nodes = True
    use_group = hasattr(sc, "compositing_node_group")
    if use_group:
        ng = sc.compositing_node_group
        if ng is None:
            ng = __import__("bpy").data.node_groups.new("rig3d_cmp", "CompositorNodeTree")
            sc.compositing_node_group = ng
        nt = ng
    else:
        nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers"); rl.location = (0, 0)
    # Isolate the painted muscle in the beauty (more reliable than the Emission pass).
    mul = _cmp_mix(nt, "MULTIPLY", 1.0); mul.name = "GF_MUL"; mul.location = (280, -80)
    img = rl.outputs["Image"]
    mask = rl.outputs.get("glow_mask")
    if mask is None:
        # Do not fall back to the beauty — multiplying/inverting the image
        # itself lifts G/B on the core and washes saturation.
        blk = nt.nodes.new("CompositorNodeRGB"); blk.name = "GF_NOMASK"
        blk.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        mask = blk.outputs[0]
    nt.links.new(img, mul.inputs[1]); nt.links.new(mask, mul.inputs[2])
    gl = nt.nodes.new("CompositorNodeGlare"); gl.location = (520, -80)
    try:
        gl.glare_type = "FOG_GLOW"
    except Exception:
        try:
            gl.inputs["Type"].default_value = "Fog Glow"
        except Exception:
            pass
    try:
        gl.quality = "HIGH"
    except Exception:
        try:
            gl.inputs["Quality"].default_value = "High"
        except Exception:
            pass
    try:
        gl.threshold = 0.22
    except Exception:
        gl.inputs["Threshold"].default_value = 0.22
    try:
        gl.size = 9
    except Exception:
        if "Size" in gl.inputs:
            gl.inputs["Size"].default_value = 0.70
    try:
        gl.mix = -1.0
    except Exception:
        if "Strength" in gl.inputs:
            gl.inputs["Strength"].default_value = float(bloom_strength)
    if "Smoothness" in getattr(gl, "inputs", {}):
        try:
            gl.inputs["Smoothness"].default_value = 0.18
        except Exception:
            pass
    nt.links.new(mul.outputs[0], gl.inputs["Image"])
    glow_out = gl.outputs.get("Glare") or gl.outputs["Image"]
    # Halo = bloom that spilled *outside* the sharp muscle. Zero the interior so
    # Fog-Glow cannot lift G/B on the already-correct opaque core.
    inv = _cmp_mix(nt, "SUBTRACT", 1.0); inv.name = "GF_INVM"; inv.location = (740, -160)
    inv.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(mask, inv.inputs[2])
    ring = _cmp_mix(nt, "MULTIPLY", 1.0); ring.name = "GF_RING"; ring.location = (740, -80)
    nt.links.new(glow_out, ring.inputs[1])
    nt.links.new(inv.outputs[0], ring.inputs[2])
    fade = _cmp_mix(nt, "MIX", float(bloom_strength)); fade.name = "GF_FADE"; fade.location = (960, -80)
    fade.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(ring.outputs[0], fade.inputs[2])
    add = _cmp_mix(nt, "ADD", 1.0); add.name = "GF_ADD"; add.location = (1180, 40)
    nt.links.new(img, add.inputs[1]); nt.links.new(fade.outputs[0], add.inputs[2])
    if use_group and hasattr(nt, "interface"):
        for item in list(getattr(nt.interface, "items_tree", []) or []):
            try:
                nt.interface.remove(item)
            except Exception:
                pass
        nt.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        gout = nt.nodes.new("NodeGroupOutput"); gout.location = (1420, 40)
        nt.links.new(add.outputs[0], gout.inputs[0])
    else:
        comp = nt.nodes.new("CompositorNodeComposite"); comp.location = (1420, 40)
        nt.links.new(add.outputs[0], comp.inputs["Image"])
    sc.render.use_compositing = True
    try:
        sc.render.use_sequencer = False
    except Exception:
        pass


def teardown_compositor(sc):
    sc.render.use_compositing = False
