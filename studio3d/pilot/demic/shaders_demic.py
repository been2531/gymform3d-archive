"""DEMIC-style: single body surface, UV/vertex-mask emission + fibre stripes + contour."""
from __future__ import annotations

import bpy


def srgb_to_linear(c):
    out = []
    for u in c:
        u = u / 255.0
        out.append(u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def mat_body_demic(name, base_srgb, mask_attr, glow_srgb, glow_strength, mix_glow, fibre_scale, contour_boost):
    base = srgb_to_linear(base_srgb)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.12

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = fibre_scale
    wave.inputs["Distortion"].default_value = 2.0
    wave.inputs["Detail"].default_value = 3.0
    nt.links.new(texcoord.outputs["UV"], wave.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    nt.links.new(wave.outputs["Color"], ramp.inputs["Fac"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.48
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    vcol = nt.nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = mask_attr

    # Mask strength = max of relevant channels (set per form on mesh)
    mask_val = nt.nodes.new("ShaderNodeSeparateColor")
    mask_val.mode = "RGB"
    nt.links.new(vcol.outputs["Color"], mask_val.inputs["Color"])

    # R channel = active muscle mask (baked per form on the mesh)
    glow_w = nt.nodes.new("ShaderNodeMath")
    glow_w.operation = "MULTIPLY"
    glow_w.inputs[1].default_value = 1.0
    nt.links.new(mask_val.outputs["Red"], glow_w.inputs[0])

    # Contour rim inside glow: (1 - facing) * mask
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.28
    rim = nt.nodes.new("ShaderNodeMath")
    rim.operation = "SUBTRACT"
    rim.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], rim.inputs[1])
    rim_m = nt.nodes.new("ShaderNodeMath")
    rim_m.operation = "MULTIPLY"
    nt.links.new(rim.outputs["Value"], rim_m.inputs[0])
    nt.links.new(glow_w.outputs["Value"], rim_m.inputs[1])
    rim_boost = nt.nodes.new("ShaderNodeMath")
    rim_boost.operation = "MULTIPLY"
    rim_boost.inputs[1].default_value = contour_boost
    nt.links.new(rim_m.outputs["Value"], rim_boost.inputs[0])

    glow_core = nt.nodes.new("ShaderNodeMath")
    glow_core.operation = "MULTIPLY"
    nt.links.new(ramp.outputs["Color"], glow_core.inputs[0])
    nt.links.new(glow_w.outputs["Value"], glow_core.inputs[1])

    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation = "MAXIMUM"
    nt.links.new(glow_core.outputs["Value"], strength.inputs[0])
    nt.links.new(rim_boost.outputs["Value"], strength.inputs[1])

    em = nt.nodes.new("ShaderNodeEmission")
    g = srgb_to_linear(glow_srgb)
    em.inputs["Color"].default_value = (*g, 1)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = glow_strength
    nt.links.new(strength.outputs["Value"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], em.inputs["Strength"])

    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = mix_glow
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(add.outputs["Shader"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def mat_shorts_demic(name, rgb_srgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*srgb_to_linear(rgb_srgb), 1)
    p.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in p.inputs:
        p.inputs["Specular IOR Level"].default_value = 0.04
    return m


def mat_machine_demic(name, rgb_srgb, rough):
    return mat_shorts_demic(name, rgb_srgb, rough)


def mat_bar_steel(name, rgb_srgb):
    return mat_shorts_demic(name, rgb_srgb, 0.35)


def mat_bar_accent(name, rgb_srgb, strength=8.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    em = nt.nodes.new("ShaderNodeEmission")
    c = srgb_to_linear(rgb_srgb)
    bs.inputs["Base Color"].default_value = (*c, 1)
    bs.inputs["Roughness"].default_value = 0.4
    em.inputs["Color"].default_value = (*c, 1)
    em.inputs["Strength"].default_value = strength
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bs.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], o.inputs["Surface"])
    return m
