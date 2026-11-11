"""Muscle fibre + emissive glow materials (Day-4-style: fibres stay visible through glow)."""
from __future__ import annotations

import bpy


def srgb_to_linear(c):
    out = []
    for u in c:
        u = u / 255.0
        out.append(u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def mat_muscle_fibre(name, base_srgb, glow_srgb=None, glow_strength=0.0, mix_glow=0.55):
    """Charcoal muscle with UV-aligned striations; optional emissive add (not opaque overlay)."""
    base = srgb_to_linear(base_srgb)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Roughness"].default_value = 0.78
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.06

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 32.0
    wave.inputs["Distortion"].default_value = 2.2
    wave.inputs["Detail"].default_value = 4.0
    nt.links.new(texcoord.outputs["UV"], wave.inputs["Vector"])

    wave2 = nt.nodes.new("ShaderNodeTexWave")
    wave2.wave_type = "BANDS"
    wave2.bands_direction = "X"
    wave2.inputs["Scale"].default_value = 58.0
    wave2.inputs["Distortion"].default_value = 0.8
    wave2.inputs["Detail"].default_value = 3.0
    nt.links.new(texcoord.outputs["UV"], wave2.inputs["Vector"])

    mix_w = nt.nodes.new("ShaderNodeMixRGB")
    mix_w.blend_type = "MULTIPLY"
    mix_w.inputs["Fac"].default_value = 0.55
    nt.links.new(wave.outputs["Color"], mix_w.inputs[1])
    nt.links.new(wave2.outputs["Color"], mix_w.inputs[2])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[1].position = 0.62
    nt.links.new(mix_w.outputs["Color"], ramp.inputs["Fac"])

    dark = srgb_to_linear([max(0, c - 14) for c in base_srgb])
    bright = srgb_to_linear([min(255, c + 10) for c in base_srgb])
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*bright, 1)

    albedo = nt.nodes.new("ShaderNodeMixRGB")
    albedo.blend_type = "MIX"
    albedo.inputs["Fac"].default_value = 0.72
    albedo.inputs["Color1"].default_value = (*base, 1)
    nt.links.new(ramp.outputs["Color"], albedo.inputs["Color2"])
    nt.links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.42
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if glow_srgb and glow_strength > 0:
        em = nt.nodes.new("ShaderNodeEmission")
        g = srgb_to_linear(glow_srgb)
        em.inputs["Color"].default_value = (*g, 1)
        # Modulate emission by striation so fibres stay readable (Day 4 behaviour)
        mod = nt.nodes.new("ShaderNodeMath")
        mod.operation = "MULTIPLY_ADD"
        mod.inputs[1].default_value = 0.28
        mod.inputs[2].default_value = 0.72
        nt.links.new(ramp.outputs["Color"], mod.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = glow_strength
        nt.links.new(mod.outputs["Value"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
        nt.links.new(em.outputs["Emission"], add.inputs[1])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = mix_glow
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(add.outputs["Shader"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)
