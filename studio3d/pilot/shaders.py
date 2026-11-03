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
    wave.inputs["Scale"].default_value = 18.0
    wave.inputs["Distortion"].default_value = 1.5
    wave.inputs["Detail"].default_value = 2.0
    nt.links.new(texcoord.outputs["UV"], wave.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].position = 0.58
    nt.links.new(wave.outputs["Color"], ramp.inputs["Fac"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if glow_srgb and glow_strength > 0:
        em = nt.nodes.new("ShaderNodeEmission")
        g = srgb_to_linear(glow_srgb)
        em.inputs["Color"].default_value = (*g, 1)
        # Modulate emission by striation so fibres stay readable (Day 4 behaviour)
        mod = nt.nodes.new("ShaderNodeMath")
        mod.operation = "MULTIPLY_ADD"
        mod.inputs[1].default_value = 0.35
        mod.inputs[2].default_value = 0.65
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
