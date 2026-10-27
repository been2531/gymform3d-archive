with open("/workspace/daily_info_reels/preview_v45_luxury_vectors.py", "r") as f:
    code = f.read()

# Add custom draw_v45_white_callout_badge with larger 21pt pure white sub-label + 56pt extrabold number
badge_fn = '''
F_V45_NUM = v30.f_barlow(56, extrabold=True)
F_V45_SUB = v30.f_pretendard(21, "extrabold")

def draw_v45_white_callout_badge(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    align: str = "right",
):
    """
    Renders a high-contrast broadcast callout badge:
    - 56pt extrabold Pure White number (`18°`, `0°`, `+28°`, `95°`, `84%`, etc.)
    - Sleek 3px colored accent underline bar
    - Larger 21pt extrabold PURE WHITE (`#FFFFFF`) sub-label (`VALGUS CAVE`, `KNEES TRACK TOES`, etc.)
      with deep dark shadow cushion for instant readability on mobile Reels.
    """
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_V45_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_V45_SUB)
    tw = tb[2] - tb[0]

    if align == "right":
        nx = x - nw
        tx = x - tw
        bar_x0, bar_x1 = x - min(max(nw, tw), 110), x
    elif align == "center":
        nx = x - nw // 2
        tx = x - tw // 2
        bar_x0, bar_x1 = x - 45, x + 45
    else:
        nx = x
        tx = x
        bar_x0, bar_x1 = x, x + min(max(nw, tw), 110)

    # Deep multi-pass shadow behind number and pure-white sub-label
    for dx_s, dy_s in [(0, 3), (0, -2), (3, 0), (-3, 0), (2, 2)]:
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_V45_NUM, fill=(0, 0, 0, 235))
        sd.text((tx + dx_s, y + 62 + dy_s), tag_str, font=F_V45_SUB, fill=(0, 0, 0, 245))

    # Crisp Pure White Number
    od.text((nx, y), num_str, font=F_V45_NUM, fill=(255, 255, 255, 255))

    # Sleek 3px colored accent bar between number and sub-label
    sd.rounded_rectangle([bar_x0 - 1, y + 55, bar_x1 + 1, y + 59], radius=2, fill=(0, 0, 0, 220))
    od.rounded_rectangle([bar_x0, y + 55, bar_x1, y + 58], radius=2, fill=(*col_main, 255))

    # Crisp Larger (21pt) PURE WHITE Sub-Label
    od.text((tx, y + 62), tag_str, font=F_V45_SUB, fill=(255, 255, 255, 255))
'''

target_block = '''    if step_idx == 1:
        v30.draw_floating_number_badge(od, sd, (325, 1020), "18°", "VALGUS CAVE", col_main, col_soft, align="right")
        v30.draw_floating_number_badge(od, sd, (SAFE_X_RIGHT_BADGE, 875), "84%", "ACL & MCL LOAD", col_main, col_soft, align="left")
    elif step_idx == 2:
        v30.draw_floating_number_badge(od, sd, (325, 1005), "0°", "KNEES TRACK TOES", col_main, col_soft, align="right")
        v30.draw_floating_number_badge(od, sd, (SAFE_X_RIGHT_BADGE, 875), "96%", "VMO & GLUTE LOCK", col_main, col_soft, align="left")
    elif step_idx == 3:
        v30.draw_floating_number_badge(od, sd, (345, 1030), "+28°", "HEELS LIFTING", col_main, col_soft, align="right")
        v30.draw_floating_number_badge(od, sd, (SAFE_X_RIGHT_BADGE, 875), "79%", "LUMBAR & PATELLA", col_main, col_soft, align="left")
    else:
        v30.draw_floating_number_badge(od, sd, (335, 1020), "95°", "PARALLEL DEPTH", col_main, col_soft, align="right")
        v30.draw_floating_number_badge(od, sd, (SAFE_X_RIGHT_BADGE, 875), "98%", "QUADS & GLUTES", col_main, col_soft, align="left")'''

replacement_block = '''    if step_idx == 1:
        draw_v45_white_callout_badge(od, sd, (335, 1010), "18°", "VALGUS CAVE", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (745, 865), "84%", "ACL & MCL LOAD", col_main, col_soft, align="left")
    elif step_idx == 2:
        draw_v45_white_callout_badge(od, sd, (345, 995), "0°", "KNEES TRACK TOES", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (735, 865), "96%", "VMO & GLUTE LOCK", col_main, col_soft, align="left")
    elif step_idx == 3:
        draw_v45_white_callout_badge(od, sd, (348, 1020), "+28°", "HEELS LIFTING", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (730, 865), "79%", "LUMBAR & PATELLA", col_main, col_soft, align="left")
    else:
        draw_v45_white_callout_badge(od, sd, (345, 1010), "95°", "PARALLEL DEPTH", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (745, 865), "98%", "QUADS & GLUTES", col_main, col_soft, align="left")'''

assert target_block in code, "target_block not found"
code = code.replace("def render_v45_luxury_callouts_and_vectors(", badge_fn + "\n\ndef render_v45_luxury_callouts_and_vectors(")
code = code.replace(target_block, replacement_block)

# Also adjust right-side leader pin endpoint from (792, 905) to (738, 898) so it meets the updated badge position cleanly
code = code.replace("(775.0, 965.0), (792.0, 905.0)", "(715.0, 955.0), (735.0, 898.0)")

with open("/workspace/daily_info_reels/preview_v45_luxury_vectors.py", "w") as f:
    f.write(code)
print("Updated preview_v45_luxury_vectors.py successfully")
