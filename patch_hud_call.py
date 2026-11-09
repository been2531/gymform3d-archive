with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "r") as f:
    code = f.read()

old_call = """        p58.draw_v58_safezone_bottom_hud(
            canvas,
            cue_1line=cue_1line,
            tag_pill=tag_pill,
            tag_ctx=tag_ctx,
            pct_val=pct_val,
            is_wrong=is_wrong,
        )"""

new_call = """        p58.draw_v58_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )"""

assert old_call in code, "old_call not found"
code = code.replace(old_call, new_call)
with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "w") as f:
    f.write(code)
print("Patched HUD call")
