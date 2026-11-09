with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "r") as f:
    code = f.read()

# Fix v30.fnt(size, bold=True) -> v30.f_pretendard(size, "extrabold")
code = code.replace('v30.fnt(28, bold=True)', 'v30.f_pretendard(28, "extrabold")')
code = code.replace('v30.fnt(19, bold=True)', 'v30.f_pretendard(19, "bold")')
code = code.replace('v30.fnt(22, bold=True)', 'v30.f_pretendard(22, "extrabold")')

# Replace get_or_build_v62_pools with cached 20-frame pools for steps 2, 3, 4
old_load = "    pools = p62.get_or_build_v62_pools()"
new_load = """    import pickle
    cache_p = "/workspace/daily_info_reels/_cache_fast_arrow_test_pools.pkl"
    if os.path.exists(cache_p):
        with open(cache_p, "rb") as fp:
            pools = pickle.load(fp)
    else:
        pools = {}
        for s_idx in (2, 3, 4):
            fn, i0, i1, is_w, sc, y_st, y_sq, ys_st, ys_sq, x_sh = p62.STEP_POOL_SPECS[s_idx]
            raw = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=24)
            pools[s_idx] = p62.build_v62_fullbleed_pool(raw, s_idx, is_w, scale=sc, y_head_target_stand=y_st, y_head_target_squat=y_sq, y_head_src_stand_720=ys_st, y_head_src_squat_720=ys_sq, x_shift=x_sh)
        with open(cache_p, "wb") as fp:
            pickle.dump(pools, fp)"""

code = code.replace(old_load, new_load)

with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "w") as f:
    f.write(code)
print("Patched build_arrow_4concepts_test_video.py")
