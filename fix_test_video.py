with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "r") as f:
    code = f.read()

# Replace multiprocessing + raw_frames list with streaming single-process write directly to ffmpeg stdin
old_tail = code[code.index("    tasks = []"):]
new_tail = """    out_mp4 = os.path.join(OUT_DIR, "arrow_4concepts_live_test.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "18",
        "-movflags", "+faststart",
        out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    total_written = 0
    for f_start, f_end, s_idx, mode in segments:
        frames_s, kps_s = pools[s_idx]
        n_p = len(frames_s)
        seg_len = f_end - f_start
        for i in range(seg_len):
            g_idx = f_start + i
            u_seg = i / float(max(1, seg_len - 1))
            u_rep = 0.5 * (1.0 - math.cos(2.0 * math.pi * u_seg))
            p_idx = int(round(u_rep * (n_p - 1)))
            p_idx = max(0, min(n_p - 1, p_idx))
            phase = (g_idx % 30) / 30.0
            _, rgb = _render_video_frame_worker((g_idx, s_idx, mode, frames_s[p_idx], kps_s[p_idx], u_rep, phase))
            proc.stdin.write(rgb.tobytes())
            total_written += 1

    proc.stdin.close()
    proc.wait()
    print(f"[test_video] Rendered {out_mp4} ({total_written} frames) in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
"""
code = code[:code.index("    tasks = []")] + new_tail
with open("/workspace/daily_info_reels/build_arrow_4concepts_test_video.py", "w") as f:
    f.write(code)
print("Updated build_arrow_4concepts_test_video.py")
