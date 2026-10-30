"""QC for the locked 3D set (t562u: 'structure it so it never drifts').
(L1) every asset file matches its sha256 in studio3d/set_lock.json (mannequin, rig, weights, morphs, anatomy)
(L2) set_v1.json + builder/compositor code hash == lock (materials, lights, cameras, equipment can't change silently)
(L3) every rendered frame's meta set_hash == lock (all 4 steps / both cameras rendered from the identical set)
(L4) the posed 3D joint angles meet configs/motion_standards.json for the form (true 3D, no 2D projection error)
usage: python3 studio3d/qc_set.py [meta.json ...]      |   --write-lock (only when a NEW set version is approved)"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_hash(S):        # must equal build_scene.set_hash
    h = hashlib.sha256(json.dumps({k: v for k, v in S.items() if not k.startswith("_")}, sort_keys=True).encode())
    for a in S["assets"].values():
        h.update(sha(os.path.join(S["asset_dir"], a["file"])).encode())
    for k in ("build_scene.py", "composite.py"):
        h.update(sha(os.path.join(HERE, k)).encode())
    return h.hexdigest()[:16]


def check_angles(meta, std):
    ex = std["exercises"][meta["exercise"]]
    out = []
    if meta["phase"] > 0.05:
        return out
    for sd, a in meta["angles_3d"].items():
        if meta["form"] == "wrong":
            for m in ex["mistakes"]:
                sg = m["signature"]
                if sg.get("metric") == "arm_torso":
                    ok = a["arm_torso"] > sg["value"]
                    out.append((f"{sd} {m['id']}: arm_torso {a['arm_torso']} > {sg['value']}", ok))
        else:
            for r in ex["right_form"]:
                if r["metric"] == "arm_torso":
                    lo, hi = r["range"]; t = r.get("tol_deg", 0)
                    out.append((f"{sd} {r['id']}: arm_torso {a['arm_torso']} in {lo}-{hi} (+-{t})", lo - t <= a["arm_torso"] <= hi + t))
        for r in ex["right_form"]:
            if r["metric"] == "elbow_flexion":
                lo, hi = r["range"]; t = r.get("tol_deg", 0)
                out.append((f"{sd} {r['id']}: elbow_flexion {a['elbow_flexion']} in {lo}-{hi} (+-{t})", lo - t <= a["elbow_flexion"] <= hi + t))
    return out


def main():
    S = json.load(open(os.path.join(HERE, "set_v1.json")))
    lock_p = os.path.join(HERE, "set_lock.json")
    files = {k: sha(os.path.join(S["asset_dir"], a["file"])) for k, a in S["assets"].items()}
    hs = set_hash(S)
    if "--write-lock" in sys.argv:
        json.dump({"version": S["version"], "set_hash": hs, "assets_sha256": files}, open(lock_p, "w"), indent=1)
        print("lock written", hs); return 0
    lock = json.load(open(lock_p)); res = []
    bad = [k for k, v in files.items() if lock["assets_sha256"].get(k) != v]
    res.append(("(L1) asset files identical to lock", not bad, ", ".join(bad) or f"{len(files)} files"))
    res.append(("(L2) set config + builder hash == lock", hs == lock["set_hash"], f"{hs} vs {lock['set_hash']}"))
    std = json.load(open(os.path.join(ROOT, "configs", "motion_standards.json")))
    for mp in [a for a in sys.argv[1:] if a.endswith(".json")]:
        m = json.load(open(mp))
        res.append((f"(L3) {os.path.basename(mp)} rendered from the locked set", m["set_hash"] == lock["set_hash"], m["set_hash"]))
        for txt, ok in check_angles(m, std):
            res.append((f"(L4) {os.path.basename(mp)} {txt}", ok, ""))
    for name, ok, info in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}")
    return 0 if all(r[1] for r in res) else 5


if __name__ == "__main__":
    sys.exit(main())
