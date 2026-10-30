"""Download the locked 3D set's free assets (CC0 MakeHuman, CC BY 4.0 BodyParts3D) into set_v1.asset_dir and
verify each against studio3d/set_lock.json. Zero cost; assets never go into git."""
import hashlib, json, os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "set_v1.json"))); L = json.load(open(os.path.join(HERE, "set_lock.json")))
for k, a in S["assets"].items():
    p = os.path.join(S["asset_dir"], a["file"]); os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        urllib.request.urlretrieve(a["url"], p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(("OK  " if h == L["assets_sha256"][k] else "BAD ") + k)
