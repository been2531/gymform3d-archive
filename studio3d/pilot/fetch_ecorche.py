#!/usr/bin/env python3
"""Download écorché glb (CC BY-SA 4.0) — not committed to git."""
import hashlib
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "assets3d", "ecorche.glb")
URL = "https://github.com/Charleeh/ecorche/raw/main/model/ecorche.glb"
if not os.path.exists(OUT):
    urllib.request.urlretrieve(URL, OUT)
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print("OK ecorche.glb sha256", h[:16])
