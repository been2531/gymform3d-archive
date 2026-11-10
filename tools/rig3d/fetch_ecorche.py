#!/usr/bin/env python3
"""Download the free Charleeh/ecorche GLB into gitignored scratch.

The binary is third-party and must not enter git. See ECORCHE_LICENSE.md.
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST_DIR = os.path.join(ROOT, "_scratch", "ecorche")
DEST = os.path.join(DEST_DIR, "ecorche.glb")
# raw.githubusercontent.com: the release-size file (not the meshopt build).
URL = "https://raw.githubusercontent.com/Charleeh/ecorche/main/model/ecorche.glb"
EXPECTED = 14_314_536
STRUCT_URL = "https://raw.githubusercontent.com/Charleeh/ecorche/main/model/ecorche.structures.json"
STRUCT_DEST = os.path.join(DEST_DIR, "ecorche.structures.json")


def _ok(path, expected=None):
    if not os.path.isfile(path):
        return False
    if expected is None:
        return os.path.getsize(path) > 0
    return os.path.getsize(path) == int(expected)


def _get(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"fetch {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(f"  wrote {os.path.getsize(dest)} bytes", flush=True)


def main():
    if _ok(DEST, EXPECTED):
        print(f"already present: {DEST} ({EXPECTED} bytes)")
    else:
        _get(URL, DEST)
        if not _ok(DEST, EXPECTED):
            size = os.path.getsize(DEST) if os.path.isfile(DEST) else 0
            raise SystemExit(f"size mismatch: got {size}, expected {EXPECTED}")
    if not _ok(STRUCT_DEST):
        try:
            _get(STRUCT_URL, STRUCT_DEST)
        except Exception as e:
            print(f"structures.json skipped: {e}")
    print("ok", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
