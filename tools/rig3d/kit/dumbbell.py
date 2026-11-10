"""Diamond knurl on a procedural dumbbell handle (authored geometry, CC0)."""
from __future__ import annotations
import math
import numpy as np


def add_handle_knurl(ob, axis=(1.0, 0.0, 0.0), amp=0.0007, around=14, along=22, radius=0.018):
    """Displace handle verts (near the cylinder radius) with a crossed-sine knurl."""
    me = ob.data
    n = len(me.vertices)
    if n < 16:
        return {"knurl_verts": 0}
    co = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ax = np.asarray(axis, np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    # project off-axis
    along_v = co @ ax
    radial = co - np.outer(along_v, ax)
    r = np.linalg.norm(radial, axis=1)
    mid = float(np.median(r[r > 1e-5])) if np.any(r > 1e-5) else 0.016
    handle = (r > mid * 0.55) & (r < mid * 1.45) & (np.abs(along_v) < 0.095)
    if not np.any(handle):
        handle = r < float(radius) * 1.6
    # angle around axis: use a stable perpendicular
    ref = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, ref); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(ax, u)
    rn = radial / (r[:, None] + 1e-12)
    theta = np.arctan2(radial @ v, radial @ u)
    span = float(np.ptp(along_v[handle])) or 0.2
    x01 = (along_v - along_v.min()) / (span + 1e-12)
    wave = np.sin(around * theta + along * math.pi * x01) * np.sin(around * theta - along * math.pi * x01)
    off = np.zeros(n)
    off[handle] = float(amp) * wave[handle]
    co = co + rn * off[:, None]
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    return {"knurl_verts": int(handle.sum()), "amp": float(amp), "median_r": round(mid, 4)}
