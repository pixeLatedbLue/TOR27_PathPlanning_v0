import json
import os

import numpy as np

from tracks import (Track, _offset_boundaries, _resample_closed, _closed_length,
                    TRACKDRIVE)
from geometry import resample_loop, smooth_loop

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "assets", "f1_circuits.geojson")
_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with open(_DATA) as fh:
            feats = json.load(fh)["features"]
        _CACHE = {}
        for f in feats:
            for key in (f["properties"].get("Location", ""),
                        f["properties"].get("Name", "")):
                if key:
                    _CACHE[key.lower()] = f
    return _CACHE


def f1_names():
    out = []
    for f in _load().values():
        loc = f["properties"].get("Location", "")
        if loc and loc not in out:
            out.append(loc)
    return sorted(out)


def _find(key):
    feats = _load()
    k = key.lower()
    if k in feats:
        return feats[k]
    for name, f in feats.items():
        if k in name:
            return f
    raise ValueError(f"unknown F1 circuit: {key!r} (try one of {f1_names()})")


def _project(coords):
    lon, lat = coords[:, 0], coords[:, 1]
    lat0 = np.radians(lat.mean())
    r = 6371000.0
    x = np.radians(lon - lon.mean()) * np.cos(lat0) * r
    y = np.radians(lat - lat.mean()) * r
    return np.column_stack([x, y])


def load_f1(key, extent=820.0, width=6.5, cone_spacing=4.5, laps_required=3):
    feat = _find(key)
    c = _project(np.array(feat["geometry"]["coordinates"], float))
    if np.linalg.norm(c[0] - c[-1]) < 2.0:
        c = c[:-1]
    span = float((c.max(axis=0) - c.min(axis=0)).max())
    c = c * (extent / span)
    c = resample_loop(c, 420)
    c = smooth_loop(c, 5)
    center = resample_loop(c, 360)
    left = _resample_closed(_offset_boundaries(center, width)[0], cone_spacing)
    right = _resample_closed(_offset_boundaries(center, width)[1], cone_spacing)
    p0 = center[0]
    heading = np.arctan2(center[1, 1] - center[-1, 1], center[1, 0] - center[-1, 0])
    track = Track(TRACKDRIVE, left, right, start_pose=(p0[0], p0[1], heading),
                  centerline=center, closed=True, laps_required=laps_required,
                  length=_closed_length(center))
    track.f1_name = feat["properties"].get("Location", key)
    return track
