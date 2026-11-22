#!/usr/bin/env python3
"""Render plan and verification overlays from the live data dir for visual review."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace")

from coresection.analysis import resolve_parameters  # noqa: E402
from coresection.baffle import build_plan, verification_overlay  # noqa: E402
from coresection.craze import trace_centerlines  # noqa: E402
from coresection.geometry import region_domain, specimen_frame  # noqa: E402
from coresection.store import Store  # noqa: E402

OUT = "/tmp/cs_baffle_render"
store = Store("/tmp/cs_live")
tile = next(item for item in store.list_tiles() if store.results(item["id"]))
craze = next(item for item in store.results(tile["id"]) if item["kind"] == "craze")
payload = {"kind": "craze", "options": {}, "apply_neutral": True}
params, _ = resolve_parameters(store, tile, payload)
frame = specimen_frame(store.blobs, tile, params["photo_id"])
thin, _, _, _ = trace_centerlines(frame, region_domain(frame.extent, tile, None), params["options"])
height, width = thin.shape
rows, cols = np.nonzero(thin)
interior = (rows > 40) & (rows < height - 40) & (cols > 40) & (cols < width - 40)
k = np.flatnonzero(interior)[len(np.flatnonzero(interior)) // 2]
inlet = [(cols[k] + 0.5) / width, (rows[k] + 0.5) / height]


def nearest(u, v):
    d = (rows - v * height) ** 2 + (cols - u * width) ** 2
    i = int(np.argmin(d))
    return [(cols[i] + 0.5) / width, (rows[i] + 0.5) / height]


plan = build_plan(store, tile["id"], {
    "title": "Render check plan",
    "photo_id": params["photo_id"],
    "craze_result_id": craze["id"],
    "baffle_radius_px": 14,
    "inlets": [inlet],
    "segments": [
        {"label": "Observed crack L1", "points": [inlet, nearest(inlet[0] + 0.05, inlet[1] + 0.04),
                                                   nearest(inlet[0] + 0.09, inlet[1] + 0.08)]},
        {"label": "Observed crack L2", "points": [inlet, nearest(inlet[0] - 0.05, inlet[1] - 0.04),
                                                   nearest(inlet[0] - 0.09, inlet[1] - 0.08)]},
    ],
    "exits": [nearest(0.06, 0.5), nearest(0.94, 0.5), nearest(0.5, 0.06)],
    "candidates": [nearest(inlet[0] + 0.045, inlet[1] + 0.035),
                   nearest(inlet[0] - 0.045, inlet[1] - 0.035)],
})
plan_blob = store.blobs.require(plan["pictures"]["plan_overlay"])
with Image.open(plan_blob) as image:
    image.save(OUT + "-plan.png")
    print("plan overlay saved", image.size)

# Record two observations (one tracer, one clear, rest blank) and render verification.
from coresection.baffle import expected_table, record_verification  # noqa: E402
keys = [row["key"] for row in expected_table(plan)]
updated = record_verification(store, plan["id"], {
    "version": plan["version"],
    "photo_id": params["photo_id"],
    "note": "Render check",
    "entries": [
        {"key": keys[0], "state": "tracer", "notes": "colored halo at inlet"},
        {"key": keys[1], "state": "clear", "notes": "no color reached this crack"},
    ],
})
with Image.open(store.blobs.require(updated["pictures"]["verification_overlay"])) as image:
    image.save(OUT + "-verify.png")
    print("verification overlay saved", image.size)
print("chosen:", plan["candidates"][plan["selected_candidate_index"]]["candidate_key"])
for row in expected_table(plan):
    print(" ", row["key"], row["kind"], row["expected_state"])
