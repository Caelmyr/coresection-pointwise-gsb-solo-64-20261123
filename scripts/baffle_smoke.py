#!/usr/bin/env python3
"""End-to-end smoke check for the tracer baffle plan workflow.

Runs entirely against a throwaway data directory and the synthetic training
specimens, so it needs no uploaded photographs.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace")

from coresection.baffle import (  # noqa: E402
    build_plan,
    clear_verification,
    decode_mask,
    public_plan,
    record_verification,
    verification_hints,
)
from coresection.demo import load_training  # noqa: E402
from coresection.store import Store  # noqa: E402
from coresection.web import create_app  # noqa: E402


def find_trace_geometry(network, extent):
    """Pick an inlet, two branch polylines, edge exits and two cut candidates."""
    rows, cols = np.nonzero(network)
    height, width = network.shape
    # Choose an interior traced pixel as the inlet.
    interior = (rows > 40) & (rows < height - 40) & (cols > 40) & (cols < width - 40)
    idx = np.flatnonzero(interior)[len(np.flatnonzero(interior)) // 2]
    inlet = [(cols[idx] + 0.5) / width, (rows[idx] + 0.5) / height]

    def nearest_pixel(point):
        rr, cc = np.nonzero(network)
        d = (rr - point[1] * height) ** 2 + (cc - point[0] * width) ** 2
        k = int(np.argmin(d))
        return (cc[k] + 0.5) / width, (rr[k] + 0.5) / height

    # Two "fine cracks": short polyline walks ending in distinct areas.
    segments = [
        {"label": "Fine crack A", "points": [inlet, nearest_pixel((inlet[0] + 0.05, inlet[1] + 0.04)),
                                              nearest_pixel((inlet[0] + 0.09, inlet[1] + 0.08))]},
        {"label": "Fine crack B", "points": [inlet, nearest_pixel((inlet[0] - 0.05, inlet[1] - 0.04)),
                                              nearest_pixel((inlet[0] - 0.09, inlet[1] - 0.08))]},
    ]
    exits = [
        nearest_pixel((0.06, 0.5)),
        nearest_pixel((0.94, 0.5)),
        nearest_pixel((0.5, 0.06)),
    ]
    candidates = [
        nearest_pixel((inlet[0] + 0.045, inlet[1] + 0.035)),
        nearest_pixel((inlet[0] - 0.045, inlet[1] - 0.035)),
    ]
    return inlet, segments, exits, candidates


def main():
    store = Store("/tmp/cs_baffle_smoke/data")
    tiles, created = load_training(store)
    print("training loaded, created =", created, "tiles =", len(tiles))
    tile = tiles[0]

    # First produce a Fine-line network so the plan can reference its settings.
    app = create_app(data_dir="/tmp/cs_baffle_smoke/data")
    client = app.test_client()
    payload = {"kind": "craze", "photo_id": None, "region_id": None,
               "apply_neutral": True, "options": {}}
    resp = client.post(f"/api/specimens/{tile['id']}/results", json=payload)
    assert resp.status_code == 201, resp.get_data(as_text=True)
    craze = resp.get_json()
    print("craze traced pixels:", craze["metrics"]["traced_pixels"])

    # Re-derive the same raster to locate markers on real traced pixels.
    from coresection.analysis import resolve_parameters
    from coresection.craze import trace_centerlines
    from coresection.geometry import region_domain, specimen_frame
    params, _ = resolve_parameters(store, tile, payload)
    frame = specimen_frame(store.blobs, tile, params["photo_id"])
    area = region_domain(frame.extent, tile, None)
    thin, _, support, _ = trace_centerlines(frame, area, params["options"])
    inlet, segments, exits, candidates = find_trace_geometry(thin, frame.extent)
    print("inlet", inlet, "exits", exits)

    plan_payload = {
        "title": "Smoke tracer plan",
        "photo_id": params["photo_id"],
        "craze_result_id": craze["id"],
        "baffle_radius_px": 12,
        "inlets": [inlet],
        "segments": segments,
        "exits": [[x, y] for x, y in exits],
        "candidates": [[x, y] for x, y in candidates],
    }
    plan = build_plan(store, tile["id"], plan_payload)
    public = public_plan(plan)
    print("plan id:", plan["id"])
    print("selected candidate index:", plan["selected_candidate_index"])
    for candidate in plan["candidates"]:
        print("  candidate", candidate["candidate_key"],
              "edge routes open:", candidate["edge_routes_open"],
              "inlet px:", candidate["inlet_connected_trace_pixels"])
        for exit_row in candidate["exits"]:
            print("    exit", exit_row["key"], exit_row["state"],
                  "margin:", exit_row["marked_at_visible_margin"])
        for seg in candidate["segments"]:
            print("    segment", seg["key"], seg["state"],
                  "inlet frac:", seg["inlet_trace_fraction"], "hidden:", seg["hidden_fraction"])

    # The overlay must exist and be a readable PNG.
    overlay = store.blobs.require(plan["pictures"]["plan_overlay"])
    with Image.open(overlay) as image:
        print("plan overlay:", image.size, image.mode)
        assert image.mode == "RGBA"

    # Stored network round-trips to the same raster.
    blob = store.blobs.get(plan["network"]["digest"])
    restored = decode_mask(blob, plan["network"]["width"], plan["network"]["height"])
    assert restored.shape == thin.shape and np.array_equal(restored, thin)
    print("network raster round-trip ok:", int(np.count_nonzero(restored)), "pixels")

    # Scenario invariants: every marker gets one of the three states.
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    states = {row["state"] for row in chosen["exits"]} | {row["state"] for row in chosen["segments"]}
    assert states <= {"connected", "severed", "obscured"}, states
    print("states present:", sorted(states))

    # Verification flow on the same registered capture (acting as follow-up).
    follow_up = tile["plane"]["photo_id"]
    hints = verification_hints(store, plan, follow_up)
    print("hint radius:", hints["hint_radius_px"], "marker hints:", len(hints["markers"]))
    entries = []
    for row in public["expected_rows"]:
        entries.append({"key": row["key"], "state": "tracer", "notes": "smoke evidence"})
    updated = record_verification(store, plan["id"], {
        "version": plan["version"], "photo_id": follow_up, "note": "same-capture check", "entries": entries,
    })
    assert len(updated["observations"]) == len(entries)
    assert "verification_overlay" in updated["pictures"]
    with Image.open(store.blobs.require(updated["pictures"]["verification_overlay"])) as image:
        print("verification overlay:", image.size, image.mode)

    # Setting one recorded marker back to blank removes its evidence.
    blank_key = entries[0]["key"]
    revised = record_verification(store, plan["id"], {
        "version": updated["version"], "photo_id": follow_up, "note": "one marker unphotographed",
        "entries": [{"key": blank_key, "state": "", "notes": ""}],
    })
    assert blank_key not in revised["observations"]
    print("blank marker removed from evidence:", blank_key)

    # A stale version is rejected.
    try:
        record_verification(store, plan["id"], {
            "version": 1, "photo_id": follow_up, "note": "stale",
            "entries": [{"key": entries[1]["key"], "state": "clear", "notes": ""}],
        })
        raise AssertionError("stale verification version should be rejected")
    except Exception as conflict:  # ConflictError is a RuntimeError
        assert type(conflict).__name__ == "ConflictError", type(conflict).__name__
        print("stale version rejected ok")

    # Blank-clear path.
    cleared = clear_verification(store, plan["id"], {"version": revised["version"]})
    assert cleared["observations"] == {} and "verification_overlay" not in cleared["pictures"]
    print("verification clear ok; final version", cleared["version"])

    # HTTP surface check.
    resp = client.get(f"/api/specimens/{tile['id']}/baffle-plans")
    assert resp.status_code == 200 and resp.get_json()["items"], resp.get_data(as_text=True)
    resp = client.get(f"/api/baffle-plans/{plan['id']}")
    assert resp.status_code == 200
    resp = client.get(f"/api/baffle-plans/{plan['id']}/image/plan_overlay")
    assert resp.status_code == 200 and resp.mimetype == "image/png"
    print("HTTP surface ok")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
