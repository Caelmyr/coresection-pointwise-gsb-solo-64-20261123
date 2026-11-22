#!/usr/bin/env python3
"""Live-server client for the baffle workflow, run against a throwaway data dir."""
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, "/workspace")

BASE = "http://127.0.0.1:19983/api"


def call(path, method="GET", body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read() or b"null")


def main():
    status, training = call("/training", "POST", {})
    assert status in (200, 201)
    tile_id = training["items"][0]["id"]
    print("training tile:", tile_id)

    status, craze = call(f"/specimens/{tile_id}/results", "POST",
                         {"kind": "craze", "options": {}, "apply_neutral": True})
    assert status == 201
    print("craze run:", craze["id"], craze["metrics"]["traced_pixels"], "px")

    # Reuse the trace to choose points that lie on real network pixels.
    from coresection.store import Store
    store = Store("/tmp/cs_live")
    # The plan endpoint computes the trace itself; derive markers from the same
    # pipeline directly so every click lands on a traced pixel.
    from coresection.analysis import resolve_parameters
    from coresection.craze import trace_centerlines
    from coresection.geometry import region_domain, specimen_frame
    payload = {"kind": "craze", "options": {}, "apply_neutral": True}
    params, _ = resolve_parameters(store, store.tile(tile_id), payload)
    frame = specimen_frame(store.blobs, store.tile(tile_id), params["photo_id"])
    thin, _, _, _ = trace_centerlines(frame, region_domain(frame.extent, store.tile(tile_id), None),
                                      params["options"])
    height, width = thin.shape
    rows, cols = np.nonzero(thin)
    interior = (rows > 40) & (rows < height - 40) & (cols > 40) & (cols < width - 40)
    k = np.flatnonzero(interior)[len(np.flatnonzero(interior)) // 2]
    inlet = [(cols[k] + 0.5) / width, (rows[k] + 0.5) / height]

    def nearest(u, v):
        d = (rows - v * height) ** 2 + (cols - u * width) ** 2
        i = int(np.argmin(d))
        return [(cols[i] + 0.5) / width, (rows[i] + 0.5) / height]

    plan_body = {
        "title": "Live API plan",
        "photo_id": params["photo_id"],
        "craze_result_id": craze["id"],
        "baffle_radius_px": 12,
        "inlets": [inlet],
        "segments": [
            {"label": "Crack L1", "points": [inlet, nearest(inlet[0] + 0.05, inlet[1] + 0.04),
                                              nearest(inlet[0] + 0.09, inlet[1] + 0.08)]},
            {"label": "Crack L2", "points": [inlet, nearest(inlet[0] - 0.05, inlet[1] - 0.04),
                                              nearest(inlet[0] - 0.09, inlet[1] - 0.08)]},
        ],
        "exits": [nearest(0.06, 0.5), nearest(0.94, 0.5), nearest(0.5, 0.06)],
        "candidates": [nearest(inlet[0] + 0.045, inlet[1] + 0.035),
                       nearest(inlet[0] - 0.045, inlet[1] - 0.035)],
    }
    status, plan = call(f"/specimens/{tile_id}/baffle-plans", "POST", plan_body)
    assert status == 201, (status, plan)
    print("plan created:", plan["id"], "chosen:",
          plan["candidates"][plan["selected_candidate_index"]]["candidate_key"])
    print("expected rows:", [(r["key"], r["expected_state"]) for r in plan["expected_rows"]])

    status, listing = call(f"/specimens/{tile_id}/baffle-plans")
    assert status == 200 and any(item["id"] == plan["id"] for item in listing["items"])

    status, hints = call(f"/baffle-plans/{plan['id']}/hints", "POST",
                         {"photo_id": params["photo_id"]})
    assert status == 200 and hints["markers"]

    entries = [{"key": row["key"], "state": "tracer", "notes": "live check"}
               for row in plan["expected_rows"][:3]]
    status, updated = call(f"/baffle-plans/{plan['id']}/verification", "PUT",
                           {"version": plan["version"], "photo_id": params["photo_id"],
                            "note": "partial evidence", "entries": entries})
    assert status == 200 and len(updated["observations"]) == 3
    blanks = [row["key"] for row in updated["expected_rows"] if row["key"] not in updated["observations"]]
    print("recorded:", sorted(updated["observations"]), "left blank:", blanks)
    assert blanks, "unrecorded markers must stay blank"

    # Stale version must be rejected with 409.
    try:
        call(f"/baffle-plans/{plan['id']}/verification", "PUT",
             {"version": 1, "photo_id": params["photo_id"], "entries": entries})
        raise AssertionError("stale version should be rejected")
    except urllib.error.HTTPError as error:
        assert error.code == 409, error.code
        print("stale verification 409 ok")

    # Explicit blank removes previously recorded evidence.
    status, trimmed = call(f"/baffle-plans/{plan['id']}/verification", "PUT", {
        "version": updated["version"], "photo_id": params["photo_id"], "entries": [
            {"key": entries[0]["key"], "state": "", "notes": ""},
        ],
    })
    assert status == 200 and entries[0]["key"] not in trimmed["observations"]
    print("blank submission removed:", entries[0]["key"])

    # Validation error path: baffle candidate off the network must return 422.
    bad = dict(plan_body)
    bad["title"] = "Live API plan - invalid candidate"
    bad["candidates"] = [[0.02, 0.02], plan_body["candidates"][1]]
    try:
        call(f"/specimens/{tile_id}/baffle-plans", "POST", bad)
        raise AssertionError("off-network baffle should be rejected")
    except urllib.error.HTTPError as error:
        assert error.code == 422, error.code
        print("validation 422 ok:", json.loads(error.read())["error"])

    print("LIVE API OK")


if __name__ == "__main__":
    main()
