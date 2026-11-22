#!/usr/bin/env python3
"""Controlled raster tests for the baffle connectivity logic."""
import sys

import numpy as np

sys.path.insert(0, "/workspace")

from coresection.baffle import (  # noqa: E402
    CONNECTED,
    OBSCURED,
    SEVERED,
    candidate_footprints,
    choose_baffle,
    disk_mask,
    edge_labels,
    encode_mask,
    decode_mask,
    evaluate_scenario,
)


def tree_network(shape):
    """Inlet trunk splits once; each branch ends at a different margin."""
    height, width = shape
    network = np.zeros(shape, dtype=bool)
    junction_col = 60
    junction_row = 60
    # Trunk: left margin (inlet side) to the junction.
    network[junction_row, : junction_col + 1] = True
    # Upper branch: junction up to row 2, then right to the right margin.
    network[: junction_row + 1, junction_col] = True
    network[2, junction_col:] = True
    # Lower branch: junction down to row 118, then right to the right margin.
    network[junction_row:, junction_col] = True
    network[height - 3, junction_col:] = True
    return network, (junction_row, junction_col)


def main():
    height, width = 120, 160
    extent = (width, height)
    network, (junction_row, junction_col) = tree_network((height, width))
    support = np.ones((height, width), dtype=bool)
    _, edge_reaching, _ = edge_labels(network, support)
    assert edge_reaching, "tree network should reach the visible margin"

    inlet = [(2 + 0.5) / width, (junction_row + 0.5) / height]
    markers = {
        "inlet": [{"key": "I1", "kind": "inlet", "point": inlet}],
        "exit": [
            {"key": "E1", "kind": "exit",
             "point": [(width - 3 + 0.5) / width, (2 + 0.5) / height]},
            {"key": "E2", "kind": "exit",
             "point": [(width - 3 + 0.5) / width, (height - 3 + 0.5) / height]},
        ],
        "candidate": [
            # C1 blocks the upper branch right above the junction -> only E1 cut.
            {"key": "C1", "kind": "candidate",
             "point": [(junction_col + 0.5) / width, (40 + 0.5) / height]},
            # C2 blocks the trunk before the junction -> both branches cut.
            {"key": "C2", "kind": "candidate",
             "point": [(30 + 0.5) / width, (junction_row + 0.5) / height]},
        ],
        "segment": [
            {"key": "L1", "kind": "segment",
             "point": [(junction_col + 0.5) / width, (20 + 0.5) / height]},
            {"key": "L2", "kind": "segment",
             "point": [(junction_col + 0.5) / width, (100 + 0.5) / height]},
        ],
    }
    segments = [
        {"key": "L1", "label": "upper fine crack",
         "points": [[(junction_col + 0.5) / width, (junction_row + 0.5) / height],
                    [(junction_col + 0.5) / width, (20 + 0.5) / height],
                    [(100 + 0.5) / width, (2 + 0.5) / height]]},
        {"key": "L2", "label": "lower fine crack",
         "points": [[(junction_col + 0.5) / width, (junction_row + 0.5) / height],
                    [(junction_col + 0.5) / width, (100 + 0.5) / height],
                    [(100 + 0.5) / width, (height - 3 + 0.5) / height]]},
    ]

    baseline = evaluate_scenario(network, support, markers, segments,
                                 np.zeros(network.shape, bool), inlet, extent)
    assert baseline["edge_routes_open"] >= 1
    assert all(row["state"] == CONNECTED for row in baseline["exits"]), "no baffle: all exits reachable"
    assert all(row["state"] == CONNECTED for row in baseline["segments"])
    baseline_open = baseline["edge_routes_open"]

    footprints = candidate_footprints(network, markers, 3.0, extent)
    assert len(footprints) == 2 and all(item["covered_trace_pixels"] > 0 for item in footprints)

    scenarios = []
    for index, footprint in enumerate(footprints):
        result = evaluate_scenario(network, support, markers, segments, footprint["footprint"], inlet, extent)
        result.update({"candidate_index": index, "candidate_key": footprint["key"], "viable": True})
        scenarios.append(result)
    c1, c2 = scenarios

    e1_c1 = next(row for row in c1["exits"] if row["key"] == "E1")
    e2_c1 = next(row for row in c1["exits"] if row["key"] == "E2")
    assert e1_c1["state"] == SEVERED and e2_c1["state"] == CONNECTED, "C1 cuts only the upper branch"
    assert e1_c1["marked_at_visible_margin"] and e2_c1["marked_at_visible_margin"]
    l1_c1 = next(row for row in c1["segments"] if row["key"] == "L1")
    l2_c1 = next(row for row in c1["segments"] if row["key"] == "L2")
    assert l1_c1["state"] == SEVERED and 0 < l1_c1["hidden_fraction"] < 1
    assert l2_c1["state"] == CONNECTED

    # Trunk blockage severs both branches.
    assert all(row["state"] == SEVERED for row in c2["exits"])
    assert all(row["state"] == SEVERED for row in c2["segments"])
    assert c2["edge_routes_open"] <= c1["edge_routes_open"] <= baseline_open
    # The branch baffle keeps one branch's margin outlet open; the trunk baffle
    # removes more inlet-connected margin trace than the branch baffle.
    assert c1["edge_routes_open"] > 0
    assert c2["edge_routes_open"] < c1["edge_routes_open"] < baseline_open

    # The obscure rule: a marker sitting inside the blocked disk cannot be judged.
    hidden_marker = dict(markers)
    hidden_marker["exit"] = [
        {"key": "E1", "kind": "exit", "point": markers["candidate"][0]["point"]},
        markers["exit"][1],
    ]
    hidden_scenario = evaluate_scenario(network, support, hidden_marker, segments,
                                        footprints[0]["footprint"], inlet, extent)
    assert hidden_scenario["exits"][0]["state"] == OBSCURED

    # Choice rule: fewer open edge routes, then longest inlet-connected trace.
    chosen = choose_baffle(scenarios)
    ordered = sorted(scenarios, key=lambda item: (item["edge_routes_open"],
                                                  -item["inlet_connected_trace_pixels"],
                                                  item["candidate_index"]))
    assert ordered[0]["candidate_index"] == chosen

    # Mask codec and disk geometry.
    blob, stored_width, stored_height = encode_mask(network)
    assert np.array_equal(decode_mask(blob, stored_width, stored_height), network)
    disk = disk_mask(extent, markers["candidate"][0]["point"], 3.0)
    assert np.count_nonzero(disk & network) >= 5
    print("controlled baffle logic ok; chosen =", scenarios[chosen]["candidate_key"],
          "routes open:", [item["edge_routes_open"] for item in scenarios],
          "inlet px:", [item["inlet_connected_trace_pixels"] for item in scenarios])


if __name__ == "__main__":
    main()
