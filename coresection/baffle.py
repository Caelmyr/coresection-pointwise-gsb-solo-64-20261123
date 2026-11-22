"""Single-baffle sealing plans for tracer tests on a traced fine-line network.

The plan reuses the exact Fine-line network raster produced by ``craze.py`` and
answers a connectivity question on the registered plane:

* which traced routes still carry tracer from the inlet once one baffle is set,
* which routes are visibly cut by the chosen baffle,
* and which routes pass under the baffle footprint and cannot be judged from the
  surface image.

After the experiment the same plan is opened on a newly registered capture. The
original expectations stay frozen; observed marker states start blank and are
only filled in from evidence on the follow-up photograph.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from .analysis import capture_provenance, specimen_provenance
from .art import INK, SHADOW, base_rgba, font, stored_preview
from .craze import trace_centerlines
from .geometry import region_domain, specimen_frame
from .model import (
    InputError,
    collection,
    dictionary,
    enumeration,
    identity,
    model_version,
    numeric_pair,
    scalar,
    string,
)
from .parameters import TOOLS


# Three surface states rendered on the plan overlay.
CONNECTED = "connected"
SEVERED = "severed"
OBSCURED = "obscured"
STATE_LABELS = {
    CONNECTED: "Still connected to inlet",
    SEVERED: "Visibly cut by baffle",
    OBSCURED: "Hidden under baffle - cannot judge",
}
STATE_COLORS = {
    CONNECTED: (94, 205, 154),
    SEVERED: (232, 116, 92),
    OBSCURED: (196, 178, 122),
}
# Follow-up evidence states. A missing record means "not photographed / blank".
TRACER_SEEN = "tracer"
TRACER_ABSENT = "clear"
OBSERVATION_STATES = {
    TRACER_SEEN: "Tracer color visible",
    TRACER_ABSENT: "No tracer visible",
}
OBSERVATION_COLORS = {
    TRACER_SEEN: (232, 92, 126),
    TRACER_ABSENT: (120, 196, 220),
}

SEGMENT_SAMPLE_STEP_PX = 2.0
HINT_MIN_RADIUS_PX = 7
HINT_MAX_RADIUS_PX = 60
MARKER_LIMITS = {"inlet": (1, 1), "segment": (2, 2), "exit": (2, 8), "candidate": (2, 2)}


# --------------------------------------------------------------------------- #
# Raster persistence
# --------------------------------------------------------------------------- #

def encode_mask(mask):
    image = Image.fromarray(np.packbits(mask.astype(np.uint8), axis=1))
    stream = BytesIO()
    image.save(stream, format="PNG", compress_level=9)
    return stream.getvalue(), mask.shape[1], mask.shape[0]


def decode_mask(blob, width, height):
    with Image.open(BytesIO(blob)) as image:
        packed = np.asarray(image, dtype=np.uint8)
    restored = np.unpackbits(packed, axis=1)[:, :width].astype(bool)
    if restored.shape != (height, width):
        raise InputError("The stored network raster no longer matches its declared size.")
    return restored


# --------------------------------------------------------------------------- #
# Marker parsing
# --------------------------------------------------------------------------- #

def parse_markers(payload):
    raw = dictionary(payload, "Baffle plan markers")
    parsed = {}
    point_kinds = ("inlet", "exit", "candidate")
    for kind in point_kinds:
        minimum, maximum = MARKER_LIMITS[kind]
        entries = collection(raw.get(kind + "s"), kind + " markers", minimum=minimum, maximum=maximum)
        parsed[kind] = []
        for index, entry in enumerate(entries):
            point = numeric_pair(entry, kind + " marker", lower=0, upper=1)
            parsed[kind].append({
                "key": f"{kind[0].upper()}{index + 1}",
                "kind": kind,
                "point": [round(point[0], 6), round(point[1], 6)],
            })
    segments_raw = collection(raw.get("segments"), "observed crack segments", minimum=2, maximum=2)
    segments = []
    for index, entry in enumerate(segments_raw):
        item = dictionary(entry, "crack segment")
        vertices = collection(item.get("points"), "crack segment vertices", minimum=2, maximum=12)
        points = [numeric_pair(coords, "segment vertex", lower=0, upper=1) for coords in vertices]
        segments.append({
            "key": f"L{index + 1}",
            "kind": "segment",
            "label": string(item.get("label"), "Segment label", maximum=40, required=False)
            or f"Observed crack {index + 1}",
            "points": [[round(x, 6), round(y, 6)] for x, y in points],
        })
    parsed["segment"] = [
        {"key": segment["key"], "kind": "segment", "point": segment["points"][0]}
        for segment in segments
    ]
    return parsed, segments


def parse_plan_payload(payload):
    data = dictionary(payload, "Baffle plan")
    title = string(data.get("title"), "Plan title", maximum=110)
    photo_id = identity(data.get("photo_id"), "Photo ID")
    region_id = identity(data.get("region_id"), "Region ID") if data.get("region_id") else None
    craze_id = identity(data.get("craze_result_id"), "Fine-line network result ID") \
        if data.get("craze_result_id") else None
    radius = scalar(data.get("baffle_radius_px", 14), "Baffle radius", lower=2.0, upper=120.0)
    parsed, segments = parse_markers(data)
    return {
        "title": title,
        "photo_id": photo_id,
        "region_id": region_id,
        "craze_result_id": craze_id,
        "radius_px": round(float(radius), 3),
        "markers": parsed,
        "segments": segments,
    }


# --------------------------------------------------------------------------- #
# Raster geometry helpers
# --------------------------------------------------------------------------- #

def pixel_at(point, extent):
    width, height = extent
    column = int(round(point[0] * width - 0.5))
    row = int(round(point[1] * height - 0.5))
    return min(max(row, 0), height - 1), min(max(column, 0), width - 1)


def disk_mask(extent, point, radius):
    width, height = extent
    row, column = pixel_at(point, extent)
    yy, xx = np.ogrid[:height, :width]
    return (xx - column) ** 2 + (yy - row) ** 2 <= radius ** 2


def disk_on_network(network, point, radius):
    return disk_mask(network.shape[::-1], point, radius) & network


def nearest_network(network, point):
    rows, cols = np.nonzero(network)
    if not len(rows):
        return None
    tree = cKDTree(np.stack((cols + 0.5, rows + 0.5), axis=1))
    width, height = network.shape[1], network.shape[0]
    distance, _ = tree.query([point[0] * width, point[1] * height])
    return float(distance)


def label_inlet_component(network, inlet_point, blocked=None):
    active = network if blocked is None else network & ~blocked
    labels, count = ndimage.label(active, structure=ndimage.generate_binary_structure(2, 2))
    if count == 0:
        return labels, 0, active
    row, column = pixel_at(inlet_point, (network.shape[1], network.shape[0]))
    inlet_label = int(labels[row, column])
    if inlet_label == 0:
        nearest = nearest_network(active, inlet_point)
        if nearest is None or nearest > 6:
            raise InputError("Mark the inlet directly on a traced fine line.")
        # Snap to the closest traced pixel when the click landed beside the line.
        rows, cols = np.nonzero(active)
        tree = cKDTree(np.stack((cols + 0.5, rows + 0.5), axis=1))
        _, index = tree.query([inlet_point[0] * network.shape[1], inlet_point[1] * network.shape[0]])
        inlet_label = int(labels[rows[index], cols[index]])
    return labels, inlet_label, active


def edge_labels(network, support):
    """Component labels and the traced pixels that touch the visible margin."""
    eroded = ndimage.binary_erosion(support, structure=np.ones((3, 3), dtype=bool), border_value=0)
    margin = support & ~eroded
    labels, _ = ndimage.label(network, structure=ndimage.generate_binary_structure(2, 2))
    reaching = {int(value) for value in np.unique(labels[margin]) if value}
    return labels, reaching, margin & network


# --------------------------------------------------------------------------- #
# Scenario evaluation
# --------------------------------------------------------------------------- #

def marker_state_on_labels(labels, inlet_label, blocked, point, extent, snap=6.0):
    """Classify one point marker against a labeled network scenario."""
    row, column = pixel_at(point, extent)
    if blocked[row, column]:
        return OBSCURED
    label = int(labels[row, column])
    if label == 0:
        active = labels > 0
        distance = nearest_network(active, point)
        if distance is None or distance > snap:
            raise InputError("Each marked crack point must sit on a traced fine line.")
        rows, cols = np.nonzero(active)
        tree = cKDTree(np.stack((cols + 0.5, rows + 0.5), axis=1))
        _, index = tree.query([point[0] * extent[0], point[1] * extent[1]])
        label = int(labels[rows[index], cols[index]])
    return CONNECTED if label == inlet_label else SEVERED


def polyline_samples(points, extent, step=SEGMENT_SAMPLE_STEP_PX):
    width, height = extent
    xy = np.asarray(points, dtype=np.float64) * [width, height]
    samples = []
    for start, finish in zip(xy[:-1], xy[1:]):
        length = float(np.linalg.norm(finish - start))
        count = max(2, int(np.ceil(length / step)))
        samples.extend(np.linspace(start, finish, count, endpoint=False))
    samples.append(xy[-1])
    return np.asarray(samples, dtype=np.float32)


def segment_state(labels, inlet_label, blocked, points, extent):
    samples = polyline_samples(points, extent)
    rows = np.clip(np.rint(samples[:, 1] - 0.5).astype(int), 0, extent[1] - 1)
    cols = np.clip(np.rint(samples[:, 0] - 0.5).astype(int), 0, extent[0] - 1)
    sample_labels = labels[rows, cols]
    hidden = blocked[rows, cols]
    total = len(samples)
    hidden_fraction = float(np.count_nonzero(hidden)) / total
    inlet_fraction = float(np.count_nonzero(sample_labels == inlet_label)) / total
    # A segment with no traced support at all is an input problem.
    if np.count_nonzero(sample_labels) == 0 and not hidden.any():
        raise InputError("Each observed crack segment must follow a traced fine line.")
    if inlet_fraction >= 0.5:
        return CONNECTED, round(inlet_fraction, 4), round(hidden_fraction, 4)
    if hidden_fraction >= 0.5:
        return OBSCURED, round(inlet_fraction, 4), round(hidden_fraction, 4)
    return SEVERED, round(inlet_fraction, 4), round(hidden_fraction, 4)


def evaluate_scenario(network, support, markers, segments, blocked, inlet_point, extent):
    labels, inlet_label, active = label_inlet_component(network, inlet_point, blocked)
    base_labels, edge_reaching, edge_trace = edge_labels(network, support)
    inlet_pixels = int(np.count_nonzero(labels == inlet_label))
    # Count traced inlet-connected pixels that still reach the visible margin.
    inlet_edge_pixels = int(np.count_nonzero(edge_trace & (labels == inlet_label))) if inlet_label else 0
    open_edge_routes = inlet_edge_pixels
    exit_rows = []
    for marker in markers["exit"]:
        state = marker_state_on_labels(labels, inlet_label, blocked, marker["point"], extent)
        base_row, base_col = pixel_at(marker["point"], extent)
        base_label = int(base_labels[base_row, base_col])
        at_margin = base_label in edge_reaching
        exit_rows.append({
            "key": marker["key"],
            "point": marker["point"],
            "state": state,
            "marked_at_visible_margin": at_margin,
        })
    segment_rows = []
    for segment in segments:
        state, inlet_fraction, hidden_fraction = segment_state(
            labels, inlet_label, blocked, segment["points"], extent,
        )
        segment_rows.append({
            "key": segment["key"],
            "label": segment["label"],
            "points": segment["points"],
            "state": state,
            "inlet_trace_fraction": inlet_fraction,
            "hidden_fraction": hidden_fraction,
        })
    return {
        "inlet_connected_trace_pixels": int(inlet_pixels),
        "edge_routes_open": open_edge_routes,
        "exits": exit_rows,
        "segments": segment_rows,
        "labels": labels,
        "inlet_label": inlet_label,
    }


def candidate_footprints(network, markers, radius, extent):
    footprints = []
    for marker in markers["candidate"]:
        footprint = disk_on_network(network, marker["point"], radius)
        covered = int(np.count_nonzero(footprint))
        if covered == 0:
            distance = nearest_network(network, marker["point"])
            raise InputError(
                f"Baffle candidate {marker['key']} does not cover any traced line "
                f"(nearest trace is {distance:.1f} px). Move it onto a crack."
            )
        footprints.append({
            "key": marker["key"],
            "point": marker["point"],
            "radius_px": radius,
            "covered_trace_pixels": covered,
            "footprint": footprint,
        })
    return footprints


def choose_baffle(scenarios):
    """Minimize open edge routes, then maximize inlet-reachable trace."""
    viable = [item for item in scenarios if item["viable"]]
    if not viable:
        return None
    viable.sort(key=lambda item: (
        item["edge_routes_open"],
        -item["inlet_connected_trace_pixels"],
        item["candidate_index"],
    ))
    return viable[0]["candidate_index"]


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #

def resolve_craze_settings(store, tile, craze_id):
    if craze_id is None:
        return None
    record = store.result(craze_id)
    if record["kind"] != "craze":
        raise InputError("The referenced result is not a Fine-line network measurement.")
    if record["tile_id"] != tile["id"]:
        raise InputError("Choose a Fine-line network measured on this specimen.")
    return record


def build_plan(store, specimen_id, payload):
    settings = parse_plan_payload(payload)
    tile = store.tile(specimen_id)
    craze = resolve_craze_settings(store, tile, settings["craze_result_id"])
    photo_id = settings["photo_id"]
    if craze is not None:
        photo_id = craze["parameters"]["photo_id"]
        settings["region_id"] = craze["parameters"].get("region_id")
        options = dict(craze["parameters"]["options"])
    else:
        # Re-trace with the same Fine-line network defaults the measurement form uses.
        options = {key: field["default"] for key, field in TOOLS["craze"]["fields"].items()}
    frame = specimen_frame(store.blobs, tile, photo_id)
    area = region_domain(frame.extent, tile, _region(tile, settings["region_id"]))
    thin, _, support, _ = trace_centerlines(frame, area, options)
    extent = frame.extent
    if np.count_nonzero(thin) < 30:
        raise InputError("The Fine-line network trace is too sparse to plan a baffle on.")
    markers = settings["markers"]
    segments = settings["segments"]
    inlet_point = markers["inlet"][0]["point"]
    radius = settings["radius_px"]
    footprints = candidate_footprints(thin, markers, radius, extent)
    inlet_row, inlet_col = pixel_at(inlet_point, extent)
    for footprint in footprints:
        if footprint["footprint"][inlet_row, inlet_col]:
            raise InputError(
                f"Baffle candidate {footprint['key']} covers the injection point; "
                "a baffle cannot seal the inlet itself."
            )

    baseline = evaluate_scenario(thin, support, markers, segments, np.zeros(thin.shape, bool),
                                 inlet_point, extent)
    scenarios = []
    for index, footprint in enumerate(footprints):
        result = evaluate_scenario(thin, support, markers, segments, footprint["footprint"],
                                   inlet_point, extent)
        result.update({
            "candidate_index": index,
            "candidate_key": footprint["key"],
            "viable": True,
            "radius_px": radius,
        })
        scenarios.append(result)
    chosen_index = choose_baffle(scenarios)
    if chosen_index is None:
        raise InputError("No baffle candidate can be placed on the traced network.")
    chosen = scenarios[chosen_index]
    chosen_footprint = footprints[chosen_index]["footprint"]

    network_blob, network_width, network_height = encode_mask(thin)
    network_digest = store.blobs.put(network_blob)
    overlay = plan_overlay(frame, thin, chosen, chosen_footprint, markers, segments, radius, extent)
    overlay_digest = stored_preview(overlay, store.blobs)

    plan = {
        "title": settings["title"],
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "region_id": settings["region_id"],
        "craze_result_id": settings["craze_result_id"],
        "craze_options": options,
        "network": {
            "digest": network_digest,
            "width": network_width,
            "height": network_height,
        },
        "markers": _frozen_markers(markers, segments),
        "baffle_radius_px": radius,
        "baseline": _public_scenario(baseline),
        "candidates": [
            {
                **_public_scenario(item),
                "candidate_key": item["candidate_key"],
                "candidate_point": footprints[item["candidate_index"]]["point"],
                "covered_trace_pixels": footprints[item["candidate_index"]]["covered_trace_pixels"],
            }
            for item in scenarios
        ],
        "selected_candidate_index": chosen_index,
        "selection_reason": (
            f"Least inlet-connected trace still reaching the visible margin "
            f"({chosen['edge_routes_open']} px), then the longest inlet-connected trace."
        ),
        "pictures": {"plan_overlay": overlay_digest},
        "observations": {},
        "observation_photo_id": None,
        "observation_note": "",
        "version": 1,
    }
    return store.save_baffle_plan(plan)


def _region(tile, region_id):
    if not region_id:
        return None
    for region in tile["regions"]:
        if region["id"] == region_id:
            return region
    raise InputError("The chosen sampling region was removed.")


def _frozen_markers(markers, segments):
    return {
        "inlet": markers["inlet"],
        "exit": markers["exit"],
        "candidate": markers["candidate"],
        "segment": markers["segment"],
        "segments": segments,
    }


def _public_scenario(scenario):
    return {
        "inlet_connected_trace_pixels": scenario["inlet_connected_trace_pixels"],
        "edge_routes_open": scenario["edge_routes_open"],
        "exits": scenario["exits"],
        "segments": scenario["segments"],
    }


# --------------------------------------------------------------------------- #
# Plan overlay drawing
# --------------------------------------------------------------------------- #

def _draw_marker(draw, point, extent, color, label, glyph_radius, filled=False):
    row, column = pixel_at(point, extent)
    box = (column - glyph_radius, row - glyph_radius, column + glyph_radius, row + glyph_radius)
    draw.ellipse(box, outline=SHADOW, width=4)
    if filled:
        draw.ellipse(box, fill=color, outline=SHADOW, width=2)
    draw.ellipse(box, outline=color, width=3)
    draw.text((column + glyph_radius + 5, row - glyph_radius - 3), label,
              fill=color, stroke_width=3, stroke_fill=SHADOW)


def plan_overlay(frame, network, scenario, blocked, markers, segments, radius, extent):
    image = base_rgba(frame)
    width, height = image.size
    short_edge = min(width, height)
    line_width = max(2, round(short_edge / 240))
    glyph_radius = max(6, round(short_edge / 82))
    rgb = np.asarray(image, dtype=np.float32).copy()
    labels = scenario["labels"]
    inlet_label = scenario["inlet_label"]
    connected = network & (labels == inlet_label)
    severed = network & (labels != inlet_label) & (labels > 0)
    rgb[connected, :3] = rgb[connected, :3] * 0.22 + np.array(STATE_COLORS[CONNECTED], np.float32) * 0.78
    rgb[severed, :3] = rgb[severed, :3] * 0.22 + np.array(STATE_COLORS[SEVERED], np.float32) * 0.78
    # Widen one-pixel skeleton lines so the states stay legible.
    widened = ndimage.maximum_filter(network, size=3)
    rgb[widened & ~network, :3] = rgb[widened & ~network, :3] * 0.5 + np.array((118, 108, 92), np.float32) * 0.5
    image = Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)).convert("RGBA")

    # Baffle footprint: opaque hatched disk with a strong ring so it reads as a
    # physical cover rather than another trace color.
    baffle_point = markers["candidate"][scenario["candidate_index"]]["point"]
    brow, bcol = pixel_at(baffle_point, extent)
    rpx = max(4, round(radius))
    overlay_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_layer)
    box = (bcol - rpx, brow - rpx, bcol + rpx, brow + rpx)
    overlay_draw.ellipse(box, fill=(*STATE_COLORS[OBSCURED], 138))
    for offset in range(-rpx, rpx + 1, max(4, rpx // 2)):
        overlay_draw.line((bcol - rpx, brow + offset, bcol + rpx, brow + offset),
                          fill=(48, 42, 34, 110), width=1)
    overlay_draw.ellipse(box, outline=(38, 32, 26, 255), width=4)
    overlay_draw.ellipse((bcol - 3, brow - 3, bcol + 3, brow + 3), fill=(38, 32, 26, 255))
    image = Image.alpha_composite(image, overlay_layer)
    draw = ImageDraw.Draw(image)
    draw.text((bcol + rpx + 6, brow - 8), "Baffle " + scenario["candidate_key"],
              fill=INK, stroke_width=3, stroke_fill=SHADOW)

    # Observed crack polylines on top of the trace.
    for segment in scenario["segments"]:
        points = [pixel_at(point, extent)[::-1] for point in
                  next(item["points"] for item in segments if item["key"] == segment["key"])]
        color = STATE_COLORS[segment["state"]]
        draw.line(points, fill=SHADOW, width=line_width + 4, joint="curve")
        draw.line(points, fill=color, width=line_width + 1, joint="curve")

    _draw_marker(draw, markers["inlet"][0]["point"], extent, (245, 226, 130), "Inlet",
                 glyph_radius, filled=True)
    for marker in markers["exit"]:
        row = next(item for item in scenario["exits"] if item["key"] == marker["key"])
        _draw_marker(draw, marker["point"], extent, STATE_COLORS[row["state"]], marker["key"], glyph_radius)
    for marker in markers["candidate"]:
        if marker["key"] == scenario["candidate_key"]:
            continue
        rrow, rcol = pixel_at(marker["point"], extent)
        ring = (rcol - glyph_radius, rrow - glyph_radius, rcol + glyph_radius, rrow + glyph_radius)
        draw.ellipse(ring, outline=(214, 210, 196), width=3)
        draw.text((rcol + glyph_radius + 5, rrow - glyph_radius - 3), marker["key"] + " unused",
                  fill=(226, 224, 212), stroke_width=3, stroke_fill=SHADOW)

    # Legend on a dark bottom strip, clear of the specimen markings.
    typeface = font(max(12, round(short_edge / 52)))
    strip_height = max(30, round(short_edge * 0.075))
    strip = Image.new("RGBA", (width, strip_height), (30, 34, 32, 214))
    image.alpha_composite(strip, (0, height - strip_height))
    draw = ImageDraw.Draw(image)
    x_cursor = 12
    y_text = height - strip_height + strip_height // 2 - 7
    for state in (CONNECTED, SEVERED, OBSCURED):
        draw.rectangle((x_cursor, y_text, x_cursor + 20, y_text + 15), fill=STATE_COLORS[state])
        draw.text((x_cursor + 26, y_text - 2), STATE_LABELS[state], font=typeface, fill=INK)
        x_cursor += 30 + int(draw.textlength(STATE_LABELS[state], font=typeface)) + 26
    return image


# --------------------------------------------------------------------------- #
# Follow-up verification
# --------------------------------------------------------------------------- #

def hint_radius_px(extent):
    short_edge = min(extent)
    return int(min(HINT_MAX_RADIUS_PX, max(HINT_MIN_RADIUS_PX, round(short_edge / 45))))


def local_tracer_hint(frame, point, radius):
    """A saturation-based hint; never an automatic verdict.

    The user confirms what the follow-up photograph shows. The hint only reports
    the fraction of strongly saturated pixels that differ in hue from the local
    neutral matrix, so colored tracer stands out against gray-brown rock.
    """
    row, column = pixel_at(point, frame.extent)
    height, width = frame.valid.shape
    r0, r1 = max(0, row - radius), min(height, row + radius + 1)
    c0, c1 = max(0, column - radius), min(width, column + radius + 1)
    linear = frame.linear[r0:r1, c0:c1]
    valid = frame.valid[r0:r1, c0:c1] & ~frame.clipped[r0:r1, c0:c1]
    if np.count_nonzero(valid) < 12:
        return {"supported_fraction": 0.0, "saturated_fraction": 0.0, "sampled_pixels": 0}
    rgb = np.clip(linear[valid], 1e-5, 1)
    strongest = rgb.max(axis=1)
    weakest = rgb.min(axis=1)
    saturation = (strongest - weakest) / strongest
    median_sat = float(np.median(saturation))
    tracer_like = saturation > max(0.32, median_sat + 0.18)
    return {
        "supported_fraction": round(float(np.count_nonzero(valid)) / linear[:, :, 0].size, 4),
        "saturated_fraction": round(float(np.count_nonzero(tracer_like)) / len(rgb), 4),
        "sampled_pixels": int(np.count_nonzero(valid)),
    }


def verify_hints(store, plan, photo_id):
    tile = store.tile(plan["tile_id"])
    frame = specimen_frame(store.blobs, tile, photo_id)
    radius = hint_radius_px(frame.extent)
    hints = {}
    for kind in ("inlet", "exit", "segment"):
        for marker in plan["markers"][kind]:
            hints[marker["key"]] = local_tracer_hint(frame, marker["point"], radius)
    return frame, radius, hints


def observation_keys(plan):
    keys = [plan["markers"]["inlet"][0]["key"]]
    keys.extend(item["key"] for item in plan["markers"]["segment"])
    keys.extend(item["key"] for item in plan["markers"]["exit"])
    return keys


def parse_observations(plan, payload):
    data = dictionary(payload, "Baffle verification")
    entries = collection(data.get("entries"), "Observation entries", minimum=1, maximum=32)
    allowed = set(observation_keys(plan))
    result = {}
    for entry in entries:
        item = dictionary(entry, "Observation entry")
        key = string(item.get("key"), "Marker key", maximum=8)
        if key not in allowed:
            raise InputError(f"Marker {key} is not part of this plan.")
        if key in result:
            raise InputError("Each marker can carry one observation.")
        raw_state = item.get("state", "")
        if raw_state == "":
            # Explicit blank means the marker was not photographed; any earlier
            # evidence for it must be removed rather than silently retained.
            result[key] = None
            continue
        state = enumeration(raw_state, OBSERVATION_STATES, "Observed state")
        notes = string(item.get("notes"), "Observation notes", maximum=280, required=False)
        result[key] = {"state": state, "notes": notes, "photo_id": None}
    return result


def verification_overlay(frame, plan):
    image = base_rgba(frame).convert("RGBA")
    width, height = image.size
    short_edge = min(width, height)
    glyph = max(6, round(short_edge / 80))
    observations = plan.get("observations", {})
    hint = hint_radius_px(frame.extent)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    groups = (
        ("inlet", plan["markers"]["inlet"]),
        ("segment", plan["markers"]["segment"]),
        ("exit", plan["markers"]["exit"]),
    )
    for _, markers in groups:
        for marker in markers:
            row, column = pixel_at(marker["point"], frame.extent)
            seen = observations.get(marker["key"])
            if seen is None:
                # Blank = not photographed: dashed hollow hint circle, no verdict.
                color = (236, 232, 216)
                box = (column - hint, row - hint, column + hint, row + hint)
                for angle in range(0, 360, 24):
                    draw.arc(box, start=angle, end=angle + 12, fill=(*color, 220), width=2)
            else:
                color = OBSERVATION_COLORS[seen["state"]]
            marker_box = (column - glyph, row - glyph, column + glyph, row + glyph)
            draw.ellipse(marker_box, fill=(*color, 235 if seen else 90), outline=SHADOW, width=3)
            draw.text((column + glyph + 5, row - glyph - 3), marker["key"],
                      fill=color, stroke_width=3, stroke_fill=SHADOW)
    # Baffle positions drawn as hollow rings, including the unused candidate.
    for index, marker in enumerate(plan["markers"]["candidate"]):
        row, column = pixel_at(marker["point"], frame.extent)
        chosen = index == plan["selected_candidate_index"]
        tone = STATE_COLORS[OBSCURED] if chosen else (170, 170, 158)
        rpx = max(4, round(plan["baffle_radius_px"]))
        draw.ellipse((column - rpx, row - rpx, column + rpx, row + rpx),
                     outline=(*tone, 240), width=4)
        draw.text((column + rpx + 5, row - rpx - 2),
                  marker["key"] + (" placed" if chosen else " unused"),
                  fill=tone, stroke_width=3, stroke_fill=SHADOW)
    image = Image.alpha_composite(image, layer)

    # Legend distinguishes frozen evidence from deliberately blank markers.
    draw = ImageDraw.Draw(image)
    typeface = font(max(12, round(short_edge / 52)))
    strip_height = max(30, round(short_edge * 0.075))
    strip = Image.new("RGBA", (width, strip_height), (30, 34, 32, 214))
    image.alpha_composite(strip, (0, height - strip_height))
    draw = ImageDraw.Draw(image)
    legend = (
        (OBSERVATION_COLORS[TRACER_SEEN], OBSERVATION_STATES[TRACER_SEEN]),
        (OBSERVATION_COLORS[TRACER_ABSENT], OBSERVATION_STATES[TRACER_ABSENT]),
        ((236, 232, 216), "Not photographed - left blank"),
        (STATE_COLORS[OBSCURED], "Baffle footprint"),
    )
    x_cursor = 12
    y_text = height - strip_height + strip_height // 2 - 7
    for color, label in legend:
        draw.rectangle((x_cursor, y_text, x_cursor + 20, y_text + 15), fill=color)
        draw.text((x_cursor + 26, y_text - 2), label, font=typeface, fill=INK)
        x_cursor += 30 + int(draw.textlength(label, font=typeface)) + 26
    return image


def expected_table(plan):
    """Flat expectation rows reused by the API and the evidence table."""
    baseline = {item["key"]: item for item in plan["baseline"]["exits"]}
    baseline_segments = {item["key"]: item for item in plan["baseline"]["segments"]}
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    chosen_exits = {item["key"]: item for item in chosen["exits"]}
    chosen_segments = {item["key"]: item for item in chosen["segments"]}
    rows = []
    inlet = plan["markers"]["inlet"][0]
    rows.append({
        "key": inlet["key"], "kind": "inlet", "label": "Injection point",
        "point": inlet["point"], "expected_state": CONNECTED,
    })
    for marker in plan["markers"]["segment"]:
        scenario = chosen_segments[marker["key"]]
        rows.append({
            "key": marker["key"], "kind": "segment",
            "label": next(item["label"] for item in plan["markers"]["segments"] if item["key"] == marker["key"]),
            "point": marker["point"], "expected_state": scenario["state"],
            "inlet_trace_fraction": scenario["inlet_trace_fraction"],
            "hidden_fraction": scenario["hidden_fraction"],
        })
    for marker in plan["markers"]["exit"]:
        scenario = chosen_exits[marker["key"]]
        rows.append({
            "key": marker["key"], "kind": "exit",
            "label": f"Margin outlet {marker['key']}",
            "point": marker["point"], "expected_state": scenario["state"],
            "marked_at_visible_margin": scenario["marked_at_visible_margin"],
            "baseline_state": baseline[marker["key"]]["state"],
        })
    return rows


def public_plan(plan):
    return {
        **{key: plan[key] for key in plan if key not in {"network"}},
        "network": plan["network"],
        "expected_rows": expected_table(plan),
        "state_labels": STATE_LABELS,
        "observation_state_labels": OBSERVATION_STATES,
    }


def summarize_plan(plan):
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    recorded = sum(1 for value in plan.get("observations", {}).values() if value)
    return {
        "id": plan["id"],
        "title": plan["title"],
        "created_at": plan["created_at"],
        "selected_candidate": chosen["candidate_key"],
        "edge_routes_open": chosen["edge_routes_open"],
        "observation_count": recorded,
        "observation_photo_id": plan.get("observation_photo_id"),
        "version": plan["version"],
        "overlay_url": None,
    }


def verification_hints(store, plan, photo_id):
    frame, radius, hints = verify_hints(store, plan, photo_id)
    return {
        "photo_id": photo_id,
        "hint_radius_px": radius,
        "hint_note": "Saturation is a viewing aid only; the observed state is set from the photograph.",
        "markers": hints,
    }


def record_verification(store, plan_id, payload):
    data = dictionary(payload, "Baffle verification")
    plan = store.baffle_plan(plan_id)
    expected_version = model_version(data)
    store.require_version(plan, expected_version)
    photo_id = identity(data.get("photo_id"), "Follow-up photo ID")
    tile = store.tile(plan["tile_id"])
    registered = {tile["plane"]["photo_id"], *tile["registrations"]}
    if photo_id not in registered:
        raise InputError("Register the follow-up photograph on the specimen plane before verifying.")
    frame = specimen_frame(store.blobs, tile, photo_id)
    changes = parse_observations(plan, data)
    note = string(data.get("note"), "Verification note", maximum=600, required=False)

    def apply(record):
        merged = dict(record.get("observations", {}))
        for key, entry in changes.items():
            if entry is None:
                merged.pop(key, None)
            else:
                merged[key] = {**entry, "photo_id": photo_id}
        record["observations"] = merged
        record["observation_photo_id"] = photo_id
        record["observation_note"] = note
        if merged:
            record["pictures"]["verification_overlay"] = stored_preview(
                verification_overlay(frame, record), store.blobs,
            )
        else:
            record["pictures"].pop("verification_overlay", None)

    return store.update_baffle_plan(
        plan_id, apply, f"Recorded tracer verification for {plan['title']}",
    )


def clear_verification(store, plan_id, payload):
    plan = store.baffle_plan(plan_id)
    expected_version = model_version(dictionary(payload, "Baffle verification"))
    store.require_version(plan, expected_version)

    def apply(record):
        record["observations"] = {}
        record["observation_photo_id"] = None
        record["observation_note"] = ""
        record["pictures"].pop("verification_overlay", None)

    return store.update_baffle_plan(plan_id, apply, f"Cleared tracer verification for {plan['title']}")
