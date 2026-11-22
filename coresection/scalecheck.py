import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, INK, MAX_PREVIEW, SHADOW, curve_points, font, stored_preview
from .geometry import (
    DEFAULT_CORNERS,
    MIN_SUPPORT_ALPHA,
    SampleFrame,
    canvas_dimensions,
    corners_parameter,
    homography,
    region_domain,
    specimen_frame,
    warp_capture,
)
from .metrology import path_cumulative, path_segments, path_turns, sample_support_along
from .model import (
    InputError,
    dictionary,
    flag,
    identity,
    match_slot,
    model_version,
    numeric_pair,
    scalar,
    string,
    timestamp,
)
from .photos import open_working, uint8_image
from .store import make_id


MAX_CHECKS = 8
MAX_CANDIDATES = 4
QUAD_TOLERANCE = 0.002
ADOPTED_INK = (174, 233, 188)
ALTERNATE_INK = (238, 184, 120)
CANVAS_SHADE = (26, 33, 36, 255)


def scale_state(tile):
    state = dict(tile.get("scale") or {})
    state["checks"] = list(state.get("checks") or [])
    state["candidates"] = list(state.get("candidates") or [])
    state["decision"] = state.get("decision")
    return state


def base_photo(tile):
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the tile plane first.")
    return match_slot(tile["photos"], plane["photo_id"], "photo")


def apply_homography(matrix, points):
    source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mapped = np.c_[source, np.ones(len(source))] @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


def check_fields(payload, photo):
    data = dictionary(payload, "Scale check")
    a = numeric_pair(data.get("a"), "Distance start", lower=0, upper=1)
    b = numeric_pair(data.get("b"), "Distance end", lower=0, upper=1)
    known = scalar(data.get("known_mm"), "Taped distance mm", lower=0.1, upper=100000)
    start, stop = np.asarray([a, b]) * [photo["width"], photo["height"]]
    if np.linalg.norm(stop - start) < 8:
        raise InputError("Mark a taped distance of at least eight photo pixels.")
    return {"name": string(data.get("name"), "Distance name", maximum=60), "a": a, "b": b, "known_mm": known}


def candidate_fields(payload, tile, photo):
    data = dictionary(payload, "Scale candidate")
    corners = corners_parameter(data.get("corners"))
    homography(DEFAULT_CORNERS, corners)
    width = data.get("width_mm") or tile.get("width_mm")
    height = data.get("height_mm") or tile.get("height_mm")
    if width is None or height is None:
        raise InputError("Enter the plate's physical size; the specimen has no dimensions to inherit.")
    width = scalar(width, "Plate width mm", lower=0.1, upper=100000)
    height = scalar(height, "Plate height mm", lower=0.1, upper=100000)
    return {
        "name": string(data.get("name"), "Candidate name", maximum=60),
        "corners": corners,
        "width_mm": width,
        "height_mm": height,
        "size": list(canvas_dimensions(photo, corners, width, height)),
    }


def add_scale_check(store, identifier, payload, check_id=None):
    expected = model_version(payload)
    tile = store.tile(identifier)
    store.require_version(tile, expected)
    check = check_fields(payload, base_photo(tile))
    chosen = identity(check_id, "Scale check ID") if check_id else make_id()
    check["id"] = chosen

    def change(tile):
        state = scale_state(tile)
        if check_id:
            previous = match_slot(state["checks"], chosen, "measured distance")
            state["checks"][state["checks"].index(previous)] = check
        else:
            if len(state["checks"]) >= MAX_CHECKS:
                raise InputError(f"A specimen supports at most {MAX_CHECKS} measured distances.")
            state["checks"].append(check)
        tile["scale"] = state

    return store.edit_tile(identifier, expected, change, "scale", f"Saved measured distance '{check['name']}'")


def remove_scale_check(store, identifier, check_id, payload):
    expected = model_version(payload)
    chosen = identity(check_id, "Scale check ID")

    def change(tile):
        state = scale_state(tile)
        state["checks"].remove(match_slot(state["checks"], chosen, "measured distance"))
        tile["scale"] = state

    return store.edit_tile(identifier, expected, change, "scale", "Removed a measured distance")


def add_scale_candidate(store, identifier, payload):
    expected = model_version(payload)
    tile = store.tile(identifier)
    store.require_version(tile, expected)
    candidate = candidate_fields(payload, tile, base_photo(tile))
    candidate["id"] = make_id()

    def change(tile):
        state = scale_state(tile)
        if len(state["candidates"]) >= MAX_CANDIDATES:
            raise InputError(f"A specimen supports at most {MAX_CANDIDATES} plate candidates.")
        state["candidates"].append(candidate)
        tile["scale"] = state

    return store.edit_tile(identifier, expected, change, "scale", f"Saved plate candidate '{candidate['name']}'")


def remove_scale_candidate(store, identifier, candidate_id, payload):
    expected = model_version(payload)
    chosen = identity(candidate_id, "Scale candidate ID")

    def change(tile):
        state = scale_state(tile)
        state["candidates"].remove(match_slot(state["candidates"], chosen, "plate candidate"))
        tile["scale"] = state

    return store.edit_tile(identifier, expected, change, "scale", "Removed a plate candidate")


def scale_definitions(tile):
    definitions = []
    plane = tile.get("plane")
    if plane:
        definitions.append({
            "id": "plane",
            "name": "Specimen plane",
            "corners": plane["corners"],
            "size": list(plane["size"]),
            "width_mm": tile.get("width_mm"),
            "height_mm": tile.get("height_mm"),
        })
    definitions.extend(dict(candidate) for candidate in scale_state(tile)["candidates"])
    return definitions


def evaluate_definition(definition, checks):
    inverse = np.linalg.inv(homography(DEFAULT_CORNERS, definition["corners"]))
    physical = definition.get("width_mm") and definition.get("height_mm")
    results = []
    for check in checks:
        start, stop = apply_homography(inverse, [check["a"], check["b"]])
        inside = bool(
            np.all(np.abs(np.r_[start, stop] - 0.5) <= 0.5 + QUAD_TOLERANCE)
        )
        row = {"check_id": check["id"], "inside": inside}
        if inside and physical:
            measured = float(np.linalg.norm((stop - start) * [definition["width_mm"], definition["height_mm"]]))
            row["measured_mm"] = round(measured, 4)
            row["residual_mm"] = round(measured - check["known_mm"], 4)
            row["residual_percent"] = round(100 * (measured - check["known_mm"]) / check["known_mm"], 3)
        results.append(row)
    residuals = [row["residual_mm"] for row in results if row.get("residual_mm") is not None]
    width, height = definition["size"]
    return {
        "id": definition["id"],
        "name": definition["name"],
        "corners": definition["corners"],
        "size_px": [int(width), int(height)],
        "physical_mm": [definition.get("width_mm"), definition.get("height_mm")] if physical else None,
        "mm_per_px": [round(definition["width_mm"] / width, 6), round(definition["height_mm"] / height, 6)] if physical else None,
        "results": results,
        "evaluated_checks": len(residuals),
        "max_abs_mm": round(max(map(abs, residuals)), 4) if residuals else None,
        "rms_mm": round(float(math.sqrt(np.mean(np.square(residuals)))), 4) if residuals else None,
    }


def evaluate_scales(tile):
    if not tile.get("plane"):
        raise InputError("Set a base photo and mark the tile plane first.")
    checks = scale_state(tile)["checks"]
    if not checks:
        raise InputError("Mark at least one taped distance on the original photo first.")
    candidates = [evaluate_definition(definition, checks) for definition in scale_definitions(tile)]
    usable = [candidate for candidate in candidates if candidate["evaluated_checks"]]
    complete = [candidate for candidate in usable if candidate["evaluated_checks"] == len(checks)]
    pool = complete or usable
    best = min(pool, key=lambda candidate: (candidate["max_abs_mm"], candidate["rms_mm"])) if pool else None
    recommendation = None
    if best is not None:
        recommendation = {
            "candidate_id": best["id"],
            "reason": (
                f"{best['name']} shows the smallest worst-case residual over {best['evaluated_checks']} "
                f"taped distance(s): max |Δ| {best['max_abs_mm']} mm, RMS {best['rms_mm']} mm."
            ),
        }
    return {"checks": checks, "candidates": candidates, "recommendation": recommendation}


def photo_canvas(blobs, photo):
    rgb, alpha = open_working(blobs, photo)
    image = uint8_image(rgb, alpha)
    image.thumbnail((MAX_PREVIEW, MAX_PREVIEW), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", image.size, CANVAS_SHADE)
    canvas.alpha_composite(image.convert("RGBA"))
    return canvas


def draw_quad(draw, transform, extent, color, label, typeface, row=0):
    quad = [[float(x), float(y)] for x, y in apply_homography(transform, DEFAULT_CORNERS)]
    points = curve_points(quad, extent)
    draw.line([*points, points[0]], fill=SHADOW, width=5, joint="curve")
    draw.line([*points, points[0]], fill=color, width=2, joint="curve")
    top = points[0][1] + 5 + row * (typeface.size + 6)
    draw.text((points[0][0] + 5, top), label, font=typeface, fill=color, stroke_width=2, stroke_fill=SHADOW)


def decision_overlay(blobs, tile, evaluation, chosen_id):
    image = photo_canvas(blobs, base_photo(tile))
    draw = ImageDraw.Draw(image)
    typeface = font(max(15, min(image.size) // 46))
    for row, candidate in enumerate(evaluation["candidates"]):
        chosen = candidate["id"] == chosen_id
        label = candidate["name"] + (" · adopted" if chosen else "")
        transform = homography(DEFAULT_CORNERS, candidate["corners"])
        draw_quad(draw, transform, image.size, ADOPTED_INK if chosen else ALTERNATE_INK, label, typeface, row)
    adopted = next(candidate for candidate in evaluation["candidates"] if candidate["id"] == chosen_id)
    results = {row["check_id"]: row for row in adopted["results"]}
    stroke = max(2, min(image.size) // 300)
    for check in evaluation["checks"]:
        start, stop = curve_points([check["a"], check["b"]], image.size)
        draw.line([start, stop], fill=SHADOW, width=stroke + 3)
        draw.line([start, stop], fill=ACCENT, width=stroke)
        for x, y in (start, stop):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW)
        row = results.get(check["id"], {})
        if row.get("residual_mm") is None:
            label = f"{check['name']}: not covered"
        else:
            label = f"{check['name']}: {check['known_mm']:.1f} → {row['measured_mm']:.1f} mm (Δ {row['residual_mm']:+.2f})"
        middle = ((start[0] + stop[0]) / 2 + 6, (start[1] + stop[1]) / 2 - 12)
        draw.text(middle, label, font=typeface, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def adopt_scale(store, identifier, payload):
    expected = model_version(payload)
    data = dictionary(payload, "Scale decision")
    chosen = string(data.get("candidate_id"), "Scale candidate", maximum=40)
    note = string(data.get("note"), "Decision note", maximum=240, required=False)
    apply = flag(data.get("apply_plane", False), "Apply as specimen plane")
    tile = store.tile(identifier)
    store.require_version(tile, expected)
    evaluation = evaluate_scales(tile)
    match = next((candidate for candidate in evaluation["candidates"] if candidate["id"] == chosen), None)
    if match is None:
        raise InputError("Choose one of the evaluated scale candidates.")
    if not match["evaluated_checks"]:
        raise InputError("That candidate has no usable taped-distance evidence.")
    overlay = stored_preview(decision_overlay(store.blobs, tile, evaluation, chosen), store.blobs)
    decision = {
        "candidate_id": chosen,
        "name": match["name"],
        "note": note,
        "decided_at": timestamp(),
        "max_abs_mm": match["max_abs_mm"],
        "rms_mm": match["rms_mm"],
        "results": match["results"],
        "checks": [{"id": check["id"], "name": check["name"], "known_mm": check["known_mm"]} for check in evaluation["checks"]],
        "definition": {"corners": match["corners"], "size": match["size_px"], "physical_mm": match["physical_mm"]},
        "overlay": overlay,
    }

    def change(tile):
        state = scale_state(tile)
        state["decision"] = decision
        if apply and chosen != "plane":
            candidate = next((item for item in state["candidates"] if item["id"] == chosen), None)
            if candidate is None:
                raise InputError("The chosen plate candidate was removed.")
            plane = tile["plane"]
            previous = {
                "id": make_id(),
                "name": "Previous specimen plane",
                "corners": plane["corners"],
                "size": list(plane["size"]),
                "width_mm": tile.get("width_mm"),
                "height_mm": tile.get("height_mm"),
            }
            tile["plane"] = {"photo_id": plane["photo_id"], "corners": candidate["corners"], "size": list(candidate["size"])}
            tile["width_mm"] = candidate["width_mm"]
            tile["height_mm"] = candidate["height_mm"]
            tile["registrations"] = {}
            tile["regions"] = []
            tile["calibration"] = None
            if previous["width_mm"] and not any(
                np.allclose(previous["corners"], other["corners"], atol=1e-6) for other in state["candidates"]
            ):
                if len(state["candidates"]) >= MAX_CANDIDATES:
                    raise InputError("Remove a plate candidate before applying this one; the list is full.")
                state["candidates"].append(previous)
        tile["scale"] = state

    return store.edit_tile(identifier, expected, change, "scale", f"Adopted plane scale '{match['name']}' for the shift report")


def trace_detail(frame, available, points):
    segments, units = path_segments(frame, points)
    cumulative = path_cumulative(segments)
    pieces = sample_support_along(available, frame, points)
    samples = sum(item["samples"] for item in pieces)
    present = sum(item["supported_samples"] for item in pieces)
    return {
        "points": points,
        "segments": segments,
        "trace_length": cumulative[-1],
        "cumulative": cumulative,
        "turn_angles": path_turns(segments),
        "support_along_line": pieces,
        "overall_support": round(100 * present / samples, 3),
        "length_unit": units,
    }


def candidate_frame(blobs, photo, definition):
    transform = homography(DEFAULT_CORNERS, definition["corners"])
    rgb, alpha = open_working(blobs, photo)
    size = tuple(definition["size"])
    linear, valid, clipped = warp_capture(rgb, alpha, size, transform)
    scale = None
    if definition.get("width_mm") and definition.get("height_mm"):
        scale = (definition["width_mm"] / size[0], definition["height_mm"] / size[1])
    return SampleFrame(linear, valid, photo, tuple(transform.ravel()), scale, clipped)


def classify_visibility(old_percent, new_percent):
    full, tiny = 99.5, 0.5
    if old_percent >= full and new_percent >= full:
        return "both"
    if old_percent >= full and new_percent < tiny:
        return "old_only"
    if new_percent >= full and old_percent < tiny:
        return "new_only"
    if old_percent < tiny and new_percent < tiny:
        return "neither"
    return "partial"


def photo_visibility(alpha, extent, old_inverse, new_inverse, photo_points):
    width, height = extent
    weight = alpha.astype(np.float32)
    segments = []
    for index, (start, stop) in enumerate(zip(photo_points, photo_points[1:]), 1):
        length = float(np.linalg.norm((stop - start) * [width, height]))
        count = min(1800, int(math.ceil(length)) + 1)
        sample = start[None, :] + np.linspace(0, 1, count)[:, None] * (stop - start)[None, :]
        rows, cols = sample[:, 1] * height - 0.5, sample[:, 0] * width - 0.5
        solid = ndimage.map_coordinates(weight, (rows, cols), order=1, mode="nearest") >= MIN_SUPPORT_ALPHA
        old_uv = apply_homography(old_inverse, sample)
        new_uv = apply_homography(new_inverse, sample)
        in_old = np.all(np.abs(old_uv - 0.5) <= 0.5 + QUAD_TOLERANCE, axis=1)
        in_new = np.all(np.abs(new_uv - 0.5) <= 0.5 + QUAD_TOLERANCE, axis=1)
        old_percent = 100 * float(np.mean(solid & in_old))
        new_percent = 100 * float(np.mean(solid & in_new))
        segments.append({
            "index": index,
            "old_plane_percent": round(old_percent, 2),
            "new_plane_percent": round(new_percent, 2),
            "visibility": classify_visibility(old_percent, new_percent),
        })
    return segments


def compare_rows(old_shape, new_shape, visibility):
    same_units = old_shape["length_unit"] == new_shape["length_unit"]
    rows = []
    for old, new, vis in zip(old_shape["segments"], new_shape["segments"], visibility):
        row = {
            "index": new["index"],
            "previous_length": old["length"],
            "new_length": new["length"],
            "old_plane_percent": vis["old_plane_percent"],
            "new_plane_percent": vis["new_plane_percent"],
            "visibility": vis["visibility"],
        }
        if same_units:
            delta = new["length"] - old["length"]
            row["delta"] = round(delta, 5)
            row["delta_percent"] = round(100 * delta / old["length"], 3) if old["length"] else None
        rows.append(row)
    return rows


def remeasure_overlay(blobs, photo, definition, photo_points, shape, old_transform, new_transform):
    image = photo_canvas(blobs, photo)
    draw = ImageDraw.Draw(image)
    typeface = font(max(15, min(image.size) // 46))
    draw_quad(draw, old_transform, image.size, ALTERNATE_INK, "Saved plane", typeface)
    draw_quad(draw, new_transform, image.size, ADOPTED_INK, definition["name"], typeface, 1)
    points = [[float(x), float(y)] for x, y in photo_points]
    pixels = curve_points(points, image.size)
    stroke = max(2, min(image.size) // 300)
    draw.line(pixels, fill=SHADOW, width=stroke + 3, joint="curve")
    draw.line(pixels, fill=ACCENT, width=stroke, joint="curve")
    for segment, (start, stop) in zip(shape["segments"], zip(pixels, pixels[1:])):
        middle = ((start[0] + stop[0]) / 2 + 4, (start[1] + stop[1]) / 2 - 10)
        draw.text(middle, f"{segment['length']:.2f} {segment['units']}", font=typeface, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    for index, (x, y) in enumerate(pixels, 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW)
        draw.text((x + 6, y + 3), str(index), font=typeface, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def remeasure_trace(store, identifier, record_id, payload):
    data = dictionary(payload, "Trace re-measurement")
    chosen = string(data.get("candidate_id"), "Scale candidate", maximum=40)
    tile = store.tile(identifier)
    record = store.metrology(identity(record_id, "Geometry measurement ID"))
    if record["tile_id"] != tile["id"]:
        raise InputError("That geometry record belongs to another specimen.")
    if record["mode"] != "trace":
        raise InputError("Only a traced path can be re-measured under another scale.")
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the tile plane first.")
    if record["source"]["id"] != plane["photo_id"]:
        raise InputError("The saved trace belongs to a different source photo than the current base.")
    old_transform = record["source"].get("plane_to_capture")
    if not old_transform:
        raise InputError("The saved reading has no plane transform to anchor against.")
    definitions = {item["id"]: item for item in scale_definitions(tile)}
    if chosen not in definitions:
        raise InputError("Choose one of the stored scale candidates.")
    definition = definitions[chosen]
    old_transform = np.asarray(old_transform, dtype=np.float64).reshape(3, 3)
    new_transform = homography(DEFAULT_CORNERS, definition["corners"])
    if np.allclose(old_transform, new_transform, atol=1e-9):
        raise InputError("The saved reading already uses that scale.")
    photo = match_slot(tile["photos"], plane["photo_id"], "photo")
    photo_points = apply_homography(old_transform, record["shape"]["points"])
    new_points = apply_homography(np.linalg.inv(new_transform), photo_points)
    point_list = [[round(float(x), 6), round(float(y), 6)] for x, y in new_points]
    notes = []
    if chosen == "plane":
        frame = specimen_frame(store.blobs, tile)
        available = region_domain(frame.extent, tile) & frame.valid
    else:
        frame = candidate_frame(store.blobs, photo, definition)
        available = frame.valid.copy()
        available[0, :] = available[-1, :] = False
        available[:, 0] = available[:, -1] = False
        notes.append("Sampling regions and exclusions belong to the saved plane; the candidate measurement uses raw photo support.")
    shape = trace_detail(frame, available, point_list)
    _, alpha = open_working(store.blobs, photo)
    visibility = photo_visibility(
        alpha, (photo["width"], photo["height"]), np.linalg.inv(old_transform), np.linalg.inv(new_transform), photo_points,
    )
    rows = compare_rows(record["shape"], shape, visibility)
    counts = {key: 0 for key in ("both", "old_only", "new_only", "partial", "neither")}
    for row in rows:
        counts[row["visibility"]] += 1
    old_shape = record["shape"]
    summary = {**counts, "previous_total": old_shape["trace_length"], "new_total": shape["trace_length"]}
    if old_shape["length_unit"] == shape["length_unit"]:
        delta = shape["trace_length"] - old_shape["trace_length"]
        summary["total_delta"] = round(delta, 5)
        summary["total_delta_percent"] = round(100 * delta / old_shape["trace_length"], 3) if old_shape["trace_length"] else None
    title = string(data.get("title") or f"{record['title']} · {definition['name']}", "Measurement title", maximum=90)
    overlay = remeasure_overlay(store.blobs, photo, definition, photo_points, shape, old_transform, new_transform)
    return store.save_metrology({
        "title": title,
        "mode": "trace",
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "region": None,
        "exclusions": [],
        "shape": shape,
        "derived_from": record["id"],
        "comparison": {
            "basis": "plane_scale",
            "derived_from": record["id"],
            "derived_title": record["title"],
            "previous_units": old_shape["length_unit"],
            "new_scale": {"id": definition["id"], "name": definition["name"], "units": shape["length_unit"]},
            "segments": rows,
            "summary": summary,
            "notes": notes,
        },
        "image": stored_preview(overlay, store.blobs),
    }, message=f"Re-measured trace under scale '{definition['name']}': {title}")
