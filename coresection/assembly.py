"""Reassemble broken core segments into one archived core string.

The intake keeps every photographed segment with its own provenance.  Two field
orderings are scored against independent evidence:

* lamina continuity across each fracture (tone, bedding angle, edge profile);
* agreement of the legacy depth records where they exist.

Exactly one ordering is archived.  The decision record always states which
consideration lost and why.  Unphotographed intervals are kept as transparent
gap blocks of known length - pixels from other segments are never copied in.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .model import InputError, dictionary, model_version, scalar, string, timestamp
from .photos import open_working, png_bytes, store_capture
from .store import make_id
from .geometry import DEFAULT_CORNERS, canvas_dimensions


SEGMENT_KEYS = ("A", "B", "C")
ARROW_CHOICES = ("down", "up")
GAP_END_CHOICES = ("uphole", "downhole")
MAX_PLANE_EDGE = 1800
EDGE_BAND_FRACTION = 0.08
LAMINA_DECISIVE_MARGIN = 0.08
DEPTH_TOLERANCE_M = 0.05
DEPTH_SCORE_SPAN_M = 0.5
LATERAL_MARGIN = 0.08

# Score weights for a single fracture join.
WEIGHT_TONE = 0.35
WEIGHT_ANGLE = 0.30
WEIGHT_PROFILE = 0.35


@dataclass(frozen=True)
class OrientedSegment:
    key: str
    rgba: np.ndarray
    rotated: bool
    original_size: tuple
    length_mm: float


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------

def _optional_depth(value, label):
    if value in (None, ""):
        return None
    return scalar(value, label, lower=-100000.0, upper=100000.0)


def parse_segment_document(raw, index):
    data = dictionary(raw, f"Segment {index + 1}")
    key = string(data.get("segment_key"), f"Segment {index + 1} key", maximum=8)
    if key not in SEGMENT_KEYS:
        raise InputError(f"Segment key must be one of {', '.join(SEGMENT_KEYS)}.")
    length_mm = scalar(data.get("length_mm"), f"Segment {key} length", lower=0.1, upper=100000.0)
    arrow = string(data.get("downhole_arrow", "down"), f"Segment {key} downhole direction", maximum=8)
    if arrow not in ARROW_CHOICES:
        raise InputError(f"Segment {key} arrow must be 'down' or 'up'.")
    gap_end = data.get("gap_end")
    gap_length_mm = None
    if gap_end:
        gap_end = string(gap_end, f"Segment {key} gap end", maximum=12)
        if gap_end not in GAP_END_CHOICES:
            raise InputError(f"Segment {key} gap end must be 'uphole' or 'downhole'.")
        gap_length_mm = scalar(
            data.get("gap_length_mm"), f"Segment {key} missing interval length", lower=0.1, upper=100000.0,
        )
    return {
        "segment_key": key,
        "length_mm": float(length_mm),
        "downhole_arrow": arrow,
        "old_label": string(data.get("old_label"), f"Segment {key} old number", maximum=60, required=False),
        "old_depth_top_m": _optional_depth(data.get("old_depth_top_m"), f"Segment {key} legacy top depth"),
        "old_depth_bottom_m": _optional_depth(data.get("old_depth_bottom_m"), f"Segment {key} legacy bottom depth"),
        "gap_end": gap_end,
        "gap_length_mm": float(gap_length_mm) if gap_length_mm is not None else None,
        "note": string(data.get("note"), f"Segment {key} note", maximum=600, required=False),
    }


def parse_orders(raw, fixed_key):
    orders_raw = raw.get("orders", raw.get("orderings"))
    if not isinstance(orders_raw, (list, tuple)) or len(orders_raw) != 2:
        raise InputError("Provide exactly two field orderings.")
    orders = []
    for index, raw_order in enumerate(orders_raw):
        item = dictionary(raw_order, f"Field ordering {index + 1}")
        label = string(item.get("label"), f"Field ordering {index + 1} label", maximum=120)
        keys = item.get("segment_keys")
        if not isinstance(keys, (list, tuple)) or len(keys) != 3:
            raise InputError(f"Ordering '{label}' must list all three segment keys.")
        keys = [string(value, f"Ordering '{label}' segment key", maximum=8) for value in keys]
        if sorted(keys) != sorted(SEGMENT_KEYS):
            raise InputError(f"Ordering '{label}' must be a permutation of {', '.join(SEGMENT_KEYS)}.")
        if keys[0] != fixed_key:
            raise InputError(
                f"Ordering '{label}' must keep segment {fixed_key} fixed at the uphole end.",
            )
        orders.append({"order_id": f"order-{index + 1}", "label": label, "segment_keys": keys})
    if orders[0]["segment_keys"] == orders[1]["segment_keys"]:
        raise InputError("The two field orderings must differ in the last two segments.")
    return orders


def duplicate_labels(segments):
    seen = {}
    duplicates = []
    for segment in segments:
        label = segment["old_label"]
        if not label:
            continue
        if label in seen:
            duplicates.append([seen[label], segment["segment_key"], label])
        else:
            seen[label] = segment["segment_key"]
    return duplicates


# ---------------------------------------------------------------------------
# Image orientation and stitching
# ---------------------------------------------------------------------------

def _resize_tile(rgba, target_height):
    image = Image.fromarray(np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8), mode="RGBA")
    ratio = target_height / image.height
    target_width = max(8, round(image.width * ratio))
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8).astype(np.float32) / 255.0, target_width


def build_string(blobs, segments, photo_records, order_keys, total_length_mm):
    """Return stitched RGBA float image and row-based source blocks."""
    by_key = {segment["segment_key"]: segment for segment in segments}
    oriented = {
        key: OrientedSegment(
            key,
            _rgba(blobs, photo_records[key], by_key[key]),
            by_key[key]["downhole_arrow"] == "up",
            (photo_records[key]["width"], photo_records[key]["height"]),
            by_key[key]["length_mm"],
        )
        for key in order_keys
    }
    ppm = min(item.rgba.shape[0] / item.length_mm for item in oriented.values())
    ppm = min(ppm, (MAX_PLANE_EDGE - 1) / total_length_mm)
    target_heights = {key: max(8, round(oriented[key].length_mm * ppm)) for key in order_keys}
    resized = {key: _resize_tile(oriented[key].rgba, target_heights[key]) for key in order_keys}
    canvas_width = max(width for _, width in resized.values())
    gap_px = {
        segment["segment_key"]: max(1, round((segment["gap_length_mm"] or 0) * ppm))
        for segment in segments
    }

    blocks = []
    height = 0
    depth_cursor = 0.0
    for key in order_keys:
        spec = by_key[key]
        if spec["gap_end"] == "uphole":
            row_start = height
            height += gap_px[key]
            blocks.append({
                "kind": "gap", "segment_key": key, "row_start": row_start, "row_end": height,
                "depth_offset_start_mm": round(depth_cursor, 5),
                "depth_offset_end_mm": round(depth_cursor + spec["gap_length_mm"], 5),
                "length_mm": spec["gap_length_mm"],
            })
            depth_cursor += spec["gap_length_mm"]
        tile, tile_width = resized[key]
        x_offset = (canvas_width - tile_width) // 2
        row_start = height
        height += tile.shape[0]
        blocks.append({
            "kind": "segment", "segment_key": key, "row_start": row_start, "row_end": height,
            "x_offset": x_offset, "tile_width": tile_width,
            "rotated": oriented[key].rotated,
            "original_size": list(oriented[key].original_size),
            "depth_offset_start_mm": round(depth_cursor, 5),
            "depth_offset_end_mm": round(depth_cursor + spec["length_mm"], 5),
            "length_mm": spec["length_mm"],
        })
        depth_cursor += spec["length_mm"]
        if spec["gap_end"] == "downhole":
            gap_start = height
            height += gap_px[key]
            blocks.append({
                "kind": "gap", "segment_key": key, "row_start": gap_start, "row_end": height,
                "depth_offset_start_mm": round(depth_cursor, 5),
                "depth_offset_end_mm": round(depth_cursor + spec["gap_length_mm"], 5),
                "length_mm": spec["gap_length_mm"],
            })
            depth_cursor += spec["gap_length_mm"]

    canvas = np.zeros((height, canvas_width, 4), dtype=np.float32)
    for block in blocks:
        if block["kind"] != "segment":
            continue
        tile, _ = resized[block["segment_key"]]
        top, left = block["row_start"], block["x_offset"]
        canvas[top:top + tile.shape[0], left:left + tile.shape[1], :] = tile
    return canvas, blocks, ppm


def _rgba(blobs, photo, spec):
    rgb, alpha = open_working(blobs, photo)
    if spec["downhole_arrow"] == "up":
        rgb = np.rot90(rgb, 2)
        alpha = np.rot90(alpha, 2)
    return np.dstack((rgb, alpha))


# ---------------------------------------------------------------------------
# Fracture evidence: lamina continuity and legacy depth agreement
# ---------------------------------------------------------------------------

def _luma(rgba):
    opaque = rgba[..., 3] >= 0.995
    rgb = rgba[..., :3]
    luma = (rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(axis=2)
    return luma, opaque


def _central_band(rgba, edge, band_rows):
    luma, opaque = _luma(rgba)
    width = luma.shape[1]
    left = int(width * LATERAL_MARGIN)
    right = width - left
    if edge == "bottom":
        band = luma[-band_rows:, left:right]
        mask = opaque[-band_rows:, left:right]
    else:
        band = luma[:band_rows, left:right]
        mask = opaque[:band_rows, left:right]
    return np.where(mask, band, np.nan)


def _dominant_angle(band):
    filled = np.where(np.isnan(band), np.nanmean(band), band)
    gy, gx = np.gradient(filled)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    angle = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    # Eigenvector of the smallest eigenvalue = line direction, modulo 180 deg.
    line_angle = (angle + 90.0) % 180.0
    return line_angle


def _column_profile(band):
    with np.errstate(invalid="ignore"):
        return np.nanmean(band, axis=0)


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3 or np.nanstd(values) < 1e-6:
        return None
    return (values - np.nanmean(values)) / np.nanstd(values)


def _profile_correlation(first, second):
    size = min(len(first), len(second), 64)
    if size < 4:
        return 0.0
    a = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(first)), first)
    b = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(second)), second)
    za, zb = _zscore(a), _zscore(b)
    if za is None or zb is None:
        return 0.0
    return float(np.clip(np.mean(za * zb), -1.0, 1.0))


def join_lamina(upper_rgba, lower_rgba):
    upper_rows = max(4, round(upper_rgba.shape[0] * EDGE_BAND_FRACTION))
    lower_rows = max(4, round(lower_rgba.shape[0] * EDGE_BAND_FRACTION))
    upper_band = _central_band(upper_rgba, "bottom", upper_rows)
    lower_band = _central_band(lower_rgba, "top", lower_rows)
    tone_delta = abs(float(np.nanmean(upper_band)) - float(np.nanmean(lower_band)))
    angle_delta = abs(_dominant_angle(upper_band) - _dominant_angle(lower_band))
    angle_delta = min(angle_delta, 180.0 - angle_delta)
    correlation = _profile_correlation(_column_profile(upper_band), _column_profile(lower_band))
    tone_score = max(0.0, 1.0 - tone_delta / 0.25)
    angle_score = max(0.0, 1.0 - angle_delta / 45.0)
    profile_score = (correlation + 1.0) / 2.0
    score = WEIGHT_TONE * tone_score + WEIGHT_ANGLE * angle_score + WEIGHT_PROFILE * profile_score
    return {
        "score": round(float(score), 4),
        "tone_delta_linear": round(tone_delta, 5),
        "tone_score": round(float(tone_score), 4),
        "lamina_angle_delta_deg": round(float(angle_delta), 3),
        "angle_score": round(float(angle_score), 4),
        "edge_profile_correlation": round(correlation, 4),
        "profile_score": round(float(profile_score), 4),
    }


def join_depth(upper_spec, lower_spec):
    top = upper_spec["old_depth_bottom_m"]
    bottom = lower_spec["old_depth_top_m"]
    if top is None or bottom is None:
        return {"evaluable": False, "delta_m": None, "monotonic": None, "score": None}
    delta = float(bottom) - float(top)
    score = max(0.0, 1.0 - abs(delta) / DEPTH_SCORE_SPAN_M)
    return {
        "evaluable": True,
        "delta_m": round(delta, 5),
        "within_tolerance": abs(delta) <= DEPTH_TOLERANCE_M,
        "monotonic": delta >= 0.0,
        "score": round(score, 4),
    }


def evaluate_order(order, segments, oriented_rgba):
    by_key = {segment["segment_key"]: segment for segment in segments}
    joins = []
    for upper_key, lower_key in zip(order["segment_keys"], order["segment_keys"][1:]):
        lamina = join_lamina(oriented_rgba[upper_key], oriented_rgba[lower_key])
        depth = join_depth(by_key[upper_key], by_key[lower_key])
        joins.append({
            "uphole_segment": upper_key,
            "downhole_segment": lower_key,
            "lamina": lamina,
            "depth": depth,
        })
    lamina_score = round(float(np.mean([join["lamina"]["score"] for join in joins])), 4)
    depth_values = [join["depth"]["score"] for join in joins if join["depth"]["evaluable"]]
    depth_score = round(float(np.mean(depth_values)), 4) if depth_values else None
    return {
        "order_id": order["order_id"],
        "label": order["label"],
        "segment_keys": order["segment_keys"],
        "lamina_score": lamina_score,
        "depth_score": depth_score,
        "depth_joins_evaluable": len(depth_values),
        "joins": joins,
    }


def choose_order(evaluations):
    first, second = evaluations
    lamina_gap = first["lamina_score"] - second["lamina_score"]
    if abs(lamina_gap) > LAMINA_DECISIVE_MARGIN:
        winner, loser = (first, second) if lamina_gap > 0 else (second, first)
        basis = "lamina"
        rationale = lamina_rationale(winner, loser)
        return winner, loser, basis, rationale
    if first["depth_score"] is not None and second["depth_score"] is not None and first["depth_score"] != second["depth_score"]:
        winner, loser = (first, second) if first["depth_score"] > second["depth_score"] else (second, first)
        basis = "depth"
        rationale = (
            f"Lamina continuity is within {LAMINA_DECISIVE_MARGIN:.2f} across both orderings "
            f"({first['lamina_score']:.3f} vs {second['lamina_score']:.3f}), so legacy depth continuity "
            f"breaks the tie. '{winner['label']}' keeps the depth records monotonic "
            f"(score {winner['depth_score']:.3f} against {loser['depth_score']:.3f})."
        )
        return winner, loser, basis, rationale
    winner, loser = (first, second) if first["lamina_score"] >= second["lamina_score"] else (second, first)
    basis = "lamina"
    rationale = (
        f"Neither ordering has decisive lamina evidence and depth records cannot separate them; "
        f"'{winner['label']}' is retained on the slightly stronger lamina score "
        f"({winner['lamina_score']:.3f} vs {loser['lamina_score']:.3f})."
    )
    return winner, loser, basis, rationale


def lamina_rationale(winner, loser):
    worst = max(loser["joins"], key=lambda join: 1 - join["lamina"]["score"])
    pair = f"{worst['uphole_segment']}→{worst['downhole_segment']}"
    lamina = worst["lamina"]
    depth_clause = ""
    if loser["depth_score"] is not None:
        depth_clause = (
            f" The legacy-depth ordering favoured by '{loser['label']}' (depth score {loser['depth_score']:.3f}) "
            "was set aside because the duplicated old number makes that numbering unreliable, and visible "
            "lamina continuity across the fracture is the primary physical evidence."
        )
    return (
        f"'{winner['label']}' is adopted: its fracture joins continue the visible lamina "
        f"(score {winner['lamina_score']:.3f} vs {loser['lamina_score']:.3f}). In '{loser['label']}' the {pair} "
        f"join breaks the bedding (tone Δ {lamina['tone_delta_linear']:.3f}, lamina angle "
        f"{lamina['lamina_angle_delta_deg']:.1f}°, edge profile r={lamina['edge_profile_correlation']:.3f})."
        + depth_clause
    )


# ---------------------------------------------------------------------------
# Source provenance mapping
# ---------------------------------------------------------------------------

def locate_point(source_map, x_norm, y_norm):
    width, height = source_map["plane_size"]
    column = min(max(x_norm, 0.0), 1.0) * width
    row = min(max(y_norm, 0.0), 1.0) * height
    for block in source_map["blocks"]:
        if not (block["row_start"] <= row < block["row_end"]):
            continue
        fraction = (row - block["row_start"]) / max(1, block["row_end"] - block["row_start"])
        depth_offset = block["depth_offset_start_mm"] + fraction * (
            block["depth_offset_end_mm"] - block["depth_offset_start_mm"]
        )
        if block["kind"] == "gap":
            return {
                "kind": "gap",
                "segment_key": block["segment_key"],
                "known_missing_length_mm": block["length_mm"],
                "depth_offset_mm": round(depth_offset, 4),
            }
        inside = block["x_offset"] <= column < block["x_offset"] + block["tile_width"]
        x_tile = (column - block["x_offset"]) / max(1, block["tile_width"])
        x_tile = min(max(x_tile, 0.0), 1.0)
        if block["rotated"]:
            original_x, original_y = 1.0 - x_tile, 1.0 - fraction
        else:
            original_x, original_y = x_tile, fraction
        orig_w, orig_h = block["original_size"]
        return {
            "kind": "segment",
            "segment_key": block["segment_key"],
            "within_face": bool(inside),
            "position_in_original": [round(original_x, 6), round(original_y, 6)],
            "pixel_in_original": [round(original_x * (orig_w - 1)), round(original_y * (orig_h - 1))],
            "original_size": [orig_w, orig_h],
            "depth_offset_mm": round(depth_offset, 4),
        }
    return {"kind": "gap", "depth_offset_mm": None}


def annotate_profile_rows(tile, rows, frame):
    source_map = (tile.get("assembly") or {}).get("source_map")
    if not source_map:
        return rows
    origin_m = tile["assembly"].get("origin_depth_m")
    for row in rows:
        x_norm, y_norm = row["relative_position"]
        location = locate_point(source_map, x_norm, y_norm)
        row["depth_offset_mm"] = location["depth_offset_mm"]
        row["depth_m"] = (
            round(origin_m + location["depth_offset_mm"] / 1000.0, 6)
            if origin_m is not None and location["depth_offset_mm"] is not None
            else None
        )
        if location["kind"] == "gap":
            row["source"] = None
            continue
        key = location["segment_key"]
        segment = next(item for item in tile["assembly"]["segments"] if item["segment_key"] == key)
        row["source"] = {
            "assembly_id": tile["assembly"]["assembly_id"],
            "segment_key": key,
            "old_label": segment["old_label"],
            "position_in_original": location["position_in_original"],
            "pixel_in_original": location["pixel_in_original"],
            "original_size": location["original_size"],
            "within_face": location["within_face"],
            "photo_url": f"/api/assemblies/{tile['assembly']['assembly_id']}/segments/{key}/photo/working",
        }
    return rows


# ---------------------------------------------------------------------------
# Assembly orchestration
# ---------------------------------------------------------------------------

def _segment_photo(blobs, binary, filename, key):
    return store_capture(
        blobs, binary, "diffuse", filename or f"segment-{key}.png",
        f"Archived fracture-face photograph of core segment {key}.", make_id(),
    )


def create_assembly(store, document_payload, files, *, synthetic=False):
    payload = dictionary(document_payload, "Assembly")
    name = string(payload.get("name"), "Assembly name", maximum=120)
    tray = string(payload.get("tray"), "Shelf / tray", maximum=120, required=False)
    origin_m = _optional_depth(payload.get("origin_depth_m"), "Origin depth")
    diameter_mm = scalar(payload.get("core_diameter_mm"), "Core diameter", lower=1.0, upper=100000.0)
    shelf_length = payload.get("shelf_length_mm")
    shelf_length = scalar(shelf_length, "Shelf capacity", lower=1.0, upper=100000.0) if shelf_length not in (None, "") else None

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, (list, tuple)) or len(raw_segments) != 3:
        raise InputError("An assembly needs exactly three broken segments.")
    segments = [parse_segment_document(raw, index) for index, raw in enumerate(raw_segments)]
    keys = [item["segment_key"] for item in segments]
    if sorted(keys) != sorted(SEGMENT_KEYS):
        raise InputError(f"Segments must use each key once: {', '.join(SEGMENT_KEYS)}.")
    segments.sort(key=lambda item: SEGMENT_KEYS.index(item["segment_key"]))
    fixed_key = segments[0]["segment_key"]
    orders = parse_orders(payload, fixed_key)
    duplicates = duplicate_labels(segments)

    photo_records = {}
    for spec in segments:
        key = spec["segment_key"]
        upload = files.get(f"photo-{key}")
        if upload is None:
            raise InputError(f"Upload the fracture-face photograph for segment {key}.")
        photo_records[key] = _segment_photo(
            store.blobs, upload.read(), upload.filename, key,
        )

    total_length_mm = sum(item["length_mm"] for item in segments)
    total_gap_mm = sum(item["gap_length_mm"] or 0 for item in segments)
    occupied_mm = total_length_mm + total_gap_mm
    if shelf_length is not None and occupied_mm > shelf_length:
        raise InputError(
            f"The string occupies {occupied_mm:.1f} mm and does not fit the {shelf_length:.1f} mm shelf.",
        )

    oriented_rgba = {
        spec["segment_key"]: _rgba(store.blobs, photo_records[spec["segment_key"]], spec)
        for spec in segments
    }
    evaluations = [evaluate_order(order, segments, oriented_rgba) for order in orders]
    winner, loser, basis, rationale = choose_order(evaluations)

    stitched, blocks, ppm = build_string(
        store.blobs, segments, photo_records, winner["segment_keys"], occupied_mm,
    )
    stitched_image = Image.fromarray(np.rint(np.clip(stitched, 0, 1) * 255).astype(np.uint8), mode="RGBA")
    stitched_photo = store_capture(
        store.blobs, png_bytes(stitched_image), "diffuse",
        f"{name.lower().replace(' ', '-')}-stitched.png",
        "Derived stitched core string; transparent bands are documented missing intervals, not imagery.",
        make_id(), synthetic=synthetic,
    )
    plane_size = canvas_dimensions(stitched_photo, DEFAULT_CORNERS)
    source_map = {
        "plane_size": list(plane_size),
        "px_per_mm": round(ppm, 6),
        "blocks": blocks,
    }

    assembly_id = make_id()
    assembly = {
        "id": assembly_id,
        "version": 1,
        "name": name,
        "tray": tray,
        "core_diameter_mm": float(diameter_mm),
        "origin_depth_m": origin_m,
        "shelf_length_mm": float(shelf_length) if shelf_length is not None else None,
        "total_segment_length_mm": round(total_length_mm, 4),
        "total_gap_length_mm": round(total_gap_mm, 4),
        "occupied_length_mm": round(occupied_mm, 4),
        "segments": [
            {**spec, "photo": photo_records[spec["segment_key"]]} for spec in segments
        ],
        "field_orders": orders,
        "duplicate_old_labels": duplicates,
        "evaluations": evaluations,
        "recommended_order_id": winner["order_id"],
        "decision": {
            "order_id": winner["order_id"],
            "basis": basis,
            "rationale": rationale,
            "defeated_order_id": loser["order_id"],
            "defeated_reason": (
                f"'{loser['label']}' loses on {basis} evidence; its supporting consideration "
                + ("(legacy depth records) " if loser["depth_score"] is not None else "")
                + "is recorded but not adopted."
            ),
            "manual_note": None,
            "decided_at": None,
        },
        "source_map": source_map,
        "stitched_photo_id": stitched_photo["id"],
        "specimen_id": None,
        "synthetic": synthetic,
    }

    tile = store.create_tile({
        "name": f"{name} · adopted string",
        "batch": tray or "Reassembled core string",
        "body": "Reassembled fracture-face core string",
        "glaze": "Archived segment join",
        "firing": None,
        "notes": rationale,
        "width_mm": float(diameter_mm),
        "height_mm": round(occupied_mm, 4),
        "photos": [stitched_photo],
        "plane": {
            "photo_id": stitched_photo["id"],
            "corners": DEFAULT_CORNERS,
            "size": list(plane_size),
        },
        "regions": [],
        "derived": True,
        "assembly": {
            "assembly_id": assembly_id,
            "adopted_order_id": winner["order_id"],
            "adopted_segment_keys": winner["segment_keys"],
            "origin_depth_m": origin_m,
            "source_map": source_map,
            "segments": [
                {
                    "segment_key": item["segment_key"],
                    "old_label": item["old_label"],
                    "length_mm": item["length_mm"],
                    "gap_end": item["gap_end"],
                    "gap_length_mm": item["gap_length_mm"],
                }
                for item in segments
            ],
        },
    })
    assembly["specimen_id"] = tile["id"]
    assembly["decision"]["decided_at"] = tile["created_at"]
    store.save_assembly(assembly)
    return assembly


def override_decision(store, assembly_id, payload):
    version = model_version(payload)
    order_id = string(payload.get("order_id"), "Ordering", maximum=40)
    note = string(payload.get("note"), "Override note", maximum=1200)
    assembly = store.assembly(assembly_id)
    store.require_version(assembly, version)
    chosen = next((item for item in assembly["evaluations"] if item["order_id"] == order_id), None)
    if chosen is None:
        raise InputError("Choose one of the two evaluated field orderings.")
    recommended = next(item for item in assembly["evaluations"] if item["order_id"] == assembly["recommended_order_id"])
    assembly["decision"] = {
        "order_id": chosen["order_id"],
        "basis": "manual_override",
        "rationale": (
            f"Manual archive decision: '{chosen['label']}' adopted despite the automatic recommendation "
            f"of '{recommended['label']}' (lamina {recommended['lamina_score']:.3f} vs {chosen['lamina_score']:.3f})."
        ),
        "defeated_order_id": recommended["order_id"],
        "defeated_reason": (
            f"The automatic lamina-based recommendation '{recommended['label']}' was set aside by the operator."
        ),
        "manual_note": note,
        "decided_at": timestamp(),
    }
    assembly["version"] = version + 1
    store.save_assembly(assembly)
    return assembly
