"""Plan and audit a single circular core extraction on the registered survey plane.

The workbench user marks brittle rims, matrix they want to keep, fine veins to
retain, two clamp-accessible drill centers and (after drilling) the actual
missing gap.  Every calculation is rasterized on the same registered plane so
pre-drill planning and post-drill auditing share one coordinate system.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import SHADOW, base_rgba, font, stored_preview
from .geometry import polygon_mask, specimen_frame, validate_polygon
from .model import (
    InputError,
    collection,
    dictionary,
    identity,
    numeric_pair,
    string,
)


CORE_DIAMETER_MM = 16.0
MAX_LAYERS = 6
MAX_CENTERS = 2
MAX_GAP_POLYGONS = 6
MIN_LAYER_PIXELS = 20

# Annotation layer identity used in payloads and stored documents.
LAYER_BRITTLE = "brittle"
LAYER_MATRIX = "matrix"
LAYER_VEIN = "vein"
LAYER_UNREADABLE = "unreadable"
# Planning accepts an optional explicit unreadable layer; the audit always
# carries one so unclear junctions can be separated from the decision totals.
PLAN_LAYERS = (LAYER_BRITTLE, LAYER_MATRIX, LAYER_VEIN, LAYER_UNREADABLE)
AUDIT_LAYERS = (LAYER_BRITTLE, LAYER_MATRIX, LAYER_VEIN, LAYER_UNREADABLE)
LAYER_TITLES = {
    LAYER_BRITTLE: "Brittle edge",
    LAYER_MATRIX: "Intact matrix to keep",
    LAYER_VEIN: "Fine mineral vein",
    LAYER_UNREADABLE: "Boundary not clear enough",
}
LAYER_COLORS = {
    LAYER_BRITTLE: (231, 105, 96),
    LAYER_MATRIX: (122, 196, 138),
    LAYER_VEIN: (116, 171, 226),
    LAYER_UNREADABLE: (214, 184, 106),
}
LAYER_FILL = {
    LAYER_BRITTLE: (231, 105, 96, 78),
    LAYER_MATRIX: (122, 196, 138, 66),
    LAYER_VEIN: (116, 171, 226, 84),
    LAYER_UNREADABLE: (214, 184, 106, 86),
}
CHOSEN_COLOR = (244, 201, 109)
ALTERNATE_COLOR = (189, 226, 214)
UNREADABLE_TINT = (94, 88, 78)
LEFTOVER_TINT = (122, 196, 138)
OVERCUT_TINT = (236, 126, 92)


@dataclass(frozen=True)
class PlaneMasks:
    extent: tuple
    brittle: np.ndarray
    matrix: np.ndarray
    vein: np.ndarray
    unreadable: np.ndarray
    unsupported: np.ndarray


# --------------------------------------------------------------------------- #
# Payload parsing
# --------------------------------------------------------------------------- #

def polygon_layers(payload, allowed):
    data = dictionary(payload, "Drill annotations")
    layers = dictionary(data.get("layers", {}), "Annotation layers")
    parsed = {}
    for name in allowed:
        polygons = collection(layers.get(name, []), f"{LAYER_TITLES[name]} outline",
                              minimum=0, maximum=MAX_LAYERS)
        outlines = []
        for polygon in polygons:
            vertices = collection(polygon, "Polygon vertices", minimum=3, maximum=40)
            points = [numeric_pair(item, "Vertex", lower=0, upper=1) for item in vertices]
            outlines.append(validate_polygon(points))
        parsed[name] = outlines
    return parsed


def center_points(payload):
    raw = collection(payload.get("centers"), "Drill centers", minimum=1, maximum=MAX_CENTERS)
    points = [numeric_pair(item, "Drill center", lower=0, upper=1) for item in raw]
    if len(points) != len({tuple(point) for point in points}):
        raise InputError("The two drill centers must mark different positions.")
    return points


def gap_polygons(payload):
    raw = collection(payload.get("gap"), "Actual gap outline", minimum=1, maximum=MAX_GAP_POLYGONS)
    outlines = []
    for polygon in raw:
        vertices = collection(polygon, "Gap vertices", minimum=3, maximum=40)
        points = [numeric_pair(item, "Vertex", lower=0, upper=1) for item in vertices]
        outlines.append(validate_polygon(points))
    return outlines


# --------------------------------------------------------------------------- #
# Raster helpers
# --------------------------------------------------------------------------- #

def combine_polygons(extent, outlines):
    mask = np.zeros((extent[1], extent[0]), dtype=bool)
    for vertices in outlines:
        mask |= polygon_mask(extent, vertices)
    return mask


def layer_masks(frame, outlines, *, include_unreadable=True):
    extent = frame.extent
    brittle = combine_polygons(extent, outlines.get(LAYER_BRITTLE, []))
    matrix = combine_polygons(extent, outlines.get(LAYER_MATRIX, []))
    vein = combine_polygons(extent, outlines.get(LAYER_VEIN, []))
    unreadable = combine_polygons(extent, outlines.get(LAYER_UNREADABLE, [])) if include_unreadable \
        else np.zeros((extent[1], extent[0]), dtype=bool)
    if np.count_nonzero(brittle) < MIN_LAYER_PIXELS:
        raise InputError("Outline at least one brittle edge area before planning the drill.")
    if np.count_nonzero(matrix) < MIN_LAYER_PIXELS:
        raise InputError("Outline the intact matrix you want to keep before planning the drill.")
    if np.count_nonzero(vein) < MIN_LAYER_PIXELS:
        raise InputError("Outline the fine mineral vein that must not be missed.")
    return PlaneMasks(extent, brittle, matrix, vein, unreadable, ~frame.valid)


def ellipse_mask(extent, center, radius_xy):
    """Rasterize the physical 16 mm circle, which may map to an ellipse."""
    width, height = extent
    grid = Image.new("1", extent)
    cx, cy = center[0] * width - 0.5, center[1] * height - 0.5
    ImageDraw.Draw(grid).ellipse(
        (cx - radius_xy[0], cy - radius_xy[1], cx + radius_xy[0], cy + radius_xy[1]), fill=1,
    )
    return np.asarray(grid, dtype=bool)


def radius_in_pixels(frame):
    if frame.scale is None:
        raise InputError("Set the physical specimen width and height before planning a 16 mm core.")
    radius_x = CORE_DIAMETER_MM / frame.scale[0] / 2
    radius_y = CORE_DIAMETER_MM / frame.scale[1] / 2
    if min(radius_x, radius_y) < 4:
        raise InputError("The 16 mm core would cover fewer than eight registered pixels on an edge.")
    return float(radius_x), float(radius_y)


def clearance_to(mask, footprint):
    """Nearest pixel distance from the footprint to a forbidden mask (0 on overlap)."""
    if np.any(mask & footprint):
        return 0.0
    if not mask.any():
        return None
    return float(ndimage.distance_transform_edt(~mask)[footprint].min() - 0.5)


def area_mm2(frame, mask):
    if frame.scale is None:
        return float(np.count_nonzero(mask))
    return float(np.count_nonzero(mask) * frame.scale[0] * frame.scale[1])


def boundary_distance_mm(frame, first, second):
    """Minimum sampled boundary-to-boundary distance between two mask sets."""
    if not first.any() or not second.any():
        return None
    scale = frame.scale or (1.0, 1.0)
    return float(ndimage.distance_transform_edt(~second, sampling=(scale[1], scale[0]))[first].min())


# --------------------------------------------------------------------------- #
# Pre-drill planning
# --------------------------------------------------------------------------- #

def evaluate_center(frame, masks, center, radius_xy):
    footprint = ellipse_mask(frame.extent, center, radius_xy)
    forbidden = {
        "brittle_edge": masks.brittle,
        "unreadable_or_off_photo": masks.unreadable | masks.unsupported,
    }
    overlaps = {name: int(np.count_nonzero(footprint & mask)) for name, mask in forbidden.items()}
    scale = frame.scale or (1.0, 1.0)
    clearances_mm = {}
    for name, mask in forbidden.items():
        gap_px = clearance_to(mask, footprint)
        clearances_mm[name] = None if gap_px is None else round(max(gap_px, 0.0) * min(scale), 4)
    footprint_count = int(np.count_nonzero(footprint))
    matrix_inside = int(np.count_nonzero(footprint & masks.matrix))
    vein_inside = int(np.count_nonzero(footprint & masks.vein))
    feasible = all(count == 0 for count in overlaps.values())
    return {
        "center": [round(center[0], 6), round(center[1], 6)],
        "feasible": feasible,
        "footprint_pixels": footprint_count,
        "footprint_area_mm2": area_mm2(frame, footprint),
        "overlap_pixels": overlaps,
        "clearance_mm": clearances_mm,
        "matrix_pixels_inside": matrix_inside,
        "matrix_area_mm2_inside": area_mm2(frame, footprint & masks.matrix),
        "vein_pixels_inside": vein_inside,
        "vein_area_mm2_inside": area_mm2(frame, footprint & masks.vein),
        "vein_components_inside": vein_components_inside(masks.vein, footprint),
        "vein_fraction_of_circle_percent": round(float(100 * vein_inside / footprint_count), 3),
        "matrix_fraction_of_circle_percent": round(float(100 * matrix_inside / footprint_count), 3),
        "_footprint": footprint,
    }


def vein_components_inside(vein, footprint):
    inside = vein & footprint
    if not np.any(inside):
        return 0
    identities, _ = ndimage.label(vein)
    return int(np.count_nonzero(np.unique(identities[inside]) > 0))


def rank_candidates(candidates):
    feasible = [item for item in candidates if item["feasible"]]
    pool = feasible or candidates
    if len(pool) < 2:
        return pool[0], None
    matrix_values = np.array([item["matrix_area_mm2_inside"] for item in pool], dtype=np.float64)
    vein_values = np.array([item["vein_area_mm2_inside"] for item in pool], dtype=np.float64)
    matrix_span = max(matrix_values.max() - matrix_values.min(), 1e-9)
    vein_span = max(vein_values.max() - vein_values.min(), 1e-9)
    scores = 0.65 * (matrix_values - matrix_values.min()) / matrix_span \
        + 0.35 * (vein_values - vein_values.min()) / vein_span
    order = np.argsort(-scores)
    return pool[int(order[0])], pool[int(order[1])]


def advantage_notes(chosen, alternate):
    if alternate is None:
        return ["Only one clamp-accessible center was marked."]
    notes = []
    matrix_gain = chosen["matrix_area_mm2_inside"] - alternate["matrix_area_mm2_inside"]
    if matrix_gain > 0:
        notes.append(
            f"Retains {matrix_gain:.3f} mm² more of the marked intact matrix "
            f"({chosen['matrix_fraction_of_circle_percent']:.1f}% vs "
            f"{alternate['matrix_fraction_of_circle_percent']:.1f}% of the circle)."
        )
    elif matrix_gain < 0:
        notes.append(
            f"Gives up {abs(matrix_gain):.3f} mm² of marked matrix relative to the other center."
        )
    vein_gain = chosen["vein_area_mm2_inside"] - alternate["vein_area_mm2_inside"]
    if vein_gain > 0:
        notes.append(
            f"Keeps {vein_gain:.3f} mm² more of the fine vein inside the core "
            f"({chosen['vein_components_inside']} vs {alternate['vein_components_inside']} "
            "marked vein patch(es))."
        )
    elif vein_gain < 0:
        notes.append(
            f"The other center holds {abs(vein_gain):.3f} mm² more vein, but loses on the matrix objective."
        )
    for label, key in (("brittle edge", "brittle_edge"),
                       ("unreadable or off-photo ground", "unreadable_or_off_photo")):
        chosen_gap = chosen["clearance_mm"].get(key)
        other_gap = alternate["clearance_mm"].get(key)
        if chosen_gap is not None and other_gap is not None and chosen_gap > other_gap:
            notes.append(f"Sits {chosen_gap - other_gap:.3f} mm further from the {label}.")
    if not notes:
        notes.append("The two centers are nearly equivalent; this one edges ahead on the combined matrix/vein score.")
    return notes


def plan_drill(store, tile, payload):
    data = dictionary(payload, "Drill plan")
    title = string(data.get("title"), "Plan title", maximum=90)
    outlines = polygon_layers(data, PLAN_LAYERS)
    centers = center_points(data)
    frame = specimen_frame(store.blobs, tile)
    radius_xy = radius_in_pixels(frame)
    masks = layer_masks(frame, outlines)
    candidates = [evaluate_center(frame, masks, center, radius_xy) for center in centers]
    chosen, alternate = rank_candidates(candidates)
    if not chosen["feasible"]:
        reasons = [f"{name}: {count} px inside the 16 mm circle"
                   for name, count in chosen["overlap_pixels"].items() if count]
        raise InputError(
            "No marked center keeps the full 16 mm circle clear. Move both centers off the brittle edge "
            "and fully inside the readable photo. " + "; ".join(reasons)
        )
    layer_areas = {
        name: {
            "pixels": int(np.count_nonzero(getattr(masks, name))),
            "area_mm2": area_mm2(frame, getattr(masks, name)),
        }
        for name in (LAYER_BRITTLE, LAYER_MATRIX, LAYER_VEIN, LAYER_UNREADABLE)
    }
    overlay = plan_overlay(frame, outlines, candidates, chosen, alternate, radius_xy)
    document = {
        "title": title,
        "tile_id": tile["id"],
        "core_diameter_mm": CORE_DIAMETER_MM,
        "radius_px": [round(radius_xy[0], 4), round(radius_xy[1], 4)],
        "layers": outlines,
        "layer_areas": layer_areas,
        "centers": centers,
        "candidates": [public_candidate(item) for item in candidates],
        "chosen_center": chosen["center"],
        "alternate_center": alternate["center"] if alternate else None,
        "advantage": advantage_notes(chosen, alternate),
        "frame_extent": list(frame.extent),
        "scale_xy": list(frame.scale) if frame.scale else None,
        "source": {"photo_id": frame.photo["id"], "photo_name": frame.photo["name"]},
        "audit": None,
    }
    return document, overlay


def public_candidate(item):
    return {key: value for key, value in item.items() if not key.startswith("_")}


# --------------------------------------------------------------------------- #
# Post-drill audit
# --------------------------------------------------------------------------- #

def audit_drill(store, tile, plan, payload):
    data = dictionary(payload, "Drill audit")
    outlines = polygon_layers(data, AUDIT_LAYERS)
    gap_outlines = gap_polygons(data)
    photo_id = identity(data.get("photo_id"), "Repeat photo ID")
    frame = specimen_frame(store.blobs, tile, photo_id)
    radius_xy = radius_in_pixels(frame)
    masks = layer_masks(frame, outlines)
    circle = ellipse_mask(frame.extent, plan["chosen_center"], radius_xy)
    gap = combine_polygons(frame.extent, gap_outlines)
    unreadable = masks.unreadable | (~frame.valid)
    decidable = ~unreadable
    circle_decided = circle & decidable
    gap_decided = gap & decidable
    leftover = circle_decided & ~gap
    overcut = gap_decided & ~circle
    matched = circle_decided & gap
    boundary = (circle | gap) & unreadable
    metrics = {
        "circle_area_mm2": area_mm2(frame, circle),
        "gap_area_mm2": area_mm2(frame, gap),
        "uncut_inside_circle_mm2": area_mm2(frame, leftover),
        "uncut_inside_circle_percent": round(float(
            100 * np.count_nonzero(leftover) / max(np.count_nonzero(circle_decided), 1)), 3),
        "overcut_outside_circle_mm2": area_mm2(frame, overcut),
        "overcut_outside_circle_percent": round(float(
            100 * np.count_nonzero(overcut) / max(np.count_nonzero(gap_decided), 1)), 3),
        "matched_mm2": area_mm2(frame, matched),
        "indeterminate_boundary_mm2": area_mm2(frame, boundary),
        "indeterminate_boundary_percent": round(float(
            100 * np.count_nonzero(boundary) / max(np.count_nonzero(circle | gap), 1)), 3),
        "gap_center_offset_mm": centroid_offset(frame, circle, gap),
        "gap_to_circle_boundary_mm": boundary_distance_mm(frame, gap, circle),
        "units": "mm²" if frame.scale else "px²",
    }
    overlay = audit_overlay(frame, outlines, gap_outlines, circle, gap, unreadable, leftover, overcut,
                            boundary, plan["chosen_center"], radius_xy)
    record = {
        "photo_id": photo_id,
        "photo_name": frame.photo["name"],
        "layers": outlines,
        "gap": gap_outlines,
        "metrics": metrics,
        "frame_extent": list(frame.extent),
    }
    return record, overlay


def centroid_offset(frame, circle, gap):
    def centroid(mask):
        rows, cols = np.nonzero(mask)
        if not len(rows):
            return None
        return np.array([cols.mean(), rows.mean()])
    circle_center, gap_center = centroid(circle), centroid(gap)
    if circle_center is None or gap_center is None:
        return None
    delta = gap_center - circle_center
    scale = frame.scale or (1.0, 1.0)
    return round(float(np.hypot(delta[0] * scale[0], delta[1] * scale[1])), 4)


# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #

def _draw_polygons(draw, extent, outlines, color, fill, width):
    for vertices in outlines:
        points = [(x * extent[0] - 0.5, y * extent[1] - 0.5) for x, y in vertices]
        draw.polygon(points, fill=fill)
        draw.line([*points, points[0]], fill=color, width=width, joint="curve")


def _draw_ellipse(draw, extent, center, radius_xy, color, width, label=None):
    cx = center[0] * extent[0] - 0.5
    cy = center[1] * extent[1] - 0.5
    rx, ry = radius_xy
    box = (cx - rx, cy - ry, cx + rx, cy + ry)
    draw.ellipse(box, outline=SHADOW, width=width + 3)
    draw.ellipse(box, outline=color, width=width)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color, outline=SHADOW)
    if label:
        draw.text((cx + 7, cy - 8), label, fill=color, stroke_width=2, stroke_fill=SHADOW)


def _legend(draw, image_size, entries):
    typeface = font(max(13, round(min(image_size) / 52)))
    legend_y = 10
    for label, color in entries:
        draw.rectangle((10, legend_y, 24, legend_y + 14), fill=color, outline=SHADOW)
        draw.text((30, legend_y - 1), label, fill=(255, 245, 225),
                  stroke_width=2, stroke_fill=SHADOW, font=typeface)
        legend_y += 20
    return typeface


def plan_overlay(frame, outlines, candidates, chosen, alternate, radius_xy):
    image = base_rgba(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(1, round(min(image.size) / 340))
    for name in PLAN_LAYERS:
        if outlines.get(name):
            _draw_polygons(draw, frame.extent, outlines[name], LAYER_COLORS[name], LAYER_FILL[name], width)
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    for item in candidates:
        is_chosen = item["center"] == chosen["center"]
        color = CHOSEN_COLOR if is_chosen else ALTERNATE_COLOR
        _draw_ellipse(draw, frame.extent, item["center"], radius_xy, color,
                      width + 1, "CHOSEN" if is_chosen else "ALTERNATE")
    legend_entries = [(LAYER_TITLES[name], LAYER_COLORS[name]) for name in PLAN_LAYERS if outlines.get(name)]
    _legend(draw, image.size, legend_entries)
    typeface = font(max(13, round(min(image.size) / 52)))
    draw.text((10, image.size[1] - 26), f"Planned core diameter {CORE_DIAMETER_MM:.0f} mm",
              fill=CHOSEN_COLOR, stroke_width=2, stroke_fill=SHADOW, font=typeface)
    return image


def audit_overlay(frame, outlines, gap_outlines, circle, gap, unreadable, leftover, overcut,
                  boundary, chosen_center, radius_xy):
    image = base_rgba(frame)
    array = np.asarray(image, dtype=np.float32).copy()
    array[leftover, :3] = array[leftover, :3] * 0.45 + np.array(LEFTOVER_TINT, dtype=np.float32) * 0.55
    array[overcut, :3] = array[overcut, :3] * 0.42 + np.array(OVERCUT_TINT, dtype=np.float32) * 0.58
    array[boundary, :3] = array[boundary, :3] * 0.35 + np.array(UNREADABLE_TINT, dtype=np.float32) * 0.65
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(1, round(min(image.size) / 340))
    # Annotation outlines stay visible on top of the decision tints.
    for name in AUDIT_LAYERS:
        _draw_polygons(draw, frame.extent, outlines[name], LAYER_COLORS[name], LAYER_FILL[name], width)
    _draw_polygons(draw, frame.extent, gap_outlines, GAP_POLYGON_COLOR, (236, 126, 92, 60), width + 1)
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    _draw_ellipse(draw, frame.extent, chosen_center, radius_xy, CHOSEN_COLOR, width + 1, "PLANNED 16 mm")
    _legend(draw, image.size, [
        ("Actual gap outline", GAP_POLYGON_COLOR),
        ("Uncut inside circle", LEFTOVER_TINT),
        ("Overcut outside circle", OVERCUT_TINT),
        ("Boundary not decidable", UNREADABLE_TINT),
    ])
    return image


GAP_POLYGON_COLOR = (236, 126, 92)


def stored_overlay(image, blobs):
    return stored_preview(image, blobs)
