"""Separate hand-drawn legacy grey rules from thin raking-light fissures.

The analysis uses three registered captures of the same specimen plane:
a diffuse front plus left and right raking-light views. A pencil/ink rule
darkens the surface in every capture and keeps its sign when the raking
lamp changes sides. A genuine fissure is a relief edge instead: its
darkness flips to brightness between the two raking captures while the
diffuse view shows no stable dark line. Stations are therefore classified
only where all three photos actually see the surface.

The user draws one polyline for the trunk and the strike that appears to
continue it, plus a short two-point strike for the competing fork branch.
Exactly one branch at the junction is allowed to remain the legacy rule's
continuation; the losing branch is demoted to crack-only or unclear
evidence.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import base_rgba, font
from .geometry import polygon_mask
from .model import InputError, collection, dictionary, numeric_pair, scalar, integer
from .photos import luminance, png_bytes


# Raster codes stored in the frozen evidence mask.
CODE_OUTSIDE = 0
CODE_LEGACY = 1
CODE_CRACK = 2
CODE_UNCLEAR = 3
CODE_COMMON_UNTRACED = 4
CODE_UNSUPPORTED = 5

STATE_LABELS = {
    CODE_LEGACY: "legacy",
    CODE_CRACK: "crack-only",
    CODE_UNCLEAR: "unclear",
}
STATE_COLORS = {
    CODE_LEGACY: (248, 202, 102),    # warm amber: adopted hand-drawn rule
    CODE_CRACK: (150, 205, 232),     # cool blue: raking-only fissure evidence
    CODE_UNCLEAR: (170, 178, 190),   # neutral grey: cannot be decided
}

MIN_STATIONS_PER_PATH = 10
MAX_STATIONS = 1200
MIN_TRACE_PX = 12
JUNCTION_END_MARGIN = 0.08
MAX_FORK_DISTANCE_PX = 26.0


def _floor_noise(values):
    """Typical background variation; strong minority lines do not inflate it."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.003
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(0.003, mad * 1.4826)


@dataclass(frozen=True)
class Frames:
    diffuse: object
    rake_left: object
    rake_right: object

    def all(self):
        return (self.diffuse, self.rake_left, self.rake_right)


def parse_trace(payload):
    data = dictionary(payload, "Legacy trace")
    trace = collection(data.get("trace"), "Legacy trace path", minimum=2, maximum=60)
    strike = collection(data.get("fork_strike"), "Fork strike", minimum=2, maximum=2)
    points = [numeric_pair(item, "Trace vertex", lower=0, upper=1) for item in trace]
    fork = [numeric_pair(item, "Fork strike point", lower=0, upper=1) for item in strike]
    if len(points) != len({tuple(point) for point in points}):
        raise InputError("The trace path cannot repeat a vertex.")
    if tuple(fork[0]) == tuple(fork[1]):
        raise InputError("The fork strike needs two distinct points.")
    return points, fork


def legacy_options(raw=None):
    data = dictionary(raw or {}, "Measurement options")
    return {
        "radius": integer(data.get("radius", 7), "Cross-section half width px", lower=2, upper=40),
        "spacing_px": scalar(data.get("spacing_px", 2.0), "Station spacing px", lower=0.8, upper=12.0),
        "diffuse_floor": scalar(data.get("diffuse_floor", 0.022), "Diffuse darkness floor",
                                lower=0.004, upper=0.35),
        "flip_floor": scalar(data.get("flip_floor", 0.03), "Raking flip floor",
                             lower=0.005, upper=0.6),
        "rake_dark_floor": scalar(data.get("rake_dark_floor", 0.018), "Raking darkness floor",
                                  lower=0.004, upper=0.35),
    }


def common_support(frames, area):
    common = area.copy()
    for frame in frames.all():
        common &= frame.valid
    if int(np.count_nonzero(common)) < 200:
        raise InputError("The three photos need a larger common visible area inside the selected region.")
    unclipped = common.copy()
    for frame in frames.all():
        unclipped &= ~frame.clipped
    if int(np.count_nonzero(unclipped)) < 140:
        raise InputError("Too little shared, unclipped surface remains to separate the rule from fissures.")
    return common, unclipped


def resample_polyline(points, extent, spacing):
    """Dense stations in normalized coordinates plus cumulative pixel arc length."""
    width, height = extent
    xy = np.asarray(points, dtype=np.float64) * [width, height]
    deltas = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    if float(segment_lengths.sum()) < MIN_TRACE_PX:
        raise InputError("Mark a longer trace spanning at least twelve registered pixels.")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    count = int(np.clip(np.ceil(total / spacing), MIN_STATIONS_PER_PATH, MAX_STATIONS))
    targets = np.linspace(0.0, total, count)
    sampled = np.empty((count, 2), dtype=np.float64)
    for axis in (0, 1):
        sampled[:, axis] = np.interp(targets, cumulative, xy[:, axis])
    normalized = sampled / [width, height]
    return normalized, targets


def locate_junction(trace_points, fork, extent, total_length):
    """Nearest point on the drawn polyline to the fork strike's first point."""
    width, height = extent
    xy = np.asarray(trace_points, dtype=np.float64) * [width, height]
    strike_start = np.asarray(fork[0], dtype=np.float64) * [width, height]
    deltas = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    best = None
    for index, segment in enumerate(deltas):
        length = segment_lengths[index]
        if length < 1e-9:
            continue
        ratio = float(np.dot(strike_start - xy[index], segment) / (length * length))
        ratio = min(1.0, max(0.0, ratio))
        point = xy[index] + ratio * segment
        distance = float(np.linalg.norm(strike_start - point))
        arc = cumulative[index] + ratio * length
        if best is None or distance < best[0]:
            best = (distance, arc, point)
    distance, junction_arc, junction_px = best
    if distance > MAX_FORK_DISTANCE_PX:
        raise InputError(
            "The fork strike must start on (or within 26 px of) the drawn trace at the junction."
        )
    margin = JUNCTION_END_MARGIN * total_length
    if junction_arc < margin or junction_arc > total_length - margin:
        raise InputError("The fork junction must be interior to the drawn trace, not at its endpoints.")
    return junction_arc, junction_px / [width, height], distance


def cross_section_coords(stations, extent, radius):
    width, height = extent
    xy = np.asarray(stations, dtype=np.float64) * [width, height]
    tangents = np.gradient(xy, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    centers = xy[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    coords = (centers[..., 1], centers[..., 0])
    core = np.abs(offsets) <= max(1, radius / 3)
    # A near-line ring keeps the local background while skipping unrelated texture farther out.
    reference = (np.abs(offsets) > radius * 0.45) & (np.abs(offsets) <= radius * 0.85)
    if not np.any(reference):
        raise InputError("Use a wider cross-section so surface beside the line is sampled too.")
    return coords, core, reference


def _luma(frame, coords):
    return ndimage.map_coordinates(
        luminance(frame.linear), coords, order=1, mode="constant", cval=0.0,
    )


def _valid(frame, coords):
    return ndimage.map_coordinates(
        frame.valid.astype(np.float32), coords, order=1, mode="constant", cval=0.0,
    ) > 0.999


def rake_balance(frames, support_mask):
    """Median left/right exposure ratio over shared, middle-tone surface pixels."""
    left_full = luminance(frames.rake_left.linear)
    right_full = luminance(frames.rake_right.linear)
    eligible = (
        support_mask
        & (left_full >= 0.025) & (right_full >= 0.025)
        & (left_full <= 0.92) & (right_full <= 0.92)
    )
    if int(np.count_nonzero(eligible)) < 120:
        raise InputError("Too little middle-tone common surface to balance the two raking captures.")
    gain = float(np.median(left_full[eligible] / right_full[eligible]))
    if not 0.2 <= gain <= 5.0:
        raise InputError("The two raking captures differ too much in exposure to compare their signs.")
    return gain


def _rolling_median(values, window):
    """Centered running median that preserves edges while absorbing broad strata drift."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    half = max(1, window // 2)
    padded = np.pad(values, (half, half), mode="edge")
    return np.array([np.median(padded[i:i + 2 * half + 1]) for i in range(len(values))])


def classify_path(frames, stations, extent, common, unclipped, options):
    """Classify every station of one polyline; unsupported stations get CODE_UNSUPPORTED.

    Darkness is the core's residual against a local along-track background, so
    broad bedding drift is absorbed and only narrow lines produce a signal.
    """
    radius = options["radius"]
    coords, core, reference = cross_section_coords(stations, extent, radius)
    d_luma, l_luma, r_luma = (_luma(frame, coords) for frame in frames.all())
    d_valid, l_valid, r_valid = (_valid(frame, coords) for frame in frames.all())

    gain = rake_balance(frames, unclipped)
    r_luma = r_luma * gain

    count = len(stations)
    core_d = np.full(count, np.nan)
    core_l = np.full(count, np.nan)
    core_r = np.full(count, np.nan)
    ref_d = np.full(count, np.nan)
    ref_l = np.full(count, np.nan)
    ref_r = np.full(count, np.nan)
    flip_raw = np.full(count, np.nan)
    supported = np.zeros(count, dtype=bool)

    for index in range(count):
        usable = d_valid[index] & l_valid[index] & r_valid[index]
        ref = reference & usable
        inner = core & usable
        if not np.any(ref) or not np.any(inner):
            continue
        supported[index] = True
        core_d[index] = float(np.median(d_luma[index, inner]))
        core_l[index] = float(np.median(l_luma[index, inner]))
        core_r[index] = float(np.median(r_luma[index, inner]))
        ref_d[index] = float(np.median(d_luma[index, ref]))
        ref_l[index] = float(np.median(l_luma[index, ref]))
        ref_r[index] = float(np.median(r_luma[index, ref]))
        inner_l, inner_r = l_luma[index, inner], r_luma[index, inner]
        flip_raw[index] = float(np.median((inner_l - inner_r) / np.maximum(inner_l + inner_r, 0.045)))

    window = max(7, min(31, count // 4))
    # Local background follows each capture's reference band along the trace.
    base_d = _rolling_median(np.where(supported, ref_d, np.nanmedian(ref_d)), window)
    base_l = _rolling_median(np.where(supported, ref_l, np.nanmedian(ref_l)), window)
    base_r = _rolling_median(np.where(supported, ref_r, np.nanmedian(ref_r)), window)
    # Positive darkness = the narrow core is darker than its local background.
    dark_d = base_d - core_d
    dark_l = base_l - core_l
    dark_r = base_r - core_r
    # The raking sign flip is measured against a local flip baseline, removing a fixed lamp bias.
    flip = flip_raw - float(np.nanmedian(flip_raw))

    good = supported
    diffuse_limit = max(options["diffuse_floor"], 1.8 * _floor_noise(dark_d[good]))
    flip_limit = max(options["flip_floor"], 2.6 * _floor_noise(np.abs(flip[good])))
    rake_limit = max(options["rake_dark_floor"], 1.8 * _floor_noise(
        np.concatenate((dark_l[good], dark_r[good])),
    ))

    states = np.full(count, CODE_UNSUPPORTED, dtype=np.uint8)
    for index in np.flatnonzero(good):
        dark_diffuse = dark_d[index] >= diffuse_limit
        dark_in_both_rakes = (dark_l[index] >= rake_limit) and (dark_r[index] >= rake_limit)
        sign_flip = abs(flip[index]) >= flip_limit
        # A pigment rule is dark in diffuse and dark from either raking side without flipping.
        if dark_diffuse and not sign_flip and dark_in_both_rakes:
            states[index] = CODE_LEGACY
        elif sign_flip and not dark_diffuse:
            states[index] = CODE_CRACK
        else:
            states[index] = CODE_UNCLEAR
    thresholds = {
        "diffuse_darkness_limit": round(diffuse_limit, 6),
        "raking_flip_limit": round(flip_limit, 6),
        "raking_darkness_limit": round(rake_limit, 6),
        "raking_balance_gain": round(gain, 5),
    }
    return states, thresholds


def branch_tally(states):
    return {
        "legacy": int(np.count_nonzero(states == CODE_LEGACY)),
        "crack_only": int(np.count_nonzero(states == CODE_CRACK)),
        "unclear": int(np.count_nonzero(states == CODE_UNCLEAR)),
        "unsupported": int(np.count_nonzero(states == CODE_UNSUPPORTED)),
    }


def decide_fork(drawn_branch_states, strike_states):
    """Only one branch at the junction continues the legacy rule.

    Prefer whichever branch carries more diffuse-backed legacy stations;
    ties fall back to the branch with fewer raking-flip crack stations,
    then to the drawn continuation to keep the result deterministic.
    """
    drawn, strike = branch_tally(drawn_branch_states), branch_tally(strike_states)
    drawn_key = (drawn["legacy"], -drawn["crack_only"])
    strike_key = (strike["legacy"], -strike["crack_only"])
    adopted = "drawn" if drawn_key >= strike_key else "strike"
    return adopted, drawn, strike


def demote_losing_branch(states):
    """A rejected fork branch can no longer claim legacy continuity."""
    result = states.copy()
    result[result == CODE_LEGACY] = CODE_UNCLEAR
    return result


def run_slices(states):
    if len(states) == 0:
        return []
    changes = np.flatnonzero(np.diff(states.astype(np.int16)) != 0)
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes, [len(states) - 1]))
    return list(zip(starts.tolist(), ends.tolist()))


def path_runs(prefix, stations, arc, states, extent, scale):
    unit_scale = float(np.hypot(scale[0], scale[1]) / np.sqrt(2)) if scale else 1.0
    rows = []
    for start, end in run_slices(states):
        code = int(states[start])
        midpoint = stations[(start + end) // 2]
        length = float(arc[end] - arc[start])
        rows.append({
            "path": prefix,
            "state": STATE_LABELS.get(code, "unsupported"),
            "stations": end - start + 1,
            "from_px": round(float(arc[start]), 3),
            "to_px": round(float(arc[end]), 3),
            "length_mm": round(length * unit_scale, 4) if scale else None,
            "midpoint_normalized": [round(float(midpoint[0]), 6), round(float(midpoint[1]), 6)],
        })
    return rows


def _band_canvas(stations, extent, radius):
    width, height = extent
    canvas = Image.new("L", extent, 0)
    ImageDraw.Draw(canvas).line(
        [(float(x * width), float(y * height)) for x, y in stations],
        fill=255, width=max(2, radius * 2 + 1), joint="curve",
    )
    return np.asarray(canvas, dtype=bool)


def evidence_raster(paths, extent, radius, common, area):
    """Freeze per-pixel evidence so later footprint overlaps stay reproducible."""
    label = np.full(extent[::-1], CODE_OUTSIDE, dtype=np.uint8)
    label[common] = CODE_COMMON_UNTRACED
    label[area & ~common] = CODE_UNSUPPORTED
    # Drawn trunk/continuation is painted first; the fork strike fills only
    # pixels the drawn path does not already claim at the shared junction.
    for stations, states in paths:
        band = _band_canvas(stations, extent, radius)
        for start, end in run_slices(states):
            code = int(states[start])
            if code == CODE_UNSUPPORTED:
                continue
            segment = _band_canvas(stations[start:end + 1], extent, radius)
            mask = segment & common & (label == CODE_COMMON_UNTRACED)
            label[mask] = code
        unsupported_band = band & area & ~common
        label[unsupported_band & (label == CODE_OUTSIDE)] = CODE_UNSUPPORTED
    return label


def support_mask_png(mask):
    return png_bytes(Image.fromarray((mask.astype(np.uint8) * 255), mode="L"))


def evidence_mask_png(label):
    return png_bytes(Image.fromarray(label, mode="L"))


def legacy_overlay(frames, paths, junction, extent, radius):
    image = base_rgba(frames.diffuse).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = extent
    shadow_width = max(3, radius * 2 + 4)
    line_width = max(2, radius * 2 + 1)
    for stations, states in paths:
        points = [(float(x * width), float(y * height)) for x, y in stations]
        for start, end in run_slices(states):
            code = int(states[start])
            color = (120, 124, 128) if code == CODE_UNSUPPORTED else STATE_COLORS.get(code, (200, 200, 200))
            segment = points[start:end + 1]
            draw.line(segment, fill=(24, 30, 33), width=shadow_width, joint="curve")
            draw.line(segment, fill=color, width=line_width, joint="curve")
    if junction is not None:
        jx, jy = junction[0] * width, junction[1] * height
        r = max(6, radius + 3)
        draw.ellipse((jx - r, jy - r, jx + r, jy + r), fill=(24, 30, 33),
                     outline=(246, 139, 96), width=3)
        draw.text((jx + r + 3, jy - 9), "fork", fill=(246, 139, 96), font=font(15),
                  stroke_width=2, stroke_fill=(24, 30, 33))
    return image


def mask_state_fractions(label, footprint_mask):
    """Split a saved observation footprint over the frozen legacy evidence states."""
    footprint = footprint_mask.astype(bool)
    total = int(np.count_nonzero(footprint))
    if total == 0:
        raise InputError("The observation footprint has no pixels on this plane.")
    counts = {
        "legacy": int(np.count_nonzero(footprint & (label == CODE_LEGACY))),
        "crack_only": int(np.count_nonzero(footprint & (label == CODE_CRACK))),
        "unclear": int(np.count_nonzero(footprint & (label == CODE_UNCLEAR))),
        "common_untraced": int(np.count_nonzero(footprint & (label == CODE_COMMON_UNTRACED))),
        "unsupported": int(np.count_nonzero(footprint & (label == CODE_UNSUPPORTED))),
        "outside_region": int(np.count_nonzero(footprint & (label == CODE_OUTSIDE))),
    }
    fractions = {key: round(100 * value / total, 3) for key, value in counts.items()}
    no_photo = counts["unsupported"] + counts["outside_region"]
    return {
        "footprint_pixels": total,
        "pixels": counts,
        "percent": fractions,
        "no_photo_evidence_pixels": no_photo,
        "no_photo_evidence_percent": round(100 * no_photo / total, 3),
        "photographed_percent": round(100 * (total - no_photo) / total, 3),
    }


def observation_footprint(extent, observation):
    outline = observation.get("outline") or []
    if len(outline) < 3:
        raise InputError("This observation has no saved outline footprint. Re-run the surface observation tool.")
    return polygon_mask(extent, observation["outline"])


def measure_legacy(store, tile, frames, area, options):
    """Full triple-capture separation, returning frozen metrics and an overlay."""
    extent = frames.diffuse.extent
    common, unclipped = common_support(frames, area)
    trace_points, fork = parse_trace({"trace": options["trace"], "fork_strike": options["fork_strike"]})

    main_stations, main_arc = resample_polyline(trace_points, extent, options["spacing_px"])
    junction_arc, junction_xy, junction_gap = locate_junction(
        trace_points, fork, extent, float(main_arc[-1]),
    )
    strike_stations, strike_arc = resample_polyline(fork, extent, options["spacing_px"])

    main_states, thresholds = classify_path(frames, main_stations, extent, common, unclipped, options)
    strike_states, strike_thresholds = classify_path(frames, strike_stations, extent, common, unclipped, options)
    thresholds = {
        **thresholds,
        "strike_raking_balance_gain": strike_thresholds["raking_balance_gain"],
    }

    after_junction = main_arc > junction_arc + 1e-9
    drawn_branch_states = main_states[after_junction]
    trunk_states = main_states[~after_junction]
    adopted, drawn_tally, strike_tally = decide_fork(drawn_branch_states, strike_states)

    final_main = main_states.copy()
    final_strike = strike_states.copy()
    if adopted == "strike":
        # The drawn continuation loses: no legacy claim survives on that branch.
        final_main[after_junction] = demote_losing_branch(drawn_branch_states)
    else:
        final_strike = demote_losing_branch(strike_states)

    paths = [
        ("main", main_stations, main_arc, final_main),
        ("fork_strike", strike_stations, strike_arc, final_strike),
    ]
    label = evidence_raster(
        [(stations, states) for _, stations, _, states in paths],
        extent, options["radius"], common, area,
    )
    overlay = legacy_overlay(frames, [(stations, states) for _, stations, _, states in paths],
                             junction_xy, extent, options["radius"])

    scale = frames.diffuse.scale
    runs = (
        path_runs("main", main_stations, main_arc, final_main, extent, scale)
        + path_runs("fork_strike", strike_stations, strike_arc, final_strike, extent, scale)
    )
    supported_codes = (CODE_LEGACY, CODE_CRACK, CODE_UNCLEAR)
    def state_counts(states):
        return {STATE_LABELS[code]: int(np.count_nonzero(states == code)) for code in supported_codes}

    metrics = {
        "adopted_continuation": adopted,
        "junction_normalized": [round(float(junction_xy[0]), 6), round(float(junction_xy[1]), 6)],
        "junction_gap_px": round(junction_gap, 3),
        "fork_decision": {
            "drawn_branch": drawn_tally,
            "strike_branch": strike_tally,
            "rule": "Most diffuse-backed legacy stations wins; ties break on fewer crack stations, then the drawn branch.",
        },
        "thresholds": thresholds,
        "trunk_station_counts": state_counts(trunk_states),
        "runs": runs,
        "common_support_pixels": int(np.count_nonzero(common)),
        "common_unclipped_pixels": int(np.count_nonzero(unclipped)),
        "plane_size": list(extent),
        "length_unit": "mm" if scale else "px",
        "evidence_mask_digest": store.blobs.put(evidence_mask_png(label)),
        "common_support_digest": store.blobs.put(support_mask_png(common)),
        "note": (
            "Legacy = dark in diffuse without a raking sign flip. Crack-only = raking sign flip without "
            "diffuse darkness. Unclear = conflicting or weak evidence. Stations outside the three-photo "
            "common visible area are not classified."
        ),
    }
    legend = {"mode": "categorical", "labels": [
        {"name": "Adopted legacy rule", "color": "#f8ca66"},
        {"name": "Crack-only evidence", "color": "#96cde8"},
        {"name": "Unclear", "color": "#aab2be"},
    ]}
    return metrics, overlay, legend


def load_evidence_mask(store, digest):
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(store.blobs.get(digest))) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def footprint_overlap(store, legacy_record, defect_record, observation_key):
    if legacy_record["kind"] != "legacy":
        raise InputError("Choose a Legacy rule separation measurement.")
    if defect_record["kind"] != "defects":
        raise InputError("Choose a Surface observations measurement.")
    observation = next(
        (item for item in defect_record["metrics"]["observations"] if item["key"] == observation_key),
        None,
    )
    if observation is None:
        raise InputError("That observation is no longer part of the selected surface measurement.")
    extent = tuple(legacy_record["metrics"]["plane_size"])
    label = load_evidence_mask(store, legacy_record["metrics"]["evidence_mask_digest"])
    if tuple(label.shape[::-1]) != extent:
        raise InputError("The stored evidence mask no longer matches this measurement's plane size.")
    footprint = observation_footprint(extent, observation)
    breakdown = mask_state_fractions(label, footprint)
    return {
        "legacy_measurement_id": legacy_record["id"],
        "defect_measurement_id": defect_record["id"],
        "observation_key": observation_key,
        "observation_type": observation["type"],
        **breakdown,
    }

