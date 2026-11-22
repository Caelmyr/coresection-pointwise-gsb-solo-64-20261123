"""Redacted crop evidence packets for external reviewers.

A release packet never contains the full capture.  It carries one continuous
``CROP_WIDTH x CROP_HEIGHT`` pixel crop cut from the registered working image,
a thumbnail derived from that crop, the pinned measurement readings and a
report that identifies the artefact as a partial crop.  The marked
confidentiality bands only define geometry: none of their pixels are rendered
into any packet member.
"""

from hashlib import sha256
from html import escape
from io import StringIO
import csv

from PIL import Image

from .evidence import artifact_json, csv_cell, result_entry_names
from .model import (
    ANALYSIS_LABELS,
    InputError,
    collection,
    dictionary,
    identity,
    numeric_pair,
    scalar,
    string,
    tags,
    unique_ids,
)
from .photos import THUMBNAIL_EDGE, bounded_preview, decode_capture, png_bytes


CROP_WIDTH = 900
CROP_HEIGHT = 650
# Extra working pixels kept between a confidentiality band edge and the crop.
BAND_GUARD_PX = 8
# A crack tip must sit at least this far inside the crop so reviewers can see it.
TIP_MARGIN_PX = 24
MAX_BANDS = 8
MAX_RELEASE_RESULTS = 12
MAX_NOTES = 2400

CROP_MEMBER = "crop/crop-900x650.png"
THUMB_MEMBER = "crop/thumbnail.png"
REPORT_MEMBER = "report.html"


# ---------------------------------------------------------------------------
# Mark validation and geometry
# ---------------------------------------------------------------------------

def unit_rectangle(value, label):
    data = dictionary(value, label) if isinstance(value, dict) else {}
    left = scalar(data.get("left"), f"{label} left", lower=0.0, upper=1.0)
    top = scalar(data.get("top"), f"{label} top", lower=0.0, upper=1.0)
    right = scalar(data.get("right"), f"{label} right", lower=0.0, upper=1.0)
    bottom = scalar(data.get("bottom"), f"{label} bottom", lower=0.0, upper=1.0)
    if right - left < 0.001 or bottom - top < 0.001:
        raise InputError(f"{label} needs a visible area at least 0.1% of the capture.")
    if right <= left or bottom <= top:
        raise InputError(f"{label} right/bottom edges must follow its left/top edges.")
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def release_marks(data):
    photo_id = identity(data.get("photo_id"), "Capture ID")
    bands = [
        unit_rectangle(item, "Confidentiality band")
        for item in collection(data.get("bands"), "Confidentiality bands", minimum=1, maximum=MAX_BANDS)
    ]
    ruler = unit_rectangle(data.get("ruler"), "Core-number ruler")
    tip = numeric_pair(data.get("crack_tip"), "Crack tip", lower=0.0, upper=1.0)
    return {"photo_id": photo_id, "bands": bands, "ruler": ruler, "crack_tip": tip}


def pixel_rect(rect, width, height):
    return (
        rect["left"] * width,
        rect["top"] * height,
        rect["right"] * width,
        rect["bottom"] * height,
    )


def expanded(rect, guard):
    return (rect[0] - guard, rect[1] - guard, rect[2] + guard, rect[3] + guard)


def intersects(first, second):
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def contains(rect, point):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def overlap_area(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def plan_crop(photo, marks):
    width, height = photo["width"], photo["height"]
    if width < CROP_WIDTH or height < CROP_HEIGHT:
        raise InputError(
            f"The working capture is {width} x {height} px; a continuous {CROP_WIDTH} x {CROP_HEIGHT} crop does not fit."
        )
    bands_px = [pixel_rect(rect, width, height) for rect in marks["bands"]]
    ruler_px = pixel_rect(marks["ruler"], width, height)
    tip_px = (marks["crack_tip"][0] * width, marks["crack_tip"][1] * height)

    if any(contains(rect, tip_px) for rect in bands_px):
        raise InputError("The crack tip is marked inside a confidentiality band; re-mark it on visible rock.")
    if any(intersects(ruler_px, rect) for rect in bands_px):
        raise InputError("The core-number ruler overlaps a confidentiality band; the marks cannot both be honoured.")

    guard = BAND_GUARD_PX
    # Candidate window starts: every frame edge and every band edge on either
    # side, so the optimum (which hugs an edge) is always considered.  The
    # guard-enlarged intersection test below is the authoritative clearance
    # check and also handles full-width/full-height bands correctly.
    x_starts = {0, width - CROP_WIDTH}
    y_starts = {0, height - CROP_HEIGHT}
    for rect in bands_px:
        for edge in (rect[0] - guard - CROP_WIDTH, rect[2] + guard,
                     rect[0] - guard, rect[2] + guard - CROP_WIDTH):
            if 0 <= edge <= width - CROP_WIDTH:
                x_starts.add(int(edge))
        for edge in (rect[1] - guard - CROP_HEIGHT, rect[3] + guard,
                     rect[1] - guard, rect[3] + guard - CROP_HEIGHT):
            if 0 <= edge <= height - CROP_HEIGHT:
                y_starts.add(int(edge))

    def feasible(x0, y0):
        window = (x0, y0, x0 + CROP_WIDTH, y0 + CROP_HEIGHT)
        if any(intersects(window, expanded(rect, guard)) for rect in bands_px):
            return False
        inside_x = x0 + TIP_MARGIN_PX <= tip_px[0] <= x0 + CROP_WIDTH - TIP_MARGIN_PX
        inside_y = y0 + TIP_MARGIN_PX <= tip_px[1] <= y0 + CROP_HEIGHT - TIP_MARGIN_PX
        return inside_x and inside_y

    ranked = []
    for x0 in x_starts:
        for y0 in y_starts:
            if not feasible(x0, y0):
                continue
            window = (x0, y0, x0 + CROP_WIDTH, y0 + CROP_HEIGHT)
            ruler_keep = overlap_area(window, ruler_px)
            ruler_total = (ruler_px[2] - ruler_px[0]) * (ruler_px[3] - ruler_px[1])
            fraction = ruler_keep / ruler_total if ruler_total else 0.0
            fully = fraction >= 0.999
            tip_x = min(tip_px[0] - x0, x0 + CROP_WIDTH - tip_px[0])
            tip_y = min(tip_px[1] - y0, y0 + CROP_HEIGHT - tip_px[1])
            # Ruler retention dominates; a fully retained ruler wins outright,
            # then a centred crack tip breaks ties deterministically.
            score = (
                1 if fully else 0,
                round(fraction, 6),
                round(min(tip_x, tip_y), 3),
                -x0,
                -y0,
            )
            ranked.append((score, x0, y0, fraction, fully))
    if not ranked:
        raise InputError(
            "No 900 x 650 crop can clear the bands and keep the crack tip with a 24 px viewing margin."
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, x0, y0, fraction, fully = ranked[0]
    window = (x0, y0, x0 + CROP_WIDTH, y0 + CROP_HEIGHT)
    return {
        "left": x0,
        "top": y0,
        "width": CROP_WIDTH,
        "height": CROP_HEIGHT,
        "ruler_retained_percent": round(fraction * 100, 2),
        "ruler_fully_retained": fully,
        "crack_tip_crop_xy": [round(tip_px[0] - x0, 2), round(tip_px[1] - y0, 2)],
        "crack_tip_margin_px": round(
            min(tip_px[0] - x0, x0 + CROP_WIDTH - tip_px[0], tip_px[1] - y0, y0 + CROP_HEIGHT - tip_px[1]), 2
        ),
        "bands_excluded": len(marks["bands"]),
        # A feasible window never intersects a guard-expanded band, so this is
        # always zero; the field makes the exclusion explicit in the manifest.
        "band_pixels_in_crop": 0,
        "band_guard_px": BAND_GUARD_PX,
        "tip_margin_px": TIP_MARGIN_PX,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_crop(store, photo, geometry):
    with Image.open(store.blobs.require(photo["working_digest"])) as image:
        image.load()
        if image.width != photo["width"] or image.height != photo["height"]:
            raise InputError("The stored working capture does not match its recorded dimensions.")
        crop = image.crop((geometry["left"], geometry["top"],
                           geometry["left"] + CROP_WIDTH, geometry["top"] + CROP_HEIGHT))
        if crop.size != (CROP_WIDTH, CROP_HEIGHT):
            raise InputError("The crop planner returned an unusable crop window.")
        crop_rgba = crop.convert("RGBA")
        crop_bytes = png_bytes(crop_rgba)
        thumb = bounded_preview(crop_rgba, THUMBNAIL_EDGE)
        thumb_bytes = png_bytes(thumb)
    return crop_bytes, thumb_bytes, thumb.size


def assert_band_clear(photo, marks, geometry):
    """Independent geometry guard proving no marked band reaches the crop."""
    width, height = photo["width"], photo["height"]
    window = (
        geometry["left"], geometry["top"],
        geometry["left"] + CROP_WIDTH, geometry["top"] + CROP_HEIGHT,
    )
    for index, rect in enumerate(marks["bands"], 1):
        band = pixel_rect(rect, width, height)
        if intersects(window, band):
            raise InputError(f"Confidentiality band {index} would be cropped; refuse to build the packet.")


# ---------------------------------------------------------------------------
# Measurement readings
# ---------------------------------------------------------------------------

def reading_descriptor(record):
    capture = record["provenance"]["photo"]
    scalar_metrics = {
        key: value
        for key, value in record["metrics"].items()
        if isinstance(value, (str, bool, int, float)) and len(str(value)) <= 180
    }
    return {
        "id": record["id"],
        "kind": record["kind"],
        "label": record["label"],
        "source_version": record["source_version"],
        "basis": record["basis"],
        "capture": {
            "id": capture["id"],
            "name": capture["display_name"],
            "role": capture["role"],
            "original_digest": capture["original_digest"],
            "working_digest": capture["working_digest"],
        },
        "metric_values": scalar_metrics,
    }


def readings_csv(descriptors):
    stream = StringIO(newline="")
    out = csv.writer(stream, lineterminator="\r\n")
    for descriptor in descriptors:
        out.writerow(["measurement_id", "kind", "label", "source_version", "basis"])
        out.writerow([
            descriptor["id"], descriptor["kind"], descriptor["label"],
            descriptor["source_version"], descriptor["basis"],
        ])
        out.writerow([])
        out.writerow(["metric", "value"])
        for key, value in sorted(descriptor["metric_values"].items()):
            out.writerow([key, csv_cell(value)])
        out.writerow([])
    return stream.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Packet assembly
# ---------------------------------------------------------------------------

def release_fields(store, specimen_id, payload):
    data = dictionary(payload, "Crop evidence packet")
    marks = release_marks(data)
    targets = unique_ids(data.get("result_ids"), "Measurement", minimum=1, maximum=MAX_RELEASE_RESULTS)
    title = string(data.get("title"), "Evidence title", maximum=115)
    notes = string(data.get("notes"), "Evidence notes", maximum=MAX_NOTES, required=False)
    tile = store.tile(specimen_id)
    photo = next((item for item in tile["photos"] if item["id"] == marks["photo_id"]), None)
    if photo is None:
        raise InputError("The marked capture is not part of this specimen.")
    records = []
    for target in targets:
        record = store.result(target)
        if record["tile_id"] != tile["id"]:
            raise InputError("A crop packet can contain measurements from its own specimen only.")
        if record["provenance"]["photo"]["id"] != photo["id"]:
            raise InputError(
                f"Measurement {record['label']} is pinned to another capture; select readings from the marked photo."
            )
        records.append(record)
    return data, tile, photo, marks, records, title, notes


def create_release(store, specimen_id, payload):
    data, tile, photo, marks, records, title, notes = release_fields(store, specimen_id, payload)
    geometry = plan_crop(photo, marks)
    assert_band_clear(photo, marks, geometry)
    crop_bytes, thumb_bytes, thumb_size = render_crop(store, photo, geometry)
    crop_digest = store.blobs.put(crop_bytes)
    thumb_digest = store.blobs.put(thumb_bytes)
    descriptors = [reading_descriptor(record) for record in records]
    readings = [
        {
            "member": f"readings/{result_entry_names(record)}.json",
            "csv_member": f"readings/{result_entry_names(record)}.csv",
            "digest": sha256(artifact_json(descriptor)).hexdigest(),
            "descriptor": descriptor,
        }
        for record, descriptor in zip(records, descriptors)
    ]
    entry = {
        "packet_kind": "crop_release",
        "title": title,
        "notes": notes,
        "tags": tags(data.get("tags", [])),
        "tile_id": tile["id"],
        "tile_name": tile["name"],
        "tile_version_at_pin": tile["version"],
        "synthetic": tile["synthetic"],
        "capture": {
            "id": photo["id"],
            "name": photo["name"],
            "role": photo["role"],
            "width": photo["width"],
            "height": photo["height"],
            "original_digest": photo["source_digest"],
            "working_digest": photo["working_digest"],
        },
        "marks": marks,
        "crop": {
            **geometry,
            "member": CROP_MEMBER,
            "thumbnail_member": THUMB_MEMBER,
            "crop_digest": crop_digest,
            "thumbnail_digest": thumb_digest,
            "thumbnail_size": list(thumb_size),
        },
        "readings": readings,
    }
    return store.save_evidence(entry)


def release_html(entry):
    title = escape(entry["title"])
    name = escape(entry["tile_name"])
    date = escape(entry["created_at"])
    notes = escape(entry["notes"]).replace("\n", "<br>") or "No inspection notes."
    crop = entry["crop"]
    capture = entry["capture"]
    tip = crop["crack_tip_crop_xy"]
    reading_rows = "".join(
        f"<tr><td>{escape(item['descriptor']['label'])}</td>"
        f"<td>{escape(ANALYSIS_LABELS.get(item['descriptor']['kind'], item['descriptor']['kind']))}</td>"
        f"<td>{escape(str(item['descriptor']['source_version']))}</td></tr>"
        for item in entry["readings"]
    )
    ruler_line = (
        "The full marked core-number ruler is inside the crop."
        if crop["ruler_fully_retained"]
        else f"{crop['ruler_retained_percent']:.2f}% of the marked core-number ruler area is inside the crop."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:920px;margin:46px auto;padding:0 20px;background:#f9f5eb;color:#252e2d}}
h1,h2{{line-height:1.12}}
.notice{{border:2px solid #9a5b32;background:#f3e4d4;padding:14px 16px;border-radius:10px}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border-bottom:1px solid #ccc;text-align:left;padding:0.45rem 0;vertical-align:top}}
img{{max-width:100%;height:auto;background:#242b2b;border:1px solid #8b8578}}
.small{{font-size:0.92rem;color:#4d5855}}
</style></head><body>
<h1>{title}</h1>
<p class="notice"><strong>Partial crop, not the original photograph.</strong> This report contains one continuous
{crop['width']} &times; {crop['height']} pixel crop of a larger capture. The full original photograph is not included.</p>
<p>{name} · specimen revision {entry['tile_version_at_pin']} · {date}</p>
<p>{notes}</p>
<h2>Crop position on the registered capture</h2>
<table>
<tr><th>Source capture</th><td>{escape(capture['name'])} ({capture['width']} &times; {capture['height']} px)</td></tr>
<tr><th>Crop upper-left pixel</th><td>({crop['left']}, {crop['top']})</td></tr>
<tr><th>Crop size</th><td>{crop['width']} &times; {crop['height']} px, continuous, no resampling</td></tr>
<tr><th>Confidentiality bands excluded</th><td>{crop['bands_excluded']} marked band(s); none of their pixels are exported</td></tr>
<tr><th>Core-number ruler</th><td>{escape(ruler_line)}</td></tr>
<tr><th>Crack tip inside crop at</th><td>({tip[0]}, {tip[1]}) with {crop['crack_tip_margin_px']:.1f} px clearance to the nearest crop edge</td></tr>
<tr><th>Original capture digest (SHA-256)</th><td class="small">{escape(capture['original_digest'])}</td></tr>
<tr><th>Working image digest (SHA-256)</th><td class="small">{escape(capture['working_digest'])}</td></tr>
</table>
<figure><img src="crop/crop-900x650.png" alt="Continuous 900 by 650 pixel evidence crop, not the full photograph">
<figcaption>Continuous {crop['width']} &times; {crop['height']} px crop. Pixels outside this rectangle stay with the sender.</figcaption></figure>
<h2>Pinned measurement readings</h2>
<table><thead><tr><th>Measurement</th><th>Type</th><th>Specimen revision</th></tr></thead><tbody>{reading_rows}</tbody></table>
<p class="small">Full scalar readings and CSV tables are in <code>readings/</code>. Packet members are verified against the
SHA-256 digests in <code>manifest.json</code>. An intranet comparison against the retained original capture is required to
confirm this crop's position and content against the source.</p>
<footer>CoreSection · Redacted crop evidence</footer></body></html>""".encode("utf8")


def release_zip(store, entry):
    from zipfile import ZIP_DEFLATED, ZipFile
    from io import BytesIO

    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        archive.writestr("manifest.json", artifact_json(entry))
        archive.writestr(REPORT_MEMBER, release_html(entry))
        archive.writestr(CROP_MEMBER, store.blobs.get(entry["crop"]["crop_digest"]))
        archive.writestr(THUMB_MEMBER, store.blobs.get(entry["crop"]["thumbnail_digest"]))
        for record in entry["readings"]:
            descriptor = record["descriptor"]
            archive.writestr(record["member"], artifact_json(descriptor))
            archive.writestr(record["csv_member"], readings_csv([descriptor]))
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Intranet-only comparison against the retained original capture
# ---------------------------------------------------------------------------

def compare_original(store, received, raw_original):
    """Re-cut the declared window from the original capture and compare.

    This is the step a reviewer can only perform once the same-session
    original is available on the internal network.  Everything before it is
    verifiable from the packet alone.
    """
    if received.get("packet_kind") != "crop_release":
        raise InputError("Original comparison is available for redacted crop packets only.")
    capture = received["capture"]
    crop = received["crop"]
    photo = decode_capture(raw_original)
    if (photo.width, photo.height) != (capture["width"], capture["height"]):
        raise InputError(
            f"The original is {photo.width} x {photo.height} px but the packet records "
            f"{capture['width']} x {capture['height']} px."
        )
    working_bytes = png_bytes(photo.image)
    digest_match = sha256(working_bytes).hexdigest() == capture["working_digest"]
    raw_digest = sha256(raw_original).hexdigest()
    raw_match = raw_digest == capture["original_digest"]

    geometry = {key: crop[key] for key in ("left", "top", "width", "height")}
    re_cut = photo.image.crop((
        geometry["left"], geometry["top"],
        geometry["left"] + geometry["width"], geometry["top"] + geometry["height"],
    )).convert("RGBA")
    with Image.open(store.blobs.require(crop["image"])) as sent:
        sent.load()
        if sent.size != re_cut.size:
            raise InputError("The packet crop and the re-cut window have different dimensions.")
        sent_rgba = sent.convert("RGBA")
        differences = sum(
            1 for a, b in zip(sent_rgba.getdata(), re_cut.getdata()) if a != b
        )
    total = geometry["width"] * geometry["height"]
    pixel_match = differences == 0

    # Independently confirm the marked bands would still be excluded today.
    marks = received["marks"]
    band_breaches = 0
    for rect in marks["bands"]:
        left, top, right, bottom = pixel_rect(rect, photo.width, photo.height)
        clipped = (
            max(left, geometry["left"]),
            max(top, geometry["top"]),
            min(right, geometry["left"] + geometry["width"]),
            min(bottom, geometry["top"] + geometry["height"]),
        )
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            band_breaches += 1

    verdict = "match" if digest_match and raw_match and pixel_match and band_breaches == 0 else "mismatch"
    return {
        "verdict": verdict,
        "original_digest": raw_digest,
        "original_digest_match": raw_match,
        "working_digest_match": digest_match,
        "pixel_match": pixel_match,
        "mismatched_pixels": differences,
        "crop_pixels": total,
        "mismatched_percent": round(100 * differences / total, 4),
        "band_breaches": band_breaches,
        "re_cut_geometry": geometry,
        "capture_name": capture["name"],
        "note": (
            "Crop position, content and the original capture align."
            if verdict == "match"
            else "At least one check failed; treat the packet as unconfirmed against this original."
        ),
    }
