from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

from .evidence import result_entry_names
from .model import ANALYSIS_LABELS, InputError, digest_identity, identity, integer, string
from .photos import bounded_preview, png_bytes
from .release import (
    CROP_HEIGHT,
    CROP_MEMBER,
    CROP_WIDTH,
    REPORT_MEMBER,
    THUMB_MEMBER,
    TIP_MARGIN_PX,
    readings_csv,
    release_html,
)
from . import release as release_module


MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_EXAMINED_BYTES = 72 * 1024 * 1024
MAX_MEMBERS = 130
MAX_MEMBER_BYTES = 24 * 1024 * 1024
MAX_MANIFEST = 6 * 1024 * 1024
MAX_PACKET_PIXELS = 3_000_000


def archive_key(value):
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or value != str(path)
        or ".." in path.parts
        or len(path.parts) > 4
        or any(len(part) > 125 for part in path.parts)
    ):
        raise InputError("Archive contains a filename outside the evidence packet layout.")
    return value


def archive_index(archive):
    items = archive.infolist()
    if not items or len(items) > MAX_MEMBERS:
        raise InputError("Choose an evidence ZIP with 130 entries or fewer.")
    names = {}
    size = 0
    for info in items:
        name = archive_key(info.filename)
        if name in names:
            raise InputError("The evidence archive has repeated file names.")
        if info.is_dir() or info.flag_bits & 1:
            raise InputError("Evidence archives require regular, readable entries.")
        if info.file_size > MAX_MEMBER_BYTES:
            raise InputError("An evidence entry exceeds the per-entry size limit.")
        size += info.file_size
        if size > MAX_EXAMINED_BYTES:
            raise InputError("The expanded evidence archive exceeds 72 MiB.")
        names[name] = info
    return names


def read_member(archive, names, key, limit=MAX_MEMBER_BYTES):
    if key not in names:
        raise InputError(f"The archive is missing {key}.")
    info = names[key]
    if info.file_size > limit:
        raise InputError(f"{key} exceeds the reader's size limit.")
    with archive.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) != info.file_size or len(data) > limit:
        raise InputError(f"{key} does not match its declared size.")
    return data


def reject_duplicate_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InputError("Evidence JSON has repeated property names.")
        value[key] = item
    return value


def reject_nonfinite(value):
    raise InputError(f"Unsupported numerical constant in evidence JSON: {value}.")


def archive_json(data, title):
    try:
        record = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_nonfinite,
            object_pairs_hook=reject_duplicate_fields,
        )
    except (UnicodeError, ValueError, RecursionError) as error:
        raise InputError(f"{title} is not a readable evidence record.") from error
    if not isinstance(record, dict):
        raise InputError(f"{title} must contain an object.")
    return record


def member_digest(archive, names, key, claimed):
    digest_identity(claimed)
    data = read_member(archive, names, key)
    actual = sha256(data).hexdigest()
    if actual != claimed:
        raise InputError(f"{key} does not match the packet's SHA-256 digest.")
    return data


def decoded_panel(data, label):
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise InputError(f"{label} must be a single PNG image.")
            if min(image.size) < 12 or image.width * image.height > MAX_PACKET_PIXELS:
                raise InputError(f"{label} falls outside the evidence display size.")
            panel = bounded_preview(image.convert("RGBA"), 1280)
            return png_bytes(panel)
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InputError(f"{label} cannot be decoded.") from error


def original_index(names):
    entries = {}
    for key in names:
        if not key.startswith("originals/"):
            continue
        stem = PurePosixPath(key).stem
        short = stem.rsplit("-", 1)[-1]
        if short in entries:
            raise InputError("More than one evidence original uses the same digest prefix.")
        entries[short] = key
    return entries


def read_originals(archive, names, documents):
    if not isinstance(documents, list) or len(documents) > 24:
        raise InputError("Evidence contains too many original captures.")
    index = original_index(names)
    listed = {}
    for document in documents:
        if not isinstance(document, dict):
            raise InputError("An original capture descriptor is incomplete.")
        expected = digest_identity(document.get("digest"))
        key = index.get(expected[:10])
        if key is None or expected in listed:
            raise InputError("An original capture does not have a unique archive entry.")
        info = names[key]
        member_digest(archive, names, key, expected)
        listed[expected] = {
            "digest": expected,
            "bytes": info.file_size,
            "name": string(document.get("name"), "Original name", maximum=140),
            "role": string(document.get("role"), "Lighting role", maximum=32),
        }
    if len(index) != len(listed):
        raise InputError("Some original photos were not declared in the packet.")
    return list(listed.values())


def measurement_descriptor(entry, expected_tile):
    if not isinstance(entry, dict):
        raise InputError("Evidence has an invalid measurement descriptor.")
    record_id = identity(entry.get("id"), "Measurement ID")
    if entry.get("tile_id") != expected_tile:
        raise InputError("An evidence measurement belongs to a different specimen.")
    kind = entry.get("kind")
    label = entry.get("label")
    if not isinstance(kind, str) or kind not in ANALYSIS_LABELS or ANALYSIS_LABELS[kind] != label:
        raise InputError("Evidence measurement type or label is unsupported.")
    return record_id, kind, label


def trusted_provenance(record, expected_id):
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("specimen"), dict):
        raise InputError("A measurement lacks specimen provenance.")
    source = provenance["specimen"]
    if source.get("id") != expected_id:
        raise InputError("Measurement provenance points to a different specimen.")
    photo = provenance.get("photo")
    if not isinstance(photo, dict):
        raise InputError("A measurement lacks capture provenance.")
    return {
        "name": string(photo.get("display_name"), "Capture name", maximum=140),
        "role": string(photo.get("role"), "Capture role", maximum=32),
        "source_digest": digest_identity(photo.get("original_digest"), "Original photograph digest"),
        "working_digest": digest_identity(photo.get("working_digest"), "Working image digest"),
        "source_version": integer(record.get("source_version"), "Source revision", lower=1, upper=2_000_000_000),
    }


def import_measurements(store, archive, names, records, specimen):
    if not isinstance(records, list) or not 1 <= len(records) <= 12:
        raise InputError("Evidence needs between 1 and 12 measurement records.")
    collected = []
    seen = set()
    for record in records:
        identifier, kind, label = measurement_descriptor(record, specimen)
        if identifier in seen:
            raise InputError("Evidence repeats a measurement.")
        seen.add(identifier)
        stem = "measurements/" + result_entry_names(record)
        receipt = archive_json(read_member(archive, names, stem + ".json", MAX_MANIFEST), label)
        if record != receipt:
            raise InputError(f"The separate {label} record differs from the manifest.")
        pictures = record.get("pictures")
        if not isinstance(pictures, dict):
            raise InputError("Measurement picture manifest is missing.")
        images = {}
        for layer in ("source", "overlay", "paired"):
            digest = pictures.get(layer)
            if layer == "paired" and digest is None:
                continue
            data = member_digest(archive, names, stem + f"-{layer}.png", digest)
            images[layer] = store.blobs.put(decoded_panel(data, f"{label} {layer}"))
        provenance = trusted_provenance(record, specimen)
        collected.append({
            "id": identifier,
            "label": label,
            "kind": kind,
            "source_version": provenance["source_version"],
            "capture": provenance,
            "pictures": images,
        })
    return collected


def import_sheets(store, archive, names, declarations):
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 2:
        raise InputError("Evidence needs one or two sheet digests.")
    sanitized = []
    for index, expected in enumerate(declarations, 1):
        key = f"sheets/sheet-{index:02d}.png"
        data = member_digest(archive, names, key, expected)
        sanitized.append(store.blobs.put(decoded_panel(data, f"Inspection sheet {index}")))
    return sanitized


# ---------------------------------------------------------------------------
# Redacted crop packets
# ---------------------------------------------------------------------------


def crop_geometry(manifest):
    crop = manifest.get("crop")
    if not isinstance(crop, dict):
        raise InputError("Crop evidence is missing its crop descriptor.")
    for key in ("left", "top"):
        integer(crop.get(key), f"Crop {key}", lower=0, upper=2_000_000_000)
    if integer(crop.get("width"), "Crop width", lower=1, upper=100_000) != CROP_WIDTH:
        raise InputError("Crop evidence must keep a 900 pixel wide crop.")
    if integer(crop.get("height"), "Crop height", lower=1, upper=100_000) != CROP_HEIGHT:
        raise InputError("Crop evidence must keep a 650 pixel tall crop.")
    if integer(crop.get("band_pixels_in_crop"), "Band pixel check", lower=0, upper=0) != 0:
        raise InputError("The crop descriptor admits confidentiality band pixels.")
    if integer(crop.get("bands_excluded"), "Excluded band count", lower=1, upper=8) < 1:
        raise InputError("Crop evidence must declare at least one excluded confidentiality band.")
    if float(crop.get("crack_tip_margin_px", 0)) < TIP_MARGIN_PX:
        raise InputError("The declared crack tip sits too close to the crop edge.")
    return crop


def crop_capture(manifest):
    capture = manifest.get("capture")
    if not isinstance(capture, dict):
        raise InputError("Crop evidence is missing its capture descriptor.")
    width = integer(capture.get("width"), "Capture width", lower=CROP_WIDTH, upper=20_000)
    height = integer(capture.get("height"), "Capture height", lower=CROP_HEIGHT, upper=20_000)
    return {
        "id": identity(capture.get("id"), "Capture ID"),
        "name": string(capture.get("name"), "Capture name", maximum=140),
        "role": string(capture.get("role"), "Capture role", maximum=32),
        "width": width,
        "height": height,
        "original_digest": digest_identity(capture.get("original_digest"), "Original photograph digest"),
        "working_digest": digest_identity(capture.get("working_digest"), "Working image digest"),
    }


def crop_marks(manifest):
    marks = manifest.get("marks")
    if not isinstance(marks, dict):
        raise InputError("Crop evidence is missing its release marks.")
    if identity(marks.get("photo_id"), "Marked capture ID") != manifest["capture"]["id"]:
        raise InputError("The release marks point at a different capture.")
    parsed = release_module.release_marks(marks)
    return parsed


def verify_crop_geometry(manifest, marks):
    """Re-run the sender's planner independently from the declared marks."""
    photo = {
        "width": manifest["capture"]["width"],
        "height": manifest["capture"]["height"],
    }
    planned = release_module.plan_crop(photo, marks)
    crop = manifest["crop"]
    for key in ("left", "top", "width", "height", "ruler_retained_percent",
                "ruler_fully_retained", "crack_tip_crop_xy", "crack_tip_margin_px",
                "bands_excluded", "band_pixels_in_crop"):
        if crop.get(key) != planned[key]:
            raise InputError(f"The declared crop geometry ({key}) does not follow its marked bands and tip.")
    capture = manifest["capture"]
    if crop["left"] + CROP_WIDTH > capture["width"] or crop["top"] + CROP_HEIGHT > capture["height"]:
        raise InputError("The crop window falls outside the declared capture.")


def crop_png_member(archive, names, key, digest, label):
    data = member_digest(archive, names, key, digest)
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise InputError(f"{label} must be a single PNG image.")
            return image.size, data
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InputError(f"{label} cannot be decoded.") from error


def import_crop_readings(archive, names, declarations, capture_id):
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 12:
        raise InputError("Crop evidence needs between 1 and 12 measurement readings.")
    collected = []
    seen = set()
    for entry in declarations:
        if not isinstance(entry, dict):
            raise InputError("A reading descriptor is incomplete.")
        member = string(entry.get("member"), "Reading member", maximum=125)
        csv_member = string(entry.get("csv_member"), "Reading CSV member", maximum=125)
        if not member.startswith("readings/") or not csv_member.startswith("readings/"):
            raise InputError("Readings must live under the readings/ folder.")
        data = member_digest(archive, names, member, digest_identity(entry.get("digest"), "Reading digest"))
        descriptor = archive_json(data, "Measurement reading")
        if entry.get("descriptor") != descriptor:
            raise InputError("A reading descriptor differs from its archived JSON.")
        csv_data = read_member(archive, names, csv_member)
        expected_csv = readings_csv([descriptor])
        if sha256(csv_data).hexdigest() != sha256(expected_csv).hexdigest():
            raise InputError("A reading CSV does not rebuild from its declared JSON.")
        record_id = identity(descriptor.get("id"), "Measurement ID")
        if record_id in seen:
            raise InputError("Crop evidence repeats a measurement reading.")
        seen.add(record_id)
        kind = descriptor.get("kind")
        label = descriptor.get("label")
        if not isinstance(kind, str) or kind not in ANALYSIS_LABELS or ANALYSIS_LABELS[kind] != label:
            raise InputError("Crop evidence contains an unsupported measurement type.")
        reading_capture = descriptor.get("capture")
        if not isinstance(reading_capture, dict) or reading_capture.get("id") != capture_id:
            raise InputError("A reading belongs to a different capture than the crop.")
        digest_identity(reading_capture.get("original_digest"), "Reading original digest")
        digest_identity(reading_capture.get("working_digest"), "Reading working digest")
        collected.append({"descriptor": descriptor, "member": member, "csv_member": csv_member,
                          "digest": entry["digest"]})
    return collected


def check_crop_layout(names):
    allowed = {"manifest.json", REPORT_MEMBER, CROP_MEMBER, THUMB_MEMBER}
    for name in names:
        if name in allowed or name.startswith("readings/"):
            continue
        raise InputError("Crop evidence packets contain only crop, readings and report members.")
    if any(name.startswith("originals/") for name in names):
        raise InputError("Crop evidence must not include original photographs.")


def import_crop_packet(store, archive, names, manifest):
    check_crop_layout(names)
    tile_id = identity(manifest.get("tile_id"), "Specimen ID")
    caption = string(manifest.get("title"), "Evidence title", maximum=115)
    notes = string(manifest.get("notes"), "Evidence notes", maximum=2400, required=False)
    revision = integer(manifest.get("tile_version_at_pin"), "Specimen revision", lower=1, upper=2_000_000_000)
    capture = crop_capture(manifest)
    marks = crop_marks(manifest)
    crop = crop_geometry(manifest)
    verify_crop_geometry({**manifest, "capture": capture, "crop": crop}, marks)
    crop_size, crop_data = crop_png_member(
        archive, names, CROP_MEMBER,
        digest_identity(crop.get("crop_digest"), "Crop image digest"), "Evidence crop",
    )
    if crop_size != (CROP_WIDTH, CROP_HEIGHT):
        raise InputError("The crop PNG is not exactly 900 x 650 pixels.")
    thumb_data = member_digest(
        archive, names, THUMB_MEMBER,
        digest_identity(crop.get("thumbnail_digest"), "Thumbnail digest"),
    )
    try:
        with Image.open(BytesIO(thumb_data)) as thumb:
            thumb_size = thumb.size
            if thumb.format != "PNG" or getattr(thumb, "n_frames", 1) != 1:
                raise InputError("Crop thumbnail must be a single PNG image.")
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InputError("Crop thumbnail cannot be decoded.") from error
    declared_thumb_size = crop.get("thumbnail_size")
    if not isinstance(declared_thumb_size, list) or list(thumb_size) != [
        integer(value, "Thumbnail size", lower=1, upper=CROP_WIDTH) for value in declared_thumb_size
    ]:
        raise InputError("The thumbnail dimensions do not match its declaration.")
    # The thumbnail must be a downscale of the crop, never another picture.
    with Image.open(BytesIO(crop_data)) as source:
        rebuilt_thumb = png_bytes(bounded_preview(source.convert("RGBA"), 480))
    if rebuilt_thumb != thumb_data:
        raise InputError("The thumbnail is not derived from the included crop.")
    readings = import_crop_readings(archive, names, manifest.get("readings"), capture["id"])
    report_data = read_member(archive, names, REPORT_MEMBER)
    record_for_report = {**manifest, "capture": capture, "crop": crop}
    if report_data != release_html(record_for_report):
        raise InputError("The report does not rebuild from the verified crop manifest.")
    stored_crop = store.blobs.put(crop_data)
    stored_thumb = store.blobs.put(thumb_data)
    return {
        "packet_kind": "crop_release",
        "source_evidence_id": identity(manifest.get("id"), "Evidence ID"),
        "external_tile_id": tile_id,
        "title": caption,
        "notes": notes,
        "specimen": string(manifest.get("tile_name"), "Specimen name"),
        "specimen_revision": revision,
        "capture": capture,
        "crop": {**crop, "image": stored_crop, "thumbnail": stored_thumb, "thumbnail_size": list(thumb_size)},
        "marks": marks,
        "readings": readings,
        "report_date": string(manifest.get("created_at"), "Recorded date", maximum=100),
        "synthetic": manifest.get("synthetic") is True,
    }


def viewable_packet(store, archive):
    entries = archive_index(archive)
    manifest = archive_json(read_member(archive, entries, "manifest.json", MAX_MANIFEST), "Manifest")
    if manifest.get("packet_kind") == "crop_release":
        return import_crop_packet(store, archive, entries, manifest)
    return import_full_packet(store, archive, entries, manifest)


def import_full_packet(store, archive, entries, manifest):
    source_id = identity(manifest.get("id"), "Evidence ID")
    tile_id = identity(manifest.get("tile_id"), "Specimen ID")
    caption = string(manifest.get("title"), "Evidence title", maximum=115)
    notes = string(manifest.get("notes"), "Evidence notes", maximum=2400, required=False)
    original_images = read_originals(archive, entries, manifest.get("originals", []))
    measurements = import_measurements(store, archive, entries, manifest.get("measurements"), tile_id)
    sheets = import_sheets(store, archive, entries, manifest.get("sheets"))
    revision = integer(manifest.get("tile_version_at_pin"), "Specimen revision", lower=1, upper=2_000_000_000)
    return {
        "source_evidence_id": source_id,
        "external_tile_id": tile_id,
        "title": caption,
        "notes": notes,
        "specimen": string(manifest.get("tile_name"), "Specimen name"),
        "specimen_revision": revision,
        "originals": original_images,
        "measurements": measurements,
        "sheets": sheets,
        "report_date": string(manifest.get("created_at"), "Recorded date", maximum=100),
        "synthetic": manifest.get("synthetic") is True,
    }


def inspect_packet(store, raw):
    if not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise InputError("Choose a nonempty CoreSection evidence ZIP of at most 24 MiB.")
    digest = sha256(raw).hexdigest()
    existing = store.received_by_digest(digest)
    if existing:
        return existing, False
    try:
        with ZipFile(BytesIO(raw)) as archive:
            record = viewable_packet(store, archive)
    except (BadZipFile, NotImplementedError, OSError, RuntimeError) as error:
        raise InputError("The file could not be read as a CoreSection evidence ZIP.") from error
    record["archive_digest"] = digest
    record["packed_bytes"] = len(raw)
    return store.save_received(record)
