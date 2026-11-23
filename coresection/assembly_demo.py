"""Synthetic practice material for the core segment assembly workbench.

The scenario mirrors a real reassignment job:

* three photographed fracture faces with different downhole-arrow directions;
* one duplicated legacy number among the last two segments;
* two field orderings that both fit the same shelf;
* the legacy depth books favour the swapped ordering, while the visible
  laminae only continue cleanly in the other ordering;
* each segment carries one documented, unphotographed gap of known length.
"""

import numpy as np
from PIL import Image, ImageDraw

from .assembly import create_assembly
from .photos import linear_to_srgb, png_bytes


FACE_WIDTH = 220
DEMO_ASSEMBLY_KEY = "coresection.assembly.demo.v1"

PALETTE = np.array([
    (58, 49, 42),
    (118, 96, 74),
    (178, 152, 116),
    (206, 184, 150),
], dtype=np.float32) / 255.0


def _beds(y, x, phase, distortion=0.0):
    wave = 9.5 * np.sin(0.020 * x + phase) + 3.2 * np.sin(0.071 * x + 1.7 * phase)
    fold = 1.0 + distortion * np.exp(-((y - 6.0) / 26.0) ** 2)
    return np.sin(2 * np.pi * (y * fold + wave) / 15.0)


def _render_segment(seed, phase, length_mm, arrow, old_label, *, distortion=0.0, tone_bias=0.0):
    generator = np.random.default_rng(seed)
    height = 210 + seed % 5 * 26
    yy, xx = np.mgrid[0:height, 0:FACE_WIDTH].astype(np.float32)
    grain = generator.normal(size=(height, FACE_WIDTH)).astype(np.float32) * 0.16
    broad = np.sin(2 * np.pi * yy / 95.0 + seed) * 0.10
    signal = _beds(yy, xx, phase, distortion) * 0.62 + broad + grain + tone_bias
    stops = np.linspace(-1.05, 1.05, len(PALETTE))
    linear = np.stack(
        [np.interp(signal, stops, PALETTE[:, channel]) for channel in range(3)], axis=2,
    ).astype(np.float32)
    # Slightly irregular, darker fracture rims.
    rim = int(round(height * 0.045))
    linear[:rim, :, :] *= (0.62 + 0.08 * np.sin(xx[:rim, :] / 7.0))[..., None]
    linear[-rim:, :, :] *= (0.60 + 0.10 * np.cos(xx[-rim:, :] / 9.0))[..., None]
    image = Image.fromarray(np.rint(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    arrow_x = 16
    if arrow == "down":
        draw.line((arrow_x, 26, arrow_x, height - 26), fill=(245, 238, 220), width=4)
        draw.polygon([(arrow_x - 9, height - 30), (arrow_x + 9, height - 30), (arrow_x, height - 14)],
                     fill=(245, 238, 220))
    else:
        draw.line((arrow_x, 26, arrow_x, height - 26), fill=(245, 238, 220), width=4)
        draw.polygon([(arrow_x - 9, 30), (arrow_x + 9, 30), (arrow_x, 14)], fill=(245, 238, 220))
    draw.rectangle((FACE_WIDTH - 92, 10, FACE_WIDTH - 10, 34), fill=(28, 31, 30))
    draw.text((FACE_WIDTH - 86, 15), old_label, fill=(240, 232, 214))
    return png_bytes(image)


def _segment_specs():
    # Segments A and B are adjacent cuts of the same bedded interval; C carries
    # a folded disturbance at its photographed uphole rim and a different phase.
    return [
        {
            "segment_key": "A",
            "length_mm": 215.0,
            "downhole_arrow": "down",
            "old_label": "R-14",
            "old_depth_top_m": 412.30,
            "old_depth_bottom_m": 412.515,
            "gap_end": "uphole",
            "gap_length_mm": 18.0,
            "note": "Fixed uphole segment.",
            "seed": 41,
            "phase": 0.7,
            "distortion": 0.0,
            "tone_bias": 0.0,
        },
        {
            "segment_key": "B",
            "length_mm": 240.0,
            "downhole_arrow": "up",
            "old_label": "R-15",
            "old_depth_top_m": 412.54,
            "old_depth_bottom_m": 412.78,
            "gap_end": "downhole",
            "gap_length_mm": 25.0,
            "note": "Arrow points uphole in the photograph.",
            "seed": 53,
            "phase": 0.72,
            "distortion": 0.0,
            "tone_bias": 0.02,
        },
        {
            "segment_key": "C",
            "length_mm": 198.0,
            "downhole_arrow": "down",
            "old_label": "R-15",  # duplicated legacy number
            # The legacy books place C directly under A, favouring the swap.
            "old_depth_top_m": 412.52,
            "old_depth_bottom_m": 412.718,
            "gap_end": "downhole",
            "gap_length_mm": 12.0,
            "note": "Duplicate legacy number R-15; folded rim disturbance.",
            "seed": 67,
            "phase": 2.95,
            "distortion": 2.6,
            "tone_bias": -0.24,
        },
    ]


def load_assembly_demo(store):
    existing = store.seeded_tiles(DEMO_ASSEMBLY_KEY)
    if existing:
        assembly_id = existing[0].get("assembly", {}).get("assembly_id")
        if assembly_id:
            return store.assembly(assembly_id), False

    specs = _segment_specs()
    files = {}
    documents = []
    for spec in specs:
        binary = _render_segment(
            spec["seed"], spec["phase"], spec["length_mm"], spec["downhole_arrow"],
            spec["old_label"], distortion=spec["distortion"], tone_bias=spec["tone_bias"],
        )

        class Upload:
            filename = f"demo-{spec['segment_key']}.png"

            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return self._payload

        files[f"photo-{spec['segment_key']}"] = Upload(binary)
        documents.append({key: value for key, value in spec.items() if key not in ("seed", "phase", "distortion", "tone_bias")})

    payload = {
        "name": "Demo reassembled run · R-14/R-15",
        "tray": "Shelf 7 · layer 3",
        "core_diameter_mm": 84.0,
        "origin_depth_m": 412.30,
        "shelf_length_mm": 760.0,
        "segments": documents,
        "orders": [
            {"label": "Field sheet 1 · lamina join", "segment_keys": ["A", "B", "C"]},
            {"label": "Field sheet 2 · legacy numbering", "segment_keys": ["A", "C", "B"]},
        ],
    }
    assembly = create_assembly(store, payload, files, synthetic=True)
    store.seed_tiles(DEMO_ASSEMBLY_KEY, [store.tile(assembly["specimen_id"])])
    return assembly, True
