"""Tests for the core segment assembly workflow.

Run with: python -m unittest discover -s tests
"""

import io
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from coresection.analysis import evaluate
from coresection.assembly import (
    create_assembly,
    locate_point,
    override_decision,
    parse_orders,
    parse_segment_document,
)
from coresection.assembly_demo import load_assembly_demo
from coresection.model import InputError
from coresection.store import Store


def make_face(seed, phase, tone_bias=0.0, distortion=0.0):
    generator = np.random.default_rng(seed)
    height, width = 200, 200
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    fold = 1.0 + distortion * np.exp(-((yy - 6.0) / 26.0) ** 2)
    beds = np.sin(2 * np.pi * (yy * fold + 9.0 * np.sin(0.02 * xx + phase)) / 15.0)
    noise = generator.normal(size=(height, width)) * 0.14 + beds * 0.6 + tone_bias
    rgb = np.repeat(np.clip(0.45 + noise[..., None] * 0.22, 0, 1), 3, axis=2)
    image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class Upload:
    def __init__(self, payload, name):
        self.filename = name
        self._payload = payload

    def read(self):
        return self._payload


def build_payload(**overrides):
    segments = [
        {
            "segment_key": "A", "length_mm": 100.0, "downhole_arrow": "down",
            "old_label": "X-1", "old_depth_top_m": 100.0, "old_depth_bottom_m": 100.1,
            "gap_end": "uphole", "gap_length_mm": 10.0,
        },
        {
            "segment_key": "B", "length_mm": 120.0, "downhole_arrow": "up",
            "old_label": "X-2", "old_depth_top_m": 100.12, "old_depth_bottom_m": 100.24,
            "gap_end": "downhole", "gap_length_mm": 8.0,
        },
        {
            "segment_key": "C", "length_mm": 90.0, "downhole_arrow": "down",
            "old_label": "X-2",  # duplicate old number
            "old_depth_top_m": 100.11, "old_depth_bottom_m": 100.2,
            "gap_end": None, "gap_length_mm": None,
        },
    ]
    payload = {
        "name": "Unit assembly",
        "tray": "Tray Z",
        "core_diameter_mm": 80.0,
        "origin_depth_m": 100.0,
        "shelf_length_mm": 400.0,
        "segments": segments,
        "orders": [
            {"label": "Sheet one", "segment_keys": ["A", "B", "C"]},
            {"label": "Sheet two", "segment_keys": ["A", "C", "B"]},
        ],
    }
    payload.update(overrides)
    return payload


def build_files():
    return {
        "photo-A": Upload(make_face(1, 0.7), "a.png"),
        "photo-B": Upload(make_face(2, 0.72, 0.02), "b.png"),
        "photo-C": Upload(make_face(3, 2.9, -0.22, 2.4), "c.png"),
    }


class AssemblyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, **overrides):
        return create_assembly(self.store, build_payload(**overrides), build_files())

    def test_full_assembly_archives_one_order_and_records_defeat(self):
        assembly = self.create()
        self.assertEqual(len(assembly["field_orders"]), 2)
        self.assertEqual(assembly["duplicate_old_labels"], [["B", "C", "X-2"]])
        adopted = next(
            order for order in assembly["field_orders"]
            if order["order_id"] == assembly["decision"]["order_id"]
        )
        self.assertEqual(adopted["segment_keys"][0], "A")  # first segment fixed
        self.assertIn(assembly["decision"]["defeated_order_id"], {"order-1", "order-2"})
        self.assertNotEqual(
            assembly["decision"]["order_id"], assembly["decision"]["defeated_order_id"],
        )
        self.assertTrue(assembly["decision"]["rationale"])
        self.assertTrue(assembly["decision"]["defeated_reason"])
        tile = self.store.tile(assembly["specimen_id"])
        self.assertEqual(tile["assembly"]["assembly_id"], assembly["id"])

    def test_gaps_become_transparent_blocks_with_known_length(self):
        assembly = self.create()
        gaps = [block for block in assembly["source_map"]["blocks"] if block["kind"] == "gap"]
        self.assertEqual({gap["length_mm"] for gap in gaps}, {10.0, 8.0})
        for gap in gaps:
            self.assertGreater(gap["row_end"], gap["row_start"])
        tile = self.store.tile(assembly["specimen_id"])
        # Stitched image keeps an alpha channel; gap rows are fully transparent.
        from coresection.geometry import specimen_frame

        frame = specimen_frame(self.store.blobs, tile)
        gap_rows = sorted(
            row for block in assembly["source_map"]["blocks"]
            if block["kind"] == "gap"
            for row in range(block["row_start"], block["row_end"])
        )
        self.assertFalse(frame.valid[gap_rows, :].any())

    def test_line_scan_gap_stations_have_no_values_but_keep_depth(self):
        assembly = self.create()
        tile_id = assembly["specimen_id"]
        result = evaluate(self.store, tile_id, {
            "kind": "profile",
            "options": {
                "start": [0.5, 0.02], "end": [0.5, 0.98],
                "samples": 100, "band_pixels": 2,
            },
            "apply_neutral": False,
        })
        rows = result["metrics"]["rows"]
        gap_rows = [row for row in rows if row["samples"] == 0]
        self.assertTrue(gap_rows)
        for row in gap_rows:
            self.assertIsNone(row["lightness"])
            self.assertIsNone(row["a"])
            self.assertIsNone(row["b"])
            self.assertIsNone(row["source"])
            self.assertIsNotNone(row["depth_offset_mm"])
        sourced = [row for row in rows if row["source"]]
        self.assertTrue(sourced)
        # Every measured station is sourced; nothing borrowed into a gap.
        self.assertEqual(
            {row["samples"] > 0 for row in rows}, {True, False},
        )
        self.assertTrue(all(row["samples"] > 0 for row in sourced))
        for row in sourced:
            self.assertIn(row["source"]["segment_key"], {"A", "B", "C"})
            self.assertTrue(row["source"]["photo_url"].startswith("/api/assemblies/"))
            self.assertEqual(len(row["source"]["position_in_original"]), 2)
            self.assertIsNotNone(row["depth_m"])

    def test_rotated_arrow_segment_maps_back_flipped(self):
        assembly = self.create()
        source_map = assembly["source_map"]
        b_block = next(
            block for block in assembly["source_map"]["blocks"]
            if block["kind"] == "segment" and block["segment_key"] == "B"
        )
        self.assertTrue(b_block["rotated"])  # arrow was 'up'
        y_norm = (b_block["row_start"] + 2) / source_map["plane_size"][1]
        location = locate_point(source_map, 0.5, y_norm)
        self.assertEqual(location["kind"], "segment")
        self.assertEqual(location["segment_key"], "B")
        # Near the top of the stitched B block corresponds to the rotated bottom.
        self.assertGreater(location["position_in_original"][1], 0.9)

    def test_override_switches_archive_order_and_keeps_reason(self):
        assembly = self.create()
        original = assembly["decision"]["order_id"]
        other = next(
            order["order_id"] for order in assembly["field_orders"]
            if order["order_id"] != original
        )
        updated = override_decision(self.store, assembly["id"], {
            "version": assembly["version"], "order_id": other, "note": "Curator override.",
        })
        self.assertEqual(updated["decision"]["order_id"], other)
        self.assertEqual(updated["decision"]["basis"], "manual_override")
        self.assertEqual(updated["decision"]["manual_note"], "Curator override.")
        self.assertEqual(updated["decision"]["defeated_order_id"], original)

    def test_validation_rejects_bad_permutation_and_oversized_string(self):
        with self.assertRaises(InputError):
            parse_orders(
                {"orders": [
                    {"label": "s1", "segment_keys": ["B", "A", "C"]},
                    {"label": "s2", "segment_keys": ["A", "C", "B"]},
                ]},
                "A",
            )
        with self.assertRaises(InputError):
            self.create(shelf_length_mm=10.0)

    def test_segment_parser_requires_gap_length(self):
        with self.assertRaises(InputError):
            parse_segment_document({
                "segment_key": "A", "length_mm": 100, "gap_end": "downhole",
            }, 0)

    def test_demo_scenario_prefers_lamina_over_legacy_depth(self):
        assembly, created = load_assembly_demo(self.store)
        self.assertTrue(created)
        lamina_by_order = {item["order_id"]: item["lamina_score"] for item in assembly["evaluations"]}
        depth_by_order = {item["order_id"]: item["depth_score"] for item in assembly["evaluations"]}
        winner = assembly["decision"]["order_id"]
        loser = assembly["decision"]["defeated_order_id"]
        self.assertGreater(lamina_by_order[winner], lamina_by_order[loser])
        self.assertIn(
            "depth", assembly["decision"]["rationale"].lower() + assembly["decision"]["defeated_reason"].lower(),
        )
        self.assertEqual(assembly["duplicate_old_labels"][0][2], "R-15")

        # Persisted and re-openable with its specimen.
        again = self.store.assembly(assembly["id"])
        self.assertEqual(again["specimen_id"], assembly["specimen_id"])
        listed = self.store.list_assemblies()
        self.assertEqual(len(listed), 1)
        self.assertIn("orders", json.dumps(again))


if __name__ == "__main__":
    unittest.main()
