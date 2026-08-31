#!/usr/bin/env python3
"""Parser-level schema tests for geometry-report v1 and placement-report v1."""

import json
import re as _re
import unittest
from pathlib import Path


def _load_schema():
    here = Path(__file__).resolve().parent
    schema_path = here / "geometry-schema-v1.json"
    with open(schema_path) as f:
        return json.load(f)


def _validate(data, schema):
    """Lightweight JSON Schema validation without jsonschema dependency."""
    def _check(val, sch, path):
        if sch.get("const") is not None:
            if val != sch["const"]:
                raise AssertionError(f"{path}: expected const {sch['const']!r}, got {val!r}")
        if sch.get("type") == "object":
            if not isinstance(val, dict):
                raise AssertionError(f"{path}: expected object, got {type(val).__name__}")
            for req in sch.get("required", []):
                if req not in val:
                    raise AssertionError(f"{path}: missing required key {req!r}")
            for k, vs in sch.get("properties", {}).items():
                if k in val:
                    _check(val[k], vs, f"{path}.{k}")
            extra = set(val.keys()) - set(sch.get("properties", {}).keys())
            if extra and not sch.get("additionalProperties"):
                raise AssertionError(f"{path}: unexpected keys {extra}")
            if "enum" in sch and val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")
        elif sch.get("type") == "array":
            if not isinstance(val, list):
                raise AssertionError(f"{path}: expected array, got {type(val).__name__}")
            if "items" in sch:
                for idx, item in enumerate(val):
                    _check(item, sch["items"], f"{path}[{idx}]")
            if "enum" in sch and val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")
        elif sch.get("type") == "string":
            if not isinstance(val, str):
                raise AssertionError(f"{path}: expected string, got {type(val).__name__}")
            if "pattern" in sch:
                if not _re.match(sch["pattern"], val):
                    raise AssertionError(f"{path}: pattern mismatch {sch['pattern']!r}")
            if "enum" in sch and val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")
        elif sch.get("type") == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                raise AssertionError(f"{path}: expected integer, got {type(val).__name__} {val!r}")
            if "minimum" in sch and val < sch["minimum"]:
                raise AssertionError(f"{path}: {val} < minimum {sch['minimum']}")
            if "enum" in sch and val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")
        elif sch.get("type") == "boolean":
            if not isinstance(val, bool):
                raise AssertionError(f"{path}: expected bool, got {type(val).__name__}")
            if "enum" in sch and val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")
        elif "enum" in sch:
            if val not in sch["enum"]:
                raise AssertionError(f"{path}: {val!r} not in {sch['enum']}")

    _check(data, schema, "$")
    return True


# ---- Fixtures ----

SEPARATE_FIXTURE = {
    "format": 1,
    "model_name": "test-model-separate",
    "model_sha256": "a" * 64,
    "expert_count": 256,
    "top_k": 8,
    "n_layers": 1,
    "all_repeated_moe_blocks_equal": True,
    "varying_fields": {},
    "layers": [
        {
            "layer": 0,
            "expert_count": 256,
            "top_k": 8,
            "layout": "separate",
            "gate_type": "Q4_K",
            "gate_shape": [2048, 2048, 256],
            "gate_bytes": 576 * 1024,
            "gate_scale_bytes": 2048,
            "up_type": "Q4_K",
            "up_shape": [2048, 2048, 256],
            "up_bytes": 576 * 1024,
            "up_scale_bytes": 2048,
            "down_type": "Q6_K",
            "down_shape": [2048, 2048, 256],
            "down_bytes": 800 * 1024,
            "down_scale_bytes": 4096,
            "complete_bundle_bytes": 576 * 1024 + 576 * 1024 + 800 * 1024 + 2048 + 2048 + 4096,
            "complete_bank_bytes": (576 * 1024 + 576 * 1024 + 800 * 1024 + 2048 + 2048 + 4096) * 256,
            "scales": {
                "gate_scale": {"type": "Q4_K", "bytes": 2048, "shape": [2048, 256]},
                "up_scale": {"type": "Q4_K", "bytes": 2048, "shape": [2048, 256]},
                "down_scale": {"type": "Q6_K", "bytes": 4096, "shape": [2048, 256]},
            },
        }
    ],
}

FUSED_FIXTURE = {
    "format": 1,
    "model_name": "test-model-fused",
    "model_sha256": "b" * 64,
    "expert_count": 256,
    "top_k": 8,
    "n_layers": 1,
    "all_repeated_moe_blocks_equal": True,
    "varying_fields": {},
    "layers": [
        {
            "layer": 0,
            "expert_count": 256,
            "top_k": 8,
            "layout": "fused",
            "gate_type": "Q4_K",
            "gate_shape": [2048, 4096, 256],
            "gate_bytes": 1152 * 1024,
            "gate_scale_bytes": 4096,
            "up_type": "Q4_K",
            "up_shape": [2048, 4096, 256],
            "up_bytes": 1152 * 1024,
            "up_scale_bytes": 4096,
            "down_type": "Q6_K",
            "down_shape": [2048, 2048, 256],
            "down_bytes": 800 * 1024,
            "down_scale_bytes": 4096,
            "complete_bundle_bytes": 1152 * 1024 + 800 * 1024 + 4096 + 4096,
            "complete_bank_bytes": (1152 * 1024 + 800 * 1024 + 4096 + 4096) * 256,
            "scales": {
                "gate_up_scale": {"type": "Q4_K", "bytes": 4096, "shape": [4096, 256]},
                "down_scale": {"type": "Q6_K", "bytes": 4096, "shape": [2048, 256]},
            },
        }
    ],
}


class TestGeometryReportSchema(unittest.TestCase):

    def test_separate_projection_fixture(self):
        row = SEPARATE_FIXTURE["layers"][0]
        expected = 576 * 1024 + 576 * 1024 + 800 * 1024 + 2048 + 2048 + 4096
        self.assertEqual(row["complete_bundle_bytes"], expected)
        self.assertTrue(SEPARATE_FIXTURE["all_repeated_moe_blocks_equal"])
        self.assertIn("down_scale", row["scales"])

    def test_fused_projection_fixture(self):
        row = FUSED_FIXTURE["layers"][0]
        expected = 1152 * 1024 + 800 * 1024 + 4096 + 4096
        self.assertEqual(row["complete_bundle_bytes"], expected)
        self.assertTrue(FUSED_FIXTURE["all_repeated_moe_blocks_equal"])
        self.assertIn("gate_up_scale", row["scales"])

    def test_separate_validates_against_schema(self):
        schema = _load_schema()
        self.assertTrue(_validate(SEPARATE_FIXTURE, schema))

    def test_fused_validates_against_schema(self):
        schema = _load_schema()
        self.assertTrue(_validate(FUSED_FIXTURE, schema))

    def test_format_constant(self):
        schema = _load_schema()
        copy = dict(SEPARATE_FIXTURE, format=2)
        with self.assertRaises(AssertionError):
            _validate(copy, schema)

    def test_sha256_pattern(self):
        schema = _load_schema()
        copy = dict(SEPARATE_FIXTURE, model_sha256="short")
        with self.assertRaises(AssertionError):
            _validate(copy, schema)

    def test_missing_required_field(self):
        schema = _load_schema()
        copy = {k: v for k, v in SEPARATE_FIXTURE.items() if k != "model_name"}
        with self.assertRaises(AssertionError):
            _validate(copy, schema)

    def test_invalid_layout(self):
        schema = _load_schema()
        copy = json.loads(json.dumps(SEPARATE_FIXTURE))
        copy["layers"][0]["layout"] = "mixed"
        with self.assertRaises(AssertionError):
            _validate(copy, schema)

    def test_zero_bundle_bytes(self):
        schema = _load_schema()
        copy = json.loads(json.dumps(SEPARATE_FIXTURE))
        copy["layers"][0]["complete_bundle_bytes"] = 0
        self.assertTrue(_validate(copy, schema))

    def test_negative_bundle_bytes(self):
        schema = _load_schema()
        copy = json.loads(json.dumps(SEPARATE_FIXTURE))
        copy["layers"][0]["complete_bundle_bytes"] = -1
        with self.assertRaises(AssertionError):
            _validate(copy, schema)

    def test_scale_bytes_included(self):
        row = SEPARATE_FIXTURE["layers"][0]
        weight_bytes = row["gate_bytes"] + row["up_bytes"] + row["down_bytes"]
        scale_bytes = row["gate_scale_bytes"] + row["up_scale_bytes"] + row["down_scale_bytes"]
        self.assertEqual(row["complete_bundle_bytes"], weight_bytes + scale_bytes)

    def test_fused_scale_bytes_included(self):
        row = FUSED_FIXTURE["layers"][0]
        weight_bytes = row["gate_bytes"] + row["down_bytes"]
        scale_bytes = row["gate_scale_bytes"] + row["down_scale_bytes"]
        self.assertEqual(row["complete_bundle_bytes"], weight_bytes + scale_bytes)

    def test_varying_fields_reports_differences(self):
        schema = _load_schema()
        unequal = json.loads(json.dumps(SEPARATE_FIXTURE))
        unequal["all_repeated_moe_blocks_equal"] = False
        unequal["varying_fields"] = {"gate_bytes": {"0": 576 * 1024, "1": 512 * 1024}}
        unequal["n_layers"] = 2
        unequal["layers"] = [
            SEPARATE_FIXTURE["layers"][0],
            dict(SEPARATE_FIXTURE["layers"][0], layer=1, gate_bytes=512 * 1024),
        ]
        self.assertTrue(_validate(unequal, schema))


# ---- Placement report schema tests ----

PLACEMENT_SCHEMA = {
    "type": "object",
    "required": [
        "format",
        "model_name",
        "model_sha256",
        "expert_count",
        "top_k",
        "n_layers",
        "cache_mib",
        "layers",
    ],
    "additionalProperties": True,
    "properties": {
        "format": {"const": 1},
        "cache_mib": {"type": "integer", "minimum": 0},
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["layer", "placement", "host_moe_tensors"],
                "properties": {
                    "layer": {"type": "integer", "minimum": 0},
                    "placement": {"type": "string", "enum": ["gpu", "cpu", "split"]},
                    "host_moe_tensors": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

PLACEMENT_FIXTURE = {
    "format": 1,
    "model_name": "test-model-placement",
    "model_sha256": "c" * 64,
    "expert_count": 256,
    "top_k": 8,
    "n_layers": 3,
    "cache_mib": 128,
    "layers": [
        {"layer": 0, "placement": "gpu", "host_moe_tensors": []},
        {"layer": 1, "placement": "cpu", "host_moe_tensors": [
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
        ]},
        {"layer": 2, "placement": "split", "host_moe_tensors": [
            "blk.2.ffn_gate_exps.weight",
        ]},
    ],
}


class TestPlacementReportSchema(unittest.TestCase):

    def test_placement_fixture_validates(self):
        self.assertTrue(_validate(PLACEMENT_FIXTURE, PLACEMENT_SCHEMA))

    def test_invalid_placement_label(self):
        copy = json.loads(json.dumps(PLACEMENT_FIXTURE))
        copy["layers"][0]["placement"] = "quantum"
        with self.assertRaises(AssertionError):
            _validate(copy, PLACEMENT_SCHEMA)

    def test_gpu_host_moe_tensors_empty(self):
        self.assertEqual(PLACEMENT_FIXTURE["layers"][0]["host_moe_tensors"], [])
        self.assertEqual(PLACEMENT_FIXTURE["layers"][0]["placement"], "gpu")

    def test_cpu_host_moe_tensors_nonempty(self):
        self.assertTrue(len(PLACEMENT_FIXTURE["layers"][1]["host_moe_tensors"]) > 0)
        self.assertEqual(PLACEMENT_FIXTURE["layers"][1]["placement"], "cpu")


if __name__ == "__main__":
    unittest.main()
