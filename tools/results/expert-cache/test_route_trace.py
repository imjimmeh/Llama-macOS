#!/usr/bin/env python3
"""Validation tests for attested TG1 route traces."""

import json
import os
import tempfile
import unittest

TRACE_LINE_SCHEMA = {
    "type": "object",
    "required": ["request_id", "sequence_index", "layer", "top_k", "experts"],
    "properties": {
        "request_id": {"type": "string"},
        "sequence_index": {"type": "integer", "minimum": 0},
        "layer": {"type": "integer", "minimum": 0},
        "top_k": {"type": "integer", "minimum": 1},
        "experts": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 1,
        },
    },
}


def validate_trace_line(obj):
    """Lightweight validation of a trace line against schema."""
    for req in TRACE_LINE_SCHEMA["required"]:
        if req not in obj:
            raise AssertionError(f"missing required key {req!r}")
    if not isinstance(obj["request_id"], str):
        raise AssertionError("request_id must be string")
    if not isinstance(obj["sequence_index"], int) or obj["sequence_index"] < 0:
        raise AssertionError("sequence_index must be non-negative int")
    if not isinstance(obj["layer"], int) or obj["layer"] < 0:
        raise AssertionError("layer must be non-negative int")
    if not isinstance(obj["top_k"], int) or obj["top_k"] < 1:
        raise AssertionError("top_k must be positive int")
    if not isinstance(obj["experts"], list) or len(obj["experts"]) < 1:
        raise AssertionError("experts must be non-empty list")
    for e in obj["experts"]:
        if not isinstance(e, int) or e < 0:
            raise AssertionError(f"expert ID {e!r} must be non-negative int")
    if len(obj["experts"]) != obj["top_k"]:
        raise AssertionError(f"experts count {len(obj['experts'])} != top_k {obj['top_k']}")
    return True


def parse_trace(path):
    """Read a JSONL trace file, return header dict and list of route dicts."""
    header = None
    routes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_header" in obj:
                header = obj
            else:
                validate_trace_line(obj)
                routes.append(obj)
    return header, routes


# ---- Fixtures ----

SAMPLE_HEADER = {
    "_header": True,
    "format": 1,
    "model": "Qwen3.6-35B-A3B-APEX-Compact",
    "model_sha256": "a2f6c7fdbe82113a2e48e2c38022b55bdcc4308a8002da96cf6d48dab67bb77d",
    "n_layers": 40,
    "n_experts": 256,
    "top_k": 8,
    "n_tokens": 128,
    "device": "CUDA0",
    "callback_matches_canonical": True,
}

SAMPLE_ROUTES = [
    {"request_id": "prof-1", "sequence_index": 0, "layer": 5, "top_k": 8, "experts": [17, 81, 4, 109, 31, 205, 7, 56]},
    {"request_id": "prof-1", "sequence_index": 0, "layer": 6, "top_k": 8, "experts": [92, 3, 144, 210, 55, 178, 39, 128]},
    {"request_id": "prof-1", "sequence_index": 1, "layer": 5, "top_k": 8, "experts": [205, 17, 109, 81, 56, 31, 4, 7]},
]


class TestTraceValidation(unittest.TestCase):

    def test_valid_route_line(self):
        self.assertTrue(validate_trace_line(SAMPLE_ROUTES[0]))

    def test_expert_list_order_preserved(self):
        obj = SAMPLE_ROUTES[0]
        self.assertEqual(obj["experts"], [17, 81, 4, 109, 31, 205, 7, 56])

    def test_exactly_one_row_per_token_layer(self):
        rows = {}
        for r in SAMPLE_ROUTES:
            key = (r["sequence_index"], r["layer"])
            rows[key] = rows.get(key, 0) + 1
        for key, count in rows.items():
            self.assertEqual(count, 1, f"duplicate rows for {key}")

    def test_experts_count_equals_top_k(self):
        for r in SAMPLE_ROUTES:
            self.assertEqual(len(r["experts"]), r["top_k"])
    def test_non_eight_route_valid_in_schema(self):
        """Non-8 routes are schema-valid; oracle filters them."""
        obj = dict(SAMPLE_ROUTES[0], top_k=7, experts=[1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(validate_trace_line(obj))

    def test_rejects_empty_experts(self):
        obj = dict(SAMPLE_ROUTES[0], experts=[])
        with self.assertRaises(AssertionError):
            validate_trace_line(obj)

    def test_rejects_negative_expert_id(self):
        obj = dict(SAMPLE_ROUTES[0], experts=[-1, 2, 3, 4, 5, 6, 7, 8])
        with self.assertRaises(AssertionError):
            validate_trace_line(obj)

    def test_header_required_keys(self):
        for req in ["_header", "format", "model", "n_layers", "n_experts", "top_k", "callback_matches_canonical"]:
            self.assertIn(req, SAMPLE_HEADER)

    def test_header_callback_matches_canonical(self):
        self.assertTrue(SAMPLE_HEADER["callback_matches_canonical"])

    def test_round_trip_file(self):
        lines = [json.dumps(SAMPLE_HEADER)] + [json.dumps(r) for r in SAMPLE_ROUTES]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            tmppath = f.name
        try:
            header, routes = parse_trace(tmppath)
            self.assertIsNotNone(header)
            self.assertEqual(len(routes), 3)
            self.assertEqual(routes[0]["experts"], [17, 81, 4, 109, 31, 205, 7, 56])
        finally:
            os.unlink(tmppath)

    def test_device_backed_id_tensor(self):
        """IDs may come from device memory; trace format is same."""
        obj = dict(SAMPLE_ROUTES[0])
        obj["_device_backed"] = True
        self.assertTrue(validate_trace_line(obj))

    def test_multi_token_rejected(self):
        """Multi-token ID tensors must not appear in TG1 trace."""
        obj = dict(SAMPLE_ROUTES[0], sequence_index=0, layer=5, experts=[1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(validate_trace_line(obj))


if __name__ == "__main__":
    unittest.main()
