"""Tests for the bounded GGUF header inspector."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from inspect_gguf_header import (
    GgufHeaderError,
    inspect_gguf_header,
    summarize_gguf_header,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _tiny_header() -> bytes:
    metadata = [
        _string("general.alignment") + struct.pack("<II", 4, 32),
        _string("clip.projector_type") + struct.pack("<I", 8) + _string("lfm2"),
        _string("clip.vision.image_mean")
        + struct.pack("<IIQ", 9, 6, 3)
        + struct.pack("<fff", 0.5, 0.25, 0.125),
    ]
    tensors = [
        _string("v.patch_embd.weight")
        + struct.pack("<IQQQQIQ", 4, 2, 2, 3, 8, 1, 0),
        _string("mm.1.weight") + struct.pack("<IQQIQ", 2, 32, 4, 8, 96),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    header += b"".join(metadata) + b"".join(tensors)
    return header + bytes((-len(header)) % 32)


def test_inspects_metadata_and_reverses_physical_dimensions(tmp_path: Path):
    path = tmp_path / "header.gguf"
    path.write_bytes(_tiny_header())
    result = inspect_gguf_header(
        path,
        source_url="https://example.invalid/mmproj.gguf",
        source_revision="0123456789abcdef",
        byte_range="0-255",
    )
    gguf = result["gguf"]
    assert gguf["version"] == 3
    assert gguf["tensor_count"] == 2
    assert gguf["metadata"]["clip.projector_type"] == "lfm2"
    assert gguf["metadata"]["clip.vision.image_mean"] == [0.5, 0.25, 0.125]
    patch = gguf["tensors"]["v.patch_embd.weight"]
    assert patch["gguf_dimensions"] == [2, 2, 3, 8]
    assert patch["shape"] == [8, 3, 2, 2]
    assert patch["dtype"] == "F16"
    assert patch["nbytes"] == 192
    projector = gguf["tensors"]["mm.1.weight"]
    assert projector["shape"] == [4, 32]
    assert projector["dtype"] == "Q8_0"
    assert projector["nbytes"] == 136
    assert result["prefix"]["contains_tensor_payload"] is False
    assert result["prefix"]["contains_complete_file"] is False
    summary = summarize_gguf_header(result)
    assert summary["gguf"]["dtype_counts"] == {"F16": 1, "Q8_0": 1}
    assert len(summary["gguf"]["tensor_names_sha256"]) == 64


def test_rejects_a_truncated_header(tmp_path: Path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(_tiny_header()[:64])
    with pytest.raises(GgufHeaderError, match="prefix ended"):
        inspect_gguf_header(path)


def test_rejects_non_gguf_input(tmp_path: Path):
    path = tmp_path / "wrong.bin"
    path.write_bytes(b"nope")
    with pytest.raises(GgufHeaderError, match="magic"):
        inspect_gguf_header(path)


def test_locked_official_header_contract_is_payload_free_and_consistent():
    lock = json.loads((HERE.parent / "reference-lock.json").read_text(encoding="utf-8"))
    source = next(
        item
        for item in lock["model_repositories"]
        if item["id"] == "LiquidAI/LFM2.5-VL-450M-GGUF"
    )
    contract = source["gguf_header_contract"]
    assert source["revision"] == "166cd80bbe157dc86d65f964eb8cc6a2cede62ca"
    assert source["adaptation"]["tensor_payload_bytes_read_for_retained_evidence"] == 0
    assert contract["version"] == 3
    assert contract["metadata_records"] == 32
    assert contract["tensor_records"] == 201
    assert contract["header_end"] == 12708
    assert contract["tensor_data_offset"] == contract["header_bytes"] == 12736
    assert contract["ranges_read"] == ["bytes=0-12735"]
    assert contract["tensor_payload_bytes_read"] == 0
    assert (
        contract["tensor_names_sha256"]
        == "45e3f6cf0b51dc9f5e458b8af3375d368cc59daff70b79e2938c7490a94df828"
    )
    schema = contract["tensor_schema"]
    assert len(schema["fixed_names"]) + 12 * len(schema["per_layer_names"]) == 201
    assert schema["patch_shape"] == [768, 3, 16, 16]
    assert schema["projector_shapes"] == {
        "mm.1.weight": [2048, 3072],
        "mm.2.weight": [1024, 2048],
    }
    assert source["files"][0]["dtype_counts"] == {"F16": 75, "F32": 126}
    assert source["files"][1]["dtype_counts"] == {"F32": 127, "Q8_0": 74}
    assert [sum(item["dtype_counts"].values()) for item in source["files"]] == [201, 201]
