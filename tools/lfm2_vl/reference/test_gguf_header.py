"""Tests for the bounded GGUF header inspector."""

from __future__ import annotations

import hashlib
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
    main,
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


def _unicode_header() -> bytes:
    metadata = [
        _string("general.name") + struct.pack("<I", 8) + _string("token Ċ")
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    header += b"".join(metadata)
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


def test_full_file_mode_hashes_only_the_payload_free_header(tmp_path: Path):
    header = _tiny_header()
    payload_bytes = 96 + 136
    path = tmp_path / "complete.gguf"
    path.write_bytes(header + bytes(payload_bytes))

    result = inspect_gguf_header(path, full_file=True)

    assert result["prefix"] == {
        "bytes": len(header),
        "sha256": hashlib.sha256(header).hexdigest(),
        "contains_tensor_payload": False,
        "contains_complete_file": False,
    }
    assert result["file"] == {
        "bytes": len(header) + payload_bytes,
        "all_tensor_sizes_known": True,
        "matches_declared_file_size": True,
    }
    assert summarize_gguf_header(result)["file"] == result["file"]


def test_cli_stdout_escapes_unicode_for_windows_code_pages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "unicode.gguf"
    path.write_bytes(_unicode_header())

    assert main([str(path)]) == 0

    assert "token \\u010a" in capsys.readouterr().out


def test_cli_quiet_retains_utf8_report_without_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "unicode.gguf"
    output = tmp_path / "report.json"
    path.write_bytes(_unicode_header())

    assert main([str(path), "--output", str(output), "--quiet"]) == 0

    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["gguf"]["metadata"]["general.name"] == "token Ċ"


def test_cli_output_requires_explicit_overwrite(tmp_path: Path):
    path = tmp_path / "unicode.gguf"
    output = tmp_path / "report.json"
    path.write_bytes(_unicode_header())

    assert main([str(path), "--output", str(output), "--quiet"]) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="GGUF report output already exists"):
        main([str(path), "--output", str(output), "--quiet"])
    assert output.read_bytes() == original
    assert (
        main(
            [
                str(path),
                "--output",
                str(output),
                "--overwrite",
                "--quiet",
            ]
        )
        == 0
    )


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
    assert [sum(item["dtype_counts"].values()) for item in source["files"]] == [
        201,
        201,
        148,
    ]


def test_locked_official_text_gguf_is_exact_and_distinct_from_mmproj():
    lock = json.loads((HERE.parent / "reference-lock.json").read_text(encoding="utf-8"))
    source = next(
        item
        for item in lock["model_repositories"]
        if item["id"] == "LiquidAI/LFM2.5-VL-450M-GGUF"
    )
    contract = source["text_gguf_header_contract"]
    text_file = next(
        item for item in source["files"] if item["path"] == "LFM2.5-VL-450M-Q4_0.gguf"
    )

    assert contract["ranges_read"] == ["bytes=0-2388127"]
    assert contract["header_bytes"] == contract["tensor_data_offset"] == 2_388_128
    assert contract["tensor_payload_bytes_read"] == 0
    assert contract["physical_size_matches_declared_extent"] is True
    assert contract["physical_file_bytes"] == contract["declared_file_bytes"] == 219_311_264
    assert contract["tensor_records"] == 148
    assert contract["dtype_counts"] == {"F32": 55, "Q4_0": 92, "Q6_K": 1}
    assert sum(contract["dtype_counts"].values()) == contract["tensor_records"]
    assert contract["metadata"]["general.architecture"] == "lfm2"
    assert contract["metadata"]["lfm2.embedding_length"] == 1024
    assert contract["metadata"]["lfm2.feed_forward_length"] == 4608
    assert contract["metadata"]["tokenizer.ggml.tokens.count"] == 65536
    assert text_file["sha256"] == "6d2757dd0f0b98aea7dc90477bb5b3a0df1089be85ef92943f8cecb05121ccbf"
    assert text_file["header_prefix_sha256"] == contract["header_prefix_sha256"]
    assert text_file["declared_file_bytes"] == contract["declared_file_bytes"]
