"""Deterministic contract tests for the split LFM2-VL MMProj exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPORTER = ROOT / "tools" / "export_lfm2_vl_mmproj.py"
SOURCE = ROOT / "tests" / "fixtures" / "lfm2_vl_tiny" / "tensors.safetensors"
BUNDLE = ROOT / "tests" / "fixtures" / "lfm2_vl_mmproj_tiny"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"

_SPEC = importlib.util.spec_from_file_location("lfm2_vl_mmproj_exporter", EXPORTER)
assert _SPEC is not None and _SPEC.loader is not None
_EXPORTER_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXPORTER_MODULE)


def _run(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--input",
            str(SOURCE),
            "--model-config",
            str(BUNDLE / "source_model_config.json"),
            "--processor-config",
            str(BUNDLE / "processor_config.json"),
            "--output-dir",
            str(output),
            "--source-model",
            "LiquidAI/LFM2.5-VL-450M-tiny-random",
            "--source-revision",
            REVISION,
            "--source-prefix",
            "weights.",
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(path: Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def test_export_is_byte_identical_and_contains_only_canonical_mmproj(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _run(first)
    second_result = _run(second)
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    for name in ("mmproj.safetensors", "mmproj.json", "processor_config.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).read_bytes() == (BUNDLE / name).read_bytes()

    header = _header(first / "mmproj.safetensors")
    names = set(header) - {"__metadata__"}
    assert len(names) == 43
    assert all(
        name.startswith("model.vision_tower.vision_model.")
        or name.startswith("model.multi_modal_projector.")
        for name in names
    )
    assert not any("language_model" in name or name.startswith("stage.") for name in names)

    manifest = json.loads((first / "mmproj.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "candle-mmproj"
    assert manifest["version"] == 1
    assert manifest["tensor_namespace_version"] == 1
    assert manifest["tensor_count"] == len(manifest["tensor_inventory"]) == 43
    assert set(manifest["tensor_inventory"]) == names
    assert manifest["vision_layer_count"] == 2
    assert manifest["expected_text_layer_count"] == 2
    assert manifest["mmproj_safetensors_sha256"] == _sha256(
        first / "mmproj.safetensors"
    )
    assert manifest["processor_config_sha256"] == _sha256(
        first / "processor_config.json"
    )


def test_existing_bundle_is_rejected_before_any_output_is_changed(tmp_path: Path):
    output = tmp_path / "bundle"
    first = _run(output)
    assert first.returncode == 0, first.stderr
    before = {
        name: (output / name).read_bytes()
        for name in ("mmproj.safetensors", "mmproj.json", "processor_config.json")
    }
    refused = _run(output)
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
    assert all((output / name).read_bytes() == payload for name, payload in before.items())


def test_processor_model_mismatch_is_actionable(tmp_path: Path):
    bad_processor = tmp_path / "processor.json"
    processor = json.loads((BUNDLE / "processor_config.json").read_text(encoding="utf-8"))
    processor["encoder_patch_size"] = 4
    bad_processor.write_text(json.dumps(processor), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--input",
            str(SOURCE),
            "--model-config",
            str(BUNDLE / "source_model_config.json"),
            "--processor-config",
            str(bad_processor),
            "--output-dir",
            str(tmp_path / "bad"),
            "--source-model",
            "tiny",
            "--source-revision",
            REVISION,
            "--source-prefix",
            "weights.",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "processor patch size 4 does not match model 2" in result.stderr


def test_missing_container_prefix_does_not_silently_emit_empty_bundle(tmp_path: Path):
    result = _run(tmp_path / "bad", "--source-prefix", "wrong.")
    assert result.returncode != 0
    assert "contains no LFM2-VL vision tensors" in result.stderr


def test_config_derived_inventory_rejects_missing_unexpected_and_wrong_shapes():
    config = json.loads((BUNDLE / "source_model_config.json").read_text(encoding="utf-8"))
    expected = _EXPORTER_MODULE._expected_tensor_shapes(config)
    inventory = {
        name: {
            "dtype": "F32",
            "shape": shape.copy(),
            "nbytes": 4 * _product(shape),
        }
        for name, shape in expected.items()
    }
    _EXPORTER_MODULE._validate_inventory(inventory, expected)

    missing = {name: info.copy() for name, info in inventory.items()}
    missing.pop(next(iter(expected)))
    with pytest.raises(ValueError, match="missing="):
        _EXPORTER_MODULE._validate_inventory(missing, expected)

    unexpected = {name: info.copy() for name, info in inventory.items()}
    unexpected["model.multi_modal_projector.unexpected.weight"] = {
        "dtype": "F32",
        "shape": [1],
        "nbytes": 4,
    }
    with pytest.raises(ValueError, match="unexpected="):
        _EXPORTER_MODULE._validate_inventory(unexpected, expected)

    wrong_shape = {name: info.copy() for name, info in inventory.items()}
    first_name = next(iter(expected))
    wrong_shape[first_name] = {
        **wrong_shape[first_name],
        "shape": [expected[first_name][0] + 1, *expected[first_name][1:]],
    }
    with pytest.raises(ValueError, match="mismatches="):
        _EXPORTER_MODULE._validate_inventory(wrong_shape, expected)


@pytest.mark.parametrize(
    ("source_model", "source_revision"),
    [
        ("", REVISION),
        (" model", REVISION),
        ("model", "main"),
        ("model", REVISION.upper()),
        ("model", "0" * 39),
    ],
)
def test_provenance_requires_model_id_and_immutable_revision(
    source_model: str, source_revision: str
):
    with pytest.raises(ValueError):
        _EXPORTER_MODULE._validate_provenance(source_model, source_revision)


def _product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result
