"""Focused contract tests for the LFM2-VL reference harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import export_fixtures
from inspect_config import inspect_config
from manifest import REFERENCE_PACKAGE_PINS, load_reference_lock, model_entry
from tensor_dump import validate_bundle


EXPORTER = HERE / "export_fixtures.py"


def _tiny_dependencies():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")


def _run_export(*args: str):
    return subprocess.run(
        [sys.executable, str(EXPORTER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_config_only_uses_lock_without_heavy_imports():
    summary_450 = inspect_config(model="450m")
    summary_16 = inspect_config(model="1.6b")
    assert summary_450["text"]["effective_ffn_size"] == 4608
    assert summary_16["text"]["effective_ffn_size"] == 8192
    assert summary_450["projector"]["input_size"] == 3072
    assert summary_16["projector"]["input_size"] == 4608
    assert summary_450["image_token_id"] == 396
    assert summary_450["model_revision"] != summary_16["model_revision"]


def test_config_only_rejects_non_json_weight_like_input(tmp_path: Path):
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="JSON files"):
        inspect_config(model="450m", config_path=weight_path)


def test_tiny_uses_official_classes_and_exports_required_stages():
    _tiny_dependencies()
    model = export_fixtures.build_official_tiny(seed=1234)
    inputs = export_fixtures.tiny_inputs()
    tensors, details = export_fixtures.run_official_tiny(model, inputs)
    required = {
        "stage.vision.patch_embedding",
        "stage.vision.embeddings_with_resized_position",
        "stage.vision.resized_position_embedding",
        "stage.vision.encoder_layer.0",
        "stage.vision.encoder_layer.1",
        "stage.vision.post_layernorm",
        "stage.projector.pixel_unshuffle",
        "stage.projector.layer_norm",
        "stage.projector.linear_1",
        "stage.projector.activation",
        "stage.projector.linear_2",
        "stage.projector.output",
        "stage.multimodal.merged_embeddings",
        "stage.language.prefill_logits",
        "stage.language.decode_logits",
    }
    assert required.issubset(tensors)
    assert export_fixtures.TINY_CONFIG["spatial_shape"] == [2, 4]
    assert int(inputs["pixel_attention_mask"].sum()) == 8
    assert int((inputs["input_ids"] == export_fixtures.TINY_CONFIG["image_token_id"]).sum()) == 2
    assert details["attention_and_short_convolution_present"] == {
        "attention": True,
        "short_convolution": True,
    }
    assert tensors["stage.projector.pixel_unshuffle"].shape[-1] == 64
    assert tensors["stage.language.decode_logits"].shape == (1, 3, 32)
    assert tensors["input.decode_token_ids"].shape == (1, 3)
    assert "weights.lm_head.weight" not in tensors
    assert details["omitted_tied_lm_head"] is True


def test_tiny_export_is_byte_identical_and_validates(tmp_path: Path):
    _tiny_dependencies()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _run_export(
        "--mode",
        "tiny-random",
        "--seed",
        "1234",
        "--output",
        str(first),
    )
    second_result = _run_export(
        "--mode",
        "tiny-random",
        "--seed",
        "1234",
        "--output",
        str(second),
    )
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    validate_bundle(first)
    for name in ("tensors.safetensors", "metadata.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_existing_output_requires_overwrite(tmp_path: Path):
    _tiny_dependencies()
    output = tmp_path / "existing"
    first = _run_export("--mode", "tiny-random", "--output", str(output))
    assert first.returncode == 0, first.stderr
    refused = _run_export("--mode", "tiny-random", "--output", str(output))
    assert refused.returncode == 2
    assert "--overwrite" in refused.stderr
    replaced = _run_export(
        "--mode",
        "tiny-random",
        "--output",
        str(output),
        "--overwrite",
    )
    assert replaced.returncode == 0, replaced.stderr


def test_production_requires_explicit_opt_in(tmp_path: Path):
    result = _run_export(
        "--mode",
        "production",
        "--output",
        str(tmp_path / "production"),
    )
    assert result.returncode == 2
    assert "--allow-production" in result.stderr


def test_production_loader_is_mockable_and_never_serializes_weights(tmp_path: Path, monkeypatch):
    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    calls = {}

    class FakeModel:
        config = SimpleNamespace(to_dict=lambda: entry["source_confirmed_config"])

        def eval(self):
            return self

    def fake_loader(model_id, revision, *, allow_download):
        calls.update(
            model_id=model_id,
            revision=revision,
            allow_download=allow_download,
        )
        return FakeModel()

    monkeypatch.setattr(export_fixtures, "load_production_model", fake_loader)
    args = export_fixtures.build_parser().parse_args(
        [
            "--mode",
            "production",
            "--model",
            "450m",
            "--allow-production",
            "--load-model",
            "--output",
            str(tmp_path / "production"),
        ]
    )
    result = export_fixtures._production_metadata(args)
    assert result["weights_loaded"] is True
    assert calls == {
        "model_id": entry["id"],
        "revision": entry["revision"],
        "allow_download": False,
    }
    manifest = validate_bundle(tmp_path / "production", require_tensors=False)
    assert manifest["tensor_payload_generated"] is False
    assert not (tmp_path / "production" / "tensors.safetensors").exists()


def test_manifest_hash_failure_is_detected(tmp_path: Path):
    _tiny_dependencies()
    output = tmp_path / "bundle"
    result = _run_export("--mode", "tiny-random", "--output", str(output))
    assert result.returncode == 0, result.stderr
    validate_bundle(output)
    tensor_path = output / "tensors.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_bundle(output)


def test_requirements_and_lock_pins_are_explicit():
    requirements = (HERE / "requirements-reference.in").read_text(encoding="utf-8")
    assert "torch==2.8.0+cpu" in requirements
    assert "torchvision==0.23.0+cpu" in requirements
    assert "fd12552d770f745fdbe41031ff4daa688f5ed57e" in requirements
    assert "safetensors==0.8.0" in requirements
    assert REFERENCE_PACKAGE_PINS["torch"] == "2.8.0+cpu"
    assert REFERENCE_PACKAGE_PINS["torchvision"] == "0.23.0+cpu"
    assert REFERENCE_PACKAGE_PINS["python"] == "3.10.12"
    resolved = (HERE / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "MANAGER-RESOLUTION-PENDING" not in resolved
    assert "tokenizers==0.22.2" in resolved
    assert "numpy==2.2.6" in resolved
