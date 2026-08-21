"""Focused contract tests for the LFM2-VL reference harness."""

from __future__ import annotations

import json
import platform
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
import inspect_artifact
import manifest
import production_trace
import verify_environment
from compare_traces import (
    compare_hybrid_traces,
    REQUIRED_TENSORS,
    _validate_artifact_identity,
    compare_traces,
    main as compare_main,
)
from inspect_config import inspect_config, main as inspect_config_main
from manifest import (
    REFERENCE_PACKAGE_PINS,
    REFERENCE_PYTHON_PINS,
    load_reference_lock,
    model_entry,
    remote_code_admission,
    reference_package_pins,
)
from tensor_dump import validate_bundle, write_tensor_bundle


EXPORTER = HERE / "export_fixtures.py"


def _tiny_dependencies():
    try:
        manifest.require_reference_environment()
    except RuntimeError as exc:
        pytest.skip(str(exc))
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


def _synthetic_trace_tensors(torch, *, offset: float = 0.0):
    """Build a small complete trace contract without loading a model."""

    integer_shapes = {
        "input.decode_token_ids": (1, 3),
        "input.attention_mask": (1, 8),
        "input.input_ids": (1, 8),
        "input.pixel_attention_mask": (1, 4),
        "input.projector_crop_ranges": (1, 2),
        "input.spatial_shapes": (1, 2),
    }
    tensors = {}
    for name in REQUIRED_TENSORS | {"stage.vision.encoder_layer.0"}:
        if name in integer_shapes:
            tensors[name] = torch.zeros(integer_shapes[name], dtype=torch.int64)
        elif name == "input.image_rgb_u8":
            tensors[name] = torch.zeros((2, 2, 3), dtype=torch.uint8)
        elif name == "input.pixel_values":
            tensors[name] = torch.zeros((1, 4, 3), dtype=torch.float32)
        elif name == "stage.language.decode_logits":
            tensors[name] = torch.ones((1, 3, 4), dtype=torch.float32) + offset
        elif name == "stage.language.prefill_logits":
            tensors[name] = torch.ones((1, 8, 4), dtype=torch.float32) + offset
        elif name in {
            "stage.language.hidden_states",
            "stage.multimodal.merged_embeddings",
            "stage.text.embeddings",
        }:
            tensors[name] = torch.ones((1, 8, 6), dtype=torch.float32) + offset
        else:
            tensors[name] = torch.ones((1, 4, 6), dtype=torch.float32) + offset
    return tensors


def _synthetic_artifact_contract():
    records = [
        ("config.json", "Official model config", 11, "1" * 64),
        ("processor_config.json", "Official processor config", 12, "2" * 64),
        ("tokenizer.json", "Pinned tokenizer", 13, "3" * 64),
        ("model.safetensors", "safetensors weight shard", 14, "4" * 64),
    ]
    artifact = {
        "schema_version": 1,
        "format": "lfm2-vl-artifact-manifest",
        "model_id": "LiquidAI/LFM2.5-VL-450M",
        "revision": "test-revision",
        "files": [
            {
                "path": name,
                "purpose": purpose,
                "bytes": byte_count,
                "sha256": digest,
                "regular_file": True,
            }
            for name, purpose, byte_count, digest in records
        ],
        "file_count": len(records),
        "total_bytes": sum(record[2] for record in records),
        "weights_hashed_not_serialized": True,
    }
    model_inputs = [
        {
            "path": str(Path("C:/snapshot") / name),
            "kind": "file",
            "bytes": byte_count,
            "sha256": digest,
        }
        for name, _purpose, byte_count, digest in records
    ]
    return artifact, model_inputs


def _synthetic_hybrid_tensors(torch, *, offset: float = 0.0):
    tensors = {
        "input.decode_token_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "input.image_rgb_u8": torch.zeros((2, 2, 3), dtype=torch.uint8),
        "input.attention_mask": torch.ones((1, 8), dtype=torch.int64),
        "input.input_ids": torch.tensor([[1, 3, 3, 2, 4, 5, 6, 7]], dtype=torch.int64),
        "input.pixel_attention_mask": torch.ones((1, 4), dtype=torch.int64),
        "input.pixel_values": torch.ones((1, 4, 3), dtype=torch.float32),
        "input.projector_crop_ranges": torch.tensor([[0, 4]], dtype=torch.int64),
        "input.spatial_shapes": torch.tensor([[2, 2]], dtype=torch.int64),
        "stage.projector.output": torch.ones((4, 6), dtype=torch.float32) + offset,
        "stage.language.prefill_logits": torch.ones((1, 8, 4), dtype=torch.float32) + offset,
        "stage.language.decode_logits": torch.ones((1, 3, 4), dtype=torch.float32) + offset,
    }
    return tensors


def test_config_only_uses_lock_without_heavy_imports():
    summary_450 = inspect_config(model="450m")
    summary_16 = inspect_config(model="1.6b")
    assert summary_450["text"]["effective_ffn_size"] == 4608
    assert summary_16["text"]["effective_ffn_size"] == 8192
    assert summary_450["projector"]["input_size"] == 3072
    assert summary_16["projector"]["input_size"] == 4608
    assert summary_450["image_token_id"] == 396
    assert summary_450["model_revision"] != summary_16["model_revision"]


def test_3b_lock_alias_and_checkpoint_contract_are_explicit():
    summary = inspect_config(model="3b")
    assert summary["model_id"] == "LiquidAI/LFM2.5-VL-3B"
    assert summary["text"]["hidden_size"] == 2048
    assert summary["text"]["vocab_size"] == 128000
    assert summary["text"]["num_hidden_layers"] == 30
    assert summary["text"]["max_position_embeddings"] == 128000
    assert summary["text"]["eos_token_id"] == 124900
    assert summary["vision"]["hidden_size"] == 1152
    assert summary["vision"]["num_hidden_layers"] == 27
    assert summary["projector"]["input_size"] == 4608
    assert summary["image_token_id"] == 124907
    entry = model_entry(load_reference_lock(), "3b")
    assert entry["remote_code_policy"]["trust_remote_code"] is False
    assert entry["remote_code_policy"]["files"] == []
    assert entry["memory_bounds"]["max_input_tokens"] == 4096
    assert entry["safetensors_header"]["tensor_records"] == 707


def test_config_only_validates_local_3b_config_and_processor_contract(tmp_path: Path):
    entry = model_entry(load_reference_lock(), "3b")
    locked = entry["source_confirmed_config"]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": [locked["architecture"]],
                "model_type": locked["model_type"],
                "image_token_id": locked["image_token_id"],
                "downsample_factor": locked["downsample_factor"],
                "projector_hidden_size": locked["projector_hidden_size"],
                "projector_hidden_act": locked["projector_hidden_act"],
                "projector_bias": locked["projector_bias"],
                "projector_use_layernorm": locked["projector_use_layernorm"],
                "text_config": locked["text"],
                "vision_config": locked["vision"],
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    processor_path = tmp_path / "processor_config.json"
    processor_path.write_text(
        json.dumps({"image_processor": locked["processor"]}), encoding="utf-8"
    )
    summary = inspect_config(
        model="3b",
        config_path=config_path,
        processor_config_path=processor_path,
    )
    assert summary["locked_values_validated"] is True
    assert summary["processor"]["min_tiles"] == 1
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('"hidden_size": 2048', '"hidden_size": 1024'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the locked checkpoint"):
        inspect_config(model="3b", config_path=config_path)


def test_config_only_rejects_non_json_weight_like_input(tmp_path: Path):
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="JSON files"):
        inspect_config(model="450m", config_path=weight_path)


def test_config_only_reports_tokenizer_image_marker_ids(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "<image>": 396,
                        "<|image_start|>": 401,
                        "<|image_end|>": 402,
                    }
                },
                "added_tokens": [
                    {"content": "<|img_thumbnail|>", "id": 403},
                    {"content": "<|img_row_1_col_1|>", "id": 404},
                    {"content": "<|img_row_2_col_3|>", "id": 405},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = inspect_config(model="450m", tokenizer_path=tokenizer_path)
    markers = summary["image_marker_tokens"]
    assert markers["fixed"]["image"] == {"token": "<image>", "id": 396}
    assert markers["fixed"]["thumbnail"]["id"] == 403
    assert markers["row_column"] == [
        {"row": 1, "column": 1, "token": "<|img_row_1_col_1|>", "id": 404},
        {"row": 2, "column": 3, "token": "<|img_row_2_col_3|>", "id": 405},
    ]


def test_config_only_rejects_conflicting_or_wrong_image_marker_ids(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = {
        "model": {
            "vocab": {
                "<image>": 395,
                "<|image_start|>": 401,
                "<|image_end|>": 402,
                "<|img_thumbnail|>": 403,
            }
        },
        "added_tokens": [],
    }
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match model ID 396"):
        inspect_config(model="450m", tokenizer_path=tokenizer_path)

    tokenizer["model"]["vocab"]["<image>"] = 396
    tokenizer["added_tokens"] = [{"content": "<image>", "id": 397}]
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting IDs 396 and 397"):
        inspect_config(model="450m", tokenizer_path=tokenizer_path)


def test_config_only_rejects_missing_aliased_or_out_of_range_grid_markers(
    tmp_path: Path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = {
        "model": {
            "vocab": {
                "<image>": 396,
                "<|image_start|>": 401,
                "<|image_end|>": 402,
                "<|img_thumbnail|>": 403,
            }
        },
        "added_tokens": [],
    }
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required image row/column markers"):
        inspect_config(model="450m", tokenizer_path=tokenizer_path)

    tokenizer["added_tokens"] = [
        {"content": "<|img_row_1_col_1|>", "id": 403}
    ]
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(ValueError, match="share token ID 403"):
        inspect_config(model="450m", tokenizer_path=tokenizer_path)

    tokenizer["added_tokens"][0]["id"] = 65536
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(ValueError, match="outside model vocabulary size 65536"):
        inspect_config(model="450m", tokenizer_path=tokenizer_path)


def test_config_only_output_requires_explicit_overwrite(tmp_path: Path):
    output = tmp_path / "config-summary.json"
    assert inspect_config_main(["--model", "450m", "--output", str(output)]) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="JSON output already exists"):
        inspect_config_main(["--model", "450m", "--output", str(output)])
    assert output.read_bytes() == original
    assert (
        inspect_config_main(
            ["--model", "450m", "--output", str(output), "--overwrite"]
        )
        == 0
    )


def test_artifact_manifest_records_pinned_regular_files_without_loading_weights(tmp_path: Path):
    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = [item["path"] for item in entry["files"]]
    names.append(entry["safetensors_header"]["file"])
    for name in names:
        (snapshot / name).write_bytes(f"{name}\n".encode("utf-8"))

    artifact = inspect_artifact.build_artifact_manifest("450m", snapshot)
    assert artifact["model_id"] == entry["id"]
    assert artifact["revision"] == entry["revision"]
    assert artifact["file_count"] == len(names)
    assert all(item["regular_file"] for item in artifact["files"])
    assert all(len(item["sha256"]) == 64 for item in artifact["files"])
    output = tmp_path / "artifact.json"
    inspect_artifact.write_artifact_manifest(output, artifact, overwrite=False)
    assert output.is_file()


def test_artifact_manifest_rejects_unlocked_model_python_code(tmp_path: Path):
    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = [item["path"] for item in entry["files"]]
    names.append(entry["safetensors_header"]["file"])
    for name in names:
        (snapshot / name).write_bytes(f"{name}\n".encode("utf-8"))
    nested = snapshot / "nested"
    nested.mkdir()
    (nested / "modeling_lfm2_vl.py").write_text("# unpinned\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted Python code"):
        inspect_artifact.build_artifact_manifest("450m", snapshot)


def test_remote_code_admission_requires_exact_hash_bound_code_files():
    entry = {
        "id": "LiquidAI/LFM2.5-VL-3B",
        "revision": "r",
        "files": [
            {
                "path": "modeling_lfm2_vl.py",
                "purpose": "model-provided Python code",
                "bytes": 7,
                "sha256": "a" * 64,
            }
        ],
        "remote_code_policy": {
            "trust_remote_code": True,
            "files": ["modeling_lfm2_vl.py"],
        },
    }
    artifact = {
        "model_id": entry["id"],
        "revision": entry["revision"],
        "files": [
            {
                "path": "modeling_lfm2_vl.py",
                "purpose": "model-provided Python code",
                "bytes": 7,
                "sha256": "a" * 64,
                "regular_file": True,
            }
        ],
    }
    assert remote_code_admission(entry, artifact) is True
    artifact["files"].append(
        {
            "path": "cache_only.py",
            "bytes": 1,
            "sha256": "b" * 64,
            "regular_file": True,
        }
    )
    with pytest.raises(ValueError, match="unlisted model Python"):
        remote_code_admission(entry, artifact)


def test_artifact_manifest_no_clobber_is_atomic_against_a_racing_writer(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "artifact.json"
    real_link = manifest.os.link

    def racing_link(source, destination):
        Path(destination).write_text("owner manifest", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(manifest.os, "link", racing_link)

    with pytest.raises(FileExistsError, match="appeared during publication"):
        inspect_artifact.write_artifact_manifest(
            output,
            {"format": "test"},
            overwrite=False,
        )

    assert output.read_text(encoding="utf-8") == "owner manifest"
    assert list(tmp_path.glob(".artifact.json.tmp-*")) == []


def test_artifact_manifest_rejects_ambiguous_lock_and_index_names(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entry = {
        "files": [{"path": "config.json"}, {"path": "config.json"}],
        "safetensors_header": {"file": "model.safetensors"},
    }
    with pytest.raises(ValueError, match="duplicate path"):
        inspect_artifact._required_files(entry, snapshot)

    index_entry = {
        "files": [{"path": "config.json"}],
        "safetensors_header": {"file": "model.safetensors"},
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direct filename"):
        inspect_artifact._required_files(index_entry, snapshot)


def test_artifact_manifest_rejects_nul_and_invalid_index_tensor_names(tmp_path: Path):
    with pytest.raises(ValueError, match="NUL"):
        inspect_artifact._safe_filename("bad\x00name", "test filename")

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"": "shard.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid tensor name"):
        inspect_artifact._read_index_shards(snapshot, "model.safetensors.index.json")


def test_model_snapshot_resolution_rejects_repository_and_records_external_identity(
    tmp_path: Path,
):
    with pytest.raises(PermissionError, match="outside the repository"):
        inspect_artifact.resolve_model_snapshot("450m", ROOT)

    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = [item["path"] for item in entry["files"]]
    names.append(entry["safetensors_header"]["file"])
    for name in names:
        (snapshot / name).write_bytes(f"{name}\n".encode("utf-8"))
    resolved, artifact = inspect_artifact.resolve_model_snapshot("450m", snapshot)
    assert resolved == snapshot.resolve()
    assert artifact["revision"] == entry["revision"]
    assert artifact["weights_hashed_not_serialized"] is True


def test_production_trace_requires_external_snapshot_before_model_import(tmp_path: Path):
    args = export_fixtures.build_parser().parse_args(
        [
            "--mode",
            "production",
            "--trace",
            "--allow-production",
            "--load-model",
            "--output",
            str(tmp_path / "trace"),
            "--image",
            str(tmp_path / "image.png"),
            "--prompt",
            "Describe this image.",
        ]
    )
    with pytest.raises(ValueError, match="--model-dir"):
        export_fixtures._production_trace(args)


def test_trace_loaders_use_external_snapshot_and_never_download(tmp_path: Path, monkeypatch):
    calls = {}

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def set_num_threads(value):
            calls["threads"] = value

        @staticmethod
        def manual_seed(value):
            calls["seed"] = value

        @staticmethod
        def use_deterministic_algorithms(value):
            calls["deterministic"] = value

    class FakeModel:
        def to(self, **kwargs):
            calls["model_to"] = kwargs
            return self

        def eval(self):
            calls["model_eval"] = True
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["model"] = (source, kwargs)
            return FakeModel()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["processor"] = (source, kwargs)
            return object()

    monkeypatch.setattr(
        production_trace,
        "_torch_and_transformers",
        lambda: (FakeTorch, FakeAutoModel, FakeAutoProcessor),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    production_trace.load_trace_model(
        "LiquidAI/LFM2.5-VL-450M",
        "revision",
        allow_download=False,
        model_dir=snapshot,
    )
    production_trace.load_trace_processor(
        "LiquidAI/LFM2.5-VL-450M",
        "revision",
        allow_download=False,
        model_dir=snapshot,
    )
    assert calls["model"][0] == str(snapshot)
    assert calls["model"][1]["local_files_only"] is True
    assert calls["model"][1]["trust_remote_code"] is False
    assert calls["processor"][0] == str(snapshot)
    assert calls["processor"][1]["local_files_only"] is True
    assert calls["processor"][1]["trust_remote_code"] is False
    with pytest.raises(ValueError, match="external artifact manifest"):
        production_trace.load_trace_model(
            "LiquidAI/LFM2.5-VL-3B",
            "revision",
            allow_download=False,
            model_dir=snapshot,
            trust_remote_code=True,
        )
    with pytest.raises(ValueError, match="external artifact manifest"):
        production_trace.load_trace_processor(
            "LiquidAI/LFM2.5-VL-3B",
            "revision",
            allow_download=False,
            model_dir=snapshot,
            trust_remote_code=True,
        )
    with pytest.raises(ValueError, match="--allow-download"):
        production_trace.load_trace_model(
            "LiquidAI/LFM2.5-VL-450M",
            "revision",
            allow_download=True,
            model_dir=snapshot,
        )


def test_production_trace_rejects_snapshot_changes_after_inference():
    initial = {"revision": "one", "files": [{"sha256": "1" * 64}]}
    inspect_artifact.verify_artifact_unchanged(
        initial,
        dict(initial),
        operation="the trace",
    )
    changed = {"revision": "one", "files": [{"sha256": "2" * 64}]}
    with pytest.raises(RuntimeError, match="changed during the trace"):
        inspect_artifact.verify_artifact_unchanged(
            initial,
            changed,
            operation="the trace",
        )


def test_processor_inputs_executes_mapping_guard():
    class FakeTokenizer:
        @staticmethod
        def convert_ids_to_tokens(_token_id):
            return "<image>"

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        image_token = "<image>"

        @staticmethod
        def apply_chat_template(_conversation, **kwargs):
            return [] if kwargs.get("tokenize") else "<image>"

    with pytest.raises(ValueError, match="mapping-like BatchFeature"):
        production_trace._processor_inputs(
            FakeProcessor(),
            object(),
            "Describe this image.",
            torch=object(),
            image_token_id=396,
            max_input_tokens=4096,
            max_image_patches=1024,
        )


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


def test_trace_comparator_reports_exact_and_float_contract(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    artifact, model_inputs = _synthetic_artifact_contract()
    metadata = {
        "schema_version": 1,
        "mode": "production-trace",
        "model_id": "LiquidAI/LFM2.5-VL-450M",
        "model_revision": "test-revision",
        "processor_revision": "test-revision",
        "source_image_sha256": "a" * 64,
        "prompt": "test prompt",
        "max_new_tokens": 3,
        "dtype": "float32",
        "device": "cpu",
        "weights_serialized": False,
        "cache_reset_exact": True,
        "artifact_manifest": artifact,
        "artifact_manifest_reverified": True,
    }
    manifest = {
        "schema_version": 1,
        "mode": "production-trace",
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "processor_revision": metadata["processor_revision"],
        "source_image_sha256": metadata["source_image_sha256"],
        "max_new_tokens": metadata["max_new_tokens"],
        "weights_serialized": False,
    }
    oracle = tmp_path / "oracle"
    native = tmp_path / "native"
    write_tensor_bundle(
        oracle,
        _synthetic_trace_tensors(torch),
        metadata,
        manifest,
        overwrite=False,
    )
    write_tensor_bundle(
        native,
        _synthetic_trace_tensors(torch),
        {
            **metadata,
            "mode": "native-trace",
            "model_id": None,
            "model_revision": None,
            "processor_revision": None,
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    report = compare_traces(oracle, native)
    assert report["passed"] is True
    assert report["failure_count"] == 0
    assert report["tensor_count_compared"] == len(REQUIRED_TENSORS) + 1

    different = tmp_path / "different"
    write_tensor_bundle(
        different,
        _synthetic_trace_tensors(torch, offset=0.1),
        {
            **metadata,
            "mode": "native-trace",
            "model_id": None,
            "model_revision": None,
            "processor_revision": None,
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    report = compare_traces(oracle, different)
    assert report["passed"] is False
    assert report["failure_count"] > 0
    failed_output = tmp_path / "failed-comparison.json"
    assert (
        compare_main(
            [
                "--oracle",
                str(oracle),
                "--native",
                str(different),
                "--output",
                str(failed_output),
            ]
        )
        == 1
    )
    assert json.loads(failed_output.read_text(encoding="utf-8"))["passed"] is False
    owner_bytes = failed_output.read_bytes()
    assert (
        compare_main(
            [
                "--oracle",
                str(oracle),
                "--native",
                str(different),
                "--output",
                str(failed_output),
            ]
        )
        == 2
    )
    assert failed_output.read_bytes() == owner_bytes


def test_trace_comparator_rejects_contract_mismatch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    artifact, model_inputs = _synthetic_artifact_contract()
    tensors = _synthetic_trace_tensors(torch)
    base_metadata = {
        "schema_version": 1,
        "mode": "production-trace",
        "source_image_sha256": "a" * 64,
        "prompt": "test prompt",
        "max_new_tokens": 3,
        "dtype": "float32",
        "device": "cpu",
        "weights_serialized": False,
        "cache_reset_exact": True,
        "model_id": artifact["model_id"],
        "model_revision": artifact["revision"],
        "artifact_manifest": artifact,
        "artifact_manifest_reverified": True,
    }
    oracle = tmp_path / "oracle"
    native = tmp_path / "native"
    write_tensor_bundle(
        oracle,
        tensors,
        base_metadata,
        {"schema_version": 1, "mode": "production-trace", "weights_serialized": False},
        overwrite=False,
    )
    write_tensor_bundle(
        native,
        tensors,
        {
            **base_metadata,
            "mode": "native-trace",
            "source_image_sha256": "b" * 64,
            "artifact_manifest": None,
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_traces(oracle, native)

    weighted = tmp_path / "weighted"
    write_tensor_bundle(
        weighted,
        tensors,
        {
            **base_metadata,
            "mode": "native-trace",
            "weights_serialized": True,
            "artifact_manifest": None,
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    with pytest.raises(ValueError, match="must not serialize weights"):
        compare_traces(oracle, weighted)


def test_trace_comparator_uses_cpu_f32_phase_contract(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    artifact, model_inputs = _synthetic_artifact_contract()
    tensors = _synthetic_trace_tensors(torch)
    candidate = {name: value.clone() for name, value in tensors.items()}
    candidate["stage.vision.encoder_layer.0"].reshape(-1)[0] += 1.0e-3
    candidate["stage.projector.output"].reshape(-1)[0] += 1.0e-3
    candidate["stage.language.hidden_states"].reshape(-1)[0] += 1.0e-3
    candidate["stage.language.prefill_logits"].reshape(-1)[0] += 9.0e-4
    metadata = {
        "schema_version": 1,
        "mode": "production-trace",
        "source_image_sha256": "a" * 64,
        "prompt": "test prompt",
        "max_new_tokens": 3,
        "dtype": "float32",
        "device": "cpu",
        "weights_serialized": False,
        "cache_reset_exact": True,
        "model_id": artifact["model_id"],
        "model_revision": artifact["revision"],
        "artifact_manifest": artifact,
        "artifact_manifest_reverified": True,
    }
    oracle = tmp_path / "oracle"
    native = tmp_path / "native"
    write_tensor_bundle(
        oracle,
        tensors,
        metadata,
        {"schema_version": 1, "mode": "production-trace", "weights_serialized": False},
        overwrite=False,
    )
    write_tensor_bundle(
        native,
        candidate,
        {
            **metadata,
            "mode": "native-trace",
            "model_id": None,
            "model_revision": None,
            "processor_revision": None,
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    report = compare_traces(oracle, native)
    assert report["passed"] is True
    assert report["failure_count"] == 0
    by_name = {tensor["name"]: tensor for tensor in report["tensors"]}
    assert by_name["stage.vision.encoder_layer.0"]["kind"] == "cosine_or_allclose"
    assert by_name["stage.projector.output"]["kind"] == "cosine_or_allclose"
    assert by_name["stage.language.hidden_states"]["kind"] == "cosine_or_allclose"
    assert by_name["stage.language.prefill_logits"]["kind"] == "max_abs"
    assert by_name["stage.language.prefill_logits"]["allowed_max_abs"] == 1.0e-3


def test_hybrid_comparator_requires_direct_q8_and_shared_inputs(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    base_metadata = {
        "schema_version": 1,
        "mode": "hybrid-trace",
        "source_image_sha256": "a" * 64,
        "prompt": "rendered prompt",
        "max_new_tokens": 3,
        "dtype": "float32",
        "device": "cpu",
        "weights_serialized": False,
        "cache_reset_exact": True,
        "model_inputs_reverified": True,
        "generation": {"generated_ids": [1, 2, 3]},
    }
    shared_inputs = [
        {"path": "text.gguf", "bytes": 10, "sha256": "1" * 64, "kind": "file"},
        {"path": "tokenizer.json", "bytes": 11, "sha256": "2" * 64, "kind": "file"},
        {"path": "processor_config.json", "bytes": 12, "sha256": "3" * 64, "kind": "file"},
    ]
    dense = tmp_path / "dense"
    q8 = tmp_path / "q8"
    write_tensor_bundle(
        dense,
        _synthetic_hybrid_tensors(torch),
        {
            **base_metadata,
            "execution_mode": "dense-dequantized",
            "q8_tensor_count": 0,
            "model_inputs": [
                *shared_inputs,
                {"path": "mmproj-dense.gguf", "bytes": 13, "sha256": "4" * 64, "kind": "file"},
            ],
        },
        {"schema_version": 1, "mode": "hybrid-trace", "weights_serialized": False},
        overwrite=False,
    )
    write_tensor_bundle(
        q8,
        _synthetic_hybrid_tensors(torch, offset=1.0e-3),
        {
            **base_metadata,
            "execution_mode": "q8_0-native",
            "q8_tensor_count": 14,
            "model_inputs": [
                *shared_inputs,
                {"path": "mmproj-q8.gguf", "bytes": 14, "sha256": "5" * 64, "kind": "file"},
            ],
        },
        {"schema_version": 1, "mode": "hybrid-trace", "weights_serialized": False},
        overwrite=False,
    )
    report = compare_hybrid_traces(dense, q8)
    assert report["passed"] is True
    assert report["q8_execution"] == "q8_0-native"
    assert report["q8_tensor_count"] == 14

    bad_metadata = json.loads((q8 / "metadata.json").read_text(encoding="utf-8"))
    bad_metadata["q8_tensor_count"] = 0
    (q8 / "metadata.json").write_text(json.dumps(bad_metadata), encoding="utf-8")
    q8_manifest = json.loads((q8 / "manifest.json").read_text(encoding="utf-8"))
    q8_manifest["metadata_sha256"] = manifest.sha256_file(q8 / "metadata.json")
    (q8 / "manifest.json").write_text(json.dumps(q8_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="without retained Q8"):
        compare_hybrid_traces(dense, q8)


def test_trace_artifact_identity_requires_matching_native_inputs():
    artifact, model_inputs = _synthetic_artifact_contract()
    oracle = {
        "model_id": artifact["model_id"],
        "model_revision": artifact["revision"],
        "artifact_manifest": artifact,
        "artifact_manifest_reverified": True,
    }
    native = {"model_inputs": model_inputs, "model_inputs_reverified": True}
    summary = _validate_artifact_identity(oracle, native)
    assert summary["model_id"] == artifact["model_id"]
    assert summary["model_revision"] == artifact["revision"]
    assert summary["native_consumed_file_count"] == 4
    assert "model.safetensors" in summary["required_files"]

    changed = [dict(record) for record in model_inputs]
    changed[-1]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="artifact mismatch"):
        _validate_artifact_identity(
            oracle, {"model_inputs": changed, "model_inputs_reverified": True}
        )

    missing_weight = model_inputs[:-1]
    with pytest.raises(ValueError, match="missing_native"):
        _validate_artifact_identity(
            oracle, {"model_inputs": missing_weight, "model_inputs_reverified": True}
        )


def test_trace_comparator_rejects_optional_stage_inventory_drift(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    artifact, model_inputs = _synthetic_artifact_contract()
    base = {
        "schema_version": 1,
        "source_image_sha256": "a" * 64,
        "prompt": "rendered prompt",
        "max_new_tokens": 3,
        "dtype": "float32",
        "device": "cpu",
        "weights_serialized": False,
        "cache_reset_exact": True,
    }
    oracle_tensors = _synthetic_trace_tensors(torch)
    oracle_tensors["stage.projector.layer_norm"] = torch.ones(
        (1, 2, 2, 6), dtype=torch.float32
    )
    oracle = tmp_path / "oracle-stage"
    native = tmp_path / "native-stage"
    write_tensor_bundle(
        oracle,
        oracle_tensors,
        {
            **base,
            "mode": "production-trace",
            "model_id": artifact["model_id"],
            "model_revision": artifact["revision"],
            "artifact_manifest": artifact,
            "artifact_manifest_reverified": True,
        },
        {"schema_version": 1, "mode": "production-trace", "weights_serialized": False},
        overwrite=False,
    )
    write_tensor_bundle(
        native,
        _synthetic_trace_tensors(torch),
        {
            **base,
            "mode": "native-trace",
            "model_inputs": model_inputs,
            "model_inputs_reverified": True,
        },
        {"schema_version": 1, "mode": "native-trace", "weights_serialized": False},
        overwrite=False,
    )
    with pytest.raises(ValueError, match="stage inventory mismatch"):
        compare_traces(oracle, native)


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


def test_production_trace_requires_explicit_inputs_and_load(tmp_path: Path):
    output = tmp_path / "trace"
    args = export_fixtures.build_parser().parse_args(
        [
            "--mode",
            "production",
            "--trace",
            "--allow-production",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(PermissionError, match="--load-model"):
        export_fixtures._production_trace(args)

    args = export_fixtures.build_parser().parse_args(
        [
            "--mode",
            "production",
            "--trace",
            "--allow-production",
            "--load-model",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="--image"):
        export_fixtures._production_trace(args)


def test_trace_flag_is_rejected_outside_production(tmp_path: Path):
    result = _run_export(
        "--mode",
        "config-only",
        "--trace",
        "--output",
        str(tmp_path / "ignored"),
    )
    assert result.returncode == 2
    assert "requires --mode production" in result.stderr


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


def test_production_loader_reverifies_external_snapshot(tmp_path: Path, monkeypatch):
    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact = {
        "revision": entry["revision"],
        "files": [{"path": "model.safetensors", "sha256": "1" * 64}],
    }
    resolve_calls = []

    def fake_resolve(model, model_dir):
        resolve_calls.append((model, model_dir))
        return snapshot.resolve(), dict(artifact)

    class FakeModel:
        config = SimpleNamespace(to_dict=lambda: entry["source_confirmed_config"])

    def fake_loader(model_id, revision, *, allow_download, model_dir):
        assert model_id == entry["id"]
        assert revision == entry["revision"]
        assert allow_download is False
        assert model_dir == snapshot.resolve()
        return FakeModel()

    monkeypatch.setattr(export_fixtures, "resolve_model_snapshot", fake_resolve)
    monkeypatch.setattr(export_fixtures, "load_production_model", fake_loader)
    output = tmp_path / "production"
    args = export_fixtures.build_parser().parse_args(
        [
            "--mode",
            "production",
            "--model",
            "450m",
            "--model-dir",
            str(snapshot),
            "--allow-production",
            "--load-model",
            "--output",
            str(output),
        ]
    )
    export_fixtures._production_metadata(args)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_manifest_reverified"] is True
    assert metadata["environment_lock"]["platform"] == platform.system()
    assert len(resolve_calls) == 2


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


def test_bundle_manifest_rejects_path_escape(tmp_path: Path):
    output = tmp_path / "bundle"
    output.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "lfm2-vl-reference-metadata",
                "metadata_file": "../outside.json",
                "metadata_sha256": manifest.sha256_file(outside),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direct filename"):
        validate_bundle(output, require_tensors=False)


def test_bundle_manifest_rejects_non_object_json(tmp_path: Path):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "manifest.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest must be a JSON object"):
        validate_bundle(output, require_tensors=False)


def test_requirements_and_lock_pins_are_explicit():
    requirements = (HERE / "requirements-reference.in").read_text(encoding="utf-8")
    assert "torch==2.8.0+cpu" in requirements
    assert "torchvision==0.23.0+cpu" in requirements
    assert "fd12552d770f745fdbe41031ff4daa688f5ed57e" in requirements
    assert "safetensors==0.8.0" in requirements
    assert REFERENCE_PACKAGE_PINS["torch"] == "2.8.0+cpu"
    assert REFERENCE_PACKAGE_PINS["torchvision"] == "0.23.0+cpu"
    assert REFERENCE_PYTHON_PINS == {"Linux": "3.10.12", "Windows": "3.10.11"}
    assert reference_package_pins("Linux")["python"] == "3.10.12"
    assert reference_package_pins("Windows")["python"] == "3.10.11"
    with pytest.raises(ValueError, match="unsupported reference platform"):
        reference_package_pins("Plan9")


def test_reference_environment_guard_rejects_unpinned_runtime(monkeypatch):
    installed = {
        name: expected
        for name, expected in REFERENCE_PACKAGE_PINS.items()
        if name != "transformers"
    }
    installed["torch"] = "2.10.0.dev20250910+cu130"
    installed["transformers"] = "5.5.4"
    monkeypatch.setattr(manifest, "package_versions", lambda: installed)
    monkeypatch.setattr(manifest, "_installed_vcs_revision", lambda _name: "wrong-revision")

    mismatches = manifest.reference_environment_mismatches()
    assert mismatches["torch"]["expected"] == "2.8.0+cpu"
    assert mismatches["transformers"]["expected"] == "fd12552d770f745fdbe41031ff4daa688f5ed57e"
    assert "pytest" not in mismatches
    with pytest.raises(RuntimeError, match="pinned reference environment mismatch"):
        manifest.require_reference_environment()
    resolved = (HERE / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "MANAGER-RESOLUTION-PENDING" not in resolved
    assert "tokenizers==0.22.2" in resolved
    assert "numpy==2.2.6" in resolved
    windows_resolved = (HERE / "requirements-reference-windows.txt").read_text(
        encoding="utf-8"
    )
    assert "Python 3.10.11 / Windows x86_64" in windows_resolved
    assert "colorama==0.4.6" in windows_resolved
    assert "pip==23.0.1" in windows_resolved
    assert "torch==2.8.0+cpu" in windows_resolved
    assert "fd12552d770f745fdbe41031ff4daa688f5ed57e" in windows_resolved


def test_environment_report_keeps_pytest_test_only(monkeypatch):
    expected = reference_package_pins()
    installed = dict(expected)
    installed["pytest"] = "missing"
    monkeypatch.setattr(verify_environment, "package_versions", lambda: installed)
    monkeypatch.setattr(
        verify_environment,
        "reference_environment_mismatches",
        lambda _system_name: {},
    )

    runtime_report = verify_environment.environment_report()
    assert runtime_report["passed"] is True
    assert runtime_report["test_mismatches"] == {
        "pytest": {"expected": "8.4.1", "installed": "missing"}
    }
    test_report = verify_environment.environment_report(require_tests=True)
    assert test_report["passed"] is False


def test_environment_report_verifies_complete_platform_lock(monkeypatch):
    expected = reference_package_pins()
    monkeypatch.setattr(verify_environment, "package_versions", lambda: expected)
    monkeypatch.setattr(
        verify_environment,
        "reference_environment_mismatches",
        lambda _system_name: {},
    )
    lock_lines = verify_environment._lock_lines(manifest.reference_requirements_path())
    monkeypatch.setattr(verify_environment, "_pip_freeze_lines", lambda: lock_lines)

    report = verify_environment.environment_report(verify_lock=True)
    assert report["passed"] is True
    assert report["resolved_lock"]["distribution_count"] == len(lock_lines)
    assert len(report["resolved_lock"]["sha256"]) == 64
    assert report["resolved_lock"]["missing"] == []
    assert report["resolved_lock"]["unexpected"] == []

    monkeypatch.setattr(
        verify_environment,
        "_pip_freeze_lines",
        lambda: [*lock_lines, "unexpected-package==1.0"],
    )
    mismatch = verify_environment.environment_report(verify_lock=True)
    assert mismatch["passed"] is False
    assert mismatch["resolved_lock"]["unexpected"] == ["unexpected-package==1.0"]
