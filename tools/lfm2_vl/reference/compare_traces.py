"""Compare one pinned Transformers trace with one native Candle trace.

The comparator loads tensors one pair at a time from safetensors so a trace
comparison does not retain both bundles in memory. It validates bundle hashes
and inventories before comparing values, requires the shared input contract,
and writes only a small JSON report (never model weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .manifest import write_bytes_atomic
    from .tensor_dump import validate_bundle
except ImportError:  # pragma: no cover - direct script execution
    from manifest import write_bytes_atomic  # type: ignore
    from tensor_dump import validate_bundle  # type: ignore


DEFAULT_ATOL = 2.0e-4
DEFAULT_RTOL = 2.0e-4
VISION_PROJECTOR_MIN_COSINE = 0.99999
POSITION_MAX_ABS = 2.0e-5
PREFILL_MAX_ABS = 1.0e-3
INTEGER_TENSORS = {
    "input.decode_token_ids",
    "input.image_rgb_u8",
    "input.attention_mask",
    "input.input_ids",
    "input.pixel_attention_mask",
    "input.projector_crop_ranges",
    "input.spatial_shapes",
}
REQUIRED_TENSORS = {
    "input.decode_token_ids",
    "input.image_rgb_u8",
    "input.attention_mask",
    "input.input_ids",
    "input.pixel_attention_mask",
    "input.pixel_values",
    "input.projector_crop_ranges",
    "input.spatial_shapes",
    "stage.language.decode_logits",
    "stage.language.hidden_states",
    "stage.language.prefill_logits",
    "stage.multimodal.merged_embeddings",
    "stage.projector.activation",
    "stage.projector.input",
    "stage.projector.linear_1",
    "stage.projector.linear_2",
    "stage.projector.output",
    "stage.projector.pixel_unshuffle",
    "stage.text.embeddings",
    "stage.vision.embeddings_with_resized_position",
    "stage.vision.last_hidden_state",
    "stage.vision.patch_embedding",
    "stage.vision.post_layernorm",
    "stage.vision.resized_position_embedding",
}
HYBRID_TRACE_MODE = "hybrid-trace"
TRACE_MODES = {"native-trace", "production-trace", HYBRID_TRACE_MODE}
HYBRID_REQUIRED_TENSORS = {
    "input.decode_token_ids",
    "input.image_rgb_u8",
    "input.attention_mask",
    "input.input_ids",
    "input.pixel_attention_mask",
    "input.pixel_values",
    "input.projector_crop_ranges",
    "input.spatial_shapes",
    "stage.language.decode_logits",
    "stage.language.prefill_logits",
    "stage.projector.output",
}
HYBRID_PROJECTOR_MIN_COSINE = 0.9999
HYBRID_LOGIT_MAX_ABS = 2.0e-2
REQUIRED_SHARED_CONTRACT_FIELDS = {
    "source_image_sha256",
    "prompt",
    "max_new_tokens",
    "dtype",
    "device",
}
REQUIRED_NATIVE_SNAPSHOT_FILES = {
    "config.json",
    "processor_config.json",
    "tokenizer.json",
}


def _torch_and_safe_open():
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError(
            "trace comparison requires the pinned CPU torch and safetensors environment"
        ) from exc
    return torch, safe_open


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_paths(bundle: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ValueError(f"trace bundle is not a directory: {bundle}")
    tensor_name = manifest.get("tensor_file")
    metadata_name = manifest.get("metadata_file")
    if not isinstance(tensor_name, str) or not isinstance(metadata_name, str):
        raise ValueError(f"trace bundle has invalid file names: {bundle}")
    tensor_path = (bundle / tensor_name).resolve()
    metadata_path = (bundle / metadata_name).resolve()
    if tensor_path.parent != bundle or metadata_path.parent != bundle:
        raise ValueError(f"trace bundle contains a path outside its directory: {bundle}")
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"trace bundle is missing its tensor or metadata file: {bundle}")
    return tensor_path, metadata_path


def _validate_trace_manifest(manifest: Mapping[str, Any], bundle: Path) -> None:
    if manifest.get("format") != "lfm2-vl-reference-bundle":
        raise ValueError(f"trace bundle has an unsupported format: {bundle}")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"trace bundle has an unsupported schema version: {bundle}")
    if manifest.get("mode") not in TRACE_MODES:
        raise ValueError(f"trace bundle has an unsupported mode: {bundle}")
    if manifest.get("weights_serialized") is not False:
        raise ValueError(f"trace bundle must not serialize weights: {bundle}")
    inventory = manifest.get("tensor_inventory")
    tensor_count = manifest.get("tensor_count")
    if not isinstance(inventory, Mapping):
        raise ValueError(f"trace bundle has no tensor inventory: {bundle}")
    if not isinstance(tensor_count, int) or isinstance(tensor_count, bool):
        raise ValueError(f"trace bundle has an invalid tensor count: {bundle}")
    if tensor_count != len(inventory):
        raise ValueError(
            f"trace bundle tensor count does not match its inventory: {bundle}"
        )


def _validate_trace_metadata(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any], bundle: Path
) -> None:
    if metadata.get("schema_version") != manifest.get("schema_version"):
        raise ValueError(f"trace metadata/schema mismatch: {bundle}")
    if metadata.get("mode") != manifest.get("mode"):
        raise ValueError(f"trace metadata/mode mismatch: {bundle}")
    if metadata.get("weights_serialized") is not False:
        raise ValueError(f"trace metadata must not serialize weights: {bundle}")
    if metadata.get("cache_reset_exact") is not True:
        raise ValueError(f"trace metadata must prove exact cache reset: {bundle}")


def _source_image_sha256(metadata: Mapping[str, Any]) -> str | None:
    direct = metadata.get("source_image_sha256")
    if isinstance(direct, str):
        return direct
    images = metadata.get("image_files")
    if isinstance(images, list) and len(images) == 1 and isinstance(images[0], Mapping):
        value = images[0].get("sha256")
        if isinstance(value, str):
            return value
    return None


def _contract_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "processor_revision": metadata.get("processor_revision"),
        "source_image_sha256": _source_image_sha256(metadata),
        "prompt": metadata.get("prompt"),
        "max_new_tokens": metadata.get("max_new_tokens"),
        "dtype": metadata.get("dtype"),
        "device": metadata.get("device"),
    }


def _validate_shared_contract(contract: Mapping[str, Any], label: str) -> None:
    missing = sorted(
        field for field in REQUIRED_SHARED_CONTRACT_FIELDS if contract.get(field) is None
    )
    if missing:
        raise ValueError(f"{label} trace lacks required contract fields: {missing!r}")
    _sha256(contract["source_image_sha256"], f"{label} source image sha256")
    prompt = contract["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{label} trace prompt must be a non-empty string")
    max_new_tokens = contract["max_new_tokens"]
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
        or max_new_tokens > 32
    ):
        raise ValueError(f"{label} trace max_new_tokens must be between 1 and 32")
    if contract["dtype"] != "float32" or contract["device"] != "cpu":
        raise ValueError(f"{label} trace must use the CPU/float32 parity lane")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    digest = value.lower()
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a hexadecimal SHA-256 digest")
    return digest


def _byte_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _direct_filename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty filename")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{label} must be a direct filename")
    return value


def _oracle_artifact_files(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, str]], set[str], Mapping[str, Any]]:
    artifact = metadata.get("artifact_manifest")
    if not isinstance(artifact, Mapping):
        raise ValueError("oracle trace metadata has no artifact_manifest")
    if artifact.get("format") != "lfm2-vl-artifact-manifest":
        raise ValueError("oracle trace artifact manifest has an unsupported format")
    if artifact.get("schema_version") != 1:
        raise ValueError("oracle trace artifact manifest has an unsupported schema version")
    if artifact.get("model_id") != metadata.get("model_id"):
        raise ValueError("oracle trace model ID does not match its artifact manifest")
    if artifact.get("revision") != metadata.get("model_revision"):
        raise ValueError("oracle trace model revision does not match its artifact manifest")
    if artifact.get("weights_hashed_not_serialized") is not True:
        raise ValueError("oracle trace artifact manifest does not identify hashed weights")

    records = artifact.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("oracle trace artifact manifest has no files")
    files: dict[str, tuple[int, str]] = {}
    required = set(REQUIRED_NATIVE_SNAPSHOT_FILES)
    total_bytes = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"oracle artifact file {index} is not an object")
        name = _direct_filename(record.get("path"), f"oracle artifact file {index} path")
        if name in files:
            raise ValueError(f"oracle artifact manifest repeats filename {name!r}")
        if record.get("regular_file") is not True:
            raise ValueError(f"oracle artifact file {name!r} is not a regular file")
        byte_count = _byte_count(record.get("bytes"), f"oracle artifact file {name!r} bytes")
        digest = _sha256(record.get("sha256"), f"oracle artifact file {name!r} sha256")
        files[name] = (byte_count, digest)
        total_bytes += byte_count
        purpose = record.get("purpose")
        if purpose in {"safetensors index", "safetensors weight shard"}:
            required.add(name)

    if artifact.get("file_count") != len(files):
        raise ValueError("oracle artifact file count does not match its inventory")
    if artifact.get("total_bytes") != total_bytes:
        raise ValueError("oracle artifact byte count does not match its inventory")
    absent_required = sorted(required - files.keys())
    if absent_required:
        raise ValueError(
            f"oracle artifact manifest lacks native-required files: {absent_required!r}"
        )
    return files, required, artifact


def _native_artifact_files(metadata: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    records = metadata.get("model_inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("native trace metadata has no model_inputs evidence")
    files: dict[str, tuple[int, str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"native model input {index} is not an object")
        path = record.get("path")
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError(f"native model input {index} has an invalid path")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if not name:
            raise ValueError(f"native model input {index} has no filename")
        if name in files:
            raise ValueError(f"native model inputs repeat filename {name!r}")
        if record.get("kind") != "file":
            raise ValueError(f"native model input {name!r} is not a regular file")
        byte_count = _byte_count(record.get("bytes"), f"native model input {name!r} bytes")
        digest = _sha256(record.get("sha256"), f"native model input {name!r} sha256")
        files[name] = (byte_count, digest)
    return files


def _validate_artifact_identity(
    oracle_metadata: Mapping[str, Any], native_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    if oracle_metadata.get("artifact_manifest_reverified") is not True:
        raise ValueError("oracle trace did not reverify its model snapshot after inference")
    if native_metadata.get("model_inputs_reverified") is not True:
        raise ValueError("native trace did not reverify its model inputs after inference")
    oracle_files, required, artifact = _oracle_artifact_files(oracle_metadata)
    native_files = _native_artifact_files(native_metadata)
    unexpected = sorted(native_files.keys() - oracle_files.keys())
    missing = sorted(required - native_files.keys())
    mismatches = {
        name: {
            "oracle_bytes": oracle_files[name][0],
            "native_bytes": native_files[name][0],
            "oracle_sha256": oracle_files[name][1],
            "native_sha256": native_files[name][1],
        }
        for name in sorted(native_files.keys() & oracle_files.keys())
        if native_files[name] != oracle_files[name]
    }
    if unexpected or missing or mismatches:
        raise ValueError(
            "trace model artifact mismatch: "
            f"unexpected_native={unexpected!r}, missing_native={missing!r}, "
            f"content_mismatches={mismatches!r}"
        )
    return {
        "model_id": artifact["model_id"],
        "model_revision": artifact["revision"],
        "native_consumed_file_count": len(native_files),
        "required_files": sorted(required),
        "verified_files": sorted(native_files),
    }


def _tolerance(name: str) -> tuple[float, float]:
    if name in INTEGER_TENSORS:
        return 0.0, 0.0
    if name.startswith("input."):
        return 2.0e-5, 2.0e-5
    if name.startswith("stage.vision."):
        return 2.0e-4, 2.0e-4
    if name.startswith("stage.projector."):
        return 2.0e-4, 2.0e-4
    if name.startswith("stage.language."):
        return 4.0e-4, 4.0e-4
    return DEFAULT_ATOL, DEFAULT_RTOL


def _comparison_policy(name: str) -> tuple[str, float | None]:
    """Return the phase contract instead of applying one global allclose rule.

    CPU F32 reductions can differ at small-magnitude elements even when the
    vision/projector tensor is directionally identical. The written parity
    contract therefore keeps allclose where it passes and otherwise uses the
    cosine floor for vision/projector stages (except the structural
    pixel-unshuffle stage), an absolute bound for resized positions and
    prefill logits, and the tighter allclose rule for the remaining
    floating-point stages.
    """

    if name == "stage.vision.resized_position_embedding":
        return "max_abs", POSITION_MAX_ABS
    if name == "stage.projector.pixel_unshuffle":
        return "allclose", None
    if name.startswith("stage.vision.") or name.startswith("stage.projector."):
        return "cosine_or_allclose", VISION_PROJECTOR_MIN_COSINE
    if name == "stage.language.hidden_states":
        return "cosine_or_allclose", VISION_PROJECTOR_MIN_COSINE
    if name == "stage.language.prefill_logits":
        return "max_abs", PREFILL_MAX_ABS
    return "allclose", None


def _compare_tensor(
    torch: Any,
    reference: Any,
    candidate: Any,
    name: str,
    *,
    hybrid: bool = False,
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "name": name,
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if reference.dtype != candidate.dtype:
        return {
            "name": name,
            "passed": False,
            "reason": "dtype_mismatch",
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
        }
    atol, rtol = _tolerance(name)
    if hybrid and name == "stage.projector.output":
        policy, policy_limit = "cosine_or_allclose", HYBRID_PROJECTOR_MIN_COSINE
    elif hybrid and name.startswith("stage.language."):
        policy, policy_limit = "max_abs", HYBRID_LOGIT_MAX_ABS
    else:
        policy, policy_limit = _comparison_policy(name)
    if name in INTEGER_TENSORS:
        exact = bool(torch.equal(reference, candidate))
        return {
            "name": name,
            "passed": exact,
            "kind": "exact",
            "max_abs": 0.0 if exact else None,
            "atol": 0.0,
            "rtol": 0.0,
        }
    reference_f32 = reference.to(dtype=torch.float32)
    candidate_f32 = candidate.to(dtype=torch.float32)
    if not bool(torch.isfinite(reference_f32).all()) or not bool(torch.isfinite(candidate_f32).all()):
        return {
            "name": name,
            "passed": False,
            "reason": "non_finite",
            "atol": atol,
            "rtol": rtol,
        }
    delta = (reference_f32 - candidate_f32).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    reference_scale = float(reference_f32.abs().max().item()) if reference_f32.numel() else 0.0
    allowed = atol + rtol * reference_scale
    reference_norm = float(torch.linalg.vector_norm(reference_f32).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate_f32).item())
    if reference_norm and candidate_norm:
        cosine = float(
            torch.dot(reference_f32.reshape(-1), candidate_f32.reshape(-1)).item()
            / (reference_norm * candidate_norm)
        )
    else:
        cosine = 1.0 if torch.equal(reference_f32, candidate_f32) else 0.0
    cosine = max(-1.0, min(1.0, cosine))
    allclose_passed = bool(torch.allclose(reference_f32, candidate_f32, atol=atol, rtol=rtol))
    if policy == "cosine_or_allclose":
        passed = allclose_passed or cosine >= policy_limit
    elif policy == "max_abs":
        passed = max_abs <= policy_limit
    else:
        passed = allclose_passed
    result = {
        "name": name,
        "passed": passed,
        "kind": policy,
        "max_abs": max_abs,
        "allowed_max_abs_at_reference_scale": allowed,
        "cosine": cosine,
        "atol": atol,
        "rtol": rtol,
    }
    if policy == "cosine_or_allclose":
        result["min_cosine"] = policy_limit
        result["allclose_passed"] = allclose_passed
        result["accepted_by"] = "allclose" if allclose_passed else "cosine"
    elif policy == "max_abs":
        result["allowed_max_abs"] = policy_limit
    return result


def compare_traces(oracle: Path, native: Path) -> dict[str, Any]:
    oracle = oracle.resolve()
    native = native.resolve()
    if oracle == native:
        raise ValueError("oracle and native trace paths must be different")
    oracle_manifest = validate_bundle(oracle, require_tensors=True)
    native_manifest = validate_bundle(native, require_tensors=True)
    oracle_manifest_path = oracle / "manifest.json"
    native_manifest_path = native / "manifest.json"
    oracle_manifest = _load_json(oracle_manifest_path)
    native_manifest = _load_json(native_manifest_path)
    _validate_trace_manifest(oracle_manifest, oracle)
    _validate_trace_manifest(native_manifest, native)
    oracle_tensor_path, oracle_metadata_path = _bundle_paths(oracle, oracle_manifest)
    native_tensor_path, native_metadata_path = _bundle_paths(native, native_manifest)
    oracle_metadata = _load_json(oracle_metadata_path)
    native_metadata = _load_json(native_metadata_path)
    _validate_trace_metadata(oracle_metadata, oracle_manifest, oracle)
    _validate_trace_metadata(native_metadata, native_manifest, native)

    oracle_names = set(oracle_manifest.get("tensor_inventory", {}))
    native_names = set(native_manifest.get("tensor_inventory", {}))
    missing_from_native = sorted(REQUIRED_TENSORS - native_names)
    missing_from_oracle = sorted(REQUIRED_TENSORS - oracle_names)
    if missing_from_native or missing_from_oracle:
        raise ValueError(
            "required trace tensors are missing: "
            f"native={missing_from_native!r}, oracle={missing_from_oracle!r}"
        )
    oracle_encoder_names = {
        name for name in oracle_names if name.startswith("stage.vision.encoder_layer.")
    }
    native_encoder_names = {
        name for name in native_names if name.startswith("stage.vision.encoder_layer.")
    }
    if oracle_encoder_names != native_encoder_names:
        raise ValueError(
            "vision encoder trace inventory mismatch: "
            f"oracle={sorted(oracle_encoder_names)!r}, native={sorted(native_encoder_names)!r}"
        )
    oracle_stage_names = {name for name in oracle_names if name.startswith("stage.")}
    native_stage_names = {name for name in native_names if name.startswith("stage.")}
    if oracle_stage_names != native_stage_names:
        raise ValueError(
            "trace stage inventory mismatch: "
            f"oracle_only={sorted(oracle_stage_names - native_stage_names)!r}, "
            f"native_only={sorted(native_stage_names - oracle_stage_names)!r}"
        )

    oracle_contract = _contract_fields(oracle_metadata)
    native_contract = _contract_fields(native_metadata)
    _validate_shared_contract(oracle_contract, "oracle")
    _validate_shared_contract(native_contract, "native")
    contract_mismatches = {
        key: {"oracle": oracle_contract[key], "native": native_contract[key]}
        for key in oracle_contract
        if oracle_contract[key] is not None
        and native_contract[key] is not None
        and oracle_contract[key] != native_contract[key]
    }
    if contract_mismatches:
        raise ValueError(f"trace input contract mismatch: {contract_mismatches}")
    artifact_identity = _validate_artifact_identity(oracle_metadata, native_metadata)

    torch, safe_open = _torch_and_safe_open()
    results: list[dict[str, Any]] = []
    names = sorted(REQUIRED_TENSORS | (oracle_names & native_names))
    with safe_open(str(oracle_tensor_path), framework="pt", device="cpu") as oracle_file:
        with safe_open(str(native_tensor_path), framework="pt", device="cpu") as native_file:
            for name in names:
                if name not in oracle_names or name not in native_names:
                    continue
                results.append(
                    _compare_tensor(
                        torch,
                        oracle_file.get_tensor(name),
                        native_file.get_tensor(name),
                        name,
                    )
                )
    failures = [item for item in results if not item["passed"]]
    return {
        "schema_version": 1,
        "format": "lfm2-vl-trace-comparison",
        "oracle": str(oracle),
        "native": str(native),
        "oracle_manifest_sha256": _sha256_file(oracle_manifest_path),
        "native_manifest_sha256": _sha256_file(native_manifest_path),
        "contract": oracle_contract,
        "artifact_identity": artifact_identity,
        "tensor_count_compared": len(results),
        "failure_count": len(failures),
        "passed": not failures,
        "tensors": results,
    }


def _hybrid_input_identity(metadata: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    records = metadata.get("model_inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("hybrid trace metadata has no consumed-file evidence")
    result: dict[str, tuple[int, str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"hybrid model input {index} is not an object")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"hybrid model input {index} has no path")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or name in result:
            raise ValueError(f"hybrid model inputs repeat filename {name!r}")
        result[name] = (
            _byte_count(record.get("bytes"), f"hybrid model input {name} bytes"),
            _sha256(record.get("sha256"), f"hybrid model input {name} sha256"),
        )
    return result


def _validate_hybrid_metadata(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any], label: str
) -> None:
    _validate_trace_manifest(manifest, Path(label))
    if metadata.get("mode") != HYBRID_TRACE_MODE:
        raise ValueError(f"{label} is not a hybrid trace")
    if metadata.get("model_inputs_reverified") is not True:
        raise ValueError(f"{label} did not reverify consumed GGUF inputs")
    if metadata.get("schema_version") != manifest.get("schema_version"):
        raise ValueError(f"{label} metadata/schema mismatch")
    if metadata.get("weights_serialized") is not False:
        raise ValueError(f"{label} serialized weights")
    if metadata.get("cache_reset_exact") is not True:
        raise ValueError(f"{label} did not prove exact cache reset")
    execution = metadata.get("execution_mode")
    if execution not in {"dense-dequantized", "q8_0-native"}:
        raise ValueError(f"{label} has an unsupported hybrid execution mode")
    q8_count = metadata.get("q8_tensor_count")
    if not isinstance(q8_count, int) or isinstance(q8_count, bool) or q8_count < 0:
        raise ValueError(f"{label} has an invalid Q8 tensor count")
    if execution == "q8_0-native" and q8_count <= 0:
        raise ValueError(f"{label} claims native Q8 execution without retained Q8 tensors")
    if execution == "dense-dequantized" and q8_count != 0:
        raise ValueError(f"{label} claims dense execution with retained Q8 tensors")


def compare_hybrid_traces(dense: Path, q8: Path) -> dict[str, Any]:
    """Compare two direct-GGUF receipts and require the candidate to be native Q8."""

    dense = dense.resolve()
    q8 = q8.resolve()
    if dense == q8:
        raise ValueError("dense and q8 hybrid trace paths must be different")
    validate_bundle(dense, require_tensors=True)
    validate_bundle(q8, require_tensors=True)
    dense_manifest = _load_json(dense / "manifest.json")
    q8_manifest = _load_json(q8 / "manifest.json")
    _validate_hybrid_metadata(_load_json(dense / "metadata.json"), dense_manifest, "dense hybrid trace")
    _validate_hybrid_metadata(_load_json(q8 / "metadata.json"), q8_manifest, "q8 hybrid trace")
    dense_metadata = _load_json(dense / "metadata.json")
    q8_metadata = _load_json(q8 / "metadata.json")
    if dense_metadata["execution_mode"] != "dense-dequantized":
        raise ValueError("dense comparison input did not resolve to dense-dequantized execution")
    if q8_metadata["execution_mode"] != "q8_0-native":
        raise ValueError("Q8 comparison input did not resolve to native Q8_0 execution")

    dense_contract = _contract_fields(dense_metadata)
    q8_contract = _contract_fields(q8_metadata)
    _validate_shared_contract(dense_contract, "dense hybrid")
    _validate_shared_contract(q8_contract, "q8 hybrid")
    mismatches = {
        key: {"dense": dense_contract[key], "q8": q8_contract[key]}
        for key in dense_contract
        if dense_contract[key] != q8_contract[key]
    }
    if mismatches:
        raise ValueError(f"hybrid input contract mismatch: {mismatches}")
    dense_inputs = _hybrid_input_identity(dense_metadata)
    q8_inputs = _hybrid_input_identity(q8_metadata)
    shared_names = sorted(set(dense_inputs) & set(q8_inputs))
    for name in shared_names:
        if dense_inputs[name] != q8_inputs[name]:
            raise ValueError(f"hybrid shared input changed between dense and Q8: {name}")
    dense_mmproj = {name for name in dense_inputs if "mmproj" in name.lower()}
    q8_mmproj = {name for name in q8_inputs if "mmproj" in name.lower()}
    if len(dense_mmproj) != 1 or len(q8_mmproj) != 1:
        raise ValueError("hybrid receipts must each contain exactly one direct MMProj input")
    if dense_inputs[next(iter(dense_mmproj))] == q8_inputs[next(iter(q8_mmproj))]:
        raise ValueError("dense and Q8 receipts unexpectedly use the same MMProj bytes")
    if dense_metadata.get("generation", {}).get("generated_ids") != q8_metadata.get("generation", {}).get("generated_ids"):
        raise ValueError("dense and Q8 generated token IDs differ")

    dense_names = set(dense_manifest.get("tensor_inventory", {}))
    q8_names = set(q8_manifest.get("tensor_inventory", {}))
    missing = sorted((HYBRID_REQUIRED_TENSORS - dense_names) | (HYBRID_REQUIRED_TENSORS - q8_names))
    if missing:
        raise ValueError(f"hybrid evidence is missing required tensors: {missing!r}")
    dense_tensor_path, _ = _bundle_paths(dense, dense_manifest)
    q8_tensor_path, _ = _bundle_paths(q8, q8_manifest)
    torch, safe_open = _torch_and_safe_open()
    results: list[dict[str, Any]] = []
    with safe_open(str(dense_tensor_path), framework="pt", device="cpu") as dense_file:
        with safe_open(str(q8_tensor_path), framework="pt", device="cpu") as q8_file:
            for name in sorted(HYBRID_REQUIRED_TENSORS):
                results.append(
                    _compare_tensor(
                        torch,
                        dense_file.get_tensor(name),
                        q8_file.get_tensor(name),
                        name,
                        hybrid=True,
                    )
                )
    failures = [item for item in results if not item["passed"]]
    return {
        "schema_version": 1,
        "format": "lfm2-vl-hybrid-comparison",
        "dense": str(dense),
        "q8": str(q8),
        "dense_execution": dense_metadata["execution_mode"],
        "q8_execution": q8_metadata["execution_mode"],
        "q8_tensor_count": q8_metadata["q8_tensor_count"],
        "contract": dense_contract,
        "shared_input_files": shared_names,
        "tensor_count_compared": len(results),
        "failure_count": len(failures),
        "passed": not failures,
        "tensors": results,
        "tolerances": {
            "projector_min_cosine": HYBRID_PROJECTOR_MIN_COSINE,
            "language_max_abs": HYBRID_LOGIT_MAX_ABS,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="pinned Transformers trace directory")
    parser.add_argument("--native", type=Path, help="native Candle trace directory")
    parser.add_argument("--dense", type=Path, help="direct-GGUF dense evidence directory")
    parser.add_argument("--q8", type=Path, help="direct-GGUF native-Q8 evidence directory")
    parser.add_argument("--output", type=Path, help="optional JSON report outside the repository")
    return parser


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.parent.resolve(strict=False) / path.name
    root = Path(__file__).resolve().parents[3]
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(f"comparison output must be outside the repository: {path}")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_atomic(
        path,
        payload,
        overwrite=False,
        label="comparison output",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dense is not None or args.q8 is not None:
            if args.dense is None or args.q8 is None or args.oracle is not None or args.native is not None:
                raise ValueError("hybrid comparison requires exactly --dense and --q8")
            report = compare_hybrid_traces(args.dense, args.q8)
        elif args.oracle is not None and args.native is not None:
            report = compare_traces(args.oracle, args.native)
        else:
            raise ValueError("comparison requires --oracle/--native or --dense/--q8")
        if args.output:
            _write_report(args.output, report)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"compare_traces: {exc}", file=sys.stderr)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["compare_hybrid_traces", "compare_traces", "main"]
