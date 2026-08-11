#!/usr/bin/env python3
"""Export a versioned dense LFM2-VL mmproj bundle without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO


MAX_HEADER_BYTES = 64 * 1024 * 1024
COPY_CHUNK_BYTES = 8 * 1024 * 1024
MAX_MMPROJ_TENSORS = 16_384
MAX_VISION_LAYERS = 512
VISION_PREFIX = "model.vision_tower."
CANONICAL_VISION_PREFIX = "model.vision_tower.vision_model."
PROJECTOR_PREFIX = "model.multi_modal_projector."
DTYPE_BYTES = {
    "BF16": 2,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}
DENSE_MMPROJ_DTYPES = {"BF16", "F16", "F32", "F64"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_temporary(
    temporary: Path, destination: Path, *, overwrite: bool
) -> None:
    """Publish one flushed temporary file without an unapproved replacement."""

    try:
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"output appeared during publication and was not replaced: {destination}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_temporary(temporary, path, overwrite=overwrite)


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value, raw


def _read_safetensors_header(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > MAX_HEADER_BYTES:
            raise ValueError(f"{path} has an invalid safetensors header length {header_size}")
        if header_size > file_size - 8:
            raise ValueError(f"{path} safetensors header exceeds the file length")
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"{path} safetensors metadata must contain only strings")

    data_start = 8 + header_size
    data_size = file_size - data_start
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, info in header.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ValueError(f"{path} has an invalid tensor header entry")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"{path} tensor {name!r} has no dtype")
        if not isinstance(shape, list) or not shape or not all(
            isinstance(value, int) and value > 0 for value in shape
        ):
            raise ValueError(f"{path} tensor {name!r} has an invalid shape")
        if dtype not in DTYPE_BYTES:
            raise ValueError(
                f"{path} tensor {name!r} has unsupported dense mmproj dtype {dtype!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise ValueError(f"{path} tensor {name!r} has invalid data offsets")
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        expected_bytes = element_count * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(
                f"{path} tensor {name!r} stores {offsets[1] - offsets[0]} bytes, "
                f"expected {expected_bytes} for {dtype}{shape}"
            )
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        ranges.append((offsets[0], offsets[1], name))
    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start != previous_end:
            relation = "overlaps another tensor" if start < previous_end else "leaves a payload gap"
            raise ValueError(f"{path} tensor {name!r} {relation}")
        previous_end = end
    if previous_end != data_size:
        raise ValueError(f"{path} has {data_size - previous_end} unclaimed payload bytes")
    return tensors, metadata, data_start, file_size


def _canonical_name(name: str, source_prefix: str) -> str | None:
    if source_prefix:
        if not name.startswith(source_prefix):
            return None
        name = name[len(source_prefix) :]
    if name.startswith(VISION_PREFIX):
        suffix = name[len(VISION_PREFIX) :]
        if suffix.startswith("vision_model."):
            return name
        return CANONICAL_VISION_PREFIX + suffix
    if name.startswith(PROJECTOR_PREFIX):
        return name
    return None


def _positive_int(values: dict[str, Any], name: str, default: int | None = None) -> int:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _boolean(values: dict[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _expected_tensor_shapes(model_config: dict[str, Any]) -> dict[str, list[int]]:
    text = model_config.get("text_config")
    vision = model_config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("model config must contain text_config and vision_config objects")

    text_hidden = _positive_int(text, "hidden_size")
    vision_hidden = _positive_int(vision, "hidden_size", 768)
    vision_intermediate = _positive_int(vision, "intermediate_size", 3072)
    vision_layers = _positive_int(vision, "num_hidden_layers", 12)
    if vision_layers > MAX_VISION_LAYERS:
        raise ValueError(
            f"vision num_hidden_layers {vision_layers} exceeds {MAX_VISION_LAYERS}"
        )
    num_channels = _positive_int(vision, "num_channels", 3)
    patch_size = _positive_int(vision, "patch_size", 16)
    num_patches = _positive_int(vision, "num_patches", 256)
    downsample_factor = _positive_int(model_config, "downsample_factor", 2)
    projector_hidden = _positive_int(model_config, "projector_hidden_size", 2560)
    projector_bias = _boolean(model_config, "projector_bias", True)
    projector_use_layernorm = model_config.get(
        "projector_use_layernorm",
        model_config.get("projector_use_layer_norm", True),
    )
    if not isinstance(projector_use_layernorm, bool):
        raise ValueError(
            f"invalid projector_use_layernorm: {projector_use_layernorm!r}"
        )

    patch_dimension = num_channels * patch_size * patch_size
    projector_input = vision_hidden * downsample_factor * downsample_factor
    shapes: dict[str, list[int]] = {
        f"{CANONICAL_VISION_PREFIX}embeddings.patch_embedding.weight": [
            vision_hidden,
            patch_dimension,
        ],
        f"{CANONICAL_VISION_PREFIX}embeddings.patch_embedding.bias": [vision_hidden],
        f"{CANONICAL_VISION_PREFIX}embeddings.position_embedding.weight": [
            num_patches,
            vision_hidden,
        ],
    }
    for layer in range(vision_layers):
        root = f"{CANONICAL_VISION_PREFIX}encoder.layers.{layer}"
        for norm in ("layer_norm1", "layer_norm2"):
            shapes[f"{root}.{norm}.weight"] = [vision_hidden]
            shapes[f"{root}.{norm}.bias"] = [vision_hidden]
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{root}.self_attn.{projection}.weight"] = [
                vision_hidden,
                vision_hidden,
            ]
            shapes[f"{root}.self_attn.{projection}.bias"] = [vision_hidden]
        shapes[f"{root}.mlp.fc1.weight"] = [vision_intermediate, vision_hidden]
        shapes[f"{root}.mlp.fc1.bias"] = [vision_intermediate]
        shapes[f"{root}.mlp.fc2.weight"] = [vision_hidden, vision_intermediate]
        shapes[f"{root}.mlp.fc2.bias"] = [vision_hidden]
    shapes[f"{CANONICAL_VISION_PREFIX}post_layernorm.weight"] = [vision_hidden]
    shapes[f"{CANONICAL_VISION_PREFIX}post_layernorm.bias"] = [vision_hidden]

    if projector_use_layernorm:
        shapes[f"{PROJECTOR_PREFIX}layer_norm.weight"] = [projector_input]
        shapes[f"{PROJECTOR_PREFIX}layer_norm.bias"] = [projector_input]
    shapes[f"{PROJECTOR_PREFIX}linear_1.weight"] = [
        projector_hidden,
        projector_input,
    ]
    shapes[f"{PROJECTOR_PREFIX}linear_2.weight"] = [text_hidden, projector_hidden]
    if projector_bias:
        shapes[f"{PROJECTOR_PREFIX}linear_1.bias"] = [projector_hidden]
        shapes[f"{PROJECTOR_PREFIX}linear_2.bias"] = [text_hidden]
    if not shapes or len(shapes) > MAX_MMPROJ_TENSORS:
        raise ValueError(
            f"config-derived mmproj tensor count {len(shapes)} is outside the supported range"
        )
    return shapes


def _validate_inventory(
    inventory: dict[str, dict[str, Any]], expected_shapes: dict[str, list[int]]
) -> None:
    actual_names = set(inventory)
    expected_names = set(expected_shapes)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatches: list[str] = []
    for name in sorted(actual_names & expected_names):
        info = inventory[name]
        shape = info.get("shape")
        dtype = info.get("dtype")
        nbytes = info.get("nbytes")
        expected_shape = expected_shapes[name]
        if dtype not in DENSE_MMPROJ_DTYPES:
            mismatches.append(f"{name}: unsupported dtype {dtype!r}")
            continue
        expected_nbytes = DTYPE_BYTES[dtype]
        for dimension in expected_shape:
            expected_nbytes *= dimension
        if shape != expected_shape or nbytes != expected_nbytes:
            mismatches.append(
                f"{name}: found {dtype}{shape} ({nbytes} bytes), "
                f"expected {dtype}{expected_shape} ({expected_nbytes} bytes)"
            )
    if missing or unexpected or mismatches:
        raise ValueError(
            "mmproj inventory does not match model config; "
            f"missing={missing}, unexpected={unexpected}, mismatches={mismatches}"
        )


def _validate_provenance(source_model: str, source_revision: str) -> tuple[str, str]:
    normalized_model = source_model.strip()
    if not normalized_model or normalized_model != source_model:
        raise ValueError("source model must be a non-empty identifier without outer whitespace")
    if len(source_revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError(
            "source revision must be an immutable 40- or 64-character lowercase hex digest"
        )
    return normalized_model, source_revision


def _copy_exact(source: BinaryIO, target: BinaryIO, length: int) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise EOFError("safetensors payload ended during mmproj extraction")
        target.write(chunk)
        remaining -= len(chunk)


def _export_safetensors(
    source_path: Path,
    output_path: Path,
    *,
    source_prefix: str,
    expected_shapes: dict[str, list[int]],
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    source_tensors, source_metadata, data_start, _ = _read_safetensors_header(source_path)
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for source_name, info in source_tensors.items():
        output_name = _canonical_name(source_name, source_prefix)
        if output_name is None:
            continue
        if output_name in seen:
            raise ValueError(f"duplicate normalized mmproj tensor name {output_name!r}")
        if info["dtype"] not in DENSE_MMPROJ_DTYPES:
            raise ValueError(
                f"mmproj tensor {source_name!r} has unsupported dense dtype {info['dtype']!r}"
            )
        seen.add(output_name)
        selected.append((output_name, {**info, "source_name": source_name}))
    selected.sort(key=lambda item: item[0])
    if not any(name.startswith(CANONICAL_VISION_PREFIX) for name, _ in selected):
        raise ValueError("source safetensors contains no LFM2-VL vision tensors")
    if not any(name.startswith(PROJECTOR_PREFIX) for name, _ in selected):
        raise ValueError("source safetensors contains no LFM2-VL projector tensors")

    output_header: dict[str, Any] = {}
    offset = 0
    inventory: dict[str, dict[str, Any]] = {}
    for output_name, info in selected:
        source_start, source_end = info["data_offsets"]
        length = source_end - source_start
        output_header[output_name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + length],
        }
        inventory[output_name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "nbytes": length,
        }
        offset += length
    _validate_inventory(inventory, expected_shapes)
    output_header["__metadata__"] = {
        **source_metadata,
        "candle-mmproj-format": "1",
    }
    raw_header = json.dumps(output_header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    raw_header += b" " * (-len(raw_header) % 8)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as target:
        temporary = Path(target.name)
        target.write(struct.pack("<Q", len(raw_header)))
        target.write(raw_header)
        with source_path.open("rb") as source:
            for _, info in selected:
                source_start, source_end = info["data_offsets"]
                source.seek(data_start + source_start)
                _copy_exact(source, target, source_end - source_start)
        target.flush()
        os.fsync(target.fileno())
    _publish_temporary(temporary, output_path, overwrite=overwrite)
    return inventory


def _processor_object(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("image_processor")
    if nested is None:
        return document
    if not isinstance(nested, dict):
        raise ValueError("processor image_processor field must be an object")
    return nested


def export(args: argparse.Namespace) -> dict[str, Any]:
    source_model, source_revision = _validate_provenance(
        args.source_model, args.source_revision
    )
    model_config, model_config_raw = _read_json(args.model_config)
    processor_config, _ = _read_json(args.processor_config)
    processor_values = _processor_object(processor_config)

    model_type = model_config.get("model_type", "lfm2_vl")
    if model_type not in ("lfm2_vl", "lfm2-vl"):
        raise ValueError(f"unsupported model_type {model_type!r}")
    text_config = model_config.get("text_config")
    vision_config = model_config.get("vision_config")
    if not isinstance(text_config, dict) or not isinstance(vision_config, dict):
        raise ValueError("model config must contain text_config and vision_config objects")
    text_hidden_size = text_config.get("hidden_size")
    text_layer_count = text_config.get("num_hidden_layers")
    vision_hidden_size = vision_config.get("hidden_size")
    vision_layer_count = vision_config.get("num_hidden_layers")
    patch_size = vision_config.get("patch_size")
    downsample_factor = model_config.get("downsample_factor", 2)
    image_token_id = model_config.get(
        "image_token_id", model_config.get("image_token_index", 396)
    )
    for name, value in (
        ("text hidden size", text_hidden_size),
        ("text layer count", text_layer_count),
        ("vision hidden size", vision_hidden_size),
        ("vision layer count", vision_layer_count),
        ("patch size", patch_size),
        ("downsample factor", downsample_factor),
        ("image token id", image_token_id),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid {name}: {value!r}")
    processor_patch_size = processor_values.get(
        "encoder_patch_size", processor_values.get("patch_size", 16)
    )
    processor_downsample = processor_values.get("downsample_factor", 2)
    if processor_patch_size != patch_size:
        raise ValueError(
            f"processor patch size {processor_patch_size} does not match model {patch_size}"
        )
    if processor_downsample != downsample_factor:
        raise ValueError(
            "processor downsample factor "
            f"{processor_downsample} does not match model {downsample_factor}"
        )
    expected_shapes = _expected_tensor_shapes(model_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "mmproj.safetensors"
    manifest_path = args.output_dir / "mmproj.json"
    processor_path = args.output_dir / "processor_config.json"
    if not args.overwrite:
        existing = [
            str(path)
            for path in (weights_path, manifest_path, processor_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing split mmproj output: " + ", ".join(existing)
            )
    inventory = _export_safetensors(
        args.input,
        weights_path,
        source_prefix=args.source_prefix,
        expected_shapes=expected_shapes,
        overwrite=args.overwrite,
    )
    canonical_processor = _json_bytes(processor_config)
    _write_atomic(processor_path, canonical_processor, overwrite=args.overwrite)

    manifest = {
        "format": "candle-mmproj",
        "version": 1,
        "architecture": "lfm2_vl",
        "source_model": source_model,
        "source_revision": source_revision,
        "source_safetensors": args.input.name,
        "source_safetensors_sha256": _sha256(args.input),
        "source_model_config_sha256": hashlib.sha256(model_config_raw).hexdigest(),
        "expected_text_hidden_size": text_hidden_size,
        "expected_text_layer_count": text_layer_count,
        "vision_hidden_size": vision_hidden_size,
        "vision_layer_count": vision_layer_count,
        "patch_size": patch_size,
        "downsample_factor": downsample_factor,
        "image_token_id": image_token_id,
        "tensor_namespace_version": 1,
        "tensor_count": len(inventory),
        "tensor_inventory": inventory,
        "mmproj_safetensors_sha256": _sha256(weights_path),
        "processor_config_sha256": hashlib.sha256(canonical_processor).hexdigest(),
        "model_config": model_config,
    }
    _write_atomic(manifest_path, _json_bytes(manifest), overwrite=args.overwrite)
    return {
        "manifest": str(manifest_path),
        "processor_config": str(processor_path),
        "safetensors": str(weights_path),
        "tensor_count": len(inventory),
        "safetensors_sha256": manifest["mmproj_safetensors_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Strip a fixture/container prefix before matching canonical model tensors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = export(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
