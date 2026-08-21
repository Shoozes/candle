"""Guarded CPU-F32 tensor tracing for a pinned LFM2.5-VL checkpoint.

This module is intentionally separate from the tiny/configuration exporter.  It
loads a production model only after the caller has opted into production,
requires a local image and prompt, bounds the input and decode sizes, and
writes the trace outside the repository.  It never downloads implicitly and
never serializes model weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        REFERENCE_PACKAGE_PINS,
        assert_secret_safe,
        image_sha256,
        load_reference_lock,
        model_entry,
        package_versions,
        reference_environment_lock,
        require_reference_environment,
        repo_root,
        remote_code_admission,
        sha256_bytes,
        transformers_entry,
    )
    from .inspect_artifact import resolve_model_snapshot, verify_artifact_unchanged
    from .tensor_dump import write_tensor_bundle
except ImportError:  # pragma: no cover - direct script/module execution
    from manifest import (  # type: ignore
        REFERENCE_PACKAGE_PINS,
        assert_secret_safe,
        image_sha256,
        load_reference_lock,
        model_entry,
        package_versions,
        reference_environment_lock,
        require_reference_environment,
        repo_root,
        remote_code_admission,
        sha256_bytes,
        transformers_entry,
    )
    from inspect_artifact import (  # type: ignore
        resolve_model_snapshot,
        verify_artifact_unchanged,
    )
    from tensor_dump import write_tensor_bundle  # type: ignore


DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_IMAGE_PATCHES = 1024
MAX_NEW_TOKENS = 32
MAX_SOURCE_IMAGE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_IMAGE_CROPS = 64


def _admit_remote_code(
    model_id: str,
    revision: str,
    trust_remote_code: bool,
    artifact_manifest: Mapping[str, Any] | None,
) -> bool:
    """Derive the loader flag only from the locked, rehashed snapshot."""

    if not trust_remote_code:
        return False
    if artifact_manifest is None:
        raise ValueError(
            "trust_remote_code requires the external artifact manifest for the locked snapshot"
        )
    lock = load_reference_lock()
    entry = model_entry(lock, model_id)
    if entry.get("id") != "LiquidAI/LFM2.5-VL-3B" or revision != entry.get("revision"):
        raise ValueError(
            "trust_remote_code is permitted only for the exact locked LFM2.5-VL-3B snapshot"
        )
    if not remote_code_admission(entry, artifact_manifest):
        raise ValueError(
            "the locked snapshot does not admit trust_remote_code without model-provided code"
        )
    return True


def _torch_and_transformers():
    require_reference_environment()
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError(
            "production trace requires the pinned CPU torch and Transformers environment"
        ) from exc
    return torch, AutoModelForImageTextToText, AutoProcessor


def _set_deterministic(torch: Any) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def load_trace_model(
    model_id: str,
    revision: str,
    *,
    allow_download: bool,
    model_dir: Path | None = None,
    trust_remote_code: bool = False,
    artifact_manifest: Mapping[str, Any] | None = None,
):
    """Load one pinned production model on CPU F32 with no device auto-placement."""

    if model_dir is not None and allow_download:
        raise ValueError("an external model snapshot cannot be combined with --allow-download")
    trust_remote_code = _admit_remote_code(
        model_id, revision, trust_remote_code, artifact_manifest
    )
    torch, auto_model, _ = _torch_and_transformers()
    _set_deterministic(torch)
    source = str(model_dir) if model_dir is not None else model_id
    kwargs = {
        "revision": revision,
        "local_files_only": model_dir is not None or not allow_download,
        "trust_remote_code": trust_remote_code,
        "dtype": torch.float32,
    }
    try:
        model = auto_model.from_pretrained(source, **kwargs)
    except TypeError as exc:
        # Transformers versions before the dtype keyword used torch_dtype.  A
        # keyword-only retry is safe because the first call failed before a
        # model could be constructed.
        if "dtype" not in str(exc):
            raise
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = torch.float32
        model = auto_model.from_pretrained(source, **kwargs)
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return model


def load_trace_processor(
    model_id: str,
    revision: str,
    *,
    allow_download: bool,
    model_dir: Path | None = None,
    trust_remote_code: bool = False,
    artifact_manifest: Mapping[str, Any] | None = None,
):
    """Load the processor at the same pinned revision as the model."""

    if model_dir is not None and allow_download:
        raise ValueError("an external model snapshot cannot be combined with --allow-download")
    trust_remote_code = _admit_remote_code(
        model_id, revision, trust_remote_code, artifact_manifest
    )
    _, _, auto_processor = _torch_and_transformers()
    source = str(model_dir) if model_dir is not None else model_id
    return auto_processor.from_pretrained(
        source,
        revision=revision,
        local_files_only=model_dir is not None or not allow_download,
        trust_remote_code=trust_remote_code,
    )


def _load_rgb_image(image_path: Path):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError("production trace requires Pillow") from exc
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise ValueError(f"source image is not a regular file: {image_path}")
    source_bytes = image_path.read_bytes()
    if not source_bytes:
        raise ValueError(f"source image is empty: {image_path}")
    if len(source_bytes) > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError(
            f"source image is {len(source_bytes)} bytes; maximum is {MAX_SOURCE_IMAGE_BYTES}"
        )
    with Image.open(image_path) as opened:
        if opened.width <= 0 or opened.height <= 0:
            raise ValueError(f"source image has invalid dimensions: {image_path}")
        if opened.width * opened.height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(
                f"source image has {opened.width * opened.height} pixels; "
                f"maximum is {MAX_SOURCE_IMAGE_PIXELS}"
            )
        image = opened.convert("RGB")
        image.load()
    return image, source_bytes


def _tensor_value(torch: Any, value: Any, *, name: str):
    if not torch.is_tensor(value):
        raise ValueError(f"processor field {name!r} is not a tensor")
    return value.detach().to(device="cpu")


def _bundle_tensor(torch: Any, tensor: Any):
    tensor = tensor.detach().to(device="cpu")
    if tensor.dtype == torch.bool:
        tensor = tensor.to(dtype=torch.int64)
    return tensor


def _processor_inputs(
    processor: Any,
    image: Any,
    prompt: str,
    *,
    torch: Any,
    image_token_id: int,
    max_input_tokens: int,
    max_image_patches: int,
) -> tuple[Mapping[str, Any], str, dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("production trace prompt must not be empty")
    if len(prompt) > 1_000_000:
        raise ValueError("production trace prompt exceeds the 1,000,000-character bound")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise ValueError("processor did not return a rendered prompt string")
    tokenizer = getattr(processor, "tokenizer", None)
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert_ids_to_tokens is None:
        raise ValueError("processor tokenizer cannot resolve the configured image token")
    image_token = convert_ids_to_tokens(image_token_id)
    if not isinstance(image_token, str) or not image_token:
        raise ValueError("configured image token ID did not resolve to a token string")
    processor_image_token = getattr(processor, "image_token", image_token)
    if processor_image_token != image_token:
        raise ValueError("processor and model config disagree on the image token")
    if rendered_prompt.count(image_token) != 1:
        raise ValueError(
            "rendered prompt must contain exactly one processor image token; "
            f"found {rendered_prompt.count(image_token)}"
        )
    batch = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    if not isinstance(batch, Mapping):
        raise ValueError("processor did not return a mapping-like BatchFeature")
    normalized: dict[str, Any] = {}
    for name, value in batch.items():
        if torch.is_tensor(value):
            normalized[name] = _tensor_value(torch, value, name=name)

    input_ids = normalized.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("processor must return rank-2 input_ids")
    input_tokens = int(input_ids.shape[-1])
    if input_tokens <= 0 or input_tokens > max_input_tokens:
        raise ValueError(
            f"processor input token count {input_tokens} exceeds bound {max_input_tokens}"
        )
    text_attention_mask = normalized.get("attention_mask")
    if text_attention_mask is None or text_attention_mask.ndim != 2:
        raise ValueError("processor must return a rank-2 attention_mask")
    if tuple(text_attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError("processor attention_mask shape differs from input_ids")
    if not bool((text_attention_mask == 1).all()):
        raise ValueError("production trace requires one unpadded all-ones attention_mask")
    pixel_values = normalized.get("pixel_values")
    if pixel_values is None or pixel_values.ndim != 3:
        raise ValueError("processor must return rank-3 pixel_values")
    if pixel_values.dtype != torch.float32:
        pixel_values = pixel_values.to(dtype=torch.float32)
        normalized["pixel_values"] = pixel_values
    if int(pixel_values.shape[1]) > max_image_patches:
        raise ValueError(
            f"processor patch capacity {int(pixel_values.shape[1])} exceeds bound "
            f"{max_image_patches}"
        )
    if int(pixel_values.shape[0]) > MAX_IMAGE_CROPS:
        raise ValueError(
            f"processor returned {int(pixel_values.shape[0])} crops; maximum is {MAX_IMAGE_CROPS}"
        )
    if int(pixel_values.shape[0]) != 1:
        raise ValueError(
            "production trace currently requires exactly one image crop; "
            "use a non-tiled deterministic source image"
        )
    attention_mask = normalized.get("pixel_attention_mask")
    spatial_shapes = normalized.get("spatial_shapes")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("processor must return rank-2 pixel_attention_mask")
    if spatial_shapes is None or spatial_shapes.ndim != 2 or spatial_shapes.shape[-1] != 2:
        raise ValueError("processor must return rank-2 spatial_shapes")
    if int(attention_mask.shape[0]) != int(pixel_values.shape[0]):
        raise ValueError("processor crop count differs between pixel values and attention mask")
    if int(spatial_shapes.shape[0]) != int(pixel_values.shape[0]):
        raise ValueError("processor crop count differs between pixel values and spatial shapes")
    if int(attention_mask.shape[1]) != int(pixel_values.shape[1]):
        raise ValueError("processor patch capacity differs between pixel values and mask")
    valid_counts = attention_mask.to(dtype=torch.int64).sum(dim=1)
    if bool((valid_counts <= 0).any()):
        raise ValueError("processor returned an empty image crop")
    if bool((valid_counts > max_image_patches).any()):
        raise ValueError("processor returned a crop beyond the patch bound")

    model_inputs = {
        name: value
        for name, value in normalized.items()
        if name in {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask", "spatial_shapes"}
    }
    crop_ranges: list[list[int]] = []
    cursor = 0
    for count in valid_counts.tolist():
        next_cursor = cursor + int(count)
        crop_ranges.append([cursor, next_cursor])
        cursor = next_cursor
    details = {
        "input_tokens": input_tokens,
        "image_token": image_token,
        "image_token_id": image_token_id,
        "attention_mask_shape": list(text_attention_mask.shape),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_attention_mask_shape": list(attention_mask.shape),
        "spatial_shapes_shape": list(spatial_shapes.shape),
        "valid_patch_counts": [int(value) for value in valid_counts.tolist()],
        "projector_crop_ranges": crop_ranges,
    }
    return model_inputs, rendered_prompt, {"normalized": normalized, "details": details}


def _register_vision_hooks(model: Any, stages: dict[str, Any], save_hook: Any):
    vision_tower = model.model.vision_tower
    vision_core = getattr(vision_tower, "vision_model", vision_tower)
    embeddings = getattr(vision_core, "embeddings", None)
    encoder = getattr(vision_core, "encoder", None)
    post_layernorm = getattr(vision_core, "post_layernorm", None)
    if embeddings is None or encoder is None or post_layernorm is None:
        raise RuntimeError("pinned vision model is missing a required trace stage")
    patch_embedding = getattr(embeddings, "patch_embedding", None)
    position_embedding = getattr(embeddings, "position_embedding", None)
    if patch_embedding is None or position_embedding is None:
        raise RuntimeError("pinned vision embeddings are missing patch or position weights")
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("pinned vision encoder has no layers")
    handles = [
        patch_embedding.register_forward_hook(
            save_hook(stages, "stage.vision.patch_embedding")
        ),
        embeddings.register_forward_hook(
            save_hook(stages, "stage.vision.embeddings_with_resized_position")
        ),
        post_layernorm.register_forward_hook(
            save_hook(stages, "stage.vision.post_layernorm")
        ),
    ]
    handles.extend(
        layer.register_forward_hook(save_hook(stages, f"stage.vision.encoder_layer.{index}"))
        for index, layer in enumerate(layers)
    )
    def save_vision_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "detach"):
            stages["stage.vision.last_hidden_state"] = output.detach().clone()

    handles.append(vision_tower.register_forward_hook(save_vision_output))
    return vision_core, embeddings, handles


def _model_inputs_for_trace(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in batch.items()
        if name in {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask", "spatial_shapes"}
    }


def export_production_trace(
    *,
    model: str,
    revision: str | None,
    image_path: Path,
    prompt: str,
    output: Path,
    allow_download: bool,
    model_dir: Path | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_image_patches: int = DEFAULT_MAX_IMAGE_PATCHES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export one bounded production trace to an external directory."""

    if max_new_tokens <= 0 or max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if max_image_patches <= 0:
        raise ValueError("max_image_patches must be positive")
    root = repo_root()
    image_path = image_path.resolve()
    output = output.resolve()
    for label, path in (("trace output", output), ("source image", image_path)):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise PermissionError(f"{label} must be outside the repository: {path}")
    lock = load_reference_lock()
    entry = model_entry(lock, model)
    locked_revision = str(entry["revision"])
    if revision and revision != locked_revision:
        raise ValueError(f"revision {revision} is not the locked revision {locked_revision}")
    if model_dir is None:
        raise ValueError(
            "production trace requires --model-dir for an identified regular-file snapshot"
        )
    if allow_download:
        raise ValueError("production trace with --model-dir cannot use --allow-download")
    resolved_model_dir, artifact_manifest = resolve_model_snapshot(model, model_dir)
    trust_remote_code = remote_code_admission(entry, artifact_manifest)
    image, source_bytes = _load_rgb_image(image_path)
    torch, _, _ = _torch_and_transformers()
    _set_deterministic(torch)
    model_object = load_trace_model(
        str(entry["id"]),
        locked_revision,
        allow_download=allow_download,
        model_dir=resolved_model_dir,
        trust_remote_code=trust_remote_code,
        artifact_manifest=artifact_manifest,
    )
    processor = load_trace_processor(
        str(entry["id"]),
        locked_revision,
        allow_download=allow_download,
        model_dir=resolved_model_dir,
        trust_remote_code=trust_remote_code,
        artifact_manifest=artifact_manifest,
    )

    model_inputs, rendered_prompt, processor_evidence = _processor_inputs(
        processor,
        image,
        prompt,
        torch=torch,
        image_token_id=int(model_object.config.image_token_id),
        max_input_tokens=max_input_tokens,
        max_image_patches=max_image_patches,
    )
    normalized = processor_evidence["normalized"]
    input_evidence = processor_evidence["details"]

    # Importing these helpers lazily keeps config-only usage free of the heavy
    # model imports and avoids a module cycle during direct script execution.
    try:
        from .export_fixtures import (
            _embed_tokens,
            _module_class_names,
            _project_valid_features,
            _replace_image_embeddings,
            _save_hook,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from export_fixtures import (  # type: ignore
            _embed_tokens,
            _module_class_names,
            _project_valid_features,
            _replace_image_embeddings,
            _save_hook,
        )

    stages: dict[str, Any] = {}
    _vision_core, embeddings, handles = _register_vision_hooks(
        model_object, stages, _save_hook
    )
    try:
        with torch.no_grad():
            position_table = embeddings.position_embedding.weight
            base_side = math.isqrt(int(position_table.shape[0]))
            if base_side * base_side != int(position_table.shape[0]):
                raise ValueError("production position table is not a square grid")
            resize_positions = getattr(embeddings, "resize_positional_embeddings", None)
            if resize_positions is None:
                raise RuntimeError("pinned vision embeddings have no resize method")
            resized_position = resize_positions(
                position_table.reshape(base_side, base_side, position_table.shape[-1]),
                model_inputs["spatial_shapes"],
                int(model_inputs["pixel_values"].shape[1]),
            )
            stages["stage.vision.resized_position_embedding"] = resized_position.detach().clone()
            prefill = model_object(
                **_model_inputs_for_trace(model_inputs),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    vision_hidden = stages.get("stage.vision.last_hidden_state")
    if vision_hidden is None:
        raise RuntimeError("vision trace did not capture last hidden state")
    projected, projector_stages = _project_valid_features(
        model_object,
        vision_hidden,
        model_inputs["pixel_attention_mask"],
        model_inputs["spatial_shapes"],
        capture_inputs=True,
    )
    stages.update(projector_stages)
    input_ids = model_inputs["input_ids"]
    text_embeddings, merged_embeddings = _replace_image_embeddings(
        model_object, input_ids, projected
    )
    if not hasattr(prefill, "past_key_values") or prefill.past_key_values is None:
        raise RuntimeError("production prefill did not return a cache")

    reset_prefill = model_object(
        **_model_inputs_for_trace(model_inputs),
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    reset_delta = (prefill.logits - reset_prefill.logits).abs()
    reset_max_abs = float(reset_delta.max().item())
    reset_exact = bool(torch.equal(prefill.logits, reset_prefill.logits))
    if not reset_exact:
        raise RuntimeError(
            f"production cache reset prefill is not bit-exact (max abs {reset_max_abs:.9g})"
        )

    cache = prefill.past_key_values
    next_input_ids = prefill.logits[:, -1:, :].argmax(dim=-1)
    decode_input_ids = []
    decode_logits = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            decode_input_ids.append(next_input_ids.detach().clone())
            decode = model_object(
                input_ids=next_input_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            decode_logits.append(decode.logits[:, -1:, :].detach().clone())
            cache = decode.past_key_values
            if cache is None:
                raise RuntimeError("production decode dropped the model cache")
            next_input_ids = decode.logits[:, -1:, :].argmax(dim=-1)

    input_tensors: dict[str, Any] = {}
    for name, value in normalized.items():
        if torch.is_tensor(value):
            input_tensors[f"input.{name}"] = _bundle_tensor(torch, value)
    rgb_tensor = torch.tensor(list(image.tobytes()), dtype=torch.uint8).reshape(
        image.height, image.width, 3
    )
    input_tensors["input.image_rgb_u8"] = rgb_tensor
    ranges = torch.tensor(input_evidence["projector_crop_ranges"], dtype=torch.int64)
    input_tensors["input.projector_crop_ranges"] = ranges
    if not prefill.hidden_states:
        raise RuntimeError("production prefill did not return hidden states")
    tensors = {
        **input_tensors,
        **stages,
        "stage.projector.output": projected,
        "stage.text.embeddings": text_embeddings,
        "stage.multimodal.merged_embeddings": merged_embeddings,
        "stage.language.hidden_states": prefill.hidden_states[-1],
        "stage.language.prefill_logits": prefill.logits,
        "stage.language.decode_logits": torch.cat(decode_logits, dim=1),
        "input.decode_token_ids": torch.cat(decode_input_ids, dim=1),
    }
    _, final_artifact_manifest = resolve_model_snapshot(model, resolved_model_dir)
    verify_artifact_unchanged(
        artifact_manifest,
        final_artifact_manifest,
        operation="the trace",
    )
    source_hash = image_sha256(source_bytes)
    decoded_hash = sha256_bytes(image.tobytes())
    metadata = {
        "schema_version": 1,
        "mode": "production-trace",
        "device": "cpu",
        "dtype": "float32",
        "seed": 0,
        "source_image_path_name": image_path.name,
        "source_image_sha256": source_hash,
        "source_image_decoded_rgb_sha256": decoded_hash,
        "source_image_shape": [image.height, image.width, 3],
        "prompt": rendered_prompt,
        "user_prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "package_versions": package_versions(),
        "package_pins": REFERENCE_PACKAGE_PINS,
        "environment_lock": reference_environment_lock(),
        "transformers_revision": transformers_entry(lock)["revision"],
        "model_id": entry["id"],
        "model_revision": locked_revision,
        "processor_revision": locked_revision,
        "model_snapshot_mode": "external-regular-file",
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_reverified": True,
        "model_class": type(model_object).__name__,
        "processor_class": type(processor).__name__,
        "official_module_classes": _module_class_names(model_object),
        "processor_evidence": input_evidence,
        "max_new_tokens": max_new_tokens,
        "cache_reset_exact": reset_exact,
        "cache_reset_prefill_max_abs": reset_max_abs,
        "weights_serialized": False,
    }
    manifest = {
        "schema_version": 1,
        "mode": "production-trace",
        "device": "cpu",
        "dtype": "float32",
        "seed": 0,
        "source_image_sha256": source_hash,
        "source_image_decoded_rgb_sha256": decoded_hash,
        "transformers_revision": transformers_entry(lock)["revision"],
        "model_id": entry["id"],
        "model_revision": locked_revision,
        "processor_revision": locked_revision,
        "model_snapshot_mode": "external-regular-file",
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_reverified": True,
        "max_new_tokens": max_new_tokens,
        "cache_reset_exact": reset_exact,
        "weights_serialized": False,
    }
    assert_secret_safe(metadata)
    write_tensor_bundle(output, tensors, metadata, manifest, overwrite=overwrite)
    return {
        "mode": "production-trace",
        "output": str(output),
        "model_revision": locked_revision,
        "source_image_sha256": source_hash,
        "input_tokens": input_evidence["input_tokens"],
        "projected_tokens": int(projected.shape[0]),
        "decode_steps": max_new_tokens,
        "cache_reset_exact": reset_exact,
        "tensor_count": len(tensors),
    }


__all__ = [
    "DEFAULT_MAX_IMAGE_PATCHES",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_NEW_TOKENS",
    "MAX_IMAGE_CROPS",
    "MAX_NEW_TOKENS",
    "MAX_SOURCE_IMAGE_BYTES",
    "MAX_SOURCE_IMAGE_PIXELS",
    "export_production_trace",
    "load_trace_model",
    "load_trace_processor",
]
