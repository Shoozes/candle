"""Export config, deterministic tiny-random, or explicit production metadata.

The tiny path instantiates the pinned Hugging Face Transformers LFM2, SigLIP2,
and LFM2-VL classes with reduced dimensions.  It does not provide a lookalike
network: the operation classes under test are the official classes at the
locked Transformers revision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .inspect_config import inspect_config
    from .inspect_artifact import resolve_model_snapshot, verify_artifact_unchanged
    from .manifest import (
        REFERENCE_PACKAGE_PINS,
        assert_secret_safe,
        image_sha256,
        load_reference_lock,
        model_entry,
        normalized_model_summary,
        package_versions,
        reference_environment_lock,
        require_reference_environment,
        repo_root,
        transformers_entry,
    )
    from .tensor_dump import write_metadata_bundle, write_tensor_bundle
except ImportError:  # pragma: no cover - direct script execution
    from inspect_config import inspect_config  # type: ignore
    from inspect_artifact import (  # type: ignore
        resolve_model_snapshot,
        verify_artifact_unchanged,
    )
    from manifest import (  # type: ignore
        REFERENCE_PACKAGE_PINS,
        assert_secret_safe,
        image_sha256,
        load_reference_lock,
        model_entry,
        normalized_model_summary,
        package_versions,
        reference_environment_lock,
        require_reference_environment,
        repo_root,
        transformers_entry,
    )
    from tensor_dump import write_metadata_bundle, write_tensor_bundle  # type: ignore


TINY_CONFIG = {
    "vocab_size": 32,
    "text_hidden_size": 12,
    "text_layers": 2,
    "decode_steps": 3,
    "text_attention_heads": 3,
    "text_key_value_heads": 1,
    "text_intermediate_size": 32,
    "text_layer_types": ["conv", "full_attention"],
    "vision_hidden_size": 16,
    "vision_layers": 2,
    "vision_attention_heads": 4,
    "vision_intermediate_size": 32,
    "vision_num_patches": 16,
    "patch_size": 2,
    "downsample_factor": 2,
    "projector_hidden_size": 24,
    "image_token_id": 3,
    "max_patches": 10,
    "spatial_shape": [2, 4],
}


def _torch_and_official_classes():
    require_reference_environment()
    try:
        import torch
        from transformers import (
            Lfm2Config,
            Lfm2VlConfig,
            Lfm2VlForConditionalGeneration,
            Siglip2VisionConfig,
        )
    except ImportError:
        try:
            import torch
            from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
            from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
            from transformers.models.lfm2_vl.modeling_lfm2_vl import (
                Lfm2VlForConditionalGeneration,
            )
            from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
        except ImportError as exc:  # pragma: no cover - manager environment
            raise RuntimeError(
                "tiny-random mode requires the exact reference requirements; "
                "config-only mode is stdlib-only"
            ) from exc
    return torch, Lfm2Config, Lfm2VlConfig, Lfm2VlForConditionalGeneration, Siglip2VisionConfig


def _set_deterministic(torch: Any, seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_official_tiny(seed: int):
    """Construct a small official LFM2-VL model on CPU."""

    torch, Lfm2Config, Lfm2VlConfig, Lfm2VlForConditionalGeneration, Siglip2VisionConfig = (
        _torch_and_official_classes()
    )
    _set_deterministic(torch, seed)

    text_config = Lfm2Config(
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["text_hidden_size"],
        intermediate_size=TINY_CONFIG["text_intermediate_size"],
        num_hidden_layers=TINY_CONFIG["text_layers"],
        num_attention_heads=TINY_CONFIG["text_attention_heads"],
        num_key_value_heads=TINY_CONFIG["text_key_value_heads"],
        layer_types=list(TINY_CONFIG["text_layer_types"]),
        block_auto_adjust_ff_dim=False,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=8,
        conv_L_cache=3,
        conv_bias=False,
        max_position_embeddings=64,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    vision_config = Siglip2VisionConfig(
        hidden_size=TINY_CONFIG["vision_hidden_size"],
        intermediate_size=TINY_CONFIG["vision_intermediate_size"],
        num_hidden_layers=TINY_CONFIG["vision_layers"],
        num_attention_heads=TINY_CONFIG["vision_attention_heads"],
        num_channels=3,
        patch_size=TINY_CONFIG["patch_size"],
        num_patches=TINY_CONFIG["vision_num_patches"],
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=False,
    )
    config = Lfm2VlConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=TINY_CONFIG["image_token_id"],
        projector_hidden_act="gelu",
        projector_hidden_size=TINY_CONFIG["projector_hidden_size"],
        projector_bias=True,
        projector_use_layernorm=True,
        downsample_factor=TINY_CONFIG["downsample_factor"],
        tie_word_embeddings=True,
        use_image_special_tokens=True,
    )
    model = Lfm2VlForConditionalGeneration(config)
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return model


def tiny_inputs():
    torch, *_ = _torch_and_official_classes()
    rows, cols = TINY_CONFIG["spatial_shape"]
    patch_size = TINY_CONFIG["patch_size"]
    height, width = rows * patch_size, cols * patch_size
    image = torch.arange(height * width * 3, dtype=torch.uint8).reshape(height, width, 3)
    image_bytes = bytes(int(value) for value in image.contiguous().reshape(-1).tolist())
    patches = (
        image.to(dtype=torch.float32)
        .reshape(rows, patch_size, cols, patch_size, 3)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(rows * cols, patch_size * patch_size * 3)
    )
    pixel_values = (patches / 255.0).unsqueeze(0)
    padding_rows = TINY_CONFIG["max_patches"] - rows * cols
    if padding_rows < 0:
        raise ValueError("tiny max_patches is smaller than the synthetic image patch count")
    if padding_rows:
        padding = torch.zeros(
            1,
            padding_rows,
            patch_size * patch_size * 3,
            dtype=pixel_values.dtype,
        )
        pixel_values = torch.cat((pixel_values, padding), dim=1)
    pixel_attention_mask = torch.tensor(
        [[True] * (rows * cols) + [False] * (TINY_CONFIG["max_patches"] - rows * cols)],
        dtype=torch.bool,
    )
    spatial_shapes = torch.tensor([TINY_CONFIG["spatial_shape"]], dtype=torch.long)
    input_ids = torch.tensor([[1, 3, 3, 2, 4]], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "spatial_shapes": spatial_shapes,
        "input_ids": input_ids,
        "image_bytes": image_bytes,
        "image_shape": [height, width, 3],
    }


def _embed_tokens(language_model: Any, input_ids: Any):
    if hasattr(language_model, "embed_tokens"):
        return language_model.embed_tokens(input_ids)
    embeddings = language_model.get_input_embeddings()
    return embeddings(input_ids)


def _save_hook(stages: dict[str, Any], name: str):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "detach"):
            stages[name] = output.detach().clone()

    return hook


def _project_valid_features(
    model: Any,
    vision_hidden: Any,
    mask: Any,
    spatial_shapes: Any,
    *,
    capture_inputs: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Use the official projector class after official vision execution."""

    torch, *_ = _torch_and_official_classes()
    projected = []
    projector_inputs = []
    stages: dict[str, Any] = {}
    projector = model.model.multi_modal_projector
    factor = int(model.config.downsample_factor)
    for batch_index in range(vision_hidden.shape[0]):
        valid = vision_hidden[batch_index][mask[batch_index].bool()]
        rows, cols = (int(value) for value in spatial_shapes[batch_index].tolist())
        if rows * cols != int(valid.shape[0]):
            raise ValueError("tiny spatial shape does not match valid patch count")
        if rows % factor or cols % factor:
            raise ValueError("tiny spatial shape is not divisible by downsample factor")
        grid = valid.reshape(1, rows, cols, valid.shape[-1])
        if capture_inputs:
            projector_inputs.append(valid.detach().clone())

        # pixel_unshuffle is a plain method on the pinned projector class, not
        # a child module.  Keep this explicit stage sequence aligned with its
        # forward method, then compare it with the official composite call.
        unshuffled_features = projector.pixel_unshuffle(grid)
        stages["stage.projector.pixel_unshuffle"] = unshuffled_features.detach().clone()
        layer_norm = getattr(projector, "layer_norm", None)
        if layer_norm is None:
            normalized_features = unshuffled_features
        else:
            normalized_features = layer_norm(unshuffled_features)
            stages["stage.projector.layer_norm"] = normalized_features.detach().clone()
        linear_1 = projector.linear_1(normalized_features)
        activation = projector.act(linear_1)
        linear_2 = projector.linear_2(activation)
        stages["stage.projector.linear_1"] = linear_1.detach().clone()
        stages["stage.projector.activation"] = activation.detach().clone()
        stages["stage.projector.linear_2"] = linear_2.detach().clone()
        output = projector(grid)
        torch.testing.assert_close(output, linear_2, rtol=0.0, atol=0.0)
        projected.append(output.reshape(-1, output.shape[-1]))
    if not projected:
        raise ValueError("no image crops to project")
    if capture_inputs:
        stages["stage.projector.input"] = torch.cat(projector_inputs, dim=0)
    return torch.cat(projected, dim=0), stages


def _replace_image_embeddings(model: Any, input_ids: Any, projected: Any):
    language_model = model.model.language_model
    embeds = _embed_tokens(language_model, input_ids)
    image_id = int(model.config.image_token_id)
    locations = (input_ids == image_id).nonzero(as_tuple=False)
    if locations.shape[0] != projected.shape[0]:
        raise ValueError(
            f"image placeholder count {locations.shape[0]} does not match "
            f"projected feature count {projected.shape[0]}"
        )
    merged = embeds.clone()
    for feature_index, location in enumerate(locations):
        merged[tuple(int(value) for value in location)] = projected[feature_index]
    return embeds, merged


def _module_class_names(module: Any) -> list[str]:
    return sorted({type(child).__name__ for child in module.modules()})


def run_official_tiny(model: Any, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, *_ = _torch_and_official_classes()
    stages: dict[str, Any] = {}
    vision_tower = model.model.vision_tower
    vision_core = getattr(vision_tower, "vision_model", vision_tower)
    embedding_module = getattr(vision_core, "embeddings", None)
    if embedding_module is None:
        raise RuntimeError("pinned SigLIP2 vision model has no embeddings module")
    vision_hooks = [
        embedding_module.patch_embedding.register_forward_hook(
            _save_hook(stages, "stage.vision.patch_embedding")
        ),
        embedding_module.register_forward_hook(
            _save_hook(stages, "stage.vision.embeddings_with_resized_position")
        ),
        vision_core.post_layernorm.register_forward_hook(
            _save_hook(stages, "stage.vision.post_layernorm")
        ),
    ]
    vision_hooks.extend(
        layer.register_forward_hook(
            _save_hook(stages, f"stage.vision.encoder_layer.{index}")
        )
        for index, layer in enumerate(vision_core.encoder.layers)
    )
    with torch.no_grad():
        position_table = embedding_module.position_embedding.weight
        base_side = math.isqrt(int(position_table.shape[0]))
        if base_side * base_side != int(position_table.shape[0]):
            raise ValueError("tiny base position table is not a square grid")
        resized_position = embedding_module.resize_positional_embeddings(
            position_table.reshape(base_side, base_side, position_table.shape[-1]),
            inputs["spatial_shapes"],
            int(inputs["pixel_values"].shape[1]),
        )
        stages["stage.vision.resized_position_embedding"] = (
            resized_position.detach().clone()
        )
        try:
            vision_outputs = model.model.vision_tower(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
                return_dict=True,
            )
        finally:
            for handle in vision_hooks:
                handle.remove()
        vision_hidden = vision_outputs.last_hidden_state
        stages["stage.vision.last_hidden_state"] = vision_hidden.detach().clone()
        projected, projector_stages = _project_valid_features(
            model,
            vision_hidden,
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        stages.update(projector_stages)
        text_embeds, merged_embeds = _replace_image_embeddings(
            model, inputs["input_ids"], projected
        )
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        next_input_id = outputs.logits[:, -1:, :].argmax(dim=-1)
        decode_input_ids = []
        decode_logits = []
        for _ in range(TINY_CONFIG["decode_steps"]):
            decode_input_ids.append(next_input_id.detach().clone())
            decode_output = model(
                input_ids=next_input_id,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            decode_logits.append(decode_output.logits[:, -1, :].detach().clone())
            cache = decode_output.past_key_values
            next_input_id = decode_output.logits[:, -1:, :].argmax(dim=-1)
    hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
    tensors = {
        "input.pixel_values": inputs["pixel_values"],
        # Store masks as integer tensors for broad safetensors compatibility;
        # the official model still receives the original boolean mask above.
        "input.pixel_attention_mask": inputs["pixel_attention_mask"].to(dtype=torch.int64),
        "input.spatial_shapes": inputs["spatial_shapes"],
        "input.input_ids": inputs["input_ids"],
        "input.decode_token_ids": torch.cat(decode_input_ids, dim=1),
        **stages,
        "stage.projector.output": projected,
        "stage.text.embeddings": text_embeds,
        "stage.multimodal.merged_embeddings": merged_embeds,
        "stage.language.hidden_states": hidden_states,
        "stage.language.prefill_logits": outputs.logits,
        "stage.language.decode_logits": torch.stack(decode_logits, dim=1),
    }
    tensors.update(
        {
            f"weights.{name}": value
            for name, value in model.state_dict().items()
            if name != "lm_head.weight"
        }
    )
    details = {
        "model_class": type(model).__name__,
        "vision_class": type(model.model.vision_tower).__name__,
        "projector_class": type(model.model.multi_modal_projector).__name__,
        "language_class": type(model.model.language_model).__name__,
        "official_module_classes": _module_class_names(model),
        "attention_and_short_convolution_present": {
            "attention": any("attention" in name.lower() for name in _module_class_names(model)),
            "short_convolution": any(
                "conv" in name.lower() or "convolution" in name.lower()
                for name in _module_class_names(model)
            ),
        },
        "omitted_tied_lm_head": bool(
            model.config.tie_word_embeddings
            and "lm_head.weight" in model.state_dict()
        ),
    }
    return tensors, details


def _check_existing_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {path.resolve()}; pass --overwrite to reuse it"
        )


def _tiny_export(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    _check_existing_output(output, overwrite=args.overwrite)
    lock = load_reference_lock()
    entry = model_entry(lock, "450m")
    model = build_official_tiny(args.seed)
    inputs = tiny_inputs()
    tensors, details = run_official_tiny(model, inputs)
    transformers_revision = transformers_entry(lock)["revision"]
    source_hash = image_sha256(inputs["image_bytes"])
    metadata = {
        "schema_version": 1,
        "mode": "tiny-random",
        "device": "cpu",
        "dtype": "float32",
        "seed": args.seed,
        "source_image_sha256": source_hash,
        "source_image_shape": inputs["image_shape"],
        "source_image_format": "raw-rgb-u8-row-major",
        "source_image_to_patches": "reshape[rows,patch_h,cols,patch_w,channels] -> permute[rows,cols,patch_h,patch_w,channels] -> flatten / 255",
        "package_versions": package_versions(),
        "package_pins": REFERENCE_PACKAGE_PINS,
        "transformers_revision": transformers_revision,
        "model_id": entry["id"],
        "model_revision": entry["revision"],
        "processor_revision": entry["revision"],
        "tiny_config": TINY_CONFIG,
        "official_classes": details,
        "input_contract": {
            "pixel_values": [1, TINY_CONFIG["max_patches"], 12],
            "pixel_attention_mask": [1, TINY_CONFIG["max_patches"]],
            "spatial_shapes": [1, 2],
            "input_ids": [1, 5],
            "decode_token_ids": [1, TINY_CONFIG["decode_steps"]],
        },
    }
    manifest = {
        "schema_version": 1,
        "mode": "tiny-random",
        "device": "cpu",
        "dtype": "float32",
        "seed": args.seed,
        "source_image_sha256": source_hash,
        "source_image_shape": inputs["image_shape"],
        "transformers_revision": transformers_revision,
        "model_id": entry["id"],
        "model_revision": entry["revision"],
        "processor_revision": entry["revision"],
    }
    assert_secret_safe(metadata)
    write_tensor_bundle(
        output,
        tensors,
        metadata,
        manifest,
        overwrite=args.overwrite,
    )
    return {
        "mode": "tiny-random",
        "output": str(output),
        "tensor_count": len(tensors),
        "source_image_sha256": source_hash,
        "model_revision": entry["revision"],
    }


def _config_export(args: argparse.Namespace) -> dict[str, Any]:
    summary = inspect_config(
        model=args.model,
        config_path=args.config,
        processor_config_path=args.processor_config,
    )
    if not args.output:
        return summary
    output = args.output.resolve()
    entry = model_entry(load_reference_lock(), args.model)
    metadata = {
        "schema_version": 1,
        "mode": "config-only",
        "device": "cpu",
        "dtype": "not-applicable",
        "seed": None,
        "source_image_sha256": None,
        "package_versions": package_versions(),
        "package_pins": REFERENCE_PACKAGE_PINS,
        "summary": summary,
    }
    manifest = {
        "schema_version": 1,
        "mode": "config-only",
        "device": "cpu",
        "dtype": "not-applicable",
        "seed": None,
        "source_image_sha256": None,
        "transformers_revision": summary["transformers_revision"],
        "model_id": entry["id"],
        "model_revision": entry["revision"],
        "processor_revision": entry["revision"],
    }
    write_metadata_bundle(output, metadata, manifest, overwrite=args.overwrite)
    return {"mode": "config-only", "output": str(output), "summary": summary}


def _production_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_production:
        raise PermissionError(
            "production mode is disabled; pass --allow-production explicitly"
        )
    if not args.output:
        raise ValueError("production mode requires --output outside the repository")
    output = args.output.resolve()
    root = repo_root()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise PermissionError(
            "production metadata output must be outside the repository; "
            "production tensors are never generated by this harness"
        )

    lock = load_reference_lock()
    entry = model_entry(lock, args.model)
    if args.revision and args.revision != entry["revision"]:
        raise ValueError(
            f"revision {args.revision} is not the locked revision {entry['revision']}"
        )
    revision = entry["revision"]
    model_dir = None
    artifact_manifest = None
    if args.model_dir:
        if args.allow_download:
            raise ValueError("an external model snapshot cannot be combined with --allow-download")
        model_dir, artifact_manifest = resolve_model_snapshot(args.model, args.model_dir)
    loaded_model = None
    if args.load_model:
        loader_args = {"allow_download": args.allow_download}
        if model_dir is not None:
            loader_args["model_dir"] = model_dir
        loaded_model = load_production_model(entry["id"], revision, **loader_args)
    artifact_manifest_reverified = False
    if model_dir is not None and artifact_manifest is not None:
        _, final_artifact_manifest = resolve_model_snapshot(args.model, model_dir)
        verify_artifact_unchanged(
            artifact_manifest,
            final_artifact_manifest,
            operation="production loading",
        )
        artifact_manifest_reverified = True
    if args.config:
        summary = inspect_config(
            model=args.model,
            config_path=args.config,
            processor_config_path=args.processor_config,
        )
        source = "local-small-json"
    elif loaded_model is not None:
        summary = normalized_model_summary(
            entry,
            config=loaded_model.config.to_dict(),
            source="loaded-pinned-production-config",
        )
        source = "loaded-pinned-production-config"
    else:
        require_reference_environment()
        try:
            from transformers import AutoConfig
        except ImportError as exc:  # pragma: no cover - manager environment
            raise RuntimeError("production metadata mode requires pinned transformers") from exc
        config_source = str(model_dir) if model_dir is not None else entry["id"]
        config = AutoConfig.from_pretrained(
            config_source,
            revision=revision,
            local_files_only=model_dir is not None or not args.allow_download,
            trust_remote_code=False,
        )
        processor_config = None
        if args.processor_config:
            processor_config = json.loads(args.processor_config.read_text(encoding="utf-8"))
        summary = normalized_model_summary(
            entry,
            config=config.to_dict(),
            processor=processor_config,
            source="pinned-production-config",
        )
        source = "pinned-production-config"
    image_hash = image_sha256(args.image.read_bytes()) if args.image else None
    metadata = {
        "schema_version": 1,
        "mode": "production",
        "source": source,
        "device": "cpu",
        "dtype": "not-loaded" if loaded_model is None else "model-default",
        "seed": args.seed,
        "source_image_sha256": image_hash,
        "package_versions": package_versions(),
        "package_pins": REFERENCE_PACKAGE_PINS,
        "environment_lock": reference_environment_lock(),
        "transformers_revision": transformers_entry(lock)["revision"],
        "model_id": entry["id"],
        "model_revision": revision,
        "processor_revision": revision,
        "model_snapshot_mode": (
            "external-regular-file" if model_dir is not None else "hub-cache-or-local"
        ),
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_reverified": artifact_manifest_reverified,
        "summary": summary,
        "weights_loaded": loaded_model is not None,
        "tensor_payload_generated": False,
        "model_class": type(loaded_model).__name__ if loaded_model is not None else None,
    }
    manifest = {
        "schema_version": 1,
        "mode": "production",
        "device": "cpu",
        "dtype": "not-loaded" if loaded_model is None else "model-default",
        "seed": args.seed,
        "source_image_sha256": image_hash,
        "transformers_revision": transformers_entry(lock)["revision"],
        "model_id": entry["id"],
        "model_revision": revision,
        "processor_revision": revision,
        "model_snapshot_mode": (
            "external-regular-file" if model_dir is not None else "hub-cache-or-local"
        ),
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_reverified": artifact_manifest_reverified,
        "weights_loaded": loaded_model is not None,
        "tensor_payload_generated": False,
    }
    write_metadata_bundle(output, metadata, manifest, overwrite=args.overwrite)
    return {
        "mode": "production",
        "output": str(output),
        "model_revision": revision,
        "weights_loaded": loaded_model is not None,
    }


def _production_trace(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_production:
        raise PermissionError(
            "production trace is disabled; pass --allow-production explicitly"
        )
    if not args.load_model:
        raise PermissionError("production trace requires --load-model explicitly")
    if not args.output:
        raise ValueError("production trace requires --output outside the repository")
    if not args.image:
        raise ValueError("production trace requires --image")
    if not args.prompt or not args.prompt.strip():
        raise ValueError("production trace requires a non-empty --prompt")
    if not args.model_dir:
        raise ValueError(
            "production trace requires --model-dir for an identified regular-file snapshot"
        )
    if args.allow_download:
        raise ValueError("production trace with --model-dir cannot use --allow-download")
    root = repo_root()
    output = args.output.resolve()
    image = args.image.resolve()
    for label, path in (("trace output", output), ("source image", image)):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise PermissionError(f"{label} must be outside the repository: {path}")
    try:
        from .production_trace import (
            DEFAULT_MAX_IMAGE_PATCHES,
            DEFAULT_MAX_INPUT_TOKENS,
            DEFAULT_MAX_NEW_TOKENS,
            export_production_trace,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from production_trace import (  # type: ignore
            DEFAULT_MAX_IMAGE_PATCHES,
            DEFAULT_MAX_INPUT_TOKENS,
            DEFAULT_MAX_NEW_TOKENS,
            export_production_trace,
        )
    return export_production_trace(
        model=args.model,
        revision=args.revision,
        image_path=image,
        prompt=args.prompt,
        output=output,
        allow_download=args.allow_download,
        model_dir=args.model_dir,
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else DEFAULT_MAX_NEW_TOKENS
        ),
        max_input_tokens=(
            args.max_input_tokens
            if args.max_input_tokens is not None
            else DEFAULT_MAX_INPUT_TOKENS
        ),
        max_image_patches=(
            args.max_image_patches
            if args.max_image_patches is not None
            else DEFAULT_MAX_IMAGE_PATCHES
        ),
        overwrite=args.overwrite,
    )


def load_production_model(
    model_id: str,
    revision: str,
    *,
    allow_download: bool,
    model_dir: Path | None = None,
):
    """Load a pinned model only after the CLI has explicitly authorized it.

    The returned model is intentionally not serialized.  This helper is kept
    separate so tests can mock it without network access or a model cache.
    """

    if model_dir is not None and allow_download:
        raise ValueError("an external model snapshot cannot be combined with --allow-download")
    require_reference_environment()
    try:
        from transformers import Lfm2VlForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError("production model loading requires pinned transformers") from exc
    source = str(model_dir) if model_dir is not None else model_id
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        source,
        revision=revision,
        local_files_only=model_dir is not None or not allow_download,
        trust_remote_code=False,
    )
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("config-only", "tiny-random", "production"),
        default="config-only",
    )
    parser.add_argument("--model", default="450m")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="external regular-file snapshot used for identified production loading/tracing",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--processor-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-production", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="in production mode, load the pinned model without serializing its tensors",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="in production mode, export bounded CPU-F32 component tensors outside the repository",
    )
    parser.add_argument("--revision")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-image-patches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.trace and args.mode != "production":
            raise ValueError("--trace requires --mode production")
        if args.mode == "config-only":
            result = _config_export(args)
        elif args.mode == "tiny-random":
            if not args.output:
                raise ValueError("tiny-random mode requires --output")
            result = _tiny_export(args)
        else:
            result = _production_trace(args) if args.trace else _production_metadata(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (FileExistsError, ImportError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"reference harness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
