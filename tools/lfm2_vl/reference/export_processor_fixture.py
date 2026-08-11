"""Export deterministic raw-image processor fixtures from the pinned oracle.

This is intentionally separate from export_fixtures.py: that harness keeps
its three-mode config-only/tiny-random/production contract.  This exporter
uses only the official Lfm2VlImageProcessor and never constructs or loads a
model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        REFERENCE_PACKAGE_PINS,
        package_versions,
        repo_root,
        require_reference_environment,
    )
    from .tensor_dump import write_tensor_bundle
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        REFERENCE_PACKAGE_PINS,
        package_versions,
        repo_root,
        require_reference_environment,
    )
    from tensor_dump import write_tensor_bundle  # type: ignore


TINY_PROCESSOR_CONFIG = {
    "downsample_factor": 2,
    "encoder_patch_size": 2,
    "do_image_splitting": True,
    "min_tiles": 2,
    "max_tiles": 4,
    "use_thumbnail": True,
    "tile_size": 8,
    "min_image_tokens": 1,
    "max_image_tokens": 16,
    "max_num_patches": 64,
    "max_pixels_tolerance": 2.0,
    "do_resize": True,
    "do_rescale": True,
    "rescale_factor": 1.0 / 255.0,
    "do_normalize": True,
    "do_pad": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

CASES = {
    "square": [(8, 8, 3, "RGB")],
    "wide": [(12, 4, 11, "RGB")],
    "tall": [(4, 12, 23, "RGB")],
    "very_wide": [(32, 4, 37, "RGB")],
    "very_tall": [(4, 32, 41, "RGB")],
    "odd": [(7, 5, 53, "RGB")],
    "grayscale": [(7, 5, 59, "L")],
    "rgba": [(7, 5, 61, "RGBA")],
    "small_upscaled": [(2, 2, 67, "RGB")],
    "large_tiled": [(64, 32, 71, "RGB")],
    "tiled_thumbnail": [(48, 32, 79, "RGB")],
    "multiple_images": [(8, 8, 83, "RGB"), (12, 4, 89, "RGB")],
}


def _official_processor():
    require_reference_environment()
    try:
        from transformers.models.lfm2_vl.image_processing_lfm2_vl import (
            Lfm2VlImageProcessor,
        )
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError(
            "processor fixture export requires the pinned Transformers/TorchVision oracle"
        ) from exc
    return Lfm2VlImageProcessor


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError("processor fixture export requires pinned torch") from exc
    return torch


def _prompt_oracle():
    require_reference_environment()
    try:
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from transformers.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor
    except ImportError as exc:  # pragma: no cover - manager environment
        raise RuntimeError("prompt fixture export requires the pinned tokenizer oracle") from exc

    vocab = {
        "<unk>": 0,
        "hello": 1,
        "world": 2,
        "<image>": 3,
        "<|image_start|>": 4,
        "<|image_end|>": 5,
        "<|img_thumbnail|>": 6,
        "<|img_row_1_col_1|>": 7,
        "<|img_row_1_col_2|>": 8,
        "<|img_row_1_col_3|>": 9,
        "<|img_row_2_col_1|>": 10,
        "<|img_row_2_col_2|>": 11,
        "<|img_row_2_col_3|>": 12,
        "and": 13,
        "Describe": 14,
        "this": 15,
        "image": 16,
        "turn": 17,
        "one": 18,
        "two": 19,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    marker_tokens = [
        "<image>",
        "<|image_start|>",
        "<|image_end|>",
        "<|img_thumbnail|>",
        "<|img_row_1_col_1|>",
        "<|img_row_1_col_2|>",
        "<|img_row_1_col_3|>",
        "<|img_row_2_col_1|>",
        "<|img_row_2_col_2|>",
        "<|img_row_2_col_3|>",
    ]
    backend.add_special_tokens([AddedToken(token, special=True) for token in marker_tokens])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens[1:]})
    tokenizer.image_token = "<image>"
    tokenizer.image_token_id = 3
    tokenizer.image_start_token = "<|image_start|>"
    tokenizer.image_end_token = "<|image_end|>"
    tokenizer.image_thumbnail_token = "<|img_thumbnail|>"
    return Lfm2VlProcessor, tokenizer


def _find_spans(input_ids: list[int], image_token_id: int, lengths: list[int]) -> list[list[int]]:
    spans = []
    cursor = 0
    for length in lengths:
        while cursor < len(input_ids) and input_ids[cursor] != image_token_id:
            cursor += 1
        end = cursor + length
        if end > len(input_ids) or any(token != image_token_id for token in input_ids[cursor:end]):
            raise RuntimeError("official prompt fixture did not produce expected image spans")
        spans.append([cursor, end])
        cursor = end
    if image_token_id in input_ids[cursor:]:
        raise RuntimeError("official prompt fixture has unexpected image placeholders")
    return spans


def _raw_bytes(width: int, height: int, offset: int, mode: str) -> bytes:
    values = bytearray()
    for row in range(height):
        for col in range(width):
            red = (offset + row * 17 + col * 3) % 256
            green = (offset + row * 5 + col * 19 + 31) % 256
            blue = (offset + row * 23 + col * 7 + 67) % 256
            if mode == "L":
                values.append((red + green + blue) % 256)
            elif mode == "RGBA":
                values.extend((red, green, blue, (offset + row * 11 + col * 13) % 256))
            elif mode == "RGB":
                values.extend((red, green, blue))
            else:
                raise ValueError(f"unsupported fixture image mode {mode!r}")
    return bytes(values)


def _make_image(raw: bytes, width: int, height: int, mode: str):
    from PIL import Image

    return Image.frombytes(mode, (width, height), raw)


def _processor(Lfm2VlImageProcessor: Any):
    return Lfm2VlImageProcessor(**TINY_PROCESSOR_CONFIG, size={"height": 8, "width": 8})


def export(output: Path, *, overwrite: bool) -> Path:
    torch = _torch()
    Lfm2VlImageProcessor = _official_processor()
    processor = _processor(Lfm2VlImageProcessor)
    default_processor = Lfm2VlImageProcessor()
    tensors: dict[str, Any] = {}
    case_metadata: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    case_images: dict[str, list[Any]] = {}

    for case_name, specs in CASES.items():
        images = []
        raw_inputs = []
        for image_index, (width, height, offset, mode) in enumerate(specs):
            raw = _raw_bytes(width, height, offset, mode)
            raw_inputs.append({"width": width, "height": height, "bytes": raw, "mode": mode})
            images.append(_make_image(raw, width, height, mode))
            tensor_name = (
                f"input.{case_name}.{mode.lower()}_u8"
                if len(specs) == 1
                else f"input.{case_name}.{image_index}.{mode.lower()}_u8"
            )
            channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
            shape = (height, width) if channels == 1 else (height, width, channels)
            tensors[tensor_name] = torch.tensor(list(raw), dtype=torch.uint8).reshape(shape)
        case_images[case_name] = images

        output_features = processor(
            images=images,
            return_tensors="pt",
            return_row_col_info=True,
            do_convert_rgb=True,
        )
        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "spatial_shapes",
            "image_rows",
            "image_cols",
            "image_sizes",
        ):
            if key not in output_features:
                raise RuntimeError(f"official processor did not return {key!r} for {case_name}")
            tensors[f"output.{case_name}.{key}"] = output_features[key]
        case_metadata[case_name] = {
            "input_shapes": [
                [item["height"], item["width"]]
                if item["mode"] == "L"
                else [item["height"], item["width"], {"RGB": 3, "RGBA": 4}[item["mode"]]]
                for item in raw_inputs
            ],
            "input_modes": [item["mode"] for item in raw_inputs],
            "output_shapes": {
                key: [int(value) for value in output_features[key].shape]
                for key in (
                    "pixel_values",
                    "pixel_attention_mask",
                    "spatial_shapes",
                    "image_rows",
                    "image_cols",
                    "image_sizes",
                )
            },
            "image_rows": output_features["image_rows"].tolist(),
            "image_cols": output_features["image_cols"].tolist(),
            "image_sizes": output_features["image_sizes"].tolist(),
            "crop_count": int(output_features["pixel_values"].shape[0]),
        }
        source_hashes[case_name] = [
            __import__("hashlib").sha256(item["bytes"]).hexdigest() for item in raw_inputs
        ]

    Lfm2VlProcessor, tokenizer = _prompt_oracle()
    prompt_processor = Lfm2VlProcessor(processor, tokenizer)
    prompt_specs = {
        "image_before_text": ("<image>Describe this image", ["square"]),
        "image_between_text": ("hello <image> world", ["square"]),
        "two_images": ("<image> and <image>", ["square", "wide"]),
        "images_across_turns": ("turn one <image> turn two <image>", ["square", "wide"]),
        "tiled_thumbnail_prompt": ("<image>", ["tiled_thumbnail"]),
    }
    prompt_metadata: dict[str, Any] = {}
    for prompt_name, (text, image_names) in prompt_specs.items():
        prompt_images = [case_images[name][0] for name in image_names]
        processed_images, replacements = prompt_processor._process_images(
            prompt_images,
            use_image_special_tokens=True,
            return_tensors="pt",
            return_row_col_info=True,
            do_convert_rgb=True,
        )
        expanded_text, _ = prompt_processor.get_text_with_replacements([text], replacements)
        result = prompt_processor(
            images=prompt_images,
            text=[text],
            return_tensors="pt",
            use_image_special_tokens=True,
            truncation=False,
            do_convert_rgb=True,
        )
        input_ids = result["input_ids"][0].detach().cpu()
        input_id_values = [int(value) for value in input_ids.tolist()]
        crop_lengths: list[int] = []
        image_info = []
        for image_index in range(len(prompt_images)):
            rows = int(processed_images["image_rows"][image_index])
            cols = int(processed_images["image_cols"][image_index])
            image_size = [int(value) for value in processed_images["image_sizes"][image_index]]
            tokens_per_tile, tokens_for_image = prompt_processor._get_image_num_tokens(image_size)
            if rows > 1 or cols > 1:
                crop_lengths.extend([int(tokens_per_tile)] * (rows * cols))
                if processor.use_thumbnail:
                    crop_lengths.append(int(tokens_for_image))
            else:
                crop_lengths.append(int(tokens_for_image))
            image_info.append({
                "rows": rows,
                "cols": cols,
                "image_size": image_size,
            })
        spans = _find_spans(input_id_values, 3, crop_lengths)
        tensor_name = f"prompt.{prompt_name}.input_ids"
        tensors[tensor_name] = input_ids
        prompt_metadata[prompt_name] = {
            "text": text,
            "expanded_text": expanded_text[0],
            "input_ids": input_id_values,
            "image_token_id": 3,
            "images": image_info,
            "crop_lengths": crop_lengths,
            "image_spans": spans,
        }

    real_dimensions = {}
    real_cases = {
        "256x256": (256, 256),
        "277x512": (277, 512),
        "512x277": (512, 277),
        "512x384": (512, 384),
        "384x512": (384, 512),
        "512x512": (512, 512),
        "128x512": (128, 512),
        "512x128": (512, 128),
        "1000x3000": (1000, 3000),
        "3000x1000": (3000, 1000),
    }
    for name, (width, height) in real_cases.items():
        smart_width, smart_height = default_processor.smart_resize(
            height,
            width,
            default_processor.downsample_factor,
            default_processor.min_image_tokens,
            default_processor.max_image_tokens,
            default_processor.encoder_patch_size,
        )
        too_large = default_processor._is_image_too_large(
            height=height,
            width=width,
            max_image_tokens=default_processor.max_image_tokens,
            encoder_patch_size=default_processor.encoder_patch_size,
            downsample_factor=default_processor.downsample_factor,
            max_pixels_tolerance=default_processor.max_pixels_tolerance,
        )
        if too_large:
            grid_width, grid_height, target_width, target_height, _ = default_processor._get_grid_layout(
                height,
                width,
                default_processor.min_tiles,
                default_processor.max_tiles,
                default_processor.tile_size,
            )
            crop_order = [
                {"kind": "tile", "row": row, "col": col}
                for row in range(grid_height)
                for col in range(grid_width)
            ]
            if default_processor.use_thumbnail and grid_width * grid_height != 1:
                crop_order.append({"kind": "thumbnail"})
            selected_grid = [grid_width, grid_height]
            tile_canvas = [target_width, target_height]
        else:
            crop_order = [{"kind": "whole"}]
            selected_grid = [1, 1]
            tile_canvas = None
        real_dimensions[name] = {
            "input_width": width,
            "input_height": height,
            "smart_width": int(smart_width),
            "smart_height": int(smart_height),
            "too_large": bool(too_large),
            "selected_grid": selected_grid,
            "tile_canvas": tile_canvas,
            "crop_order": crop_order,
        }
    if real_dimensions["3000x1000"]["smart_width"] != 864 or real_dimensions["3000x1000"]["smart_height"] != 288:
        raise RuntimeError("official 3000x1000 smart-resize oracle changed")

    metadata = {
        "schema_version": 1,
        "mode": "processor-fixture",
        "oracle": {
            "transformers_revision": "fd12552d770f745fdbe41031ff4daa688f5ed57e",
            "processor_class": "Lfm2VlImageProcessor",
            "processor_module": "transformers.models.lfm2_vl.image_processing_lfm2_vl",
        },
        "package_pins": REFERENCE_PACKAGE_PINS,
        "package_versions": package_versions(),
        "device": "cpu",
        "dtype": "float32",
        "processor_config": TINY_PROCESSOR_CONFIG,
        "cases": case_metadata,
        "source_image_sha256": source_hashes,
        "real_dimension_oracles": real_dimensions,
        "prompt_cases": prompt_metadata,
    }
    manifest = {
        "schema_version": 1,
        "mode": "processor-fixture",
        "source": "official-transformers-lfm2-vl-image-processor",
        "transformers_revision": "fd12552d770f745fdbe41031ff4daa688f5ed57e",
    }
    return write_tensor_bundle(output, tensors, metadata, manifest, overwrite=overwrite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="fixture directory (default: tests/fixtures/lfm2_vl_processor_tiny)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    output = args.output or root / "tests" / "fixtures" / "lfm2_vl_processor_tiny"
    path = export(output, overwrite=args.overwrite)
    print(json.dumps({"output": str(path), "cases": sorted(CASES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
