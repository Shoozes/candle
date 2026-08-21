"""Config-only inspection for the pinned LFM2.5-VL checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        load_reference_lock,
        model_entry,
        normalized_model_summary,
        validate_summary_against_lock,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        load_reference_lock,
        model_entry,
        normalized_model_summary,
        validate_summary_against_lock,
        write_json,
    )


def _read_small_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.suffix.lower() != ".json":
        raise ValueError(f"config-only accepts JSON files, not {path.name!r}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"refusing unusually large config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config JSON must contain an object: {path}")
    return value


_FIXED_IMAGE_MARKERS = {
    "image": "<image>",
    "image_start": "<|image_start|>",
    "image_end": "<|image_end|>",
    "thumbnail": "<|img_thumbnail|>",
}
_TILE_MARKER = re.compile(r"^<\|img_row_([1-9][0-9]*)_col_([1-9][0-9]*)\|>$")


def _token_id_map(tokenizer: dict[str, Any]) -> dict[str, int]:
    """Read token IDs from a tokenizer JSON without importing tokenizers."""

    token_ids: dict[str, int] = {}

    def add(token: Any, token_id: Any, source: str) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError(f"{source} contains an invalid token string")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{source} contains an invalid token ID for {token!r}")
        previous = token_ids.get(token)
        if previous is not None and previous != token_id:
            raise ValueError(
                f"tokenizer assigns conflicting IDs {previous} and {token_id} to {token!r}"
            )
        token_ids[token] = token_id

    model = tokenizer.get("model")
    if model is not None and not isinstance(model, dict):
        raise ValueError("tokenizer model must be an object")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        for token, token_id in vocab.items():
            add(token, token_id, "tokenizer model vocabulary")
    elif isinstance(vocab, list):
        for token_id, entry in enumerate(vocab):
            token = entry[0] if isinstance(entry, list) and entry else entry
            add(token, token_id, "tokenizer model vocabulary")
    elif vocab is not None:
        raise ValueError("tokenizer model vocabulary must be an object or array")

    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens must be an array")
    for index, entry in enumerate(added_tokens):
        if not isinstance(entry, dict):
            raise ValueError(f"tokenizer added token {index} must be an object")
        add(entry.get("content"), entry.get("id"), f"tokenizer added token {index}")
    return token_ids


def _image_marker_summary(
    tokenizer: dict[str, Any], *, expected_image_token_id: int, vocabulary_size: int
) -> dict[str, Any]:
    token_ids = _token_id_map(tokenizer)
    if vocabulary_size <= 0:
        raise ValueError("model vocabulary size must be positive")
    out_of_range = sorted(
        (token, token_id)
        for token, token_id in token_ids.items()
        if token_id >= vocabulary_size
    )
    if out_of_range:
        token, token_id = out_of_range[0]
        raise ValueError(
            f"tokenizer ID {token_id} for {token!r} is outside model vocabulary "
            f"size {vocabulary_size}"
        )
    missing = [token for token in _FIXED_IMAGE_MARKERS.values() if token not in token_ids]
    if missing:
        raise ValueError(f"tokenizer is missing required image markers: {', '.join(missing)}")
    image_token_id = token_ids[_FIXED_IMAGE_MARKERS["image"]]
    if image_token_id != expected_image_token_id:
        raise ValueError(
            "tokenizer image token ID "
            f"{image_token_id} does not match model ID {expected_image_token_id}"
        )

    row_column = []
    for token, token_id in token_ids.items():
        match = _TILE_MARKER.fullmatch(token)
        if match is None:
            continue
        row_column.append(
            {
                "row": int(match.group(1)),
                "column": int(match.group(2)),
                "token": token,
                "id": token_id,
            }
        )
    row_column.sort(key=lambda item: (item["row"], item["column"], item["id"]))
    if not row_column:
        raise ValueError("tokenizer is missing required image row/column markers")

    selected_markers = [
        (token, token_ids[token]) for token in _FIXED_IMAGE_MARKERS.values()
    ] + [(item["token"], item["id"]) for item in row_column]
    marker_by_id: dict[int, str] = {}
    for token, token_id in selected_markers:
        previous = marker_by_id.get(token_id)
        if previous is not None and previous != token:
            raise ValueError(
                f"image markers {previous!r} and {token!r} share token ID {token_id}"
            )
        marker_by_id[token_id] = token
    return {
        "source": "local-tokenizer-json",
        "fixed": {
            name: {"token": token, "id": token_ids[token]}
            for name, token in _FIXED_IMAGE_MARKERS.items()
        },
        "row_column": row_column,
        "row_column_count": len(row_column),
    }


def inspect_config(
    *,
    model: str = "450m",
    config_path: Path | None = None,
    processor_config_path: Path | None = None,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Read the lock and optional small JSON files, never a model weight file."""

    lock = load_reference_lock()
    entry = model_entry(lock, model)
    config = _read_small_json(config_path) if config_path else None
    processor = _read_small_json(processor_config_path) if processor_config_path else None
    tokenizer = _read_small_json(tokenizer_path) if tokenizer_path else None
    summary = normalized_model_summary(
        entry,
        config=config,
        processor=processor,
        source=(
            "local-small-json"
            if config_path or processor_config_path or tokenizer_path
            else "reference-lock"
        ),
    )
    if config_path or processor_config_path:
        validate_summary_against_lock(entry, summary)
        summary["locked_values_validated"] = True
    if tokenizer is not None:
        summary["image_marker_tokens"] = _image_marker_summary(
            tokenizer,
            expected_image_token_id=int(summary["image_token_id"]),
            vocabulary_size=int(summary["text"]["vocab_size"]),
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="450m",
        help="pinned model alias (450m, 1.6b, or 3b) or full LiquidAI model ID",
    )
    parser.add_argument("--config", type=Path, help="optional local config.json")
    parser.add_argument(
        "--processor-config",
        type=Path,
        help="optional local processor_config.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="optional local tokenizer.json for image-marker ID inspection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output file; stdout is always stable JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --output file explicitly",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_config(
        model=args.model,
        config_path=args.config,
        processor_config_path=args.processor_config,
        tokenizer_path=args.tokenizer,
    )
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    print(payload)
    if args.output:
        write_json(args.output, summary, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
