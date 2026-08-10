"""Config-only inspection for the pinned LFM2.5-VL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        load_reference_lock,
        model_entry,
        normalized_model_summary,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        load_reference_lock,
        model_entry,
        normalized_model_summary,
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


def inspect_config(
    *,
    model: str = "450m",
    config_path: Path | None = None,
    processor_config_path: Path | None = None,
) -> dict[str, Any]:
    """Read the lock and optional small JSON files, never a model weight file."""

    lock = load_reference_lock()
    entry = model_entry(lock, model)
    config = _read_small_json(config_path) if config_path else None
    processor = _read_small_json(processor_config_path) if processor_config_path else None
    return normalized_model_summary(
        entry,
        config=config,
        processor=processor,
        source="local-small-json" if config_path or processor_config_path else "reference-lock",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="450m",
        help="pinned model alias (450m or 1.6b) or full LiquidAI model ID",
    )
    parser.add_argument("--config", type=Path, help="optional local config.json")
    parser.add_argument(
        "--processor-config",
        type=Path,
        help="optional local processor_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output file; stdout is always stable JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_config(
        model=args.model,
        config_path=args.config,
        processor_config_path=args.processor_config,
    )
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    print(payload)
    if args.output:
        write_json(args.output.resolve(), summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
