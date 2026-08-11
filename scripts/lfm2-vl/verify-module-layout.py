#!/usr/bin/env python3
"""Verify the bounded, same-module source layout for the LFM2-VL overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_RE = re.compile(r'include!\("([^"]+)"\);')


@dataclass(frozen=True)
class Split:
    wrapper: str
    parts: tuple[str, ...]
    wrapper_limit: int = 900
    part_limit: int = 500


SPLITS = (
    Split(
        "candle-transformers/src/models/lfm2.rs",
        (
            "candle-transformers/src/models/lfm2/config.rs",
            "candle-transformers/src/models/lfm2/cache.rs",
            "candle-transformers/src/models/lfm2/layers.rs",
            "candle-transformers/src/models/lfm2/model.rs",
        ),
    ),
    Split(
        "candle-transformers/src/models/siglip2.rs",
        (
            "candle-transformers/src/models/siglip2/config.rs",
            "candle-transformers/src/models/siglip2/embeddings.rs",
            "candle-transformers/src/models/siglip2/encoder.rs",
            "candle-transformers/src/models/siglip2/model.rs",
            "candle-transformers/src/models/siglip2/interpolation.rs",
        ),
    ),
    Split(
        "candle-transformers/src/models/lfm2_vl/gguf.rs",
        (
            "candle-transformers/src/models/lfm2_vl/gguf/types.rs",
            "candle-transformers/src/models/lfm2_vl/gguf/loading.rs",
            "candle-transformers/src/models/lfm2_vl/gguf/metadata.rs",
            "candle-transformers/src/models/lfm2_vl/gguf/inventory.rs",
            "candle-transformers/src/models/lfm2_vl/gguf/metadata_values.rs",
        ),
    ),
    Split(
        "candle-transformers/src/models/lfm2_vl/weights.rs",
        (
            "candle-transformers/src/models/lfm2_vl/weights/manifest.rs",
            "candle-transformers/src/models/lfm2_vl/weights/runtime.rs",
            "candle-transformers/src/models/lfm2_vl/weights/safetensors.rs",
        ),
    ),
    Split(
        "candle-transformers/src/models/lfm2_vl/model.rs",
        (
            "candle-transformers/src/models/lfm2_vl/model/types.rs",
            "candle-transformers/src/models/lfm2_vl/model/runtime.rs",
            "candle-transformers/src/models/lfm2_vl/model/encoding.rs",
            "candle-transformers/src/models/lfm2_vl/model/merge.rs",
            "candle-transformers/src/models/lfm2_vl/model/config_ext.rs",
        ),
    ),
    Split(
        "candle-vlm/src/lfm2_vl/processor.rs",
        (
            "candle-vlm/src/lfm2_vl/processor/types.rs",
            "candle-vlm/src/lfm2_vl/processor/entry.rs",
            "candle-vlm/src/lfm2_vl/processor/budget.rs",
            "candle-vlm/src/lfm2_vl/processor/crops.rs",
            "candle-vlm/src/lfm2_vl/processor/helpers.rs",
        ),
    ),
    Split(
        "candle-vlm/src/lfm2_vl/prompt.rs",
        (
            "candle-vlm/src/lfm2_vl/prompt/types.rs",
            "candle-vlm/src/lfm2_vl/prompt/tokens.rs",
            "candle-vlm/src/lfm2_vl/prompt/expand.rs",
            "candle-vlm/src/lfm2_vl/prompt/validation.rs",
            "candle-vlm/src/lfm2_vl/prompt/image_block.rs",
            "candle-vlm/src/lfm2_vl/prompt/helpers.rs",
        ),
    ),
    Split(
        "candle-examples/examples/lfm2-vl/runner.rs",
        (
            "candle-examples/examples/lfm2-vl/runner/types.rs",
            "candle-examples/examples/lfm2-vl/runner/runtime.rs",
            "candle-examples/examples/lfm2-vl/runner/run.rs",
            "candle-examples/examples/lfm2-vl/runner/generation.rs",
            "candle-examples/examples/lfm2-vl/runner/evidence.rs",
        ),
    ),
    Split(
        "candle-examples/examples/lfm2-vl/native_loading.rs",
        (
            "candle-examples/examples/lfm2-vl/native_loading/types.rs",
            "candle-examples/examples/lfm2-vl/native_loading/load.rs",
            "candle-examples/examples/lfm2-vl/native_loading/inventory.rs",
        ),
    ),
)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    for split in SPLITS:
        wrapper = ROOT / split.wrapper
        if not wrapper.is_file():
            fail(f"missing wrapper {split.wrapper}")
        wrapper_text = wrapper.read_text(encoding="utf-8")
        wrapper_lines = line_count(wrapper)
        if wrapper_lines > split.wrapper_limit:
            fail(
                f"wrapper {split.wrapper} has {wrapper_lines} lines, "
                f"limit is {split.wrapper_limit}"
            )

        declared = {
            str((wrapper.parent / relative).resolve().relative_to(ROOT))
            for relative in INCLUDE_RE.findall(wrapper_text)
        }
        expected = set(split.parts)
        if declared != expected:
            fail(
                f"wrapper {split.wrapper} include set mismatch; "
                f"missing={sorted(expected - declared)}, "
                f"unexpected={sorted(declared - expected)}"
            )

        part_sizes: list[str] = []
        for relative in split.parts:
            part = ROOT / relative
            if not part.is_file():
                fail(f"missing source part {relative}")
            lines = line_count(part)
            if lines > split.part_limit:
                fail(f"source part {relative} has {lines} lines, limit is {split.part_limit}")
            if "mod tests" in part.read_text(encoding="utf-8"):
                fail(f"source part {relative} owns tests; tests must remain in {split.wrapper}")
            part_sizes.append(f"{relative}={lines}")

        print(
            f"module-layout wrapper={split.wrapper} lines={wrapper_lines} "
            + " ".join(part_sizes)
        )

    print("module-layout: passed")


if __name__ == "__main__":
    main()
