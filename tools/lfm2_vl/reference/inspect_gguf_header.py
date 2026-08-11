"""Inspect a bounded GGUF header prefix without reading tensor payloads."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any

try:
    from .manifest import write_bytes_atomic
except ImportError:  # pragma: no cover - direct script execution
    from manifest import write_bytes_atomic  # type: ignore


MAX_PREFIX_BYTES = 4 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024
MAX_ENTRIES = 16_384
MAX_ARRAY_ELEMENTS = 65_536
MAX_VALUE_DEPTH = 16
MAX_TENSOR_DIMS = 4

VALUE_TYPES = {
    0: "u8",
    1: "i8",
    2: "u16",
    3: "i16",
    4: "u32",
    5: "i32",
    6: "f32",
    7: "bool",
    8: "string",
    9: "array",
    10: "u64",
    11: "i64",
    12: "f64",
}

# Numeric values follow ggml_type. Size entries are (elements per block, bytes).
GGML_DTYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}


class GgufHeaderError(ValueError):
    """Raised when a bounded prefix does not contain a valid GGUF header."""


class _Reader:
    def __init__(self, data: bytes | mmap.mmap, *, limit: int | None = None):
        self.data = data
        self.limit = len(data) if limit is None else min(len(data), limit)
        self.offset = 0

    def read(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > self.limit:
            raise GgufHeaderError(
                f"GGUF header prefix ended at byte {self.limit} while reading "
                f"{size} bytes at offset {self.offset}"
            )
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def unpack(self, code: str) -> int | float:
        size = struct.calcsize(code)
        return struct.unpack(code, self.read(size))[0]

    def u8(self) -> int:
        return int(self.unpack("<B"))

    def i8(self) -> int:
        return int(self.unpack("<b"))

    def u16(self) -> int:
        return int(self.unpack("<H"))

    def i16(self) -> int:
        return int(self.unpack("<h"))

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def i32(self) -> int:
        return int(self.unpack("<i"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def i64(self) -> int:
        return int(self.unpack("<q"))

    def f32(self) -> float:
        return float(self.unpack("<f"))

    def f64(self) -> float:
        return float(self.unpack("<d"))


def _read_length(reader: _Reader, version: int) -> int:
    return reader.u32() if version == 1 else reader.u64()


def _read_string(reader: _Reader, version: int) -> str:
    length = _read_length(reader, version)
    if length > MAX_STRING_BYTES:
        raise GgufHeaderError(
            f"GGUF string length {length} exceeds bound {MAX_STRING_BYTES}"
        )
    try:
        return reader.read(length).rstrip(b"\0").decode("utf-8")
    except UnicodeDecodeError as error:
        raise GgufHeaderError("GGUF header contains invalid UTF-8") from error


def _read_value(reader: _Reader, version: int, value_type: int, depth: int) -> Any:
    if depth > MAX_VALUE_DEPTH:
        raise GgufHeaderError(
            f"GGUF metadata nesting exceeds bound {MAX_VALUE_DEPTH}"
        )
    if value_type not in VALUE_TYPES:
        raise GgufHeaderError(f"unknown GGUF metadata value type {value_type}")
    if value_type == 0:
        return reader.u8()
    if value_type == 1:
        return reader.i8()
    if value_type == 2:
        return reader.u16()
    if value_type == 3:
        return reader.i16()
    if value_type == 4:
        return reader.u32()
    if value_type == 5:
        return reader.i32()
    if value_type == 6:
        value = reader.f32()
    elif value_type == 7:
        value = reader.u8()
        if value not in (0, 1):
            raise GgufHeaderError(f"invalid GGUF boolean value {value}")
        return bool(value)
    elif value_type == 8:
        return _read_string(reader, version)
    elif value_type == 9:
        element_type = reader.u32()
        length = _read_length(reader, version)
        if length > MAX_ARRAY_ELEMENTS:
            raise GgufHeaderError(
                f"GGUF array length {length} exceeds bound {MAX_ARRAY_ELEMENTS}"
            )
        return [
            _read_value(reader, version, element_type, depth + 1)
            for _ in range(length)
        ]
    elif value_type == 10:
        return reader.u64()
    elif value_type == 11:
        return reader.i64()
    else:
        value = reader.f64()
    if not math.isfinite(value):
        raise GgufHeaderError("GGUF metadata contains a non-finite float")
    return value


def _alignment(metadata: dict[str, Any]) -> int:
    value = metadata.get("general.alignment", 32)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GgufHeaderError("GGUF general.alignment must be a positive integer")
    return value


def _tensor_nbytes(shape: list[int], dtype_code: int) -> int | None:
    dtype = GGML_DTYPES.get(dtype_code)
    if dtype is None:
        return None
    _, block_size, type_size = dtype
    element_count = math.prod(shape)
    if element_count % block_size:
        raise GgufHeaderError(
            f"tensor with {element_count} elements is not divisible by "
            f"{dtype[0]} block size {block_size}"
        )
    return element_count // block_size * type_size


def _inspect_gguf_data(
    data: bytes | mmap.mmap,
    *,
    source_url: str | None = None,
    source_revision: str | None = None,
    byte_range: str | None = None,
    full_file_bytes: int | None = None,
) -> dict[str, Any]:
    reader = _Reader(
        data,
        limit=MAX_PREFIX_BYTES if full_file_bytes is not None else None,
    )
    if reader.read(4) != b"GGUF":
        raise GgufHeaderError("invalid GGUF magic")
    version = reader.u32()
    if version not in (1, 2, 3):
        raise GgufHeaderError(f"unsupported GGUF version {version}")
    tensor_count = _read_length(reader, version)
    metadata_count = _read_length(reader, version)
    if tensor_count > MAX_ENTRIES or metadata_count > MAX_ENTRIES:
        raise GgufHeaderError(
            f"GGUF counts [{tensor_count}, {metadata_count}] exceed bound {MAX_ENTRIES}"
        )

    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        name = _read_string(reader, version)
        if name in metadata:
            raise GgufHeaderError(f"duplicate GGUF metadata key {name!r}")
        metadata[name] = _read_value(reader, version, reader.u32(), 0)

    tensors: dict[str, dict[str, Any]] = {}
    for _ in range(tensor_count):
        name = _read_string(reader, version)
        if name in tensors:
            raise GgufHeaderError(f"duplicate GGUF tensor name {name!r}")
        dimension_count = reader.u32()
        if dimension_count == 0 or dimension_count > MAX_TENSOR_DIMS:
            raise GgufHeaderError(
                f"GGUF tensor {name!r} dimension count {dimension_count} is invalid"
            )
        dimensions = [
            reader.u32() if version == 1 else reader.u64()
            for _ in range(dimension_count)
        ]
        if any(dimension == 0 for dimension in dimensions):
            raise GgufHeaderError(f"GGUF tensor {name!r} has a zero dimension")
        dtype_code = reader.u32()
        offset = reader.u64()
        shape = list(reversed(dimensions))
        dtype = GGML_DTYPES.get(dtype_code)
        tensors[name] = {
            "dtype": dtype[0] if dtype else f"GGML_TYPE_{dtype_code}",
            "dtype_code": dtype_code,
            "gguf_dimensions": dimensions,
            "shape": shape,
            "nbytes": _tensor_nbytes(shape, dtype_code),
            "relative_offset": offset,
        }

    header_end = reader.offset
    alignment = _alignment(metadata)
    tensor_data_offset = ((header_end + alignment - 1) // alignment) * alignment
    if tensor_data_offset > reader.limit:
        raise GgufHeaderError(
            f"GGUF tensor data offset {tensor_data_offset} exceeds bounded header "
            f"prefix {reader.limit}"
        )
    maximum_payload_end = tensor_data_offset
    all_tensor_sizes_known = True
    for tensor in tensors.values():
        nbytes = tensor["nbytes"]
        if nbytes is not None:
            tensor["absolute_offset"] = tensor_data_offset + tensor["relative_offset"]
            maximum_payload_end = max(
                maximum_payload_end, tensor["absolute_offset"] + nbytes
            )
        else:
            tensor["absolute_offset"] = None
            all_tensor_sizes_known = False

    prefix_bytes = tensor_data_offset if full_file_bytes is not None else len(data)
    result = {
        "format": "gguf-header-prefix",
        "source": {
            "url": source_url,
            "revision": source_revision,
            "byte_range": byte_range,
        },
        "prefix": {
            "bytes": prefix_bytes,
            "sha256": hashlib.sha256(data[:prefix_bytes]).hexdigest(),
            "contains_tensor_payload": prefix_bytes > tensor_data_offset,
            "contains_complete_file": prefix_bytes >= maximum_payload_end,
        },
        "gguf": {
            "version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
            "header_end": header_end,
            "alignment": alignment,
            "tensor_data_offset": tensor_data_offset,
            "declared_file_size": maximum_payload_end,
            "metadata": dict(sorted(metadata.items())),
            "tensors": dict(sorted(tensors.items())),
        },
    }
    if full_file_bytes is not None:
        result["file"] = {
            "bytes": full_file_bytes,
            "all_tensor_sizes_known": all_tensor_sizes_known,
            "matches_declared_file_size": (
                full_file_bytes == maximum_payload_end
                if all_tensor_sizes_known
                else None
            ),
        }
    return result


def inspect_gguf_header(
    path: Path,
    *,
    source_url: str | None = None,
    source_revision: str | None = None,
    byte_range: str | None = None,
    full_file: bool = False,
) -> dict[str, Any]:
    """Return stable metadata from a bounded prefix or local full GGUF file."""

    path = path.resolve()
    size = path.stat().st_size
    if size == 0 or (not full_file and size > MAX_PREFIX_BYTES):
        noun = "GGUF file" if full_file else "GGUF header prefix"
        maximum = "unbounded" if full_file else str(MAX_PREFIX_BYTES)
        raise GgufHeaderError(f"{noun} size {size} is outside 1..{maximum} bytes")
    if not full_file:
        return _inspect_gguf_data(
            path.read_bytes(),
            source_url=source_url,
            source_revision=source_revision,
            byte_range=byte_range,
        )

    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), length=0, access=mmap.ACCESS_READ
    ) as data:
        return _inspect_gguf_data(
            data,
            source_url=source_url,
            source_revision=source_revision,
            byte_range=byte_range,
            full_file_bytes=size,
        )


def summarize_gguf_header(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact integrity summary for a full inspection result."""

    gguf = result["gguf"]
    tensors = gguf["tensors"]
    names = sorted(tensors)
    summary = {
        "format": result["format"],
        "source": result["source"],
        "prefix": result["prefix"],
        "gguf": {
            "version": gguf["version"],
            "metadata_count": gguf["metadata_count"],
            "tensor_count": gguf["tensor_count"],
            "header_end": gguf["header_end"],
            "alignment": gguf["alignment"],
            "tensor_data_offset": gguf["tensor_data_offset"],
            "declared_file_size": gguf["declared_file_size"],
            "dtype_counts": dict(sorted(Counter(t["dtype"] for t in tensors.values()).items())),
            "tensor_names_sha256": hashlib.sha256(
                ("\n".join(names) + "\n").encode("utf-8")
            ).hexdigest(),
        },
    }
    if "file" in result:
        summary["file"] = result["file"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="local bounded GGUF header prefix")
    parser.add_argument("--source-url")
    parser.add_argument("--source-revision")
    parser.add_argument("--byte-range")
    parser.add_argument(
        "--full-file",
        action="store_true",
        help=(
            "inspect a complete local GGUF while reading and hashing only its "
            "bounded header prefix"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="emit counts and hashes without the full metadata/tensor table",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --output file explicitly",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress stdout when --output retains the JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quiet and args.output is None:
        parser.error("--quiet requires --output")
    result = inspect_gguf_header(
        args.path,
        source_url=args.source_url,
        source_revision=args.source_revision,
        byte_range=args.byte_range,
        full_file=args.full_file,
    )
    if args.summary_only:
        result = summarize_gguf_header(result)
    if args.output:
        write_bytes_atomic(
            args.output,
            (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
            overwrite=args.overwrite,
            label="GGUF report output",
        )
    if not args.quiet:
        compact = json.dumps(
            result, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        print(compact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
