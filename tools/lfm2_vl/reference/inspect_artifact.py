"""Create a hash-only manifest for one pinned local LFM2-VL snapshot.

The command is intentionally stdlib-only and local-only.  It records the
locked repository/revision and the required regular-file names, sizes, and
SHA-256 values without copying or serializing model payloads.  Production use
requires an explicit opt-in because hashing a checkpoint is an owner action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import (
        assert_secret_safe,
        canonical_json_bytes,
        load_reference_lock,
        model_entry,
        repo_root,
        write_bytes_atomic,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        assert_secret_safe,
        canonical_json_bytes,
        load_reference_lock,
        model_entry,
        repo_root,
        write_bytes_atomic,
    )


MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MANIFEST_FORMAT = "lfm2-vl-artifact-manifest"
MANIFEST_SCHEMA_VERSION = 1


def _safe_filename(name: Any, label: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty filename")
    if "\x00" in name:
        raise ValueError(f"{label} must not contain a NUL byte")
    if Path(name).is_absolute() or "/" in name or "\\" in name:
        raise ValueError(f"{label} must be a direct filename: {name!r}")
    return name


def _resolve_regular_file(root: Path, name: str, label: str) -> Path:
    name = _safe_filename(name, label)
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a regular file, not a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"missing {label}: {candidate}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the model directory: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} is not a regular file: {path}")
            if before.st_size > MAX_FILE_BYTES:
                raise ValueError(
                    f"{label} is {before.st_size} bytes; maximum is {MAX_FILE_BYTES}"
                )
            total = 0
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError(
                        f"{label} grew beyond the {MAX_FILE_BYTES}-byte limit while hashing"
                    )
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"could not hash {label}: {path}") from exc
    if after.st_size != before.st_size or total != before.st_size:
        raise ValueError(f"{label} changed while it was hashed: {path}")
    return total, digest.hexdigest()


def _read_index_shards(root: Path, index_name: str) -> list[str]:
    index_path = _resolve_regular_file(root, index_name, "safetensors index")
    size = index_path.stat().st_size
    if size > MAX_INDEX_BYTES:
        raise ValueError(f"safetensors index exceeds {MAX_INDEX_BYTES} bytes: {index_path}")
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse safetensors index: {index_path}") from exc
    weight_map = value.get("weight_map") if isinstance(value, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"safetensors index has no weight_map: {index_path}")
    for tensor_name in weight_map:
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"safetensors index has an invalid tensor name: {tensor_name!r}")
    shard_names = {
        _safe_filename(shard, "safetensors shard name")
        for shard in weight_map.values()
    }
    return [index_name, *sorted(shard_names)]


def _pinned_file_names(entry: Mapping[str, Any]) -> list[str]:
    raw_files = entry.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("pinned model files must be a non-empty list")
    names: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise ValueError(f"pinned model file entry {index} must be an object")
        name = _safe_filename(item.get("path"), f"pinned model filename at index {index}")
        if name in seen:
            raise ValueError(f"pinned model files contain duplicate path: {name}")
        seen.add(name)
        names.append(name)
    return names


def _required_files(entry: Mapping[str, Any], root: Path) -> tuple[list[str], set[str]]:
    names = set(_pinned_file_names(entry))
    header = entry.get("safetensors_header", {})
    weight_name = _safe_filename(
        header.get("file") if isinstance(header, Mapping) else None,
        "pinned safetensors filename",
    )
    index_name = "model.safetensors.index.json"
    has_weight = (root / weight_name).exists()
    has_index = (root / index_name).exists()
    if has_weight and has_index:
        raise ValueError(
            f"model snapshot contains both {weight_name} and {index_name}; refusing ambiguity"
        )
    weight_names: set[str]
    if has_weight:
        names.add(weight_name)
        weight_names = {weight_name}
    elif has_index:
        indexed = _read_index_shards(root, index_name)
        names.update(indexed)
        weight_names = set(indexed[1:])
    else:
        raise ValueError(f"model snapshot has neither {weight_name} nor {index_name}")
    return sorted(names), weight_names


def _validate_snapshot_code_inventory(entry: Mapping[str, Any], root: Path) -> None:
    """Reject direct model Python files that are not explicitly hash-locked."""

    policy = entry.get("remote_code_policy", {})
    locked_code = {
        name
        for name in policy.get("files", [])
        if isinstance(name, str)
    } if isinstance(policy, Mapping) else set()
    try:
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() != ".py":
                continue
            relative_name = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file() or relative_name not in locked_code:
                raise ValueError(
                    "model snapshot contains unlisted Python code; refusing "
                    f"cache-only or moving code: {relative_name}"
                )
    except OSError as exc:
        raise ValueError(f"could not inspect model snapshot code inventory: {root}") from exc


def build_artifact_manifest(model: str, model_dir: Path) -> dict[str, Any]:
    """Hash the pinned local files required by ``model`` without loading them."""

    root = model_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"model directory is not a directory: {root}")
    repository_root = repo_root()
    try:
        root.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise PermissionError(f"production model directory must be outside the repository: {root}")

    lock = load_reference_lock()
    entry = model_entry(lock, model)
    _validate_snapshot_code_inventory(entry, root)
    names, weight_names = _required_files(entry, root)
    purposes = {
        str(item["path"]): str(item.get("purpose", ""))
        for item in entry.get("files", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for name in names:
        path = _resolve_regular_file(root, name, f"pinned model file {name}")
        byte_count, digest = _hash_regular_file(path, f"pinned model file {name}")
        total_bytes += byte_count
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError(
                f"pinned model files exceed the {MAX_TOTAL_BYTES}-byte aggregate limit"
            )
        records.append(
            {
                "path": name,
                "purpose": purposes.get(
                    name,
                    (
                        "safetensors index"
                        if name.endswith(".index.json")
                        else "safetensors weight shard"
                        if name in weight_names
                        else ""
                    ),
                ),
                "bytes": byte_count,
                "sha256": digest,
                "regular_file": True,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": MANIFEST_FORMAT,
        "model_id": entry["id"],
        "repository": entry["repository"],
        "revision": entry["revision"],
        "revision_url": entry["revision_url"],
        "files": records,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "weights_hashed_not_serialized": bool(weight_names),
    }
    assert_secret_safe(manifest)
    return manifest


def resolve_model_snapshot(model: str, model_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve and hash one external regular-file snapshot before model loading."""

    try:
        root = model_dir.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"model snapshot is not readable: {model_dir}") from exc
    if not root.is_dir():
        raise ValueError(f"model snapshot is not a directory: {root}")
    return root, build_artifact_manifest(model, root)


def verify_artifact_unchanged(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    """Reject model input changes across a bounded load or inference operation."""

    if final != initial:
        raise RuntimeError(f"production model snapshot changed during {operation}")


def write_artifact_manifest(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> Path:
    """Write one small manifest atomically outside the repository."""

    path = Path(os.path.abspath(path.expanduser()))
    resolved_parent = path.parent.resolve(strict=False)
    path = resolved_parent / path.name
    root = repo_root()
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PermissionError(f"artifact manifest output must be outside the repository: {path}")
    return write_bytes_atomic(
        path,
        canonical_json_bytes(value),
        overwrite=overwrite,
        label="artifact manifest",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="450m",
        help="pinned model alias (450m, 1.6b, or 3b) or full LiquidAI model ID",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="explicitly authorize hashing a local production checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.allow_production:
            raise PermissionError(
                "artifact inspection is disabled; pass --allow-production explicitly"
            )
        manifest = build_artifact_manifest(args.model, args.model_dir)
        output = write_artifact_manifest(args.output, manifest, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "format": MANIFEST_FORMAT,
                    "output": str(output),
                    "model_id": manifest["model_id"],
                    "revision": manifest["revision"],
                    "file_count": manifest["file_count"],
                    "total_bytes": manifest["total_bytes"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (FileExistsError, ImportError, OSError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"artifact inspection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "build_artifact_manifest",
    "build_parser",
    "main",
    "resolve_model_snapshot",
    "verify_artifact_unchanged",
    "write_artifact_manifest",
]
