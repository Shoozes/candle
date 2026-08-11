"""Acquire one pinned LFM2-VL snapshot through an explicit, atomic owner action.

The command downloads only the direct filenames and immutable revision recorded
in ``reference-lock.json``. Files enter a caller-owned Hugging Face cache first,
then stream into an external staging directory while their locked source
identity and SHA-256 are verified. The complete regular-file snapshot is
published by one directory rename and is never loaded as a model here. The
download-only path requires the pinned Hugging Face Hub package, not the full
Torch/Transformers oracle environment.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .inspect_artifact import (
        MAX_FILE_BYTES,
        _safe_filename,
        build_artifact_manifest,
        write_artifact_manifest,
    )
    from .manifest import (
        REFERENCE_SHARED_PACKAGE_PINS,
        assert_secret_safe,
        load_reference_lock,
        model_entry,
        repo_root,
    )
except ImportError:  # pragma: no cover - direct script execution
    from inspect_artifact import (  # type: ignore
        MAX_FILE_BYTES,
        _safe_filename,
        build_artifact_manifest,
        write_artifact_manifest,
    )
    from manifest import (  # type: ignore
        REFERENCE_SHARED_PACKAGE_PINS,
        assert_secret_safe,
        load_reference_lock,
        model_entry,
        repo_root,
    )


ACQUISITION_FORMAT = "lfm2-vl-snapshot-acquisition"
ACQUISITION_SCHEMA_VERSION = 2
CHUNK_BYTES = 1024 * 1024
FORBIDDEN_PATH_PARTS = frozenset({".git", ".tools", ".secrets"})
HUB_DISABLE_XET_ENV = "HF_HUB_DISABLE_XET"
TRANSFER_POLICY = "serial-files-resumable-http-xet-disabled"
NETWORK_POLICY_DISABLED = "disabled"
NETWORK_POLICY_CACHE_AWARE = "permitted-cache-aware"

Downloader = Callable[..., str | Path]


def _is_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _pinned_names(entry: Mapping[str, Any]) -> tuple[list[str], str]:
    raw_files = entry.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("pinned model files must be a non-empty list")
    names = [
        _safe_filename(
            item.get("path") if isinstance(item, Mapping) else None,
            f"pinned model filename at index {index}",
        )
        for index, item in enumerate(raw_files)
    ]
    header = entry.get("safetensors_header")
    weight_name = _safe_filename(
        header.get("file") if isinstance(header, Mapping) else None,
        "pinned safetensors filename",
    )
    names.append(weight_name)
    if len(names) != len(set(names)):
        raise ValueError("pinned acquisition filenames must be unique")
    return names, weight_name


def acquisition_contract(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the checked-in acquisition contract."""

    raw = entry.get("acquisition")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("model has no supported acquisition contract")
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("acquisition files must be a non-empty list")

    pinned_names, weight_name = _pinned_names(entry)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    for index, raw_record in enumerate(raw_files):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"acquisition file entry {index} must be an object")
        name = _safe_filename(raw_record.get("path"), f"acquisition path at index {index}")
        if name in seen:
            raise ValueError(f"acquisition files contain duplicate path: {name}")
        seen.add(name)
        byte_count = raw_record.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError(f"acquisition bytes must be a positive integer for {name}")
        if byte_count > MAX_FILE_BYTES:
            raise ValueError(f"acquisition file exceeds {MAX_FILE_BYTES} bytes: {name}")
        identity = raw_record.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"acquisition identity must be an object for {name}")
        kind = identity.get("kind")
        value = identity.get("value")
        if kind == "git-blob-sha1":
            if not _is_hex(value, 40):
                raise ValueError(f"invalid git-blob-sha1 identity for {name}")
        elif kind == "sha256":
            if not _is_hex(value, 64):
                raise ValueError(f"invalid sha256 identity for {name}")
        else:
            raise ValueError(f"unsupported acquisition identity kind for {name}: {kind!r}")
        records.append(
            {
                "path": name,
                "bytes": byte_count,
                "identity": {"kind": kind, "value": value},
            }
        )
        total_bytes += byte_count

    contract_names = [record["path"] for record in records]
    if contract_names != pinned_names:
        raise ValueError(
            "acquisition files must exactly match pinned metadata files followed by the weight file"
        )
    if records[-1]["path"] != weight_name:
        raise ValueError("the safetensors weight must be acquired last")
    declared_total = raw.get("snapshot_bytes")
    if declared_total != total_bytes:
        raise ValueError(
            f"acquisition snapshot_bytes mismatch: expected {total_bytes}, found {declared_total!r}"
        )
    minimum_free = raw.get("minimum_free_bytes")
    if (
        not isinstance(minimum_free, int)
        or isinstance(minimum_free, bool)
        or minimum_free < total_bytes
    ):
        raise ValueError("acquisition minimum_free_bytes must cover the complete snapshot")
    if raw.get("token_policy") != "public-no-token":
        raise ValueError("acquisition token_policy must be public-no-token")
    return {
        "schema_version": 1,
        "files": records,
        "snapshot_bytes": total_bytes,
        "minimum_free_bytes": minimum_free,
        "token_policy": "public-no-token",
    }


def _external_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    absolute = Path(os.path.abspath(expanded))
    for ancestor in (absolute, *absolute.parents):
        if not ancestor.exists():
            continue
        metadata = os.lstat(ancestor)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        if ancestor.is_symlink() or file_attributes & reparse_flag:
            raise ValueError(f"{label} must not cross a link or reparse point: {ancestor}")
    resolved = absolute.resolve(strict=False)
    root = repo_root()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise PermissionError(f"{label} must be outside the repository: {resolved}")
    if resolved.parent == resolved:
        raise ValueError(f"{label} must not be a filesystem root: {resolved}")
    if any(part.lower() in FORBIDDEN_PATH_PARTS for part in resolved.parts):
        raise PermissionError(f"{label} must not use a Git, tool, or secret directory")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_existing_parent(path: Path, label: str) -> Path:
    parent = path.parent
    if parent.is_symlink():
        raise ValueError(f"{label} parent must not be a symlink: {parent}")
    try:
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} parent does not exist: {parent}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} parent is not a directory: {resolved}")
    return resolved


def _refuse_stale_siblings(target: Path, prefix: str, label: str) -> None:
    try:
        candidates = target.parent.iterdir()
        for candidate in candidates:
            if candidate.name.startswith(prefix):
                raise FileExistsError(
                    f"stale {label} path exists; verify no acquisition owner "
                    f"is running, then inspect and remove it before retry: {candidate}"
                )
    except FileExistsError:
        raise
    except OSError as exc:
        raise ValueError(
            f"could not inspect {label} parent for stale paths: {target.parent}"
        ) from exc


def _rename_directory_no_replace(source: Path, destination: Path, label: str) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    if os.name == "nt":
        try:
            os.rename(source, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"{label} appeared during publication and was not replaced: {destination}"
                ) from exc
            raise
        return

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "atomic no-clobber directory publication is supported only on Windows and Linux"
        )

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            "Linux acquisition requires renameat2 for atomic no-clobber publication"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            f"{label} appeared during publication and was not replaced: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _disk_admission(paths: list[Path], minimum_free: int) -> list[dict[str, Any]]:
    volumes: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        probe = path if path.exists() else path.parent
        probe = probe.resolve(strict=True)
        device = int(os.stat(probe).st_dev)
        drive = probe.drive.lower()
        key = (device, drive)
        if key in volumes:
            continue
        usage = shutil.disk_usage(probe)
        if usage.free < minimum_free:
            raise RuntimeError(
                f"disk admission failed at {probe}: {usage.free} bytes free, "
                f"{minimum_free} required"
            )
        volumes[key] = {
            "probe_path": str(probe),
            "free_bytes": usage.free,
            "minimum_free_bytes": minimum_free,
        }
    return list(volumes.values())


def build_acquisition_plan(
    model: str,
    output_dir: Path,
    cache_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate paths, lock facts, and disk headroom without network or writes."""

    lock = load_reference_lock()
    entry = model_entry(lock, model)
    contract = acquisition_contract(entry)
    output = _external_path(output_dir, "snapshot output")
    cache = _external_path(cache_dir, "download cache")
    manifest = _external_path(manifest_path, "acquisition manifest")
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "snapshot output already exists and is not automatically trusted; "
            f"inspect its matching manifest and hashes before load or retry: {output}"
        )
    if manifest.exists() or manifest.is_symlink():
        raise FileExistsError(
            "acquisition manifest already exists and blocks a new publication; "
            f"inspect it with the final snapshot before retry: {manifest}"
        )
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError(f"download cache must be a directory, not a link or file: {cache}")
    _require_existing_parent(output, "snapshot output")
    _require_existing_parent(cache, "download cache")
    _require_existing_parent(manifest, "acquisition manifest")
    _refuse_stale_siblings(
        output,
        f".{output.name}.partial-",
        "acquisition staging",
    )
    _refuse_stale_siblings(
        manifest,
        f".{manifest.name}.tmp-",
        "acquisition manifest staging",
    )
    if output == cache or _contains(output, cache) or _contains(cache, output):
        raise ValueError("snapshot output and download cache must be separate, non-nested paths")
    if _contains(output, manifest) or _contains(cache, manifest):
        raise ValueError("acquisition manifest must be outside the snapshot and download cache")
    disk = _disk_admission([output, cache], contract["minimum_free_bytes"])
    plan = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "format": ACQUISITION_FORMAT,
        "model_id": entry["id"],
        "revision": entry["revision"],
        "snapshot_path": str(output),
        "cache_path": str(cache),
        "manifest_path": str(manifest),
        "files": contract["files"],
        "file_count": len(contract["files"]),
        "snapshot_bytes": contract["snapshot_bytes"],
        "minimum_free_bytes": contract["minimum_free_bytes"],
        "disk": disk,
        "network_policy": NETWORK_POLICY_DISABLED,
        "network_used": False,
        "model_loaded": False,
        "token_policy": contract["token_policy"],
        "transfer_policy": TRANSFER_POLICY,
    }
    assert_secret_safe(plan)
    return plan


def _copy_verified(
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    cache_root = cache_root.resolve(strict=True)
    if not _contains(cache_root, source):
        raise ValueError(
            f"downloaded source escaped the caller-owned cache: {expected['path']}"
        )
    if not source.is_file():
        raise ValueError(f"downloaded cache entry is not a regular file: {source}")
    expected_bytes = int(expected["bytes"])
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {expected_bytes}\0".encode("ascii"))
    total = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"downloaded cache entry is not regular: {source}")
            if before.st_size != expected_bytes:
                raise ValueError(
                    f"downloaded size mismatch for {expected['path']}: "
                    f"expected {expected_bytes}, found {before.st_size}"
                )
            while True:
                chunk = reader.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise ValueError(f"downloaded file grew while copying: {expected['path']}")
                sha256.update(chunk)
                git_blob.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
    except OSError as exc:
        raise ValueError(f"could not copy downloaded file {expected['path']}") from exc
    if (
        total != expected_bytes
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError(f"downloaded file changed while copying: {expected['path']}")
    identity = expected["identity"]
    actual_identity = sha256.hexdigest() if identity["kind"] == "sha256" else git_blob.hexdigest()
    if actual_identity != identity["value"]:
        raise ValueError(
            f"downloaded identity mismatch for {expected['path']}: "
            f"expected {identity['value']}, found {actual_identity}"
        )
    copied = destination.stat()
    if not stat.S_ISREG(copied.st_mode) or copied.st_size != expected_bytes:
        raise ValueError(f"published staging file is not exact: {destination}")
    return {
        "path": expected["path"],
        "bytes": expected_bytes,
        "sha256": sha256.hexdigest(),
        "source_identity": dict(identity),
        "source_identity_verified": True,
        "regular_file": True,
    }


def _load_default_downloader() -> Downloader:
    expected = REFERENCE_SHARED_PACKAGE_PINS["huggingface-hub"]
    try:
        installed = importlib.metadata.version("huggingface-hub")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"snapshot acquisition requires huggingface-hub=={expected}"
        ) from exc
    if installed != expected:
        raise RuntimeError(
            "snapshot acquisition package mismatch; "
            f"expected huggingface-hub=={expected}, installed {installed}"
        )
    os.environ[HUB_DISABLE_XET_ENV] = "1"
    try:
        from huggingface_hub import constants as hub_constants
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            f"snapshot acquisition requires huggingface-hub=={expected}"
        ) from exc
    if not hub_constants.HF_HUB_DISABLE_XET:
        raise RuntimeError(
            "snapshot acquisition requires Xet to be disabled before Hub import; "
            "run this command in a fresh Python process"
        )
    return hf_hub_download


def _validate_artifact_records(
    copied: list[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> None:
    copied_by_name = {record["path"]: record for record in copied}
    artifact_files = artifact.get("files")
    if not isinstance(artifact_files, list):
        raise ValueError("artifact verifier returned no file inventory")
    artifact_by_name: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(artifact_files):
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError(f"artifact verifier file entry {index} is malformed")
        name = record["path"]
        if name in artifact_by_name:
            raise ValueError(f"artifact verifier returned duplicate file: {name}")
        artifact_by_name[name] = record
    if set(artifact_by_name) != set(copied_by_name):
        raise ValueError("artifact verifier file inventory differs from acquired files")
    for name, record in copied_by_name.items():
        verified = artifact_by_name[name]
        if verified.get("bytes") != record["bytes"] or verified.get("sha256") != record["sha256"]:
            raise ValueError(f"artifact verifier disagrees with acquired file: {name}")


def acquire_snapshot(
    model: str,
    output_dir: Path,
    cache_dir: Path,
    manifest_path: Path,
    *,
    allow_production_download: bool,
) -> dict[str, Any]:
    """Acquire, verify, and atomically publish one pinned regular snapshot."""

    if not allow_production_download:
        raise PermissionError(
            "snapshot download is disabled; pass --allow-production-download explicitly"
        )
    plan = build_acquisition_plan(model, output_dir, cache_dir, manifest_path)
    download = _load_default_downloader()
    cache = Path(plan["cache_path"])
    cache.mkdir(exist_ok=True)
    plan = build_acquisition_plan(
        model,
        Path(plan["snapshot_path"]),
        cache,
        Path(plan["manifest_path"]),
    )
    cache = Path(plan["cache_path"])
    output = Path(plan["snapshot_path"])
    manifest = Path(plan["manifest_path"])
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.partial-{os.getpid()}-",
            dir=str(output.parent),
        )
    )
    copied: list[dict[str, Any]] = []
    published = False
    try:
        for expected in plan["files"]:
            try:
                source = download(
                    repo_id=plan["model_id"],
                    filename=expected["path"],
                    repo_type="model",
                    revision=plan["revision"],
                    cache_dir=str(cache),
                    token=False,
                    local_files_only=False,
                    force_download=False,
                    library_name="candle-lfm2-vl-reference",
                )
            except Exception as exc:  # downloader implementations expose varied network exceptions
                raise RuntimeError(
                    f"download failed for {expected['path']}: {type(exc).__name__}"
                ) from None
            copied.append(
                _copy_verified(
                    Path(source),
                    stage / expected["path"],
                    expected,
                    cache,
                )
            )
        artifact = build_artifact_manifest(model, stage)
        _validate_artifact_records(copied, artifact)
        result = {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "format": ACQUISITION_FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": plan["model_id"],
            "revision": plan["revision"],
            "snapshot_path": str(output),
            "cache_path": str(cache),
            "manifest_path": str(manifest),
            "files": copied,
            "file_count": len(copied),
            "total_bytes": sum(record["bytes"] for record in copied),
            "artifact_manifest": artifact,
            "token_policy": "public-no-token",
            "transfer_policy": TRANSFER_POLICY,
            "network_policy": NETWORK_POLICY_CACHE_AWARE,
            "network_used": None,
            "model_loaded": False,
            "snapshot_published_atomically": True,
        }
        assert_secret_safe(result)
        _rename_directory_no_replace(stage, output, "snapshot output")
        published = True
        try:
            write_artifact_manifest(manifest, result, overwrite=False)
        except Exception:
            try:
                _rename_directory_no_replace(
                    output,
                    stage,
                    "snapshot rollback staging",
                )
            except OSError as rollback_error:
                raise RuntimeError(
                    "acquisition manifest publication and snapshot rollback both failed; "
                    f"do not load the snapshot remaining at {output}"
                ) from rollback_error
            published = False
            raise
        return result
    finally:
        if not published and stage.exists():
            try:
                shutil.rmtree(stage)
            except OSError as cleanup_error:
                raise RuntimeError(
                    "acquisition failed and staging cleanup also failed; verify no owner "
                    f"is running, then inspect and remove before retry: {stage}"
                ) from cleanup_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="1.6b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="validate without network or writes")
    action.add_argument(
        "--allow-production-download",
        action="store_true",
        help="explicitly authorize the pinned production snapshot download",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.plan:
            result = build_acquisition_plan(
                args.model, args.output_dir, args.cache_dir, args.manifest
            )
        else:
            result = acquire_snapshot(
                args.model,
                args.output_dir,
                args.cache_dir,
                args.manifest,
                allow_production_download=args.allow_production_download,
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (FileExistsError, ImportError, OSError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"snapshot acquisition error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "acquire_snapshot",
    "acquisition_contract",
    "build_acquisition_plan",
    "build_parser",
    "main",
]
