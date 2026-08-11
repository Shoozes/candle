"""Safetensors bundle writing and integrity validation.

The module imports PyTorch and safetensors only inside the functions that need
them.  Config-only inspection therefore remains usable on the manager's bare
Python 3.10 installation.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import (
        assert_secret_safe,
        prepare_output_dir,
        resolve_regular_file,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        assert_secret_safe,
        prepare_output_dir,
        resolve_regular_file,
        sha256_file,
        write_json,
    )


TENSOR_FILE = "tensors.safetensors"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"


def _torch_and_safetensors():
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised in manager environment
        raise RuntimeError(
            "tiny-random export requires the pinned torch and safetensors packages; "
            "config-only does not"
        ) from exc
    return torch, save_file


def tensor_inventory(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return stable shape/dtype metadata without serializing tensor values."""

    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        tensor = tensors[name]
        shape = [int(value) for value in tensor.shape]
        dtype = str(tensor.dtype).split(".")[-1]
        inventory[name] = {"dtype": dtype, "shape": shape}
    return inventory


def _ordered_cpu_tensors(tensors: Mapping[str, Any]) -> OrderedDict[str, Any]:
    torch, _ = _torch_and_safetensors()
    ordered: OrderedDict[str, Any] = OrderedDict()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor value for {name!r} is not a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"tensor {name!r} is not strided")
        # Clone every value so tied lm_head/embed aliases cannot trigger the
        # safetensors shared-storage rejection.
        ordered[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return ordered


def write_tensor_bundle(
    output_dir: Path,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    overwrite: bool,
) -> Path:
    """Write deterministic tensors plus separately hashed stable metadata."""

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    assert_secret_safe(metadata)
    assert_secret_safe(manifest)
    ordered = _ordered_cpu_tensors(tensors)
    _, save_file = _torch_and_safetensors()
    tensor_path = output_dir / TENSOR_FILE
    metadata_path = output_dir / METADATA_FILE
    manifest_path = output_dir / MANIFEST_FILE
    save_file(ordered, str(tensor_path))
    write_json(metadata_path, metadata, overwrite=overwrite)

    final_manifest = dict(manifest)
    final_manifest.update(
        {
            "format": "lfm2-vl-reference-bundle",
            "tensor_file": TENSOR_FILE,
            "metadata_file": METADATA_FILE,
            "tensor_sha256": sha256_file(tensor_path),
            "metadata_sha256": sha256_file(metadata_path),
            "tensor_count": len(ordered),
            "tensor_inventory": tensor_inventory(ordered),
        }
    )
    assert_secret_safe(final_manifest)
    write_json(manifest_path, final_manifest, overwrite=overwrite)
    return output_dir


def write_metadata_bundle(
    output_dir: Path,
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    overwrite: bool,
) -> Path:
    """Write production/config metadata without creating a tensor payload."""

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    assert_secret_safe(metadata)
    assert_secret_safe(manifest)
    metadata_path = output_dir / METADATA_FILE
    write_json(metadata_path, metadata, overwrite=overwrite)
    final_manifest = dict(manifest)
    final_manifest.update(
        {
            "format": "lfm2-vl-reference-metadata",
            "metadata_file": METADATA_FILE,
            "metadata_sha256": sha256_file(metadata_path),
            "tensor_count": 0,
            "tensor_file": None,
            "tensor_sha256": None,
        }
    )
    assert_secret_safe(final_manifest)
    write_json(output_dir / MANIFEST_FILE, final_manifest, overwrite=overwrite)
    return output_dir


def validate_bundle(output_dir: Path, *, require_tensors: bool = True) -> dict[str, Any]:
    """Validate stable JSON and hashes; optionally validate every safetensor header."""

    import json

    output_dir = output_dir.resolve()
    manifest_path = resolve_regular_file(output_dir, MANIFEST_FILE, "bundle manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"bundle manifest must be a JSON object: {manifest_path}")
    assert_secret_safe(manifest)
    if require_tensors and manifest.get("format") != "lfm2-vl-reference-bundle":
        raise ValueError("manifest is not a tensor bundle")
    metadata_file = manifest.get("metadata_file")
    if not metadata_file:
        raise ValueError("manifest does not name metadata_file")
    metadata_path = resolve_regular_file(output_dir, metadata_file, "metadata file")
    if sha256_file(metadata_path) != manifest.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 does not match manifest")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"bundle metadata must be a JSON object: {metadata_path}")
    assert_secret_safe(metadata)

    tensor_file = manifest.get("tensor_file")
    if require_tensors:
        if not tensor_file:
            raise ValueError("tensor bundle has no tensor_file")
        tensor_path = resolve_regular_file(output_dir, tensor_file, "tensor file")
        if sha256_file(tensor_path) != manifest.get("tensor_sha256"):
            raise ValueError("tensor SHA-256 does not match manifest")
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - manager environment
            raise RuntimeError("manifest validation requires safetensors") from exc
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            names = sorted(handle.keys())
            inventory = manifest.get("tensor_inventory", {})
            if names != sorted(inventory):
                raise ValueError("safetensors names do not match manifest inventory")
            for name in names:
                tensor = handle.get_tensor(name)
                expected = inventory[name]
                if list(tensor.shape) != expected.get("shape"):
                    raise ValueError(f"shape mismatch for tensor {name}")
                dtype = str(tensor.dtype).split(".")[-1]
                if dtype != expected.get("dtype"):
                    raise ValueError(f"dtype mismatch for tensor {name}")
    return manifest


__all__ = [
    "MANIFEST_FILE",
    "METADATA_FILE",
    "TENSOR_FILE",
    "validate_bundle",
    "write_metadata_bundle",
    "write_tensor_bundle",
]
