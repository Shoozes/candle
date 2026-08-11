"""Stable metadata, lock-file, and output-directory helpers for the reference harness.

This module deliberately has no PyTorch, Transformers, or Hub dependency.  The
config-only path can therefore validate the checked-in source lock in a bare
Python installation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping


REFERENCE_PYTHON_PINS = {
    "Linux": "3.10.12",
    "Windows": "3.10.11",
}
REFERENCE_REQUIREMENTS_LOCKS = {
    "Linux": "requirements-reference.txt",
    "Windows": "requirements-reference-windows.txt",
}
REFERENCE_SHARED_PACKAGE_PINS = {
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "safetensors": "0.8.0",
    "transformers": "git+https://github.com/huggingface/transformers.git@fd12552d770f745fdbe41031ff4daa688f5ed57e",
    "huggingface-hub": "1.5.0",
    "tokenizers": "0.22.2",
    "regex": "2025.10.22",
    "pytest": "8.4.1",
    "Pillow": "11.3.0",
}


def reference_package_pins(system_name: str | None = None) -> dict[str, str]:
    """Return the exact oracle pins for one supported host platform."""

    system_name = system_name or platform.system()
    try:
        python_version = REFERENCE_PYTHON_PINS[system_name]
    except KeyError as exc:
        supported = ", ".join(sorted(REFERENCE_PYTHON_PINS))
        raise ValueError(
            f"unsupported reference platform {system_name!r}; choose {supported}"
        ) from exc
    return {"python": python_version, **REFERENCE_SHARED_PACKAGE_PINS}


REFERENCE_PACKAGE_PINS = reference_package_pins()
REFERENCE_RUNTIME_PACKAGE_NAMES = tuple(
    name for name in REFERENCE_PACKAGE_PINS if name != "pytest"
)

_MODEL_ALIASES = {
    "450m": "LiquidAI/LFM2.5-VL-450M",
    "1.6b": "LiquidAI/LFM2.5-VL-1.6B",
}
_SECRET_KEY_PARTS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the byte-stable JSON representation used by all artifacts."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
    label: str = "output",
) -> Path:
    """Publish bytes atomically without clobbering an unapproved destination."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    absolute = Path(os.path.abspath(expanded))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    parent = absolute.parent.resolve(strict=True)
    destination = parent / absolute.name
    if destination.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"{label} is not a regular file: {destination}")
        if not overwrite:
            raise FileExistsError(f"{label} already exists: {destination}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            prefix=f".{destination.name}.tmp-",
            dir=parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"{label} appeared during publication and was not replaced: "
                    f"{destination}"
                ) from exc
        return destination
    finally:
        if temporary is not None and (temporary.exists() or temporary.is_symlink()):
            try:
                temporary.unlink()
            except OSError:
                pass


def write_json(path: Path, value: Any, *, overwrite: bool) -> Path:
    """Write stable JSON atomically without exposing environment state."""

    return write_bytes_atomic(
        path,
        canonical_json_bytes(value),
        overwrite=overwrite,
        label="JSON output",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_regular_file(root: Path, name: str, label: str) -> Path:
    """Resolve one direct, regular file named by a bundle manifest.

    Bundle manifests are portable metadata, not trusted path specifications.
    Only a simple filename in the bundle root is accepted; traversal, absolute
    paths, nested paths, directories, and links are rejected before the caller
    reads or hashes the file.
    """

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"bundle root is not a directory: {root}")
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty filename")
    if Path(name).is_absolute() or "/" in name or "\\" in name:
        raise ValueError(f"{label} must be a direct filename in {root}")

    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a regular file, not a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is not a readable file: {candidate}") from exc
    if resolved.parent != root:
        raise ValueError(f"{label} resolves outside the bundle root: {candidate}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def repo_root(start: Path | None = None) -> Path:
    """Find the checkout containing the checked-in reference lock."""

    candidate = (start or Path(__file__)).resolve()
    if not candidate.is_dir():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "tools" / "lfm2_vl" / "reference-lock.json").is_file():
            return parent
    raise FileNotFoundError("could not locate tools/lfm2_vl/reference-lock.json")


def reference_requirements_path(system_name: str | None = None) -> Path:
    """Return the checked-in resolved lock for one supported platform."""

    system_name = system_name or platform.system()
    try:
        filename = REFERENCE_REQUIREMENTS_LOCKS[system_name]
    except KeyError as exc:
        supported = ", ".join(sorted(REFERENCE_REQUIREMENTS_LOCKS))
        raise ValueError(
            f"unsupported reference platform {system_name!r}; choose {supported}"
        ) from exc
    return Path(__file__).resolve().parent / filename


def reference_environment_lock(system_name: str | None = None) -> dict[str, str]:
    """Identify the resolved environment lock without reading installed packages."""

    system_name = system_name or platform.system()
    path = reference_requirements_path(system_name)
    if not path.is_file():
        raise FileNotFoundError(f"reference environment lock is missing: {path}")
    return {
        "platform": system_name,
        "filename": path.name,
        "sha256": sha256_file(path),
    }


def load_reference_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = path or (repo_root() / "tools" / "lfm2_vl" / "reference-lock.json")
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1:
        raise ValueError(f"unsupported reference-lock schema: {lock.get('schema_version')!r}")
    if lock.get("lock_policy", {}).get("production_weight_payloads_downloaded"):
        raise ValueError("reference lock unexpectedly claims production payloads were downloaded")
    return lock


def model_id_for_alias(alias: str) -> str:
    try:
        return _MODEL_ALIASES[alias.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_MODEL_ALIASES))
        raise ValueError(f"unknown model alias {alias!r}; choose {choices}") from exc


def model_entry(lock: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    model_id = _MODEL_ALIASES.get(model.lower(), model)
    for entry in lock.get("model_repositories", []):
        if entry.get("id") == model_id:
            return entry
    raise ValueError(f"model is not pinned in reference-lock.json: {model_id}")


def transformers_entry(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    for entry in lock.get("repositories", []):
        if entry.get("id") == "huggingface-transformers":
            return entry
    raise ValueError("huggingface-transformers is missing from reference-lock.json")


def package_versions() -> dict[str, str]:
    """Report installed versions without reading or printing environment variables."""

    versions: dict[str, str] = {"python": platform.python_version()}
    for label in (
        "torch",
        "torchvision",
        "safetensors",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "regex",
        "pytest",
        "Pillow",
    ):
        try:
            versions[label] = importlib.metadata.version(label)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "missing"
    return versions


def _installed_vcs_revision(distribution_name: str) -> str | None:
    """Return a VCS commit recorded by pip's direct_url metadata, if present."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, Mapping):
        commit_id = vcs_info.get("commit_id")
        if isinstance(commit_id, str):
            return commit_id
    return None


def reference_environment_mismatches(
    system_name: str | None = None,
) -> dict[str, dict[str, str]]:
    """Return installed-package differences from the locked oracle environment."""

    expected_pins = reference_package_pins(system_name)
    installed = package_versions()
    mismatches: dict[str, dict[str, str]] = {}
    for name in REFERENCE_RUNTIME_PACKAGE_NAMES:
        expected = expected_pins[name]
        if name == "transformers":
            expected_revision = expected.rsplit("@", 1)[-1]
            actual_revision = _installed_vcs_revision(name)
            if actual_revision != expected_revision:
                mismatches[name] = {
                    "expected": expected_revision,
                    "installed": actual_revision or installed.get(name, "missing"),
                }
            continue
        actual = installed.get(name, "missing")
        if actual != expected:
            mismatches[name] = {"expected": expected, "installed": actual}
    return mismatches


def require_reference_environment(system_name: str | None = None) -> None:
    """Refuse production loading unless every locked runtime pin is installed."""

    selected_system = system_name or platform.system()
    mismatches = reference_environment_mismatches(selected_system)
    if mismatches:
        details = "; ".join(
            f"{name}: expected {values['expected']!r}, installed {values['installed']!r}"
            for name, values in sorted(mismatches.items())
        )
        requirements_name = REFERENCE_REQUIREMENTS_LOCKS[selected_system]
        raise RuntimeError(
            "pinned reference environment mismatch; use the owner-managed "
            f"{selected_system} environment from {requirements_name} ({details})"
        )


def prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    """Create an output directory, refusing any pre-existing directory by default."""

    path = path.resolve()
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {path}; pass --overwrite to reuse it"
            )
    else:
        path.mkdir(parents=True)
    return path


def image_sha256(image_bytes: bytes) -> str:
    return sha256_bytes(image_bytes)


def _value(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def effective_ffn_dim(text_config: Mapping[str, Any]) -> int:
    hidden_size = int(_value(text_config, "hidden_size", default=0))
    raw = int(_value(text_config, "block_ff_dim", "intermediate_size", default=hidden_size * 4))
    if bool(_value(text_config, "block_auto_adjust_ff_dim", default=False)):
        raw = (2 * raw) // 3
        raw = int(float(_value(text_config, "block_ffn_dim_multiplier", default=1.0)) * raw)
        multiple = int(_value(text_config, "block_multiple_of", default=1))
        if multiple <= 0:
            raise ValueError("block_multiple_of must be positive")
        raw = math.ceil(raw / multiple) * multiple
    return raw


def _full_attention_layers(text_config: Mapping[str, Any]) -> list[int]:
    layer_types = list(text_config.get("layer_types", []))
    if not layer_types and "full_attention_layers" in text_config:
        return [int(index) for index in text_config["full_attention_layers"]]
    return [index for index, layer_type in enumerate(layer_types) if "attention" in str(layer_type)]


def normalized_model_summary(
    entry: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    processor: Mapping[str, Any] | None = None,
    source: str = "reference-lock",
) -> dict[str, Any]:
    """Normalize either lock-derived or locally supplied small JSON files."""

    locked = entry.get("source_confirmed_config", {})
    config = config or locked
    text = dict(config.get("text_config") or config.get("text") or locked.get("text", {}))
    vision = dict(config.get("vision_config") or config.get("vision") or locked.get("vision", {}))
    processor_defaults = dict(locked.get("processor", {}))
    if processor:
        processor_defaults.update(processor)
    if "type" not in processor_defaults and "image_processor_type" in processor_defaults:
        processor_defaults["type"] = processor_defaults["image_processor_type"]
    processor = processor_defaults

    architecture = _value(config, "architectures", default=locked.get("architecture"))
    if isinstance(architecture, list):
        architecture = architecture[0] if architecture else locked.get("architecture")
    model_type = _value(config, "model_type", default=locked.get("model_type"))
    text_hidden = int(_value(text, "hidden_size", default=0))
    vision_hidden = int(_value(vision, "hidden_size", default=0))
    factor = int(_value(config, "downsample_factor", default=locked.get("downsample_factor", 0)))
    projector_hidden = int(
        _value(config, "projector_hidden_size", default=locked.get("projector_hidden_size", 0))
    )
    summary = {
        "source": source,
        "model_id": entry["id"],
        "model_revision": entry["revision"],
        "processor_revision": entry["revision"],
        "transformers_revision": transformers_entry(load_reference_lock())[
            "revision"
        ],
        "architecture": architecture,
        "model_type": model_type,
        "image_token_id": int(_value(config, "image_token_id", default=locked.get("image_token_id", 0))),
        "downsample_factor": factor,
        "projector": {
            "hidden_size": projector_hidden,
            "activation": _value(config, "projector_hidden_act", default=locked.get("projector_hidden_act")),
            "bias": bool(_value(config, "projector_bias", default=locked.get("projector_bias", True))),
            "use_layernorm": bool(
                _value(
                    config,
                    "projector_use_layernorm",
                    default=locked.get("projector_use_layernorm", False),
                )
            ),
            "input_size": vision_hidden * factor * factor,
            "output_size": text_hidden,
        },
        "text": {
            "model_type": _value(text, "model_type", default="lfm2"),
            "hidden_size": text_hidden,
            "vocab_size": int(_value(text, "vocab_size", default=0)),
            "num_hidden_layers": int(_value(text, "num_hidden_layers", default=0)),
            "num_attention_heads": int(_value(text, "num_attention_heads", "num_heads", default=0)),
            "num_key_value_heads": int(_value(text, "num_key_value_heads", default=0)),
            "intermediate_size": int(_value(text, "intermediate_size", default=0)),
            "effective_ffn_size": effective_ffn_dim(text),
            "conv_L_cache": int(_value(text, "conv_L_cache", "conv_l_cache", default=0)),
            "max_position_embeddings": int(_value(text, "max_position_embeddings", default=0)),
            "layer_types": list(text.get("layer_types", [])),
            "full_attention_layers": _full_attention_layers(text),
            "eos_token_id": int(_value(text, "eos_token_id", default=7)),
        },
        "vision": {
            "model_type": _value(vision, "model_type", default="siglip2_vision_model"),
            "hidden_size": vision_hidden,
            "intermediate_size": int(_value(vision, "intermediate_size", default=0)),
            "num_hidden_layers": int(_value(vision, "num_hidden_layers", default=0)),
            "num_attention_heads": int(_value(vision, "num_attention_heads", default=0)),
            "num_channels": int(_value(vision, "num_channels", default=0)),
            "patch_size": int(_value(vision, "patch_size", default=0)),
            "num_patches": int(_value(vision, "num_patches", default=0)),
            "vision_use_head": bool(_value(vision, "vision_use_head", default=False)),
        },
        "processor": {
            key: processor[key]
            for key in (
                "type",
                "tile_size",
                "min_tiles",
                "max_tiles",
                "min_image_tokens",
                "max_image_tokens",
                "max_num_patches",
                "use_thumbnail",
                "image_mean",
                "image_std",
            )
            if key in processor
        },
        "special_token_ids": dict(
            locked.get("special_token_ids", {"bos": 1, "eos": 7, "pad": 0})
        ),
        "tie_word_embeddings": bool(
            _value(config, "tie_word_embeddings", "tie_embedding", default=True)
        ),
    }
    validate_summary(summary)
    return summary


def validate_summary(summary: Mapping[str, Any]) -> None:
    text = summary["text"]
    vision = summary["vision"]
    projector = summary["projector"]
    required_positive = (
        ("text.hidden_size", text["hidden_size"]),
        ("text.num_hidden_layers", text["num_hidden_layers"]),
        ("vision.hidden_size", vision["hidden_size"]),
        ("vision.patch_size", vision["patch_size"]),
        ("downsample_factor", summary["downsample_factor"]),
        ("projector.hidden_size", projector["hidden_size"]),
    )
    for label, value in required_positive:
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive")
    if projector["input_size"] != vision["hidden_size"] * summary["downsample_factor"] ** 2:
        raise ValueError("projector input width does not match vision width and downsample factor")
    if projector["output_size"] != text["hidden_size"]:
        raise ValueError("projector output width does not match text hidden size")
    if text["hidden_size"] % text["num_attention_heads"]:
        raise ValueError("text hidden size is not divisible by attention heads")
    if vision["hidden_size"] % vision["num_attention_heads"]:
        raise ValueError("vision hidden size is not divisible by attention heads")


def assert_secret_safe(value: Any) -> None:
    """Reject credentials before JSON or manifest serialization."""

    encoded = canonical_json_bytes(value).lower()
    for part in _SECRET_KEY_PARTS:
        if part.encode("ascii") in encoded:
            raise ValueError(f"refusing to serialize secret-like metadata key: {part}")
