"""Offline contract tests for guarded production snapshot acquisition."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import acquire_snapshot
from manifest import load_reference_lock, model_entry


def _git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _synthetic_entry(payloads: dict[str, bytes]) -> dict:
    metadata_names = [name for name in payloads if name != "model.safetensors"]
    records = []
    for name in [*metadata_names, "model.safetensors"]:
        payload = payloads[name]
        if name == "model.safetensors":
            identity = {"kind": "sha256", "value": hashlib.sha256(payload).hexdigest()}
        else:
            identity = {"kind": "git-blob-sha1", "value": _git_blob_oid(payload)}
        records.append({"path": name, "bytes": len(payload), "identity": identity})
    total = sum(len(payload) for payload in payloads.values())
    return {
        "id": "test-model",
        "repository": "https://example.invalid/test-model",
        "revision": "a" * 40,
        "revision_url": "https://example.invalid/test-model/tree/" + "a" * 40,
        "files": [{"path": name, "purpose": "test metadata"} for name in metadata_names],
        "safetensors_header": {"file": "model.safetensors"},
        "acquisition": {
            "schema_version": 1,
            "token_policy": "public-no-token",
            "snapshot_bytes": total,
            "minimum_free_bytes": total,
            "files": records,
        },
    }


def _install_lock(monkeypatch, entry: dict) -> None:
    monkeypatch.setattr(
        acquire_snapshot,
        "load_reference_lock",
        lambda: {"model_repositories": [entry]},
    )


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    external = tmp_path / "external"
    external.mkdir()
    return external, external / "snapshot", external / "cache", external / "acquisition.json"


def _artifact_builder(_model: str, root: Path) -> dict:
    records = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        payload = path.read_bytes()
        records.append(
            {
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "regular_file": True,
            }
        )
    return {"files": records, "file_count": len(records)}


def test_official_16b_acquisition_contract_is_exact():
    entry = model_entry(load_reference_lock(), "1.6b")
    contract = acquire_snapshot.acquisition_contract(entry)
    assert contract["snapshot_bytes"] == 3_198_084_631
    assert contract["minimum_free_bytes"] == 12 * 1024**3
    assert len(contract["files"]) == 8
    assert contract["files"][-1] == {
        "path": "model.safetensors",
        "bytes": 3_193_334_216,
        "identity": {
            "kind": "sha256",
            "value": "7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d",
        },
    }


def test_runtime_api_has_no_downloader_or_verifier_bypass():
    parameters = inspect.signature(acquire_snapshot.acquire_snapshot).parameters
    assert tuple(parameters) == (
        "model",
        "output_dir",
        "cache_dir",
        "manifest_path",
        "allow_production_download",
    )


def test_plan_runs_without_site_packages_or_creating_targets():
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    code = "\n".join(
        [
            "import sys, tempfile",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(HERE)!r})",
            "import acquire_snapshot",
            f"entry = {entry!r}",
            "acquire_snapshot.load_reference_lock = "
            "lambda: {'model_repositories': [entry]}",
            "with tempfile.TemporaryDirectory() as temporary:",
            "    root = Path(temporary)",
            "    output = root / 'snapshot'",
            "    cache = root / 'cache'",
            "    manifest = root / 'acquisition.json'",
            "    plan = acquire_snapshot.build_acquisition_plan("
            "'test-model', output, cache, manifest)",
            "    assert plan['schema_version'] == 2",
            "    assert plan['network_used'] is False",
            "    assert plan['network_policy'] == 'disabled'",
            "    assert plan['model_loaded'] is False",
            "    assert plan['transfer_policy'] == "
            "'serial-files-resumable-http-xet-disabled'",
            "    assert not output.exists()",
            "    assert not cache.exists()",
            "    assert not manifest.exists()",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_downloader_requires_only_the_pinned_hub_package(monkeypatch):
    queries = []
    monkeypatch.delenv(acquire_snapshot.HUB_DISABLE_XET_ENV, raising=False)
    monkeypatch.setattr(
        acquire_snapshot.importlib.metadata,
        "version",
        lambda name: queries.append(name) or "1.5.0",
    )
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.constants = SimpleNamespace(HF_HUB_DISABLE_XET=True)
    fake_hub.hf_hub_download = lambda **_kwargs: "unused"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    downloader = acquire_snapshot._load_default_downloader()

    assert queries == ["huggingface-hub"]
    assert os.environ[acquire_snapshot.HUB_DISABLE_XET_ENV] == "1"
    assert downloader is fake_hub.hf_hub_download


def test_default_downloader_rejects_unpinned_hub_before_import(monkeypatch):
    monkeypatch.setattr(
        acquire_snapshot.importlib.metadata,
        "version",
        lambda _name: "0.0.0",
    )

    with pytest.raises(RuntimeError, match="expected huggingface-hub==1.5.0"):
        acquire_snapshot._load_default_downloader()


def test_default_downloader_refuses_preimported_xet_enabled_hub(monkeypatch):
    monkeypatch.delenv(acquire_snapshot.HUB_DISABLE_XET_ENV, raising=False)
    monkeypatch.setattr(
        acquire_snapshot.importlib.metadata,
        "version",
        lambda _name: "1.5.0",
    )
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.constants = SimpleNamespace(HF_HUB_DISABLE_XET=False)
    fake_hub.hf_hub_download = lambda **_kwargs: "unused"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match="Xet to be disabled before Hub import"):
        acquire_snapshot._load_default_downloader()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["files"].append(dict(raw["files"][0])), "duplicate path"),
        (lambda raw: raw["files"].reverse(), "exactly match pinned"),
        (lambda raw: raw.update(snapshot_bytes=1), "snapshot_bytes mismatch"),
        (
            lambda raw: raw["files"][0]["identity"].update(value="bad"),
            "invalid git-blob-sha1",
        ),
        (lambda raw: raw.update(token_policy="implicit"), "public-no-token"),
    ],
)
def test_acquisition_contract_rejects_drift(mutation, message):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    mutation(entry["acquisition"])
    with pytest.raises(ValueError, match=message):
        acquire_snapshot.acquisition_contract(entry)


def test_plan_is_read_only_and_rejects_repository_output(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    plan = acquire_snapshot.build_acquisition_plan("test-model", output, cache, manifest)
    assert plan["schema_version"] == 2
    assert plan["network_used"] is False
    assert plan["network_policy"] == "disabled"
    assert plan["model_loaded"] is False
    assert plan["transfer_policy"] == "serial-files-resumable-http-xet-disabled"
    assert plan["file_count"] == 2
    assert not output.exists()
    assert not cache.exists()
    assert not manifest.exists()

    with pytest.raises(PermissionError, match="outside the repository"):
        acquire_snapshot.build_acquisition_plan(
            "test-model", ROOT / "artifacts" / "snapshot", cache, manifest
        )
    assert list(external.iterdir()) == []


def test_plan_refuses_stale_staging_without_deleting_it(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    _external, output, cache, manifest = _paths(tmp_path)
    stale = output.parent / f".{output.name}.partial-1234-abcd"
    stale.mkdir()

    with pytest.raises(FileExistsError, match="stale acquisition staging path"):
        acquire_snapshot.build_acquisition_plan(
            "test-model", output, cache, manifest
        )

    assert stale.is_dir()
    assert not output.exists()
    assert not cache.exists()
    assert not manifest.exists()


def test_plan_refuses_stale_manifest_staging_without_deleting_it(
    tmp_path: Path, monkeypatch
):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    _external, output, cache, manifest = _paths(tmp_path)
    stale = manifest.with_name(f".{manifest.name}.tmp-1234")
    stale.write_bytes(b"partial manifest")

    with pytest.raises(FileExistsError, match="stale acquisition manifest staging path"):
        acquire_snapshot.build_acquisition_plan(
            "test-model", output, cache, manifest
        )

    assert stale.read_bytes() == b"partial manifest"
    assert not output.exists()
    assert not cache.exists()
    assert not manifest.exists()


def test_download_requires_explicit_opt_in_before_paths_or_network(monkeypatch):
    called = False

    def downloader(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not download")

    monkeypatch.setattr(acquire_snapshot, "_load_default_downloader", lambda: downloader)

    with pytest.raises(PermissionError, match="--allow-production-download"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            Path("missing-output"),
            Path("missing-cache"),
            Path("missing-manifest"),
            allow_production_download=False,
        )
    assert called is False


def test_insufficient_disk_refuses_before_downloader_or_network(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    _external, output, cache, manifest = _paths(tmp_path)
    monkeypatch.setattr(
        acquire_snapshot.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=100, free=0),
    )
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: pytest.fail("downloader guard must not run after failed disk admission"),
    )
    with pytest.raises(RuntimeError, match="disk admission failed"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )
    assert not output.exists()
    assert not manifest.exists()


def test_paths_are_revalidated_after_cache_creation(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    _external, output, cache, manifest = _paths(tmp_path)
    called = False

    def downloader(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not download")

    def load_downloader():
        stale = output.parent / f".{output.name}.partial-race"
        stale.mkdir()
        return downloader

    monkeypatch.setattr(acquire_snapshot, "_load_default_downloader", load_downloader)

    with pytest.raises(FileExistsError, match="stale acquisition staging path"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert called is False
    assert cache.is_dir()
    assert not output.exists()
    assert not manifest.exists()


def test_download_failure_preserves_cache_but_removes_staging(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)

    def downloader(**_kwargs):
        marker = cache / "resumable.partial"
        marker.write_bytes(b"partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr(acquire_snapshot, "_load_default_downloader", lambda: downloader)

    with pytest.raises(
        RuntimeError, match="download failed for config.json: OSError"
    ) as caught:
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert caught.value.__cause__ is None
    assert (cache / "resumable.partial").read_bytes() == b"partial"
    assert not output.exists()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_staging_cleanup_failure_is_actionable_and_blocks_retry(tmp_path: Path, monkeypatch):
    entry = _synthetic_entry({"config.json": b"{}\n", "model.safetensors": b"weights"})
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **_kwargs: (_ for _ in ()).throw(OSError("download failure"))),
    )
    monkeypatch.setattr(
        acquire_snapshot.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup failure")),
    )

    with pytest.raises(RuntimeError, match="staging cleanup also failed"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    stages = list(external.glob(".snapshot.partial-*"))
    assert len(stages) == 1
    assert stages[0].is_dir()
    assert not output.exists()
    assert not manifest.exists()


def test_downloaded_source_must_resolve_inside_owned_cache(tmp_path: Path, monkeypatch):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    escaped = external / "escaped-sources"
    escaped.mkdir()
    for name, payload in payloads.items():
        (escaped / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: escaped / kwargs["filename"]),
    )

    with pytest.raises(ValueError, match="escaped the caller-owned cache"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert cache.is_dir()
    assert not output.exists()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_acquisition_is_serial_public_atomic_and_reverified(tmp_path: Path, monkeypatch):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return sources / kwargs["filename"]

    monkeypatch.setattr(acquire_snapshot, "_load_default_downloader", lambda: downloader)
    monkeypatch.setattr(acquire_snapshot, "build_artifact_manifest", _artifact_builder)

    result = acquire_snapshot.acquire_snapshot(
        "test-model",
        output,
        cache,
        manifest,
        allow_production_download=True,
    )
    assert [call["filename"] for call in calls] == ["config.json", "model.safetensors"]
    assert all(call["revision"] == "a" * 40 for call in calls)
    assert all(call["token"] is False for call in calls)
    assert all(call["local_files_only"] is False for call in calls)
    assert output.is_dir()
    assert (output / "config.json").read_bytes() == payloads["config.json"]
    assert (output / "model.safetensors").read_bytes() == payloads["model.safetensors"]
    assert result["snapshot_published_atomically"] is True
    assert result["schema_version"] == 2
    assert result["model_loaded"] is False
    assert result["network_policy"] == "permitted-cache-aware"
    assert result["network_used"] is None
    assert result["transfer_policy"] == "serial-files-resumable-http-xet-disabled"
    assert result["manifest_path"] == str(manifest.resolve())
    assert json.loads(manifest.read_text(encoding="utf-8"))["total_bytes"] == sum(
        map(len, payloads.values())
    )
    assert list(external.glob(".snapshot.partial-*")) == []


def test_snapshot_publication_does_not_replace_a_racing_destination(
    tmp_path: Path, monkeypatch
):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )

    def racing_artifact_builder(model: str, root: Path) -> dict:
        artifact = _artifact_builder(model, root)
        output.mkdir()
        (output / "owner.txt").write_text("do not replace", encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        acquire_snapshot,
        "build_artifact_manifest",
        racing_artifact_builder,
    )

    with pytest.raises(FileExistsError, match="was not replaced"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert (output / "owner.txt").read_text(encoding="utf-8") == "do not replace"
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_manifest_publication_race_rolls_back_without_replacing_owner_file(
    tmp_path: Path, monkeypatch
):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )

    def racing_artifact_builder(model: str, root: Path) -> dict:
        artifact = _artifact_builder(model, root)
        manifest.write_text("owner manifest", encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        acquire_snapshot,
        "build_artifact_manifest",
        racing_artifact_builder,
    )

    with pytest.raises(FileExistsError, match="artifact manifest already exists"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert manifest.read_text(encoding="utf-8") == "owner manifest"
    assert not output.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_duplicate_artifact_inventory_never_publishes(tmp_path: Path, monkeypatch):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )

    def duplicate_artifact_builder(model: str, root: Path) -> dict:
        artifact = _artifact_builder(model, root)
        artifact["files"].append(dict(artifact["files"][0]))
        return artifact

    monkeypatch.setattr(
        acquire_snapshot,
        "build_artifact_manifest",
        duplicate_artifact_builder,
    )

    with pytest.raises(ValueError, match="duplicate file"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert not output.exists()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_identity_failure_removes_staging_and_never_publishes(tmp_path: Path, monkeypatch):
    expected = {"config.json": b"{}\n", "model.safetensors": b"weights-a"}
    entry = _synthetic_entry(expected)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    (sources / "config.json").write_bytes(expected["config.json"])
    (sources / "model.safetensors").write_bytes(b"weights-b")
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )
    monkeypatch.setattr(acquire_snapshot, "build_artifact_manifest", _artifact_builder)

    with pytest.raises(ValueError, match="identity mismatch"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )
    assert not output.exists()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_manifest_failure_rolls_back_published_snapshot(tmp_path: Path, monkeypatch):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    monkeypatch.setattr(
        acquire_snapshot,
        "write_artifact_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest failure")),
    )
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )
    monkeypatch.setattr(acquire_snapshot, "build_artifact_manifest", _artifact_builder)

    with pytest.raises(OSError, match="manifest failure"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )
    assert not output.exists()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []


def test_manifest_and_rollback_failure_reports_remaining_snapshot(tmp_path: Path, monkeypatch):
    payloads = {"config.json": b"{}\n", "model.safetensors": b"weights"}
    entry = _synthetic_entry(payloads)
    _install_lock(monkeypatch, entry)
    external, output, cache, manifest = _paths(tmp_path)
    sources = cache / "sources"
    sources.mkdir(parents=True)
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    monkeypatch.setattr(
        acquire_snapshot,
        "_load_default_downloader",
        lambda: (lambda **kwargs: sources / kwargs["filename"]),
    )
    monkeypatch.setattr(acquire_snapshot, "build_artifact_manifest", _artifact_builder)
    monkeypatch.setattr(
        acquire_snapshot,
        "write_artifact_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest failure")),
    )
    real_rename = acquire_snapshot._rename_directory_no_replace

    def rename(source, destination, label):
        if Path(source) == output:
            raise OSError("rollback failure")
        return real_rename(source, destination, label)

    monkeypatch.setattr(acquire_snapshot, "_rename_directory_no_replace", rename)

    with pytest.raises(RuntimeError, match="do not load the snapshot remaining"):
        acquire_snapshot.acquire_snapshot(
            "test-model",
            output,
            cache,
            manifest,
            allow_production_download=True,
        )

    assert output.is_dir()
    assert not manifest.exists()
    assert list(external.glob(".snapshot.partial-*")) == []
