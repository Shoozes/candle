"""Report whether the current interpreter satisfies the pinned oracle lane."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        REFERENCE_SHARED_PACKAGE_PINS,
        package_versions,
        reference_environment_lock,
        reference_environment_mismatches,
        reference_package_pins,
        reference_requirements_path,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest import (  # type: ignore
        REFERENCE_SHARED_PACKAGE_PINS,
        package_versions,
        reference_environment_lock,
        reference_environment_mismatches,
        reference_package_pins,
        reference_requirements_path,
    )


def _lock_lines(path: Path) -> list[str]:
    return sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("--")
    )


def _pip_freeze_lines() -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--all",
            "--disable-pip-version-check",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"pip freeze failed with exit {result.returncode}: {detail}")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _resolved_lock_report(system_name: str) -> dict[str, Any]:
    path = reference_requirements_path(system_name)
    expected = _lock_lines(path)
    installed = _pip_freeze_lines()
    expected_set = set(expected)
    installed_set = set(installed)
    return {
        **reference_environment_lock(system_name),
        "distribution_count": len(expected),
        "missing": sorted(expected_set - installed_set),
        "unexpected": sorted(installed_set - expected_set),
        "passed": expected == installed,
    }


def environment_report(
    *, require_tests: bool = False, verify_lock: bool = False
) -> dict[str, Any]:
    """Build a secret-free, import-light environment verification report."""

    system_name = platform.system()
    expected = reference_package_pins(system_name)
    installed = package_versions()
    mismatches = reference_environment_mismatches(system_name)
    test_mismatches: dict[str, dict[str, str]] = {}
    expected_pytest = REFERENCE_SHARED_PACKAGE_PINS["pytest"]
    actual_pytest = installed.get("pytest", "missing")
    if actual_pytest != expected_pytest:
        test_mismatches["pytest"] = {
            "expected": expected_pytest,
            "installed": actual_pytest,
        }
    lock_report = _resolved_lock_report(system_name) if verify_lock else None
    passed = (
        not mismatches
        and (not require_tests or not test_mismatches)
        and (lock_report is None or lock_report["passed"])
    )
    return {
        "schema_version": 1,
        "format": "lfm2-vl-reference-environment",
        "platform": {
            "system": system_name,
            "machine": platform.machine(),
        },
        "expected": expected,
        "installed": installed,
        "runtime_mismatches": mismatches,
        "test_mismatches": test_mismatches,
        "tests_required": require_tests,
        "resolved_lock": lock_report,
        "lock_verification_required": verify_lock,
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-tests",
        action="store_true",
        help="also require the pinned pytest version",
    )
    parser.add_argument(
        "--verify-lock",
        action="store_true",
        help="require pip freeze --all to match the platform lock",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = environment_report(
            require_tests=args.require_tests,
            verify_lock=args.verify_lock,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"verify_environment: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["environment_report", "main"]
