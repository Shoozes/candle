# LFM2.5-VL Status

## Baseline

- Upstream: Hugging Face Candle
- Base version: 0.11.0
- Working branch: `feat/lfm2-vl-mmproj`
- Baseline commit: `31f35b147389700ed2a178ee66a91c3cc25cc80d`
- Baseline tag: `lfm2-vl-baseline-candle-0.11.0`
- Bootstrap checkpoint tag: `lfm2-vl-phase-0-bootstrap` after checkpoint commit

## Current Phase

- Phase: 0 — Bootstrap and baseline verification
- Task: Bootstrap controls and CPU-only baseline proof
- Scope: LFM2-VL control documentation, environment report, baseline verifier, and fixture/tooling readmes
- Status: green

## Last Green Verification

- Date: 2026-08-09 22:35 EDT (`2026-08-10T02:35:09Z` to `2026-08-10T02:35:12Z`)
- Environment: WSL2 `NVIDIA-Workbench`; Ubuntu 22.04.5 LTS; Linux `6.6.87.2-microsoft-standard-WSL2`; CPU-only lane
- Verification HEAD: `31f35b147389700ed2a178ee66a91c3cc25cc80d` with the reviewed bootstrap paths staged in the detached Linux verification worktree
- Command: from `/tmp`, `bash /home/workbench/code/candle-lfm2-vl-verify/scripts/lfm2-vl/verify-baseline.sh`
- Results: passed `cargo fmt --all -- --check`; locked/offline checks for `candle-core`, `candle-nn`, `candle-transformers`, `lfm2`, and `quantized-lfm2`; `git diff --check`; and `git diff --cached --check`
- Lockfile: local-only ignored `Cargo.lock`; SHA-256 `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`
- Retained baseline log: `artifacts/verification/bootstrap/baseline.log`; SHA-256 `a4f77d1b007eb267865be01ef1c239754ac0e093dd1c27ad457d77242b614f22`
- Retained environment log: `artifacts/verification/bootstrap/env-report.log`; SHA-256 `5f4fd70b4dd5ca6a956c9678d386598ca2ff6bcdb2e75ef3ba3aa6a10775e4d8`

## Environment Snapshot

- WSL2 distribution: `NVIDIA-Workbench`
- Linux home: `/home/workbench`
- Rust compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Cargo: `cargo 1.97.1 (c980f4866 2026-06-30)`
- Python: `Python 3.10.12`
- CMake: `cmake 3.22.1`
- Ninja: missing
- System pip: missing
- Available space for the Linux-home filesystem: `829G`

## Proven

- Static inspection identified package `candle-examples`.
- Static inspection identified examples `lfm2` and `quantized-lfm2`.
- Bootstrap scripts use the actual package and example names from the workspace.
- The untouched Candle 0.11 baseline and both existing LFM2 examples compile in the locked, offline, CPU-only lane.
- The baseline verifier resolves the repository root correctly when invoked from `/tmp`.
- The bootstrap checkpoint changes no Candle source or Cargo manifest.

## Bootstrap Static Checks

- WSL `bash -n` for `env-report.sh` and `verify-baseline.sh`: passed.
- Expected-file and Markdown/path inspection: passed; 11 requested artifact paths exist and no JSON fixture was created.
- `git diff --check` and `git diff --cached --check`: passed after normalizing the two supplied attachment copies from CRLF to LF.
- Environment report invoked from `/tmp`: passed.

## Known Failures

- None.

## Blockers

- None.

## Active Files

- No Candle source file is active.
- The bootstrap checkpoint paths are awaiting the phase commit and tag.

## Reference Pins

- Transformers: not pinned in Bootstrap Phase
- LiquidAI 450M: not pinned in Bootstrap Phase
- LiquidAI 1.6B: not pinned in Bootstrap Phase
- mistral.rs: not pinned in Bootstrap Phase
- llama.cpp: not pinned in Bootstrap Phase

## Next Task

Perform Source Lock Phase only: pin authoritative revisions and licenses without changing Candle Rust source or downloading production weights.

---
AI-edited: 2026-08-09T22:38:03-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=coordination | change=recorded green bootstrap proof and next task
