# LFM2.5-VL Failure Log

## F-0001: Baseline Replay Pending

Status: Resolved; not a code failure.

Context:
The Windows edit worktree is linked to the authoritative WSL repository. The task explicitly forbids running the full Cargo baseline there, so manager replay from the Linux-home verification worktree was required.

Reproduction:
Not run in this worktree by design.

Expected:
The Linux-home verifier records exact locked CPU-only results without model downloads or CUDA features.

Actual:
The manager replay passed in the Linux-home verification worktree from `2026-08-10T02:35:09Z` to `2026-08-10T02:35:12Z`.

Resolution:
Retained baseline log SHA-256: `a4f77d1b007eb267865be01ef1c239754ac0e093dd1c27ad457d77242b614f22`.

## F-0002: Missing Optional Environment Tools

Status: Known environment gap from manager audit; not a verifier result.

Observed:

- `ninja` is missing.
- System `pip` is missing.

Impact:
Future reference-harness setup may need a manager-owned environment decision. Bootstrap scripts report missing optional tools without installing or downloading anything.

## F-0003: Offline Cargo Cache Incomplete

Status: Resolved environment preparation issue; not a Candle code failure.

Context:
Upstream Candle 0.11 does not track `Cargo.lock`, and the initial WSL Cargo cache lacked required crate archives.

Observed:

- `cargo generate-lockfile --offline` could not resolve `accelerate-src`.
- The first locked verifier run stopped at missing `aho-corasick v1.1.5` because network access was disabled.

Resolution:
Generated the ignored local lockfile with one bounded crates.io index resolution, ran `cargo fetch --locked --target x86_64-unknown-linux-gnu`, and reran the verifier offline. Lockfile SHA-256: `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`.

---
AI-edited: 2026-08-09T22:38:03-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=resolved bootstrap replay and Cargo cache failures
