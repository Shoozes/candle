# Workspace dependency security maintenance

## Selected root-lock repair, 2026-10-25

This local batch starts at published main `f1b002c3733da283c57aebddd317346f5751330b`
plus the preserved eight-document handoff cleanup. It changes only compatible
registry package selections in `Cargo.lock`; manifests, features, model math,
public commands and consumer pins are unchanged.

| Package | Before | Candidate | Retained advisory addressed |
| --- | --- | --- | --- |
| h2 | 0.4.15 | 0.4.16 | RUSTSEC-2026-0258 |
| rustls | 0.23.43 | 0.23.46 | RUSTSEC-2026-0285, -0346, -0348, -0349 |
| rustls-webpki | 0.103.13 | 0.103.16 | RUSTSEC-2026-0382 |
| webpki-root-certs | 1.0.9 | 1.0.10 | RUSTSEC-2026-0383 |
| webpki-roots | 1.0.9 | 1.0.10 | RUSTSEC-2026-0381 |
| tokio | 1.53.1 | 1.53.3 | RUSTSEC-2026-0386, Unix AsyncFd I/O-safety warning |

The first five patches close the selected eight vulnerability records. The
full audit also exposed the existing patchable Tokio unsoundness warning;
the sixth compatible patch closes it without suppressing the advisory.
All patched sources were already cached. Their declared MSRV is at most 1.71,
below this repository's pinned Rust 1.97.1. No package version was added or
removed: the root lock still contains 805 packages. Apart from these six
version/checksum updates, only the existing webpki-roots 0.26.11 wrapper edge
changes to refer to 1.0.10. Cargo's unrelated compatible Windows/getrandom
edge reselections were restored to the original valid locked choices.

## Caller and proof scope

The Windows normal/build/dev reverse graphs reach these dependencies through
HF Hub 1.0.0, Hyper/Hyper-Rustls, Reqwest and the Xet transport stack, consumed
by `candle-datasets` and `candle-examples`. The actual caller surfaces include
`candle-datasets/src/hub.rs` and `candle-examples/src/lib.rs`. This is a
network acquisition dependency repair; it does not exercise a model or change
the private d1/ordinary VLM execution policies.

The full unfiltered audit uses cargo-audit 0.22.2 and the unchanged local
RustSec database commit `21d909e4c8bd4e14c7aea58bf980c881a0053ab1`, with no
ignores, target filters or network fetch. Candidate lock SHA-256 is
`b7c0c561aaf0b193909317f41cb88910c04f7257d8fea4a71fa67708c836e735`.
It reports zero vulnerabilities and zero unsoundness warnings. This is an
exact-source/database claim, not a freshly fetched advisory database claim.

Five unmaintained-package notices remain visible: bincode 1.3.3 and 2.0.1
(RUSTSEC-2025-0141), paste 1.0.15 (RUSTSEC-2024-0436), proc-macro-error 1.0.4
(RUSTSEC-2024-0370) and ttf-parser 0.25.1 (RUSTSEC-2026-0192). They are unchanged
from intake; replacing those APIs is outside this compatible security patch batch.
No ignore list or blanket maintenance disposition is introduced.

Exact audit arguments, raw reports, incoming-document recovery patch, runtime
verification and final source bindings belong under
`D:/Compact_Evidence/candle/runs/root-lock-security-20261025-c24a6353/`.
The unmodified native gate `pwsh -NoProfile -File .tools/verify-before-push.ps1`
passed on Rust 1.97.1/MSVC with `CARGO_INCREMENTAL=0` and two build jobs:
585 Rust tests passed, 14 were ignored, all library/example checks and strict
Clippy passed, and both PowerShell hosts passed 197 release assertions.
Format, module layout, model-free controls, routing, overlays and whitespace
also passed. This proves the selected local Done When. Clean-commit verification
and the confirmed remote identity belong to `publication.json` in that run.
The supplemental datasets unit/doctests call the Hub; their request/cache
activity was not instrumented, so their success is not deterministic offline
or model-acquisition proof. They are not repeated by the required gate.
That supplemental run passed one unit test and two doctests. No Candle WSL
replay or production-model numerical run was performed in this batch.

Published Candle `81e1aa28`, GenEye `3200310f`, all sealed packages, frozen
inputs, original limits and consumed allowances retain their original bytes.
This lock repair does not repin or qualify a consumer. Closeout/Publish selects
scoped Candle publication through its guarded helper. Package assembly, model
execution, installation and activation remain held.

---
AI-edited: 2026-10-25 | agent=Codex/root | task=root-lock-security-publish | change=bind compatible security proof to scoped guarded publication
