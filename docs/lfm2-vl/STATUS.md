# Candle Fork Current Status

## Current phase and baseline

Verified ZIP parent repair on main, based on
`b70f5f3264d88413ee77e7e7c9827644590de3c7`. Scoped commit and guarded
publication are authorized; the final result is recorded in
`artifacts/publication/last-push.json`. Require ZIP 9.0.4 with defaults
disabled; preserve decoded CP437 NPZ/PyTorch keys. Pinned upstream stays
`c68b24997319b8d76e5a4e775dab197c7f26322a`; released 0.2.0 tag/assets stay
at `9a7c65c2ebca67e7997ae1a73b652e38d0a3781f`. Consumer graphs are untouched.

## Last green verification and proven behavior

Archive gate: 14 CPU tests; strict archive Clippy passes. An independent local
workspace builds all four consumer crates and loads NPZ via VarBuilder and a
tiny PyTorch fixture without a consumer override. Its audit has zero
vulnerabilities, with the existing paste unmaintained warning retained.
The complete native gate passes on the D: build/temp route: 585 Rust tests,
14 ignored; both PowerShell release-contract lanes pass. Runtime/precision,
Q8/Nomic and optional CUDA behavior are unchanged by this repair. Details and
held-pin adoption envelope: [ZIP_COMPATIBILITY](../fork-compat/ZIP_COMPATIBILITY.md).

## Active files and limitations

Twelve local paths: workspace dependency/lock, core npy.rs/pickle.rs/archive tests,
compatibility docs/manifest, STATUS/TODO/DECISIONS, overlay registry and Summary
Bank. The 12-file batch is selected for guarded main publication. Prior local
source hashes and commands are preserved in
`D:/Compact_Evidence/candle/runs/zip-parent-20261017/summary.json`; the final
commit, native gate and remote equality are bound in the adjacent
`publication.json` after the helper succeeds.

The workspace audit removes all three ZIP findings but retains five unrelated
h2/rustls findings. C: filled during d1 Clippy after 569 tests passed; the fresh
D: route passed with two Cargo jobs, no incrementals/debug symbols and scoped
TEMP/TMP. Existing C: caches stay preserved; Cargo's usage-tracking writes also
warned while C: was full. Space later recovered without a cleanup by this task.
WSL replay is unavailable: Codex-Compat's registered VHD path is missing.
No production-model/GPU run, download, activation or Edge qualification ran.

## Exact next task

Use the guarded publication receipt to select the immutable successor. Edge
owns review of intervening GPT-OSS runtime changes and adoption of that exact
pin. It must update all four locks and pass all four audits/consumer checks
before fresh
package/native qualification. Earlier allowances remain closed. The missing
WSL disk needs its owner. Preserve historical proof and retained build caches.

---
AI-edited: 2026-10-17T17:19:22-04:00 | agent=Codex/root | task=zip-parent-closeout-publish | change=bound authorized guarded publication and preserved consumer acceptance
