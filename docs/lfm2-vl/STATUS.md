# Candle Fork Current Status

## Current phase

Independent overlay MVP 0.2.0 closeout. Baseline is published clean `main`
`63f0d2cd275e4653b268939b429cbb9af25aba1a`; upstream integration base remains
`c68b24997319b8d76e5a4e775dab197c7f26322a`. This record describes the verified
pre-publication candidate. The external identity receipt owns its eventual
checkpoint/tree/publication state; this source file claims no tag or release.

Scope is the five existing overlays and release tooling, source package,
support matrix and replay instructions. No new model capability is required.
[Release contract](../releases/CANDLE_OVERLAYS_MVP_0.2.0.md) owns the exact
support boundary; [TODO](TODO.md#release-blockers) owns finite blockers.

## Proven behavior

Existing LFM/VL/d1, retained Q8/F32, Q8Rows, diffusion, compatibility and Qwen
implementation/proof boundaries remain unchanged. See the release support
matrix, [d1 loading](D1_LOADING.md), [Qwen status](../qwen35/STATUS.md), and
[GPT experimental status](../gpt-oss/STATUS.md). Original receipts keep their
runtime source, inputs, numerical limits and cleanup evidence.

Edge's core uses its existing `a0c70e1d` graph; GenEye retains `3fa7075e`;
SnapFlash retains `1660f9fc`; Symbio retains Codex 0.154.0. These independent
consumers are not repinned by this source release.

## Last green verification and active files

The current 16-file working candidate passes the complete native gate, including
188 release-contract assertions on each of PS7 and PS5.1, 461 model-free GPT
controls, maintained Rust tests/checks, core doctests, example tests and strict
Clippy. Two additional diffusion transaction/rollback integration tests and all
three owning manifest checks pass. Native inventory is 326 paths/five overlays/
31 shared declarations (30 actual overlaps). Exact commands/results/logs are in
`artifacts/releases/mvp-0.2.0-20261014/`; the 16 GiB/1,800-second Job exits zero
after 256.679 seconds, peak Job memory 3,012,374,528 bytes. Its native launcher
and invoker are absent after cleanup. Final source binding follows a separately
authorized clean checkpoint; a real candidate archive is not yet claimed.

Active files: release writer/test/expectations/contract, native gate, owning
LFM/diffusion inventories/verifiers, current status/backlog/history, registry,
GPT research disposition and Summary Bank release route. Experimental Short
source is parked outside the release source; its exact bytes/proof are retained.

## Known limitations and blockers

The real clean candidate package/receipt and publication proof remain pending. Commits,
guarded main publication, annotated tag and hosted assets require explicit
authorization for this batch. Existing model allowances are closed.

Only named native Windows lanes are advertised. WSL2 `Codex-Compat` overlay
inventory replay passes; locked/offline `candle-core` check exits 101 because
`accelerate-src` is not cached. No dependency was fetched. Metal is unrun.
GPT-OSS remains experimental: short parity is not chat/coding qualification;
637.218s of 8k fixed-token forward work exceeds 90s/600s routes; physical 16k/32k
is unqualified. Native 3B/400M MMProj and real-adapter quality remain deferred
with their original acceptance criteria.

## Exact next task

Authorize the reviewed scoped local checkpoint to bind the clean candidate
source archive and receipt, then prepare consolidated remaining native and
publication actions. Coordinate Edge installed-core and Symbio native-manager
proof independently. After publication, stop in maintenance; do not start the
deferred research backlog automatically.

Completed narrative and prior states are in
[HISTORY](HISTORY.md#2026-10-14---independent-mvp-scope-freeze).

---
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=overlay-mvp-release | change=froze independent MVP scope and retained historical proof
