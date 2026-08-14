# Candle Fork Overlay Registry

This fork keeps reusable framework changes in Candle while preserving the
review and release identity of each independently promoted feature family.
The registry is the authority for overlay ownership; it is not an application
integration plan.

## Registered overlays

| Overlay | Manifest | Current boundary |
| --- | --- | --- |
| LFM2-VL/MMProj | `docs/lfm2-vl/MOD_MANIFEST.md` | Proven model, loader, processor, fixture, and verification work |
| SnapFlash-derived diffusion | `docs/snapflash/MOD_MANIFEST.md` | Generic three-component SDXL LoRA transaction, controlled unsupported flash-attention failure, and exact residual/opt-in `text_time` UNet conditioning |

The repository-wide `scripts/verify-fork-overlays.sh` gate requires every
baseline-to-current path to belong to at least one registered manifest. Each
overlay-specific verifier may validate only its own paths, so unfinished work
in one overlay cannot silently become release evidence for another.

The `candle-overlays-mvp-0.2.0` closeout admits only build or verification
fixes, reproducibility and packaging changes, release metadata, distribution
security fixes, and documentation corrections that reflect actual behavior.
Its exact scope and non-goals are frozen in
`docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md`.

## Dependency direction

The permanent direction is:

```text
SnapFlash-Server experiment
    -> generic Candle primitive
    -> SnapFlash-Server reconsumption
    -> EdgeSymbio product acceptance
```

For LFM2-VL it is:

```text
Candle implementation -> EdgeSymbio integration -> optional SnapFlash use
```

## Coordinated progress

The table is the concise cross-repository lineage map. Detailed implementation
and verification narratives live in each repository's history/proof records;
later branch heads do not relabel the acceptance revision shown here.

| Order | Repository | Focused result | State / release condition |
| --- | --- | --- | --- |
| 1 | Candle | Overlay registry plus public LFM2-VL hybrid loader | Complete and published at `c0fb3a9fe098e50d07ec1b749c77015d7bd8d9a5` |
| 2 | EdgeSymbio | Exact-revision pin plus separate proof-only 450M CPU/F32 LFM2-VL lane | Complete and published at `d535a4f56f5a8e06407cb4b8f5be0df7f3121327` |
| 3 | Candle | Generic three-component SDXL LoRA transaction | Complete and published at `37584ecd2738ba1eb4ec4c1ab218667681f54973` |
| 4 | SnapFlash-Server | Reconsume Candle LoRA and delete duplicate tensor/transaction code | Complete and published at `6e64320fe26e7c3be91262bc0dac99ce53f4c628` |
| 5 | EdgeSymbio | Reconsume Candle LoRA and add both SDXL text encoders | Complete and published at `633f774a3690df5a8a35b6cac000df4b390316d5` |
| 6 | SnapFlash-Server | Adopt typed immutable runtime context, retained-file checks, and completion-last publication | Complete and published at implementation `d66c1c35158aca7b37e6e1d82e527334b209d93a`; proof-record head `b83db70ba4027535e4e55f6509e6011feeead850` |
| 7 | Candle | Evaluate and harden the existing ControlNet residual hook | Complete and published at `95ac9ff815fbac4f252b4ef6780b5e4a7843f328`; model-level numerical parity remains the separate INT-5 fixture gate |
| INT-5A | SnapFlash-Server | Reject unsupported official-style attention/`text_time` inventories before payload load or mutation | Complete and published at `9bc58ccaef77e7ceac0ab4e75a1a4c93acc1cdff` |
| INT-5B | Candle | Add opt-in generic SDXL pooled-text/time-ID addition conditioning | Complete and published at `ba1e8acc142c4683995e4cdbc8b1d933c81e96c6` |
| INT-5B.1 | Candle | Match Diffusers' F32 timestep projection and lower-precision learned-MLP cast order | Complete and published at `aed7f062bbfb825675efaf21c98029983312d336` |
| INT-5C/D | SnapFlash-Server | Consume the public conditioning hooks, implement the faithful ControlNet graph, and prove tiny differential plus installed Canny/Depth behavior | Complete and published at `b90f7c6bb76f1d73c70cd69e483fdfb1278de4ca` |
| REL-8A | Candle | Expose deterministic named-component LoRA write failure only to opted-in consumer tests | Complete and published at `1660f9fca8d6c8eb70937791e796203527f7be26` |
| REL-8B | SnapFlash-Server | Repin the exact six-package graph and prove later-component rollback through the application wrapper | Complete and published at `a6eaffb3f4ffdc465192dd293c61ed0ae7a4ca95` |

Detailed current completion conditions live only in the owning repository's
active TODO. This table preserves order and state without duplicating those
implementation contracts.

Candle must never depend on EdgeSymbio or SnapFlash-Server. Candle owns model,
tensor, loader, preprocessing, scheduler, and mutation primitives. It does not
own application request schemas, HTTP or Tauri types, queues, filesystem
policy, resource brokers, proof-report schemas, licensing allowlists, or
product-specific names.

## Shared-path registry

An overlapping path must be listed here before two overlay manifests may claim
it. Every change to such a path must state which overlay owns each hunk and
must pass both affected focused gates plus the repository-wide overlay gate.

<!-- shared-paths:start -->
- `.github/workflows/rust-ci.yml`
- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
- `CHANGELOG.md`
- `candle-examples/Cargo.toml`
- `candle-transformers/Cargo.toml`
- `candle-transformers/src/models/mod.rs`
- `candle-transformers/src/models/stable_diffusion/mod.rs`
- `docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md`
- `rust-toolchain.toml`
- `scripts/release/test-write-candle-overlays-receipt.ps1`
- `scripts/release/write-candle-overlays-receipt.ps1`
<!-- shared-paths:end -->

Shared registration permits coexistence; it does not let one overlay claim
another overlay's implementation or proof. Generic public names are required
in Candle. `Snapflash*`, EdgeSymbio report types, application paths, and
product policy remain outside the framework API.

## Review contract

1. Add a path to exactly one overlay manifest before staging it.
2. If two overlays genuinely touch the same path, register it above and list it
   in both manifests.
3. Run the affected overlay verifier independently.
4. Run `bash scripts/verify-fork-overlays.sh` before publication.
5. Never move or reuse the immutable `lfm2-vl-mvp-0.1.0` tag for a composite
   runtime release.
6. Create any future composite tag only after both consumer repositories pin
   the same exact Candle revision and pass their local acceptance gates.

---
AI-edited: 2026-08-13T20:08:58-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity | change=kept the overlay map concise and delegated detailed evidence to owned histories
