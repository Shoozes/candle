# Candle Fork Overlay Registry

This fork keeps reusable framework changes in Candle while preserving the
review and release identity of each independently promoted feature family.
The registry is the authority for overlay ownership; it is not an application
integration plan.

## Registered overlays

| Overlay | Manifest | Current boundary |
| --- | --- | --- |
| LFM2-VL/MMProj | `docs/lfm2-vl/MOD_MANIFEST.md` | Proven model, loader, processor, fixture, and verification work |
| SnapFlash-derived diffusion | `docs/snapflash/MOD_MANIFEST.md` | Generic three-component SDXL LoRA parsing, evidence, and rollback-capable mutable replacement |

The repository-wide `scripts/verify-fork-overlays.sh` gate requires every
baseline-to-current path to belong to at least one registered manifest. Each
overlay-specific verifier may validate only its own paths, so unfinished work
in one overlay cannot silently become release evidence for another.

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

The reusable loader parent is Candle
`c0fb3a9fe098e50d07ec1b749c77015d7bd8d9a5`. EdgeSymbio Round 2 is published
at `d535a4f56f5a8e06407cb4b8f5be0df7f3121327`; it pins that Candle revision and
passes the bounded 450M CPU/F32 token-level proof. The current SnapFlash-Server
behavior donor remains `de68a751a055d55caf9daecf19e3733719cecbf0` until it
consumes the LoRA promotion commit produced by order 3.

| Order | Repository | Focused result | State / release condition |
| --- | --- | --- | --- |
| 1 | Candle | Overlay registry plus public LFM2-VL hybrid loader | Complete and published at `c0fb3a9fe098e50d07ec1b749c77015d7bd8d9a5` |
| 2 | EdgeSymbio | Exact-revision pin plus separate proof-only 450M CPU/F32 LFM2-VL lane | Complete and published at `d535a4f56f5a8e06407cb4b8f5be0df7f3121327` |
| 3 | Candle | Generic three-component SDXL LoRA transaction | Current candidate; complete when local gates and guarded remote-main equality pass |
| 4 | SnapFlash-Server | Reconsume Candle LoRA and delete duplicate tensor/transaction code | Exact next consumer after order 3 publication; done when A -> B -> base and live regression proof remain green |
| 5 | EdgeSymbio | Reconsume Candle LoRA and add both SDXL text encoders | Held until order 4; done when both consumers report the same targets/delta hashes on one Candle revision |
| 6 | SnapFlash-Server | Adopt typed immutable runtime context, retained-file checks, and completion-last publication | Held until shared LoRA migration; done when direct, queued, and inpaint paths share the hardened boundary |
| 7 | Candle | Evaluate ControlNet hooks, then inpainting math | Later proposal only; no implementation starts without differential fixtures and a focused acceptance contract |

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
- `Cargo.toml`
- `CHANGELOG.md`
- `candle-examples/Cargo.toml`
- `candle-transformers/src/models/mod.rs`
- `candle-transformers/src/models/stable_diffusion/mod.rs`
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
AI-edited: 2026-08-12T16:05:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-3 | change=closed Edge acceptance and defined the rollback-capable Candle LoRA promotion gate
