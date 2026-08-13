# SnapFlash-Derived Candle Overlay Manifest

This manifest owns only generic diffusion primitives deliberately promoted
from application experiments into Candle. SnapFlash-Server remains the donor
and first regression witness; EdgeSymbio remains the product acceptance owner.

## Current state

Two reusable diffusion boundaries are implemented. Candle owns validated SDXL
LoRA tensor parsing and a rollback-capable replacement transaction over the
UNet and both text encoders. It also validates the existing UNet
additional-residual hook as an exact, configuration-derived tensor contract
before ControlNet values are added. The LoRA transaction retains independent
base tensors, computes adapter B from base rather than adapter A, restores
exact base values, and rolls every component back if a later write fails under
the consumer's exclusive model-execution boundary.

Applications still own filename and directory policy, source licensing,
adapter catalogs, Kohya/model-family name conversion, resource admission,
request/report schemas, and live-generation proof. No application code or
production model artifact is part of this overlay.

## Fork-origin files intentionally modified

| Path | Ownership |
| --- | --- |
| `CHANGELOG.md` | Record the public LoRA transaction and additional-residual contract. |
| `candle-transformers/src/models/stable_diffusion/mod.rs` | Export the generic LoRA parser and mutable transaction modules. |
| `candle-transformers/src/models/stable_diffusion/unet_2d.rs` | Fail closed on malformed additional down/mid residual inventories before ControlNet addition. |

## Overlay-owned additions

- `docs/FORK_OVERLAYS.md`
- `docs/snapflash/MOD_MANIFEST.md`
- `candle-transformers/src/models/stable_diffusion/lora.rs`
- `candle-transformers/src/models/stable_diffusion/mutable.rs`
- `scripts/snapflash/verify-mod-manifest.sh`
- `scripts/verify-fork-overlays.sh`

## Public behavior boundary

- `SdxlLoraComponent` identifies UNet, text encoder 1, and text encoder 2.
- `parse_sdxl_lora_pairs` recognizes paired down/up tensors plus optional
  positive finite alpha, and rejects unknown layouts or incomplete pairs.
- `SdxlLoraTargetResolver` leaves application/model-family key conversion in
  the consumer and makes unmatched targets explicit.
- `VarMapSwapTransaction` prepares every component before mutation, binds a
  plan to the current revision and exact live snapshots, applies all writes as
  one rollback-capable operation, and clears to independent base copies. The
  consumer must hold its exclusive model execution/mutation lease throughout
  prepare and apply; Candle does not own application inference admission.
- `canonical_lora_tensor_sha256` hashes shape plus canonical finite F32 values
  under the `candle-sdxl-lora-tensor-f32-v1` contract. Per-target base, delta,
  and merged hashes let consumers compare the same adapter without importing
  application proof schemas into Candle.
- `UNet2DConditionModel::forward_with_additional_residuals` preserves its
  existing signature and `None` fast path while requiring the exact skip
  inventory derived from the configured down blocks and layers. SDXL's three
  blocks with two layers require nine down residuals. Every down and mid
  residual must exactly match the receiving tensor's shape, dtype, and device;
  broadcastable or otherwise mismatched tensors return controlled errors.

## Behavior provenance

- Primary behavior donor: `Shoozes/SnapFlash-Server` at
  `de68a751a055d55caf9daecf19e3733719cecbf0`, specifically
  `source/src-tauri/src/engine/loader/lora.rs` and `head_swap.rs`.
- Independent rollback/base-copy witness: `Shoozes/EdgeSymbio` at
  `d535a4f56f5a8e06407cb4b8f5be0df7f3121327`, specifically its existing
  `source/backend/src/image_snapflash/head_swap.rs` transaction tests.
- This is a fresh Candle-native implementation. No source block was copied
  from either application. No standalone license file was present at the
  reviewed SnapFlash-Server tip, so the donor is used only as a behavioral
  reference until its owner records a code license.

## Promotion rules

- Use generic Candle names; never expose `Snapflash*` or EdgeSymbio product
  types in Candle APIs.
- Validate every component and immutable base copy before the first mutation.
- Keep API schemas, Tauri/Axum code, queues, catalogs, licensing policy,
  filesystem resolution, resource claims, and proof JSON in applications.
- Consumers must pin one exact published Candle revision; never a moving
  branch or an uncommitted worktree.
- Add source paths here only in the focused Candle promotion commit that owns
  them. Shared paths must also appear in `docs/FORK_OVERLAYS.md`.

## Completion contract

Done when all of the following are true:

- UNet-only, text-only, and mixed three-component adapters pass deterministic
  tensor tests.
- Adapter A -> B is computed from immutable base, and clear restores exact
  base values.
- Injected failures in components 2 and 3 restore every earlier write.
- Revision exhaustion fails before the first tensor write.
- Missing pairs, invalid alpha/rank, shape mismatches, unknown tensor names,
  duplicate/unmatched targets, non-finite strength, all-zero effect, and stale
  plans fail before an invalid state can commit.
- BF16 1x1 convolution targets and canonical target/delta hashes are proven.
- Short, long, broadcastable-shape, dtype-mismatched, and malformed mid
  residuals fail before addition; `None` and exact zero residuals preserve the
  original UNet result.
- Focused transformer tests and strict Clippy, the independent overlay
  verifier, repository-wide overlay union, summary-bank checks, and the full
  local Candle gate pass before direct-main publication.

## Never publish

Models, adapters, generated images, local caches, `.tools/`, secrets, runtime
logs, and application artifacts are not part of this overlay.

---
AI-edited: 2026-08-12T20:41:11-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-round-7 | change=registered and bounded the generic ControlNet residual contract
