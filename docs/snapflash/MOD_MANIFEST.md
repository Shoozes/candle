# SnapFlash-Derived Candle Overlay Manifest

This manifest owns only generic diffusion primitives deliberately promoted
from application experiments into Candle. SnapFlash-Server remains the donor
and first regression witness; EdgeSymbio remains the product acceptance owner.

## Current state

No diffusion runtime primitive has been promoted yet. The current overlay is
boundary scaffolding for the ordered three-repository migration. It does not
claim LFM2-VL files, application code, or release evidence.

## Fork-origin files intentionally modified

| Path | Ownership |
| --- | --- |
| `CHANGELOG.md` | Shared fork changelog; this slice registers the overlay boundary while the LFM2-VL slice records its public loader promotion. |

## Overlay-owned additions

- `docs/FORK_OVERLAYS.md`
- `docs/snapflash/MOD_MANIFEST.md`
- `scripts/verify-fork-overlays.sh`

## Promotion rules

- Use generic Candle names; never expose `Snapflash*` or EdgeSymbio product
  types in Candle APIs.
- Validate every component and immutable base copy before the first mutation.
- Keep API schemas, Tauri/Axum code, queues, catalogs, licensing policy,
  filesystem resolution, resource claims, and proof JSON in applications.
- Do not pin SnapFlash-Server to the fork until the promoted API and exact
  integration revision exist.
- Add source paths here only in the focused Candle promotion commit that owns
  them. Shared paths must also appear in `docs/FORK_OVERLAYS.md`.

## Never publish

Models, adapters, generated images, local caches, `.tools/`, secrets, runtime
logs, and application artifacts are not part of this overlay.

---
AI-edited: 2026-08-12T12:42:54-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=three-repo-boundary | change=created empty diffusion promotion boundary without importing application code
