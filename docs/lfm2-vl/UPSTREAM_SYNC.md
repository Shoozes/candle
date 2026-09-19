# Overlay vs Hugging Face sync inventory

Read-only. No merge. No gpt-oss overlay. No `candle-overlays-mvp-0.2.0` tag.

Fetched 2026-09-18 against `C:\DevStuff\candle`.

| Clock | SHA | Note |
| --- | --- | --- |
| Local overlay `main` (unpublished S3 merge) | `25d676f5663f152cf9371b405236d75fe110d14f` | `--no-ff` of exact HF `#3950` onto `238cc176`. |
| Published overlay `origin/main` | `238cc176e2a7283da88588fdef47276965d0022b` | Edge pin remains here until S4. |
| Live HF integration base | `7c2e89295dad4aeebc6ef7a92c255360b6957c2c` | Selected S3 SHA. `#3950`. |
| Prior overlay HF base (frozen 0.2.0 receipts) | `6f74e7c390c717f8fd34f23ce02aceb058173370` | Not the live union gate after S3. |
| `huggingface/main` tip | `ddf1b879dc3a1760cbcb3f3c4a7c6467850cec4a` | 2026-09-04 `Remove ug (#3954)`. Do not merge. |
| Compat baseline | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | Candle 0.11.0. Not a sync target. |

HF `main` has no gpt-oss paths. Overlay still has `candle-ug/`; the tip deleted it.

HF `#3899` Mxfp4 core dtype: **absent** on this fetched tip (no `mxfp4` / `#3899` in `huggingface/main` history or tree). That is later C, not this slice.

T2/T3 gpt-oss llama.cpp receipts remain gold. They are not Candle/Edge proof.

## 1. Edge would absorb `dca98495..2ec92ace`

13 overlay commits. Crate API Edge compiles:

| Path | Change | Public API break? |
| --- | --- | --- |
| `candle-transformers/src/models/lfm2.rs` | Test that deprecated `into_config` matches `try_into_config` | no |
| `candle-transformers/src/models/lfm2/config.rs` | `#[deprecated]` on `into_config`; signature unchanged | no |
| `candle-transformers/src/models/stable_diffusion/attention.rs` | Comment | no |
| `candle-transformers/src/models/stable_diffusion/lora.rs` | `sort_by` → `sort_by_key` | no |
| `candle-vlm/README.md` | `--offline` in example | no |
| `candle-core`, `candle-nn` | untouched | no |
| Remaining files | Overlay docs, 3B/Q8 proof-gap contracts, lock added on Candle side | no |

No `lfm2` / `candle-vlm` public API break. P2 may bump Edge to `2ec92ace`.

## 2. Upstream `6f74e7c..ddf1b879`

High-risk later commits (newest last):

| SHA | Subject | Flag |
| --- | --- | --- |
| `5814a6fa` | tokenizers `fancy-regex` over `onig` (#3952) | paired Edge `tokenizers` feature change |
| `d4d130c8` | Add caching to rust ci (#3953) | after #3952 |
| `e50eece1` | Bump cutile to v0.3.1 (#3959) | after #3952 |
| `ddf1b879` | Remove ug (#3954) | deletes `candle-ug`; overlay still has it |

Also in the 41-commit window: hf-hub 1.0, tokenizers 0.22.0 → 0.23.1 (still `onig` until #3952), CI feature-gates, cutile/MoE.

## 3. Shared paths (13)

| Path | Classification |
| --- | --- |
| `.github/workflows/rust-ci.yml` | both |
| `.gitignore` | ours-only |
| `Cargo.lock` | ours-only |
| `Cargo.toml` | both |
| `CHANGELOG.md` | ours-only |
| `candle-examples/Cargo.toml` | both |
| `candle-transformers/Cargo.toml` | ours-only |
| `candle-transformers/src/models/mod.rs` | ours-only |
| `candle-transformers/src/models/stable_diffusion/mod.rs` | ours-only |
| `docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md` | ours-only |
| `rust-toolchain.toml` | ours-only |
| `scripts/release/test-write-candle-overlays-receipt.ps1` | ours-only |
| `scripts/release/write-candle-overlays-receipt.ps1` | ours-only |

`both` = overlay-owned and HF touched after `6f74e7c`. `ours-only` = overlay-owned, HF did not touch.

## Overlay fork-origin vs HF (extra collisions)

HF also touched these LFM2-VL fork-origin paths: `.github/workflows/ci_cuda.yaml`, `README.md`, `candle-core/tests/custom_op_tests.rs`, `candle-examples/examples/lfm2/main.rs`, `candle-kernels/build.rs`. Keep overlay hunks (fork-PR CUDA skip, `cuda_i32_to_f32_cast`, MSVC `/Zc:preprocessor`).

HF did **not** touch `lfm2.rs`, `quantized_lfm2.rs`, `models/mod.rs`, `gguf_file.rs`, `cast.cu`, or SnapFlash SDXL sources. Zero overlay-owned additions exist on HF `main`.

## 4. Proposed next HF integration base (P3)

**Proposed:** `7c2e89295dad4aeebc6ef7a92c255360b6957c2c`  
`Config gate candle-examples' buildtime downloader (#3950)`.

Reason: newest HF commit that still keeps `tokenizers` **`onig`** and still has `candle-ug`. `#3952` is the first rejected later commit.

Rejected later (do not merge as floating `main`):

- `5814a6fa` #3952 fancy-regex — would force an Edge `tokenizers` feature change (`onig` → `fancy-regex`)
- `d4d130c8` #3953
- `e50eece1` #3959
- `ddf1b879` #3954 Remove ug

`#3952` does **not** apply at the proposed SHA. Edge can keep `tokenizers` 0.22/`onig` until a later chosen base includes #3952. The proposed SHA already has workspace `tokenizers` 0.23.1 with `onig`; that is a version unify, not a backend switch.

This SHA is merged locally as `25d676f5…` and is **not** S4-complete or
published. Do not treat the merge as consumer-green until overlay and Edge
compatibility gates pass. Keep the Edge pin on `238cc176…` until then.

---
AI-edited: 2026-09-18T00:35:00-04:00 | agent=Grok/root | model=grok-4.6 | effort=high | task=p1-p3-inventory | change=recorded overlay vs HF path table and proposed 7c2e8929 integration base
