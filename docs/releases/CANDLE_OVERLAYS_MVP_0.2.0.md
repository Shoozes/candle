# Candle Overlays MVP 0.2.0 Release Contract

The selected product is an independently usable five-overlay source release,
proposed annotated tag `candle-overlays-mvp-0.2.0`. It is an MVP, not an LTS
commitment. No tag, hosted release, asset publication or new model proof is
claimed by this contract. Stop at maintenance after the release; research does
not extend its completion criteria.

## Candidate freeze

The starting published source is `63f0d2cd275e4653b268939b429cbb9af25aba1a`.
This release adds only tooling, tests, scope/support records and ownership
updates; existing model arithmetic is unchanged. Unpublished practical Short
source is parked with its proof and excluded. The final commit/tree pair must
be reviewed externally after a separately authorized clean checkpoint; it
cannot be embedded self-referentially in the commit it identifies.

Reviewed source expectations are in
[`candle-overlays-mvp-0.2.0.expectations.json`](../../scripts/release/candle-overlays-mvp-0.2.0.expectations.json).
They freeze the compiler, Cargo, upstream base, canonical LF hashes of the
lock/toolchain/workflow/contract/registry/all five manifests, and per-overlay
and union inventories before verification. The writer compares actual values
with those inputs; it never learns its expected counts from a verifier result.
Changing an expectation requires source review and another candidate gate.

- Upstream integration base: `c68b24997319b8d76e5a4e775dab197c7f26322a`.
- Native toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)` and
  `cargo 1.97.1 (c980f4866 2026-06-30)`.
- Five overlays: 326 union paths, 31 registered shared-path declarations and
  30 observed overlapping paths. LFM2-VL has 215 paths
  (20 modified/195 added); diffusion has 22 (9 modified/13 added). The frozen
  JSON records each remaining overlay's independent counts.
- Historical `lfm2-vl-mvp-0.1.0`: annotated object
  `7de0b4dc5f914e9b1d317da7f76f677d9f32b844`, target
  `ff885586f6d44a3d9b9ac1724032cdf5f0155384`. Both local and remote object and
  target must remain unchanged.

## Supported behavior and limits

The primary host is native Windows x64/MSVC. CPU builds remain independent of
CUDA. CUDA is optional and limited to recorded device/precision/input receipts.
The source package contains code, deterministic fixtures and instructions;
production model weights, dependency caches, runtime DLL installations and
consumer applications are separate inputs. No model or native dependency is
downloaded by release verification.

| Overlay | Included and proven boundary | Limits that remain explicit |
| --- | --- | --- |
| LFM2-VL/d1 | Config-driven LFM2 text, native and GGUF multimodal loaders, SigLIP2 NaFlex, direct Q8 MMProj, hybrid/text-only d1 and optional Q8Rows. Retained CPU-F32 official component proof, 450M placement/dtype matrix, source-bound d1 CPU/CUDA parity and deterministic fixtures. | 1.6B component proof is not universal end-to-end/device support. Native 3B/400M MMProj, arbitrary adapters and trained 2.6B quality remain gated. [Parity](../lfm2-vl/PARITY.md), [d1 loading](../lfm2-vl/D1_LOADING.md). |
| SnapFlash-derived diffusion | Generic SDXL LoRA transactions/rollback, ControlNet residual admission, opt-in `text_time`, cast ordering, VAE malformed-input errors and controlled unsupported attention behavior. | Consumer owns files, licensing, request/resource policy and live generation. SnapFlash keeps Candle `1660f9fca8d6c8eb70937791e796203527f7be26`; this release does not repin it. [Manifest](../snapflash/MOD_MANIFEST.md). |
| GPT-OSS | Experimental local F32/MXFP4 loading/session, packed CPU/CUDA operations, bounded short numerical/replay and ownership receipts. Latest execution remains `2adfa4a3`; original source identities and numerical bounds are retained. | No enabled consumer/default, end-to-end chat/coding/route, physical 16k/32k or long-context quality claim. Short CUDA decode is about 79ms/forward, not chat latency; 8k fixed-token work takes 637.218s, beyond 90s/600s routes. GPU samples approach 20e9 and can miss peaks. [Characterization](../gpt-oss/CHARACTERIZATION.md), [capacity](../gpt-oss/CAPACITY.md). |
| Fork compatibility | Retained `ug` APIs, onig selection, scoped Auto/GenericTiled/GenericRowwise CPU quantized matmul, checked generic boundaries and exact scheduling parity. | Native CUDA/ug settings and historical qualification remain source-bound; no Metal execution claim. [Manifest](../fork-compat/MOD_MANIFEST.md), [sync](../lfm2-vl/UPSTREAM_SYNC.md). |
| Qwen3.5 | Candle Q8_0/F32 text primitives, CPU numerical parity, recorded scoped CUDA qualification, failure/cancellation recovery and opt-in grouped projection fixtures. | Grouped model qualification, Edge activation, 27B/IQ/MTP, new formats and long context remain separate. [Status](../qwen35/STATUS.md). |

No consumer has to adopt the newest Candle revision. Edge's Windows core MVP
keeps its existing `a0c70e1d` graph, Q8/Nomic defaults and disabled GPT profiles;
the broader GenAce/VL profile is separate. GenEye keeps `3fa7075e`. Symbio's
engine-manager MVP keeps Codex 0.154.0; correct rejection of an invalid model
patch is not autonomous-coding qualification. Each worker records its exact
source/package/lock tuple and proof independently. These releases do not depend
on GenAce cutover or a fourth application release.

## Native acceptance and replay

Use PowerShell 7, Git for Windows (Git and Bash), the pinned Rust/MSVC toolchain,
Python and a prepared dependency cache. Canonical Git owns `C:\DevStuff\candle`.
An empty-cache setup is a separate authorized online operation; `--offline`
does not fetch missing packages. Clear `FORK_OVERLAYS_REPO_ROOT` so inventory
cannot be redirected to another checkout. No WSL checkout is used by the native
release route. WSL replay is secondary; unavailable caches are reported, not
silently fetched. Hosted CI is not release proof.

```powershell
$env:RUST_TEST_THREADS = '2' # Process-local doctest linker concurrency bound.
pwsh -NoProfile -File .\scripts\release\test-write-candle-overlays-receipt.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release\test-write-candle-overlays-receipt.ps1
pwsh -NoProfile -File .\.tools\verify-before-push.ps1
cargo test --locked --offline -j 2 -p candle-transformers --features test-utils --test stable_diffusion_mutable_tests
```

The maintained gate includes formatting, locked/offline core/nn/transformer/VLM
and example checks, maintained tests, strict scoped Clippy, GPT model-free
controls, d1 receipt tests, Summary Bank/layout, overlay union and whitespace.
Both release-contract parser lanes run inside that gate. These commands run
deterministic fixtures, not production model inference. The writer additionally
runs LFM, diffusion and GPT manifest checks and compares native inventories for
all five overlays. There is no separate Qwen or fork-compat shell verifier;
their manifests participate in the five-overlay union and frozen comparison.

This is the maintained-crate acceptance matrix, not a whole-upstream-workspace
or all-platform certification. The unshipped August candidate's wider workspace
diagnostics remain historical and are not relabeled as current execution.
Inherited runtime Python/network-dependent datasets and unrun platforms are
outside the advertised matrix. Numerical evidence is reused only through its
original implementation/input identities; this release does no new inference.

## Source package and identity receipt

After the clean checkpoint is separately authorized, freeze its exact commit
and tree in the external approval/evidence record. From that reviewed source:

```powershell
pwsh -NoProfile -File .\scripts\release\write-candle-overlays-receipt.ps1 `
  -Stage Candidate `
  -ExpectedHead <reviewed-40-character-commit> `
  -ExpectedTree <reviewed-40-character-tree> `
  -ExpectationsPath .\scripts\release\candle-overlays-mvp-0.2.0.expectations.json `
  -ExpectedExpectationsSha256 <independently-reviewed-canonical-LF-freeze-digest> `
  -SourcePackagePath <existing-external-directory>\candle-overlays-mvp-0.2.0-candidate.zip `
  -ReceiptPath <existing-external-directory>\candidate.json
```

Candidate mode is local-only and clearly reports `publication_verified=false`;
it does not require or create the proposed tag. It verifies clean named `main`,
the script's own checkout root and native Git root, reviewed source/tree,
upstream ancestry, toolchain, all frozen input hashes/counts, owning verifiers
and the immutable local historical tag. Alternate replay checkouts require an
explicit reviewed `-ExpectedRepositoryRoot`; defaults stay canonical.

The writer calls native `git archive --format=zip` at that exact commit with
prefix `candle-overlays-mvp-0.2.0/`. Only tracked source is packaged; `.git`,
ignored artifacts, caches and production weights are absent. Receipt v2 binds
archive bytes/hash, source/tree, expectations digest, inputs and all inventories.
It omits local paths, refuses existing outputs, uses fresh temporary siblings
and writes the completion receipt last. A failed receipt move removes only the
package created by that invocation after checking its hash; a subsequently
changed package is retained with an explicit error. Release/source ZIP replay
uses a Git checkout at the receipt commit for ancestry gates; the archive itself
supplies code and fixtures.

## Publication and maintenance

Commit, push, annotated tag and GitHub Release/assets require explicit authority
for this batch. Save only reviewed named files; main publication uses Candle's
`.tools/gitpush.ps1`. Never force or rewrite historical refs. Any tag publisher
must preserve the historical object and target. Proposed LTS support duration
and repository rule changes remain separate owner commitments.

After authorized main/tag publication, run the same writer with `-Stage Published`
and fresh external ZIP/receipt paths. It additionally requires remote `main`,
local/remote annotated release objects/targets and both historical refs to agree.
Upload the source ZIP, published identity receipt and this contract; verify
downloaded asset bytes/hashes against the receipt. Keep post-publication proof
external so success does not create another source candidate.

Retain one compact external evidence index linking original required numerical
receipts, exact command/runtime/results, final commit/tree, ZIP and receipt
hashes, remote/tag/asset verification, limitations and recovery. Preserve unique
proof; omit weights, compiler caches and duplicate installed engines. On install
or replay failure, preserve the failed evidence, compare the frozen tuple and
restore the previous consumer pin/package. Do not retry inference or change
providers implicitly. After the supported release passes, default to maintenance.

## Superseded August candidate

The unshipped two-overlay candidate used parent
`dca9849584e377cebc1da40de966d050733f3bbf`, base
`6f74e7c390c717f8fd34f23ce02aceb058173370`, inventories LFM 156, diffusion 20,
union 167/shared 13 and lock hash
`9b7aa15899ae8acf7b1a09b951ddba2f16462137eee2fed0db863a9d84707175`.
Those are historical inputs, superseded for the current candidate. The prior
[contract at `63f0d2cd`](https://github.com/Shoozes/candle/blob/63f0d2cd275e4653b268939b429cbb9af25aba1a/docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md)
and all original receipts remain recoverable and unchanged.

---
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=overlay-mvp-release | change=superseded unshipped candidate and froze independent five-overlay source release
