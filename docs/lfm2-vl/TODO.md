# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. Execute these items in order unless an earlier gate exposes a correctness prerequisite. All release proof is local: do not invoke, inspect, or depend on GitHub Actions or another hosted runner.

## P4 — CUDA and Distinct-Device Execution

P3 CPU-F32 parity, P4.1's public device-policy route, P4.2's tiny native
CUDA/distinct-device proof, and P4.3's official 450M CUDA parity are green.
P4.4 is now the active product task.
Do not install or update a CUDA toolkit implicitly; keep every build and model
run memory-bounded and sequential.

### P4.4 — Optimize Only Measured Bottlenecks

What: Profile and improve one proven CUDA bottleneck at a time.

Why: Speculative changes to interpolation, attention dtype, crop batching, transfers, or Q8 kernels would enlarge the risk surface after parity.

When: Now that P4.3 is green.

Where: Only the measured hot path and its focused benchmark/regression.

How: Use the opt-in `--timings` stderr diagnostic with repeated warm-up and a
fixed official 450M fixture; build the prompt with verified newline bytes and
require a quiet-host census with no Cargo/rustc, model, or llama process;
isolate decode/cache from load and vision cost; change one generation
bottleneck; replay CPU and CUDA parity; compare median, spread, and Job
memory; retain the change only when the measurement improves without contract
drift. The first stage baseline is captured in `PARITY.md`; the remaining
proof is a clean decode/cache microbenchmark.

Done when: Each retained optimization has a reproducible improvement and unchanged correctness evidence; P4 closes with a documented supported placement/dtype matrix.

Verification: Before/after bounded local benchmark with repeated warm-up,
explicit variance bound, and memory evidence; CPU/CUDA parity replay;
focused and full affected gates; no change to JSON evidence or cache-reset
semantics.

## R1 — Release Discoverability and Supported Matrix

What: Expose the proven LFM2-VL example from the repository entry points and state the actual support boundary.

Why: The dedicated example documentation is strong, but root discoverability trails the implementation and could imply unsupported lower-bit, video, or batching behavior.

When: After P4 is green; this is release polish, not a P3/P4 blocker.

Where: Root `README.md`, a concise `candle-vlm/README.md`, and `candle-examples/examples/lfm2-vl/README.md`; update another documentation index only when it owns the relevant link.

How: Add one root LFM2-VL example entry, introduce the crate briefly, link to the detailed example, publish the proven platform/artifact/device/dtype matrix, and name lower-bit MMProj, video, and true text batching as future work.

Done when: A new user can find the example and distinguish proven, optional, and unsupported modes without reading project-history files.

Verification: Root and mod-owned relative-link checks; README command smoke against `--help`; supported-matrix review against `PARITY.md`; complete diff inspection.

## C1 — Split Large Modules at Proven Seams

What: Reduce context and merge pressure in the largest LFM2-VL modules without changing public behavior or serialized evidence.

Why: `gguf.rs`, `weights.rs`, `runner.rs`, `processor.rs`, `model.rs`, `lfm2.rs`, `prompt.rs`, `siglip2.rs`, and `native_loading.rs` each exceed 1,200 lines. Splitting without measured seams would add indirection; leaving unrelated responsibilities together impedes review and scaling. The context bank now separates native model math from checkpoint loading, so a physical split is not a substitute for the P3/P4 product gates or R1 release handoff.

When: After P3, P4, and R1. The only earlier extraction allowed is the smallest proven device-policy seam required by P4.1.

Where: `candle-transformers/src/models/lfm2_vl/gguf.rs`, `candle-transformers/src/models/lfm2_vl/weights.rs`, `candle-examples/examples/lfm2-vl/runner.rs`, `candle-vlm/src/lfm2_vl/processor.rs`, `candle-transformers/src/models/lfm2_vl/model.rs`, `candle-transformers/src/models/lfm2.rs`, `candle-vlm/src/lfm2_vl/prompt.rs`, `candle-transformers/src/models/siglip2.rs`, `candle-examples/examples/lfm2-vl/native_loading.rs`, their nearest tests, and the affected `summary_bank.json` routes.

How:

1. Measure line/context size and call/test ownership before editing.
2. Split GGUF metadata/name normalization, tensor decode/layout, and model construction.
3. Split native weight inventory validation, canonical shape/layout conversion, and builder wiring.
4. Split runner artifact evidence, deterministic generation, trace serialization, and report emission.
5. Split processor/prompt only at image preprocessing versus token/span validation; retain shared checked-arithmetic helpers.
6. Re-measure routes and remove only proven duplication; do not create a generic VLM framework.

Done when: Each selected module has one clear responsibility, public APIs and evidence schemas are unchanged, routes load no more context than before, and every prior focused/full test remains green.

Verification: Before/after size report; import/compile checks; deterministic fixture and trace hashes where applicable; full local baseline; complete diff inspection.

## C2 — Adopt Gknome Safely Across Native Windows and This Linked Worktree

What: Integrate only useful Gknome project/context controls while preserving this mature repository's authority and safely refusing its Linux-absolute `.git` pointer.

Why: Native Windows is the normal product workflow, but this checkout remains a WSL-owned linked worktree even though it is now intentionally attached to local `main`. Windows Git still cannot resolve the Linux absolute pointer. The latest dry run correctly applied nothing and found four authority conflicts; bypassing them could overwrite project policy or context routing.

When: Last among the current deferred items, after product/release gates and only after Gknome produces a reviewed zero-conflict mature-repository plan. Do not use repair or implicit template replacement.

Where: Gknome at `C:\Users\jc816\OneDrive\Desktop\Gen-App\gknome`, this repo's non-secret generated controls, and `summary_bank.json`. Never read or replace `.tools/.secrets/`.

How:

1. Define a preserve-or-reviewed-merge contract for existing `.gitignore`, `AGENTS.md`, and `README.md`.
2. Map generated context routes onto existing groups or perform a staged reviewed merge; preserve the 256 KiB budget unless measured splitting proves a change.
3. Rerun the dry plan and require `existing_repository=true`, zero runtime/cache/secret inputs, explicit linked-Git refusal, byte-preserved authorities, and zero unresolved conflicts.
4. Apply only the reviewed plan as a separate action, then run generated project/context tests and add only accepted paths to the mod manifest and context bank.

Done when: Dry-run and adopted-project tests pass, no authority/runtime/secret path is treated as template input, the `.git` topology is reported truthfully, the context bank remains valid, and no commit/push/remote mutation occurs implicitly.

Verification: Gknome adoption/layout/inventory tests; zero-conflict plan JSON; before/after authority hashes; generated-file diff; context/project tests; WSL Git status when available; Candle summary-bank/mod-manifest verifiers; secret-tree exclusion with zero value exposure.

---
AI-edited: 2026-08-11T20:10:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=p4-4 | change=recorded the first stage-timing baseline and narrowed the active optimization to a measured decode/cache microbenchmark
