# GPT-OSS Experimental Candle Status

## Assignment state

- Current phase: Task 3 bounded native performance qualification is complete
  for the observed-safe 2080-token total-context envelope. Short parity remains
  accepted; the optimized-release packet proves the 8/512/2048 prompt cases
  with a 32-token autoregressive reserve, exact 20,000,000,000-byte admission,
  fail-closed sampling, and post-run cleanup. The harness still refuses target
  contexts above 8192 tokens, but the clean 8160-prompt diagnostic exceeded the
  observed GPU ceiling and is not accepted. This optional Candle track does not
  change EdgeSymbio or symbio-code.
- Current baseline before this slice: `main` at
  `f470d2de1f9370815cab2f3ebf83ba81825f1716`, tree
  `154da36bce009ff058681113ce2ece9c24a9b71f`.
- The bounded implementation checkpoint is `4a4699981ed55bb11e857c58923c67133559145d`;
  the prior short-parity checkpoint was
  `ea5900f614c35c47822b8363ff18ee676ae2159a`.
- The accepted performance receipt executed that clean source checkpoint and
  records the exact release, model, tokenizer, GPU, and runtime identities.
  Documentation closeout is layered on top without changing the receipt's
  executed source identity.
- Task 1, Task 2, and the prior Task 3 packed-executor checkpoints remain
  accepted. No model bytes entered the repository.
- Product artifact identity: the owner-selected GGUF SHA-256 is
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
  The external file remains outside this checkout; it was read in place only
  at the owner-provided path and was not copied, downloaded, or committed.

## Accepted capability rows

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Exact artifact admission | Accepted | `GptOssGgufArtifact::open` hashes before parsing and admits only the selected SHA-256. |
| GGUF directory/config normalization | Accepted | Strict v2/v3 parser, `gpt-oss` to `gpt_oss` normalization, YaRN/context and dimension checks. |
| Packed MXFP4 ownership | Accepted | CUDA keeps expert blocks/scales as device U8 tensors and dispatches a dedicated packed kernel; no dense expert fallback exists. |
| Packed expert/router/attention execution | Accepted | CUDA component trace and direct packed output match the independent oracle at absolute tolerance `1e-4`. |
| Expert contribution/residual contract | Accepted | `Mxfp4ExpertOperation::forward_contribution` returns only weighted expert output; CPU and CUDA add it directly to the post-attention hidden state. |
| Post-attention/post-MoE traces | Accepted | CPU and CUDA expose both hidden states in the two-layer independent fixture, including sliding/full attention, sinks, unequal routing, and position 3. |
| CUDA forward/cache execution | Accepted | Uncached logits and cached prefill/decode match the oracle; exact logical cache usage is reported. |
| Explicit CUDA selection | Accepted | Public feature-gated config selects device index and rejects non-F32 activation dtype with typed errors. |
| Static/runtime resource admission | Accepted | Logical static weight bytes are checked before device creation; token and exact KV-byte admission happen before forward mutation. |
| Cancellation/rollback/eviction | Accepted | Cancellation before allocation and at admitted work boundaries leaves the retained cache unchanged; eviction releases logical cache state and retry succeeds. |
| CUDA narrowed-view equivalence | Accepted | Padded/narrowed CUDA views match the materialized-copy path; pre-launch device/dtype/shape checks run before dispatch and cudarc pointer guards remain alive through launch. |
| GGML MXFP4 wire normalization | Accepted | Real serialized fused and split GGUF fixtures normalize all 32 wire coordinates, preserve mixed signs/scales, and match an independent decoder before expert execution. |
| GGUF-to-weights assembly | Accepted | Fused and converter-style split synthetic fixtures assemble and execute on CPU; the exact owner-selected 13,792,638,656-byte artifact assembles under the 32 GiB resident bound. |
| Retained-file integrity and ownership | Accepted | Admission retains one identity-verified handle, Windows read-opens deny write/delete sharing, load reads avoid reopen/rehash, cancellation is chunk-aware, and the registry lease follows the live model/runtime owner. |
| CUDA validation and commit guards | Accepted | All public CUDA config fields validate before device setup; synchronization, finite-output validation, and a final cancellation checkpoint precede cache commit, with delayed failure/cancellation recovery tests. |
| Product artifact numerical parity | Accepted for bounded short CUDA gate | The exact short runner proves the pinned tokenizer IDs, real CUDA construction, cancellation rollback, prefill/decode parity, deterministic reset/replay, and registry teardown. The robust criterion requires exact top-1 agreement, top-k union log-probability error <= `0.25`, mean full-row error <= `0.01`, and probability-space total variation <= `0.01`; all three stages pass. This is not yet a cross-device, Q8, or maintained product-path claim. |
| Performance characterization harness | Accepted for bounded native packet | `gpt-oss-performance` defaults to optimized release, records build/model/tokenizer/GPU/runtime identities, enforces the exact 20,000,000,000-byte Edge ceiling with a fail-closed physical monitor, reports checked static+KV+workspace accounting, preserves cancellation/stop reasons, and writes cleanup evidence. The accepted packet passes 8/512/2048 prompt cases with 32 generated tokens; the 8160-prompt diagnostic is superseded after an observed physical-ceiling violation. No 32k or maintained production claim is made. |
| Quantized text/split dense MMProj | Deferred | The ordered post-CUDA task; no Q8/LFM2 defaults or Edge/Harmony integration were changed. |

## Proven evidence

- Independent fixture: `tests/fixtures/gpt_oss_task2/oracle.json`, fixture ID
  `synthetic-gpt-oss-task2-oracle-v1`, 2,690 raw bytes, SHA-256
  `db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04`.
  The pinned standalone Python 3.13.3 / NumPy 2.3.3 scalar reference covers
  packed expert output, router indices/weights, attention transitions,
  uncached logits, and cached prefill/decode logits at absolute tolerance
  `1e-4`.
- Independent two-layer correction fixture:
  `tests/fixtures/gpt_oss_task3_two_layer/oracle.json`, fixture ID
  `synthetic-gpt-oss-task3-two-layer-oracle-v1`, 10,525 raw bytes, SHA-256
  `0530081353cc7268d796ae070c4168ec4e6bb036f62d54670b975a43556bc173`.
  Standalone Python 3.13.3 / NumPy 2.3.3 values cover final logits and
  attention, router, expert-contribution, post-attention, and post-MoE traces
  for a four-token, two-layer sliding/full-attention sequence at absolute
  tolerance `1e-4`.
- CUDA proof ran on native Windows/MSVC with Rust/Cargo 1.97.1, CUDA/nvcc
  13.3, NVIDIA driver 616.92, and an NVIDIA GeForce RTX 4090 with 24,564 MiB.
  The focused CPU GPT-OSS run passed `37` tests with one ignored exact-artifact
  test. The focused CUDA run passed `45` tests with the same one ignored test,
  including the real serialized fused/split loader, delayed cancellation and
  non-finite-output recovery, and public-config validation.
- The synthetic CUDA model admits `19,416` logical static weight bytes, of
  which `3,264` are packed MXFP4 block/scale bytes. Its exact cache contract
  remains `128` logical bytes per token and `512` bytes at the four-token
  bound. The tests cover the exact bound and one-over rejection.
- The CUDA component trace compares attention values, router values, sorted
  expert indices, normalized expert weights, post-attention hidden states,
  expert contributions, and post-MoE hidden states. The executor uses F32
  dense transformer tensors plus U8 packed expert tensors; it does not
  silently dequantize MXFP4 weights.
- The CUDA packed-kernel narrowed-view regression compares padded non-zero-offset
  views with a fully materialized copy. Non-contiguous or narrowed inputs are
  copied explicitly before launch, and `DevicePtr`/`DevicePtrMut` guards live
  through the kernel call.
- The GGML wire regression covers all 32 coordinates, mixed signs and scales,
  and the discriminating high-nibble lane. The real serialized fused/split
  loader regression compares loaded packed values and expert contributions with
  an independent wire decoder and compares fused/split CPU logits; the CUDA
  feature lane repeats the output comparison.
- The retained-file regressions cover same-size path replacement, Windows
  write/delete sharing denial, admission and chunk-read cancellation,
  construction failure, duplicate rejection while a model lives, and release
  after teardown. The retained handle is shared by the load session and the
  owner-bound registry handle; no production bytes are copied into the repo.
- The synthetic GGUF assembly regression covers fused experts plus converter-
  style split gate/up experts and split Q/K/V tensors, then runs one CPU
  forward. The owner-selected artifact test used only
  `C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf`, verified the pinned SHA, assembled
  `GptOssWeights`, constructed `GptOssModel`, and returned the registry count to
  zero after release. No model bytes entered the repository.
- The opt-in short-parity runner binds the supplied model, tokenizer,
  tokenizer-config, chat-template, and external llama.cpp `_logits_` artifact
  to exact SHA-256 identities. Its fixed plain-text policy is
  `add_bos=false`, `add_eos=false`; the 20 tokenizer IDs match the fixture.
  The external reference is llama-perplexity `0.4.1-dev`, build `11026`,
  commit `b49650adb`, with 1,206,604 bytes and SHA-256
  `dafa573496953142bb637bd009f1a0e38a9e36fbd10255ee98fc9bfbc6dbfc3f`.
  The successful native run reached GGUF load, CUDA model construction,
  cancellation rollback, three output captures, deterministic reset/replay,
  and registry teardown. Its receipt is
  `artifacts/gpt-oss/short-parity/receipt.json`; the candidate is explicitly
  marked `uncommitted_candidate`.
- The accepted bounded release performance packet is
  `artifacts/gpt-oss/performance/bounded-release-packet-safe.json`, with run
  directory `artifacts/gpt-oss/performance/runs/20260922T002225005Z-60f5d632eaf6`.
  It used source `4a4699981ed55bb11e857c58923c67133559145d`, release executable
  SHA-256 `f0f808c05f570e8aa8a4cbc25055f67c2de7d78feb73d2edd5cdab5ed0c1f30c`,
  the owner GGUF SHA-256
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`, the
  tokenizer SHA-256
  `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3`, and
  an RTX 4090/driver 616.92/compute 8.9. It passed the 8/512/2048 prompt
  cases with 32 generated tokens at 14.342/13.872/13.411 prefill tokens per
  second, with accounted device bytes of 17,379,651,840 /
  17,495,515,392 / 17,848,623,360. The monitor peak was 19,723,714,560
  bytes, below the exact ceiling; runner and wrapper both report
  `stop_reason: completed`, `completed_cases: 3/3`, and `exit_code: 0`.
  After process exit, no GPT-OSS process remained and current GPU use was
  835 MiB. The external model/tokenizer remain outside the repository.
- The clean-source diagnostic packet
  `artifacts/gpt-oss/performance/bounded-release-packet-committed.json` is
  retained as superseded evidence: its 8160-prompt phase sampled a peak of
  `25,206,718,464` bytes, above the exact ceiling, so its logical accounting
  and successful runner status are not an acceptance claim.
- Pinned behavioral sources remain OpenAI `gpt-oss` commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1` and llama.cpp commit
  `f072b103714dfa1eee531f80b24512faf38e3dd2`; see `SOURCES.md`. The CUDA
  kernel itself is a fresh Candle-native implementation.
- The maintained Q8/default path is unchanged. No Edge-owned Harmony/profile
  integration, training, hidden download, exact product model load, or
  tokenizer work was added.

## Public boundary

`GptOssGgufArtifact::load_weights` and
`load_gpt_oss_weights_with_cancellation` assemble admitted dense F32/BF16
text tensors and packed MXFP4 expert tensors through one bounded retained file
session. The public combined loader retains a registry lease until the caller
drops the returned handle.

`GptOssCudaModel` is an opt-in, `cuda`-feature-gated executor over admitted
synthetic/native `GptOssWeights`. `GptOssCudaConfig` makes the device index,
F32 activation dtype, static weight budget, and existing token/KV limits
explicit. `GptOssCudaError` distinguishes unsupported dtype, unavailable
device, resource limit, overflow, cancellation, invalid input, backend, and
kernel failures. Forward work stages a cloned cache and commits it only after
all checkpoints pass; Candle tensor/device handles provide cleanup ownership.

The CUDA path is not wired into the GGUF product loader, tokenizer, generation
CLI, or maintained Q8/LFM2-VL defaults. Q8_0 dense text assembly remains
explicitly deferred. The exact artifact result proves bounded weight assembly
and model construction, not tokenizer or production numerical support. The
short-parity example is an explicit opt-in diagnostic seam and does not change
the maintained product path.

## Known limitations and blockers

- The selected product GGUF is an owner-admitted external input, not a
  checked-out file. Its exact admission, bounded weight assembly, model
  construction, and short CUDA numerical gate now pass. The prefill raw full
  row still contains a clipped-tail diagnostic outlier of `0.4869547`, but
  the predeclared robust criteria pass: prefill top-k union error `0.1310458`,
  mean error `0.0020542`, and total variation `0.0048582`; decode stages also
  pass. Forward parity is limited to this pinned short CUDA receipt; Q8/default
  and maintained production support remain unproven.
- Only F32 activations are accepted by this executor. Quantized text and split
  dense MMProj execution are intentionally deferred to the next ordered task.
- The CUDA receipt is a synthetic fixture result on one Windows RTX 4090
  lane; it is not a cross-device matrix or a production throughput claim.
- The earlier owner-artifact 32k-matrix run
  `artifacts/gpt-oss/performance/runs/20260921T071321303Z-3cd84d87e5b8`
  remains historical partial evidence: it reached 8/512/2048/8192 and timed
  out during 16384 with `4/6`, no report, and no recovery receipt. This slice
  intentionally accepts only the observed-safe 2080-token envelope; no
  16k/32k optimization, batching, mixed precision, or KV-storage redesign is
  implied.
- The existing Windows linker warning `LNK4098` remains an environment/build
  warning in the CUDA test binary; it did not fail the executed tests.
- The rolling repository-wide overlay gate now passes with the live union
  baseline, including committed, staged, unstaged, and untracked candidate
  paths. The isolated `scripts/tests/test-verify-fork-overlays.sh` regression
  suite covers allowed dirty paths, an unowned path, rename/deletion ownership,
  invalid and missing baselines, and the rolling/upstream distinction.
- `--upstream-baseline 6f74e7c` remains an exact-delta frozen-receipt check. It
  correctly rejects this current rolling candidate because that older delta
  contains unrelated upstream paths outside the registered overlays; this is
  not a blocker for the live rolling gate.

## Last green verification

Native Windows/MSVC, locked dependencies for this correction and assembly
slice:

- `cargo fmt --all -- --check`: passed.
- `cargo check --locked -p candle-core`: passed.
- `cargo check --locked -p candle-nn`: passed.
- `cargo check --locked -p candle-transformers`: passed.
- `cargo check --locked -p candle-vlm`: passed.
- `cargo check --locked --features cuda -p candle-transformers`: passed.
- `cargo check --locked -p candle-examples --example lfm2`: passed.
- `cargo check --locked -p candle-examples --example quantized-lfm2`: passed.
- `cargo check --locked -p candle-examples --example lfm2-vl`: passed.
- `cargo clippy --locked -p candle-transformers --lib -- -D warnings` and
  `cargo clippy --locked --features cuda -p candle-transformers --lib -- -D
  warnings`: passed.
- `cargo test --locked -p candle-transformers --lib models::gpt_oss`:
  passed `37/37` CPU-focused tests, with one ignored exact-artifact test.
- `cargo test --locked --features cuda -p candle-transformers --lib models::gpt_oss`:
  passed `45/45`, with one ignored exact-artifact test.
- `cargo check --locked --features cuda -p candle-examples --example
  gpt-oss-short-parity`: passed.
- `cargo test --locked --features cuda -p candle-examples --example
  gpt-oss-short-parity`: passed `3/3` model-free comparison tests.
- `cargo check --locked --features cuda -p candle-examples --example
  gpt-oss-performance`: passed.
- `cargo test --locked --features cuda -p candle-examples --example
  gpt-oss-performance`: passed `5/5` model-free planning, cancellation, and
  logits-shape tests.
- `pwsh -NoProfile -File scripts/gpt-oss/test-performance.ps1`: passed `16/16`
  model-free harness tests, including single-case normalization, release
  executable selection, exact Edge ceiling, bounded-context refusal, resource-
  limit stop classification, and observed-limit precedence.
- `cargo test --locked -p candle-transformers --lib gpt_oss::runtime`: passed
  `4/4`, including exact total-device-boundary/one-over arithmetic.
- `cargo build --locked --release --features cuda -p candle-examples --example
  gpt-oss-performance`: passed with the optimized release profile.
- The accepted bounded packet command with `-Lengths 8,512,2048`,
  `-TargetContextTokens 2080`, autoregressive mode, and all three explicit
  budgets set to `20000000000` exited `0` and wrote the report plus
  runner/wrapper terminal and monitor evidence. The monitor's peak sampled GPU
  use was `19,723,714,560` bytes and the post-run process census found no
  GPT-OSS process.
- The clean-source `-Lengths 8,512,2048,8160` / `-TargetContextTokens 8192`
  diagnostic is retained but rejected because its 8160 phase sampled
  `25,206,718,464` bytes. The wrapper with `-TargetContextTokens 16384`
  exited `1` before model execution with `bounded GPT-OSS qualification
  refuses target contexts above 8192 tokens.`
- `cargo clippy --locked --features cuda -p candle-examples --example
  gpt-oss-performance -- -D warnings`: passed.
- PowerShell parse validation for `scripts/gpt-oss/run-performance.ps1`:
  passed.
- The historical 32768-token performance command remains a retained timeout
  record only; the current wrapper refuses targets above 8192 before model
  execution, and the accepted native packet is narrower because of observed
  physical-device accounting.
- `pwsh -NoProfile -File scripts/gpt-oss/run-short-parity.ps1
  -ReferenceLogits C:\Users\jc816\AppData\Local\Temp\edgesymbio-gptoss-reference-20260920\reference-c8.logits`:
  passed in `685.41s` on `Cuda(CudaDevice(DeviceId(1)))`. The receipt records
  exact top-1 agreement, prefill top-k error `0.1310458`, mean error
  `0.0020542`, total variation `0.0048582`, decode top-k errors `0.0302615`
  and `0.0602632`, reset/replay error `0.0`, cancellation cache length `0`,
  and teardown registry count `0`. The candidate tree is dirty and the
  receipt makes no publication claim.
- `cargo test --locked -p candle-transformers --lib models::gpt_oss::gguf::tests::assembles_owner_selected_product_artifact_when_requested -- --ignored --exact --nocapture`, with
  `CANDLE_GPT_OSS_GGUF=C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf` and
  `CANDLE_GPT_OSS_MAX_RESIDENT_BYTES=34359738368`: passed in `613.20s`.
  The test verified the pinned SHA, assembled the exact artifact, constructed
  a model, and released the registry lease.
- `git diff --check`: passed.
- `pwsh -NoProfile -File scripts/lfm2-vl/verify-summary-bank.ps1`: passed.
- `bash scripts/gpt-oss/verify-mod-manifest.sh a0c795a7f6a27d56175aa5c45ee764067ad7d5e3`:
  passed with 35 registered paths.
- `bash scripts/lfm2-vl/verify-mod-manifest.sh 7c2e89295dad4aeebc6ef7a92c255360b6957c2c`:
  passed with 163 total paths, 17 fork-origin modifications, and 146 additions.
- `bash scripts/verify-fork-overlays.sh --rolling-baseline d830e03078a29d39a1aacd741620475eb33b7609`:
  passed with 200 registered paths and 21 shared paths.
- `bash scripts/tests/test-verify-fork-overlays.sh`: passed all 7 isolated
  regression cases.
- The prior guarded publication helper verified `f470d2de` on `origin/main`;
  the current bounded source slice still requires its final clean-tree gate
  and direct publication.

## Exact next task

No further 32k performance work is in this bounded slice. Preserve the
observed-safe optimized-release 2080 packet as optional Candle evidence, and
keep Task 4 quantized text plus split dense MMProj separately scoped. Do not
infer Q8/default or broad maintained production support from this packet.

---

AI-edited: 2026-09-21; agent=Codex; task=gpt-oss-performance-qualification; change=recorded the observed-safe release packet, superseded the physical-ceiling diagnostic, and closed model-free verification
