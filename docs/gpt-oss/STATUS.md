# GPT-OSS Experimental Candle Status

## Assignment state

Documentation closeout (2026-10-07) reviewed native `main` at
`b274902312a39cbe1b0028141276f43e64b56096`. The corrected evidence descriptions
and deferred acceleration note preserve the executed source identities below;
this closeout adds no production-model or CUDA qualification.

- Current phase: Task 3 bounded native performance qualification is complete
  for the observed-safe 2080-token total-context envelope. Short parity remains
  accepted; the optimized-release packet proves the 8/512/2048 prompt cases
  with a 32-token autoregressive reserve, exact 20,000,000,000-byte admission,
  fail-closed sampling, and post-run cleanup. The harness still refuses target
  contexts above 8192 tokens, but the clean 8160-prompt diagnostic exceeded the
  observed GPU ceiling and is not accepted. This optional Candle track does not
  change EdgeSymbio or symbio-code.
- Historical performance qualification baseline: `main` at
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
  835 MiB. The ignored report bytes observed at delivery are bound by raw
  SHA-256 `616b188eadb04b27f7834a952d45a7cfd8ec9ff8a0b68413ce8a5aa468a71431`;
  the report remains ignored and the external model/tokenizer remain outside
  the repository.
- The clean-source diagnostic packet
  `artifacts/gpt-oss/performance/bounded-release-packet-committed.json` is
  retained as superseded evidence: its 8160-prompt phase sampled a peak of
  `25,206,718,464` bytes, above the exact ceiling, so its logical accounting
  and successful runner status are not an acceptance claim.
- Pinned behavioral sources remain OpenAI `gpt-oss` commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1` and llama.cpp commit
  `f072b103714dfa1eee531f80b24512faf38e3dd2`; see `SOURCES.md`. The CUDA
  kernel itself is a fresh Candle-native implementation.
- The maintained Q8/default path is unchanged. External model loading and
  tokenizer use belong to the explicit diagnostic examples above; no
  Edge-owned Harmony/profile integration, training, or hidden download follows
  from those receipts.

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

The maintained GGUF/Q8/LFM2-VL product routes do not select this CUDA executor.
Q8_0 dense text assembly remains explicitly deferred. Exact artifact assembly
and construction alone do not prove inference. The separate opt-in short-parity
and performance examples load the admitted model and tokenizer for their
source-bound diagnostic gates; they do not change the maintained product path.

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
- Synthetic CUDA component proof, the pinned external short-parity receipt,
  and the 2,080-token performance packet are distinct results on one Windows
  RTX 4090 lane. None establishes a cross-device matrix or maintained product
  qualification.
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
slice. The runtime results below are historical; the documentation cleanup
did not rerun builds or model/CUDA inference:

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
- At that performance closeout, the guarded helper had verified `f470d2de`
  on `origin/main` and the bounded slice still awaited publication. This is
  historical checkpoint state; the [current Candle handoff](../lfm2-vl/STATUS.md)
  records later publication without relabeling these receipts.

## Deferred optional acceleration research

Reviewed 2026-10-07 against Candle `main` at
`b274902312a39cbe1b0028141276f43e64b56096`. This note records future research;
it establishes no speedup or new qualification. GenAce MVP/LTS and GenUni's
current foundation retain priority. The short-parity and observed-safe
2,080-token performance receipts above retain their executed source identities.
The [module header](../../candle-transformers/src/models/gpt_oss/mod.rs) and
manifest describe these separate evidence boundaries; documentation cleanup
does not renew the runtime receipts.

### Profile existing safe cases first

At this source, the [packed kernel](../../candle-kernels/src/gpt_oss_mxfp4.cu)
assigns each route/output to one thread and reduces K serially. The
[CUDA executor](../../candle-transformers/src/models/gpt_oss/cuda.rs) accepts F32
activations, processes prefill tokens individually, concatenates KV tensors,
and allocates routing/activation intermediates. These are source observations,
not measured bottlenecks. Candidate techniques are:

- Prefill: bounded token chunks and grouping by expert to expose larger GEMMs,
  preserving causal/sliding/sink attention and router ordering.
- Low-batch decode: cooperative Ada packed-MXFP4 reduction; large-M prefill
  throughput does not predict one-token latency.
- Both: fusion and scratch/KV buffer reuse while preserving staged cache commit,
  cancellation and packed expert ownership. Lower precision or activation
  quantization needs separate numerical proof.

Dense LFM2 has no GPT-OSS MoE routing requirement. Dense projection,
attention/convolution fusion and buffer reuse are separate hypotheses;
GGUF text and LFM2-VL vision/projector layouts need their own proof. Preserve
LFM convolution history across chunks and cache-reset behavior.

### Primary-source shortlist

The linked revisions are research snapshots checked on 2026-10-07, not new
dependency pins or replacement golden references. No library/kernel was built
or run for this note. Preserve upstream license notices if code is later adapted.

| Reference | Hardware, precision/layout and shape constraints | Build/runtime/license and evidence limit |
| --- | --- | --- |
| [DeepGEMM `057ca596`](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md), [API exports](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/__init__.py) | Published CUDA kernels require SM90/SM100, excluding the recorded [RTX 4090's Ada SM89](https://developer.nvidia.com/cuda/gpus). FP8/FP4/BF16 APIs have distinct operand, scale, layout and alignment requirements. M-grouped contiguous prefill fixes N/K and aligns expert segments; masked decode is separate. SM90 FP32 versus SM100 packed UE8M0 scales are not Candle's U8 blocks/scales. cuBLASLt exports and legacy A100 Triton imports do not qualify Ada MXFP4. | [MIT](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/LICENSE). Python/PyTorch extension and runtime JIT; published minima: Python 3.8, PyTorch 2.3, C++20, CUDA 12.9, CUTLASS 4.0. Mega MoE additionally uses PyTorch 2.9 symmetric memory, multiple processes and NVLink dispatch/combine. Reference ideas only for this single-4090 lane; native Windows/Candle integration is unproven. |
| [CUTLASS 4.8.0 `0b55a2f6`](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/README.md), [functionality](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/media/docs/cpp/functionality.md) | SM80+ dense F16/BF16 and grouped-GEMM building blocks are candidates for Ada. Match each kernel's M/N/K, batch, layout, strides, alignment and tile limits. Dense or INT4 support does not establish packed MXFP4 support; changing Candle F32 activations requires parity. | [BSD-3-Clause](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/LICENSE.txt). C++17/CUDA templates need Candle launch integration. The README retains a C++ 3.x Windows-build warning despite newer CuTe DSL Windows support; prove the selected native MSVC/kernel slice. No Ada MXFP4 qualification or decode gain follows from library support. |
| [llama.cpp `88dcc460` MMVQ](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/mmvq.cu), [packed dot](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/vecdotq.cuh), [MMQ](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/mmq.cu) | Ada-aware MMVQ dispatch and MXFP4/Q8_1 dot techniques are relevant decode references; MMQ has separate batching criteria. Native FP4 instructions are Blackwell-gated. Dispatch depends on type, batch, expert count and architecture. GGML nibble/block layouts and quantized Q8_1 activations differ from Candle's adjacent-nibble/F32 executor. | [MIT](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/LICENSE); upstream documents [Windows CMake/MSVC and CUDA builds](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/docs/build.md). Study bounded techniques before adopting a runtime. Conversion and activation-quantization parity, memory and overhead need proof; source support supplies no Candle speedup. |
| [Triton `fa2e589b`](https://github.com/triton-lang/triton/blob/fa2e589b55706d0de73d25641840e134f20a2c65/README.md) | Upstream lists Linux and NVIDIA CC8.0+, making Ada a hardware candidate. Custom reduction/fusion kernels still need exact dtype, block/scale packing, strides, M/N/K and batch support. | [MIT](https://github.com/triton-lang/triton/blob/fa2e589b55706d0de73d25641840e134f20a2c65/LICENSE). Python/LLVM/JIT toolchain; at most a future Linux/WSL research path with compilation/cache costs included. No native Windows or Candle MXFP4 qualification. |

### Admission for a future optional path

1. Profile the existing admitted 8/512/2048-prompt plus 32-generated cases within
   the observed-safe 2,080-token envelope. Select one measured bottleneck and
   one optimization; report prefill throughput and low-batch decode latency
   separately, with GPT-OSS MoE and dense LFM2 evaluated independently.
2. Require exact device capability, driver/toolchain, compiled kernel, precision,
   blocks/scales/layout/strides, shape/batch and context support, plus explicit
   opt-in initially. Detection is not proof. Unsupported or unqualified cases
   retain the existing default fallback before mutable execution.
3. Bind independent component/router/expert/attention/logit, cache/reset and
   cancellation parity to the exact kernel/library/Candle/model/tokenizer
   versions. Keep existing tolerances. Require actual latency/throughput and
   physical peak-memory evidence on the same inputs, including conversion,
   packing, quantization, JIT, launches and allocations.
4. Existing physical VRAM/proof budgets, quiet-host checks, cancellation and
   process/lease ownership remain authoritative; this note adds no disk cap.
   On runtime failure preserve the failure/uncertain state and cache ownership;
   never silently retry another backend after cache mutation. Research does not
   reorder Task 4 or open a product qualification gate.

## Exact next task

No further 32k performance work is in this bounded slice. Preserve the
observed-safe optimized-release 2080 packet as optional Candle evidence, and
keep Task 4 quantized text plus split dense MMProj separately scoped. Do not
infer Q8/default or broad maintained production support from this packet.

---

AI-edited: 2026-10-07 | agent=Codex/root | model=unknown | effort=unknown | task=evidence-cleanup | change=reconciled historical evidence language while preserving deferred acceleration research
