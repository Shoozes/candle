# GPT-OSS local loading and session contract

The current implementation is a local F32 loading/session candidate. Synthetic
CPU/CUDA proof does not qualify production weights, performance, a 32k provider
or BF16 execution. Existing production receipts retain their original source
identities and 2,080-total-token boundary.

## Public API

`candle_transformers::models::gpt_oss` exports `GptOssSession`,
`GptOssLoadOptions`, `GptOssSource`, `GptOssDevice` and `GptOssLogits`.
Callers supply the exact local file identities, device, sequence/cache/device
limits, weight budget, shared load registry and cancellation token. There is
no cache discovery, download, tokenizer rendering, sampling or provider policy
in this session.

- `GptOssSource::Gguf` takes a `GptOssFileIdentity`: local path, byte size and
  expected SHA-256. Historical GGUF helpers still require the original selected
  artifact; `open_with_sha256` is the explicit alternative.
- `PackedSafetensors` takes `GptOssSafetensorsSource`: explicit Original or
  Transformers layout, config/tokenizer identities, optional index identity
  and exact shard identities. Multiple shards require their index. The index
  controls the shard set; filename numbering is not used to infer it.
- `GptOssSession::load` returns a CPU or feature-gated CUDA session. `config`,
  `device`, `activation_dtype`, `consumed_files` and `resource_usage` expose
  execution and provenance. Unsupported CUDA selections fail before artifact
  I/O in CPU builds.
  `static_weight_bytes` reports assembled payload residency. Both backends
  admit combined static/cache/conservative-workspace bytes before forwarding;
  a budget that cannot admit one token fails before weight assembly.
- `prefill` and `decode` return device-native `GptOssLogits`. `into_cpu` is a
  separate fallible readback; readback failure is not a rolled-back forward.
  Inference cancellation/errors preserve the previously committed cache prefix.
  `reset_cache` releases owned cache buffers.

The CUDA executor still synchronizes and reads its full final logit row on the
host for finite-output validation before cache commit. Failure in that internal
validation preserves the old cache. A later caller-requested `into_cpu` failure
leaves the completed forward committed. Returning a CUDA tensor does not yet
eliminate the executor's validation transfer; profile that transfer before
changing the safety boundary or claiming a latency improvement.

Both loaders preserve MXFP4 expert blocks/scales. Original safetensors store
adjacent low/high nibbles and alternating gate/up rows; only GGUF wire bytes
need their existing layout normalization. Non-expert F32/F16/BF16 weights, and
GGUF Q8_0 weights, currently normalize to F32 dense execution. There is no
native-Q8 non-expert storage or FastBf16 readiness claim.

## Admission and ownership

SHA-256 and size admission use the retained file handle. Windows read-sharing
denies writes/deletion while that handle is retained. An artifact owns its file
handles; assembled model weights hold the registry lease through CPU/CUDA model
lifetime. Failed/cancelled loading releases ownership for a subsequent load.

Safetensors bounds include 128 explicit shards, 16 MiB per header, 32 MiB total
headers, 4,096 total tensors and rank 1..4 positive shapes. Duplicate names,
unsupported dtypes, gaps/overlaps, truncated/trailing payloads, index mismatch,
unconsumed tensors and arithmetic overflow produce errors. Dense conversion
uses at most 1 MiB raw chunks rather than a second whole-tensor copy.

The weight budget bounds final assembled residency; metadata, conversion,
QKV concatenation and backend upload also require temporary host memory.
Physical host/GPU ceilings remain enforced by the owner launch and measured
qualification, not inferred from logical cache accounting.

Configuration accepts original and Transformers aliases, reconciles equal
duplicate aliases, and rejects conflicting expert/YaRN fields. Explicit layer
types and RMS epsilon drive both executors. Missing layer types preserve the
legacy alternating sliding/full schedule. Checkpoint capacity is distinct from
the smaller supplied execution limits and any qualified provider envelope.
Existing JSON configuration remains compatible. Rust callers constructing
`GptOssConfig` with a struct literal must now supply `layer_types`,
`rms_norm_eps` and `max_position_embeddings`; repository callers are updated.

## Remaining maintained-support gates

The accepted target is GPT-OSS-20B on native Windows Ryzen 7950X and RTX 4090,
with 32 GiB host and 20,000,000,000 GPU bytes, and 32,768 total tokens split into
30,720 prompt/history plus 2,048 generated tokens including private reasoning.
120B configuration compatibility is distinct from production qualification.

Next: clean source/binary-bound baseline profiling, then measured CPU/CUDA
prefill/decode optimizations, separately qualified optional BF16, and staged
8k/16k/32k qualification. No old model allowance may be reused. Full native
production qualification also requires retained official packed artifacts;
metadata inspection does not establish their local presence or permission to
download them.

EdgeSymbio owns Harmony, sampling, the existing in-process model slot and the
linked artifact/device/mode/route qualification. The maintained alias is
`edgesymbio.local.gpt-oss-20b.v1`; its runtime and Responses identities are
`edgesymbio.candle-gpt-oss.runtime.v1` and
`edgesymbio.codex-responses.candle-gpt-oss.v1`. Those are prepared identities,
not enabled capabilities. Preserve the experimental identity and Q8 default.

Responses retains its 90-second total deadline. Longer work uses WorkerRuntime
within its existing 600-second bound. A WorkerRuntime 32k receipt cannot qualify
Responses. Strict v1 receipts stay unchanged: their linked capacity-report
digest must bind the actual source/build/artifact/device/mode/format/settings,
route/deadline, observed envelope, physical memory, work counts and release.
Packed weights use a separately identified canonical consumed-file manifest;
its digest must never be presented as a single model-file hash.

Symbio preserves Codex's engine/tool loop and requires observed Edge
qualification. Pins/locks/provenance advance only after separately authorized
Candle publication, followed by Edge qualification and Symbio consumption.
GenEye remains pinned to `3fa7075e`.

---
AI-edited: 2026-10-12 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-local-loading | change=documented implemented loading and pending measured qualification
