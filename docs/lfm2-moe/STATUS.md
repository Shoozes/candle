# LFM2-MoE current state

The November 10 capability was published as Candle
`95df99a0bda2f6ac64dbeb8475850f826bd92ca9` from base `5f094821`.
The first Q4 parity attempt stopped during CPU comparison. The comparator
repair exposes a remaining CPU numerical disagreement. Edge qualification and
adoption follow supplier parity. D1 stays closed.

Native `models::lfm2_moe::{Config, Model, Cache}` loads the official per-expert
safetensors layout and exposes token/embedding forward, hidden states, logits
projection and cache reset. The pinned native index confirms 7,610 tensor names.
Shared attention/convolution math is reused through private mixers; dense LFM2
public behavior is preserved. Routing follows pinned sigmoid/bias/top-K semantics.

`models::quantized_lfm2_moe::{ModelWeights, inspect_gguf_metadata}` validates all
required roles, dimensions, storage types, offsets/alignment, non-overlap and
payload EOF. Q4_K/Q6_K/Q8_0 experts stay quantized; F32/F16/BF16 storage has
fixture coverage. CPU/CUDA share grouped execution. Input IDs and context are
checked before embedding allocation, and failed forwards clear both caches.
Memory estimation includes widened storage, auxiliary tensors, full input,
prefill scratch and KV/convolution cache. No silent device fallback is used.

Retained implementation proof includes eight CPU MoE tests and ten with CUDA enabled,
covering the pinned F32 oracle, routing, native untied output/missing experts,
chunking/decode/reset, malformed GGUF, lower-precision storage, retained quantized
matrices and recovery. CUDA production weights have not been loaded. Native
F32 tolerance is `atol=1e-5`, `rtol=1e-4`; quantized tolerance is separately
recorded in [DECISIONS](DECISIONS.md).

The preceding implementation gate passed 617 Rust tests with 14 ignored,
197 release assertions on each PowerShell host, library/example checks,
Clippy and existing model-free controls. These are retained source-bound results,
not new full-matrix or CUDA runs from the review/cleanup.
Overlay registration has seven independent owners and its 12 regression cases
pass. Summary Bank and the independent existing manifests remain verifiable.

Edge has an explicit `lfm2-moe-local` backend/headless CLI/API feature, new
`edgesymbio.local.lfm25-24b-a2b.q4km.v1` candidate, exact artifact admission,
48 KiB / 12,288 total / 2,048 generated limits, seeded reasoning parsing and
existing validated tool/JSON permission paths. Six focused model-free tests
pass with the local CUDA overlay. Exact rendered bytes and token IDs match four
pinned official tokenizer cases without loading model payloads. Edge's broader
backend regression and CLI/API checks are retained in the run record.

The candidate remains unapproved/held pending production qualification. Local
Cargo overlays restore original Edge locks byte-for-byte; its ordinary pin is
still `38d90519`. GenEye's `81e1aa28`, other consumer pins, frozen releases and
all original owner edits are preserved. The current publication covers Candle
only; it does not grant Edge pin/lock adoption or production execution.

The existing official Q4_K_M file remains at `C:/llamacpp/models/`, unchanged and
unloaded. Header inspection admits all 428 tensors and estimates 16,035,996,672
bytes at context 12,288/chunk 128. That is an estimate, not observed peak memory
or runtime acceptance. The pinned llama.cpp CPU probe is built separately from
`582eab71`; the installed `9df0b73c` runtime is untouched. Its no-model version
probe passed and the bounded build Job released all children.

Run record: `D:/Compact_Evidence/candle/runs/lfm2-moe-20261110-5f094821/summary.json`.
[TODO](TODO.md) owns remaining Done When gates. [QUALIFICATION](QUALIFICATION.md)
freezes two prompts/eight teacher-forced positions before testing. The prior inert
packet `1e4449ab` at `inert-packet-final/packet.json` remains preserved as historical
preparation. It no longer matches the reviewed source and is not admissible.

The review/recheck compared all 48 Candle and 16 Edge scoped source hashes with
the retained final inventory at the unchanged Candle/Edge heads. Previous token,
context, routing, memory and consumer-identity fixes remain closed on recheck.
Cleanup repaired two new defects: the comparator previously admitted matching
shortened/wrong-reference records, and the example discarded normalized EOS
metadata by requiring a U32 encoding again. Comparisons now bind the frozen case,
reference revision, full continuation and exact count; the example consumes the
inspector's checked EOS. Eleven model-free protocol tests and five quantized CPU
tests pass, including EOS acceptance/refusal. Required checks/Clippy and their
exact commands belong to the review receipt under the run's `review-20261110/`.
The main run summary and all older proof remain unchanged.

The model-free artifact refresh is complete. A fresh CUDA-enabled release binary
was built locked/offline at
`D:/CodexStorage/build/candle-lfm2-moe-refresh-20261110-5f094821/release/examples/lfm2-moe.exe`,
SHA-256 `7ee3969b125feffd96d1df2b2b78f767dafca78cf4e978c73fb6f34a3e8a82e3`.
All 788 build-input hashes stayed stable. Header inspection and four exact
tokenizer cases passed without loading weight payloads, and the retained CPU
reference reported its pinned revision. The original model's full size/hash was
freshly reverified in place. The build emitted an LNK4098 library-conflict warning;
the successful build and no-model checks do not prove production execution.

The completed refresh's inert packet is
`D:/Compact_Evidence/candle/runs/lfm2-moe-artifact-refresh-20261110-5f094821/packet/packet.json`.
Its owning run's `summary.json`, `build-binding.json` and
`binding-verification.json` bind the binary, stable build inputs, finalized
candidate inventory and all four `--inputs`/`--case` comparisons. Original limits,
old packet/binaries, frozen inputs and consumer pins are preserved. All bounded
build/no-model children exited and released their Jobs. Production dispatches
remain zero. [HISTORY](HISTORY.md) owns the completed refresh details. This
`1717f529` packet remains immutable preparation for its pre-publication source.

The source closeout's commit, exact clean-commit native gate and confirmed remote
result belong to
`D:/Compact_Evidence/candle/runs/lfm2-moe-publish-20261110-5f094821/summary.json`.
Only the guarded helper's confirmed receipt establishes publication. Its
`packet/packet.json` rebinds preparation to the final committed source inventory
without rebuilding unchanged Rust inputs or changing the binary/model identity.
Its `binding-verification.json` records the source bridge and original limits.
No historical packet or binary is replaced.

Packet `00ab6c5e` consumed its one attempt on November 11. Both CPU reference
children and Candle's CPU arithmetic child exited successfully and released
their Jobs. The first comparison failed while decoding a clipped UTF-8 tokenizer
diagnostic. CPU Unicode and both RTX 4090 children were skipped. The host returned
to its pre-run resource range. The [attempt receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-qualification-20261111-00ab6c5e/failure-review.json)
binds the three original logs and cleanup evidence.

The comparator reads JSON lines from byte logs and reports the first
top-1 mismatch. Thirteen protocol tests passed. Offline replay of the original
arithmetic pair reached the numerical gate and rejected zero-based position 2
with reference token `1113` and Candle token `6986`.
The [cleanup receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-review-cleanup-20261111-95df99a0/summary.json)
records the source and replay identities.

The retained pair already differs at the output of the 37-token prefill. Its
seven sampled logits have a maximum absolute delta of `0.70701456` at position
0, before either engine consumes a forced token. Both still select `207` there.
The current records omit intermediate tensors and the logits for both competing
tokens at position 2. They cannot identify the first unequal layer or the rank
margin at the first top-1 mismatch.

The pinned reference log identifies layer 0 ShortConv input projection as
repacked `Q4_K_8x8`. Candle's generic and packed x86 Q4_K paths use
`BlockQ8K::from_float`,
which rounds activation half-ties away from zero. The pinned reference's scalar
and AVX2 conversions round half-ties to even. A controlled 256-value block
demonstrates the difference with identical operands. The same arithmetic
comparison found equal quantization for four existing synthetic fixture rows.
Both prebuilt synthetic prefill tests passed. This establishes a source-level
difference and a fixture coverage gap. The production cause is unresolved.

The next task in TODO is a paired CPU prefill trace at the embedding, layer 0
operator normalization and ShortConv input projection. If the projection differs
on common normalized inputs, compare the Q8_K blocks with the same packed
group or tail dispatch before attributing the difference to the dot product.
The [isolation receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-cpu-isolation-20261111-95df99a0/summary.json)
binds the witness, fixture results and source anchors. Further builds and model
loads remain held. A future production attempt requires a new reviewed packet,
exact grant and fresh quiet/resource admission.

The comparator and prefill isolation closeout is recorded in its
[guarded publication receipt](D:/Compact_Evidence/candle/runs/lfm2-moe-comparator-publish-20261111-95df99a0/summary.json).
The receipt binds the verified source change to its commit and remote main.
Production runs, package assembly, installation, activation, release creation and
consumer repinning remain held. Edge runtime/CLI/API acceptance and reviewed adoption stay
open after the supplier gate. Full native/Q8 production, Metal and 131k context
qualification remain outside this delivery.

Active source: the two new model modules, private shared mixers, local example,
synthetic fixtures, reference/qualification tools and their owning records.
Known unsupported native configuration: `conv_bias=true`, non-default RoPE.
The two review findings are closed by the scoped repair. Retained failed
implementation attempts remain in the original run record. Native lower-
precision activation parity is not claimed by the CPU-F32 numerical gate.

---
AI-edited: 2026-11-11 | agent=Codex/root | task=lfm2-moe-comparator-publish | change=link scoped comparator closeout and retain the paired trace gate
