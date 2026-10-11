# Local LFM d1 example

This thin runner uses `candle_vlm::lfm2_d1` with explicit local inputs. It retains
and hashes model, projector, tokenizer, processor and template files against the
supplied manifest, then rehashes them after execution. It does not download models.

Each input JSONL row contains `request_id`, `request` and optional image paths:

```json
{"request_id":"one","request":{"state":"The assertion failed.","questions":{"ready":{"type":"noul","instructions":"Is the evidence sufficient?"},"inspect":{"type":"choice","instructions":"Pick an inspection.","criteria":{"stdout":"Read the stack trace.","report":"Read the final report."}},"clarity":{"type":"score","instructions":"Rate clarity.","criteria":["unclear","partial","clear"]}}},"images":[]}
```

```powershell
cargo run --release --locked --offline -p candle-examples --example lfm2-d1 -- `
  --model C:\models\d1-3B-Q8_0.gguf --mmproj C:\models\mmproj-d1-3B-Q8_0.gguf `
  --tokenizer C:\models\tokenizer.json --processor C:\models\processor_config.json `
  --template C:\models\chat_template.jinja `
  --artifact-manifest tools\lfm2_vl\d1\reference-lock.json `
  --requests C:\inputs\requests.jsonl --out C:\results\fresh-d1-run --device cpu
```

Use `--features cuda` and `--device cuda` only after the documented CPU parity
gate. The output directory must be fresh. `--trace` retains exact prompts, token
IDs, final logits and the first image request's processor/features. Failures keep
partial answers/work counts and remaining case IDs. JSON output preserves the
public answer and option order. No output tokens are generated.

`--text-embedding q8-rows` retains Q8_0 token rows and returns F32 embeddings,
sharing the existing output weight for tied models. The default `dense-f32`
keeps the full F32 lookup table. Both modes load the complete vision model and
retain the same model/projector files. `loaded.json` includes exact token-table
storage and output-sharing details. See [loading qualification](../../../docs/lfm2-vl/D1_LOADING.md)
for measured effects and numerical limits.

For constructor measurements, replace `--requests ...` with `--load-only`.
This excludes `--trace` and `--profile`, executes zero model forwards and emits
`load-only.json` after device synchronization, session release and the final
retained-file recheck. It does not emit an inference `report.json`. Its admission
receipt uses the separate `candle.d1_load_artifact_admission.v1` schema.

Tensor traces are streamed into sibling staging, flushed, and published without
replacing an existing or racing destination. The output filesystem must support
hard links; unsupported publication returns an error. Staging is removed on
success or failure. `report.json` is written only after the final retained-file
and path-identity recheck succeeds. A failed recheck cannot publish a complete
report; the runner exits with failure and keeps available partial case evidence.

`--profile` adds opt-in `timings` to each case: preparation, image processing,
tokenization, vision, each question's prefill/readout, and trace observation.
It synchronizes device work at phase boundaries. Use unprofiled runs for
latency comparison; profiled totals include synchronization overhead. Trace
writing is measured in observation time. Model loading remains in `loaded.json`;
case elapsed time also includes image reading and plan-file writing.

Run the executable through `scripts/lfm2-vl/run-bounded-oracle.ps1` for a bounded
native Job with deadline, memory limit, process identity and verified release.
The caller owns admission and isolation. See [D1.md](../../../docs/lfm2-vl/D1.md)
for policy, API, source pins and qualification limits.

---
AI-edited: 2026-10-10 | agent=Codex/root | model=unknown | effort=unknown | task=d1-loading | change=documented opt-in Q8 rows and zero-forward constructor measurements
