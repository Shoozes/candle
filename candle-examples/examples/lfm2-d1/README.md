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

Run the executable through `scripts/lfm2-vl/run-bounded-oracle.ps1` for a bounded
native Job with deadline, memory limit, process identity and verified release.
The caller owns admission and isolation. See [D1.md](../../../docs/lfm2-vl/D1.md)
for policy, API, source pins and qualification limits.

---
AI-edited: 2026-10-08 | agent=Codex/root | model=unknown | effort=unknown | task=lfm2-d1 | change=documented standalone explicit-input runner
