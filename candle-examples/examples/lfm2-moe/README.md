# Local LFM2-MoE

Build with `cargo build --locked --offline -p candle-examples --example lfm2-moe`.
CPU is the default; CUDA requires the Candle `cuda` feature and `--cuda 0`.
Use exactly one of `--gguf FILE` or `--native-dir DIRECTORY`. Native directories
contain `config.json`, `tokenizer.json` and local safetensors (including a shard
index when applicable). Nothing is downloaded automatically.

`--inspect` validates configuration/header inventory and estimates memory
without loading payloads or creating a GPU device. `--tokenize-only` accepts
`--prompt-file FILE` and prints exact prompt/token IDs without loading weights.
For inference, supply an already rendered prompt through `--prompt-file`,
`--context 12288` and a bounded `--max-new-tokens` (default 64, maximum 2048).
GGUF uses F32 activations; native supports the selected `--dtype`.

`--teacher-forced-ids FILE` accepts a JSON array of token IDs. The output records
the top-1 prediction at each fixed continuation position.
`--logit-probe-ids FILE` records up to 64 selected logits per position and the
teacher-forced output includes exact prompt IDs for differential comparison.
Resource admission, wall-clock limits and process ownership belong to the invoking finite packet;
this example is not permission to run a production checkpoint.

---
AI-edited: 2026-11-10 | agent=Codex/root | task=lfm2-moe | change=document local loading, inspection and bounded probes
