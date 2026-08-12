# candle-vlm

`candle-vlm` contains Rust-native multimodal preprocessing and prompt handling
for Candle's LFM2.5-VL/MMProj integration. The public `lfm2_vl` module owns
SigLIP2 NaFlex crop/patch preparation, image-token expansion, and the checked
image-span contracts used by the example runner.

## Supported boundary

The current release evidence covers the `LiquidAI/LFM2.5-VL-450M` and
`LiquidAI/LFM2.5-VL-1.6B` native checkpoint contracts, with CPU F32 as the
portable baseline. The 450M runtime has also been proven on native Windows
with CPU/CPU F32, all-CUDA F32/BF16/F16, CPU-text/CUDA-vision F32, and
CUDA-text/CPU-vision F32, using the bounded example runner. A CPU component
must use F32; BF16 and F16 are rejected before model loading when the resolved
device is CPU.

The crate performs no network access, vendors no model weights, and does not
download checkpoints. Request-wide image, crop, patch, and sequence limits are
validated from configuration before large allocations; external dimensions
use checked arithmetic, and malformed images, token spans, or count mismatches
return errors rather than truncating or falling back to text-only behavior.
It does not provide a generic VLM abstraction. Lower-bit vision execution,
video, true text batching, and WebGPU are future work rather than supported
behavior.

## Example

Run the detailed example from the repository root:

```powershell
cargo run --locked --release -p candle-examples --example lfm2-vl -- --help
```

See [`candle-examples/examples/lfm2-vl/README.md`](../candle-examples/examples/lfm2-vl/README.md)
for checkpoint preparation, native and GGUF/MMProj forms, device placement,
bounded inference, JSON evidence, and local verification commands.

---
AI-edited: 2026-08-11T23:12:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=release-closeout | change=added crate purpose, safety boundary, and supported-device contract
