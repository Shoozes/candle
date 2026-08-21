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

## Public hybrid loader

`candle_vlm::lfm2_vl::load_lfm2_vl_hybrid` assembles quantized LFM2 GGUF text
with either a split safetensors MMProj bundle or a direct llama.cpp-compatible
MMProj GGUF. The caller supplies every path, device, dtype, and execution
policy explicitly:

```rust,no_run
use candle::{DType, Device};
use candle_vlm::lfm2_vl::{
    load_lfm2_vl_hybrid, Lfm2VlHybridLoadOptions, Lfm2VlMmprojExecution,
    Lfm2VlMmprojSource,
};
use std::path::Path;

# fn load() -> candle::Result<()> {
let device = Device::Cpu;
let loaded = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
    text_gguf: Path::new("text.gguf"),
    mmproj: Lfm2VlMmprojSource::GgufFile(Path::new("mmproj.gguf")),
    tokenizer: Path::new("tokenizer.json"),
    processor_config: Some(Path::new("processor_config.json")),
    mmproj_execution: Lfm2VlMmprojExecution::Dense,
    vision_dtype: DType::F32,
    vision_device: &device,
    text_device: &device,
})?;

for path in &loaded.consumed_files {
    println!("{}", path.display());
}
# Ok(())
# }
```

The returned bundle contains the paired model, processor, prompt contract, and
the exact local file inventory consumed during construction. Applications
remain responsible for hashing, retained handles, resource admission, and
proof/report policy. The loader performs no discovery, download, or text-only
fallback.

Small metadata is validated before either model payload is opened. The
tokenizer is read through a 512 MiB ceiling and the optional processor config
through a 16 MiB ceiling; empty, oversized, malformed, or size-changing files
return controlled errors. These ceilings bound local metadata admission, not
model weights. Applications still own retained-handle and identity policy. The
returned inventory remains the authority for the exact files an application
must retain and identify.

## Example

Run the detailed example from the repository root:

```powershell
cargo run --locked --offline --release -p candle-examples --example lfm2-vl -- --help
```

See [`candle-examples/examples/lfm2-vl/README.md`](../candle-examples/examples/lfm2-vl/README.md)
for checkpoint preparation, native and GGUF/MMProj forms, device placement,
bounded inference, JSON evidence, and local verification commands.

---
AI-edited: 2026-08-13T13:36:16-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=hybrid-loader-hardening | change=documented bounded fail-fast tokenizer and processor admission
