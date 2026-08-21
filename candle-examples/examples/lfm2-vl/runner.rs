//! Deterministic, evidence-producing LFM2-VL inference.

use anyhow::{bail, Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::lfm2;
use candle_transformers::models::lfm2_vl::{
    CropKind, EncodedImages, GgufMmprojExecution, ImageTokenSpan, Lfm2VlDecodeTrace,
    Lfm2VlImageTrace, Lfm2VlModel, Lfm2VlPrefillTrace, ProcessedVisionBatch, QuantizedLfm2VlModel,
    VisionLimits,
};
use candle_vlm::lfm2_vl::{ExpandedPrompt, Lfm2VlProcessor, Lfm2VlPrompt};
use image::{DynamicImage, GenericImageView, ImageReader};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use super::trace::{self, HybridTraceCapture, NativeTraceCapture};

const CONTRACT: &str = "candle-lfm2-vl-inference-v1";
const MAX_COMPRESSED_IMAGE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_REPORTED_VOCAB: usize = 1 << 20;
const TOP_K: usize = 5;
const EOS_CANDIDATES: &[&str] = &[
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
];

include!("runner/types.rs");
include!("runner/runtime.rs");
include!("runner/run.rs");
include!("runner/generation.rs");
include!("runner/benchmark.rs");
include!("runner/evidence.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::MmprojExecutionArg;
    use crate::loading::{self, HybridLoadOptions, MmprojInput};
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::{lfm2, lfm2_vl::Lfm2VlConfig};
    use candle_vlm::lfm2_vl::{Lfm2VlProcessorConfig, PromptOptions};
    use image::{ImageFormat, Rgb, RgbImage};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::AddedToken;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let number = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "candle-lfm2-vl-runner-{}-{number}",
                std::process::id()
            ));
            std::fs::create_dir(&path)
                .with_context(|| format!("creating test directory {}", path.display()))?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tokenizer() -> Result<Tokenizer> {
        let vocab = (0..32u32)
            .map(|token_id| {
                let token = match token_id {
                    0 => "<unk>".to_owned(),
                    1 => "hello".to_owned(),
                    2 => "</s>".to_owned(),
                    3 => "<image>".to_owned(),
                    4 => "<|image_start|>".to_owned(),
                    5 => "<|image_end|>".to_owned(),
                    6 => "<|img_thumbnail|>".to_owned(),
                    7..=22 => {
                        let offset = token_id - 7;
                        format!("<|img_row_{}_col_{}|>", offset / 4 + 1, offset % 4 + 1)
                    }
                    _ => format!("token-{token_id}"),
                };
                (token, token_id)
            })
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .map_err(anyhow::Error::msg)?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let mut special_tokens = vec![
            AddedToken::from("<image>", true),
            AddedToken::from("<|image_start|>", true),
            AddedToken::from("<|image_end|>", true),
            AddedToken::from("<|img_thumbnail|>", true),
        ];
        for row in 1..=4 {
            for col in 1..=4 {
                special_tokens.push(AddedToken::from(
                    format!("<|img_row_{row}_col_{col}|>"),
                    true,
                ));
            }
        }
        tokenizer.add_special_tokens(&special_tokens);
        Ok(tokenizer)
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tiny fixture tensor {name}"))
    }

    fn tiny_text_gguf(tensors: &HashMap<String, Tensor>, config: &Lfm2VlConfig) -> Result<Vec<u8>> {
        let text = config.text_model_config()?;
        let root = "weights.model.language_model";
        let mut names = vec![
            (
                "token_embd.weight".to_owned(),
                format!("{root}.embed_tokens.weight"),
            ),
            (
                "output_norm.weight".to_owned(),
                format!("{root}.embedding_norm.weight"),
            ),
        ];
        for (layer, layer_type) in text.layer_types.iter().enumerate() {
            let native = format!("{root}.layers.{layer}");
            let gguf = format!("blk.{layer}");
            names.extend([
                (
                    format!("{gguf}.attn_norm.weight"),
                    format!("{native}.operator_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_norm.weight"),
                    format!("{native}.ffn_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_gate.weight"),
                    format!("{native}.feed_forward.w1.weight"),
                ),
                (
                    format!("{gguf}.ffn_down.weight"),
                    format!("{native}.feed_forward.w2.weight"),
                ),
                (
                    format!("{gguf}.ffn_up.weight"),
                    format!("{native}.feed_forward.w3.weight"),
                ),
            ]);
            match layer_type {
                lfm2::LayerType::Conv => names.extend([
                    (
                        format!("{gguf}.shortconv.in_proj.weight"),
                        format!("{native}.conv.in_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.out_proj.weight"),
                        format!("{native}.conv.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.conv.weight"),
                        format!("{native}.conv.conv.weight"),
                    ),
                ]),
                lfm2::LayerType::FullAttention => names.extend([
                    (
                        format!("{gguf}.attn_q.weight"),
                        format!("{native}.self_attn.q_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k.weight"),
                        format!("{native}.self_attn.k_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_v.weight"),
                        format!("{native}.self_attn.v_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_output.weight"),
                        format!("{native}.self_attn.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_q_norm.weight"),
                        format!("{native}.self_attn.q_layernorm.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k_norm.weight"),
                        format!("{native}.self_attn.k_layernorm.weight"),
                    ),
                ]),
            }
        }

        let mut qtensors = Vec::new();
        for (gguf_name, native_name) in names {
            let tensor = fixture_tensor(tensors, &native_name)?.contiguous()?;
            let dtype = if tensor.rank() == 2
                && tensor.dim(1)?.is_multiple_of(GgmlDType::Q8_0.block_size())
            {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            qtensors.push((gguf_name, QTensor::quantize(&tensor, dtype)?));
        }
        let to_u32 = |value: usize, label: &str| {
            u32::try_from(value).map_err(|_| anyhow::anyhow!("{label} exceeds u32"))
        };
        let metadata = vec![
            (
                "general.architecture".to_owned(),
                gguf_file::Value::String("lfm2".to_owned()),
            ),
            (
                "lfm2.attention.head_count".to_owned(),
                gguf_file::Value::U32(to_u32(text.num_attention_heads, "head count")?),
            ),
            (
                "lfm2.attention.head_count_kv".to_owned(),
                gguf_file::Value::Array(
                    text.layer_types
                        .iter()
                        .map(|kind| match kind {
                            lfm2::LayerType::FullAttention => {
                                to_u32(text.num_key_value_heads, "key/value head count")
                                    .map(gguf_file::Value::U32)
                            }
                            lfm2::LayerType::Conv => Ok(gguf_file::Value::U32(0)),
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
            (
                "lfm2.embedding_length".to_owned(),
                gguf_file::Value::U32(to_u32(text.hidden_size, "embedding length")?),
            ),
            (
                "lfm2.context_length".to_owned(),
                gguf_file::Value::U32(to_u32(text.max_position_embeddings, "context length")?),
            ),
            (
                "lfm2.block_count".to_owned(),
                gguf_file::Value::U32(to_u32(text.num_hidden_layers, "block count")?),
            ),
            (
                "lfm2.attention.layer_norm_rms_epsilon".to_owned(),
                gguf_file::Value::F32(text.norm_eps as f32),
            ),
            (
                "lfm2.rope.freq_base".to_owned(),
                gguf_file::Value::F32(text.rope_theta),
            ),
            (
                "lfm2.shortconv.l_cache".to_owned(),
                gguf_file::Value::U32(to_u32(text.conv_l_cache, "short-convolution cache")?),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_owned(),
                gguf_file::Value::U32(31),
            ),
        ];
        let metadata_refs: Vec<_> = metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let tensor_refs: Vec<_> = qtensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        let mut output = Cursor::new(Vec::new());
        gguf_file::write(&mut output, &metadata_refs, &tensor_refs)?;
        Ok(output.into_inner())
    }

    fn split_bundle_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_mmproj_tiny")
    }

    fn processor_and_prompt() -> Result<(Lfm2VlProcessor, Lfm2VlPrompt)> {
        let mut config = Lfm2VlProcessorConfig::default();
        config.do_resize = false;
        config.downsample_factor = 2;
        config.encoder_patch_size = 2;
        config.do_image_splitting = false;
        config.min_tiles = 1;
        config.max_tiles = 2;
        config.use_thumbnail = false;
        config.tile_size = 8;
        config.min_image_tokens = 1;
        config.max_image_tokens = 4;
        config.max_num_patches = Some(16);
        config.context_length = Some(64);
        let processor = Lfm2VlProcessor::from_config(&config)?;
        let prompt = Lfm2VlPrompt::new(
            tokenizer()?,
            Some(3),
            config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: Some(64),
            },
        )?;
        Ok((processor, prompt))
    }

    #[test]
    fn top_k_is_finite_and_breaks_ties_by_token_id() -> Result<()> {
        let tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let logits = Tensor::new(&[1f32, 3., 3., -2.], &Device::Cpu)?;
        let ranked = top_k(&logits, &tokenizer)?;
        assert_eq!(ranked[0].token_id, 1);
        assert_eq!(ranked[1].token_id, 2);
        let non_finite = Tensor::new(&[0f32, f32::NAN], &Device::Cpu)?;
        assert!(top_k(&non_finite, &tokenizer).is_err());
        Ok(())
    }

    #[test]
    fn generation_benchmark_statistics_use_median_and_mad() -> Result<()> {
        let (median, mad) = median_and_mad(&[1.0, 2.0, 3.0, 100.0])?;
        assert_eq!(median, 2.5);
        assert_eq!(mad, 1.0);
        assert!(median_and_mad(&[f64::MAX, f64::MAX])?.0.is_finite());
        assert!(median_and_mad(&[]).is_err());
        assert!(median_and_mad(&[f64::NAN]).is_err());
        let zero_duration = median_and_mad(&[0.0]).expect_err("zero duration was accepted");
        assert!(zero_duration.to_string().contains("positive"));
        assert!(median_and_mad(&[-1.0]).is_err());
        assert!(ensure_benchmark_reaches_decode(&[1, 2]).is_ok());
        let prefill_only =
            ensure_benchmark_reaches_decode(&[1]).expect_err("prefill-only benchmark was accepted");
        assert!(prefill_only.to_string().contains("cached decode"));
        assert!(ensure_benchmark_ids(0, "measured", &[1, 2], &[1, 3]).is_err());
        Ok(())
    }

    #[test]
    fn native_fixture_runs_image_prefill_decode_and_exact_cache_replay() -> Result<()> {
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &Device::Cpu)?;
        let model = Lfm2VlModel::new(&config, weights.pp("weights"))?;
        let (processor, prompt) = processor_and_prompt()?;
        let dir = TestDir::new()?;
        let image_path = dir.path().join("fixture.png");
        let pixels = RgbImage::from_fn(8, 4, |x, y| {
            Rgb([(x * 17) as u8, (y * 41) as u8, ((x + y) * 13) as u8])
        });
        DynamicImage::ImageRgb8(pixels).save_with_format(&image_path, ImageFormat::Png)?;
        let trace_output = dir.path().join("trace");
        let model_inputs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/lfm2_vl_tiny/tensors.safetensors")];
        let image_paths = vec![image_path];
        let report = run_native(
            &model,
            &processor,
            &prompt,
            InferenceRequest {
                backend: "native-fixture",
                model_inputs: &model_inputs,
                prompt: "<image> hello",
                image_paths: &image_paths,
                max_new_tokens: 3,
                vision_batch_size: 1,
                eos_token_id: None,
                timings: false,
                benchmark_generation: false,
                trace_output: Some(trace_output.as_path()),
            },
        )?;

        assert!(report.cache_reset_exact);
        assert_eq!(report.contract, CONTRACT);
        assert_eq!(
            report.model_inputs[0].sha256.as_deref().map(str::len),
            Some(64)
        );
        assert_eq!(report.image_files.len(), 1);
        assert_eq!(report.image_files[0].width, 8);
        assert_eq!(report.image_files[0].height, 4);
        assert_eq!(report.image_files[0].sha256.len(), 64);
        assert_eq!(report.processed_crops.len(), 1);
        assert_eq!(report.processed_crops[0].patch_rows, 2);
        assert_eq!(report.processed_crops[0].patch_cols, 4);
        assert_eq!(report.image_spans.len(), 1);
        assert_eq!(report.image_spans[0].end - report.image_spans[0].start, 2);
        assert_eq!(report.generation.prefill_logits_sha256.len(), 64);
        assert!(!report.generation.generated_ids.is_empty());
        assert!(report.generation.generated_ids.len() <= 3);
        let json = serde_json::to_value(&report)?;
        assert_eq!(json["cache_reset_exact"], true);
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("manifest.json"),
        )?)?;
        assert_eq!(manifest["mode"], "native-trace");
        assert_eq!(manifest["weights_serialized"], false);
        assert_eq!(manifest["model_inputs_reverified"], true);
        assert!(manifest["tensor_inventory"]["stage.language.prefill_logits"].is_object());
        let trace_metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("metadata.json"),
        )?)?;
        assert_eq!(
            trace_metadata["model_inputs"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(trace_metadata["model_inputs"][0]["kind"], "file");
        assert_eq!(
            trace_metadata["model_inputs"][0]["sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.pixel_values"]["dtype"],
            "float32"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.pixel_attention_mask"]["dtype"],
            "int32"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.attention_mask"]["dtype"],
            "int64"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.image_rgb_u8"]["dtype"],
            "uint8"
        );
        let trace_tensors =
            candle::safetensors::load(trace_output.join("tensors.safetensors"), &Device::Cpu)?;
        assert!(trace_tensors.contains_key("stage.projector.output"));
        assert!(trace_tensors.contains_key("stage.vision.encoder_layer.0"));
        assert_eq!(
            trace_tensors["input.attention_mask"].dims(),
            trace_tensors["input.input_ids"].dims()
        );
        assert_eq!(
            trace_tensors["input.projector_crop_ranges"].to_vec2::<i64>()?,
            vec![vec![0, 8]]
        );
        assert_eq!(trace_tensors["input.decode_token_ids"].dims(), [1, 3]);
        assert_eq!(
            trace_tensors["stage.language.decode_logits"].dims(),
            [1, 3, 32]
        );
        Ok(())
    }

    #[test]
    fn hybrid_fixture_reports_exact_files_eos_and_cache_replay() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let dir = TestDir::new()?;
        let text_path = dir.path().join("text.gguf");
        std::fs::write(&text_path, tiny_text_gguf(&tensors, &config)?)?;
        let tokenizer_path = dir.path().join("tokenizer.json");
        tokenizer()?
            .save(&tokenizer_path, false)
            .map_err(anyhow::Error::msg)?;
        let mut loaded = loading::load_hybrid(HybridLoadOptions {
            text_gguf: &text_path,
            mmproj: MmprojInput::SplitDirectory(&split_bundle_dir()),
            tokenizer: &tokenizer_path,
            processor_config: None,
            mmproj_execution: MmprojExecutionArg::Dense,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })?;
        let image_path = dir.path().join("fixture.png");
        let pixels = RgbImage::from_fn(8, 4, |x, y| {
            Rgb([(x * 17) as u8, (y * 41) as u8, ((x + y) * 13) as u8])
        });
        DynamicImage::ImageRgb8(pixels).save_with_format(&image_path, ImageFormat::Png)?;
        let image_paths = vec![image_path];
        let report = run_hybrid(
            &mut loaded.model,
            &loaded.processor,
            &loaded.prompt,
            InferenceRequest {
                backend: "hybrid-split-fixture",
                model_inputs: &loaded.consumed_files,
                prompt: "<image> hello",
                image_paths: &image_paths,
                max_new_tokens: 3,
                vision_batch_size: 1,
                eos_token_id: None,
                timings: false,
                benchmark_generation: false,
                trace_output: None,
            },
        )?;

        assert!(report.cache_reset_exact);
        assert_eq!(report.eos.token_id, Some(31));
        assert_eq!(report.eos.source, "gguf_metadata");
        assert_eq!(report.generation.steps.len(), 3);
        assert_eq!(report.model_inputs.len(), 5);
        assert!(report.model_inputs.iter().all(|input| {
            input.kind == "file"
                && input.bytes.is_some()
                && input.sha256.as_deref().is_some_and(|hash| hash.len() == 64)
        }));
        assert!(report
            .model_inputs
            .iter()
            .any(|input| input.path.ends_with("mmproj.safetensors")));
        let json = serde_json::to_string(&report)?;
        assert_eq!(json.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn direct_q8_fixture_publishes_hybrid_evidence_and_retains_q8_selection() -> Result<()> {
        let device = Device::Cpu;
        let fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_loader_tiny");
        let text_path = fixture_root.join("text.gguf");
        let mmproj_path = fixture_root.join("mmproj-q8.gguf");
        let tokenizer_path = fixture_root.join("tokenizer.json");
        let processor_config = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/lfm2_vl_mmproj_tiny/processor_config.json");
        let mut loaded = loading::load_hybrid(HybridLoadOptions {
            text_gguf: &text_path,
            mmproj: MmprojInput::GgufFile(&mmproj_path),
            tokenizer: &tokenizer_path,
            processor_config: Some(&processor_config),
            mmproj_execution: MmprojExecutionArg::Q8,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })?;
        assert_eq!(
            loaded.model.mmproj().gguf_execution(),
            Some(GgufMmprojExecution::Q8_0)
        );
        assert!(loaded.model.mmproj().native_quantized_tensor_count() > 0);

        let dir = TestDir::new()?;
        let image_path = dir.path().join("fixture.png");
        let pixels = RgbImage::from_fn(8, 4, |x, y| {
            Rgb([(x * 17) as u8, (y * 41) as u8, ((x + y) * 13) as u8])
        });
        DynamicImage::ImageRgb8(pixels).save_with_format(&image_path, ImageFormat::Png)?;
        let trace_output = dir.path().join("hybrid-trace");
        let image_paths = vec![image_path];
        let report = run_hybrid(
            &mut loaded.model,
            &loaded.processor,
            &loaded.prompt,
            InferenceRequest {
                backend: "hybrid-gguf-q8-fixture",
                model_inputs: &loaded.consumed_files,
                prompt: "<image> hello",
                image_paths: &image_paths,
                max_new_tokens: 3,
                vision_batch_size: 1,
                eos_token_id: None,
                timings: false,
                benchmark_generation: false,
                trace_output: Some(trace_output.as_path()),
            },
        )?;
        assert!(report.cache_reset_exact);
        let metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("metadata.json"),
        )?)?;
        assert_eq!(metadata["mode"], "hybrid-trace");
        assert_eq!(metadata["execution_mode"], "q8_0-native");
        assert!(metadata["q8_tensor_count"].as_u64().unwrap_or(0) > 0);
        assert_eq!(metadata["cache_reset_exact"], true);
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("manifest.json"),
        )?)?;
        assert_eq!(manifest["mode"], "hybrid-trace");
        let tensors = candle::safetensors::load(trace_output.join("tensors.safetensors"), &device)?;
        assert!(tensors.contains_key("stage.projector.output"));
        assert!(tensors.contains_key("stage.language.prefill_logits"));
        assert!(tensors.contains_key("stage.language.decode_logits"));
        assert_eq!(tensors["input.decode_token_ids"].dims(), [1, 3]);
        Ok(())
    }

    #[test]
    fn model_input_evidence_rejects_directories() -> Result<()> {
        let dir = TestDir::new()?;
        let error = inspect_input_paths(&[dir.path().to_path_buf()])
            .expect_err("directory evidence unexpectedly accepted");
        assert!(error.to_string().contains("not a regular file"));
        Ok(())
    }

    #[test]
    fn traced_model_input_evidence_detects_content_changes() -> Result<()> {
        let dir = TestDir::new()?;
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, b"before")?;
        let paths = vec![path.clone()];
        let expected = inspect_input_paths(&paths)?;
        verify_input_paths_unchanged(&paths, &expected)?;
        std::fs::write(path, b"after")?;
        let error = verify_input_paths_unchanged(&paths, &expected)
            .expect_err("changed trace input unexpectedly passed revalidation");
        assert!(error
            .to_string()
            .contains("changed during traced inference"));
        Ok(())
    }
}
