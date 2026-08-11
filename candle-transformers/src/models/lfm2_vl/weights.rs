//! Split/direct MMProj loading and quantized-text hybrid execution.

use super::gguf::{GgufMmprojExecution, GgufMmprojMetadata};
use super::model::{
    encode_images_with_parts, merge_projected_embeddings, preflight_packed_vision_limits,
};
use super::{
    EncodedImages, ImageTokenSpan, Lfm2VlConfig, Lfm2VlMmprojConfig, Lfm2VlProjector,
    ProcessedVisionBatch, VisionLimits,
};
use crate::models::{quantized_lfm2, siglip2};
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MMPROJ_FORMAT: &str = "candle-mmproj";
const MMPROJ_VERSION: u32 = 1;
const MMPROJ_NAMESPACE_VERSION: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_MMPROJ_TENSORS: usize = 16_384;
const MAX_VISION_LAYERS: usize = 512;
const VISION_ROOT: &str = "model.vision_tower.vision_model";
const PROJECTOR_ROOT: &str = "model.multi_modal_projector";

include!("weights/manifest.rs");
include!("weights/runtime.rs");
include!("weights/safetensors.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        lfm2,
        lfm2_vl::{GgufMmprojExecution, Lfm2VlModel},
    };
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    use candle::{IndexOp, Tensor};
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::path::PathBuf;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

    fn bundle_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_mmproj_tiny")
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
    }

    fn fixture_batch(tensors: &HashMap<String, Tensor>) -> Result<ProcessedVisionBatch> {
        Ok(ProcessedVisionBatch {
            pixel_values: fixture_tensor(tensors, "input.pixel_values")?.clone(),
            pixel_attention_mask: fixture_tensor(tensors, "input.pixel_attention_mask")?.clone(),
            spatial_shapes: fixture_tensor(tensors, "input.spatial_shapes")?.clone(),
            crops: vec![super::super::CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: super::super::CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![super::super::ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn image_spans(input_ids: &Tensor, image_token_id: u32) -> Result<Vec<ImageTokenSpan>> {
        let values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut spans = Vec::new();
        for (batch_index, row) in values.iter().enumerate() {
            let mut start = None;
            for (position, &token_id) in row.iter().enumerate() {
                if token_id == image_token_id && start.is_none() {
                    start = Some(position);
                } else if token_id != image_token_id {
                    if let Some(start) = start.take() {
                        spans.push(ImageTokenSpan::new(batch_index, start, position));
                    }
                }
            }
            if let Some(start) = start {
                spans.push(ImageTokenSpan::new(batch_index, start, row.len()));
            }
        }
        Ok(spans)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("{label}: element count mismatch")
        }
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0f32, f32::max);
        eprintln!("lfm2_vl hybrid {label}: max_abs={max_abs:.9e}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        Ok(())
    }

    fn synthetic_text_metadata(config: &Lfm2VlConfig) -> quantized_lfm2::Lfm2GgufMetadata {
        let text = config.text_model_config().expect("fixed tiny text config");
        quantized_lfm2::Lfm2GgufMetadata {
            architecture: "lfm2".to_string(),
            embedding_length: text.hidden_size,
            context_length: text.max_position_embeddings,
            block_count: text.num_hidden_layers,
            head_count: text.num_attention_heads,
            head_count_kv: text
                .layer_types
                .iter()
                .map(|kind| match kind {
                    lfm2::LayerType::FullAttention => text.num_key_value_heads,
                    lfm2::LayerType::Conv => 0,
                })
                .collect(),
            rms_norm_eps: text.norm_eps,
            rope_freq_base: text.rope_theta,
            shortconv_l_cache: text.conv_l_cache,
            eos_token_id: None,
            tied_output: true,
        }
    }

    fn tiny_text_gguf(tensors: &HashMap<String, Tensor>, config: &Lfm2VlConfig) -> Result<Vec<u8>> {
        let text = config.text_model_config()?;
        let root = "weights.model.language_model";
        let mut names = vec![
            (
                "token_embd.weight".to_string(),
                format!("{root}.embed_tokens.weight"),
            ),
            (
                "output_norm.weight".to_string(),
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
        let mut q8_count = 0usize;
        for (gguf_name, native_name) in names {
            let tensor = fixture_tensor(tensors, &native_name)?.contiguous()?;
            let dtype = if tensor.rank() == 2
                && tensor.dim(1)?.is_multiple_of(GgmlDType::Q8_0.block_size())
            {
                q8_count += 1;
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            qtensors.push((gguf_name, QTensor::quantize(&tensor, dtype)?));
        }
        if q8_count == 0 {
            candle::bail!("tiny GGUF fixture contains no Q8_0 matrices")
        }
        let metadata = vec![
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("lfm2".to_string()),
            ),
            (
                "lfm2.attention.head_count".to_string(),
                gguf_file::Value::U32(text.num_attention_heads as u32),
            ),
            (
                "lfm2.attention.head_count_kv".to_string(),
                gguf_file::Value::Array(
                    text.layer_types
                        .iter()
                        .map(|kind| match kind {
                            lfm2::LayerType::FullAttention => {
                                gguf_file::Value::U32(text.num_key_value_heads as u32)
                            }
                            lfm2::LayerType::Conv => gguf_file::Value::U32(0),
                        })
                        .collect(),
                ),
            ),
            (
                "lfm2.embedding_length".to_string(),
                gguf_file::Value::U32(text.hidden_size as u32),
            ),
            (
                "lfm2.context_length".to_string(),
                gguf_file::Value::U32(text.max_position_embeddings as u32),
            ),
            (
                "lfm2.block_count".to_string(),
                gguf_file::Value::U32(text.num_hidden_layers as u32),
            ),
            (
                "lfm2.attention.layer_norm_rms_epsilon".to_string(),
                gguf_file::Value::F32(text.norm_eps as f32),
            ),
            (
                "lfm2.rope.freq_base".to_string(),
                gguf_file::Value::F32(text.rope_theta),
            ),
            (
                "lfm2.shortconv.l_cache".to_string(),
                gguf_file::Value::U32(text.conv_l_cache as u32),
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

    #[test]
    fn split_manifest_and_tensor_inventory_are_exact() -> Result<()> {
        let device = Device::Cpu;
        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &device)?;
        assert!(mmproj.report.is_clean());
        assert_eq!(mmproj.report.loaded_tensors.len(), 43);
        let manifest = mmproj
            .metadata
            .split_manifest()
            .ok_or_else(|| candle::Error::Msg("expected split MMProj metadata".into()))?;
        assert_eq!(manifest.vision_layer_count, 2);
        assert_eq!(manifest.expected_text_layer_count, 2);
        assert_eq!(manifest.tensor_count, 43);
        assert_eq!(
            mmproj.metadata.source_revision.as_deref(),
            Some("fc6221ca597f3315e4f82fc2df606783267b34ba")
        );
        assert!(mmproj
            .report
            .loaded_tensors
            .iter()
            .all(|name| name.starts_with(VISION_ROOT) || name.starts_with(PROJECTOR_ROOT)));

        let mut bad = manifest.clone();
        let linear_2 = format!("{PROJECTOR_ROOT}.linear_2.weight");
        bad.tensor_inventory
            .get_mut(&linear_2)
            .ok_or_else(|| candle::Error::Msg("missing fixed projector weight".into()))?
            .shape[0] += 1;
        let report = inspect_safetensors(
            bundle_dir().join("mmproj.safetensors"),
            &bad,
            DType::F32,
            &device,
        )?;
        assert!(report
            .shape_or_dtype_mismatches
            .iter()
            .any(|mismatch| mismatch.contains(&linear_2)));
        let error = bad.validate().unwrap_err().to_string();
        assert!(error.contains("linear_2.weight") && error.contains("expected"));

        let mut bad_names = manifest.clone();
        let moved = bad_names
            .tensor_inventory
            .remove(&linear_2)
            .ok_or_else(|| candle::Error::Msg("missing fixed projector weight".into()))?;
        let fake = format!("{PROJECTOR_ROOT}.unexpected.weight");
        bad_names.tensor_inventory.insert(fake.clone(), moved);
        let report = inspect_safetensors(
            bundle_dir().join("mmproj.safetensors"),
            &bad_names,
            DType::F32,
            &device,
        )?;
        assert_eq!(report.missing_tensors, vec![fake]);
        assert_eq!(report.unexpected_tensors, vec![linear_2]);
        Ok(())
    }

    #[test]
    fn manifest_requires_pinned_source_provenance() -> Result<()> {
        let manifest = MmprojManifest::from_json(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)?;
        let mut missing_model = manifest.clone();
        missing_model.source_model.clear();
        assert!(missing_model
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source model"));

        let mut mutable_revision = manifest.clone();
        mutable_revision.source_revision = "main".to_string();
        assert!(mutable_revision
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source revision"));

        let mut json: serde_json::Value = serde_json::from_str(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)
        .map_err(candle::Error::wrap)?;
        json.as_object_mut()
            .ok_or_else(|| candle::Error::Msg("fixed manifest is not an object".into()))?
            .remove("source_revision");
        let json = serde_json::to_string(&json).map_err(candle::Error::wrap)?;
        let error = MmprojManifest::from_json(&json).unwrap_err().to_string();
        assert!(error.contains("source_revision"));
        Ok(())
    }

    #[test]
    fn safetensors_preflight_bounds_header_and_tensor_count() -> Result<()> {
        let oversized_header = (MAX_SAFETENSORS_HEADER_BYTES as u64 + 1)
            .to_le_bytes()
            .to_vec();
        let error = safetensors_inventory(&oversized_header)
            .unwrap_err()
            .to_string();
        assert!(error.contains("header length") && error.contains("outside"));

        let mut header = serde_json::Map::new();
        for index in 0..=MAX_MMPROJ_TENSORS {
            header.insert(
                format!("tensor.{index}"),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }),
            );
        }
        let raw_header =
            serde_json::to_vec(&serde_json::Value::Object(header)).map_err(candle::Error::wrap)?;
        assert!(raw_header.len() < MAX_SAFETENSORS_HEADER_BYTES);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(raw_header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&raw_header);
        let error = safetensors_inventory(&bytes).unwrap_err().to_string();
        assert!(error.contains("tensor count") && error.contains("outside"));
        Ok(())
    }

    #[test]
    fn pairing_rejects_every_cross_artifact_mismatch() -> Result<()> {
        let manifest = MmprojManifest::from_json(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)?;
        let text = synthetic_text_metadata(&manifest.model_config);
        let report = manifest.validate_pair(&text, 2, 2, 3)?;
        assert!(report.only_projected_features_cross_devices);
        assert_eq!(report.text_output_resolution, "tied token embeddings");

        let mut wrong = text.clone();
        wrong.architecture = "llama".to_string();
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        let mut wrong = text.clone();
        wrong.embedding_length += 1;
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        let mut wrong = text.clone();
        wrong.block_count += 1;
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        assert!(manifest.validate_pair(&text, 4, 2, 3).is_err());
        assert!(manifest.validate_pair(&text, 2, 1, 3).is_err());
        assert!(manifest.validate_pair(&text, 2, 2, 4).is_err());
        Ok(())
    }

    #[test]
    fn split_features_and_hybrid_text_match_unified_native_model() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let unified_vb = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?;
        let native = Lfm2VlModel::new(&config, unified_vb.pp("weights"))?;
        let batch = fixture_batch(&tensors)?;
        let native_encoded = native.encode_images(&batch, 1)?;

        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &device)?;
        let split_encoded = mmproj.encode_images(&batch, 1)?;
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(mmproj.device()));
        assert_close(
            &split_encoded.embeddings,
            &native_encoded.embeddings,
            1e-6,
            "split image features",
        )?;
        assert_eq!(
            split_encoded.per_crop_ranges,
            native_encoded.per_crop_ranges
        );
        assert_eq!(
            split_encoded.per_image_ranges,
            native_encoded.per_image_ranges
        );

        let direct_bytes = crate::models::lfm2_vl::gguf::tests::tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &[],
            None,
            false,
        )?;
        let direct_hash = format!("{:x}", Sha256::digest(&direct_bytes));
        assert_eq!(
            direct_hash,
            "7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256"
        );
        let mut direct_reader = Cursor::new(direct_bytes);
        let direct_mmproj = Mmproj::from_gguf(&mut direct_reader, DType::F32, &device, 3)?;
        let direct_encoded = direct_mmproj.encode_images(&batch, 1)?;
        assert_close(
            &direct_encoded.embeddings,
            &native_encoded.embeddings,
            1e-6,
            "direct GGUF image features",
        )?;
        let q8_direct_bytes = crate::models::lfm2_vl::gguf::tests::tiny_mmproj_gguf(
            &tensors,
            &config,
            true,
            &[],
            None,
            false,
        )?;
        let q8_direct_hash = format!("{:x}", Sha256::digest(&q8_direct_bytes));
        eprintln!("lfm2_vl hybrid deterministic Q8_0 MMProj GGUF SHA-256: {q8_direct_hash}");
        let q8_direct_mmproj =
            Mmproj::from_gguf_q8(&mut Cursor::new(q8_direct_bytes), DType::F32, &device, 3)?;
        assert_eq!(
            q8_direct_mmproj.gguf_execution(),
            Some(GgufMmprojExecution::Q8_0)
        );
        let q8_direct_encoded = q8_direct_mmproj.encode_images(&batch, 1)?;
        assert_close(
            &q8_direct_encoded.embeddings,
            &native_encoded.embeddings,
            5e-4,
            "native Q8_0 GGUF image features",
        )?;

        let text_config = config.text_model_config()?;
        let gguf_bytes = tiny_text_gguf(&tensors, &config)?;
        let gguf_hash = format!("{:x}", Sha256::digest(&gguf_bytes));
        eprintln!("lfm2_vl hybrid deterministic text GGUF SHA-256: {gguf_hash}");
        assert_eq!(
            gguf_hash,
            "8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1"
        );
        let mut malformed_reader = Cursor::new(gguf_bytes.clone());
        let mut malformed_gguf = gguf_file::Content::read(&mut malformed_reader)?;
        malformed_gguf.metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(2),
        );
        assert_eq!(
            quantized_lfm2::inspect_gguf_metadata(&malformed_gguf)?.eos_token_id,
            Some(2)
        );
        malformed_gguf.metadata.insert(
            "lfm2.rope.freq_base".to_string(),
            gguf_file::Value::String("not-a-frequency".to_string()),
        );
        assert!(quantized_lfm2::inspect_gguf_metadata(&malformed_gguf)
            .unwrap_err()
            .to_string()
            .contains("f32"));
        let mut gguf_reader = Cursor::new(gguf_bytes.clone());
        let gguf = gguf_file::Content::read(&mut gguf_reader)?;
        let quantized_text =
            quantized_lfm2::ModelWeights::from_gguf(gguf, &mut gguf_reader, &device)?;
        let mut split_hybrid = QuantizedLfm2VlModel::new(quantized_text, mmproj, 2, 2, 3)?;
        let mut direct_text_reader = Cursor::new(gguf_bytes.clone());
        let direct_text_gguf = gguf_file::Content::read(&mut direct_text_reader)?;
        let direct_text = quantized_lfm2::ModelWeights::from_gguf(
            direct_text_gguf,
            &mut direct_text_reader,
            &device,
        )?;
        let mut direct_hybrid = QuantizedLfm2VlModel::new(direct_text, direct_mmproj, 2, 2, 3)?;
        let mut q8_text_reader = Cursor::new(gguf_bytes);
        let q8_text_gguf = gguf_file::Content::read(&mut q8_text_reader)?;
        let q8_text =
            quantized_lfm2::ModelWeights::from_gguf(q8_text_gguf, &mut q8_text_reader, &device)?;
        let mut q8_direct_hybrid = QuantizedLfm2VlModel::new(q8_text, q8_direct_mmproj, 2, 2, 3)?;
        assert!(split_hybrid.vision_device().same_device(&device));
        assert!(split_hybrid.text_device().same_device(&device));
        assert!(direct_hybrid.vision_device().same_device(&device));
        assert!(direct_hybrid.text_device().same_device(&device));
        assert!(q8_direct_hybrid.vision_device().same_device(&device));
        assert!(q8_direct_hybrid.text_device().same_device(&device));
        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let spans = image_spans(input_ids, 3)?;
        let mut native_cache = lfm2::Cache::new(true, DType::F32, &text_config, &device)?;
        let native_prefill =
            native.prefill(input_ids, &spans, Some(&native_encoded), &mut native_cache)?;
        let hybrid_prefill = split_hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        let direct_prefill = direct_hybrid.prefill(input_ids, &spans, Some(&direct_encoded))?;
        let q8_direct_prefill =
            q8_direct_hybrid.prefill(input_ids, &spans, Some(&q8_direct_encoded))?;
        let native_last = native_prefill.i((.., input_ids.dim(1)? - 1, ..))?;
        assert_close(&hybrid_prefill, &native_last, 1e-4, "hybrid prefill logits")?;
        assert_close(
            &direct_prefill,
            &native_last,
            1e-4,
            "direct GGUF prefill logits",
        )?;
        assert_close(
            &q8_direct_prefill,
            &native_last,
            1e-3,
            "native Q8_0 GGUF prefill logits",
        )?;

        let decode_ids = fixture_tensor(&tensors, "input.decode_token_ids")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let native_logits = native.decode(&token, 5 + step, &mut native_cache)?;
            let hybrid_logits = split_hybrid.decode(&token, 5 + step)?;
            let direct_logits = direct_hybrid.decode(&token, 5 + step)?;
            let q8_direct_logits = q8_direct_hybrid.decode(&token, 5 + step)?;
            assert_close(
                &hybrid_logits,
                &native_logits,
                1e-4,
                &format!("hybrid cached decode step {step}"),
            )?;
            assert_close(
                &direct_logits,
                &native_logits,
                1e-4,
                &format!("direct GGUF cached decode step {step}"),
            )?;
            assert_close(
                &q8_direct_logits,
                &native_logits,
                1e-3,
                &format!("native Q8_0 GGUF cached decode step {step}"),
            )?;
        }

        let reset = split_hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        assert_close(&reset, &hybrid_prefill, 1e-6, "hybrid cache reset")?;
        let direct_reset = direct_hybrid.prefill(input_ids, &spans, Some(&direct_encoded))?;
        assert_close(
            &direct_reset,
            &direct_prefill,
            1e-6,
            "direct GGUF cache reset",
        )?;
        let q8_direct_reset =
            q8_direct_hybrid.prefill(input_ids, &spans, Some(&q8_direct_encoded))?;
        assert_close(
            &q8_direct_reset,
            &q8_direct_prefill,
            1e-6,
            "native Q8_0 GGUF cache reset",
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn split_vision_cuda_text_cpu_transfers_only_projected_features() -> Result<()> {
        let text_device = Device::Cpu;
        let vision_device = Device::new_cuda(0)?;
        assert!(!vision_device.same_device(&text_device));

        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &text_device)?;
        let unified_vb =
            VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &text_device)?;
        let native = Lfm2VlModel::new(&config, unified_vb.pp("weights"))?;
        let batch = fixture_batch(&tensors)?;
        assert!(batch.pixel_values.device().same_device(&text_device));
        let native_encoded = native.encode_images(&batch, 1)?;

        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &vision_device)?;
        let split_encoded = mmproj.encode_images(&batch, 1)?;
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(&vision_device));
        assert!(batch.pixel_values.device().same_device(&text_device));
        assert_close(
            &split_encoded.embeddings.to_device(&text_device)?,
            &native_encoded.embeddings,
            1e-5,
            "CUDA split image features",
        )?;

        let gguf_bytes = tiny_text_gguf(&tensors, &config)?;
        let mut gguf_reader = Cursor::new(gguf_bytes);
        let gguf = gguf_file::Content::read(&mut gguf_reader)?;
        let quantized_text =
            quantized_lfm2::ModelWeights::from_gguf(gguf, &mut gguf_reader, &text_device)?;
        let mut hybrid = QuantizedLfm2VlModel::new(quantized_text, mmproj, 2, 2, 3)?;
        assert!(hybrid.vision_device().same_device(&vision_device));
        assert!(hybrid.text_device().same_device(&text_device));
        assert!(
            hybrid
                .pairing_report()
                .only_projected_features_cross_devices
        );

        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let spans = image_spans(input_ids, 3)?;
        let text_config = config.text_model_config()?;
        let mut native_cache = lfm2::Cache::new(true, DType::F32, &text_config, &text_device)?;
        let native_prefill =
            native.prefill(input_ids, &spans, Some(&native_encoded), &mut native_cache)?;
        let hybrid_prefill = hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        assert!(hybrid_prefill.device().same_device(&text_device));
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(&vision_device));
        let native_last = native_prefill.i((.., input_ids.dim(1)? - 1, ..))?;
        assert_close(
            &hybrid_prefill,
            &native_last,
            1e-4,
            "CUDA-vision/CPU-text hybrid prefill logits",
        )?;
        Ok(())
    }
}
