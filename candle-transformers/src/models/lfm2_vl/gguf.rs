//! llama.cpp-compatible GGUF MMProj loading through dense or native Q8 execution.

use super::weights::{Mmproj, MmprojLoadReport, MmprojMetadata};
use super::{Lfm2VlMmprojConfig, Lfm2VlProjector};
use crate::models::siglip2;
use candle::quantized::{
    gguf_file::{self, Value},
    GgmlDType,
};
use candle::{DType, Device, Result};
use candle_nn::{Activation, VarBuilder};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_GGUF_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_DENSE_MMPROJ_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_ESTIMATED_MMPROJ_PEAK_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_GGUF_MMPROJ_TENSORS: usize = 16_384;
const MAX_GGUF_MMPROJ_METADATA: u64 = 16_384;
const MAX_GGUF_MMPROJ_ARRAY_ELEMENTS: u64 = 16_384;
const MAX_GGUF_MMPROJ_STRING_BYTES: u64 = 1024 * 1024;
const MAX_GGUF_MMPROJ_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_VISION_LAYERS: usize = 512;
const NATIVE_VISION_ROOT: &str = "model.vision_tower.vision_model";
const NATIVE_PROJECTOR_ROOT: &str = "model.multi_modal_projector";

include!("gguf/types.rs");
include!("gguf/loading.rs");
include!("gguf/metadata.rs");
include!("gguf/inventory.rs");
include!("gguf/metadata_values.rs");

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::models::lfm2_vl::{
        CropKind, CropMeta, ImageMeta, Lfm2VlConfig, Lfm2VlModel, ProcessedVisionBatch,
    };
    use crate::models::{lfm2, quantized_lfm2};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::Tensor;
    use sha2::{Digest, Sha256};
    use std::io::Cursor;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

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
            crops: vec![CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn max_abs(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("GGUF MMProj parity element count mismatch")
        }
        Ok(actual
            .iter()
            .zip(expected)
            .map(|(&lhs, rhs)| (lhs - rhs).abs())
            .fold(0f32, f32::max))
    }

    fn cosine_similarity(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("GGUF MMProj cosine element count mismatch")
        }
        let (dot, actual_norm, expected_norm) = actual.iter().zip(expected).fold(
            (0f64, 0f64, 0f64),
            |(dot, actual_norm, expected_norm), (&lhs, rhs)| {
                let lhs = lhs as f64;
                let rhs = rhs as f64;
                (
                    dot + lhs * rhs,
                    actual_norm + lhs * lhs,
                    expected_norm + rhs * rhs,
                )
            },
        );
        if actual_norm == 0.0 || expected_norm == 0.0 {
            candle::bail!("GGUF MMProj cosine requires non-zero tensors")
        }
        Ok((dot / (actual_norm.sqrt() * expected_norm.sqrt())) as f32)
    }

    fn block_aligned_config() -> Result<Lfm2VlConfig> {
        let mut config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        config.vision_config.hidden_size = 32;
        config.vision_config.intermediate_size = 64;
        config.vision_config.num_hidden_layers = 2;
        config.vision_config.num_attention_heads = 4;
        config.projector_hidden_size = 64;
        config.text_config.hidden_size = 32;
        config.text_config.intermediate_size = Some(64);
        config.text_config.num_attention_heads = 4;
        config.text_config.num_key_value_heads = 1;
        config.validate()?;
        Ok(config)
    }

    fn synthetic_mmproj_tensors(
        config: &Lfm2VlConfig,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let runtime = Lfm2VlMmprojConfig::from(config);
        let mut tensors = HashMap::new();
        for (tensor_index, tensor_info) in expected_tensors(&runtime)?.into_values().enumerate() {
            let element_count = tensor_info.shape.iter().try_fold(1usize, |count, &dim| {
                count.checked_mul(dim).ok_or_else(|| {
                    candle::Error::Msg("synthetic MMProj tensor size overflowed".into())
                })
            })?;
            let is_norm_weight = tensor_info.native_name.ends_with(".weight")
                && (tensor_info.native_name.contains("layer_norm")
                    || tensor_info.native_name.contains("layernorm"));
            let values = (0..element_count)
                .map(|index| {
                    if is_norm_weight {
                        1.0 + ((index + tensor_index) % 7) as f32 * 1e-3
                    } else {
                        (((index * 17 + tensor_index * 29) % 101) as f32 - 50.0) * 2e-3
                    }
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, tensor_info.shape, device)?;
            tensors.insert(format!("weights.{}", tensor_info.native_name), tensor);
        }
        Ok(tensors)
    }

    fn block_aligned_batch(device: &Device) -> Result<ProcessedVisionBatch> {
        let values = (0..(8 * 12))
            .map(|index| (index as f32 - 47.5) / 64.0)
            .collect::<Vec<_>>();
        Ok(ProcessedVisionBatch {
            pixel_values: Tensor::from_vec(values, (1, 8, 12), device)?,
            pixel_attention_mask: Tensor::ones((1, 8), DType::U32, device)?,
            spatial_shapes: Tensor::new(&[[2u32, 4u32]], device)?,
            crops: vec![CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn metadata_entries(config: &Lfm2VlConfig) -> Result<Vec<(String, Value)>> {
        let vision = &config.vision_config;
        let base_side = (vision.num_patches as f64).sqrt() as usize;
        if base_side * base_side != vision.num_patches {
            candle::bail!("tiny GGUF fixture positions must form a square")
        }
        Ok(vec![
            ("general.architecture".into(), Value::String("clip".into())),
            ("general.type".into(), Value::String("mmproj".into())),
            (
                "general.name".into(),
                Value::String("deterministic-tiny-lfm2-vl-mmproj".into()),
            ),
            ("clip.projector_type".into(), Value::String("lfm2".into())),
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            ("clip.use_gelu".into(), Value::Bool(true)),
            (
                "clip.vision.image_size".into(),
                Value::U32((base_side * vision.patch_size) as u32),
            ),
            (
                "clip.vision.patch_size".into(),
                Value::U32(vision.patch_size as u32),
            ),
            (
                "clip.vision.embedding_length".into(),
                Value::U32(vision.hidden_size as u32),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                Value::U32(vision.intermediate_size as u32),
            ),
            (
                "clip.vision.block_count".into(),
                Value::U32(vision.num_hidden_layers as u32),
            ),
            (
                "clip.vision.attention.head_count".into(),
                Value::U32(vision.num_attention_heads as u32),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                Value::F32(vision.layer_norm_eps as f32),
            ),
            (
                "clip.vision.image_mean".into(),
                Value::Array(vec![Value::F32(0.5); 3]),
            ),
            (
                "clip.vision.image_std".into(),
                Value::Array(vec![Value::F32(0.5); 3]),
            ),
            (
                "clip.vision.projection_dim".into(),
                Value::U32(config.text_config.hidden_size as u32),
            ),
            (
                "clip.vision.projector.scale_factor".into(),
                Value::U32(config.downsample_factor as u32),
            ),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tiny_mmproj_gguf(
        tensors: &HashMap<String, Tensor>,
        config: &Lfm2VlConfig,
        quantize_linears: bool,
        omitted: &[&str],
        metadata_override: Option<(&str, Value)>,
        malformed_patch_rank: bool,
    ) -> Result<Vec<u8>> {
        tiny_mmproj_gguf_with_dtypes(
            tensors,
            config,
            quantize_linears.then_some(GgmlDType::Q8_0),
            None,
            omitted,
            metadata_override,
            malformed_patch_rank,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn tiny_mmproj_gguf_with_dtypes(
        tensors: &HashMap<String, Tensor>,
        config: &Lfm2VlConfig,
        linear_dtype: Option<GgmlDType>,
        forced_dtype: Option<(&str, GgmlDType)>,
        omitted: &[&str],
        metadata_override: Option<(&str, Value)>,
        malformed_patch_rank: bool,
    ) -> Result<Vec<u8>> {
        let runtime = Lfm2VlMmprojConfig::from(config);
        let expected = expected_tensors(&runtime)?;
        let mut qtensors = Vec::new();
        for (gguf_name, tensor_info) in expected {
            if omitted.contains(&gguf_name.as_str()) {
                continue;
            }
            let fixture_name = format!("weights.{}", tensor_info.native_name);
            let fixture_name = if tensors.contains_key(&fixture_name) {
                fixture_name
            } else {
                fixture_name.replace(
                    "weights.model.vision_tower.vision_model.",
                    "weights.model.vision_tower.",
                )
            };
            let mut tensor = fixture_tensor(tensors, &fixture_name)?.clone();
            if tensor_info.patch_layout {
                let vision = &config.vision_config;
                tensor = tensor
                    .reshape((
                        vision.hidden_size,
                        vision.patch_size,
                        vision.patch_size,
                        vision.num_channels,
                    ))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?;
                if malformed_patch_rank {
                    tensor = tensor.reshape((
                        vision.hidden_size,
                        vision.num_channels,
                        vision.patch_size * vision.patch_size,
                    ))?;
                }
            }
            let last_dimension = tensor.dim(tensor.rank() - 1)?;
            let dtype = forced_dtype
                .filter(|(name, _)| *name == gguf_name)
                .map(|(_, dtype)| dtype)
                .or_else(|| {
                    linear_dtype.filter(|dtype| {
                        is_quantized_linear_name(&gguf_name)
                            && last_dimension.is_multiple_of(dtype.block_size())
                    })
                })
                .unwrap_or(GgmlDType::F32);
            qtensors.push((gguf_name, QTensor::quantize(&tensor.contiguous()?, dtype)?));
        }

        let mut metadata = metadata_entries(config)?;
        if let Some((name, value)) = metadata_override {
            let entry = metadata
                .iter_mut()
                .find(|(key, _)| key == name)
                .ok_or_else(|| candle::Error::Msg(format!("unknown test metadata key {name}")))?;
            entry.1 = value;
        }
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

    fn synthetic_text_metadata(config: &Lfm2VlConfig) -> quantized_lfm2::Lfm2GgufMetadata {
        let text = config.text_model_config().expect("fixed tiny text config");
        quantized_lfm2::Lfm2GgufMetadata {
            architecture: "lfm2".into(),
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

    #[test]
    fn dense_gguf_mmproj_matches_native_image_features() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let native = Lfm2VlModel::new(
            &config,
            VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?.pp("weights"),
        )?;
        let batch = fixture_batch(&tensors)?;
        let native_features = native.encode_images(&batch, 1)?;

        let bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        eprintln!(
            "lfm2_vl deterministic dense MMProj GGUF SHA-256: {:x}",
            Sha256::digest(&bytes)
        );
        let mut reader = Cursor::new(bytes);
        let mmproj = Mmproj::from_gguf(&mut reader, DType::F32, &device, 3)?;
        let gguf = mmproj
            .metadata
            .gguf_metadata()
            .ok_or_else(|| candle::Error::Msg("expected direct GGUF MMProj metadata".into()))?;
        assert_eq!(gguf.tensor_count, 43);
        assert_eq!(gguf.quantized_tensor_count, 0);
        assert_eq!(gguf.projector_type, "lfm2");
        assert_eq!(gguf.general_architecture, "clip");
        assert!(gguf.source_byte_count > 0);
        assert!(gguf.dense_byte_count > 0);
        assert!(gguf.estimated_peak_byte_count > gguf.dense_byte_count);
        assert!(mmproj.metadata.use_image_special_tokens);
        assert!(mmproj.metadata.split_manifest().is_none());
        let loaded_features = mmproj.encode_images(&batch, 1)?;
        let error = max_abs(&loaded_features.embeddings, &native_features.embeddings)?;
        eprintln!("lfm2_vl dense GGUF MMProj image features: max_abs={error:.9e}");
        assert!(error <= 1e-6, "dense GGUF image feature error {error}");
        assert_eq!(
            loaded_features.per_crop_ranges,
            native_features.per_crop_ranges
        );
        assert_eq!(
            loaded_features.per_image_ranges,
            native_features.per_image_ranges
        );
        Ok(())
    }

    #[test]
    fn q8_gguf_mmproj_dequantizes_and_pairs() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let dense_bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let q8_bytes = tiny_mmproj_gguf(&tensors, &config, true, &[], None, false)?;
        let mut dense_reader = Cursor::new(dense_bytes);
        let dense = Mmproj::from_gguf(&mut dense_reader, DType::F32, &device, 3)?;
        let mut q8_reader = Cursor::new(q8_bytes.clone());
        let q8 = Mmproj::from_gguf(&mut q8_reader, DType::F32, &device, 3)?;
        let q8_native = Mmproj::from_gguf_q8(&mut Cursor::new(q8_bytes), DType::F32, &device, 3)?;
        assert!(q8
            .metadata
            .gguf_metadata()
            .is_some_and(|metadata| metadata.quantized_tensor_count > 0));
        let batch = fixture_batch(&tensors)?;
        let dense_features = dense.encode_images(&batch, 1)?;
        let q8_features = q8.encode_images(&batch, 1)?;
        let q8_native_features = q8_native.encode_images(&batch, 1)?;
        let error = max_abs(&q8_features.embeddings, &dense_features.embeddings)?;
        eprintln!("lfm2_vl Q8_0 dequantized MMProj image features: max_abs={error:.9e}");
        assert!(error <= 2e-2, "Q8_0 GGUF image feature error {error}");
        let native_error = max_abs(&q8_native_features.embeddings, &dense_features.embeddings)?;
        let native_cosine =
            cosine_similarity(&q8_native_features.embeddings, &dense_features.embeddings)?;
        eprintln!(
            "lfm2_vl Q8_0 native MMProj image features: max_abs={native_error:.9e} cosine={native_cosine:.9}"
        );
        assert!(
            native_error <= 5e-4,
            "Q8_0 native image feature error {native_error}"
        );
        assert!(
            native_cosine >= 0.9999,
            "Q8_0 native image feature cosine {native_cosine}"
        );

        let text = synthetic_text_metadata(&config);
        let report = q8.metadata.validate_pair(&text, 2, 2, 3)?;
        assert_eq!(report.text_hidden_size, 12);
        assert!(q8.metadata.validate_pair(&text, 4, 2, 3).is_err());
        assert!(q8.metadata.validate_pair(&text, 2, 1, 3).is_err());
        assert!(q8.metadata.validate_pair(&text, 2, 2, 4).is_err());
        let mut wrong_text = text;
        wrong_text.embedding_length += 1;
        assert!(q8.metadata.validate_pair(&wrong_text, 2, 2, 3).is_err());
        Ok(())
    }

    #[test]
    fn native_q8_gguf_executes_all_vision_and_projector_linears() -> Result<()> {
        let device = Device::Cpu;
        let config = block_aligned_config()?;
        let tensors = synthetic_mmproj_tensors(&config, &device)?;
        let batch = block_aligned_batch(&device)?;
        let dense_bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let q8_bytes = tiny_mmproj_gguf(&tensors, &config, true, &[], None, false)?;
        eprintln!(
            "lfm2_vl block-aligned Q8_0 MMProj GGUF SHA-256: {:x}",
            Sha256::digest(&q8_bytes)
        );

        let dense = Mmproj::from_gguf(
            &mut Cursor::new(dense_bytes.clone()),
            DType::F32,
            &device,
            3,
        )?;
        let q8_dequantized =
            Mmproj::from_gguf(&mut Cursor::new(q8_bytes.clone()), DType::F32, &device, 3)?;
        let q8_native =
            Mmproj::from_gguf_q8(&mut Cursor::new(q8_bytes.clone()), DType::F32, &device, 3)?;
        let q8_auto = Mmproj::from_gguf_auto(&mut Cursor::new(q8_bytes), DType::F32, &device, 3)?;

        assert_eq!(
            dense.gguf_execution(),
            Some(GgufMmprojExecution::DenseCompatibility)
        );
        assert_eq!(dense.native_quantized_tensor_count(), 0);
        let native_metadata = q8_native.metadata.gguf_metadata().unwrap();
        assert_eq!(native_metadata.quantized_tensor_count, 14);
        assert_eq!(q8_native.gguf_execution(), Some(GgufMmprojExecution::Q8_0));
        assert_eq!(q8_native.native_quantized_tensor_count(), 14);
        assert_eq!(q8_auto.gguf_execution(), Some(GgufMmprojExecution::Q8_0));

        let dense_features = dense.encode_images(&batch, 1)?;
        let dequantized_features = q8_dequantized.encode_images(&batch, 1)?;
        let native_features = q8_native.encode_images(&batch, 1)?;
        let auto_features = q8_auto.encode_images(&batch, 1)?;
        let operator_error = max_abs(
            &native_features.embeddings,
            &dequantized_features.embeddings,
        )?;
        let quantization_error = max_abs(&native_features.embeddings, &dense_features.embeddings)?;
        let cosine = cosine_similarity(&native_features.embeddings, &dense_features.embeddings)?;
        eprintln!(
            "lfm2_vl native Q8_0 MMProj image features: operator_max_abs={operator_error:.9e} dense_max_abs={quantization_error:.9e} cosine={cosine:.9}"
        );
        assert!(
            operator_error <= 5e-3,
            "native/dequantized Q8 operator error {operator_error}"
        );
        assert!(
            quantization_error <= 1e-2,
            "native/dense Q8 image feature error {quantization_error}"
        );
        assert!(cosine >= 0.9999, "native/dense Q8 cosine {cosine}");
        assert!(max_abs(&auto_features.embeddings, &native_features.embeddings)? <= 1e-6);

        let error = Mmproj::from_gguf_q8(&mut Cursor::new(dense_bytes), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires at least one Q8_0"));
        let error = Mmproj::from_gguf_q8(
            &mut Cursor::new(tiny_mmproj_gguf(&tensors, &config, true, &[], None, false)?),
            DType::BF16,
            &device,
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires F32 activations"));

        let q4_bytes = tiny_mmproj_gguf_with_dtypes(
            &tensors,
            &config,
            Some(GgmlDType::Q4_0),
            None,
            &[],
            None,
            false,
        )?;
        let error = Mmproj::from_gguf_q8(&mut Cursor::new(q4_bytes), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not support Q4_0 tensor"));

        for dense_role in ["v.blk.0.ln1.weight", "v.position_embd.weight"] {
            let invalid = tiny_mmproj_gguf_with_dtypes(
                &tensors,
                &config,
                Some(GgmlDType::Q8_0),
                Some((dense_role, GgmlDType::Q8_0)),
                &[],
                None,
                false,
            )?;
            let auto_error =
                Mmproj::from_gguf_auto(&mut Cursor::new(invalid.clone()), DType::F32, &device, 3)
                    .unwrap_err()
                    .to_string();
            assert!(
                auto_error.contains(dense_role) && auto_error.contains("role must remain dense"),
                "unexpected automatic Q8 dense-role error: {auto_error}"
            );
            let error = Mmproj::from_gguf_q8(&mut Cursor::new(invalid), DType::F32, &device, 3)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(dense_role) && error.contains("role must remain dense"),
                "unexpected native Q8 dense-role error: {error}"
            );
        }

        let dense_bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let mut dense_content = gguf_file::Content::read(&mut Cursor::new(dense_bytes))?;
        dense_content
            .tensor_infos
            .get_mut("v.patch_embd.weight")
            .unwrap()
            .ggml_dtype = GgmlDType::Q8_0;
        let runtime = Lfm2VlMmprojConfig::from(&config);
        let error =
            validate_native_q8_tensors(&dense_content, &expected_tensors(&runtime)?, DType::F32)
                .unwrap_err()
                .to_string();
        assert!(error.contains("v.patch_embd.weight") && error.contains("role must remain dense"));

        let tiny_config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tiny_tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let tiny_bytes = tiny_mmproj_gguf(&tiny_tensors, &tiny_config, false, &[], None, false)?;
        let mut unaligned_content = gguf_file::Content::read(&mut Cursor::new(tiny_bytes))?;
        unaligned_content
            .tensor_infos
            .get_mut("v.blk.0.attn_q.weight")
            .unwrap()
            .ggml_dtype = GgmlDType::Q8_0;
        let tiny_runtime = Lfm2VlMmprojConfig::from(&tiny_config);
        let error = validate_native_q8_tensors(
            &unaligned_content,
            &expected_tensors(&tiny_runtime)?,
            DType::F32,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("input width 16") && error.contains("block size 32"));
        Ok(())
    }

    #[test]
    fn gguf_mmproj_rejects_malformed_metadata_inventory_layout_and_payload() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;

        let wrong_projector = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &[],
            Some(("clip.projector_type", Value::String("mlp".into()))),
            false,
        )?;
        let error = Mmproj::from_gguf(&mut Cursor::new(wrong_projector), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("projector_type") && error.contains("lfm2"));

        let mut missing_general_type =
            tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let key = b"general.type";
        let key_offset = missing_general_type
            .windows(key.len())
            .position(|window| window == key)
            .ok_or_else(|| candle::Error::Msg("missing general.type test key".into()))?;
        missing_general_type[key_offset] = b'x';
        let error = Mmproj::from_gguf(
            &mut Cursor::new(missing_general_type),
            DType::F32,
            &device,
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing metadata key") && error.contains("general.type"));

        let wrong_type = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &[],
            Some(("clip.vision.patch_size", Value::String("two".into()))),
            false,
        )?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(wrong_type), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("positive integer")
        );

        let missing = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["v.blk.0.attn_q.bias"],
            None,
            false,
        )?;
        let error = Mmproj::from_gguf(&mut Cursor::new(missing), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing") && error.contains("v.blk.0.attn_q.bias"));

        let mut unexpected = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let original_name = b"v.post_ln.bias";
        let name_offset = unexpected
            .windows(original_name.len())
            .position(|window| window == original_name)
            .ok_or_else(|| candle::Error::Msg("missing fixed GGUF test name".into()))?;
        unexpected[name_offset] = b'x';
        let error = Mmproj::from_gguf(&mut Cursor::new(unexpected), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing") && error.contains("unexpected"));

        let wrong_patch = tiny_mmproj_gguf(&tensors, &config, false, &[], None, true)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(wrong_patch), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("patch_embd.weight")
        );

        let mut truncated = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        truncated.truncate(truncated.len() - 1);
        let error = Mmproj::from_gguf(&mut Cursor::new(truncated), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("beyond file size"));

        let mut oversized_counts = Vec::new();
        oversized_counts.extend_from_slice(b"GGUF");
        oversized_counts.extend_from_slice(&3u32.to_le_bytes());
        oversized_counts.extend_from_slice(&1u64.to_le_bytes());
        oversized_counts.extend_from_slice(&(MAX_GGUF_MMPROJ_METADATA + 1).to_le_bytes());
        let error = Mmproj::from_gguf(&mut Cursor::new(oversized_counts), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("metadata_kv_count") && error.contains("exceeds max"));

        let mut oversized_string = Vec::new();
        oversized_string.extend_from_slice(b"GGUF");
        oversized_string.extend_from_slice(&3u32.to_le_bytes());
        oversized_string.extend_from_slice(&0u64.to_le_bytes());
        oversized_string.extend_from_slice(&1u64.to_le_bytes());
        oversized_string.extend_from_slice(&(MAX_GGUF_MMPROJ_STRING_BYTES + 1).to_le_bytes());
        oversized_string.extend_from_slice(&[0; 9]);
        let error = Mmproj::from_gguf(&mut Cursor::new(oversized_string), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("string length") && error.contains("exceeds max"));

        let valid = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(valid), DType::U8, &device, 3)
                .unwrap_err()
                .to_string()
                .contains("F32, F16, or BF16")
        );
        Ok(())
    }

    #[test]
    fn gguf_mmproj_range_validation_rejects_misalignment_overlap_and_overflow() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let runtime = Lfm2VlMmprojConfig::from(&config);
        let expected = expected_tensors(&runtime)?;
        let file_size = bytes.len() as u64;

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = 1;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not aligned"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.weight")
            .unwrap()
            .offset = 0;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = 0;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlaps another tensor"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = u64::MAX - 31;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("offset overflowed"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes))?;
        content.tensor_data_offset = file_size + 32;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("beyond file size"));
        Ok(())
    }

    #[test]
    fn gguf_mmproj_optional_projector_layer_norm_is_not_synthesized() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let bytes = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.input_norm.weight", "mm.input_norm.bias"],
            None,
            false,
        )?;
        let mmproj = Mmproj::from_gguf(&mut Cursor::new(bytes), DType::F32, &device, 3)?;
        assert_eq!(mmproj.report.loaded_tensors.len(), 41);

        let incomplete = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.input_norm.bias"],
            None,
            false,
        )?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(incomplete), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("appear together")
        );

        let no_biases = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.1.bias", "mm.2.bias"],
            None,
            false,
        )?;
        let no_biases = Mmproj::from_gguf(&mut Cursor::new(no_biases), DType::F32, &device, 3)?;
        assert_eq!(no_biases.report.loaded_tensors.len(), 41);

        let incomplete_biases =
            tiny_mmproj_gguf(&tensors, &config, false, &["mm.2.bias"], None, false)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(incomplete_biases), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("biases must appear together")
        );
        Ok(())
    }
}
