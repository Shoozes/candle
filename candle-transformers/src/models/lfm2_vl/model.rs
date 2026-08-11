use crate::models::lfm2_vl::config::{Lfm2VlConfig, VisionLimits};
use crate::models::lfm2_vl::projector::Lfm2VlProjector;
use crate::models::{lfm2, siglip2};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use std::ops::Range;

include!("model/types.rs");
include!("model/runtime.rs");
include!("model/encoding.rs");
include!("model/merge.rs");
include!("model/config_ext.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config() -> Lfm2VlConfig {
        Lfm2VlConfig::from_json(
            &serde_json::json!({
                "model_type": "lfm2_vl",
                "image_token_id": 3,
                "downsample_factor": 2,
                "projector_hidden_size": 24,
                "projector_hidden_act": "gelu",
                "projector_bias": true,
                "projector_use_layernorm": true,
                "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 12,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 3,
                    "num_key_value_heads": 1,
                    "intermediate_size": 32,
                    "block_auto_adjust_ff_dim": false,
                    "layer_types": ["conv", "full_attention"],
                    "rope_theta": 10000.0
                },
                "vision_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_channels": 3,
                    "patch_size": 2,
                    "num_patches": 16,
                    "hidden_act": "gelu_pytorch_tanh",
                    "vision_use_head": false
                }
            })
            .to_string(),
        )
        .expect("fixed LFM2-VL config")
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
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
        let mut max_abs = 0f32;
        let mut dot = 0f32;
        let mut actual_norm = 0f32;
        let mut expected_norm = 0f32;
        for (&lhs, &rhs) in actual.iter().zip(&expected) {
            max_abs = max_abs.max((lhs - rhs).abs());
            dot += lhs * rhs;
            actual_norm += lhs * lhs;
            expected_norm += rhs * rhs;
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        eprintln!("lfm2_vl composite {label}: max_abs={max_abs:.9e}, cosine={cosine:.9}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        assert!(
            cosine.is_finite() && cosine >= 0.99999,
            "{label}: cosine={cosine}"
        );
        Ok(())
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

    fn tiny_model(device: &Device) -> Result<(Lfm2VlModel, HashMap<String, Tensor>)> {
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, device)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, device)?;
        let model = Lfm2VlModel::new(&tiny_config(), weights.pp("weights"))?;
        Ok((model, tensors))
    }

    fn image_spans(input_ids: &Tensor, image_token_id: u32) -> Result<Vec<ImageTokenSpan>> {
        let values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut spans = Vec::new();
        for (batch_index, row) in values.iter().enumerate() {
            let mut start = None;
            for (position, &token_id) in row.iter().enumerate() {
                if token_id == image_token_id && start.is_none() {
                    start = Some(position);
                }
                if token_id != image_token_id {
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

    #[test]
    fn fixture_matches_projector_merged_embeddings_and_prefill_decode() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let encoded = model.encode_images(&batch, 1)?;
        assert_close(
            &encoded.embeddings,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "encoded projector output",
        )?;
        assert_eq!(encoded.per_crop_ranges, vec![0..2]);
        assert_eq!(encoded.per_image_ranges, vec![0..2]);

        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let input_embeds = model.embed_tokens(input_ids)?;
        let spans = image_spans(input_ids, 3)?;
        let merged = model.merge_image_embeddings(input_ids, &input_embeds, &spans, &encoded)?;
        assert_close(
            &merged,
            fixture_tensor(&tensors, "stage.multimodal.merged_embeddings")?,
            1e-6,
            "merged embeddings",
        )?;

        let text_config = tiny_config().text_model_config()?;
        let mut cache = lfm2::Cache::new(true, DType::F32, &text_config, &device)?;
        let prefill = model.prefill(input_ids, &spans, Some(&encoded), &mut cache)?;
        assert_close(
            &prefill,
            fixture_tensor(&tensors, "stage.language.prefill_logits")?,
            1e-3,
            "multimodal prefill logits",
        )?;
        let decode_ids = fixture_tensor(&tensors, "input.decode_token_ids")?;
        let decode_expected = fixture_tensor(&tensors, "stage.language.decode_logits")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let logits = model.decode(&token, 5 + step, &mut cache)?;
            assert_close(
                &logits,
                &decode_expected.i((.., step, ..))?,
                1e-3,
                &format!("cached decode step {step}"),
            )?;
        }
        cache.clear();
        let reset = model.prefill(input_ids, &spans, Some(&encoded), &mut cache)?;
        assert_close(
            &reset,
            fixture_tensor(&tensors, "stage.language.prefill_logits")?,
            1e-3,
            "cache-reset prefill logits",
        )?;
        let reset_decode = model.decode(&decode_ids.i((.., 0..1))?, 5, &mut cache)?;
        assert_close(
            &reset_decode,
            &decode_expected.i((.., 0, ..))?,
            1e-3,
            "cache-reset decode logits",
        )?;
        Ok(())
    }

    #[test]
    fn packed_model_boundary_enforces_exact_and_one_over_limits() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let (_, max_patches, _) = batch.pixel_values.dims3()?;
        let source_pixels = batch.images[0]
            .resized_width
            .checked_mul(batch.images[0].resized_height)
            .ok_or_else(|| candle::Error::Msg("test image surface overflow".into()))?;
        let exact = VisionLimits {
            max_source_pixels: source_pixels,
            max_images: 1,
            max_crops_per_image: 1,
            max_total_crops: 1,
            max_patches_per_crop: max_patches,
            max_total_projected_tokens: 2,
        };
        let encoded = model.encode_images_with_limits(&batch, 1, &exact)?;
        assert_eq!(encoded.embeddings.dims(), [2, 12]);

        let token_over = VisionLimits {
            max_total_projected_tokens: 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &token_over)
            .expect_err("packed boundary must reject projected-token overage");
        assert!(error.to_string().contains("2 projected tokens"));

        let patch_over = VisionLimits {
            max_patches_per_crop: max_patches - 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &patch_over)
            .expect_err("packed boundary must reject padded patch overage");
        assert!(error.to_string().contains("patch slots per crop"));

        let surface_over = VisionLimits {
            max_source_pixels: exact.max_source_pixels - 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &surface_over)
            .expect_err("packed boundary must reject resized-surface overage");
        assert!(error.to_string().contains("resized image metadata"));

        let error = model
            .encode_images_with_limits(&batch, 0, &exact)
            .expect_err("packed boundary must reject a zero batch size before execution");
        assert!(error.to_string().contains("vision_batch_size"));
        Ok(())
    }

    #[test]
    fn multiple_crops_and_images_preserve_order_and_ranges() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let single = fixture_batch(&tensors)?;
        let pixel_values = Tensor::cat(
            &[
                &single.pixel_values,
                &single.pixel_values,
                &single.pixel_values,
            ],
            0,
        )?;
        let mask = Tensor::cat(
            &[
                &single.pixel_attention_mask,
                &single.pixel_attention_mask,
                &single.pixel_attention_mask,
            ],
            0,
        )?;
        let shapes = Tensor::cat(
            &[
                &single.spatial_shapes,
                &single.spatial_shapes,
                &single.spatial_shapes,
            ],
            0,
        )?;
        let batch = ProcessedVisionBatch {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
            crops: vec![
                CropMeta {
                    image_index: 0,
                    crop_index: 0,
                    kind: CropKind::Tile { row: 0, col: 0 },
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
                CropMeta {
                    image_index: 0,
                    crop_index: 1,
                    kind: CropKind::Thumbnail,
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
                CropMeta {
                    image_index: 1,
                    crop_index: 0,
                    kind: CropKind::Whole,
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
            ],
            images: vec![
                ImageMeta {
                    crop_range: 0..2,
                    rows: 2,
                    cols: 4,
                    resized_width: 8,
                    resized_height: 2,
                },
                ImageMeta {
                    crop_range: 2..3,
                    rows: 2,
                    cols: 4,
                    resized_width: 4,
                    resized_height: 2,
                },
            ],
        };
        let encoded = model.encode_images(&batch, 2)?;
        assert_eq!(encoded.embeddings.dims(), [6, 12]);
        assert_eq!(encoded.per_crop_ranges, vec![0..2, 2..4, 4..6]);
        assert_eq!(encoded.per_image_ranges, vec![0..4, 4..6]);
        assert_close(
            &encoded.embeddings.narrow(0, 0, 2)?,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "first crop order",
        )?;
        assert_close(
            &encoded.embeddings.narrow(0, 4, 2)?,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "second image order",
        )?;
        Ok(())
    }

    #[test]
    fn rejects_malformed_packed_shapes_and_masks() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let single = fixture_batch(&tensors)?;
        let bad_mask = ProcessedVisionBatch {
            pixel_values: single.pixel_values.clone(),
            pixel_attention_mask: single.pixel_attention_mask.narrow(1, 0, 9)?,
            spatial_shapes: single.spatial_shapes.clone(),
            crops: single.crops.clone(),
            images: single.images.clone(),
        };
        assert!(model.encode_images(&bad_mask, 1).is_err());

        let bad_spatial = ProcessedVisionBatch {
            pixel_values: single.pixel_values.clone(),
            pixel_attention_mask: single.pixel_attention_mask.clone(),
            spatial_shapes: Tensor::new(&[[-1i64, 4]], &device)?,
            crops: single.crops.clone(),
            images: single.images.clone(),
        };
        assert!(model.encode_images(&bad_spatial, 1).is_err());

        let nonbinary_mask = ProcessedVisionBatch {
            pixel_values: single.pixel_values,
            pixel_attention_mask: Tensor::ones((1, 10), DType::F32, &device)?,
            spatial_shapes: Tensor::new(&[[2i64, 4]], &device)?,
            crops: single.crops,
            images: single.images,
        };
        assert!(model.encode_images(&nonbinary_mask, 1).is_err());
        Ok(())
    }

    #[test]
    fn span_validation_rejects_bad_ranges_tokens_and_counts() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let encoded = model.encode_images(&batch, 1)?;
        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let embeds = model.embed_tokens(input_ids)?;
        let spans = image_spans(input_ids, 3)?;
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[ImageTokenSpan::new(0, spans[0].start, spans[0].end + 1)],
                &encoded,
            )
            .is_err());
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[
                    ImageTokenSpan::new(0, spans[0].start, spans[0].start + 1),
                    ImageTokenSpan::new(0, spans[0].start, spans[0].start + 1),
                ],
                &EncodedImages {
                    embeddings: encoded.embeddings.clone(),
                    per_image_ranges: vec![0..1, 1..2],
                    per_crop_ranges: vec![0..1, 1..2],
                },
            )
            .is_err());
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[ImageTokenSpan::new(0, 0, input_ids.dim(1)? + 1)],
                &encoded,
            )
            .is_err());
        let wrong_token = ImageTokenSpan::new(0, 0, 1);
        assert!(model
            .merge_image_embeddings(input_ids, &embeds, &[wrong_token], &encoded)
            .is_err());
        let short = EncodedImages {
            embeddings: encoded.embeddings.narrow(0, 0, 1)?,
            per_image_ranges: std::iter::once(0..1).collect(),
            per_crop_ranges: std::iter::once(0..1).collect(),
        };
        assert!(model
            .merge_image_embeddings(input_ids, &embeds, &spans, &short)
            .is_err());
        Ok(())
    }

    #[test]
    fn multiple_spans_pair_with_exact_per_crop_feature_ranges() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let one_image = model.encode_images(&batch, 1)?;
        let three_crop_embeddings = Tensor::cat(
            &[
                &one_image.embeddings,
                &one_image.embeddings,
                &one_image.embeddings,
            ],
            0,
        )?;
        let encoded = EncodedImages {
            embeddings: three_crop_embeddings,
            per_image_ranges: vec![0..4, 4..6],
            per_crop_ranges: vec![0..2, 2..4, 4..6],
        };
        let input_ids = Tensor::new(&[[3i64, 3, 5, 3, 3, 5, 3, 3]], &device)?;
        let input_embeds = model.embed_tokens(&input_ids)?;
        let spans = [
            ImageTokenSpan::new(0, 0, 2),
            ImageTokenSpan::new(0, 3, 5),
            ImageTokenSpan::new(0, 6, 8),
        ];
        let merged = model.merge_image_embeddings(&input_ids, &input_embeds, &spans, &encoded)?;
        assert_close(
            &merged.i((.., 0..2, ..))?,
            &encoded.embeddings.i((0..2, ..))?.unsqueeze(0)?,
            0.0,
            "first crop span insertion",
        )?;
        assert_close(
            &merged.i((.., 3..5, ..))?,
            &encoded.embeddings.i((2..4, ..))?.unsqueeze(0)?,
            0.0,
            "second crop span insertion",
        )?;
        assert_close(
            &merged.i((.., 6..8, ..))?,
            &encoded.embeddings.i((4..6, ..))?.unsqueeze(0)?,
            0.0,
            "third crop span insertion",
        )?;
        assert_close(
            &merged.i((0, 2, ..))?,
            &input_embeds.i((0, 2, ..))?,
            0.0,
            "untouched tile marker embedding",
        )?;
        assert_close(
            &merged.i((0, 5, ..))?,
            &input_embeds.i((0, 5, ..))?,
            0.0,
            "untouched thumbnail marker embedding",
        )?;

        let mismatched_crop_lengths = EncodedImages {
            embeddings: encoded.embeddings.clone(),
            per_image_ranges: vec![0..2, 2..6],
            per_crop_ranges: vec![0..2, 2..5, 5..6],
        };
        assert!(model
            .merge_image_embeddings(&input_ids, &input_embeds, &spans, &mismatched_crop_lengths)
            .is_err());
        let image_range_splits_crop = EncodedImages {
            embeddings: encoded.embeddings.clone(),
            per_image_ranges: vec![0..3, 3..6],
            per_crop_ranges: vec![0..2, 2..4, 4..6],
        };
        assert!(model
            .merge_image_embeddings(&input_ids, &input_embeds, &spans, &image_range_splits_crop)
            .is_err());
        Ok(())
    }
}
