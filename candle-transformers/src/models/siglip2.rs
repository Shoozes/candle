//! SigLIP2 NaFlex vision encoding for already-patchified inputs.
//!
//! This module intentionally stops at the padded vision hidden states. Image
//! resizing, tiling, normalization, patchification, and the LFM2.5-VL
//! projector are separate phases.

#![allow(clippy::manual_is_multiple_of, clippy::needless_range_loop)]

use crate::models::lfm2_vl::linear::LinearOp;
use candle::quantized::QTensor;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use std::collections::HashMap;
use std::sync::RwLock;

include!("siglip2/config.rs");
include!("siglip2/embeddings.rs");
include!("siglip2/encoder.rs");
include!("siglip2/model.rs");
include!("siglip2/interpolation.rs");

/// Stable two-pass LayerNorm for SigLIP2 CPU-F32 parity.
fn stable_layer_norm(layer: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    let bias = layer
        .bias()
        .ok_or_else(|| candle::Error::Msg("SigLIP2 LayerNorm requires an affine bias".into()))?;
    candle_nn::ops::layer_norm_slow(xs, layer.weight(), bias, layer.eps() as f32)
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config() -> Siglip2VisionConfig {
        Siglip2VisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 2,
            num_patches: 16,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            vision_use_head: false,
        }
    }

    #[test]
    fn serde_config_parses_and_validates_dynamic_fields() -> Result<()> {
        let config = Siglip2VisionConfig::from_json(
            r#"{
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "patch_size": 2,
                "num_patches": 16,
                "hidden_act": "gelu_pytorch_tanh",
                "layer_norm_eps": 0.000001,
                "attention_dropout": 0.0,
                "vision_use_head": false
            }"#,
        )?;
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.intermediate_size, 32);
        assert_eq!(config.hidden_act, Activation::GeluPytorchTanh);
        assert_eq!(config.base_grid_side()?, 4);
        assert!(
            Siglip2VisionConfig::from_json(r#"{"hidden_size": 16, "num_patches": 15}"#).is_err()
        );
        assert!(Siglip2VisionConfig::from_json(
            r#"{"hidden_size": 16, "num_patches": 16, "vision_use_head": true}"#
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn stable_layer_norm_handles_large_offsets() -> Result<()> {
        let device = Device::Cpu;
        let layer = LayerNorm::new(
            Tensor::ones(4, DType::F32, &device)?,
            Tensor::zeros(4, DType::F32, &device)?,
            1e-6,
        );
        let input = Tensor::new(&[[[10_000.0f32, 10_001.0, 9_999.0, 10_000.5]]], &device)?;
        let actual = stable_layer_norm(&layer, &input)?.to_vec3::<f32>()?;
        let values = [10_000.0f64, 10_001.0, 9_999.0, 10_000.5];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let scale = (variance + 1e-6).sqrt();
        for (actual, expected) in actual[0][0].iter().zip(values) {
            let expected = ((expected - mean) / scale) as f32;
            assert!((actual - expected).abs() <= 1e-3);
        }
        Ok(())
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
        eprintln!("{label}: max_abs={max_abs:.9e}, cosine={cosine:.9}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        assert!(
            cosine.is_finite() && cosine >= 0.99999,
            "{label}: cosine={cosine} < 0.99999"
        );
        Ok(())
    }

    fn tiny_model() -> Result<(Siglip2VisionModel, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?
            .pp("weights")
            .pp("model")
            .pp("vision_tower");
        let model = Siglip2VisionModel::new(&tiny_config(), weights)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        Ok((model, tensors))
    }

    #[test]
    fn tiny_fixture_matches_all_vision_stages() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let pixel_values = fixture_tensor(&tensors, "input.pixel_values")?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let inputs = PackedVisionInputs {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        };
        let stages = model.forward_stages(&inputs)?;
        assert_close(
            &stages.embeddings.patch_embedding,
            fixture_tensor(&tensors, "stage.vision.patch_embedding")?,
            2e-5,
            "patch projection",
        )?;
        assert_close(
            &stages.embeddings.resized_position_embedding,
            fixture_tensor(&tensors, "stage.vision.resized_position_embedding")?,
            2e-5,
            "resized positions",
        )?;
        assert_close(
            &stages.embeddings.embeddings_with_position,
            fixture_tensor(&tensors, "stage.vision.embeddings_with_resized_position")?,
            2e-5,
            "embedding plus positions",
        )?;
        for (index, actual) in stages.encoder_layers.iter().enumerate() {
            assert_close(
                actual,
                fixture_tensor(&tensors, &format!("stage.vision.encoder_layer.{index}"))?,
                2e-5,
                &format!("encoder layer {index}"),
            )?;
        }
        assert_close(
            &stages.post_layernorm,
            fixture_tensor(&tensors, "stage.vision.last_hidden_state")?,
            2e-5,
            "returned post layer norm",
        )?;
        assert_close(
            &stages.post_layernorm,
            fixture_tensor(&tensors, "stage.vision.post_layernorm")?,
            2e-5,
            "post layer norm",
        )?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_repeat_is_deterministic() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let inputs = PackedVisionInputs {
            pixel_values: fixture_tensor(&tensors, "input.pixel_values")?,
            pixel_attention_mask: fixture_tensor(&tensors, "input.pixel_attention_mask")?,
            spatial_shapes: fixture_tensor(&tensors, "input.spatial_shapes")?,
        };
        let first = model.forward(&inputs)?;
        let second = model.forward(&inputs)?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn padding_keys_cannot_change_valid_patch_outputs() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let pixel_values = fixture_tensor(&tensors, "input.pixel_values")?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let inputs = PackedVisionInputs {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        };
        let baseline = model.forward(&inputs)?;
        let mut altered = pixel_values.to_vec3::<f32>()?;
        for value in altered[0][8..].iter_mut().flatten() {
            *value = 123.0;
        }
        let altered = Tensor::from_vec(
            altered.into_iter().flatten().flatten().collect(),
            (1, 10, 12),
            &Device::Cpu,
        )?;
        let altered_output = model.forward(&PackedVisionInputs {
            pixel_values: &altered,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        })?;
        assert_close(
            &baseline.i((.., 0..8, ..))?,
            &altered_output.i((.., 0..8, ..))?,
            1e-6,
            "padding key isolation",
        )?;
        Ok(())
    }

    #[test]
    fn repeated_crop_batch_matches_single_crop() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let pixel_values = fixture_tensor(&tensors, "input.pixel_values")?;
        let pixel_attention_mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let spatial_shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let single = model.forward(&PackedVisionInputs {
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
        })?;
        let pixel_values = Tensor::cat(&[pixel_values, pixel_values], 0)?;
        let pixel_attention_mask = Tensor::cat(&[pixel_attention_mask, pixel_attention_mask], 0)?;
        let spatial_shapes = Tensor::cat(&[spatial_shapes, spatial_shapes], 0)?;
        let repeated = model.forward(&PackedVisionInputs {
            pixel_values: &pixel_values,
            pixel_attention_mask: &pixel_attention_mask,
            spatial_shapes: &spatial_shapes,
        })?;
        assert_close(
            &repeated.i(0)?,
            &single.i(0)?,
            1e-6,
            "repeated crop batch first output",
        )?;
        assert_close(
            &repeated.i(1)?,
            &single.i(0)?,
            1e-6,
            "repeated crop batch second output",
        )?;
        Ok(())
    }

    #[test]
    fn resize_weights_match_pinned_four_to_two_and_four_to_six_oracle() -> Result<()> {
        let expected_down = [
            [3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0, 0.0],
            [0.0, 1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0],
        ];
        for (index, expected) in expected_down.into_iter().enumerate() {
            let weights = resize_weights(4, 2, index)?;
            let mut actual = [0f32; 4];
            for offset in 0..weights.indices.len() {
                actual[weights.indices[offset]] = weights.weights[offset];
            }
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
        let expected_up = [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 5.0 / 6.0, 1.0 / 6.0, 0.0],
            [0.0, 1.0 / 6.0, 5.0 / 6.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for (index, expected) in expected_up.into_iter().enumerate() {
            let weights = resize_weights(4, 6, index)?;
            let mut actual = [0f32; 4];
            for offset in 0..weights.indices.len() {
                actual[weights.indices[offset]] = weights.weights[offset];
            }
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn resize_composes_pinned_wide_and_tall_shapes() -> Result<()> {
        let width_only: Vec<Vec<f32>> = (0..16).map(|index| vec![(index % 4) as f32]).collect();
        let wide = resize_bilinear_antialias(&width_only, 4, 4, 2, 6, 1)?;
        let expected_wide = [0.0, 0.5, 7.0 / 6.0, 11.0 / 6.0, 2.5, 3.0];
        for row in 0..2 {
            for (column, expected) in expected_wide.into_iter().enumerate() {
                assert!((wide[row * 6 + column] - expected).abs() <= 1e-6);
            }
        }

        let height_only: Vec<Vec<f32>> = (0..16).map(|index| vec![(index / 4) as f32]).collect();
        let tall = resize_bilinear_antialias(&height_only, 4, 4, 6, 2, 1)?;
        let expected_tall = [0.0, 0.5, 7.0 / 6.0, 11.0 / 6.0, 2.5, 3.0];
        for (row, expected) in expected_tall.into_iter().enumerate() {
            for column in 0..2 {
                assert!((tall[row * 2 + column] - expected).abs() <= 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_malformed_packed_inputs_and_config() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let wrong_pixels = Tensor::zeros((1, 10, 11), DType::F32, &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: &wrong_pixels,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        });
        assert!(error.is_err());

        let bad_mask = Tensor::ones((1, 10), DType::F32, &Device::Cpu)?;
        let pixels = fixture_tensor(&tensors, "input.pixel_values")?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: &bad_mask,
            spatial_shapes: shapes,
        });
        assert!(error.is_err());

        let bad_shapes = Tensor::new(&[[3i64, 4i64]], &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: mask,
            spatial_shapes: &bad_shapes,
        });
        assert!(error.is_err());

        let overflowing_shapes = Tensor::new(&[[i64::MAX, 2i64]], &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: mask,
            spatial_shapes: &overflowing_shapes,
        });
        assert!(error.is_err());

        let mut nonsquare = tiny_config();
        nonsquare.num_patches = 15;
        assert!(nonsquare.validate().is_err());
        let mut head = tiny_config();
        head.vision_use_head = true;
        assert!(head.validate().is_err());
        Ok(())
    }
}
