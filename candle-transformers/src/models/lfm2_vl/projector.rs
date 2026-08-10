use crate::models::lfm2_vl::config::{Lfm2VlConfig, Lfm2VlMmprojConfig};
use crate::models::lfm2_vl::linear::LinearOp;
use candle::quantized::QTensor;
use candle::{Module, Result, Tensor};
use candle_nn::{
    layer_norm, linear, linear_no_bias, Activation, LayerNorm, LayerNormConfig, VarBuilder,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Lfm2VlProjector {
    factor: usize,
    input_size: usize,
    layer_norm: Option<LayerNorm>,
    linear_1: LinearOp,
    activation: Activation,
    linear_2: LinearOp,
    output_size: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct ProjectorStages {
    pixel_unshuffle: Tensor,
    layer_norm: Option<Tensor>,
    linear_1: Tensor,
    activation: Tensor,
    linear_2: Tensor,
    output: Tensor,
}

impl Lfm2VlProjector {
    pub fn new(config: &Lfm2VlConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        Self::from_mmproj_config(&Lfm2VlMmprojConfig::from(config), vb)
    }

    pub fn from_mmproj_config(config: &Lfm2VlMmprojConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let input_size = config.projector_input_size()?;
        let layer_norm = if config.projector_use_layernorm {
            Some(layer_norm(
                input_size,
                LayerNormConfig {
                    eps: 1e-5,
                    ..LayerNormConfig::default()
                },
                vb.pp("layer_norm"),
            )?)
        } else {
            None
        };
        let linear_1 = LinearOp::Dense(if config.projector_bias {
            linear(input_size, config.projector_hidden_size, vb.pp("linear_1"))?
        } else {
            linear_no_bias(input_size, config.projector_hidden_size, vb.pp("linear_1"))?
        });
        let linear_2 = LinearOp::Dense(if config.projector_bias {
            linear(
                config.projector_hidden_size,
                config.text_hidden_size,
                vb.pp("linear_2"),
            )?
        } else {
            linear_no_bias(
                config.projector_hidden_size,
                config.text_hidden_size,
                vb.pp("linear_2"),
            )?
        });
        Ok(Self {
            factor: config.downsample_factor,
            input_size,
            layer_norm,
            linear_1,
            activation: config.projector_hidden_act,
            linear_2,
            output_size: config.text_hidden_size,
        })
    }

    pub(crate) fn from_mmproj_config_with_quantized_linears(
        config: &Lfm2VlMmprojConfig,
        vb: VarBuilder,
        mut quantized_weights: HashMap<String, QTensor>,
    ) -> Result<Self> {
        config.validate()?;
        let input_size = config.projector_input_size()?;
        let layer_norm = if config.projector_use_layernorm {
            Some(layer_norm(
                input_size,
                LayerNormConfig {
                    eps: 1e-5,
                    ..LayerNormConfig::default()
                },
                vb.pp("layer_norm"),
            )?)
        } else {
            None
        };
        let linear_1 = mixed_linear(
            input_size,
            config.projector_hidden_size,
            config.projector_bias,
            vb.pp("linear_1"),
            &mut quantized_weights,
            "linear_1.weight",
        )?;
        let linear_2 = mixed_linear(
            config.projector_hidden_size,
            config.text_hidden_size,
            config.projector_bias,
            vb.pp("linear_2"),
            &mut quantized_weights,
            "linear_2.weight",
        )?;
        if !quantized_weights.is_empty() {
            let mut names: Vec<_> = quantized_weights.into_keys().collect();
            names.sort();
            candle::bail!("unused LFM2-VL projector quantized linear weights: {names:?}")
        }
        Ok(Self {
            factor: config.downsample_factor,
            input_size,
            layer_norm,
            linear_1,
            activation: config.projector_hidden_act,
            linear_2,
            output_size: config.text_hidden_size,
        })
    }

    pub fn pixel_unshuffle(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, rows, cols, channels) = hidden_states.dims4()?;
        if self.factor == 0 {
            candle::bail!("LFM2-VL projector factor must be greater than zero")
        }
        if rows == 0 || cols == 0 || channels == 0 {
            candle::bail!("LFM2-VL projector input dimensions must be positive")
        }
        if rows % self.factor != 0 || cols % self.factor != 0 {
            candle::bail!(
                "LFM2-VL grid [{rows}, {cols}] is not divisible by factor {}",
                self.factor
            )
        }
        let factor_squared = self
            .factor
            .checked_mul(self.factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL pixel-unshuffle factor overflow".into()))?;
        let output_channels = channels
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL pixel-unshuffle channel overflow".into()))?;
        let reshaped =
            hidden_states.reshape((batch, rows, cols / self.factor, channels * self.factor))?;
        let permuted = reshaped.permute((0, 2, 1, 3))?;
        let reshaped = permuted.reshape((
            batch,
            cols / self.factor,
            rows / self.factor,
            output_channels,
        ))?;
        reshaped.permute((0, 2, 1, 3))
    }

    fn forward_stages(&self, hidden_states: &Tensor) -> Result<ProjectorStages> {
        let pixel_unshuffle = self.pixel_unshuffle(hidden_states)?;
        let output_channels = pixel_unshuffle.dim(3)?;
        if output_channels != self.input_size {
            candle::bail!(
                "LFM2-VL pixel-unshuffle produced {output_channels} channels, expected {}",
                self.input_size
            )
        }
        let layer_norm = match &self.layer_norm {
            Some(layer_norm) => Some(layer_norm.forward(&pixel_unshuffle)?),
            None => None,
        };
        let normalized = match layer_norm.as_ref() {
            Some(layer_norm) => layer_norm,
            None => &pixel_unshuffle,
        };
        let linear_1 = self.linear_1.forward(normalized)?;
        let activation = self.activation.forward(&linear_1)?;
        let linear_2 = self.linear_2.forward(&activation)?;
        Ok(ProjectorStages {
            pixel_unshuffle,
            layer_norm,
            linear_1,
            activation,
            output: linear_2.clone(),
            linear_2,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Ok(self.forward_stages(hidden_states)?.output)
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

fn mixed_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    quantized_weights: &mut HashMap<String, QTensor>,
    weight_name: &str,
) -> Result<LinearOp> {
    match quantized_weights.remove(weight_name) {
        Some(weight) => {
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(LinearOp::from_qtensor(weight, bias))
        }
        None => Ok(LinearOp::Dense(if bias {
            linear(in_dim, out_dim, vb)?
        } else {
            linear_no_bias(in_dim, out_dim, vb)?
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config(use_layer_norm: bool) -> Lfm2VlConfig {
        Lfm2VlConfig::from_json(
            &serde_json::json!({
                "model_type": "lfm2_vl",
                "image_token_id": 3,
                "downsample_factor": 2,
                "projector_hidden_size": 24,
                "projector_hidden_act": "gelu",
                "projector_bias": true,
                "projector_use_layernorm": use_layer_norm,
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
        .expect("fixed projector config")
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
        eprintln!("lfm2_vl projector {label}: max_abs={max_abs:.9e}, cosine={cosine:.9}");
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

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
    }

    #[test]
    fn pixel_unshuffle_matches_pinned_official_order() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(false);
        let projector = Lfm2VlProjector::new(&config, VarBuilder::zeros(DType::F32, &device))?;
        let input = Tensor::arange(0f32, 48f32, &device)?.reshape((1, 4, 6, 2))?;
        let actual = projector.pixel_unshuffle(&input)?;
        let expected = Tensor::new(
            &[[
                [
                    [0f32, 1., 2., 3., 12., 13., 14., 15.],
                    [4., 5., 6., 7., 16., 17., 18., 19.],
                    [8., 9., 10., 11., 20., 21., 22., 23.],
                ],
                [
                    [24., 25., 26., 27., 36., 37., 38., 39.],
                    [28., 29., 30., 31., 40., 41., 42., 43.],
                    [32., 33., 34., 35., 44., 45., 46., 47.],
                ],
            ]],
            &device,
        )?;
        assert_close(&actual, &expected, 0.0, "monotonic pixel-unshuffle")
    }

    #[test]
    fn fixture_matches_each_projector_stage() -> Result<()> {
        let device = Device::Cpu;
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let projector = Lfm2VlProjector::new(
            &tiny_config(true),
            weights
                .pp("weights")
                .pp("model")
                .pp("multi_modal_projector"),
        )?;
        let vision = fixture_tensor(&tensors, "stage.vision.last_hidden_state")?
            .narrow(1, 0, 8)?
            .reshape((1, 2, 4, 16))?;
        let stages = projector.forward_stages(&vision)?;
        assert_close(
            &stages.pixel_unshuffle,
            fixture_tensor(&tensors, "stage.projector.pixel_unshuffle")?,
            2e-5,
            "pixel-unshuffle",
        )?;
        assert_close(
            stages.layer_norm.as_ref().ok_or_else(|| {
                candle::Error::Msg("fixture projector unexpectedly lacks layer norm".into())
            })?,
            fixture_tensor(&tensors, "stage.projector.layer_norm")?,
            2e-5,
            "layer norm",
        )?;
        assert_close(
            &stages.linear_1,
            fixture_tensor(&tensors, "stage.projector.linear_1")?,
            2e-5,
            "linear 1",
        )?;
        assert_close(
            &stages.activation,
            fixture_tensor(&tensors, "stage.projector.activation")?,
            2e-5,
            "activation",
        )?;
        assert_close(
            &stages.linear_2,
            fixture_tensor(&tensors, "stage.projector.linear_2")?,
            2e-5,
            "linear 2",
        )?;
        assert_close(
            &stages.output.reshape((2, 12))?,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "projector output",
        )?;
        Ok(())
    }

    #[test]
    fn optional_layer_norm_is_skipped_without_layer_norm_weights() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "linear_1.weight".to_string(),
            Tensor::ones((24, 64), DType::F32, &device)?,
        );
        tensors.insert(
            "linear_2.weight".to_string(),
            Tensor::ones((12, 24), DType::F32, &device)?,
        );
        let mut config = tiny_config(false);
        config.projector_bias = false;
        let projector = Lfm2VlProjector::new(
            &config,
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        let input = Tensor::zeros((1, 2, 4, 16), DType::F32, &device)?;
        let stages = projector.forward_stages(&input)?;
        assert!(stages.layer_norm.is_none());
        assert_eq!(stages.output.dims(), [1, 1, 2, 12]);
        Ok(())
    }

    #[test]
    fn rejects_nondivisible_or_wrong_channel_grids() -> Result<()> {
        let device = Device::Cpu;
        let projector =
            Lfm2VlProjector::new(&tiny_config(false), VarBuilder::zeros(DType::F32, &device))?;
        assert!(projector
            .pixel_unshuffle(&Tensor::zeros((1, 3, 4, 16), DType::F32, &device)?)
            .is_err());
        assert!(projector
            .forward(&Tensor::zeros((1, 2, 4, 15), DType::F32, &device)?)
            .is_err());
        Ok(())
    }
}
