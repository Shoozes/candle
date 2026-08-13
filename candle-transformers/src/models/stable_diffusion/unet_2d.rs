//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use super::embeddings::{TimestepEmbedding, Timesteps};
use super::unet_2d_blocks::*;
use crate::models::with_tracing::{conv2d, Conv2d};
use candle::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub out_channels: usize,
    /// When `None` no cross-attn is used, when `Some(d)` then cross-attn is used and `d` is the
    /// number of transformer blocks to be used.
    pub use_cross_attn: Option<usize>,
    pub attention_head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionModelConfig {
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub blocks: Vec<BlockConfig>,
    pub layers_per_block: usize,
    pub downsample_padding: usize,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub cross_attention_dim: usize,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig {
                    out_channels: 320,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 640,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

fn expected_down_block_additional_residuals(
    block_count: usize,
    layers_per_block: usize,
) -> Result<usize> {
    let downsample_outputs = match block_count.checked_sub(1) {
        Some(count) => count,
        None => candle::bail!("UNet additional residuals require at least one down block"),
    };
    let resnet_outputs = match block_count.checked_mul(layers_per_block) {
        Some(count) => count,
        None => candle::bail!("UNet down-block residual count overflow"),
    };
    match 1usize
        .checked_add(resnet_outputs)
        .and_then(|count| count.checked_add(downsample_outputs))
    {
        Some(count) => Ok(count),
        None => candle::bail!("UNet down-block residual count overflow"),
    }
}

fn validate_down_block_additional_residual_count(expected: usize, found: usize) -> Result<()> {
    if found != expected {
        candle::bail!(
            "down_block_additional_residuals count mismatch: expected {expected}, found {found}"
        )
    }
    Ok(())
}

fn validate_additional_residual_tensor(
    label: &str,
    expected: &Tensor,
    found: &Tensor,
) -> Result<()> {
    if found.dims() != expected.dims() {
        candle::bail!(
            "{label} shape mismatch: expected {:?}, found {:?}",
            expected.dims(),
            found.dims()
        )
    }
    if found.dtype() != expected.dtype() {
        candle::bail!(
            "{label} dtype mismatch: expected {:?}, found {:?}",
            expected.dtype(),
            found.dtype()
        )
    }
    if !found.device().same_device(expected.device()) {
        candle::bail!(
            "{label} device mismatch: expected {:?}, found {:?}",
            expected.device().location(),
            found.device().location()
        )
    }
    Ok(())
}

fn add_down_block_additional_residuals(
    down_block_res_xs: Vec<Tensor>,
    additional_residuals: Option<&[Tensor]>,
) -> Result<Vec<Tensor>> {
    let Some(additional_residuals) = additional_residuals else {
        return Ok(down_block_res_xs);
    };
    validate_down_block_additional_residual_count(
        down_block_res_xs.len(),
        additional_residuals.len(),
    )?;

    // Validate the complete inventory before allocating any full-sized sums.
    for (index, (expected, found)) in down_block_res_xs
        .iter()
        .zip(additional_residuals.iter())
        .enumerate()
    {
        validate_additional_residual_tensor(
            &format!("down_block_additional_residuals[{index}]"),
            expected,
            found,
        )?;
    }

    down_block_res_xs
        .iter()
        .zip(additional_residuals.iter())
        .map(|(xs, residual)| xs + residual)
        .collect()
}

fn add_mid_block_additional_residual(
    xs: Tensor,
    additional_residual: Option<&Tensor>,
) -> Result<Tensor> {
    let Some(additional_residual) = additional_residual else {
        return Ok(xs);
    };
    validate_additional_residual_tensor("mid_block_additional_residual", &xs, additional_residual)?;
    additional_residual + xs
}

#[derive(Debug)]
pub(crate) enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: Conv2d,
    span: tracing::Span,
    config: UNet2DConditionModelConfig,
}

impl UNet2DConditionModel {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        use_flash_attn: bool,
        config: UNet2DConditionModelConfig,
    ) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;

        let vs_db = vs.pp("down_blocks");
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let in_channels = if i > 0 {
                    config.blocks[i - 1].out_channels
                } else {
                    b_channels
                };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                    )?;
                    Ok(UNetDownBlock::CrossAttn(block))
                } else {
                    let block = DownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    )?;
                    Ok(UNetDownBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // https://github.com/huggingface/diffusers/blob/a76f2ad538e73b34d5fe7be08c8eb8ab38c7e90c/src/diffusers/models/unet_2d_condition.py#L462
        let mid_transformer_layers_per_block = match config.blocks.last() {
            None => 1,
            Some(block) => block.use_cross_attn.unwrap_or(1),
        };
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers_per_block,
            ..Default::default()
        };

        let mid_block = UNetMidBlock2DCrossAttn::new(
            vs.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            use_flash_attn,
            mid_cfg,
        )?;

        let vs_ub = vs.pp("up_blocks");
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[n_blocks - 1 - i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let prev_out_channels = if i > 0 {
                    config.blocks[n_blocks - i].out_channels
                } else {
                    bl_channels
                };
                let in_channels = {
                    let index = if i == n_blocks - 1 {
                        0
                    } else {
                        n_blocks - i - 2
                    };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig {
                        upblock: ub_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                    )?;
                    Ok(UNetUpBlock::CrossAttn(block))
                } else {
                    let block = UpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    )?;
                    Ok(UNetUpBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            b_channels,
            config.norm_eps,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(b_channels, out_channels, 3, conv_cfg, vs.pp("conv_out"))?;
        let span = tracing::span!(tracing::Level::TRACE, "unet2d");
        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.forward_with_additional_residuals(xs, timestep, encoder_hidden_states, None, None)
    }

    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(additional_residuals) = down_block_additional_residuals {
            let expected = expected_down_block_additional_residuals(
                self.config.blocks.len(),
                self.config.layers_per_block,
            )?;
            validate_down_block_additional_residual_count(expected, additional_residuals.len())?;
        }
        let (bsize, _channels, height, width) = xs.dims4()?;
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2usize.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;
        // 0. center input if necessary
        let xs = if self.config.center_input_sample {
            ((xs * 2.0)? - 1.0)?
        } else {
            xs.clone()
        };
        // 1. time
        let emb = (Tensor::ones(bsize, xs.dtype(), device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;
        // 2. pre-process
        let xs = self.conv_in.forward(&xs)?;
        // 3. down
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb))?,
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }

        // A previous version of this code used in-place addition here, which modified the tensor
        // that is also the input to the mid block.
        let mut down_block_res_xs = add_down_block_additional_residuals(
            down_block_res_xs,
            down_block_additional_residuals,
        )?;

        // 4. mid
        let xs = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;
        let xs = add_mid_block_additional_residual(xs, mid_block_additional_residual)?;
        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let split_at = match down_block_res_xs.len().checked_sub(n_resnets) {
                Some(split_at) => split_at,
                None => candle::bail!(
                    "UNet skip residual inventory exhausted at up block {i}: expected {n_resnets}, found {}",
                    down_block_res_xs.len()
                ),
            };
            let res_xs = down_block_res_xs.split_off(split_at);
            if i < n_blocks - 1 && forward_upsample_size {
                let last_tensor = match down_block_res_xs.last() {
                    Some(last_tensor) => last_tensor,
                    None => {
                        candle::bail!("UNet skip residual inventory is empty before up block {i}")
                    }
                };
                let (_, _, h, w) = last_tensor.dims4()?;
                upsample_size = Some((h, w))
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size)?,
                UNetUpBlock::CrossAttn(b) => b.forward(
                    &xs,
                    &res_xs,
                    Some(&emb),
                    upsample_size,
                    Some(encoder_hidden_states),
                )?,
            };
        }
        // 6. post-process
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn error_message<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(error) => error.to_string(),
        }
    }

    fn tiny_config() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![BlockConfig {
                // Down/up blocks currently use the ResNet default of 32 groups.
                out_channels: 32,
                use_cross_attn: None,
                attention_head_dim: 1,
            }],
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 1,
            norm_eps: 1e-5,
            cross_attention_dim: 4,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }

    #[test]
    fn additional_residual_count_uses_checked_unet_skip_inventory() -> Result<()> {
        assert_eq!(expected_down_block_additional_residuals(3, 2)?, 9);
        let error = error_message(expected_down_block_additional_residuals(usize::MAX, 2));
        assert!(error.contains("residual count overflow"), "{error}");
        Ok(())
    }

    #[test]
    fn additional_residual_count_rejects_short_and_long_inventories() {
        let short = error_message(validate_down_block_additional_residual_count(9, 8));
        assert!(short.contains("expected 9, found 8"), "{short}");

        let long = error_message(validate_down_block_additional_residual_count(9, 10));
        assert!(long.contains("expected 9, found 10"), "{long}");
    }

    #[test]
    fn down_residual_rejects_broadcastable_shape_before_addition() -> Result<()> {
        let device = Device::Cpu;
        let base = vec![
            Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?,
        ];
        let additional = [
            Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 4, 1, 1), DType::F32, &device)?,
        ];

        let error = error_message(add_down_block_additional_residuals(base, Some(&additional)));
        assert!(
            error.contains("down_block_additional_residuals[1] shape mismatch"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn down_residual_rejects_dtype_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let base = vec![Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?];
        let additional = [Tensor::zeros((1, 4, 2, 2), DType::F64, &device)?];

        let error = error_message(add_down_block_additional_residuals(base, Some(&additional)));
        assert!(
            error.contains("down_block_additional_residuals[0] dtype mismatch"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn mid_residual_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?;
        let additional = Tensor::zeros((1, 4, 1, 1), DType::F32, &device)?;

        let error = error_message(add_mid_block_additional_residual(xs, Some(&additional)));
        assert!(
            error.contains("mid_block_additional_residual shape mismatch"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn forward_rejects_bad_counts_and_preserves_none_and_zero_behavior() -> Result<()> {
        let device = Device::Cpu;
        let model = UNet2DConditionModel::new(
            candle_nn::VarBuilder::zeros(DType::F32, &device),
            2,
            2,
            false,
            tiny_config(),
        )?;
        let xs = Tensor::zeros((1, 2, 2, 2), DType::F32, &device)?;
        let encoder_hidden_states = Tensor::zeros((1, 1, 4), DType::F32, &device)?;

        let short_down = [Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?];
        let short_error = error_message(model.forward_with_additional_residuals(
            &xs,
            1.,
            &encoder_hidden_states,
            Some(&short_down),
            None,
        ));
        assert!(short_error.contains("expected 2, found 1"), "{short_error}");

        let long_down = [
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
        ];
        let long_error = error_message(model.forward_with_additional_residuals(
            &xs,
            1.,
            &encoder_hidden_states,
            Some(&long_down),
            None,
        ));
        assert!(long_error.contains("expected 2, found 3"), "{long_error}");

        let baseline = model.forward(&xs, 1., &encoder_hidden_states)?;
        let explicit_none =
            model.forward_with_additional_residuals(&xs, 1., &encoder_hidden_states, None, None)?;
        let zero_down = [
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
        ];
        let zero_mid = Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?;
        let explicit_zero = model.forward_with_additional_residuals(
            &xs,
            1.,
            &encoder_hidden_states,
            Some(&zero_down),
            Some(&zero_mid),
        )?;

        let baseline = baseline.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(explicit_none.flatten_all()?.to_vec1::<f32>()?, baseline);
        assert_eq!(explicit_zero.flatten_all()?.to_vec1::<f32>()?, baseline);
        Ok(())
    }
}
