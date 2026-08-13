//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
pub use super::embeddings::SdxlTextTimeAdditionConfig;
use super::embeddings::{SdxlTextTimeAdditionEmbedding, TimestepEmbedding, Timesteps};
use super::unet_2d_blocks::*;
use crate::models::with_tracing::{conv2d, Conv2d};
use candle::{DType, DeviceLocation, Result, Tensor};
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

#[derive(Debug, Clone, Copy)]
pub struct SdxlTextTimeConditioning<'a> {
    pub pooled_text_embeds: &'a Tensor,
    pub time_ids: &'a Tensor,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UNet2DConditioning<'a> {
    pub text_time: Option<SdxlTextTimeConditioning<'a>>,
    pub down_block_additional_residuals: Option<&'a [Tensor]>,
    pub mid_block_additional_residual: Option<&'a Tensor>,
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

fn validate_text_time_input(
    label: &str,
    found: &Tensor,
    expected_dims: &[usize],
    sample: &Tensor,
) -> Result<()> {
    if found.dims() != expected_dims {
        candle::bail!(
            "SDXL {label} shape mismatch: expected {expected_dims:?}, found {:?}",
            found.dims()
        )
    }
    if found.dtype() != sample.dtype() {
        candle::bail!(
            "SDXL {label} dtype mismatch: expected {:?}, found {:?}",
            sample.dtype(),
            found.dtype()
        )
    }
    validate_text_time_device(label, sample.device().location(), found.device().location())?;
    Ok(())
}

fn validate_text_time_device(
    label: &str,
    expected: DeviceLocation,
    found: DeviceLocation,
) -> Result<()> {
    if found != expected {
        candle::bail!("SDXL {label} device mismatch: expected {expected:?}, found {found:?}")
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
    text_time_addition_embedding: Option<SdxlTextTimeAdditionEmbedding>,
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
        Self::new_with_added_conditioning(
            vs,
            in_channels,
            out_channels,
            use_flash_attn,
            config,
            None,
        )
    }

    pub fn new_with_added_conditioning(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        use_flash_attn: bool,
        config: UNet2DConditionModelConfig,
        text_time_addition_config: Option<SdxlTextTimeAdditionConfig>,
    ) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let first_block = config
            .blocks
            .first()
            .ok_or_else(|| candle::Error::Msg("UNet requires at least one block".into()))?;
        let last_block = config
            .blocks
            .last()
            .ok_or_else(|| candle::Error::Msg("UNet requires at least one block".into()))?;
        let b_channels = first_block.out_channels;
        let bl_channels = last_block.out_channels;
        let bl_attention_head_dim = last_block.attention_head_dim;
        let time_embed_dim = b_channels
            .checked_mul(4)
            .ok_or_else(|| candle::Error::Msg("UNet timestep embedding width overflow".into()))?;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;
        let text_time_addition_embedding = match text_time_addition_config {
            Some(addition_config) => Some(SdxlTextTimeAdditionEmbedding::new(
                vs.clone(),
                time_embed_dim,
                config.flip_sin_to_cos,
                config.freq_shift,
                addition_config,
            )?),
            None => None,
        };

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
            text_time_addition_embedding,
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
        self.forward_with_conditioning(
            xs,
            timestep,
            encoder_hidden_states,
            UNet2DConditioning::default(),
        )
    }

    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_conditioning(
            xs,
            timestep,
            encoder_hidden_states,
            UNet2DConditioning {
                text_time: None,
                down_block_additional_residuals,
                mid_block_additional_residual,
            },
        )
    }

    pub fn forward_with_conditioning(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        conditioning: UNet2DConditioning<'_>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        if let Some(additional_residuals) = conditioning.down_block_additional_residuals {
            let expected = expected_down_block_additional_residuals(
                self.config.blocks.len(),
                self.config.layers_per_block,
            )?;
            validate_down_block_additional_residual_count(expected, additional_residuals.len())?;
        }
        let (bsize, _channels, height, width) = xs.dims4()?;
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let text_time_conditioning = match (
            self.text_time_addition_embedding.as_ref(),
            conditioning.text_time,
        ) {
            (None, None) => None,
            (None, Some(_)) => {
                candle::bail!("SDXL text_time conditioning was supplied to an unconfigured UNet")
            }
            (Some(_), None) => {
                candle::bail!("SDXL text_time conditioning is required by this UNet")
            }
            (Some(addition_embedding), Some(text_time)) => {
                if xs.dtype() != DType::F32 {
                    candle::bail!(
                        "SDXL text_time conditioning currently requires F32 tensors, found {:?}",
                        xs.dtype()
                    )
                }
                let expected_pooled_width = addition_embedding.config().pooled_text_embed_dim()?;
                validate_text_time_input(
                    "pooled_text_embeds",
                    text_time.pooled_text_embeds,
                    &[bsize, expected_pooled_width],
                    xs,
                )?;
                validate_text_time_input(
                    "time_ids",
                    text_time.time_ids,
                    &[bsize, addition_embedding.config().time_id_count],
                    xs,
                )?;
                Some((addition_embedding, text_time))
            }
        };
        let num_upsamplers = n_blocks - 1;
        let num_upsamplers = u32::try_from(num_upsamplers)
            .map_err(|_| candle::Error::Msg("UNet upsampler count overflow".into()))?;
        let default_overall_up_factor = 2usize
            .checked_pow(num_upsamplers)
            .ok_or_else(|| candle::Error::Msg("UNet overall upsample factor overflow".into()))?;
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
        let mut emb = self.time_embedding.forward(&emb)?;
        if let Some((addition_embedding, text_time)) = text_time_conditioning {
            let addition =
                addition_embedding.forward(text_time.pooled_text_embeds, text_time.time_ids)?;
            validate_additional_residual_tensor("SDXL text_time addition", &emb, &addition)?;
            emb = (&emb + &addition)?;
        }
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
            conditioning.down_block_additional_residuals,
        )?;

        // 4. mid
        let xs = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;
        let xs = add_mid_block_additional_residual(xs, conditioning.mid_block_additional_residual)?;
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
    use std::collections::HashMap;

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

    fn tiny_text_time_config() -> SdxlTextTimeAdditionConfig {
        SdxlTextTimeAdditionConfig {
            addition_time_embed_dim: 4,
            projection_class_embeddings_input_dim: 10,
            time_id_count: 2,
        }
    }

    fn tiny_text_time_conditioning<'a>(
        pooled_text_embeds: &'a Tensor,
        time_ids: &'a Tensor,
    ) -> UNet2DConditioning<'a> {
        UNet2DConditioning {
            text_time: Some(SdxlTextTimeConditioning {
                pooled_text_embeds,
                time_ids,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn sdxl_text_time_config_derives_official_dimensions_and_rejects_invalid_values() -> Result<()>
    {
        let official = SdxlTextTimeAdditionConfig {
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 2816,
            time_id_count: 6,
        };
        assert_eq!(official.pooled_text_embed_dim()?, 1280);

        for (config, message) in [
            (
                SdxlTextTimeAdditionConfig {
                    addition_time_embed_dim: 0,
                    ..official
                },
                "greater than zero",
            ),
            (
                SdxlTextTimeAdditionConfig {
                    addition_time_embed_dim: 255,
                    ..official
                },
                "must be even",
            ),
            (
                SdxlTextTimeAdditionConfig {
                    time_id_count: 0,
                    ..official
                },
                "time_id_count must be greater than zero",
            ),
            (
                SdxlTextTimeAdditionConfig {
                    addition_time_embed_dim: usize::MAX - 1,
                    time_id_count: 2,
                    ..official
                },
                "width overflow",
            ),
            (
                SdxlTextTimeAdditionConfig {
                    projection_class_embeddings_input_dim: 1536,
                    ..official
                },
                "pooled text embedding width must be greater than zero",
            ),
        ] {
            let error = error_message(config.pooled_text_embed_dim());
            assert!(error.contains(message), "{error}");
        }
        Ok(())
    }

    #[test]
    fn sdxl_text_time_embedding_uses_pooled_text_and_time_ids() -> Result<()> {
        let device = Device::Cpu;
        let config = SdxlTextTimeAdditionConfig {
            addition_time_embed_dim: 2,
            projection_class_embeddings_input_dim: 6,
            time_id_count: 2,
        };
        let linear_1_weight = Tensor::from_vec(
            vec![
                1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                1., 0., 0.,
            ],
            (4, 6),
            &device,
        )?;
        let linear_2_weight = Tensor::from_vec(
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            (4, 4),
            &device,
        )?;
        let mut weights = HashMap::new();
        weights.insert("add_embedding.linear_1.weight".into(), linear_1_weight);
        weights.insert(
            "add_embedding.linear_1.bias".into(),
            Tensor::zeros(4, DType::F32, &device)?,
        );
        weights.insert("add_embedding.linear_2.weight".into(), linear_2_weight);
        weights.insert(
            "add_embedding.linear_2.bias".into(),
            Tensor::zeros(4, DType::F32, &device)?,
        );
        let embedding = SdxlTextTimeAdditionEmbedding::new(
            candle_nn::VarBuilder::from_tensors(weights, DType::F32, &device),
            4,
            true,
            0.,
            config,
        )?;

        let pooled = Tensor::from_slice(&[1f32, 2.], (1, 2), &device)?;
        let changed_pooled = Tensor::from_slice(&[3f32, 2.], (1, 2), &device)?;
        let time_ids = Tensor::from_slice(&[0f32, 0.], (1, 2), &device)?;
        let changed_time_ids = Tensor::from_slice(&[1f32, 0.], (1, 2), &device)?;
        let baseline = embedding
            .forward(&pooled, &time_ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let pooled_result = embedding
            .forward(&changed_pooled, &time_ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let time_result = embedding
            .forward(&pooled, &changed_time_ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_ne!(pooled_result, baseline);
        assert_ne!(time_result, baseline);
        Ok(())
    }

    #[test]
    fn sdxl_text_time_embedding_rejects_malformed_tensor_contracts() -> Result<()> {
        let device = Device::Cpu;
        let config = SdxlTextTimeAdditionConfig {
            addition_time_embed_dim: 2,
            projection_class_embeddings_input_dim: 6,
            time_id_count: 2,
        };
        let embedding = SdxlTextTimeAdditionEmbedding::new(
            candle_nn::VarBuilder::zeros(DType::F32, &device),
            4,
            true,
            0.,
            config,
        )?;
        let pooled = Tensor::zeros((1, 2), DType::F32, &device)?;
        let time_ids = Tensor::zeros((1, 2), DType::F32, &device)?;

        let rank_error =
            error_message(embedding.forward(&Tensor::zeros(2, DType::F32, &device)?, &time_ids));
        assert!(rank_error.contains("rank 2"), "{rank_error}");

        let width_error = error_message(
            embedding.forward(&Tensor::zeros((1, 3), DType::F32, &device)?, &time_ids),
        );
        assert!(width_error.contains("width mismatch"), "{width_error}");

        let batch_error =
            error_message(embedding.forward(&pooled, &Tensor::zeros((2, 2), DType::F32, &device)?));
        assert!(batch_error.contains("batch mismatch"), "{batch_error}");

        let count_error =
            error_message(embedding.forward(&pooled, &Tensor::zeros((1, 3), DType::F32, &device)?));
        assert!(count_error.contains("count mismatch"), "{count_error}");

        let dtype_error =
            error_message(embedding.forward(&pooled, &Tensor::zeros((1, 2), DType::F64, &device)?));
        assert!(dtype_error.contains("dtype mismatch"), "{dtype_error}");
        Ok(())
    }

    #[test]
    fn configured_text_time_unet_validates_before_graph_execution() -> Result<()> {
        let device = Device::Cpu;
        let model = UNet2DConditionModel::new_with_added_conditioning(
            candle_nn::VarBuilder::zeros(DType::F32, &device),
            2,
            2,
            false,
            tiny_config(),
            Some(tiny_text_time_config()),
        )?;
        // The sample has the wrong channel count for conv_in. Conditioning errors must win first.
        let invalid_sample = Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?;
        let encoder_hidden_states = Tensor::zeros((1, 1, 4), DType::F32, &device)?;
        let pooled = Tensor::zeros((1, 2), DType::F32, &device)?;
        let wrong_pooled = Tensor::zeros((1, 3), DType::F32, &device)?;
        let time_ids = Tensor::zeros((1, 2), DType::F32, &device)?;

        let missing = error_message(model.forward(&invalid_sample, 1., &encoder_hidden_states));
        assert!(missing.contains("conditioning is required"), "{missing}");

        let wrong_shape = error_message(model.forward_with_conditioning(
            &invalid_sample,
            1.,
            &encoder_hidden_states,
            tiny_text_time_conditioning(&wrong_pooled, &time_ids),
        ));
        assert!(
            wrong_shape.contains("pooled_text_embeds shape mismatch"),
            "{wrong_shape}"
        );

        let pooled_f64 = Tensor::zeros((1, 2), DType::F64, &device)?;
        let wrong_dtype = error_message(model.forward_with_conditioning(
            &invalid_sample,
            1.,
            &encoder_hidden_states,
            tiny_text_time_conditioning(&pooled_f64, &time_ids),
        ));
        assert!(
            wrong_dtype.contains("pooled_text_embeds dtype mismatch"),
            "{wrong_dtype}"
        );

        let valid_sample = Tensor::zeros((1, 2, 2, 2), DType::F32, &device)?;
        model.forward_with_conditioning(
            &valid_sample,
            1.,
            &encoder_hidden_states,
            tiny_text_time_conditioning(&pooled, &time_ids),
        )?;

        let non_f32_sample = Tensor::zeros((1, 2, 2, 2), DType::F64, &device)?;
        let pooled_f64 = Tensor::zeros((1, 2), DType::F64, &device)?;
        let time_ids_f64 = Tensor::zeros((1, 2), DType::F64, &device)?;
        let dtype_boundary = error_message(model.forward_with_conditioning(
            &non_f32_sample,
            1.,
            &encoder_hidden_states,
            tiny_text_time_conditioning(&pooled_f64, &time_ids_f64),
        ));
        assert!(
            dtype_boundary.contains("currently requires F32"),
            "{dtype_boundary}"
        );
        Ok(())
    }

    #[test]
    fn text_time_device_contract_rejects_different_locations() -> Result<()> {
        validate_text_time_device(
            "pooled_text_embeds",
            DeviceLocation::Cpu,
            DeviceLocation::Cpu,
        )?;
        let error = error_message(validate_text_time_device(
            "pooled_text_embeds",
            DeviceLocation::Cpu,
            DeviceLocation::Cuda { gpu_id: 0 },
        ));
        assert!(error.contains("device mismatch"), "{error}");
        Ok(())
    }

    #[test]
    fn structured_conditioning_combines_text_time_and_residuals() -> Result<()> {
        let device = Device::Cpu;
        let model = UNet2DConditionModel::new_with_added_conditioning(
            candle_nn::VarBuilder::zeros(DType::F32, &device),
            2,
            2,
            false,
            tiny_config(),
            Some(tiny_text_time_config()),
        )?;
        let xs = Tensor::zeros((1, 2, 2, 2), DType::F32, &device)?;
        let encoder_hidden_states = Tensor::zeros((1, 1, 4), DType::F32, &device)?;
        let pooled = Tensor::zeros((1, 2), DType::F32, &device)?;
        let time_ids = Tensor::zeros((1, 2), DType::F32, &device)?;
        let text_time = SdxlTextTimeConditioning {
            pooled_text_embeds: &pooled,
            time_ids: &time_ids,
        };
        let baseline = model.forward_with_conditioning(
            &xs,
            1.,
            &encoder_hidden_states,
            UNet2DConditioning {
                text_time: Some(text_time),
                ..Default::default()
            },
        )?;
        let zero_down = [
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
            Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?,
        ];
        let zero_mid = Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?;
        let combined = model.forward_with_conditioning(
            &xs,
            1.,
            &encoder_hidden_states,
            UNet2DConditioning {
                text_time: Some(text_time),
                down_block_additional_residuals: Some(&zero_down),
                mid_block_additional_residual: Some(&zero_mid),
            },
        )?;
        assert_eq!(
            combined.flatten_all()?.to_vec1::<f32>()?,
            baseline.flatten_all()?.to_vec1::<f32>()?
        );

        let malformed_down = [Tensor::zeros((1, 32, 2, 2), DType::F32, &device)?];
        let error = error_message(model.forward_with_conditioning(
            &xs,
            1.,
            &encoder_hidden_states,
            UNet2DConditioning {
                text_time: Some(text_time),
                down_block_additional_residuals: Some(&malformed_down),
                mid_block_additional_residual: None,
            },
        ));
        assert!(error.contains("expected 2, found 1"), "{error}");
        Ok(())
    }

    #[test]
    fn text_time_conditioning_rejects_unconfigured_unet_and_empty_blocks() -> Result<()> {
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
        let pooled = Tensor::zeros((1, 2), DType::F32, &device)?;
        let time_ids = Tensor::zeros((1, 2), DType::F32, &device)?;
        let error = error_message(model.forward_with_conditioning(
            &xs,
            1.,
            &encoder_hidden_states,
            tiny_text_time_conditioning(&pooled, &time_ids),
        ));
        assert!(error.contains("unconfigured UNet"), "{error}");

        let mut empty = tiny_config();
        empty.blocks.clear();
        let error = error_message(UNet2DConditionModel::new(
            candle_nn::VarBuilder::zeros(DType::F32, &device),
            2,
            2,
            false,
            empty,
        ));
        assert!(error.contains("requires at least one block"), "{error}");
        Ok(())
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
        let structured_none = model.forward_with_conditioning(
            &xs,
            1.,
            &encoder_hidden_states,
            UNet2DConditioning::default(),
        )?;
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
        assert_eq!(structured_none.flatten_all()?.to_vec1::<f32>()?, baseline);
        assert_eq!(explicit_zero.flatten_all()?.to_vec1::<f32>()?, baseline);
        Ok(())
    }
}
