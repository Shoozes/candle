#![allow(dead_code)]
//! # Variational Auto-Encoder (VAE) Models.
//!
//! Auto-encoder models compress their input to a usually smaller latent space
//! before expanding it back to its original shape. This results in the latent values
//! compressing the original information.
use super::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
    UpDecoderBlock2D, UpDecoderBlock2DConfig,
};
use candle::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone)]
struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    norm_num_groups: usize,
    double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            double_z: true,
        }
    }
}

#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
    #[allow(dead_code)]
    config: EncoderConfig,
}

impl Encoder {
    fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: EncoderConfig,
    ) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            in_channels,
            config.block_out_channels[0],
            3,
            conv_cfg,
            vs.pp("conv_in"),
        )?;
        let mut down_blocks = vec![];
        let vs_down_blocks = vs.pp("down_blocks");
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block = DownEncoderBlock2D::new(
                vs_down_blocks.pp(index.to_string()),
                in_channels,
                out_channels,
                cfg,
            )?;
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config
            .block_out_channels
            .last()
            .ok_or_else(|| candle::Error::Msg("VAE encoder requires a channel layout".into()))?;
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(vs.pp("mid_block"), last_block_out_channels, None, mid_cfg)?;
        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            last_block_out_channels,
            1e-6,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out_channels = if config.double_z {
            out_channels.checked_mul(2).ok_or_else(|| {
                candle::Error::Msg("VAE doubled latent-channel width overflows".into())
            })?
        } else {
            out_channels
        };
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = nn::conv2d(
            last_block_out_channels,
            conv_out_channels,
            3,
            conv_cfg,
            vs.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        })
    }
}

impl Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv_in)?;
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)?
        }
        let xs = self
            .mid_block
            .forward(&xs, None)?
            .apply(&self.conv_norm_out)?;
        nn::ops::silu(&xs)?.apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    norm_num_groups: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
        }
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2d,
    up_blocks: Vec<UpDecoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
    #[allow(dead_code)]
    config: DecoderConfig,
}

impl Decoder {
    fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: DecoderConfig,
    ) -> Result<Self> {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config
            .block_out_channels
            .last()
            .ok_or_else(|| candle::Error::Msg("VAE decoder requires a channel layout".into()))?;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            in_channels,
            last_block_out_channels,
            3,
            conv_cfg,
            vs.pp("conv_in"),
        )?;
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(vs.pp("mid_block"), last_block_out_channels, None, mid_cfg)?;
        let mut up_blocks = vec![];
        let vs_up_blocks = vs.pp("up_blocks");
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().copied().rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                ..Default::default()
            };
            let up_block = UpDecoderBlock2D::new(
                vs_up_blocks.pp(index.to_string()),
                in_channels,
                out_channels,
                cfg,
            )?;
            up_blocks.push(up_block)
        }
        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            config.block_out_channels[0],
            1e-6,
            vs.pp("conv_norm_out"),
        )?;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = nn::conv2d(
            config.block_out_channels[0],
            out_channels,
            3,
            conv_cfg,
            vs.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            up_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        })
    }
}

impl Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.mid_block.forward(&self.conv_in.forward(xs)?, None)?;
        for up_block in self.up_blocks.iter() {
            xs = up_block.forward(&xs)?
        }
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoderKLConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
}

impl Default for AutoEncoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
        }
    }
}

pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    std: Tensor,
}

impl DiagonalGaussianDistribution {
    pub fn new(parameters: &Tensor) -> Result<Self> {
        let channels = parameters.dims().get(1).copied().ok_or_else(|| {
            candle::Error::Msg("VAE Gaussian parameters require a channel axis".into())
        })?;
        if channels == 0 || channels % 2 != 0 {
            candle::bail!("VAE Gaussian channel width must be positive and even, got {channels}")
        }
        let [mean, logvar]: [Tensor; 2] = parameters.chunk(2, 1)?.try_into().map_err(|_| {
            candle::Error::Msg("VAE Gaussian split requires mean and log-variance".into())
        })?;
        let std = (logvar * 0.5)?.exp()?;
        Ok(DiagonalGaussianDistribution { mean, std })
    }

    pub fn sample(&self) -> Result<Tensor> {
        let sample = self.mean.randn_like(0., 1.);
        &self.mean + &self.std * sample
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-5
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
#[derive(Debug)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Option<nn::Conv2d>,
    post_quant_conv: Option<nn::Conv2d>,
    pub config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: AutoEncoderKLConfig,
    ) -> Result<Self> {
        if config.block_out_channels.is_empty() || config.block_out_channels.contains(&0) {
            candle::bail!("VAE block channel layout must be nonempty and positive")
        }
        if in_channels == 0 || out_channels == 0 || config.latent_channels == 0 {
            candle::bail!("VAE input, output and latent channels must be positive")
        }
        if config.norm_num_groups == 0
            || config
                .block_out_channels
                .iter()
                .any(|channel| channel % config.norm_num_groups != 0)
        {
            candle::bail!(
                "VAE normalization groups must be positive and divide every block channel width"
            )
        }
        config
            .layers_per_block
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("VAE decoder layer count overflows".into()))?;
        let latent_channels = config.latent_channels;
        let doubled_latent_channels = latent_channels.checked_mul(2).ok_or_else(|| {
            candle::Error::Msg("VAE doubled latent-channel width overflows".into())
        })?;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            double_z: true,
        };
        let encoder = Encoder::new(vs.pp("encoder"), in_channels, latent_channels, encoder_cfg)?;
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
        };
        let decoder = Decoder::new(vs.pp("decoder"), latent_channels, out_channels, decoder_cfg)?;
        let conv_cfg = Default::default();

        let quant_conv = {
            if config.use_quant_conv {
                Some(nn::conv2d(
                    doubled_latent_channels,
                    doubled_latent_channels,
                    1,
                    conv_cfg,
                    vs.pp("quant_conv"),
                )?)
            } else {
                None
            }
        };
        let post_quant_conv = {
            if config.use_post_quant_conv {
                Some(nn::conv2d(
                    latent_channels,
                    latent_channels,
                    1,
                    conv_cfg,
                    vs.pp("post_quant_conv"),
                )?)
            } else {
                None
            }
        };
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            config,
        })
    }

    /// Returns the distribution in the latent space.
    pub fn encode(&self, xs: &Tensor) -> Result<DiagonalGaussianDistribution> {
        let xs = self.encoder.forward(xs)?;
        let parameters = match &self.quant_conv {
            None => xs,
            Some(quant_conv) => quant_conv.forward(&xs)?,
        };
        DiagonalGaussianDistribution::new(&parameters)
    }

    /// Takes as input some sampled values.
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = match &self.post_quant_conv {
            None => xs,
            Some(post_quant_conv) => &post_quant_conv.forward(xs)?,
        };
        self.decoder.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn builder() -> nn::VarBuilder<'static> {
        nn::VarBuilder::zeros(DType::F32, &Device::Cpu)
    }

    fn tiny() -> AutoEncoderKLConfig {
        AutoEncoderKLConfig {
            block_out_channels: vec![4],
            layers_per_block: 1,
            latent_channels: 2,
            norm_num_groups: 2,
            use_quant_conv: true,
            use_post_quant_conv: true,
        }
    }

    #[test]
    fn malformed_channel_layouts_fail_before_construction() {
        for channels in [vec![], vec![0], vec![4, 0]] {
            let mut config = tiny();
            config.block_out_channels = channels;
            let error = AutoEncoderKL::new(builder(), 3, 3, config).expect_err("invalid layout");
            assert!(error.to_string().contains("channel layout"));
        }
        for groups in [0, 3] {
            let mut config = tiny();
            config.norm_num_groups = groups;
            assert!(AutoEncoderKL::new(builder(), 3, 3, config)
                .expect_err("invalid groups")
                .to_string()
                .contains("normalization groups"));
        }
        for latent in [0, usize::MAX / 2 + 1] {
            let mut config = tiny();
            config.latent_channels = latent;
            assert!(AutoEncoderKL::new(builder(), 3, 3, config).is_err());
        }
        assert!(AutoEncoderKL::new(builder(), 0, 3, tiny()).is_err());
        assert!(AutoEncoderKL::new(builder(), 3, 0, tiny()).is_err());
        let mut config = tiny();
        config.layers_per_block = usize::MAX;
        assert!(AutoEncoderKL::new(builder(), 3, 3, config).is_err());
    }

    #[test]
    fn gaussian_requires_positive_even_channel_axis() -> Result<()> {
        for shape in [vec![], vec![4], vec![1, 0], vec![1, 1], vec![1, 3]] {
            let parameters = Tensor::zeros(shape, DType::F32, &Device::Cpu)?;
            assert!(DiagonalGaussianDistribution::new(&parameters).is_err());
        }
        let parameters = Tensor::new(&[[0.5f32, 1.5, 4f32.ln(), 9f32.ln()]], &Device::Cpu)?;
        let distribution = DiagonalGaussianDistribution::new(&parameters)?;
        assert_eq!(distribution.mean.to_vec2::<f32>()?, vec![vec![0.5, 1.5]]);
        for (actual, expected) in distribution.std.to_vec2::<f32>()?[0].iter().zip([2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(distribution.sample()?.dims(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn tiny_valid_vae_constructs_encodes_and_decodes() -> Result<()> {
        let model = AutoEncoderKL::new(builder(), 3, 3, tiny())?;
        let image = Tensor::zeros((1, 3, 2, 2), DType::F32, &Device::Cpu)?;
        let distribution = model.encode(&image)?;
        assert_eq!(distribution.mean.dims(), &[1, 2, 2, 2]);
        assert!(distribution
            .mean
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| *v == 0.0));
        assert!(distribution
            .std
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| *v == 1.0));
        let sample = distribution.sample()?;
        assert_eq!(sample.dims(), &[1, 2, 2, 2]);
        assert_eq!(model.decode(&sample)?.dims(), image.dims());
        Ok(())
    }
}
