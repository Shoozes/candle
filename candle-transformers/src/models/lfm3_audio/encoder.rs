use super::super::lfm2;
use candle::{DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(alias = "dim_in")]
    pub num_mel_bins: usize,
    #[serde(alias = "dim_conv")]
    pub conv_channels: usize,
    #[serde(alias = "dim_out")]
    pub hidden_size: usize,
    pub sliding_window: usize,
    #[serde(alias = "lfm")]
    pub text_config: lfm2::Lfm2Config,
}

#[derive(Debug, Clone)]
pub struct Encoder {
    config: Config,
    conv_in: Conv2d,
    conv_mid: Conv2d,
    conv_end: Conv2d,
    lin_mid: Conv2d,
    lin_end: Conv2d,
    lin_out: Conv1d,
    backbone: lfm2::FeatureEncoder,
    norm: LayerNorm,
    first: Linear,
    second: Linear,
}

impl Encoder {
    pub fn new(config: &Config, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        if config.num_mel_bins == 0
            || !config.num_mel_bins.is_multiple_of(8)
            || config.conv_channels == 0
            || config.hidden_size == 0
            || config.sliding_window == 0
            || text_hidden == 0
            || vb.dtype() != DType::F32
        {
            candle::bail!("invalid LFM3-Audio encoder dimensions, window or dtype")
        }
        let width = config
            .num_mel_bins
            .checked_mul(config.conv_channels)
            .ok_or_else(|| candle::Error::Msg("LFM3-Audio downsampler width overflow".into()))?
            / 8;
        for (a, b) in [
            (config.conv_channels, config.conv_channels),
            (config.conv_channels, 9),
            (width, config.hidden_size),
            (config.hidden_size, text_hidden),
            (text_hidden, text_hidden),
        ] {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| {
                    candle::Error::Msg("LFM3-Audio encoder tensor byte size overflow".into())
                })?;
        }
        let cfg = config.text_config.clone().try_into_config(false)?;
        if cfg.hidden_size != config.hidden_size {
            candle::bail!("audio encoder and LFM2 widths do not match")
        }
        let down = vb.pp("encoder.downsample");
        let conv = |input, output, groups, name| {
            candle_nn::conv2d(
                input,
                output,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    groups,
                    ..Default::default()
                },
                down.pp(name),
            )
        };
        let channels = config.conv_channels;
        Ok(Self {
            config: config.clone(),
            conv_in: conv(1, channels, 1, "conv_in")?,
            conv_mid: conv(channels, channels, channels, "conv_mid")?,
            conv_end: conv(channels, channels, channels, "conv_end")?,
            lin_mid: candle_nn::conv2d(
                channels,
                channels,
                1,
                Default::default(),
                down.pp("lin_mid"),
            )?,
            lin_end: candle_nn::conv2d(
                channels,
                channels,
                1,
                Default::default(),
                down.pp("lin_end"),
            )?,
            lin_out: candle_nn::conv1d(
                width,
                config.hidden_size,
                1,
                Default::default(),
                down.pp("lin_out"),
            )?,
            backbone: lfm2::FeatureEncoder::new(
                &cfg,
                config.sliding_window,
                vb.pp("encoder.lfm.lfm"),
            )?,
            norm: candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("mlp.model.0"))?,
            first: candle_nn::linear(config.hidden_size, text_hidden, vb.pp("mlp.model.1"))?,
            second: candle_nn::linear(text_hidden, text_hidden, vb.pp("mlp.model.3"))?,
        })
    }

    /// Encode one unpadded clip. The symmetric downsampler has seven feature
    /// frames of lookahead; the LFM2 encoder uses causal windowed attention.
    pub fn encode(&self, features: &Tensor, max_feature_frames: usize) -> Result<Tensor> {
        let (batch, mel, frames) = features.dims3()?;
        if batch != 1
            || mel != self.config.num_mel_bins
            || frames == 0
            || frames > max_feature_frames
            || features.dtype() != DType::F32
        {
            candle::bail!("audio encoder expects bounded nonempty F32 [1, mel, frames] features")
        }
        let x = candle_nn::ops::silu(&self.conv_in.forward(&features.unsqueeze(1)?)?)?;
        let x = candle_nn::ops::silu(&self.lin_mid.forward(&self.conv_mid.forward(&x)?)?)?;
        let x = candle_nn::ops::silu(&self.lin_end.forward(&self.conv_end.forward(&x)?)?)?;
        let x = self
            .lin_out
            .forward(&x.flatten(1, 2)?)?
            .transpose(1, 2)?
            .contiguous()?;
        if x.dim(1)? != frames.div_ceil(8) {
            candle::bail!("audio downsampler token-count mismatch")
        }
        let hidden = self.backbone.forward(&x)?;
        self.second.forward(
            &self
                .first
                .forward(&self.norm.forward(&hidden)?)?
                .gelu_erf()?,
        )
    }
}
