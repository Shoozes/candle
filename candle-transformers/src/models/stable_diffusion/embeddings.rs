use candle::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdxlTextTimeAdditionConfig {
    pub addition_time_embed_dim: usize,
    pub projection_class_embeddings_input_dim: usize,
    pub time_id_count: usize,
}

impl SdxlTextTimeAdditionConfig {
    pub fn pooled_text_embed_dim(&self) -> Result<usize> {
        if self.addition_time_embed_dim == 0 {
            candle::bail!("SDXL addition_time_embed_dim must be greater than zero")
        }
        if !self.addition_time_embed_dim.is_multiple_of(2) {
            candle::bail!("SDXL addition_time_embed_dim must be even")
        }
        if self.time_id_count == 0 {
            candle::bail!("SDXL time_id_count must be greater than zero")
        }
        let time_embed_width = self
            .addition_time_embed_dim
            .checked_mul(self.time_id_count)
            .ok_or_else(|| candle::Error::Msg("SDXL time embedding width overflow".into()))?;
        let pooled_text_embed_dim = self
            .projection_class_embeddings_input_dim
            .checked_sub(time_embed_width)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "SDXL projection input width {} is smaller than the time embedding width {time_embed_width}",
                    self.projection_class_embeddings_input_dim
                ))
            })?;
        if pooled_text_embed_dim == 0 {
            candle::bail!("SDXL pooled text embedding width must be greater than zero")
        }
        Ok(pooled_text_embed_dim)
    }
}

#[derive(Debug)]
pub struct SdxlTextTimeAdditionEmbedding {
    time_proj: Timesteps,
    add_embedding: TimestepEmbedding,
    config: SdxlTextTimeAdditionConfig,
}

impl SdxlTextTimeAdditionEmbedding {
    pub fn new(
        vs: nn::VarBuilder,
        time_embed_dim: usize,
        flip_sin_to_cos: bool,
        freq_shift: f64,
        config: SdxlTextTimeAdditionConfig,
    ) -> Result<Self> {
        config.pooled_text_embed_dim()?;
        if time_embed_dim == 0 {
            candle::bail!("SDXL base timestep embedding width must be greater than zero")
        }
        let time_proj = Timesteps::new(config.addition_time_embed_dim, flip_sin_to_cos, freq_shift);
        let add_embedding = TimestepEmbedding::new(
            vs.pp("add_embedding"),
            config.projection_class_embeddings_input_dim,
            time_embed_dim,
        )?;
        Ok(Self {
            time_proj,
            add_embedding,
            config,
        })
    }

    pub fn config(&self) -> SdxlTextTimeAdditionConfig {
        self.config
    }

    pub fn forward(&self, pooled_text_embeds: &Tensor, time_ids: &Tensor) -> Result<Tensor> {
        if pooled_text_embeds.rank() != 2 {
            candle::bail!(
                "SDXL pooled_text_embeds must have rank 2, found rank {}",
                pooled_text_embeds.rank()
            )
        }
        if time_ids.rank() != 2 {
            candle::bail!(
                "SDXL time_ids must have rank 2, found rank {}",
                time_ids.rank()
            )
        }
        let (batch_size, pooled_width) = pooled_text_embeds.dims2()?;
        let (time_batch_size, time_id_count) = time_ids.dims2()?;
        let expected_pooled_width = self.config.pooled_text_embed_dim()?;
        if pooled_width != expected_pooled_width {
            candle::bail!(
                "SDXL pooled_text_embeds width mismatch: expected {expected_pooled_width}, found {pooled_width}"
            )
        }
        if time_batch_size != batch_size {
            candle::bail!(
                "SDXL time_ids batch mismatch: expected {batch_size}, found {time_batch_size}"
            )
        }
        if time_id_count != self.config.time_id_count {
            candle::bail!(
                "SDXL time_ids count mismatch: expected {}, found {time_id_count}",
                self.config.time_id_count
            )
        }
        if time_ids.dtype() != pooled_text_embeds.dtype() {
            candle::bail!(
                "SDXL time_ids dtype mismatch: expected {:?}, found {:?}",
                pooled_text_embeds.dtype(),
                time_ids.dtype()
            )
        }
        if !time_ids.device().same_device(pooled_text_embeds.device()) {
            candle::bail!(
                "SDXL time_ids device mismatch: expected {:?}, found {:?}",
                pooled_text_embeds.device().location(),
                time_ids.device().location()
            )
        }

        let time_embeds = self.time_proj.forward(&time_ids.flatten_all()?)?;
        let time_embed_width = self
            .config
            .addition_time_embed_dim
            .checked_mul(self.config.time_id_count)
            .ok_or_else(|| candle::Error::Msg("SDXL time embedding width overflow".into()))?;
        let time_embeds = time_embeds.reshape((batch_size, time_embed_width))?;
        let add_embeds = Tensor::cat(&[pooled_text_embeds, &time_embeds], D::Minus1)?;
        self.add_embedding.forward(&add_embeds)
    }
}

#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    // act_fn: "silu"
    pub fn new(vs: nn::VarBuilder, channel: usize, time_embed_dim: usize) -> Result<Self> {
        let linear_1 = nn::linear(channel, time_embed_dim, vs.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vs.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = nn::ops::silu(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}

#[derive(Debug)]
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
        }
    }
}

impl Module for Timesteps {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let half_dim = (self.num_channels / 2) as u32;
        let exponent = (Tensor::arange(0, half_dim, xs.device())?.to_dtype(candle::DType::F32)?
            * -f64::ln(10000.))?;
        let exponent = (exponent / (half_dim as f64 - self.downscale_freq_shift))?;
        let emb = exponent.exp()?.to_dtype(xs.dtype())?;
        // emb = timesteps[:, None].float() * emb[None, :]
        let emb = xs.unsqueeze(D::Minus1)?.broadcast_mul(&emb.unsqueeze(0)?)?;
        let (cos, sin) = (emb.cos()?, emb.sin()?);
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[&cos, &sin], D::Minus1)?
        } else {
            Tensor::cat(&[&sin, &cos], D::Minus1)?
        };
        if self.num_channels % 2 == 1 {
            emb.pad_with_zeros(D::Minus2, 0, 1)
        } else {
            Ok(emb)
        }
    }
}
