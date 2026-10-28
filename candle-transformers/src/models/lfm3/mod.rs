//! LFM3 inference operations, separate from the LFM2 architecture.
//!
//! The initial execution lane is F32, with caller-bounded single-sequence caches.
//! Reference: LiquidAI/LFM3-Audio-2B at 9d49701c372d6011d900f6abb40c6b296fb706dd.

mod config;
mod layers;

pub use config::{Config, LayerType, RopeParameters};
use layers::Layer;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

#[derive(Debug, Clone, Default)]
pub(crate) struct LayerState {
    pub kv: Option<(Tensor, Tensor)>,
    pub conv: Option<Tensor>,
}

/// A bounded, append-only sequence. Clear it before reusing a model for a new request.
#[derive(Debug, Clone)]
pub struct Cache {
    config: Config,
    states: Vec<LayerState>,
    position: usize,
    max_length: usize,
}

impl Cache {
    pub fn new(config: &Config, max_length: usize) -> Result<Self> {
        config.validate()?;
        if !config.is_causal {
            candle::bail!("LFM3 bidirectional execution does not accept a persistent cache")
        }
        if max_length == 0 || max_length > config.max_position_embeddings {
            candle::bail!("LFM3 cache limit must be in 1..=max_position_embeddings")
        }
        config.checked_cache_elements(max_length)?;
        Ok(Self {
            config: config.clone(),
            states: vec![LayerState::default(); config.num_hidden_layers],
            position: 0,
            max_length,
        })
    }

    pub fn clear(&mut self) {
        self.states.fill(LayerState::default());
        self.position = 0;
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Retained attention positions, excluding constant-size convolution windows.
    pub fn attention_lengths(&self) -> Result<Vec<usize>> {
        self.states
            .iter()
            .map(|s| s.kv.as_ref().map_or(Ok(0), |(k, _)| k.dim(2)))
            .collect()
    }
}

/// Transformer/convolution body; its variable root contains `layers` and `embedding_norm`.
#[derive(Debug, Clone)]
pub struct Backbone {
    config: Config,
    layers: Vec<Layer>,
    norm: RmsNorm,
    device: Device,
}

impl Backbone {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        if vb.dtype() != DType::F32 {
            candle::bail!("LFM3 currently requires F32 execution")
        }
        let layers = (0..config.num_hidden_layers)
            .map(|i| Layer::new(config, i, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            layers,
            norm: candle_nn::rms_norm(
                config.hidden_size,
                config.norm_eps,
                vb.pp("embedding_norm"),
            )?,
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn validate_input(&self, input: &Tensor) -> Result<usize> {
        let (batch, length, hidden) = input.dims3()?;
        if batch != 1 || length == 0 || hidden != self.config.hidden_size {
            candle::bail!(
                "LFM3 expects nonempty [1, sequence, {}] embeddings",
                self.config.hidden_size
            )
        }
        if input.dtype() != DType::F32 || !input.device().same_device(&self.device) {
            candle::bail!("LFM3 embeddings must use the model's device and F32 dtype")
        }
        Ok(length)
    }

    /// Append a contiguous chunk. On error, the caller's cache remains unchanged.
    pub fn forward_embeds(&self, input: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let length = self.validate_input(input)?;
        if cache.config != self.config || !self.config.is_causal {
            candle::bail!("LFM3 cache configuration does not match this causal backbone")
        }
        let end = cache
            .position
            .checked_add(length)
            .ok_or_else(|| candle::Error::Msg("LFM3 sequence position overflow".into()))?;
        if end > cache.max_length {
            candle::bail!(
                "LFM3 sequence would exceed the caller's cache limit {}",
                cache.max_length
            )
        }
        let mut next = cache.states.clone();
        let mut hidden = input.clone();
        for (layer, state) in self.layers.iter().zip(next.iter_mut()) {
            hidden = layer.forward(&hidden, cache.position, Some(state), &self.config)?;
        }
        let output = self.norm.forward(&hidden)?;
        cache.states = next;
        cache.position = end;
        Ok(output)
    }

    /// Fresh sequence execution, including the bidirectional depth decoder.
    pub fn forward_uncached(&self, input: &Tensor) -> Result<Tensor> {
        let length = self.validate_input(input)?;
        if length > self.config.max_position_embeddings {
            candle::bail!("LFM3 input exceeds max_position_embeddings")
        }
        let mut hidden = input.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, 0, None, &self.config)?;
        }
        self.norm.forward(&hidden)
    }
}

/// Standalone LFM3 text model, also usable as the audio model's text backbone.
#[derive(Debug, Clone)]
pub struct Model {
    pub backbone: Backbone,
    embedding: Embedding,
    lm_head: Linear,
}

impl Model {
    /// `vb` is the direct Lfm3Model root; an untied head must be supplied separately.
    pub fn new(config: &Config, vb: VarBuilder, head: Option<VarBuilder>) -> Result<Self> {
        let backbone = Backbone::new(config, vb.clone())?;
        let embedding =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embedding.embeddings().clone(), None)
        } else {
            let head = head
                .ok_or_else(|| candle::Error::Msg("untied LFM3 requires an output head".into()))?;
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, head)?
        };
        Ok(Self {
            backbone,
            embedding,
            lm_head,
        })
    }

    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embedding.forward(ids)
    }

    pub fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    pub fn forward(&self, ids: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let hidden = self
            .backbone
            .forward_embeds(&self.embed_tokens(ids)?, cache)?;
        self.project_logits(&hidden)
    }
}

#[cfg(test)]
pub(crate) mod tests;
