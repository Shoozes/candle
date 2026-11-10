use super::{routing, Config};
use crate::models::{
    lfm2,
    with_tracing::{linear_no_bias, Embedding, Linear, RmsNorm},
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Cache {
    inner: lfm2::Cache,
    position: usize,
    batch: Option<usize>,
    limit: usize,
    cached: bool,
    layers: usize,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, cfg: &Config, device: &Device) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            inner: lfm2::Cache::new(use_kv_cache, dtype, &cfg.base(), device)?,
            position: 0,
            batch: None,
            limit: cfg.max_position_embeddings,
            cached: use_kv_cache,
            layers: cfg.num_hidden_layers,
        })
    }
    pub fn clear(&mut self) {
        self.inner.clear();
        self.position = 0;
        self.batch = None;
    }
}

#[derive(Debug, Clone)]
struct Sparse {
    gate: Linear,
    bias: Option<Tensor>,
    experts: Vec<lfm2::Mlp>,
    topk: usize,
    normalize: bool,
    scale: f64,
}

impl Sparse {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut expert_cfg = cfg.base();
        expert_cfg.intermediate_size = cfg.moe_intermediate_size;
        let experts = (0..cfg.num_experts)
            .map(|i| lfm2::Mlp::new(&expert_cfg, vb.pp("experts").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?,
            bias: if cfg.use_expert_bias {
                Some(
                    vb.get(cfg.num_experts, "expert_bias")?
                        .to_dtype(DType::F32)?,
                )
            } else {
                None
            },
            experts,
            topk: cfg.num_experts_per_tok,
            normalize: cfg.norm_topk_prob,
            scale: cfg.routed_scaling_factor,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, length, hidden) = x.dims3()?;
        let flat = x.reshape(((), hidden))?;
        let (ids, weights) = routing::route(
            &self.gate.forward(&flat)?,
            self.bias.as_ref(),
            self.topk,
            self.normalize,
            self.scale,
        )?;
        let weights = weights.flatten_all()?;
        let mut output = flat.zeros_like()?;
        for (expert, (tokens, slots)) in self
            .experts
            .iter()
            .zip(routing::groups(&ids, self.experts.len())?)
        {
            if tokens.is_empty() {
                continue;
            }
            let tokens = Tensor::new(tokens.as_slice(), x.device())?;
            let slots = Tensor::new(slots.as_slice(), x.device())?;
            let values = expert.forward(&flat.index_select(&tokens, 0)?)?;
            let weighted = values.broadcast_mul(&weights.index_select(&slots, 0)?.unsqueeze(1)?)?;
            output = output.index_add(&tokens, &weighted, 0)?;
        }
        output.reshape((batch, length, hidden))
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(lfm2::Mlp),
    Sparse(Sparse),
}

#[derive(Debug, Clone)]
struct Layer {
    mixer: lfm2::Mixer,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    feed_forward: FeedForward,
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    head: Linear,
    hidden: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let root = vb.pp("model");
        let base = cfg.base();
        let embeddings = Embedding::new(cfg.vocab_size, cfg.hidden_size, root.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = root.pp("layers").pp(i);
            layers.push(Layer {
                mixer: lfm2::Mixer::new(&base, i, layer.clone())?,
                operator_norm: RmsNorm::new(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    layer.pp("operator_norm"),
                )?,
                ffn_norm: RmsNorm::new(cfg.hidden_size, cfg.norm_eps, layer.pp("ffn_norm"))?,
                feed_forward: if i < cfg.num_dense_layers {
                    FeedForward::Dense(lfm2::Mlp::new(&base, layer.pp("feed_forward"))?)
                } else {
                    FeedForward::Sparse(Sparse::new(cfg, layer.pp("feed_forward"))?)
                },
            });
        }
        let head = if cfg.tie_word_embeddings {
            Linear::from_weights(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embeddings,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.norm_eps, root.pp("embedding_norm"))?,
            head,
            hidden: cfg.hidden_size,
        })
    }
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        ids.dims2()?;
        let vocab = self.embeddings.embeddings().dim(0)?;
        if ids.dtype() != DType::U32
            || ids.elem_count() == 0
            || ids
                .flatten_all()?
                .to_vec1::<u32>()?
                .iter()
                .any(|&id| id as usize >= vocab)
        {
            candle::bail!("LFM2-MoE input requires nonempty U32 token IDs within vocabulary")
        }
        self.embeddings.forward(ids)
    }
    pub fn forward_hidden(
        &self,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let result = (|| {
            let (batch, length, hidden) = embeds.dims3()?;
            let end = index_pos
                .checked_add(length)
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE position overflow".into()))?;
            if batch == 0
                || length == 0
                || cache.layers != self.layers.len()
                || hidden != self.hidden
                || end > cache.limit
                || (index_pos != 0
                    && (!cache.cached || cache.position != index_pos || cache.batch != Some(batch)))
            {
                candle::bail!("LFM2-MoE invalid input shape, context or cache position")
            }
            if index_pos == 0 {
                cache.clear();
            }
            let mut x = embeds.clone();
            for (i, layer) in self.layers.iter().enumerate() {
                x = (&x
                    + layer.mixer.forward(
                        &layer.operator_norm.forward(&x)?,
                        index_pos,
                        i,
                        &mut cache.inner,
                    )?)?;
                let normalized = layer.ffn_norm.forward(&x)?;
                let ff = match &layer.feed_forward {
                    FeedForward::Dense(ff) => ff.forward(&normalized)?,
                    FeedForward::Sparse(ff) => ff.forward(&normalized)?,
                };
                x = (x + ff)?;
            }
            let x = self.norm.forward(&x)?;
            cache.position = end;
            cache.batch = Some(batch);
            Ok(x)
        })();
        if result.is_err() {
            cache.clear();
        }
        result
    }
    pub fn project_logits(&self, hidden: &Tensor, logits_to_keep: usize) -> Result<Tensor> {
        let (_, length, _) = hidden.dims3()?;
        let keep = if logits_to_keep == 0 {
            length
        } else {
            logits_to_keep
        };
        if length == 0 || keep > length {
            candle::bail!("LFM2-MoE invalid logit selection")
        }
        self.head
            .forward(&hidden.narrow(1, length - keep, keep)?)?
            .to_dtype(DType::F32)
    }
    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let result = self
            .forward_hidden(embeds, index_pos, cache)
            .and_then(|hidden| self.project_logits(&hidden, 1))
            .and_then(|logits| logits.i((.., 0, ..))?.contiguous());
        if result.is_err() {
            cache.clear();
        }
        result
    }
    pub fn forward(&self, ids: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let result = (|| {
            let (batch, length) = ids.dims2()?;
            let end = index_pos
                .checked_add(length)
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE position overflow".into()))?;
            if batch == 0 || length == 0 || end > cache.limit {
                candle::bail!("LFM2-MoE input exceeds context before embedding allocation")
            }
            self.forward_embeds(&self.embed_tokens(ids)?, index_pos, cache)
        })();
        if result.is_err() {
            cache.clear();
        }
        result
    }
}
