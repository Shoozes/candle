/// Internal feature-only body for the released LFM3-Audio encoder. It shares
/// LFM2 model math while loading neither token embeddings nor an unused head.
#[derive(Debug, Clone)]
pub(crate) struct FeatureEncoder {
    config: Config,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    window: usize,
    device: Device,
}

impl FeatureEncoder {
    pub(crate) fn new(config: &Config, window: usize, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        if window == 0 || vb.dtype() != DType::F32 {
            candle::bail!("LFM2 feature encoder requires a positive attention window and F32")
        }
        let layers = (0..config.num_hidden_layers)
            .map(|i| DecoderLayer::new(config, i, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            layers,
            norm: RmsNorm::new(config.hidden_size, config.norm_eps, vb.pp("embedding_norm"))?,
            window,
            device: vb.device().clone(),
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, length, hidden) = input.dims3()?;
        if batch != 1
            || length == 0
            || hidden != self.config.hidden_size
            || length > self.config.max_position_embeddings
            || input.dtype() != DType::F32
        {
            candle::bail!("invalid LFM2 feature encoder input shape, dtype or length")
        }
        let mut bounded = self.config.clone();
        bounded.max_position_embeddings = length;
        let mut cache = Cache::new(false, DType::F32, &bounded, &self.device)?
            .with_attention_window(self.window)?;
        let mut hidden = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, 0, i, &mut cache)?;
        }
        self.norm.forward(&hidden)
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_from_parts(cfg, vb.pp("model"), Some(vb.pp("lm_head")))
    }

    /// Construct from the direct language-model variable root.
    ///
    /// `Model::new` is the standalone loader for checkpoints rooted at
    /// `model.*`. Nested multimodal checkpoints should pass
    /// `model.language_model` here and provide the separate `lm_head` root
    /// only when embeddings are not tied.
    pub fn new_from_parts(
        cfg: &Config,
        vb_m: VarBuilder,
        lm_head_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        cfg.validate()?;
        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            let lm_head_vb = match lm_head_vb {
                Some(lm_head_vb) => lm_head_vb,
                None => candle::bail!("untied LFM2 configuration requires an lm_head root"),
            };
            linear(cfg.hidden_size, cfg.vocab_size, lm_head_vb)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            dtype: vb_m.dtype(),
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn device(&self) -> &Device {
        self.embed_tokens.embeddings().device()
    }

    pub fn forward_hidden(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embeds.dims3()?;
        if seq_len == 0 {
            candle::bail!("LFM2 cannot forward an empty sequence")
        }
        let end_pos = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("LFM2 sequence position overflow".into()))?;
        let max_position_embeddings = cache.cos.dim(0)?;
        if end_pos > max_position_embeddings {
            candle::bail!(
                "LFM2 sequence positions [{index_pos}, {end_pos}) exceed max_position_embeddings {max_position_embeddings}"
            )
        }

        let mut hidden_states = input_embeds.clone();
        for (block_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }
        self.embedding_norm.forward(&hidden_states)
    }

    pub fn project_logits(&self, hidden_states: &Tensor, logits_to_keep: usize) -> Result<Tensor> {
        let (_, seq_len, _) = hidden_states.dims3()?;
        if seq_len == 0 {
            candle::bail!("LFM2 cannot project logits for an empty sequence")
        }
        let keep = if logits_to_keep == 0 {
            seq_len
        } else {
            logits_to_keep
        };
        if keep > seq_len {
            candle::bail!("cannot keep {keep} LFM2 logits from a sequence of length {seq_len}")
        }
        let hidden_states = hidden_states.narrow(1, seq_len - keep, keep)?;
        self.lm_head.forward(&hidden_states)?.to_dtype(DType::F32)
    }

    pub fn forward_embeds(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let hidden_states = self.forward_hidden(input_embeds, index_pos, cache)?;
        let logits = self.project_logits(&hidden_states, 1)?;
        logits.i((.., 0, ..))?.contiguous()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        input_ids.dims2()?;
        let input_embeds = self.embed_tokens(input_ids)?;
        self.forward_embeds(&input_embeds, index_pos, cache)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
