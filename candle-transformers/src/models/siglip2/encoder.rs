#[derive(Clone, Debug)]
struct Attention {
    q_proj: LinearOp,
    k_proj: LinearOp,
    v_proj: LinearOp,
    out_proj: LinearOp,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .ok_or_else(|| candle::Error::Msg("SigLIP2 attention head dimension is zero".into()))?;
        Ok(Self {
            q_proj: LinearOp::Dense(linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("q_proj"),
            )?),
            k_proj: LinearOp::Dense(linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("k_proj"),
            )?),
            v_proj: LinearOp::Dense(linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("v_proj"),
            )?),
            out_proj: LinearOp::Dense(linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("out_proj"),
            )?),
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn new_with_quantized_linears(
        config: &Siglip2VisionConfig,
        vb: VarBuilder,
        quantized_weights: &mut HashMap<String, QTensor>,
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .ok_or_else(|| candle::Error::Msg("SigLIP2 attention head dimension is zero".into()))?;
        Ok(Self {
            q_proj: mixed_linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("q_proj"),
                quantized_weights,
                &format!("{prefix}.q_proj.weight"),
            )?,
            k_proj: mixed_linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("k_proj"),
                quantized_weights,
                &format!("{prefix}.k_proj.weight"),
            )?,
            v_proj: mixed_linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("v_proj"),
                quantized_weights,
                &format!("{prefix}.v_proj.weight"),
            )?,
            out_proj: mixed_linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("out_proj"),
                quantized_weights,
                &format!("{prefix}.out_proj.weight"),
            )?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, _) = xs.dims3()?;
        let query = self.split_heads(&self.q_proj.forward(xs)?, batch_size, sequence_length)?;
        let key = self.split_heads(&self.k_proj.forward(xs)?, batch_size, sequence_length)?;
        let value = self.split_heads(&self.v_proj.forward(xs)?, batch_size, sequence_length)?;

        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let scores = (query_f32.matmul(&key_f32.t()?)? / (self.head_dim as f64).sqrt())?;
        let mask = mask
            .reshape((batch_size, 1, 1, sequence_length))?
            .broadcast_as((batch_size, self.num_heads, sequence_length, sequence_length))?;
        let valid = mask.gt(0f32)?;
        let neg_inf =
            Tensor::new(f32::MIN, scores.device())?.broadcast_as(scores.shape().clone())?;
        let scores = valid.where_cond(&scores, &neg_inf)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(query.dtype())?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj.forward(&output.to_dtype(xs.dtype())?)
    }

    fn split_heads(
        &self,
        xs: &Tensor,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<Tensor> {
        xs.reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    fc1: LinearOp,
    fc2: LinearOp,
    activation: Activation,
}

impl Mlp {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: LinearOp::Dense(linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("fc1"),
            )?),
            fc2: LinearOp::Dense(linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("fc2"),
            )?),
            activation: config.hidden_act,
        })
    }

    fn new_with_quantized_linears(
        config: &Siglip2VisionConfig,
        vb: VarBuilder,
        quantized_weights: &mut HashMap<String, QTensor>,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            fc1: mixed_linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("fc1"),
                quantized_weights,
                &format!("{prefix}.fc1.weight"),
            )?,
            fc2: mixed_linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("fc2"),
                quantized_weights,
                &format!("{prefix}.fc2.weight"),
            )?,
            activation: config.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)
    }
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps,
            ..LayerNormConfig::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm1"))?,
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn new_with_quantized_linears(
        config: &Siglip2VisionConfig,
        vb: VarBuilder,
        quantized_weights: &mut HashMap<String, QTensor>,
        index: usize,
    ) -> Result<Self> {
        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps,
            ..LayerNormConfig::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm1"))?,
            self_attn: Attention::new_with_quantized_linears(
                config,
                vb.pp("self_attn"),
                quantized_weights,
                &format!("encoder.layers.{index}.self_attn"),
            )?,
            layer_norm2: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm2"))?,
            mlp: Mlp::new_with_quantized_linears(
                config,
                vb.pp("mlp"),
                quantized_weights,
                &format!("encoder.layers.{index}.mlp"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let attended = self
            .self_attn
            .forward(&xs.apply(&self.layer_norm1)?, mask)?;
        let xs = (residual + attended)?;
        let residual = &xs;
        let feed_forward = self.mlp.forward(&xs.apply(&self.layer_norm2)?)?;
        residual + feed_forward
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct ForwardStages {
    pub(crate) embeddings: EmbeddingStages,
    pub(crate) encoder_layers: Vec<Tensor>,
    pub(crate) post_layernorm: Tensor,
}
