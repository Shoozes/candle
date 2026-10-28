use super::{Config, LayerState, LayerType};
use candle::{DType, Module, Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

#[derive(Debug, Clone)]
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w1"))?,
            up: candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w3"))?,
            down: candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("w2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}

/// Zero-tail partial RoPE uses the full head width in the frequency exponent.
pub(super) fn rotary(x: &Tensor, position: usize, cfg: &Config, kind: LayerType) -> Result<Tensor> {
    let name = match kind {
        LayerType::FullAttention => "full_attention",
        LayerType::SlidingAttention => "sliding_attention",
        LayerType::Conv => candle::bail!("convolution has no rotary embedding"),
    };
    let p = cfg
        .rope_parameters
        .get(name)
        .ok_or_else(|| candle::Error::Msg(format!("missing LFM3 {name} RoPE")))?;
    let half = cfg.head_dim / 2;
    let rotated = (p.partial_rotary_factor * cfg.head_dim as f64) as usize / 2;
    let frequency = (0..half)
        .map(|i| {
            if i < rotated {
                1f32 / (p.rope_theta as f32).powf((2 * i) as f32 / cfg.head_dim as f32)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let length = x.dim(2)?;
    let end = position
        .checked_add(length)
        .ok_or_else(|| candle::Error::Msg("LFM3 rotary position overflow".into()))?;
    let t = Tensor::arange(position as u32, end as u32, x.device())?
        .to_dtype(DType::F32)?
        .reshape((length, 1))?;
    let f = Tensor::from_vec(frequency, (1, half), x.device())?;
    let angle = t.matmul(&f)?;
    candle_nn::rotary_emb::rope(
        &x.to_dtype(DType::F32)?.contiguous()?,
        &angle.cos()?,
        &angle.sin()?,
    )
}

#[derive(Debug, Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    gate: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    kind: LayerType,
}

impl Attention {
    fn new(c: &Config, kind: LayerType, vb: VarBuilder) -> Result<Self> {
        let linear = |width, name| candle_nn::linear_no_bias(c.hidden_size, width, vb.pp(name));
        Ok(Self {
            q: linear(c.num_attention_heads * c.head_dim, "q_proj")?,
            k: linear(c.num_key_value_heads * c.head_dim, "k_proj")?,
            v: linear(c.num_key_value_heads * c.head_dim, "v_proj")?,
            out: candle_nn::linear_no_bias(
                c.num_attention_heads * c.head_dim,
                c.hidden_size,
                vb.pp("o_proj"),
            )?,
            gate: linear(c.num_attention_heads, "gate_proj")?,
            q_norm: candle_nn::rms_norm(c.head_dim, c.norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(c.head_dim, c.norm_eps, vb.pp("k_norm"))?,
            kind,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        position: usize,
        mut state: Option<&mut LayerState>,
        c: &Config,
    ) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let shape = |heads| (batch, length, heads, c.head_dim);
        let q = self
            .q_norm
            .forward(&self.q.forward(x)?.reshape(shape(c.num_attention_heads))?)?
            .transpose(1, 2)?;
        let k = self
            .k_norm
            .forward(&self.k.forward(x)?.reshape(shape(c.num_key_value_heads))?)?
            .transpose(1, 2)?;
        let q = rotary(&q, position, c, self.kind)?;
        let k = rotary(&k, position, c, self.kind)?;
        let v = self
            .v
            .forward(x)?
            .reshape(shape(c.num_key_value_heads))?
            .transpose(1, 2)?
            .contiguous()?;
        let (k, v) = match state.as_ref().and_then(|s| s.kv.as_ref()) {
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2)?.contiguous()?,
                Tensor::cat(&[pv, &v], 2)?.contiguous()?,
            ),
            None => (k, v),
        };
        let keys = k.dim(2)?;
        let previous = keys - length;
        let start = position
            .checked_sub(previous)
            .ok_or_else(|| candle::Error::Msg("invalid LFM3 cache position".into()))?;
        let sliding = self.kind == LayerType::SlidingAttention;
        let mask = (0..length)
            .flat_map(|i| {
                (0..keys).map(move |j| {
                    let query = position + i;
                    let key = start + j;
                    if (c.is_causal && key > query)
                        || (sliding && key.saturating_add(c.sliding_window) <= query)
                    {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, length, keys), x.device())?;
        let repeat = c.num_attention_heads / c.num_key_value_heads;
        let rk = crate::utils::repeat_kv(k.clone(), repeat)?;
        let rv = crate::utils::repeat_kv(v.clone(), repeat)?.contiguous()?;
        let weights = (q.matmul(&rk.t()?)? / (c.head_dim as f64).sqrt())?.broadcast_add(&mask)?;
        let y = candle_nn::ops::softmax_last_dim(&weights)?
            .matmul(&rv)?
            .transpose(1, 2)?;
        let gate = candle_nn::ops::sigmoid(&self.gate.forward(x)?)?.unsqueeze(3)?;
        let output = self.out.forward(&y.broadcast_mul(&gate)?.reshape((
            batch,
            length,
            c.num_attention_heads * c.head_dim,
        ))?)?;
        if let Some(state) = state.as_mut() {
            let keep = if sliding {
                keys.min(c.sliding_window.saturating_sub(1))
            } else {
                keys
            };
            state.kv = if keep == 0 {
                None
            } else {
                Some((
                    k.narrow(2, keys - keep, keep)?.contiguous()?,
                    v.narrow(2, keys - keep, keep)?.contiguous()?,
                ))
            };
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ShortConv {
    input: Linear,
    output: Linear,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl ShortConv {
    fn new(c: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input: candle_nn::linear_b(
                c.hidden_size,
                3 * c.hidden_size,
                c.conv_bias,
                vb.pp("in_proj"),
            )?,
            output: candle_nn::linear_b(
                c.hidden_size,
                c.hidden_size,
                c.conv_bias,
                vb.pp("out_proj"),
            )?,
            weight: vb.get((c.hidden_size, 1, c.conv_kernel_size), "conv.weight")?,
            bias: if c.conv_bias {
                Some(vb.get(c.hidden_size, "conv.bias")?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, state: Option<&mut LayerState>, c: &Config) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let bcx = self.input.forward(x)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, c.hidden_size)?;
        let gate = bcx.narrow(1, c.hidden_size, c.hidden_size)?;
        let x = (b * bcx.narrow(1, 2 * c.hidden_size, c.hidden_size)?)?;
        let history = c.conv_kernel_size - 1;
        let prefix = match state.as_ref().and_then(|s| s.conv.as_ref()) {
            Some(previous) => previous.clone(),
            None => Tensor::zeros((batch, c.hidden_size, history), DType::F32, x.device())?,
        };
        let joined = Tensor::cat(&[&prefix, &x], 2)?.contiguous()?;
        let mut y = joined.conv1d(&self.weight, 0, 1, 1, c.hidden_size)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.reshape((1, c.hidden_size, 1))?)?;
        }
        let output = self
            .output
            .forward(&(y * gate)?.transpose(1, 2)?.contiguous()?)?;
        if let Some(state) = state {
            state.conv = Some(joined.narrow(2, length, history)?.contiguous()?);
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
enum Operator {
    Attention(Attention),
    Conv(ShortConv),
}

#[derive(Debug, Clone)]
pub(super) struct Layer {
    operator: Operator,
    mlp: Mlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_operator: Option<RmsNorm>,
    post_ffn: Option<RmsNorm>,
}

impl Layer {
    pub(super) fn new(c: &Config, i: usize, vb: VarBuilder) -> Result<Self> {
        let norm = |name| candle_nn::rms_norm(c.hidden_size, c.norm_eps, vb.pp(name));
        let kind = c
            .layer_types
            .get(i)
            .ok_or_else(|| candle::Error::Msg("invalid LFM3 layer index".into()))?;
        Ok(Self {
            operator: match kind {
                LayerType::Conv => Operator::Conv(ShortConv::new(c, vb.pp("conv"))?),
                _ => Operator::Attention(Attention::new(c, *kind, vb.pp("self_attn"))?),
            },
            mlp: Mlp::new(c, vb.pp("feed_forward"))?,
            operator_norm: norm("operator_norm")?,
            ffn_norm: norm("ffn_norm")?,
            post_operator: c
                .use_sandwich_norm
                .then(|| norm("post_operator_norm"))
                .transpose()?,
            post_ffn: c
                .use_sandwich_norm
                .then(|| norm("post_ffn_norm"))
                .transpose()?,
        })
    }

    pub(super) fn forward(
        &self,
        x: &Tensor,
        position: usize,
        state: Option<&mut LayerState>,
        c: &Config,
    ) -> Result<Tensor> {
        let normed = self.operator_norm.forward(x)?;
        let mut y = match &self.operator {
            Operator::Attention(a) => a.forward(&normed, position, state, c)?,
            Operator::Conv(conv) => conv.forward(&normed, state, c)?,
        };
        if let Some(norm) = &self.post_operator {
            y = norm.forward(&y)?;
        }
        let x = (x + y)?;
        let mut y = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        if let Some(norm) = &self.post_ffn {
            y = norm.forward(&y)?;
        }
        x + y
    }
}
