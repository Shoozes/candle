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

#[cfg(test)]
#[test]
fn short_conv_matches_independent_pytorch_causal_fixture() -> Result<()> {
    let mut cfg = super::tests::tiny_config();
    cfg.hidden_size = 2;
    let device = &candle::Device::Cpu;
    let values = |count: usize, modulus: usize, center: i32, scale: f32| {
        (0..count)
            .map(|i| ((i % modulus) as i32 - center) as f32 / scale)
            .collect::<Vec<_>>()
    };
    let input = Tensor::from_vec(values(18, 13, 6, 8.), (1, 9, 2), device)?;
    let conv = ShortConv {
        input: Linear::new(
            Tensor::from_vec(values(12, 7, 3, 8.), (6, 2), device)?,
            Some(Tensor::from_vec(values(6, 5, 2, 16.), 6, device)?),
        ),
        output: Linear::new(
            Tensor::from_vec(values(4, 5, 2, 4.), (2, 2), device)?,
            Some(Tensor::from_vec(values(2, 3, 1, 16.), 2, device)?),
        ),
        weight: Tensor::from_vec(values(14, 9, 4, 16.), (2, 1, 7), device)?,
        bias: Some(Tensor::from_vec(values(2, 3, 1, 8.), 2, device)?),
    };
    // Independent PyTorch F32: linear -> B*X -> conv1d(padding=6, groups=2)
    // cropped to 9 causal positions -> multiply by C -> output linear.
    let expected = Tensor::from_slice(
        &[
            -0.07621765f32,
            0.,
            -0.07121813,
            1.5497208e-6,
            -0.06631875,
            3.0994415e-6,
            -0.060397863,
            3.5762787e-6,
            -0.053450108,
            2.3841858e-6,
            -0.046143293,
            -1.4543533e-5,
            -0.06932211,
            -0.0007581711,
            -0.07512677,
            -1.8954277e-5,
            -0.06882608,
            -1.7523766e-5,
        ],
        (1, 9, 2),
        device,
    )?;
    let mut state = LayerState::default();
    for (start, len) in [(0, 4), (4, 4), (8, 1)] {
        let actual = conv.forward(&input.narrow(1, start, len)?, Some(&mut state), &cfg)?;
        let error = (actual - expected.narrow(1, start, len)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            error <= 1e-7,
            "ShortConv chunk at {start} differs by {error}"
        );
    }
    Ok(())
}

#[test]
fn attention_matches_independent_pytorch_fixture() -> Result<()> {
    let cfg = super::tests::tiny_config();
    let device = &candle::Device::Cpu;
    let values = |count, stride| {
        (0..count)
            .map(|i: usize| ((i * stride % 19) as i32 - 9) as f32 / 32.)
            .collect::<Vec<_>>()
    };
    let input = Tensor::from_vec(values(40, 7), (1, 5, 8), device)?;
    let linear = |rows, stride| -> Result<Linear> {
        Ok(Linear::new(
            Tensor::from_vec(values(rows * 8, stride), (rows, 8), device)?,
            None,
        ))
    };
    // Independent PyTorch F32 SDPA with explicit masks, zero-tail RoPE for
    // full attention, and frequencies [1, 0.01] for sliding attention.
    let references: [(LayerType, [f32; 40]); 2] = [
        (
            LayerType::SlidingAttention,
            [
                -0.04086664,
                0.03825166,
                0.006516728,
                -0.001854553,
                0.01526780,
                -0.02896234,
                0.001401509,
                0.01756402,
                -0.009917514,
                0.01251601,
                0.006334675,
                -0.01189569,
                -0.02274532,
                -0.006131484,
                0.01964845,
                -0.007925086,
                -0.005299348,
                0.002137860,
                0.001863466,
                0.0005945601,
                -0.03717196,
                -0.01315011,
                0.03181625,
                -0.009145066,
                0.002568781,
                -0.005966877,
                -0.03095308,
                0.03416681,
                -0.03207847,
                0.004180027,
                0.03700288,
                -0.02176503,
                0.004661680,
                -0.006462518,
                -0.03396006,
                0.03522962,
                -0.03156418,
                0.007452118,
                0.03452260,
                -0.02113333,
            ],
        ),
        (
            LayerType::FullAttention,
            [
                -0.04086664,
                0.03825166,
                0.006516728,
                -0.001854553,
                0.01526780,
                -0.02896234,
                0.001401509,
                0.01756402,
                -0.009956034,
                0.01258457,
                0.006362480,
                -0.01196998,
                -0.02265259,
                -0.006097329,
                0.01957633,
                -0.007910017,
                -0.005333938,
                0.002208908,
                0.001896150,
                0.0005043596,
                -0.03707731,
                -0.01310221,
                0.03173824,
                -0.009136304,
                -0.001106646,
                0.008269798,
                -0.01792292,
                0.01112715,
                -0.01623272,
                0.004885013,
                0.01444443,
                -0.009348895,
                0.009607553,
                -0.005236536,
                -0.02777627,
                0.02666149,
                -0.03253706,
                0.003412148,
                0.02466539,
                -0.01230292,
            ],
        ),
    ];
    for (kind, reference) in references {
        let norm_weight = Tensor::ones(4, DType::F32, device)?;
        let attention = Attention {
            q: linear(8, 3)?,
            k: linear(4, 5)?,
            v: linear(4, 7)?,
            out: linear(8, 11)?,
            gate: linear(2, 13)?,
            q_norm: RmsNorm::new(norm_weight.clone(), cfg.norm_eps),
            k_norm: RmsNorm::new(norm_weight, cfg.norm_eps),
            kind,
        };
        let expected = Tensor::from_slice(&reference, (1, 5, 8), device)?;
        let mut state = LayerState::default();
        for (start, len) in [(0, 2), (2, 2), (4, 1)] {
            let actual =
                attention.forward(&input.narrow(1, start, len)?, start, Some(&mut state), &cfg)?;
            let error = (actual - expected.narrow(1, start, len)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                error <= 2e-7,
                "{kind:?} chunk at {start} differs by {error}"
            );
        }
    }
    Ok(())
}
