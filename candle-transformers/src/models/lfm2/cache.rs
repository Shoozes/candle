/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    // KV cache for attention layers: (key, value) per layer
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // Conv state cache for convolution layers
    conv_states: Vec<Option<Tensor>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        config.validate()?;
        let theta = calculate_default_inv_freq(config);
        let theta = Tensor::new(theta, device)?;

        let max_position_embeddings = config.max_position_embeddings as u32;
        let idx_theta = Tensor::arange(0u32, max_position_embeddings, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            conv_states: vec![None; num_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("LFM2 sequence position overflow".into()))?;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear(&mut self) {
        self.masks.clear();
        self.kvs.iter_mut().for_each(|v| *v = None);
        self.conv_states.iter_mut().for_each(|v| *v = None);
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    candle::bail!(
        "LFM2 flash attention was requested, but candle-transformers was built without the 'flash-attn' feature"
    )
}
