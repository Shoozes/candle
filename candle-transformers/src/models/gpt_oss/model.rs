use super::GptOssConfig;
use crate::models::gpt_oss::mxfp4::Mxfp4ExpertOperation;
use candle::{bail, Result};

/// A small dense linear container used by the CPU GPT-OSS reference path.
#[derive(Debug, Clone)]
pub struct DenseLinear {
    rows: usize,
    cols: usize,
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl DenseLinear {
    pub fn new(
        rows: usize,
        cols: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            bail!("GPT-OSS dense linear dimensions must be positive");
        }
        let expected_weights = rows.checked_mul(cols).ok_or_else(|| {
            candle::Error::Msg("GPT-OSS dense linear size overflowed".to_string())
        })?;
        if weights.len() != expected_weights {
            bail!(
                "GPT-OSS dense linear has {} weights, expected {}",
                weights.len(),
                expected_weights
            );
        }
        if let Some(bias) = &bias {
            if bias.len() != rows {
                bail!(
                    "GPT-OSS dense linear has {} bias values, expected {}",
                    bias.len(),
                    rows
                );
            }
        }
        if weights
            .iter()
            .chain(bias.iter().flatten())
            .any(|value| !value.is_finite())
        {
            bail!("GPT-OSS dense linear weights and bias must be finite");
        }
        Ok(Self {
            rows,
            cols,
            weights,
            bias,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.cols {
            bail!(
                "GPT-OSS dense linear input has {} values, expected {}",
                input.len(),
                self.cols
            );
        }
        let mut output = allocate_f32(self.rows)?;
        for row in 0..self.rows {
            let mut sum = 0.0f32;
            let row_offset = row.checked_mul(self.cols).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS linear row offset overflowed".to_string())
            })?;
            for col in 0..self.cols {
                let weight = *self.weights.get(row_offset + col).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS dense linear weight is out of bounds".to_string())
                })?;
                let value = *input.get(col).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS dense linear input is out of bounds".to_string())
                })?;
                sum += weight * value;
            }
            if let Some(bias) = &self.bias {
                sum += *bias.get(row).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS dense linear bias is out of bounds".to_string())
                })?;
            }
            if !sum.is_finite() {
                bail!("GPT-OSS dense linear output is not finite");
            }
            *output.get_mut(row).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS dense linear output is out of bounds".to_string())
            })? = sum;
        }
        Ok(output)
    }
}

/// All weights for one GPT-OSS transformer block.
#[derive(Debug, Clone)]
pub struct GptOssLayerWeights {
    attention_norm: Vec<f32>,
    qkv: DenseLinear,
    attention_out: DenseLinear,
    sinks: Vec<f32>,
    moe_norm: Vec<f32>,
    gate: DenseLinear,
    experts: Mxfp4ExpertOperation,
}

impl GptOssLayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &GptOssConfig,
        attention_norm: Vec<f32>,
        qkv: DenseLinear,
        attention_out: DenseLinear,
        sinks: Vec<f32>,
        moe_norm: Vec<f32>,
        gate: DenseLinear,
        experts: Mxfp4ExpertOperation,
    ) -> Result<Self> {
        if attention_norm.len() != config.hidden_size || moe_norm.len() != config.hidden_size {
            bail!("GPT-OSS layer norm width does not match hidden_size");
        }
        if attention_norm
            .iter()
            .chain(moe_norm.iter())
            .any(|value| !value.is_finite())
        {
            bail!("GPT-OSS layer norm weights must be finite");
        }
        let qkv_heads = config
            .num_key_value_heads
            .checked_mul(2)
            .and_then(|value| config.num_attention_heads.checked_add(value))
            .ok_or_else(|| candle::Error::Msg("GPT-OSS qkv head count overflowed".to_string()))?;
        let qkv_rows = config
            .head_dim
            .checked_mul(qkv_heads)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS qkv width overflowed".to_string()))?;
        if qkv.rows() != qkv_rows || qkv.cols() != config.hidden_size {
            bail!("GPT-OSS qkv linear dimensions do not match config");
        }
        let attention_width = config
            .head_dim
            .checked_mul(config.num_attention_heads)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS attention width overflowed".to_string()))?;
        if attention_out.rows() != config.hidden_size || attention_out.cols() != attention_width {
            bail!("GPT-OSS attention output dimensions do not match config");
        }
        if sinks.len() != config.num_attention_heads || sinks.iter().any(|value| !value.is_finite())
        {
            bail!("GPT-OSS attention sinks do not match config");
        }
        if gate.rows() != config.num_experts || gate.cols() != config.hidden_size {
            bail!("GPT-OSS router dimensions do not match config");
        }
        if experts.expert_count() != config.num_experts
            || experts.hidden_size() != config.hidden_size
            || experts.intermediate_size() != config.intermediate_size
        {
            bail!("GPT-OSS MXFP4 expert dimensions do not match config");
        }
        Ok(Self {
            attention_norm,
            qkv,
            attention_out,
            sinks,
            moe_norm,
            gate,
            experts,
        })
    }
}

/// Dense non-MoE weights required by the CPU GPT-OSS proof model.
#[derive(Debug, Clone)]
pub struct GptOssWeights {
    token_embedding: Vec<f32>,
    layers: Vec<GptOssLayerWeights>,
    final_norm: Vec<f32>,
    lm_head: DenseLinear,
}

impl GptOssWeights {
    pub fn new(
        config: &GptOssConfig,
        token_embedding: Vec<f32>,
        layers: Vec<GptOssLayerWeights>,
        final_norm: Vec<f32>,
        lm_head: DenseLinear,
    ) -> Result<Self> {
        let embedding_len = config
            .vocab_size
            .checked_mul(config.hidden_size)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS embedding size overflowed".to_string()))?;
        if token_embedding.len() != embedding_len {
            bail!(
                "GPT-OSS token embedding has {} values, expected {}",
                token_embedding.len(),
                embedding_len
            );
        }
        if layers.len() != config.num_hidden_layers {
            bail!(
                "GPT-OSS has {} layers, expected {}",
                layers.len(),
                config.num_hidden_layers
            );
        }
        if final_norm.len() != config.hidden_size {
            bail!("GPT-OSS final norm width does not match hidden_size");
        }
        if token_embedding
            .iter()
            .chain(final_norm.iter())
            .any(|value| !value.is_finite())
        {
            bail!("GPT-OSS embedding and final norm weights must be finite");
        }
        if lm_head.rows() != config.vocab_size || lm_head.cols() != config.hidden_size {
            bail!("GPT-OSS lm_head dimensions do not match config");
        }
        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct GptOssCache {
    layers: Vec<LayerCache>,
    tokens: usize,
}

type CacheLayerCheckpoint = (usize, usize, usize);

impl GptOssCache {
    fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![LayerCache::default(); layer_count],
            tokens: 0,
        }
    }

    fn checkpoint(&self) -> Result<(usize, Vec<CacheLayerCheckpoint>)> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.layers.len())
            .map_err(|error| {
                candle::Error::Msg(format!(
                    "GPT-OSS cache checkpoint allocation failed: {error}"
                ))
            })?;
        for layer in &self.layers {
            layers.push((layer.keys.len(), layer.values.len(), layer.tokens));
        }
        Ok((self.tokens, layers))
    }

    fn rollback(
        &mut self,
        tokens: usize,
        layer_checkpoints: &[CacheLayerCheckpoint],
    ) -> Result<()> {
        if layer_checkpoints.len() != self.layers.len() {
            bail!("GPT-OSS cache layer count changed during forward");
        }
        for (layer, &(keys_len, values_len, _)) in self.layers.iter().zip(layer_checkpoints) {
            if keys_len > layer.keys.len() || values_len > layer.values.len() {
                bail!("GPT-OSS cache shrank during forward");
            }
        }
        for (layer, &(keys_len, values_len, layer_tokens)) in
            self.layers.iter_mut().zip(layer_checkpoints)
        {
            layer.keys.truncate(keys_len);
            layer.values.truncate(values_len);
            layer.tokens = layer_tokens;
        }
        self.tokens = tokens;
        Ok(())
    }
}

/// CPU-only GPT-OSS forward/reference model with explicit KV-cache behavior.
///
/// This is a deterministic proof boundary for synthetic weights.  It accepts
/// packed MXFP4 MoE weights, but it does not load a production checkpoint or
/// provide CUDA kernels.
#[derive(Debug, Clone)]
pub struct GptOssModel {
    config: GptOssConfig,
    weights: GptOssWeights,
    cache: GptOssCache,
}

impl GptOssModel {
    pub fn new(config: GptOssConfig, weights: GptOssWeights) -> Result<Self> {
        config.validate()?;
        if weights.layers.len() != config.num_hidden_layers {
            bail!("GPT-OSS weight layer count does not match config");
        }
        Ok(Self {
            cache: GptOssCache::new(weights.layers.len()),
            config,
            weights,
        })
    }

    pub fn reset_cache(&mut self) {
        self.cache = GptOssCache::new(self.weights.layers.len());
    }

    pub fn cache_len(&self) -> usize {
        self.cache.tokens
    }

    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    /// Run the next token span using the model's retained KV cache and return
    /// logits for the final token in the span. A failed forward leaves the
    /// retained cache unchanged.
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let (checkpoint_tokens, layer_checkpoints) = self.cache.checkpoint()?;
        let result =
            Self::forward_with_cache(&self.config, &self.weights, input_ids, &mut self.cache);
        match result {
            Ok(logits) => Ok(logits),
            Err(error) => match self.cache.rollback(checkpoint_tokens, &layer_checkpoints) {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(candle::Error::Msg(format!(
                    "GPT-OSS forward failed: {error}; cache rollback failed: {rollback_error}"
                ))),
            },
        }
    }

    /// Run against a fresh local cache without changing the model's retained
    /// cache.  This is the reference side of the cache-equivalence proof.
    pub fn forward_uncached(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let mut cache = GptOssCache::new(self.weights.layers.len());
        Self::forward_with_cache(&self.config, &self.weights, input_ids, &mut cache)
    }

    fn forward_with_cache(
        config: &GptOssConfig,
        weights: &GptOssWeights,
        input_ids: &[u32],
        cache: &mut GptOssCache,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            bail!("GPT-OSS forward requires at least one input token");
        }
        for (index, &token) in input_ids.iter().enumerate() {
            if token as usize >= config.vocab_size {
                bail!(
                    "GPT-OSS input token {token} at position {index} exceeds vocab size {}",
                    config.vocab_size
                );
            }
        }
        let mut logits = Vec::new();
        for &token in input_ids {
            let token = token as usize;
            let embedding_offset = token.checked_mul(config.hidden_size).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS embedding offset overflowed".to_string())
            })?;
            let embedding_end = embedding_offset
                .checked_add(config.hidden_size)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS embedding range overflowed".to_string())
                })?;
            let mut hidden = weights
                .token_embedding
                .get(embedding_offset..embedding_end)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS embedding row is out of bounds".to_string())
                })?
                .to_vec();
            let position = cache.tokens;
            for (layer_index, layer) in weights.layers.iter().enumerate() {
                hidden = Self::forward_layer(config, layer, layer_index, hidden, cache, position)?;
            }
            let hidden = rms_norm(&hidden, &weights.final_norm, 1e-5)?;
            logits = weights.lm_head.forward(&hidden)?;
            cache.tokens = cache
                .tokens
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS cache length overflowed".to_string()))?;
        }
        Ok(logits)
    }

    fn forward_layer(
        config: &GptOssConfig,
        layer: &GptOssLayerWeights,
        layer_index: usize,
        hidden: Vec<f32>,
        cache: &mut GptOssCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let normalized = rms_norm(&hidden, &layer.attention_norm, 1e-5)?;
        let qkv = layer.qkv.forward(&normalized)?;
        let q_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS query width overflowed".to_string()))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".to_string()))?;
        let expected_qkv = q_width
            .checked_add(
                kv_width
                    .checked_mul(2)
                    .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".to_string()))?,
            )
            .ok_or_else(|| candle::Error::Msg("GPT-OSS qkv width overflowed".to_string()))?;
        if qkv.len() != expected_qkv {
            bail!("GPT-OSS qkv output has the wrong width");
        }
        let mut query = qkv
            .get(..q_width)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS query slice is out of bounds".to_string()))?
            .to_vec();
        let mut key = qkv
            .get(q_width..q_width + kv_width)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS key slice is out of bounds".to_string()))?
            .to_vec();
        let value_start = q_width
            .checked_add(kv_width)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS value offset overflowed".to_string()))?;
        let value = qkv.get(value_start..).ok_or_else(|| {
            candle::Error::Msg("GPT-OSS value slice is out of bounds".to_string())
        })?;
        apply_rope(
            &mut query,
            config.num_attention_heads,
            config.head_dim,
            position,
            config,
        )?;
        apply_rope(
            &mut key,
            config.num_key_value_heads,
            config.head_dim,
            position,
            config,
        )?;

        let layer_cache = cache.layers.get_mut(layer_index).ok_or_else(|| {
            candle::Error::Msg("GPT-OSS layer cache is out of bounds".to_string())
        })?;
        layer_cache
            .keys
            .try_reserve_exact(key.len())
            .map_err(|error| {
                candle::Error::Msg(format!("GPT-OSS KV cache allocation failed: {error}"))
            })?;
        layer_cache
            .values
            .try_reserve_exact(value.len())
            .map_err(|error| {
                candle::Error::Msg(format!("GPT-OSS KV cache allocation failed: {error}"))
            })?;
        layer_cache.keys.extend_from_slice(&key);
        layer_cache.values.extend_from_slice(value);
        layer_cache.tokens = layer_cache.tokens.checked_add(1).ok_or_else(|| {
            candle::Error::Msg("GPT-OSS layer cache length overflowed".to_string())
        })?;

        let attended = attention(
            &query,
            &layer_cache.keys,
            &layer_cache.values,
            &layer.sinks,
            layer_cache.tokens,
            layer_index.is_multiple_of(2),
            config,
        )?;
        let attention_update = layer.attention_out.forward(&attended)?;
        let mut hidden = add_vectors(&hidden, &attention_update)?;

        let normalized = rms_norm(&hidden, &layer.moe_norm, 1e-5)?;
        let gate = layer.gate.forward(&normalized)?;
        let (expert_indices, expert_weights) =
            route_top_k(&gate, config.experts_per_token, config.num_experts)?;
        let expert_output =
            layer
                .experts
                .forward(&normalized, 1, &expert_indices, &expert_weights)?;
        for index in 0..hidden.len() {
            let delta = *expert_output.get(index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS expert output is out of bounds".to_string())
            })? - *normalized.get(index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS normalized output is out of bounds".to_string())
            })?;
            *hidden.get_mut(index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS hidden output is out of bounds".to_string())
            })? += delta;
        }
        Ok(hidden)
    }
}

fn route_top_k(gate: &[f32], top_k: usize, expert_count: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    if gate.len() != expert_count || top_k == 0 || top_k > expert_count {
        bail!("GPT-OSS router shape is incompatible with top-k configuration");
    }
    if gate.iter().any(|value| !value.is_finite()) {
        bail!("GPT-OSS router output is not finite");
    }
    let mut order: Vec<usize> = (0..expert_count).collect();
    order.sort_by(|left, right| match gate[*right].partial_cmp(&gate[*left]) {
        Some(ordering) => ordering.then_with(|| left.cmp(right)),
        None => std::cmp::Ordering::Equal,
    });
    order.truncate(top_k);
    let max = order
        .iter()
        .filter_map(|index| gate.get(*index).copied())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::new();
    weights.try_reserve_exact(top_k).map_err(|error| {
        candle::Error::Msg(format!("GPT-OSS router allocation failed: {error}"))
    })?;
    let mut denominator = 0.0f32;
    for &index in &order {
        let value = (gate[index] - max).exp();
        denominator += value;
        weights.push(value);
    }
    if !denominator.is_finite() || denominator == 0.0 {
        bail!("GPT-OSS router softmax is not finite");
    }
    for weight in &mut weights {
        *weight /= denominator;
    }
    Ok((order, weights))
}

fn attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sinks: &[f32],
    key_tokens: usize,
    sliding: bool,
    config: &GptOssConfig,
) -> Result<Vec<f32>> {
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let q_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS attention width overflowed".to_string()))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".to_string()))?;
    let expected_kv_values = key_tokens
        .checked_mul(kv_width)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV cache size overflowed".to_string()))?;
    if query.len() != q_width || keys.len() != expected_kv_values || values.len() != keys.len() {
        bail!("GPT-OSS attention cache shapes are inconsistent");
    }
    if sinks.len() != heads {
        bail!("GPT-OSS sink count does not match attention heads");
    }
    let q_per_kv = heads / kv_heads;
    let window_start = if sliding {
        key_tokens.saturating_sub(config.sliding_window)
    } else {
        0
    };
    let mut output = allocate_f32(q_width)?;
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..heads {
        let kv_head = head / q_per_kv;
        let q = query
            .get(head * head_dim..(head + 1) * head_dim)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS query head is out of bounds".to_string()))?;
        let mut scores = Vec::new();
        let score_count = key_tokens
            .checked_sub(window_start)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| candle::Error::Msg("GPT-OSS score count overflowed".to_string()))?;
        scores.try_reserve_exact(score_count).map_err(|error| {
            candle::Error::Msg(format!("GPT-OSS attention allocation failed: {error}"))
        })?;
        for token in window_start..key_tokens {
            let key_offset = token
                .checked_mul(kv_width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| candle::Error::Msg("GPT-OSS key offset overflowed".to_string()))?;
            let key_end = key_offset
                .checked_add(head_dim)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS key range overflowed".to_string()))?;
            let key = keys.get(key_offset..key_end).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS key head is out of bounds".to_string())
            })?;
            let dot = q.iter().zip(key.iter()).map(|(q, k)| q * k).sum::<f32>() * scale;
            if !dot.is_finite() {
                bail!("GPT-OSS attention score is not finite");
            }
            scores.push(dot);
        }
        scores.push(
            *sinks
                .get(head)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS sink is out of bounds".to_string()))?,
        );
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let denominator = exp_scores.iter().sum::<f32>();
        if !denominator.is_finite() || denominator == 0.0 {
            bail!("GPT-OSS attention softmax is not finite");
        }
        let output_head = output
            .get_mut(head * head_dim..(head + 1) * head_dim)
            .ok_or_else(|| {
                candle::Error::Msg("GPT-OSS attention output is out of bounds".to_string())
            })?;
        for (score_index, token) in (window_start..key_tokens).enumerate() {
            let value_offset = token
                .checked_mul(kv_width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| candle::Error::Msg("GPT-OSS value offset overflowed".to_string()))?;
            let value_end = value_offset
                .checked_add(head_dim)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS value range overflowed".to_string()))?;
            let value = values.get(value_offset..value_end).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS value head is out of bounds".to_string())
            })?;
            let weight = exp_scores.get(score_index).copied().ok_or_else(|| {
                candle::Error::Msg("GPT-OSS attention weight is out of bounds".to_string())
            })? / denominator;
            for (out, value) in output_head.iter_mut().zip(value.iter()) {
                *out += weight * value;
            }
        }
    }
    Ok(output)
}

fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    config: &GptOssConfig,
) -> Result<()> {
    let expected_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS rotary width overflowed".to_string()))?;
    if values.len() != expected_width || !head_dim.is_multiple_of(2) {
        bail!("GPT-OSS rotary input shape is invalid");
    }
    let half = head_dim / 2;
    let (concentration, inv_freq) = rope_parameters(config, half)?;
    for head in 0..heads {
        for (index, &inv_frequency) in inv_freq.iter().enumerate() {
            let angle = position as f32 * inv_frequency;
            let cos = angle.cos() * concentration;
            let sin = angle.sin() * concentration;
            let first_index = head * head_dim + index;
            let second_index = first_index + half;
            let first = *values.get(first_index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS rotary first half is out of bounds".to_string())
            })?;
            let second = *values.get(second_index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS rotary second half is out of bounds".to_string())
            })?;
            *values.get_mut(first_index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS rotary output is out of bounds".to_string())
            })? = first * cos - second * sin;
            *values.get_mut(second_index).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS rotary output is out of bounds".to_string())
            })? = second * cos + first * sin;
        }
    }
    Ok(())
}

fn rope_parameters(config: &GptOssConfig, half: usize) -> Result<(f32, Vec<f32>)> {
    let concentration = 0.1 * config.rope_scaling_factor.ln() + 1.0;
    let mut inv_freq = Vec::new();
    inv_freq
        .try_reserve_exact(half)
        .map_err(|error| candle::Error::Msg(format!("GPT-OSS RoPE allocation failed: {error}")))?;
    let d_half = half as f32;
    let (low, high) = if config.rope_scaling_factor > 1.0 {
        let log_base = config.rope_theta.ln();
        (
            d_half
                * (config.initial_context_length as f32
                    / (config.rope_ntk_beta * 2.0 * std::f32::consts::PI))
                    .ln()
                / log_base,
            d_half
                * (config.initial_context_length as f32
                    / (config.rope_ntk_alpha * 2.0 * std::f32::consts::PI))
                    .ln()
                / log_base,
        )
    } else {
        (0.0, 0.0)
    };
    for index in 0..half {
        let frequency = config
            .rope_theta
            .powf((2 * index) as f32 / (2 * half) as f32);
        let extrapolation = frequency.recip();
        let value = if config.rope_scaling_factor > 1.0 && high > low {
            let interpolation = extrapolation / config.rope_scaling_factor;
            let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            interpolation * ramp + extrapolation * (1.0 - ramp)
        } else {
            extrapolation
        };
        inv_freq.push(value);
    }
    Ok((concentration, inv_freq))
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len() || input.is_empty() {
        bail!("GPT-OSS RMS norm input and weight widths do not match");
    }
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean + epsilon).sqrt().recip();
    let mut output = allocate_f32(input.len())?;
    for index in 0..input.len() {
        output[index] = input[index] * scale * weight[index];
    }
    Ok(output)
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        bail!("GPT-OSS residual widths do not match");
    }
    let mut output = allocate_f32(left.len())?;
    for index in 0..left.len() {
        output[index] = left[index] + right[index];
    }
    Ok(output)
}

fn allocate_f32(length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| candle::Error::Msg(format!("GPT-OSS allocation failed: {error}")))?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpt_oss::mxfp4::{
        BYTES_PER_BLOCK, FP4_VALUES, SCALE_BIAS, VALUES_PER_BLOCK,
    };

    fn config() -> GptOssConfig {
        GptOssConfig {
            model_type: "gpt_oss".to_string(),
            num_hidden_layers: 1,
            num_experts: 2,
            experts_per_token: 2,
            vocab_size: 8,
            hidden_size: 32,
            intermediate_size: 32,
            swiglu_limit: 7.0,
            head_dim: 8,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            sliding_window: 3,
            initial_context_length: 8,
            rope_theta: 10_000.0,
            rope_scaling_factor: 1.0,
            rope_ntk_alpha: 1.0,
            rope_ntk_beta: 32.0,
            architectures: vec!["GptOssForCausalLM".to_string()],
        }
    }

    fn packed(shape: &[usize], seed: u8) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        let elements = shape.iter().product::<usize>();
        let blocks = elements / VALUES_PER_BLOCK;
        let bytes: Vec<u8> = (0..blocks * BYTES_PER_BLOCK)
            .map(|index| seed.wrapping_add(index as u8).rotate_left(1))
            .collect();
        let scales = vec![SCALE_BIAS as u8; blocks];
        crate::models::gpt_oss::mxfp4::PackedMxfp4::from_parts(shape, bytes, scales).unwrap()
    }

    fn linear(rows: usize, cols: usize, seed: f32, bias: bool) -> DenseLinear {
        let weights = (0..rows * cols)
            .map(|index| ((index % 11) as f32 - 5.0) * seed)
            .collect();
        let bias = bias.then(|| (0..rows).map(|index| index as f32 * seed).collect());
        DenseLinear::new(rows, cols, weights, bias).unwrap()
    }

    fn model() -> GptOssModel {
        let config = config();
        let qkv_rows =
            config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads);
        let experts = Mxfp4ExpertOperation::new(
            packed(
                &[
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ],
                1,
            ),
            vec![0.0; config.num_experts * config.intermediate_size * 2],
            packed(
                &[
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size,
                ],
                7,
            ),
            vec![0.0; config.num_experts * config.hidden_size],
            config.swiglu_limit,
        )
        .unwrap();
        let layer = GptOssLayerWeights::new(
            &config,
            vec![1.0; config.hidden_size],
            linear(qkv_rows, config.hidden_size, 0.0003, true),
            linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                0.0002,
                true,
            ),
            vec![0.0; config.num_attention_heads],
            vec![1.0; config.hidden_size],
            linear(config.num_experts, config.hidden_size, 0.0004, true),
            experts,
        )
        .unwrap();
        let weights = GptOssWeights::new(
            &config,
            (0..config.vocab_size * config.hidden_size)
                .map(|index| (index % 17) as f32 * 0.001)
                .collect(),
            vec![layer],
            vec![1.0; config.hidden_size],
            linear(config.vocab_size, config.hidden_size, 0.0001, false),
        )
        .unwrap();
        GptOssModel::new(config, weights).unwrap()
    }

    #[test]
    fn cached_forward_matches_uncached_last_token_and_reset() -> Result<()> {
        let mut model = model();
        let prompt = [1, 2, 3, 4];
        let expected = model.forward_uncached(&prompt)?;
        let first = model.forward(&prompt[..3])?;
        let actual = model.forward(&prompt[3..])?;
        assert_eq!(first.len(), 8);
        assert_eq!(model.cache_len(), 4);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        model.reset_cache();
        assert_eq!(model.cache_len(), 0);
        let replay = model.forward(&prompt)?;
        for (actual, expected) in replay.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn rejects_empty_and_out_of_range_forward_inputs() {
        let mut model = model();
        assert!(model.forward(&[]).is_err());
        let error = model.forward(&[8]).expect_err("token id must be bounded");
        assert!(error.to_string().contains("vocab size"));
        assert_eq!(model.cache_len(), 0);
    }

    #[test]
    fn rotary_and_router_are_deterministic() -> Result<()> {
        let config = config();
        let mut values = vec![1.0; config.head_dim];
        apply_rope(&mut values, 1, config.head_dim, 0, &config)?;
        assert_eq!(values, vec![1.0; config.head_dim]);
        let (indices, weights) = route_top_k(&[1.0, 2.0], 2, 2)?;
        assert_eq!(indices, vec![1, 0]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() <= 1e-6);
        assert!(FP4_VALUES.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn yarn_scaling_uses_interpolation_ramp_edges() -> Result<()> {
        let mut config = config();
        config.head_dim = 64;
        config.initial_context_length = 4_096;
        config.rope_theta = 150_000.0;
        config.rope_scaling_factor = 32.0;
        config.rope_ntk_alpha = 1.0;
        config.rope_ntk_beta = 32.0;

        let (_, inv_freq) = rope_parameters(&config, 32)?;
        assert_eq!(inv_freq.len(), 32);
        assert!((inv_freq[0] - 1.0).abs() < 1e-6, "{}", inv_freq[0]);
        assert!((inv_freq[9] - 0.031705696).abs() < 1e-7, "{}", inv_freq[9]);
        assert!(
            (inv_freq[31] - 3.0235114e-7).abs() < 1e-12,
            "{}",
            inv_freq[31]
        );
        Ok(())
    }

    #[test]
    fn failed_forward_does_not_mutate_retained_cache() -> Result<()> {
        let mut reference = model();
        let mut candidate = model();
        candidate.forward(&[1])?;
        reference.forward(&[1])?;
        let before = candidate.cache.clone();
        let before_capacities: Vec<_> = candidate
            .cache
            .layers
            .iter()
            .map(|layer| (layer.keys.capacity(), layer.values.capacity()))
            .collect();

        let original_norm = candidate.weights.final_norm[0];
        let original_lm_head = candidate.weights.lm_head.weights[0];
        candidate.weights.final_norm[0] = f32::MAX;
        candidate.weights.lm_head.weights[0] = f32::MAX;
        let error = candidate
            .forward(&[2])
            .expect_err("finite extreme norm must fail after cache append");
        assert!(error.to_string().contains("not finite"), "{error}");
        assert_eq!(candidate.cache, before);
        for (layer, (keys_capacity, values_capacity)) in
            candidate.cache.layers.iter().zip(before_capacities)
        {
            assert!(layer.keys.capacity() >= keys_capacity);
            assert!(layer.values.capacity() >= values_capacity);
        }

        candidate.weights.final_norm[0] = original_norm;
        candidate.weights.lm_head.weights[0] = original_lm_head;
        let actual = candidate.forward(&[2])?;
        let expected = reference.forward(&[2])?;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }
}
