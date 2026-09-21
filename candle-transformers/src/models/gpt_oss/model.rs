use super::GptOssConfig;
use crate::models::gpt_oss::mxfp4::Mxfp4ExpertOperation;
use crate::models::gpt_oss::runtime::{
    cache_bytes_for_tokens, GptOssCancellationToken, GptOssLoadedHandle, GptOssResourceLimits,
    GptOssResourceUsage,
};
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

    #[cfg(feature = "cuda")]
    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
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

    #[cfg(feature = "cuda")]
    pub(crate) fn attention_norm(&self) -> &[f32] {
        &self.attention_norm
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn qkv(&self) -> &DenseLinear {
        &self.qkv
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn attention_out(&self) -> &DenseLinear {
        &self.attention_out
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sinks(&self) -> &[f32] {
        &self.sinks
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn moe_norm(&self) -> &[f32] {
        &self.moe_norm
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn gate(&self) -> &DenseLinear {
        &self.gate
    }

    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn experts(&self) -> &Mxfp4ExpertOperation {
        &self.experts
    }
}

/// Dense non-MoE weights required by the CPU GPT-OSS proof model.
#[derive(Debug, Clone)]
pub struct GptOssWeights {
    token_embedding: Vec<f32>,
    layers: Vec<GptOssLayerWeights>,
    final_norm: Vec<f32>,
    lm_head: DenseLinear,
    load_handle: Option<GptOssLoadedHandle>,
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
            load_handle: None,
        })
    }

    pub(crate) fn attach_load_handle(&mut self, handle: GptOssLoadedHandle) {
        self.load_handle = Some(handle);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn load_handle(&self) -> Option<GptOssLoadedHandle> {
        self.load_handle.clone()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn token_embedding(&self) -> &[f32] {
        &self.token_embedding
    }

    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn layers(&self) -> &[GptOssLayerWeights] {
        &self.layers
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn lm_head(&self) -> &DenseLinear {
        &self.lm_head
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

/// CPU GPT-OSS forward/reference model with explicit KV-cache behavior.
///
/// This is a deterministic proof boundary for synthetic weights.  It accepts
/// packed MXFP4 MoE weights, but it does not load a production checkpoint or
/// provide CUDA kernels. The opt-in CUDA executor is exposed separately when
/// the `cuda` feature is enabled.
#[derive(Debug, Clone)]
pub struct GptOssModel {
    config: GptOssConfig,
    weights: GptOssWeights,
    cache: GptOssCache,
    limits: Option<GptOssResourceLimits>,
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
            limits: None,
        })
    }

    pub fn new_with_limits(
        config: GptOssConfig,
        weights: GptOssWeights,
        limits: GptOssResourceLimits,
    ) -> Result<Self> {
        config.validate()?;
        limits.validate(&config)?;
        if weights.layers.len() != config.num_hidden_layers {
            bail!("GPT-OSS weight layer count does not match config");
        }
        Ok(Self {
            cache: GptOssCache::new(weights.layers.len()),
            config,
            weights,
            limits: Some(limits),
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

    pub fn resource_limits(&self) -> Option<GptOssResourceLimits> {
        self.limits
    }

    pub fn resource_usage(&self) -> Result<GptOssResourceUsage> {
        let cache_bytes = cache_bytes_for_tokens(&self.config, self.cache.tokens)?;
        let mut cache_capacity_bytes = 0usize;
        for layer in &self.cache.layers {
            let layer_capacity = layer
                .keys
                .capacity()
                .checked_add(layer.values.capacity())
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS cache capacity byte count overflowed".to_string())
                })?;
            cache_capacity_bytes = cache_capacity_bytes
                .checked_add(layer_capacity)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS cache capacity byte count overflowed".to_string())
                })?;
        }
        Ok(GptOssResourceUsage {
            sequence_tokens: self.cache.tokens,
            cache_bytes,
            cache_capacity_bytes,
        })
    }

    /// Run the next token span using the model's retained KV cache and return
    /// logits for the final token in the span. A failed forward leaves the
    /// retained cache unchanged.
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_transaction(input_ids, None)
    }

    /// Run a prompt-prefill span with cooperative cancellation and rollback.
    pub fn prefill(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> Result<Vec<f32>> {
        self.forward_transaction(input_ids, Some(cancellation))
    }

    /// Run one cached decode token with cooperative cancellation and rollback.
    pub fn decode(
        &mut self,
        input_id: u32,
        cancellation: &GptOssCancellationToken,
    ) -> Result<Vec<f32>> {
        self.forward_transaction(&[input_id], Some(cancellation))
    }

    /// Run an arbitrary retained span with cooperative cancellation and
    /// all-or-rollback cache semantics.
    pub fn forward_with_cancellation(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> Result<Vec<f32>> {
        self.forward_transaction(input_ids, Some(cancellation))
    }

    fn forward_transaction(
        &mut self,
        input_ids: &[u32],
        cancellation: Option<&GptOssCancellationToken>,
    ) -> Result<Vec<f32>> {
        validate_input_ids(&self.config, input_ids)?;
        if let Some(limits) = self.limits {
            limits.admit(&self.config, self.cache.tokens, input_ids.len())?;
        }
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let (checkpoint_tokens, layer_checkpoints) = self.cache.checkpoint()?;
        let result = Self::forward_with_cache(
            &self.config,
            &self.weights,
            self.limits.as_ref(),
            cancellation,
            input_ids,
            &mut self.cache,
        );
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
        Self::forward_with_cache(
            &self.config,
            &self.weights,
            None,
            None,
            input_ids,
            &mut cache,
        )
    }

    fn forward_with_cache(
        config: &GptOssConfig,
        weights: &GptOssWeights,
        limits: Option<&GptOssResourceLimits>,
        cancellation: Option<&GptOssCancellationToken>,
        input_ids: &[u32],
        cache: &mut GptOssCache,
    ) -> Result<Vec<f32>> {
        validate_input_ids(config, input_ids)?;
        if let Some(limits) = limits {
            limits.admit(config, cache.tokens, input_ids.len())?;
        }
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let mut logits = Vec::new();
        for &token in input_ids {
            if let Some(cancellation) = cancellation {
                cancellation.checkpoint()?;
            }
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
            if let Some(cancellation) = cancellation {
                cancellation.checkpoint()?;
            }
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
        Ok(
            Self::forward_layer_trace(config, layer, layer_index, hidden, cache, position)?
                .post_moe_hidden,
        )
    }

    fn forward_layer_trace(
        config: &GptOssConfig,
        layer: &GptOssLayerWeights,
        layer_index: usize,
        hidden: Vec<f32>,
        cache: &mut GptOssCache,
        position: usize,
    ) -> Result<CpuLayerTrace> {
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
        let post_attention_hidden = add_vectors(&hidden, &attention_update)?;

        let normalized = rms_norm(&post_attention_hidden, &layer.moe_norm, 1e-5)?;
        let gate = layer.gate.forward(&normalized)?;
        let (expert_indices, expert_weights) =
            route_top_k(&gate, config.experts_per_token, config.num_experts)?;
        let expert_contribution =
            layer
                .experts
                .forward_contribution(&normalized, 1, &expert_indices, &expert_weights)?;
        let post_moe_hidden = add_vectors(&post_attention_hidden, &expert_contribution)?;
        Ok(CpuLayerTrace {
            post_attention_hidden,
            expert_contribution,
            post_moe_hidden,
            attention: attended,
            router: gate,
            expert_indices,
            expert_weights,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CpuLayerTrace {
    post_attention_hidden: Vec<f32>,
    expert_contribution: Vec<f32>,
    post_moe_hidden: Vec<f32>,
    attention: Vec<f32>,
    router: Vec<f32>,
    expert_indices: Vec<usize>,
    expert_weights: Vec<f32>,
}

fn validate_input_ids(config: &GptOssConfig, input_ids: &[u32]) -> Result<()> {
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
    Ok(())
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

pub(crate) fn rope_parameters(config: &GptOssConfig, half: usize) -> Result<(f32, Vec<f32>)> {
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
    use crate::models::gpt_oss::cache_bytes_per_token;
    use crate::models::gpt_oss::mxfp4::{
        BYTES_PER_BLOCK, FP4_VALUES, SCALE_BIAS, VALUES_PER_BLOCK,
    };
    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    const ORACLE_BYTES: &[u8] =
        include_bytes!("../../../../tests/fixtures/gpt_oss_task2/oracle.json");
    const ORACLE_SHA256: &str = "db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04";
    const TWO_LAYER_ORACLE_BYTES: &[u8] =
        include_bytes!("../../../../tests/fixtures/gpt_oss_task3_two_layer/oracle.json");
    const TWO_LAYER_ORACLE_SHA256: &str =
        "0530081353cc7268d796ae070c4168ec4e6bb036f62d54670b975a43556bc173";

    #[derive(Debug, Deserialize)]
    struct Oracle {
        fixture_id: String,
        forward: OracleForward,
        packed: OraclePacked,
        provenance: OracleProvenance,
        transitions: OracleTransitions,
    }

    #[derive(Debug, Deserialize)]
    struct OracleForward {
        decode: Vec<u32>,
        decode_logits: Vec<f32>,
        prefill: Vec<u32>,
        prompt: Vec<u32>,
        uncached_logits: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct OraclePacked {
        input: Vec<f32>,
        output: Vec<f32>,
        selected_expert: usize,
        shape: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleProvenance {
        tolerance_abs: f32,
    }

    #[derive(Debug, Deserialize)]
    struct OracleTransitions {
        token_1: OracleTransition,
        token_3_decode: OracleTransition,
    }

    #[derive(Debug, Deserialize)]
    struct OracleTransition {
        attention: Vec<f32>,
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
        router: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracle {
        fixture_id: String,
        forward: TwoLayerOracleForward,
        provenance: OracleProvenance,
        trace: TwoLayerOracleTraceBundle,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleForward {
        prompt: Vec<u32>,
        prefill: Vec<u32>,
        decode: Vec<u32>,
        uncached_logits: Vec<f32>,
        decode_logits: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleTraceBundle {
        token: u32,
        position: usize,
        layers: Vec<TwoLayerOracleLayer>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleLayer {
        sliding: bool,
        attention: Vec<f32>,
        post_attention_hidden: Vec<f32>,
        expert_contribution: Vec<f32>,
        post_moe_hidden: Vec<f32>,
        router: Vec<f32>,
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
    }

    struct ActualTransition {
        attention: Vec<f32>,
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
        router: Vec<f32>,
    }

    fn oracle() -> Result<Oracle> {
        let actual = format!("{:x}", Sha256::digest(ORACLE_BYTES));
        if actual != ORACLE_SHA256 {
            bail!(
                "GPT-OSS Task 2 oracle fixture changed: expected {}, got {}",
                ORACLE_SHA256,
                actual
            );
        }
        serde_json::from_slice(ORACLE_BYTES).map_err(|error| {
            candle::Error::Msg(format!("invalid GPT-OSS Task 2 oracle fixture: {error}"))
        })
    }

    fn two_layer_oracle() -> Result<TwoLayerOracle> {
        let actual = format!("{:x}", Sha256::digest(TWO_LAYER_ORACLE_BYTES));
        if actual != TWO_LAYER_ORACLE_SHA256 {
            bail!(
                "GPT-OSS Task 3 two-layer oracle fixture changed: expected {}, got {}",
                TWO_LAYER_ORACLE_SHA256,
                actual
            );
        }
        serde_json::from_slice(TWO_LAYER_ORACLE_BYTES).map_err(|error| {
            candle::Error::Msg(format!(
                "invalid GPT-OSS Task 3 two-layer oracle fixture: {error}"
            ))
        })
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "oracle mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

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
        packed_with_scale(shape, seed, SCALE_BIAS as u8)
    }

    fn packed_with_scale(
        shape: &[usize],
        seed: u8,
        scale: u8,
    ) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        let elements = shape.iter().product::<usize>();
        let blocks = elements / VALUES_PER_BLOCK;
        let bytes: Vec<u8> = (0..blocks * BYTES_PER_BLOCK)
            .map(|index| seed.wrapping_add(index as u8).rotate_left(1))
            .collect();
        let scales = vec![scale; blocks];
        crate::models::gpt_oss::mxfp4::PackedMxfp4::from_parts(shape, bytes, scales).unwrap()
    }

    fn packed_or_zero(
        shape: &[usize],
        seed: u8,
        zero: bool,
        scale: u8,
    ) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        if zero {
            let elements = shape.iter().product::<usize>();
            let blocks = elements / VALUES_PER_BLOCK;
            crate::models::gpt_oss::mxfp4::PackedMxfp4::from_parts(
                shape,
                vec![0; blocks * BYTES_PER_BLOCK],
                vec![scale; blocks],
            )
            .unwrap()
        } else {
            packed_with_scale(shape, seed, scale)
        }
    }

    fn linear(rows: usize, cols: usize, seed: f32, bias: bool) -> DenseLinear {
        let weights = (0..rows * cols)
            .map(|index| ((index % 11) as f32 - 5.0) * seed)
            .collect();
        let bias = bias.then(|| (0..rows).map(|index| index as f32 * seed).collect());
        DenseLinear::new(rows, cols, weights, bias).unwrap()
    }

    fn model_with_options(
        layer_count: usize,
        zero_experts: bool,
        nonzero_sinks: bool,
    ) -> GptOssModel {
        let mut config = config();
        config.num_hidden_layers = layer_count;
        let qkv_rows =
            config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads);
        let expert_scale = if nonzero_sinks {
            (SCALE_BIAS - 4) as u8
        } else {
            SCALE_BIAS as u8
        };
        let layers = (0..layer_count)
            .map(|layer_index| {
                let layer_offset = layer_index as u8;
                let layer_seed = layer_index as f32;
                let experts = Mxfp4ExpertOperation::new(
                    packed_or_zero(
                        &[
                            config.num_experts,
                            config.intermediate_size * 2,
                            config.hidden_size,
                        ],
                        1u8.wrapping_add(layer_offset.wrapping_mul(17)),
                        zero_experts,
                        expert_scale,
                    ),
                    vec![0.0; config.num_experts * config.intermediate_size * 2],
                    packed_or_zero(
                        &[
                            config.num_experts,
                            config.hidden_size,
                            config.intermediate_size,
                        ],
                        7u8.wrapping_add(layer_offset.wrapping_mul(19)),
                        zero_experts,
                        expert_scale,
                    ),
                    vec![0.0; config.num_experts * config.hidden_size],
                    config.swiglu_limit,
                )
                .unwrap();
                let sinks = if nonzero_sinks {
                    (0..config.num_attention_heads)
                        .map(|head| (head as f32 - 1.5) * 0.125 + layer_seed * 0.25)
                        .collect()
                } else {
                    vec![0.0; config.num_attention_heads]
                };
                GptOssLayerWeights::new(
                    &config,
                    vec![1.0; config.hidden_size],
                    linear(
                        qkv_rows,
                        config.hidden_size,
                        0.0003 + layer_seed * 0.00007,
                        true,
                    ),
                    linear(
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim,
                        0.0002 + layer_seed * 0.00005,
                        true,
                    ),
                    sinks,
                    vec![1.0; config.hidden_size],
                    linear(
                        config.num_experts,
                        config.hidden_size,
                        0.0004 + layer_seed * 0.0003,
                        true,
                    ),
                    experts,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let weights = GptOssWeights::new(
            &config,
            (0..config.vocab_size * config.hidden_size)
                .map(|index| (index % 17) as f32 * 0.001)
                .collect(),
            layers,
            vec![1.0; config.hidden_size],
            linear(config.vocab_size, config.hidden_size, 0.0001, false),
        )
        .unwrap();
        GptOssModel::new(config, weights).unwrap()
    }

    fn model() -> GptOssModel {
        model_with_options(1, false, false)
    }

    fn model_with_limits(limits: GptOssResourceLimits) -> GptOssModel {
        let base = model();
        GptOssModel::new_with_limits(base.config.clone(), base.weights.clone(), limits).unwrap()
    }

    fn trace_token(
        model: &GptOssModel,
        cache: &mut GptOssCache,
        token: u32,
        position: usize,
    ) -> Result<ActualTransition> {
        let token = usize::try_from(token)
            .map_err(|_| candle::Error::Msg("oracle token does not fit usize".to_string()))?;
        let embedding_offset = token
            .checked_mul(model.config.hidden_size)
            .ok_or_else(|| candle::Error::Msg("oracle embedding offset overflowed".to_string()))?;
        let embedding_end = embedding_offset
            .checked_add(model.config.hidden_size)
            .ok_or_else(|| candle::Error::Msg("oracle embedding range overflowed".to_string()))?;
        let hidden = model
            .weights
            .token_embedding
            .get(embedding_offset..embedding_end)
            .ok_or_else(|| candle::Error::Msg("oracle embedding row is out of bounds".to_string()))?
            .to_vec();
        let layer = model
            .weights
            .layers
            .first()
            .ok_or_else(|| candle::Error::Msg("oracle layer is missing".to_string()))?;
        let normalized = rms_norm(&hidden, &layer.attention_norm, 1e-5)?;
        let qkv = layer.qkv.forward(&normalized)?;
        let q_width = model
            .config
            .num_attention_heads
            .checked_mul(model.config.head_dim)
            .ok_or_else(|| candle::Error::Msg("oracle query width overflowed".to_string()))?;
        let kv_width = model
            .config
            .num_key_value_heads
            .checked_mul(model.config.head_dim)
            .ok_or_else(|| candle::Error::Msg("oracle KV width overflowed".to_string()))?;
        let mut query = qkv
            .get(..q_width)
            .ok_or_else(|| candle::Error::Msg("oracle query is out of bounds".to_string()))?
            .to_vec();
        let mut key = qkv
            .get(q_width..q_width + kv_width)
            .ok_or_else(|| candle::Error::Msg("oracle key is out of bounds".to_string()))?
            .to_vec();
        let value_start = q_width
            .checked_add(kv_width)
            .ok_or_else(|| candle::Error::Msg("oracle value offset overflowed".to_string()))?;
        let value = qkv
            .get(value_start..)
            .ok_or_else(|| candle::Error::Msg("oracle value is out of bounds".to_string()))?
            .to_vec();
        apply_rope(
            &mut query,
            model.config.num_attention_heads,
            model.config.head_dim,
            position,
            &model.config,
        )?;
        apply_rope(
            &mut key,
            model.config.num_key_value_heads,
            model.config.head_dim,
            position,
            &model.config,
        )?;
        let layer_cache = cache
            .layers
            .first_mut()
            .ok_or_else(|| candle::Error::Msg("oracle layer cache is missing".to_string()))?;
        layer_cache.keys.extend_from_slice(&key);
        layer_cache.values.extend_from_slice(&value);
        layer_cache.tokens = layer_cache
            .tokens
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("oracle layer cache overflowed".to_string()))?;
        let attention_values = attention(
            &query,
            &layer_cache.keys,
            &layer_cache.values,
            &layer.sinks,
            layer_cache.tokens,
            true,
            &model.config,
        )?;
        let attention_update = layer.attention_out.forward(&attention_values)?;
        let hidden = add_vectors(&hidden, &attention_update)?;
        let normalized = rms_norm(&hidden, &layer.moe_norm, 1e-5)?;
        let router = layer.gate.forward(&normalized)?;
        let (expert_indices, expert_weights) = route_top_k(
            &router,
            model.config.experts_per_token,
            model.config.num_experts,
        )?;
        Ok(ActualTransition {
            attention: attention_values,
            expert_indices,
            expert_weights,
            router,
        })
    }

    fn trace_and_advance(
        model: &GptOssModel,
        cache: &mut GptOssCache,
        token: u32,
        position: usize,
    ) -> Result<ActualTransition> {
        let transition = trace_token(model, cache, token, position)?;
        cache.tokens = cache
            .tokens
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("oracle cache overflowed".to_string()))?;
        Ok(transition)
    }

    fn trace_layer_states(
        model: &GptOssModel,
        cache: &mut GptOssCache,
        token: u32,
        position: usize,
    ) -> Result<Vec<CpuLayerTrace>> {
        let token = usize::try_from(token)
            .map_err(|_| candle::Error::Msg("trace token does not fit usize".to_string()))?;
        let embedding_offset = token
            .checked_mul(model.config.hidden_size)
            .ok_or_else(|| candle::Error::Msg("trace embedding offset overflowed".to_string()))?;
        let embedding_end = embedding_offset
            .checked_add(model.config.hidden_size)
            .ok_or_else(|| candle::Error::Msg("trace embedding range overflowed".to_string()))?;
        let mut hidden = model
            .weights
            .token_embedding
            .get(embedding_offset..embedding_end)
            .ok_or_else(|| candle::Error::Msg("trace embedding row is out of bounds".to_string()))?
            .to_vec();
        let mut traces = Vec::new();
        traces
            .try_reserve_exact(model.weights.layers.len())
            .map_err(|error| candle::Error::Msg(format!("trace allocation failed: {error}")))?;
        for (layer_index, layer) in model.weights.layers.iter().enumerate() {
            let trace = GptOssModel::forward_layer_trace(
                &model.config,
                layer,
                layer_index,
                hidden,
                cache,
                position,
            )?;
            hidden = trace.post_moe_hidden.clone();
            traces.push(trace);
        }
        Ok(traces)
    }

    fn trace_layer_states_and_advance(
        model: &GptOssModel,
        cache: &mut GptOssCache,
        token: u32,
        position: usize,
    ) -> Result<Vec<CpuLayerTrace>> {
        let traces = trace_layer_states(model, cache, token, position)?;
        cache.tokens = cache
            .tokens
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("trace cache overflowed".to_string()))?;
        Ok(traces)
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
    fn task2_independent_oracle_covers_packed_router_attention_and_cache() -> Result<()> {
        let oracle = oracle()?;
        assert_eq!(oracle.fixture_id, "synthetic-gpt-oss-task2-oracle-v1");
        let tolerance = oracle.provenance.tolerance_abs;

        let packed_weight = packed(&oracle.packed.shape, 1);
        let packed_output = packed_weight.matmul_selected(
            &oracle.packed.input,
            1,
            &[oracle.packed.selected_expert],
        )?;
        assert_close(&packed_output, &oracle.packed.output, tolerance);

        let candidate = model();
        let uncached_logits = candidate.forward_uncached(&oracle.forward.prompt)?;
        assert_close(&uncached_logits, &oracle.forward.uncached_logits, tolerance);

        let mut cached = model();
        let cancellation = GptOssCancellationToken::new();
        cached.prefill(&oracle.forward.prefill, &cancellation)?;
        let decode_token = *oracle
            .forward
            .decode
            .first()
            .ok_or_else(|| candle::Error::Msg("oracle decode token is missing".to_string()))?;
        let decode_logits = cached.decode(decode_token, &cancellation)?;
        assert_close(&decode_logits, &oracle.forward.decode_logits, tolerance);
        assert_eq!(cached.cache_len(), 3);

        let mut trace_cache = GptOssCache::new(1);
        let token_1 = trace_and_advance(&candidate, &mut trace_cache, 1, 0)?;
        let _token_2 = trace_and_advance(&candidate, &mut trace_cache, 2, 1)?;
        let token_3 = trace_and_advance(&candidate, &mut trace_cache, 3, 2)?;
        for (actual, expected) in [
            (token_1, oracle.transitions.token_1),
            (token_3, oracle.transitions.token_3_decode),
        ] {
            assert_close(&actual.attention, &expected.attention, tolerance);
            assert_close(&actual.router, &expected.router, tolerance);
            assert_eq!(actual.expert_indices, expected.expert_indices);
            assert_close(&actual.expert_weights, &expected.expert_weights, tolerance);
        }
        Ok(())
    }

    #[test]
    fn task3_two_layer_independent_oracle_matches_cpu_reference_and_traces() -> Result<()> {
        let oracle = two_layer_oracle()?;
        assert_eq!(
            oracle.fixture_id,
            "synthetic-gpt-oss-task3-two-layer-oracle-v1"
        );
        assert_eq!(oracle.trace.token, 4);
        assert_eq!(oracle.trace.position, 3);
        assert_eq!(oracle.trace.layers.len(), 2);
        let tolerance = oracle.provenance.tolerance_abs;

        let uncached = model_with_options(2, false, true);
        let uncached_logits = uncached.forward_uncached(&oracle.forward.prompt)?;
        assert_close(&uncached_logits, &oracle.forward.uncached_logits, tolerance);

        let mut cached = model_with_options(2, false, true);
        cached.prefill(&oracle.forward.prefill, &GptOssCancellationToken::new())?;
        let decode_token = *oracle
            .forward
            .decode
            .first()
            .ok_or_else(|| candle::Error::Msg("oracle decode token is missing".to_string()))?;
        let decode_logits = cached.decode(decode_token, &GptOssCancellationToken::new())?;
        assert_close(&decode_logits, &oracle.forward.decode_logits, tolerance);
        assert_eq!(cached.cache_len(), 4);

        let traced_model = model_with_options(2, false, true);
        let mut trace_cache = GptOssCache::new(2);
        for (position, token) in [1u32, 2, 3].into_iter().enumerate() {
            trace_layer_states_and_advance(&traced_model, &mut trace_cache, token, position)?;
        }
        let actual = trace_layer_states_and_advance(
            &traced_model,
            &mut trace_cache,
            oracle.trace.token,
            oracle.trace.position,
        )?;
        assert_eq!(actual.len(), oracle.trace.layers.len());
        for (layer_index, (actual, expected)) in actual.iter().zip(&oracle.trace.layers).enumerate()
        {
            assert_eq!(expected.sliding, layer_index.is_multiple_of(2));
            assert_close(&actual.attention, &expected.attention, tolerance);
            assert_close(
                &actual.post_attention_hidden,
                &expected.post_attention_hidden,
                tolerance,
            );
            assert_close(
                &actual.expert_contribution,
                &expected.expert_contribution,
                tolerance,
            );
            assert_close(
                &actual.post_moe_hidden,
                &expected.post_moe_hidden,
                tolerance,
            );
            assert_close(&actual.router, &expected.router, tolerance);
            assert_eq!(actual.expert_indices, expected.expert_indices);
            assert_close(&actual.expert_weights, &expected.expert_weights, tolerance);
            assert!(
                actual
                    .expert_weights
                    .windows(2)
                    .any(|weights| { (weights[0] - weights[1]).abs() > tolerance }),
                "layer {layer_index} must exercise unequal routing weights"
            );
        }
        Ok(())
    }

    #[test]
    fn resource_limits_admit_exact_boundary_and_reject_one_over() -> Result<()> {
        let config = config();
        let bytes_per_token = cache_bytes_per_token(&config)?;
        assert_eq!(model().resource_limits(), None);
        let mut exact = model_with_limits(GptOssResourceLimits {
            max_sequence_tokens: 4,
            max_cache_bytes: bytes_per_token * 4,
        });
        let cancellation = GptOssCancellationToken::new();
        exact.prefill(&[1, 2, 3, 4], &cancellation)?;
        assert_eq!(
            exact.resource_usage()?,
            GptOssResourceUsage {
                sequence_tokens: 4,
                cache_bytes: bytes_per_token * 4,
                cache_capacity_bytes: exact.resource_usage()?.cache_capacity_bytes,
            }
        );
        let before = exact.resource_usage()?;
        let error = exact
            .decode(5, &cancellation)
            .expect_err("one token above the exact sequence bound must fail");
        assert!(error.to_string().contains("sequence admission"));
        assert_eq!(
            exact.resource_usage()?.sequence_tokens,
            before.sequence_tokens
        );
        assert_eq!(exact.resource_usage()?.cache_bytes, before.cache_bytes);

        let mut byte_limited = model_with_limits(GptOssResourceLimits {
            max_sequence_tokens: 5,
            max_cache_bytes: bytes_per_token * 4,
        });
        byte_limited.prefill(&[1, 2, 3, 4], &cancellation)?;
        let error = byte_limited
            .decode(5, &cancellation)
            .expect_err("one token above the exact byte bound must fail");
        assert!(error.to_string().contains("KV-cache admission"));
        assert_eq!(byte_limited.cache_len(), 4);

        let mut under = model_with_limits(GptOssResourceLimits {
            max_sequence_tokens: 4,
            max_cache_bytes: bytes_per_token * 4 - 1,
        });
        let error = under
            .prefill(&[1, 2, 3, 4], &cancellation)
            .expect_err("one byte below the exact bound must fail before mutation");
        assert!(error.to_string().contains("KV-cache admission"));
        assert_eq!(under.cache_len(), 0);
        assert_eq!(under.resource_usage()?.cache_bytes, 0);
        Ok(())
    }

    #[test]
    fn cancellation_and_reset_release_owned_cache_buffers() -> Result<()> {
        let mut cancelled_model = model();
        let cancellation = GptOssCancellationToken::cancel_after_checks(3);
        let error = cancelled_model
            .prefill(&[1, 2], &cancellation)
            .expect_err("mid-prefill cancellation must fail");
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(cancelled_model.cache_len(), 0);
        let partial = cancelled_model.resource_usage()?;
        assert_eq!(partial.cache_bytes, 0);
        assert!(partial.cache_capacity_bytes > 0);

        cancelled_model.reset_cache();
        assert_eq!(cancelled_model.resource_usage()?.cache_capacity_bytes, 0);

        let mut decode_model = model();
        let no_cancel = GptOssCancellationToken::new();
        decode_model.prefill(&[1, 2], &no_cancel)?;
        let before = decode_model.resource_usage()?;
        let cancellation = GptOssCancellationToken::new();
        cancellation.cancel();
        let error = decode_model
            .decode(3, &cancellation)
            .expect_err("cancelled decode must fail before mutation");
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(decode_model.resource_usage()?, before);
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
