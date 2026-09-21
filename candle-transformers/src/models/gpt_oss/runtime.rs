//! Bounded CPU proof contracts for GPT-OSS loading and cached execution.
//!
//! These contracts do not construct a production model or start worker
//! processes.  They make sequence/cache admission, cancellation, and
//! temporary load ownership explicit for the synthetic CPU proof boundary.

use super::GptOssConfig;
use candle::{bail, Result};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const F32_BYTES: usize = std::mem::size_of::<f32>();
const NO_SCHEDULED_CANCELLATION: usize = usize::MAX;

/// Product admission ceiling used by the GPT-OSS Edge qualification runner.
pub const EDGE_DEVICE_CEILING_BYTES: usize = 20_000_000_000;

/// Exact logical KV-cache bytes required for one retained token.
pub fn cache_bytes_per_token(config: &GptOssConfig) -> Result<usize> {
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".to_string()))?;
    config
        .num_hidden_layers
        .checked_mul(kv_width)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV-cache byte count overflowed".to_string()))
}

/// Exact logical KV-cache bytes required for `tokens` retained tokens.
pub fn cache_bytes_for_tokens(config: &GptOssConfig, tokens: usize) -> Result<usize> {
    cache_bytes_per_token(config)?
        .checked_mul(tokens)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV-cache byte count overflowed".to_string()))
}

/// Checked upper bound for transient F32 CUDA workspace at one retained
/// sequence length.  The bound covers the overlapping attention score and
/// value intermediates plus QKV, router, hidden, and top-expert buffers for
/// the peak layer.  Layers execute sequentially and reuse this workspace;
/// retained keys and values are accounted for separately by the KV cache.
pub fn workspace_bytes_for_tokens(config: &GptOssConfig, tokens: usize) -> Result<usize> {
    let query_width = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS query width overflowed".to_string()))?;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".to_string()))?;
    let qkv_width = query_width
        .checked_add(
            kv_width
                .checked_mul(2)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV width overflowed".to_string()))?,
        )
        .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV width overflowed".to_string()))?;
    let attention_scores = config
        .num_attention_heads
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS attention workspace overflowed".to_string()))?;
    let attention_values = config
        .num_attention_heads
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(config.head_dim))
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS attention workspace overflowed".to_string()))?;
    let qkv = qkv_width
        .checked_mul(F32_BYTES)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV workspace overflowed".to_string()))?;
    let hidden = config
        .hidden_size
        .checked_mul(F32_BYTES)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS hidden workspace overflowed".to_string()))?;
    let router = config
        .num_experts
        .checked_mul(F32_BYTES)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS router workspace overflowed".to_string()))?;
    let expert = config
        .experts_per_token
        .checked_mul(config.intermediate_size)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS expert workspace overflowed".to_string()))?;
    attention_scores
        .checked_mul(2)
        .and_then(|value| value.checked_add(attention_values.checked_mul(8)?))
        .and_then(|value| value.checked_add(qkv.checked_mul(2)?))
        .and_then(|value| value.checked_add(hidden.checked_mul(8)?))
        .and_then(|value| value.checked_add(router.checked_mul(4)?))
        .and_then(|value| value.checked_add(expert.checked_mul(4)?))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS workspace byte count overflowed".to_string()))
}

/// Checked static-plus-cache-plus-workspace admission accounting.
pub fn total_device_bytes_for_tokens(
    config: &GptOssConfig,
    static_resident_bytes: usize,
    tokens: usize,
) -> Result<usize> {
    let cache_bytes = cache_bytes_for_tokens(config, tokens)?;
    let workspace_bytes = workspace_bytes_for_tokens(config, tokens)?;
    static_resident_bytes
        .checked_add(cache_bytes)
        .and_then(|value| value.checked_add(workspace_bytes))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS total device byte count overflowed".to_string()))
}

/// Find the greatest sequence length whose checked device accounting fits the
/// supplied budget.  This is used for admission evidence, not as a runtime
/// replacement for the fail-closed per-forward check.
pub fn max_supported_sequence_tokens(
    config: &GptOssConfig,
    static_resident_bytes: usize,
    max_sequence_tokens: usize,
    max_total_device_bytes: usize,
) -> Result<usize> {
    let mut low = 0usize;
    let mut high = max_sequence_tokens;
    while low < high {
        let span = high - low;
        let upper_half = span / 2 + span % 2;
        let mid = low
            .checked_add(upper_half)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS admission search overflowed".to_string()))?;
        if total_device_bytes_for_tokens(config, static_resident_bytes, mid)?
            <= max_total_device_bytes
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

/// Admission limits for one synthetic GPT-OSS execution session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptOssResourceLimits {
    pub max_sequence_tokens: usize,
    pub max_cache_bytes: usize,
    pub max_total_device_bytes: usize,
}

impl GptOssResourceLimits {
    /// Derive a checked default from the normalized model configuration.
    pub fn for_config(config: &GptOssConfig) -> Result<Self> {
        let max_sequence_tokens = config.initial_context_length;
        let max_cache_bytes = cache_bytes_for_tokens(config, max_sequence_tokens)?;
        let limits = Self {
            max_sequence_tokens,
            max_cache_bytes,
            max_total_device_bytes: usize::MAX,
        };
        limits.validate(config)?;
        Ok(limits)
    }

    pub fn validate(&self, config: &GptOssConfig) -> Result<()> {
        if self.max_sequence_tokens == 0 {
            bail!("GPT-OSS max_sequence_tokens must be greater than zero");
        }
        if self.max_cache_bytes == 0 {
            bail!("GPT-OSS max_cache_bytes must be greater than zero");
        }
        if self.max_total_device_bytes == 0 {
            bail!("GPT-OSS max_total_device_bytes must be greater than zero");
        }
        let one_token = cache_bytes_per_token(config)?;
        if self.max_cache_bytes < one_token {
            bail!(
                "GPT-OSS max_cache_bytes {} cannot admit one token; one token requires {} bytes",
                self.max_cache_bytes,
                one_token
            );
        }
        Ok(())
    }

    pub(crate) fn admit(
        &self,
        config: &GptOssConfig,
        current_tokens: usize,
        incoming_tokens: usize,
    ) -> Result<usize> {
        if incoming_tokens == 0 {
            bail!("GPT-OSS forward requires at least one input token");
        }
        let requested_tokens = current_tokens
            .checked_add(incoming_tokens)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS sequence length overflowed".to_string()))?;
        if requested_tokens > self.max_sequence_tokens {
            bail!(
                "GPT-OSS sequence admission requests {} tokens, limit is {}",
                requested_tokens,
                self.max_sequence_tokens
            );
        }
        let requested_bytes = cache_bytes_for_tokens(config, requested_tokens)?;
        if requested_bytes > self.max_cache_bytes {
            bail!(
                "GPT-OSS KV-cache admission requests {} bytes, limit is {}",
                requested_bytes,
                self.max_cache_bytes
            );
        }
        Ok(requested_bytes)
    }
}

/// Logical and allocated-byte accounting for a retained synthetic session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptOssResourceUsage {
    pub sequence_tokens: usize,
    pub cache_bytes: usize,
    pub cache_capacity_bytes: usize,
}

/// Cooperative cancellation shared by load, prefill, and decode callers.
#[derive(Debug, Clone)]
pub struct GptOssCancellationToken {
    cancelled: Arc<AtomicBool>,
    remaining_checks: Arc<AtomicUsize>,
}

impl Default for GptOssCancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl GptOssCancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            remaining_checks: Arc::new(AtomicUsize::new(NO_SCHEDULED_CANCELLATION)),
        }
    }

    /// Create a deterministic cancellation probe for bounded tests.
    ///
    /// The first `checks` calls to the internal checkpoint pass.  The next
    /// checkpoint returns cancellation, allowing rollback to be tested after
    /// partial prefill/decode work without a timing-dependent helper thread.
    pub fn cancel_after_checks(checks: usize) -> Self {
        let token = Self::new();
        token.remaining_checks.store(checks, Ordering::Release);
        token
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn checkpoint(&self) -> Result<()> {
        if self.is_cancelled() {
            bail!("GPT-OSS operation cancelled");
        }
        let remaining = self.remaining_checks.load(Ordering::Acquire);
        if remaining == NO_SCHEDULED_CANCELLATION {
            return Ok(());
        }
        if remaining == 0 {
            self.cancel();
            bail!("GPT-OSS operation cancelled");
        }
        let _ = self.remaining_checks.compare_exchange(
            remaining,
            remaining - 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        Ok(())
    }
}

#[derive(Debug, Default)]
struct LoadState {
    loading: BTreeMap<String, ()>,
    loaded: BTreeMap<String, ()>,
}

/// Shared duplicate-load guard for one Candle-owned process.
#[derive(Debug, Clone, Default)]
pub struct GptOssLoadRegistry {
    state: Arc<Mutex<LoadState>>,
}

impl GptOssLoadRegistry {
    pub fn begin(&self, identity: impl Into<String>) -> Result<GptOssLoadLease> {
        let identity = identity.into();
        if identity.is_empty() {
            bail!("GPT-OSS load identity cannot be empty");
        }
        let mut state = self.state.lock().map_err(|_| {
            candle::Error::Msg("GPT-OSS load registry mutex was poisoned".to_string())
        })?;
        if state.loading.contains_key(&identity) || state.loaded.contains_key(&identity) {
            bail!("GPT-OSS duplicate load is already owned: {identity}");
        }
        state.loading.insert(identity.clone(), ());
        Ok(GptOssLoadLease {
            registry: self.clone(),
            identity,
            committed: false,
        })
    }

    pub fn active_count(&self) -> Result<usize> {
        let state = self.state.lock().map_err(|_| {
            candle::Error::Msg("GPT-OSS load registry mutex was poisoned".to_string())
        })?;
        Ok(state.loading.len())
    }

    pub fn loaded_count(&self) -> Result<usize> {
        let state = self.state.lock().map_err(|_| {
            candle::Error::Msg("GPT-OSS load registry mutex was poisoned".to_string())
        })?;
        Ok(state.loaded.len())
    }

    fn promote(&self, identity: &str) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            candle::Error::Msg("GPT-OSS load registry mutex was poisoned".to_string())
        })?;
        if state.loading.remove(identity).is_none() {
            bail!("GPT-OSS load lease is no longer active: {identity}");
        }
        state.loaded.insert(identity.to_string(), ());
        Ok(())
    }

    fn release_loading(&self, identity: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.loading.remove(identity);
        }
    }

    fn release_loaded(&self, identity: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.loaded.remove(identity);
        }
    }
}

/// RAII ownership of one in-progress load.
#[derive(Debug)]
pub struct GptOssLoadLease {
    registry: GptOssLoadRegistry,
    identity: String,
    committed: bool,
}

impl GptOssLoadLease {
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn commit(mut self) -> Result<GptOssLoadedHandle> {
        self.registry.promote(&self.identity)?;
        self.committed = true;
        Ok(GptOssLoadedHandle {
            inner: Arc::new(GptOssLoadedHandleInner {
                registry: self.registry.clone(),
                identity: self.identity.clone(),
            }),
        })
    }
}

impl Drop for GptOssLoadLease {
    fn drop(&mut self) {
        if !self.committed {
            self.registry.release_loading(&self.identity);
        }
    }
}

/// RAII ownership of one successfully admitted load.
#[derive(Debug, Clone)]
pub struct GptOssLoadedHandle {
    inner: Arc<GptOssLoadedHandleInner>,
}

#[derive(Debug)]
struct GptOssLoadedHandleInner {
    registry: GptOssLoadRegistry,
    identity: String,
}

impl GptOssLoadedHandle {
    pub fn identity(&self) -> &str {
        &self.inner.identity
    }
}

impl Drop for GptOssLoadedHandleInner {
    fn drop(&mut self) {
        self.registry.release_loaded(&self.identity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GptOssConfig {
        GptOssConfig {
            model_type: "gpt_oss".to_string(),
            num_hidden_layers: 1,
            num_experts: 2,
            experts_per_token: 1,
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

    #[test]
    fn derives_exact_cache_bytes() -> Result<()> {
        let config = config();
        assert_eq!(cache_bytes_per_token(&config)?, 128);
        assert_eq!(cache_bytes_for_tokens(&config, 8)?, 1_024);
        let limits = GptOssResourceLimits::for_config(&config)?;
        assert_eq!(limits.max_sequence_tokens, 8);
        assert_eq!(limits.max_cache_bytes, 1_024);
        assert_eq!(limits.max_total_device_bytes, usize::MAX);
        assert_eq!(limits.admit(&config, 7, 1)?, 1_024);
        let error = limits
            .admit(&config, 8, 1)
            .expect_err("one token above the exact sequence bound must fail");
        assert!(error.to_string().contains("sequence admission"));
        Ok(())
    }

    #[test]
    fn total_device_budget_rejects_exactly_one_token_over() -> Result<()> {
        let config = config();
        let static_bytes = 10_000;
        let exact = total_device_bytes_for_tokens(&config, static_bytes, 4)?;
        assert!(total_device_bytes_for_tokens(&config, static_bytes, 4)? <= exact);
        assert!(total_device_bytes_for_tokens(&config, static_bytes, 5)? > exact);
        assert_eq!(
            max_supported_sequence_tokens(&config, static_bytes, 8, exact)?,
            4
        );
        Ok(())
    }

    #[test]
    fn load_registry_releases_failed_and_successful_ownership() -> Result<()> {
        let registry = GptOssLoadRegistry::default();
        let lease = registry.begin("fixture")?;
        assert_eq!(registry.active_count()?, 1);
        let duplicate = registry
            .begin("fixture")
            .expect_err("active duplicate must fail closed");
        assert!(duplicate.to_string().contains("duplicate"));
        drop(lease);
        assert_eq!(registry.active_count()?, 0);

        let lease = registry.begin("fixture")?;
        let handle = lease.commit()?;
        assert_eq!(handle.identity(), "fixture");
        assert_eq!(registry.active_count()?, 0);
        assert_eq!(registry.loaded_count()?, 1);
        let duplicate = registry
            .begin("fixture")
            .expect_err("loaded duplicate must fail closed");
        assert!(duplicate.to_string().contains("duplicate"));
        drop(handle);
        assert_eq!(registry.loaded_count()?, 0);

        let retry = registry.begin("fixture")?;
        drop(retry);
        assert_eq!(registry.active_count()?, 0);
        Ok(())
    }

    #[test]
    fn cancellation_token_is_deterministic_and_shared() -> Result<()> {
        let token = GptOssCancellationToken::cancel_after_checks(1);
        token.checkpoint()?;
        assert!(!token.is_cancelled());
        let clone = token.clone();
        let error = clone
            .checkpoint()
            .expect_err("scheduled cancellation must fire at the next checkpoint");
        assert!(error.to_string().contains("cancelled"));
        assert!(token.is_cancelled());
        Ok(())
    }
}
