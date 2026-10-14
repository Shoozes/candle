//! CUDA-specific bounds; legacy CPU accounting retains its original contract.

use super::{GptOssAttentionType, GptOssConfig};
use candle::{bail, Result};

pub(crate) const PAGE_ROWS: usize = 64;

fn checked_sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA memory accounting overflowed".into()))
}

fn rows_bytes(config: &GptOssConfig, rows: usize) -> Result<usize> {
    rows.checked_mul(config.num_key_value_heads)
        .and_then(|n| n.checked_mul(config.head_dim))
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA KV byte count overflowed".into()))
}

fn layer_counts(config: &GptOssConfig) -> (usize, usize) {
    let full = if config.layer_types.is_empty() {
        config.num_hidden_layers / 2
    } else {
        config
            .layer_types
            .iter()
            .filter(|&&t| t == GptOssAttentionType::FullAttention)
            .count()
    };
    (full, config.num_hidden_layers - full)
}

fn layer_bytes(config: &GptOssConfig, full_rows: usize, sliding_rows: usize) -> Result<usize> {
    let (full, sliding) = layer_counts(config);
    let full = rows_bytes(config, full_rows)?.checked_mul(full);
    let sliding = rows_bytes(config, sliding_rows)?.checked_mul(sliding);
    full.and_then(|f| sliding.and_then(|s| f.checked_add(s)))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA layer cache overflowed".into()))
}

/// Logical retained CUDA K/V bytes, with physical sliding-window eviction.
pub fn cuda_cache_bytes_for_tokens(config: &GptOssConfig, tokens: usize) -> Result<usize> {
    config.validate()?;
    layer_bytes(config, tokens, tokens.min(config.sliding_window))
}

/// Conservative allocated CUDA K/V bytes, including a partially evicted page.
pub fn cuda_cache_capacity_bytes_for_tokens(config: &GptOssConfig, tokens: usize) -> Result<usize> {
    config.validate()?;
    let sliding_rows = checked_sum(config.sliding_window, PAGE_ROWS - 1)?;
    layer_bytes(config, tokens, tokens.min(sliding_rows))
}

/// Upper bound for grouped-query attention materialization and live F32 work.
/// It includes one layer's gathered/permuted K/V, score/softmax intermediates,
/// and three output rows (previous logits, matmul and optional biased output).
/// It does not estimate CUDA context or allocator overhead.
pub fn cuda_workspace_bytes_for_tokens(config: &GptOssConfig, tokens: usize) -> Result<usize> {
    config.validate()?;
    let kv = rows_bytes(config, tokens)?
        .checked_mul(2)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA attention workspace overflowed".into()))?;
    let scores = config
        .num_attention_heads
        .checked_mul(checked_sum(tokens, 1)?)
        .and_then(|n| n.checked_mul(16))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA score workspace overflowed".into()))?;
    let logits = config
        .vocab_size
        .checked_mul(12)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA output workspace overflowed".into()))?;
    // Keep routed hidden rows and additional live SwiGLU intermediates
    // explicit; neither is bounded by an unrouted hidden-width buffer.
    let routed = config
        .experts_per_token
        .checked_mul(config.hidden_size)
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA routed workspace overflowed".into()))?;
    let middle = config
        .experts_per_token
        .checked_mul(config.intermediate_size)
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS CUDA SwiGLU workspace overflowed".into()))?;
    let fixed = checked_sum(
        super::workspace_bytes_for_tokens(config, 0)?,
        checked_sum(routed, middle)?,
    )?;
    checked_sum(checked_sum(kv, scores)?, checked_sum(fixed, logits)?)
}

/// Peak live payload bound for a transaction, preserving its committed root.
/// Full pages are shared; old full-attention tails and old sliding windows may
/// coexist with the candidate. One additional tail covers append overlap.
pub fn cuda_peak_device_bytes_for_forward(
    config: &GptOssConfig,
    static_bytes: usize,
    committed: usize,
    incoming: usize,
) -> Result<usize> {
    config.validate()?;
    if incoming == 0 {
        bail!("GPT-OSS CUDA forward needs at least one incoming token");
    }
    let tokens = checked_sum(committed, incoming)?;
    let snapshot = layer_bytes(
        config,
        committed % PAGE_ROWS,
        committed.min(checked_sum(config.sliding_window, PAGE_ROWS - 1)?),
    )?;
    let total = checked_sum(
        static_bytes,
        cuda_cache_capacity_bytes_for_tokens(config, tokens)?,
    )?;
    let total = checked_sum(total, snapshot)?;
    let total = checked_sum(total, rows_bytes(config, tokens.min(PAGE_ROWS))?)?;
    checked_sum(total, cuda_workspace_bytes_for_tokens(config, tokens)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_twenty_b_32k_payload_bound_is_not_physical_qualification() -> Result<()> {
        let c = GptOssConfig::from_json_str(
            r#"{
            "hidden_size":2880,"num_hidden_layers":24,"num_attention_heads":64,
            "num_key_value_heads":8,"head_dim":64,"intermediate_size":2880,
            "num_experts":32,"experts_per_token":4,"vocab_size":201088,"sliding_window":128
        }"#,
        )?;
        let old_static = 17_370_183_936usize;
        let table_savings = c.vocab_size * c.hidden_size * 2 * 2;
        let static_bytes = old_static - table_savings;
        assert_eq!(table_savings, 2_316_533_760);
        assert_eq!(
            super::super::total_device_bytes_for_tokens(&c, old_static, 32768)?,
            24_903_656_192
        );
        assert_eq!(
            cuda_cache_capacity_bytes_for_tokens(&c, 32768)?,
            1_620_000_768
        );
        assert_eq!(cuda_workspace_bytes_for_tokens(&c, 32768)?, 305_643_520);
        assert_eq!(
            cuda_peak_device_bytes_for_forward(&c, static_bytes, 32767, 1)?,
            16_992_041_216
        );
        // This bound excludes context/allocator/other-process overhead by design.
        assert!(cuda_peak_device_bytes_for_forward(&c, static_bytes, 32767, 1)? < 20_000_000_000);
        Ok(())
    }

    #[test]
    fn mixed_windows_peak_snapshot_and_overflow() -> Result<()> {
        let mut c = GptOssConfig::from_json_str(
            r#"{
            "hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"intermediate_size":32,
            "num_experts":2,"experts_per_token":2,"vocab_size":8,"sliding_window":3
        }"#,
        )?;
        let row = c.num_key_value_heads * c.head_dim * 8;
        assert_eq!(
            cuda_cache_bytes_for_tokens(&c, 131)?,
            (131 + c.sliding_window) * row
        );
        assert_eq!(
            cuda_cache_capacity_bytes_for_tokens(&c, 131)?,
            (131 + c.sliding_window + 63) * row
        );
        let fresh = cuda_peak_device_bytes_for_forward(&c, 500, 0, 131)?;
        let resumed = cuda_peak_device_bytes_for_forward(&c, 500, 130, 1)?;
        assert_eq!(resumed - fresh, (130 % 64 + c.sliding_window + 63) * row);
        assert!(cuda_peak_device_bytes_for_forward(&c, 1, usize::MAX, 1).is_err());
        assert!(cuda_peak_device_bytes_for_forward(&c, usize::MAX, 0, 1).is_err());
        assert!(cuda_peak_device_bytes_for_forward(&c, 1, 0, 0).is_err());
        c.layer_types = vec![GptOssAttentionType::SlidingAttention; c.num_hidden_layers];
        assert_eq!(
            cuda_cache_bytes_for_tokens(&c, 1000)?,
            c.num_hidden_layers * c.sliding_window * row
        );
        Ok(())
    }
}
