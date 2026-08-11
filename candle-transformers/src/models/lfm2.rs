//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use std::collections::HashMap;

include!("lfm2/config.rs");
include!("lfm2/cache.rs");
include!("lfm2/layers.rs");
include!("lfm2/model.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config(tie_embedding: bool) -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 12,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: vec![LayerType::Conv, LayerType::FullAttention],
            tie_embedding,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            use_flash_attn: false,
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    #[test]
    fn flash_attention_without_feature_returns_an_error() -> Result<()> {
        let tensor = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?;
        let err = flash_attn(&tensor, &tensor, &tensor, 1.0, false).unwrap_err();
        assert!(
            err.to_string()
                .contains("built without the 'flash-attn' feature"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let max_abs = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        eprintln!("lfm2 {label}: max absolute error {max_abs:.8e}");
        assert!(
            max_abs <= tolerance,
            "{label}: max absolute error {max_abs} exceeds {tolerance}"
        );
        Ok(())
    }

    fn parse_config(value: serde_json::Value) -> Result<Lfm2Config> {
        serde_json::from_value(value)
            .map_err(|err| candle::Error::Msg(format!("invalid test LFM2 config: {err}")))
    }

    fn add_tiny_language_weights(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Config,
        device: &Device,
    ) -> Result<()> {
        let zero = |shape: &[usize]| Tensor::zeros(shape, DType::F32, device);
        weights.insert(
            format!("{prefix}.embed_tokens.weight"),
            zero(&[cfg.vocab_size, cfg.hidden_size])?,
        );
        weights.insert(
            format!("{prefix}.embedding_norm.weight"),
            Tensor::ones(cfg.hidden_size, DType::F32, device)?,
        );
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            weights.insert(
                format!("{layer_prefix}.operator_norm.weight"),
                Tensor::ones(cfg.hidden_size, DType::F32, device)?,
            );
            weights.insert(
                format!("{layer_prefix}.ffn_norm.weight"),
                Tensor::ones(cfg.hidden_size, DType::F32, device)?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w1.weight"),
                zero(&[cfg.intermediate_size, cfg.hidden_size])?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w2.weight"),
                zero(&[cfg.hidden_size, cfg.intermediate_size])?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w3.weight"),
                zero(&[cfg.intermediate_size, cfg.hidden_size])?,
            );
            match cfg.layer_types[layer_idx] {
                LayerType::Conv => {
                    weights.insert(
                        format!("{layer_prefix}.conv.in_proj.weight"),
                        zero(&[3 * cfg.hidden_size, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.conv.out_proj.weight"),
                        zero(&[cfg.hidden_size, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.conv.conv.weight"),
                        zero(&[cfg.hidden_size, 1, cfg.conv_l_cache])?,
                    );
                }
                LayerType::FullAttention => {
                    let head_dim = cfg.head_dim();
                    weights.insert(
                        format!("{layer_prefix}.self_attn.q_proj.weight"),
                        zero(&[cfg.num_attention_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.k_proj.weight"),
                        zero(&[cfg.num_key_value_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.v_proj.weight"),
                        zero(&[cfg.num_key_value_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.out_proj.weight"),
                        zero(&[cfg.hidden_size, cfg.num_attention_heads * head_dim])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.q_layernorm.weight"),
                        Tensor::ones(head_dim, DType::F32, device)?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.k_layernorm.weight"),
                        Tensor::ones(head_dim, DType::F32, device)?,
                    );
                }
            }
        }
        Ok(())
    }

    fn official_text_config(hidden_size: usize, intermediate_size: usize) -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2",
            "vocab_size": 65536,
            "hidden_size": hidden_size,
            "num_hidden_layers": 16,
            "num_attention_heads": if hidden_size == 1024 { 16 } else { 32 },
            "num_key_value_heads": 8,
            "intermediate_size": intermediate_size,
            "block_ff_dim": intermediate_size,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "conv_l_cache": 3,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"}
        })
    }

    #[test]
    fn config_aliases_and_official_ffn_widths() -> Result<()> {
        let legacy = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "intermediate_size": 64,
            "block_auto_adjust_ff_dim": false,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "conv_L_cache": 5,
            "tie_embedding": false,
            "rope_theta": 42.0
        }))?;
        assert_eq!(legacy.conv_l_cache, 5);
        assert_eq!(legacy.intermediate_size, Some(64));
        assert_eq!(legacy.block_ff_dim, None);
        assert!(!legacy.tie_embedding);
        assert_eq!(legacy.rope_theta, 42.0);
        assert_eq!(legacy.effective_ffn_dim()?, 64);

        let precedence = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "intermediate_size": 4096,
            "block_ff_dim": 6656,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "conv_l_cache": 7,
            "tie_word_embeddings": true,
            "tie_embedding": false,
            "rope_theta": 123.0,
            "rope_parameters": {"rope_theta": 456.0}
        }))?;
        assert_eq!(precedence.conv_l_cache, 7);
        assert_eq!(precedence.intermediate_size, Some(6656));
        assert_eq!(precedence.block_ff_dim, Some(6656));
        assert!(!precedence.tie_embedding);
        assert_eq!(precedence.rope_theta, 456.0);
        assert_eq!(precedence.effective_ffn_dim()?, 4608);

        let official_450 = parse_config(official_text_config(1024, 6656))?;
        assert_eq!(official_450.effective_ffn_dim()?, 4608);
        assert_eq!(official_450.tie_word_embeddings, None);
        let official_450_normalized = official_450.try_into_config(false)?;
        assert_eq!(official_450_normalized.intermediate_size, 4608);
        assert_eq!(official_450_normalized.layer_types.len(), 16);
        assert_eq!(
            official_450_normalized.layer_types[2],
            LayerType::FullAttention
        );
        assert_eq!(official_450_normalized.layer_types[1], LayerType::Conv);

        let official_16 = parse_config(official_text_config(2048, 12288))?;
        assert_eq!(official_16.effective_ffn_dim()?, 8192);
        assert!(official_16.tie_embedding);
        assert_eq!(official_16.try_into_config(false)?.intermediate_size, 8192);

        let current_untied = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false
        }))?;
        assert_eq!(current_untied.tie_word_embeddings, Some(false));
        assert!(!current_untied.tie_embedding);

        let full_attention_alias = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "full_attn_idxs": [1]
        }))?;
        assert_eq!(
            full_attention_alias.try_into_config(false)?.layer_types,
            vec![LayerType::Conv, LayerType::FullAttention]
        );

        let layer_type_precedence = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "full_attn_idxs": [1],
            "layer_types": ["conv", "conv"]
        }))?;
        assert_eq!(
            layer_type_precedence.try_into_config(false)?.layer_types,
            vec![LayerType::Conv, LayerType::Conv]
        );

        Ok(())
    }

    #[test]
    fn missing_ffn_uses_legacy_hidden_width_fallback() -> Result<()> {
        let config = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_auto_adjust_ff_dim": false
        }))?;
        assert_eq!(config.intermediate_size, None);
        assert_eq!(config.effective_ffn_dim()?, 48);
        assert!(config.tie_embedding);
        Ok(())
    }

    #[test]
    fn malformed_dimensions_are_rejected_before_model_construction() -> Result<()> {
        let base = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "conv_l_cache": 3
        });
        let cases = [
            (
                "num_attention_heads",
                serde_json::json!(0),
                "num_attention_heads must be greater than zero",
            ),
            (
                "num_key_value_heads",
                serde_json::json!(2),
                "must be divisible by num_key_value_heads",
            ),
            (
                "hidden_size",
                serde_json::json!(15),
                "attention head dimension 5 must be a positive even number",
            ),
            (
                "conv_l_cache",
                serde_json::json!(0),
                "conv_l_cache must be greater than zero",
            ),
            (
                "max_position_embeddings",
                serde_json::json!(0),
                "max_position_embeddings must be greater than zero",
            ),
            (
                "norm_eps",
                serde_json::json!(0.0),
                "norm_eps must be finite and greater than zero",
            ),
            (
                "rope_theta",
                serde_json::json!(0.0),
                "rope_theta must be finite and greater than zero",
            ),
        ];
        for (field, value, expected) in cases {
            let mut raw = base.clone();
            raw[field] = value;
            let config = parse_config(raw)?;
            let error = config.try_into_config(false).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "{field}: expected error containing {expected:?}, got {error}"
            );
        }

        let mut out_of_range = base;
        out_of_range["full_attn_idxs"] = serde_json::json!([2]);
        let config = parse_config(out_of_range)?;
        let error = config.try_into_config(false).unwrap_err().to_string();
        assert!(error.contains("full_attention_layers index 2"));
        Ok(())
    }

    #[test]
    fn cache_rejects_unrepresentable_positions_and_index_overflow() -> Result<()> {
        let device = Device::Cpu;
        let mut invalid = tiny_config(true);
        invalid.max_position_embeddings = (u32::MAX as usize).saturating_add(1);
        let error = Cache::new(true, DType::F32, &invalid, &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("supported u32 position range"), "{error}");

        let mut cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let error = cache.mask(1, usize::MAX).unwrap_err().to_string();
        assert!(error.contains("sequence position overflow"), "{error}");
        Ok(())
    }

    #[test]
    fn constructors_support_standalone_nested_and_explicit_heads() -> Result<()> {
        let device = Device::Cpu;
        let tied = tiny_config(true);
        let mut standalone_weights = HashMap::new();
        add_tiny_language_weights(&mut standalone_weights, "model", &tied, &device)?;
        let _standalone = Model::new(
            &tied,
            VarBuilder::from_tensors(standalone_weights, DType::F32, &device),
        )?;

        let mut nested_weights = HashMap::new();
        add_tiny_language_weights(&mut nested_weights, "model.language_model", &tied, &device)?;
        let nested_vb = VarBuilder::from_tensors(nested_weights, DType::F32, &device);
        let _nested =
            Model::new_from_parts(&tied, nested_vb.pp("model").pp("language_model"), None)?;

        let untied = tiny_config(false);
        let mut explicit_weights = HashMap::new();
        add_tiny_language_weights(&mut explicit_weights, "model", &untied, &device)?;
        explicit_weights.insert(
            "lm_head.weight".to_string(),
            Tensor::zeros((untied.vocab_size, untied.hidden_size), DType::F32, &device)?,
        );
        let _explicit = Model::new(
            &untied,
            VarBuilder::from_tensors(explicit_weights, DType::F32, &device),
        )?;
        Ok(())
    }

    #[test]
    fn fixture_proves_dense_embedding_and_cached_decode_parity() -> Result<()> {
        let device = Device::Cpu;
        let weights_vb = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?;
        let fixture_tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let fixture_tensor = |name: &str| match fixture_tensors.get(name) {
            Some(tensor) => Ok(tensor.clone()),
            None => candle::bail!("missing tiny fixture tensor {name}"),
        };
        let model_vb = weights_vb.pp("weights").pp("model").pp("language_model");
        let model = Model::new_from_parts(&tiny_config(true), model_vb, None)?;

        let input_ids = fixture_tensor("input.input_ids")?;
        let expected_embeddings = fixture_tensor("stage.text.embeddings")?;
        let embeddings = model.embed_tokens(&input_ids)?;
        assert_close(&embeddings, &expected_embeddings, 1e-6, "token embeddings")?;

        let merged_embeddings = fixture_tensor("stage.multimodal.merged_embeddings")?;
        let expected_hidden = fixture_tensor("stage.language.hidden_states")?;
        let expected_prefill = fixture_tensor("stage.language.prefill_logits")?;

        let mut token_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let token_logits = model.forward(&input_ids, 0, &mut token_cache)?;
        let mut embed_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let embed_logits = model.forward_embeds(&embeddings, 0, &mut embed_cache)?;
        assert_close(
            &token_logits,
            &embed_logits,
            1e-6,
            "token-ID vs embedding-driven forwarding",
        )?;

        let mut parity_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let hidden = model.forward_hidden(&merged_embeddings, 0, &mut parity_cache)?;
        assert_close(&hidden, &expected_hidden, 1e-3, "prefill hidden states")?;
        let prefill_logits = model.project_logits(&hidden, 0)?;
        assert_close(&prefill_logits, &expected_prefill, 1e-3, "prefill logits")?;

        let decode_ids = fixture_tensor("input.decode_token_ids")?;
        let expected_decode = fixture_tensor("stage.language.decode_logits")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let logits = model.forward(&token, 5 + step, &mut parity_cache)?;
            let expected = expected_decode.i((.., step, ..))?;
            assert_close(&logits, &expected, 1e-3, "cached decode logits")?;
        }

        parity_cache.clear();
        let hidden_after_reset = model.forward_hidden(&merged_embeddings, 0, &mut parity_cache)?;
        assert_close(
            &hidden_after_reset,
            &expected_hidden,
            1e-3,
            "cache-reset prefill hidden states",
        )?;
        let reset_decode = model.forward(&decode_ids.i((.., 0..1))?, 5, &mut parity_cache)?;
        assert_close(
            &reset_decode,
            &expected_decode.i((.., 0, ..))?,
            1e-3,
            "cache-reset decode logits",
        )?;

        Ok(())
    }
}
