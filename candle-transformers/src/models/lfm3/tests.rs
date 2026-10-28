use super::*;
use std::collections::HashMap;

pub(crate) fn tiny_config() -> Config {
    Config {
        vocab_size: 19,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        norm_eps: 1e-5,
        tie_word_embeddings: true,
        conv_bias: true,
        conv_kernel_size: 7,
        layer_types: vec![
            LayerType::Conv,
            LayerType::SlidingAttention,
            LayerType::FullAttention,
        ],
        sliding_window: 3,
        use_sandwich_norm: true,
        is_causal: true,
        rope_parameters: HashMap::from([
            (
                "full_attention".into(),
                RopeParameters {
                    rope_type: "default".into(),
                    rope_theta: 1_000_000.0,
                    partial_rotary_factor: 0.5,
                },
            ),
            (
                "sliding_attention".into(),
                RopeParameters {
                    rope_type: "default".into(),
                    rope_theta: 10_000.0,
                    partial_rotary_factor: 1.0,
                },
            ),
        ]),
    }
}

pub(crate) fn weights(c: &Config) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    let mut add = |name: String, shape: Vec<usize>, norm: bool| -> Result<()> {
        let n = shape.iter().product();
        let data = (0..n)
            .map(|i| {
                if norm {
                    0.8 + (i % 5) as f32 * 0.03
                } else {
                    ((i as f32 + 1.0) * 0.73).sin() * 0.1
                }
            })
            .collect::<Vec<_>>();
        tensors.insert(name, Tensor::from_vec(data, shape, &Device::Cpu)?);
        Ok(())
    };
    add(
        "embed_tokens.weight".into(),
        vec![c.vocab_size, c.hidden_size],
        false,
    )?;
    add("embedding_norm.weight".into(), vec![c.hidden_size], true)?;
    for (i, kind) in c.layer_types.iter().enumerate() {
        let p = format!("layers.{i}");
        for name in ["operator_norm", "ffn_norm"] {
            add(format!("{p}.{name}.weight"), vec![c.hidden_size], true)?;
        }
        if c.use_sandwich_norm {
            for name in ["post_operator_norm", "post_ffn_norm"] {
                add(format!("{p}.{name}.weight"), vec![c.hidden_size], true)?;
            }
        }
        for (name, out, input) in [
            ("w1", c.intermediate_size, c.hidden_size),
            ("w3", c.intermediate_size, c.hidden_size),
            ("w2", c.hidden_size, c.intermediate_size),
        ] {
            add(
                format!("{p}.feed_forward.{name}.weight"),
                vec![out, input],
                false,
            )?;
        }
        if *kind == LayerType::Conv {
            add(
                format!("{p}.conv.conv.weight"),
                vec![c.hidden_size, 1, c.conv_kernel_size],
                false,
            )?;
            if c.conv_bias {
                add(format!("{p}.conv.conv.bias"), vec![c.hidden_size], false)?;
            }
            for (name, out) in [("in_proj", 3 * c.hidden_size), ("out_proj", c.hidden_size)] {
                add(
                    format!("{p}.conv.{name}.weight"),
                    vec![out, c.hidden_size],
                    false,
                )?;
                if c.conv_bias {
                    add(format!("{p}.conv.{name}.bias"), vec![out], false)?;
                }
            }
        } else {
            for (name, out, input) in [
                ("q_proj", c.num_attention_heads * c.head_dim, c.hidden_size),
                ("k_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
                ("v_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
                ("o_proj", c.hidden_size, c.num_attention_heads * c.head_dim),
                ("gate_proj", c.num_attention_heads, c.hidden_size),
            ] {
                add(
                    format!("{p}.self_attn.{name}.weight"),
                    vec![out, input],
                    false,
                )?;
            }
            for name in ["q_norm", "k_norm"] {
                add(
                    format!("{p}.self_attn.{name}.weight"),
                    vec![c.head_dim],
                    true,
                )?;
            }
        }
    }
    Ok(tensors)
}

fn close(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
    let error = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(error <= tolerance, "max_abs={error} > {tolerance}");
    Ok(())
}

#[test]
fn chunked_decode_matches_full_with_biased_width_seven_conv_and_sliding_cache() -> Result<()> {
    let cfg = tiny_config();
    let model = Model::new(
        &cfg,
        VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu),
        None,
    )?;
    let ids = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8, 9]], &Device::Cpu)?;
    let embeds = model.embed_tokens(&ids)?;
    let expected = model.backbone.forward_uncached(&embeds)?;
    for chunks in [vec![9], vec![1; 9], vec![2, 4, 3], vec![7, 2]] {
        let mut cache = Cache::new(&cfg, 16)?;
        let mut results = vec![];
        let mut position = 0;
        for length in chunks {
            results.push(
                model
                    .backbone
                    .forward_embeds(&embeds.narrow(1, position, length)?, &mut cache)?,
            );
            position += length;
        }
        close(&Tensor::cat(&results, 1)?, &expected, 2e-5)?;
        assert_eq!(cache.position(), 9);
        assert_eq!(cache.attention_lengths()?, vec![0, 2, 9]);
        cache.clear();
        assert_eq!(cache.position(), 0);
        assert_eq!(cache.attention_lengths()?, vec![0; 3]);
        close(
            &model.backbone.forward_embeds(&embeds, &mut cache)?,
            &expected,
            2e-5,
        )?;
    }
    Ok(())
}

#[test]
fn rotary_uses_full_width_frequency_and_keeps_tail_pairs_unchanged() -> Result<()> {
    let mut cfg = tiny_config();
    cfg.head_dim = 8;
    let values = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
    let x = Tensor::from_vec(values.clone(), (1, 1, 1, 8), &Device::Cpu)?;
    let actual = layers::rotary(&x, 3, &cfg, LayerType::FullAttention)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for i in 0..4 {
        let angle = if i < 2 {
            3. / 1_000_000f64.powf((2 * i) as f64 / 8.)
        } else {
            0.
        };
        let (sin, cos) = angle.sin_cos();
        let a = values[i] as f64;
        let b = values[i + 4] as f64;
        assert!((actual[i] as f64 - (a * cos - b * sin)).abs() < 2e-6);
        assert!((actual[i + 4] as f64 - (b * cos + a * sin)).abs() < 2e-6);
    }
    assert_eq!(&actual[2..4], &values[2..4]);
    assert_eq!(&actual[6..8], &values[6..8]);
    Ok(())
}

#[test]
fn cache_limit_and_mismatched_configuration_fail_without_advancing() -> Result<()> {
    let cfg = tiny_config();
    let model = Model::new(
        &cfg,
        VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu),
        None,
    )?;
    let mut cache = Cache::new(&cfg, 2)?;
    let ids = Tensor::new(&[[1u32, 2]], &Device::Cpu)?;
    model.forward(&ids, &mut cache)?;
    assert!(model.forward(&ids, &mut cache).is_err());
    assert_eq!(cache.position(), 2);
    let mut other = cfg.clone();
    other.norm_eps = 0.01;
    let mut wrong = Cache::new(&other, 2)?;
    assert!(model.forward(&ids, &mut wrong).is_err());
    assert_eq!(wrong.position(), 0);
    assert!(Cache::new(&cfg, cfg.max_position_embeddings + 1).is_err());
    Ok(())
}

#[test]
fn causal_prefix_has_no_future_leak_and_bidirectional_depth_does() -> Result<()> {
    let mut cfg = tiny_config();
    cfg.layer_types = vec![LayerType::FullAttention];
    cfg.num_hidden_layers = 1;
    cfg.use_sandwich_norm = false;
    let ids = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu)?;
    let model = Model::new(
        &cfg,
        VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu),
        None,
    )?;
    let x = model.embed_tokens(&ids)?;
    let full = model.backbone.forward_uncached(&x)?;
    let prefix = model.backbone.forward_uncached(&x.narrow(1, 0, 1)?)?;
    close(&full.narrow(1, 0, 1)?, &prefix, 1e-6)?;
    cfg.is_causal = false;
    let depth = Backbone::new(
        &cfg,
        VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu),
    )?;
    let full = depth.forward_uncached(&x)?;
    let prefix = depth.forward_uncached(&x.narrow(1, 0, 1)?)?;
    let delta = (full.narrow(1, 0, 1)? - prefix)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(delta > 1e-4);
    assert!(Cache::new(&cfg, 4).is_err());
    Ok(())
}

#[test]
fn malformed_configurations_fail_before_tensor_construction() {
    for mutate in [
        |c: &mut Config| c.head_dim = 3,
        |c: &mut Config| c.num_key_value_heads = 3,
        |c: &mut Config| c.sliding_window = 0,
        |c: &mut Config| c.layer_types.pop().map(|_| ()).unwrap_or(()),
        |c: &mut Config| c.norm_eps = f64::NAN,
        |c: &mut Config| c.hidden_size = usize::MAX,
        |c: &mut Config| c.conv_kernel_size = usize::MAX,
        |c: &mut Config| {
            c.rope_parameters.clear();
        },
    ] {
        let mut cfg = tiny_config();
        mutate(&mut cfg);
        assert!(cfg.validate().is_err());
    }
}
