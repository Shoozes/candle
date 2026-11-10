use super::*;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

const WEIGHTS: &[u8] =
    include_bytes!("../../../../tests/fixtures/lfm2_moe_tiny/native.safetensors");
const CONFIG: &str = include_str!("../../../../tests/fixtures/lfm2_moe_tiny/native.json");

pub(crate) fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (*a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "index {index}: {a} vs {e}"
        );
    }
    Ok(())
}

#[test]
fn native_pinned_oracle_prefill_decode_and_reset() -> Result<()> {
    native_oracle_on_device(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_pinned_oracle_prefill_decode_and_reset() -> Result<()> {
    native_oracle_on_device(&Device::new_cuda(0)?)
}

fn native_oracle_on_device(device: &Device) -> Result<()> {
    let cfg: Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
    let vb = VarBuilder::from_slice_safetensors(WEIGHTS, DType::F32, &device)?;
    let model = Model::new(&cfg, vb.clone())?;
    let ids = vb.get((1, 4), "oracle.ids")?.to_dtype(DType::U32)?;
    let decode = vb.get((1, 1), "oracle.decode_ids")?.to_dtype(DType::U32)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;
    let hidden = model.forward_hidden(&model.embed_tokens(&ids)?, 0, &mut cache)?;
    assert_close(&hidden, &vb.get((1, 4, cfg.hidden_size), "oracle.hidden")?)?;
    assert_close(
        &model.project_logits(&hidden, 0)?,
        &vb.get((1, 4, cfg.vocab_size), "oracle.logits")?,
    )?;
    let first = model.project_logits(&hidden, 1)?.squeeze(1)?;
    assert_close(
        &model.forward(&decode, 4, &mut cache)?,
        &vb.get((1, 1, cfg.vocab_size), "oracle.decode_logits")?
            .squeeze(1)?,
    )?;
    cache.clear();
    assert_close(&model.forward(&ids, 0, &mut cache)?, &first)?;
    cache.clear();
    model.forward(&ids.narrow(1, 0, 2)?, 0, &mut cache)?;
    assert_close(
        &model.forward(&ids.narrow(1, 2, 2)?, 2, &mut cache)?,
        &first,
    )?;
    assert!(model.forward(&decode, 1, &mut cache).is_err());
    assert_close(&model.forward(&ids, 0, &mut cache)?, &first)?;
    assert!(model
        .forward(&Tensor::new(&[[999u32]], &device)?, 4, &mut cache)
        .is_err());
    assert_close(&model.forward(&ids, 0, &mut cache)?, &first)?;
    let bad_embeds = (model.embed_tokens(&ids)? * f64::NAN)?;
    assert!(model.forward_embeds(&bad_embeds, 0, &mut cache).is_err());
    assert!(model
        .forward(
            &Tensor::zeros((1, cfg.max_position_embeddings + 1), DType::U32, device)?,
            0,
            &mut cache
        )
        .is_err());
    assert_close(&model.forward(&ids, 0, &mut cache)?, &first)
}

#[test]
fn routing_matches_pinned_bias_normalization_and_scaling() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::from_slice_safetensors(WEIGHTS, DType::F32, &device)?;
    let logits = vb.get((3, 4), "oracle.router_logits")?;
    let bias = vb.get(4, "oracle.router_bias")?;
    for biased in [false, true] {
        for normalized in [false, true] {
            let (ids, weights) = routing::route(
                &logits,
                if biased { Some(&bias) } else { None },
                2,
                normalized,
                1.7,
            )?;
            let key = format!("oracle.route_{}_{}", u8::from(biased), u8::from(normalized));
            assert_eq!(
                ids.to_vec2::<u32>()?,
                vb.get((3, 2), &format!("{key}.ids"))?
                    .to_dtype(DType::U32)?
                    .to_vec2::<u32>()?
            );
            assert_close(&weights, &vb.get((3, 2), &format!("{key}.weights"))?)?;
        }
    }
    assert!(routing::route(&logits, None, 0, true, 1.).is_err());
    assert!(routing::route(&logits, None, 5, true, 1.).is_err());
    assert!(routing::route(&Tensor::new(&[[f32::NAN, 0.]], &device)?, None, 1, true, 1.).is_err());
    Ok(())
}

#[test]
fn invalid_moe_configuration_is_rejected() -> Result<()> {
    let base: Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
    for kind in 0..7 {
        let mut cfg = base.clone();
        match kind {
            0 => cfg.num_experts = 0,
            1 => cfg.num_experts_per_tok = cfg.num_experts + 1,
            2 => cfg.num_dense_layers = cfg.num_hidden_layers + 1,
            3 => cfg.moe_intermediate_size = usize::MAX,
            4 => cfg.routed_scaling_factor = f64::NAN,
            5 => cfg.norm_eps = f64::NAN,
            _ => cfg.num_experts = 4097,
        }
        assert!(cfg.validate().is_err());
    }
    let vb = VarBuilder::from_slice_safetensors(WEIGHTS, DType::F32, &Device::Cpu)?;
    let model = Model::new(&base, vb)?;
    let mut short = base.clone();
    short.num_hidden_layers = 2;
    short.layer_types.truncate(2);
    let mut cache = Cache::new(true, DType::F32, &short, &Device::Cpu)?;
    assert!(model
        .forward(&Tensor::new(&[[1u32]], &Device::Cpu)?, 0, &mut cache)
        .is_err());
    Ok(())
}

#[test]
fn native_untied_head_and_missing_expert_are_checked() -> Result<()> {
    let device = Device::Cpu;
    let mut cfg: Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
    let mut tensors = candle::safetensors::load_buffer(WEIGHTS, &device)?;
    let ids = Tensor::new(&[[1u32, 4, 3, 5]], &device)?;
    let expected = tensors
        .get("oracle.logits")
        .expect("oracle")
        .i((.., 3, ..))?;
    tensors.insert(
        "lm_head.weight".into(),
        tensors
            .get("model.embed_tokens.weight")
            .expect("embedding")
            .clone(),
    );
    cfg.tie_word_embeddings = false;
    let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
    let model = Model::new(&cfg, vb)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;
    assert_close(&model.forward(&ids, 0, &mut cache)?, &expected)?;
    tensors.remove("model.layers.1.feed_forward.experts.0.w1.weight");
    assert!(Model::new(&cfg, VarBuilder::from_tensors(tensors, DType::F32, &device)).is_err());
    Ok(())
}
