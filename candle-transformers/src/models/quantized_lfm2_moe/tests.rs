use super::*;
use crate::models::{lfm2::LayerType, lfm2_moe};
use candle::{
    quantized::{gguf_file, GgmlDType, QTensor},
    DType, Device, IndexOp, Result, Tensor,
};
use candle_nn::VarBuilder;
use std::{collections::BTreeMap, io::Cursor};

const NATIVE: &[u8] =
    include_bytes!("../../../../tests/fixtures/lfm2_moe_tiny/quantized.safetensors");
const CONFIG: &str = include_str!("../../../../tests/fixtures/lfm2_moe_tiny/quantized.json");

fn fixture(mode: u8) -> Result<Vec<u8>> {
    fixture_with_context(mode, None)
}

fn fixture_with_context(mode: u8, context: Option<usize>) -> Result<Vec<u8>> {
    let device = Device::Cpu;
    let cfg: lfm2_moe::Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
    let vb = VarBuilder::from_slice_safetensors(NATIVE, DType::F32, &device)?;
    let mut tensors = BTreeMap::new();
    let mut put = |name: String, value: Tensor, auxiliary: bool, down: bool| -> Result<()> {
        let dtype = if mode == 3 {
            GgmlDType::F16
        } else if mode == 4 {
            GgmlDType::BF16
        } else if auxiliary || mode == 0 {
            GgmlDType::F32
        } else if mode == 2 {
            GgmlDType::Q8_0
        } else if down {
            GgmlDType::Q6K
        } else {
            GgmlDType::Q4K
        };
        tensors.insert(name, QTensor::quantize(&value.contiguous()?, dtype)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let m = cfg.moe_intermediate_size;
    put(
        "token_embd.weight".into(),
        vb.get((cfg.vocab_size, h), "model.embed_tokens.weight")?,
        false,
        true,
    )?;
    put(
        "token_embd_norm.weight".into(),
        vb.get(h, "model.embedding_norm.weight")?,
        true,
        false,
    )?;
    for i in 0..cfg.num_hidden_layers {
        let root = vb.pp("model.layers").pp(i);
        let p = format!("blk.{i}");
        put(
            format!("{p}.attn_norm.weight"),
            root.get(h, "operator_norm.weight")?,
            true,
            false,
        )?;
        put(
            format!("{p}.ffn_norm.weight"),
            root.get(h, "ffn_norm.weight")?,
            true,
            false,
        )?;
        if matches!(cfg.layer_types[i], LayerType::Conv) {
            put(
                format!("{p}.shortconv.in_proj.weight"),
                root.get((3 * h, h), "conv.in_proj.weight")?,
                false,
                false,
            )?;
            put(
                format!("{p}.shortconv.out_proj.weight"),
                root.get((h, h), "conv.out_proj.weight")?,
                false,
                false,
            )?;
            put(
                format!("{p}.shortconv.conv.weight"),
                root.get((h, 1, cfg.conv_l_cache), "conv.conv.weight")?
                    .squeeze(1)?,
                true,
                false,
            )?;
        } else {
            let hd = h / cfg.num_attention_heads;
            for (dst, src, n) in [
                ("attn_q", "q_proj", h),
                ("attn_k", "k_proj", hd * cfg.num_key_value_heads),
                ("attn_v", "v_proj", hd * cfg.num_key_value_heads),
                ("attn_output", "out_proj", h),
            ] {
                put(
                    format!("{p}.{dst}.weight"),
                    root.get((n, h), &format!("self_attn.{src}.weight"))?,
                    false,
                    false,
                )?;
            }
            for (dst, src) in [
                ("attn_q_norm", "q_layernorm"),
                ("attn_k_norm", "k_layernorm"),
            ] {
                put(
                    format!("{p}.{dst}.weight"),
                    root.get(hd, &format!("self_attn.{src}.weight"))?,
                    true,
                    false,
                )?;
            }
        }
        let root = root.pp("feed_forward");
        if i < cfg.num_dense_layers {
            for (dst, src, shape) in [
                ("ffn_gate", "w1", (ff, h)),
                ("ffn_up", "w3", (ff, h)),
                ("ffn_down", "w2", (h, ff)),
            ] {
                put(
                    format!("{p}.{dst}.weight"),
                    root.get(shape, &format!("{src}.weight"))?,
                    false,
                    dst == "ffn_down",
                )?;
            }
        } else {
            put(
                format!("{p}.ffn_gate_inp.weight"),
                root.get((cfg.num_experts, h), "gate.weight")?,
                true,
                false,
            )?;
            put(
                format!("{p}.exp_probs_b.bias"),
                root.get(cfg.num_experts, "expert_bias")?,
                true,
                false,
            )?;
            for (dst, src, shape) in [
                ("ffn_gate_exps", "w1", (m, h)),
                ("ffn_up_exps", "w3", (m, h)),
                ("ffn_down_exps", "w2", (h, m)),
            ] {
                let bank = (0..cfg.num_experts)
                    .map(|j| {
                        root.pp("experts")
                            .pp(j)
                            .get(shape, &format!("{src}.weight"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                put(
                    format!("{p}.{dst}.weight"),
                    Tensor::stack(&bank, 0)?,
                    false,
                    dst == "ffn_down_exps",
                )?;
            }
        }
    }
    let u = |v: usize| gguf_file::Value::U32(v as u32);
    let kv = cfg
        .layer_types
        .iter()
        .map(|kind| {
            u(if matches!(kind, LayerType::Conv) {
                0
            } else {
                cfg.num_key_value_heads
            })
        })
        .collect();
    let metadata = BTreeMap::from([
        (
            "general.architecture",
            gguf_file::Value::String("lfm2moe".into()),
        ),
        ("lfm2moe.embedding_length", u(h)),
        ("lfm2moe.block_count", u(cfg.num_hidden_layers)),
        ("lfm2moe.attention.head_count", u(cfg.num_attention_heads)),
        (
            "lfm2moe.attention.head_count_kv",
            gguf_file::Value::Array(kv),
        ),
        (
            "lfm2moe.context_length",
            u(context.unwrap_or(cfg.max_position_embeddings)),
        ),
        ("lfm2moe.vocab_size", u(cfg.vocab_size)),
        ("lfm2moe.feed_forward_length", u(ff)),
        ("lfm2moe.expert_feed_forward_length", u(m)),
        ("lfm2moe.expert_count", u(cfg.num_experts)),
        ("lfm2moe.expert_used_count", u(cfg.num_experts_per_tok)),
        ("lfm2moe.leading_dense_block_count", u(cfg.num_dense_layers)),
        ("lfm2moe.expert_gating_func", u(2)),
        ("lfm2moe.shortconv.l_cache", u(cfg.conv_l_cache)),
        (
            "lfm2moe.attention.layer_norm_rms_epsilon",
            gguf_file::Value::F32(cfg.norm_eps as f32),
        ),
        ("lfm2moe.rope.freq_base", gguf_file::Value::F32(10000.)),
        ("tokenizer.ggml.eos_token_id", u(2)),
    ]);
    let mut output = Cursor::new(Vec::new());
    gguf_file::write(
        &mut output,
        &metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>(),
        &tensors
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
    )?;
    Ok(output.into_inner())
}

fn load(bytes: &[u8], device: &Device) -> Result<ModelWeights> {
    let mut reader = Cursor::new(bytes);
    let ct = gguf_file::Content::read(&mut reader)?;
    ModelWeights::from_gguf(ct, &mut reader, device)
}

fn dequantized_reference(bytes: &[u8]) -> Result<lfm2_moe::Model> {
    let device = Device::Cpu;
    let mut reader = Cursor::new(bytes);
    let ct = gguf_file::Content::read(&mut reader)?;
    let mut weights = std::collections::HashMap::new();
    for name in ct.tensor_infos.keys() {
        let tensor = ct.tensor(&mut reader, name, &device)?.dequantize(&device)?;
        if name == "token_embd.weight" {
            weights.insert("model.embed_tokens.weight".into(), tensor);
            continue;
        }
        if name == "token_embd_norm.weight" {
            weights.insert("model.embedding_norm.weight".into(), tensor);
            continue;
        }
        let mut parts = name.splitn(3, '.');
        let _ = parts.next();
        let index = parts.next().expect("fixture block");
        let role = parts.next().expect("fixture role");
        let root = format!("model.layers.{index}");
        let suffix = match role {
            "attn_norm.weight" => "operator_norm.weight",
            "ffn_norm.weight" => "ffn_norm.weight",
            "shortconv.in_proj.weight" => "conv.in_proj.weight",
            "shortconv.out_proj.weight" => "conv.out_proj.weight",
            "shortconv.conv.weight" => "conv.conv.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.out_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_layernorm.weight",
            "attn_k_norm.weight" => "self_attn.k_layernorm.weight",
            "ffn_gate.weight" => "feed_forward.w1.weight",
            "ffn_up.weight" => "feed_forward.w3.weight",
            "ffn_down.weight" => "feed_forward.w2.weight",
            "ffn_gate_inp.weight" => "feed_forward.gate.weight",
            "exp_probs_b.bias" => "feed_forward.expert_bias",
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
                let w = match role {
                    "ffn_gate_exps.weight" => "w1",
                    "ffn_up_exps.weight" => "w3",
                    _ => "w2",
                };
                for expert in 0..tensor.dim(0)? {
                    weights.insert(
                        format!("{root}.feed_forward.experts.{expert}.{w}.weight"),
                        tensor.i(expert)?,
                    );
                }
                continue;
            }
            _ => panic!("unknown fixture role {role}"),
        };
        let tensor = if role == "shortconv.conv.weight" {
            tensor.unsqueeze(1)?
        } else {
            tensor
        };
        weights.insert(format!("{root}.{suffix}"), tensor);
    }
    let cfg: lfm2_moe::Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
    lfm2_moe::Model::new(&cfg, VarBuilder::from_tensors(weights, DType::F32, &device))
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        error.is_finite() && error <= tolerance,
        "maximum error {error}"
    );
    Ok(())
}

#[test]
fn gguf_f32_matches_official_prefill_decode_and_recovery() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::from_slice_safetensors(NATIVE, DType::F32, &device)?;
    let bytes = fixture(0)?;
    let mut model = load(&bytes, &device)?;
    let ids = vb.get((1, 4), "oracle.ids")?.to_dtype(DType::U32)?;
    let first = model.forward(&ids, 0)?;
    assert_close(
        &first,
        &vb.get((1, 4, 8), "oracle.logits")?.i((.., 3, ..))?,
        1e-5,
    )?;
    let decode = Tensor::new(&[[6u32]], &device)?;
    assert_close(
        &model.forward(&decode, 4)?,
        &vb.get((1, 1, 8), "oracle.decode_logits")?.squeeze(1)?,
        1e-5,
    )?;
    assert!(model.forward(&decode, 2).is_err());
    assert_close(&model.forward(&ids, 0)?, &first, 0.)?;
    let embeds = model.embed_tokens(&ids)?;
    model.clear_cache();
    assert_close(&model.forward_embeds(&embeds, 0)?, &first, 0.)?;
    assert!(model.forward_embeds(&(&embeds * f64::NAN)?, 0).is_err());
    assert!(model
        .forward(
            &Tensor::zeros(
                (1, model.metadata().context_length + 1),
                DType::U32,
                &device
            )?,
            0
        )
        .is_err());
    assert_close(&model.forward(&ids, 0)?, &first, 0.)?;
    Ok(())
}

#[test]
fn mixed_q4_q6_and_q8_preserve_cache_and_bounded_quantization_error() -> Result<()> {
    let device = Device::Cpu;
    let ids = Tensor::new(&[[1u32, 4, 3, 5]], &device)?;
    for mode in [1, 2, 3, 4] {
        let bytes = fixture(mode)?;
        let reference = dequantized_reference(&bytes)?;
        let cfg: lfm2_moe::Config = serde_json::from_str(CONFIG).map_err(candle::Error::wrap)?;
        let mut cache = lfm2_moe::Cache::new(true, DType::F32, &cfg, &device)?;
        let reference = reference.forward(&ids, 0, &mut cache)?;
        let mut model = load(&bytes, &device)?;
        assert_eq!(
            model.quantized_expert_matrices(),
            if mode <= 2 { 6 } else { 0 }
        );
        let first = model.forward(&ids, 0)?;
        assert_close(&first, &reference, if mode <= 2 { 0.02 } else { 1e-5 })?;
        model.clear_cache();
        model.forward(&ids.narrow(1, 0, 2)?, 0)?;
        assert_close(&model.forward(&ids.narrow(1, 2, 2)?, 2)?, &first, 1e-5)?;
        model.clear_cache();
        assert_close(&model.forward(&ids, 0)?, &first, 0.)?;
        let memory = model.metadata().memory_estimate(64, 16)?;
        assert!(memory.peak_bytes > memory.resident_bytes && memory.cache_bytes > 0);
    }
    Ok(())
}

#[test]
fn mixed_quantized_prefill_matches_decode_across_chunk_boundary() -> Result<()> {
    let device = &Device::Cpu;
    let ids = (0..130u32).map(|i| (i * 5 + i / 7) % 8).collect::<Vec<_>>();
    let ids = Tensor::from_vec(ids, (1, 130), device)?;
    let bytes = fixture_with_context(1, Some(130))?;
    let mut model = load(&bytes, device)?;
    let tokenwise = (0..130)
        .map(|position| model.forward(&ids.narrow(1, position, 1)?, position))
        .collect::<Result<Vec<_>>>()?;

    for length in [127, 128, 129] {
        let expected = &tokenwise[length - 1];
        let next = ids.narrow(1, length, 1)?;
        let expected_next = &tokenwise[length];
        let prefix = ids.narrow(1, 0, length)?;
        assert_close(&model.forward(&prefix, 0)?, expected, 1e-5)?;
        assert_close(&model.forward(&next, length)?, expected_next, 1e-5)?;

        let mut position = 0;
        let embeds = model.embed_tokens(&prefix)?;
        for chunk in [19, 1, 46, length - 66] {
            let logits = model.forward_embeds(&embeds.narrow(1, position, chunk)?, position)?;
            position += chunk;
            assert_close(&logits, &tokenwise[position - 1], 1e-5)?;
        }
        assert_eq!(position, length);
        assert_close(&model.forward(&next, length)?, expected_next, 1e-5)?;
    }
    Ok(())
}

#[test]
fn malformed_metadata_and_payloads_fail_before_loading() -> Result<()> {
    let bytes = fixture(1)?;
    for case in 0..9 {
        let mut reader = Cursor::new(bytes.as_slice());
        let mut ct = gguf_file::Content::read(&mut reader)?;
        match case {
            0 => {
                ct.metadata.insert(
                    "lfm2moe.expert_used_count".into(),
                    gguf_file::Value::U32(999),
                );
            }
            1 => {
                ct.tensor_infos.remove("blk.1.exp_probs_b.bias");
            }
            2 => {
                ct.tensor_infos
                    .get_mut("blk.1.ffn_down_exps.weight")
                    .expect("fixture tensor")
                    .shape = (2, 256, 255).into();
            }
            3 => {
                ct.metadata.insert(
                    "general.architecture".into(),
                    gguf_file::Value::String("lfm2".into()),
                );
            }
            4 => {
                ct.tensor_infos
                    .get_mut("blk.1.ffn_gate_exps.weight")
                    .expect("fixture tensor")
                    .offset = u64::MAX;
            }
            5 => {
                ct.metadata.insert(
                    "lfm2moe.attention.head_count_kv".into(),
                    gguf_file::Value::Array(vec![]),
                );
            }
            6 => {
                ct.tensor_infos
                    .get_mut("blk.1.ffn_gate_exps.weight")
                    .expect("fixture tensor")
                    .ggml_dtype = GgmlDType::Q4_0;
            }
            7 => {
                ct.tensor_infos
                    .get_mut("blk.1.ffn_gate_exps.weight")
                    .expect("fixture tensor")
                    .offset = 1;
            }
            _ => {
                let offset = ct.tensor_infos["blk.1.ffn_up_exps.weight"].offset;
                ct.tensor_infos
                    .get_mut("blk.1.ffn_gate_exps.weight")
                    .expect("fixture tensor")
                    .offset = offset;
            }
        }
        assert!(inspect_gguf_metadata(&ct).is_err(), "case {case}");
    }
    let mut reader = Cursor::new(bytes.as_slice());
    let ct = gguf_file::Content::read(&mut reader)?;
    let mut truncated = Cursor::new(bytes[..bytes.len() - 4096].to_vec());
    assert!(ModelWeights::from_gguf(ct, &mut truncated, &Device::Cpu).is_err());
    Ok(())
}

#[test]
fn dense_lower_precision_storage_accounts_for_f32_widening() -> Result<()> {
    let bytes = fixture(0)?;
    let mut reader = Cursor::new(bytes);
    let mut ct = gguf_file::Content::read(&mut reader)?;
    let original = inspect_gguf_metadata(&ct)?.memory_estimate(64, 16)?;
    ct.metadata.insert(
        "lfm2moe.expert_gating_func".into(),
        gguf_file::Value::I32(2),
    );
    assert!(inspect_gguf_metadata(&ct).is_ok());
    ct.metadata.insert(
        "lfm2moe.expert_gating_func".into(),
        gguf_file::Value::I32(-2),
    );
    assert!(inspect_gguf_metadata(&ct).is_err());
    ct.metadata.insert(
        "lfm2moe.expert_gating_func".into(),
        gguf_file::Value::I32(2),
    );
    for dtype in [GgmlDType::F16, GgmlDType::BF16] {
        ct.tensor_infos
            .get_mut("blk.1.ffn_gate_exps.weight")
            .expect("fixture tensor")
            .ggml_dtype = dtype;
        assert_eq!(
            inspect_gguf_metadata(&ct)?
                .memory_estimate(64, 16)?
                .resident_bytes,
            original.resident_bytes
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_quantized_fixture_matches_cpu_and_recovers() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    for mode in [1, 2, 3, 4] {
        let bytes = fixture(mode)?;
        let ids = Tensor::new(&[[1u32, 4, 3, 5]], &Device::Cpu)?;
        let decode = Tensor::new(&[[6u32]], &Device::Cpu)?;
        let mut cpu = load(&bytes, &Device::Cpu)?;
        let expected = cpu.forward(&ids, 0)?;
        let expected_decode = cpu.forward(&decode, 4)?;
        let mut gpu = load(&bytes, &cuda)?;
        assert_eq!(
            gpu.quantized_expert_matrices(),
            if mode <= 2 { 6 } else { 0 }
        );
        let ids = ids.to_device(&cuda)?;
        let decode = decode.to_device(&cuda)?;
        let prefill = gpu.forward(&ids, 0)?;
        assert_close(&prefill.to_device(&Device::Cpu)?, &expected, 0.02)?;
        let decoded = gpu.forward(&decode, 4)?;
        assert_close(&decoded.to_device(&Device::Cpu)?, &expected_decode, 0.02)?;
        assert!(gpu.forward(&Tensor::new(&[[999u32]], &cuda)?, 5).is_err());
        assert!(gpu.forward(&decode, 5).is_err());
        assert_close(&gpu.forward(&ids, 0)?, &prefill, 1e-5)?;
        assert_close(&gpu.forward(&decode, 4)?, &decoded, 1e-5)?;
    }
    Ok(())
}

#[test]
fn eos_uses_checked_integer_normalization() -> Result<()> {
    use gguf_file::Value;
    let bytes = fixture(0)?;
    let mut reader = Cursor::new(bytes);
    let mut ct = gguf_file::Content::read(&mut reader)?;
    let key = "tokenizer.ggml.eos_token_id";
    for value in [
        Value::U8(2),
        Value::U16(2),
        Value::U32(2),
        Value::U64(2),
        Value::I8(2),
        Value::I16(2),
        Value::I32(2),
        Value::I64(2),
    ] {
        ct.metadata.insert(key.into(), value);
        assert_eq!(inspect_gguf_metadata(&ct)?.eos_token_id(), Some(2));
    }
    for value in [
        Value::I32(-1),
        Value::U64(u64::MAX),
        Value::U32(8),
        Value::F32(2.),
        Value::Bool(true),
    ] {
        ct.metadata.insert(key.into(), value);
        assert!(inspect_gguf_metadata(&ct).is_err());
    }
    ct.metadata.remove(key);
    assert_eq!(inspect_gguf_metadata(&ct)?.eos_token_id(), None);
    Ok(())
}
