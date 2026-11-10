use crate::models::quantized_lfm2::Lfm2GgufMetadata;
use candle::{
    quantized::{gguf_file, GgmlDType},
    Result,
};

#[derive(Debug, Clone)]
pub(crate) struct Role {
    pub name: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub resident_bytes: usize,
    pub auxiliary: bool,
}

/// Fully validated tensor inventory; inspection does not read weight payloads.
#[derive(Debug, Clone)]
pub struct Lfm2MoeGgufMetadata {
    pub architecture: String,
    pub vocab_size: usize,
    pub embedding_length: usize,
    pub context_length: usize,
    pub block_count: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_feed_forward_length: usize,
    pub feed_forward_length: usize,
    pub leading_dense_block_count: usize,
    pub tensor_bytes: usize,
    pub(crate) shared: Lfm2GgufMetadata,
    pub(crate) roles: Vec<Role>,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub resident_bytes: usize,
    pub cache_bytes: usize,
    pub scratch_bytes: usize,
    pub peak_bytes: usize,
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| candle::Error::Msg("LFM2-MoE dimension/byte product overflow".into()))
    })
}
fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| candle::Error::Msg("LFM2-MoE byte sum overflow".into()))
    })
}

impl Lfm2MoeGgufMetadata {
    /// EOS normalized by the same checked integer/vocabulary validation as loading.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.shared.eos_token_id
    }

    /// Conservative live tensor/storage envelope, excluding OS file-cache accounting.
    pub fn memory_estimate(&self, context: usize, prefill_chunk: usize) -> Result<MemoryEstimate> {
        if context == 0
            || context > self.context_length
            || prefill_chunk == 0
            || prefill_chunk > context
        {
            candle::bail!("invalid LFM2-MoE memory-estimation context or prefill chunk")
        }
        let h = self.embedding_length;
        let head_dim = h / self.shared.head_count;
        let attention_heads = sum(&self.shared.head_count_kv)?;
        let kv = product(&[2, context, attention_heads, head_dim, 4])?;
        let conv_layers = self
            .shared
            .head_count_kv
            .iter()
            .filter(|&&n| n == 0)
            .count();
        let conv = product(&[conv_layers, h, self.shared.shortconv_l_cache, 4])?;
        let rotary = product(&[2, self.context_length, head_dim / 2, 4])?;
        let auxiliary = self
            .roles
            .iter()
            .filter(|r| r.auxiliary)
            .map(|r| product(&r.shape).and_then(|n| product(&[n, 4])))
            .collect::<Result<Vec<_>>>()?;
        // Auxiliary bytes are counted again to cover load-time conversion ownership.
        let weights = sum(&self
            .roles
            .iter()
            .map(|r| r.resident_bytes)
            .collect::<Vec<_>>())?;
        let resident = sum(&[weights, sum(&auxiliary)?, rotary])?;
        let scores = product(&[3, self.shared.head_count, prefill_chunk, context, 4])?;
        let repeated_kv = product(&[2, context, h, 4])?;
        let ff = product(&[
            8,
            prefill_chunk,
            self.feed_forward_length
                .max(self.expert_feed_forward_length),
            4,
        ])?;
        let routes = product(&[prefill_chunk, self.expert_count, 4, 4])?;
        let logits = product(&[self.vocab_size, 4])?;
        let input_embeds = product(&[context, h, 4])?;
        let hidden_scratch = product(&[4, prefill_chunk, h, 4])?;
        let scratch = sum(&[
            scores,
            repeated_kv,
            ff,
            routes,
            logits,
            input_embeds,
            hidden_scratch,
        ])?;
        let cache = sum(&[kv, conv])?;
        let peak = sum(&[resident, cache, scratch])?;
        Ok(MemoryEstimate {
            resident_bytes: resident,
            cache_bytes: cache,
            scratch_bytes: scratch,
            peak_bytes: peak,
        })
    }
}

fn number(value: &gguf_file::Value) -> Result<usize> {
    use gguf_file::Value::*;
    match value {
        U8(v) => Ok(*v as usize),
        U16(v) => Ok(*v as usize),
        U32(v) => Ok(*v as usize),
        U64(v) => usize::try_from(*v).map_err(candle::Error::wrap),
        I8(v) => usize::try_from(*v).map_err(candle::Error::wrap),
        I16(v) => usize::try_from(*v).map_err(candle::Error::wrap),
        I32(v) => usize::try_from(*v).map_err(candle::Error::wrap),
        I64(v) => usize::try_from(*v).map_err(candle::Error::wrap),
        _ => candle::bail!("LFM2-MoE metadata requires a nonnegative integer"),
    }
}

pub fn inspect_gguf_metadata(ct: &gguf_file::Content) -> Result<Lfm2MoeGgufMetadata> {
    let get = |key: &str| {
        ct.metadata
            .get(key)
            .ok_or_else(|| candle::Error::Msg(format!("missing LFM2-MoE GGUF metadata {key}")))
    };
    let int = |key: &str| {
        number(get(&format!("lfm2moe.{key}"))?)
            .map_err(|error| candle::Error::Msg(format!("invalid lfm2moe.{key}: {error}")))
    };
    let architecture = get("general.architecture")?.to_string()?.to_owned();
    if architecture != "lfm2moe" {
        candle::bail!("unsupported LFM2-MoE GGUF architecture {architecture}")
    }
    let h = int("embedding_length")?;
    let layers = int("block_count")?;
    let heads = int("attention.head_count")?;
    let context = int("context_length")?;
    let experts = int("expert_count")?;
    let topk = int("expert_used_count")?;
    let dense = int("leading_dense_block_count")?;
    let ff = int("feed_forward_length")?;
    let expert_ff = int("expert_feed_forward_length")?;
    let vocab = int("vocab_size")?;
    if h == 0
        || h > 65536
        || heads == 0
        || !h.is_multiple_of(heads)
        || !(h / heads).is_multiple_of(2)
        || layers == 0
        || layers > 512
        || context == 0
        || context > (1 << 22)
        || experts == 0
        || experts > 4096
        || topk == 0
        || topk > experts
        || dense > layers
        || ff == 0
        || expert_ff == 0
        || vocab == 0
        || vocab > (1 << 24)
    {
        candle::bail!("invalid LFM2-MoE GGUF dimensions, experts, top-K or dense prefix")
    }
    if int("expert_gating_func")? != 2 {
        candle::bail!("LFM2-MoE GGUF requires sigmoid expert gating")
    }
    let eps = get("lfm2moe.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let theta = get("lfm2moe.rope.freq_base")?.to_f32()?;
    let l_cache = int("shortconv.l_cache")?;
    if !eps.is_finite()
        || eps <= 0.
        || !theta.is_finite()
        || theta <= 0.
        || l_cache == 0
        || l_cache > context
    {
        candle::bail!("invalid LFM2-MoE normalization, RoPE or convolution length")
    }
    if product(&[context, h / heads / 2])? > 1 << 26 {
        candle::bail!("LFM2-MoE rotary cache exceeds element bound")
    }
    let kv = match get("lfm2moe.attention.head_count_kv")? {
        gguf_file::Value::Array(v) if v.len() == layers => {
            v.iter().map(number).collect::<Result<Vec<_>>>()?
        }
        gguf_file::Value::Array(_) => candle::bail!("LFM2-MoE KV head list must match block_count"),
        v => vec![number(v)?; layers],
    };
    if kv
        .iter()
        .any(|&n| n > heads || (n != 0 && !heads.is_multiple_of(n)))
    {
        candle::bail!("LFM2-MoE invalid KV head count")
    }
    let eos = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .map(number)
        .transpose()?;
    if eos.is_some_and(|n| n >= vocab) {
        candle::bail!("LFM2-MoE EOS is outside vocabulary")
    }
    let tied_output = !ct.tensor_infos.contains_key("output.weight");
    let shared = Lfm2GgufMetadata {
        architecture: architecture.clone(),
        embedding_length: h,
        context_length: context,
        block_count: layers,
        head_count: heads,
        head_count_kv: kv.clone(),
        rms_norm_eps: eps,
        rope_freq_base: theta,
        shortconv_l_cache: l_cache,
        eos_token_id: eos
            .map(u32::try_from)
            .transpose()
            .map_err(candle::Error::wrap)?,
        tied_output,
    };
    let mut roles = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, auxiliary: bool| -> Result<()> {
        let info = ct
            .tensor_infos
            .get(&name)
            .ok_or_else(|| candle::Error::Msg(format!("missing LFM2-MoE tensor {name}")))?;
        if info.shape.dims() != shape {
            candle::bail!(
                "LFM2-MoE tensor {name}: expected {shape:?}, got {:?}",
                info.shape
            )
        }
        let dense_type = matches!(
            info.ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        if !(dense_type
            || (!auxiliary
                && matches!(
                    info.ggml_dtype,
                    GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q8_0
                )))
        {
            candle::bail!(
                "unsupported LFM2-MoE dtype {:?} for {name}",
                info.ggml_dtype
            )
        }
        let elements = product(&shape)?;
        let last = shape.last().copied().unwrap_or(0);
        if last == 0 || !last.is_multiple_of(info.ggml_dtype.block_size()) {
            candle::bail!("invalid LFM2-MoE quantized row alignment for {name}")
        }
        let bytes = product(&[
            elements / info.ggml_dtype.block_size(),
            info.ggml_dtype.type_size(),
        ])?;
        ct.tensor_data_offset
            .checked_add(info.offset)
            .and_then(|n| n.checked_add(bytes as u64))
            .ok_or_else(|| candle::Error::Msg(format!("LFM2-MoE tensor range overflow: {name}")))?;
        roles.push(Role {
            shape,
            bytes,
            resident_bytes: if dense_type {
                sum(&[
                    product(&[elements, 4])?,
                    if tied_output && name == "token_embd.weight" {
                        bytes
                    } else {
                        0
                    },
                ])?
            } else {
                bytes
            },
            name,
            auxiliary,
        });
        Ok(())
    };
    add("token_embd.weight".into(), vec![vocab, h], false)?;
    add("token_embd_norm.weight".into(), vec![h], true)?;
    if !tied_output {
        add("output.weight".into(), vec![vocab, h], false)?;
    }
    let head_dim = h / heads;
    for (i, &kv_heads) in kv.iter().enumerate() {
        let p = format!("blk.{i}");
        add(format!("{p}.attn_norm.weight"), vec![h], true)?;
        add(format!("{p}.ffn_norm.weight"), vec![h], true)?;
        if kv_heads == 0 {
            add(
                format!("{p}.shortconv.in_proj.weight"),
                vec![product(&[3, h])?, h],
                false,
            )?;
            add(format!("{p}.shortconv.out_proj.weight"), vec![h, h], false)?;
            let key = format!("{p}.shortconv.conv.weight");
            let shape = if ct
                .tensor_infos
                .get(&key)
                .is_some_and(|v| v.shape.dims() == [h, 1, l_cache])
            {
                vec![h, 1, l_cache]
            } else {
                vec![h, l_cache]
            };
            add(key, shape, true)?;
        } else {
            for (name, n) in [
                ("attn_q", h),
                ("attn_k", product(&[kv_heads, head_dim])?),
                ("attn_v", product(&[kv_heads, head_dim])?),
                ("attn_output", h),
            ] {
                add(format!("{p}.{name}.weight"), vec![n, h], false)?;
            }
            for name in ["attn_q_norm", "attn_k_norm"] {
                add(format!("{p}.{name}.weight"), vec![head_dim], true)?;
            }
        }
        if i < dense {
            add(format!("{p}.ffn_gate.weight"), vec![ff, h], false)?;
            add(format!("{p}.ffn_up.weight"), vec![ff, h], false)?;
            add(format!("{p}.ffn_down.weight"), vec![h, ff], false)?;
        } else {
            add(format!("{p}.ffn_gate_inp.weight"), vec![experts, h], true)?;
            add(format!("{p}.exp_probs_b.bias"), vec![experts], true)?;
            add(
                format!("{p}.ffn_gate_exps.weight"),
                vec![experts, expert_ff, h],
                false,
            )?;
            add(
                format!("{p}.ffn_up_exps.weight"),
                vec![experts, expert_ff, h],
                false,
            )?;
            add(
                format!("{p}.ffn_down_exps.weight"),
                vec![experts, h, expert_ff],
                false,
            )?;
        }
    }
    if roles.len() != ct.tensor_infos.len() {
        candle::bail!("LFM2-MoE GGUF contains tensors outside the validated inventory")
    }
    let alignment = ct
        .metadata
        .get("general.alignment")
        .map(number)
        .transpose()?
        .unwrap_or(32);
    if !alignment.is_power_of_two() || alignment > 4096 {
        candle::bail!("invalid LFM2-MoE GGUF tensor alignment")
    }
    let mut ranges = roles
        .iter()
        .map(|role| {
            let info = ct
                .tensor_infos
                .get(&role.name)
                .ok_or_else(|| candle::Error::Msg("missing validated tensor".into()))?;
            if !info.offset.is_multiple_of(alignment as u64) {
                candle::bail!("unaligned LFM2-MoE tensor offset: {}", role.name)
            }
            let end = info
                .offset
                .checked_add(role.bytes as u64)
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE tensor end overflow".into()))?;
            Ok((info.offset, end))
        })
        .collect::<Result<Vec<_>>>()?;
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        candle::bail!("overlapping LFM2-MoE GGUF tensor payloads")
    }
    let tensor_bytes = sum(&roles.iter().map(|r| r.bytes).collect::<Vec<_>>())?;
    Ok(Lfm2MoeGgufMetadata {
        architecture,
        vocab_size: vocab,
        embedding_length: h,
        context_length: context,
        block_count: layers,
        expert_count: experts,
        expert_used_count: topk,
        expert_feed_forward_length: expert_ff,
        feed_forward_length: ff,
        leading_dense_block_count: dense,
        tensor_bytes,
        shared,
        roles,
    })
}
