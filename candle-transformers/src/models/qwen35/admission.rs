//! Bounded GGUF admission for the Qwen3.5 Q8_0/F32 text trunk.

use candle::quantized::{gguf_file, GgmlDType};
use candle::{bail, Result};
use gguf_file::{Content, ContentReadLimits, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

/// Caller-selected limits. There are no format-wide defaults for a model load.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionLimits {
    pub max_file_bytes: u64,
    pub max_header_bytes: u64,
    pub max_tensors: u64,
    pub max_metadata: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_layers: usize,
    pub max_tensor_bytes: u64,
    pub max_total_tensor_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub block_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub attention_head_dim: usize,
    pub rope_dimension_count: usize,
    pub rope_sections: [usize; 4],
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub conv_kernel: usize,
    pub state_size: usize,
    pub key_groups: usize,
    pub value_heads: usize,
    pub inner_size: usize,
    pub full_attention_interval: usize,
}

impl Config {
    pub fn is_recurrent(&self, layer: usize) -> bool {
        self.full_attention_interval > 0
            && layer % self.full_attention_interval != self.full_attention_interval - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    Dense,
    Matrix,
}

#[derive(Debug)]
pub struct AdmittedTensor {
    pub shape: Vec<usize>,
    pub dtype: GgmlDType,
    pub byte_start: u64,
    pub byte_len: u64,
}

/// The file stays open after the identity and entire directory are validated.
#[derive(Debug)]
pub struct Artifact {
    pub config: Config,
    pub sha256: String,
    pub tensors: BTreeMap<String, AdmittedTensor>,
    file: File,
}

impl Artifact {
    pub fn open(path: &Path, expected_sha256: &str, limits: AdmissionLimits) -> Result<Self> {
        if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("qwen35: expected SHA-256 must be 64 hexadecimal characters")
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        options.share_mode(0x0000_0001);
        let mut file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limits.max_file_bytes {
            bail!("qwen35: file is not regular or exceeds caller byte limit")
        }
        let mut hasher = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let count = file.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            hasher.update(&chunk[..count]);
        }
        let sha256 = format!("{:x}", hasher.finalize());
        if !sha256.eq_ignore_ascii_case(expected_sha256) {
            bail!("qwen35: GGUF SHA-256 mismatch: expected {expected_sha256}, got {sha256}")
        }
        file.seek(SeekFrom::Start(0))?;
        let content = Content::read_with_limits(
            &mut file,
            ContentReadLimits {
                max_tensor_count: limits.max_tensors,
                max_metadata_count: limits.max_metadata,
                max_string_length: limits.max_string_bytes,
                max_array_elements: limits.max_array_elements,
                max_header_bytes: limits.max_header_bytes,
            },
        )?;
        let (config, tensors) = validate(&content, metadata.len(), &limits)?;
        Ok(Self {
            config,
            sha256,
            tensors,
            file,
        })
    }

    pub fn file(&mut self) -> &mut File {
        &mut self.file
    }
}

fn required<'a>(content: &'a Content, name: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(name)
        .ok_or_else(|| candle::Error::Msg(format!("qwen35: missing GGUF metadata {name}")))
}

fn number(content: &Content, name: &str) -> Result<usize> {
    let value = required(content, name)?.to_u64()?;
    usize::try_from(value)
        .map_err(|_| candle::Error::Msg(format!("qwen35: metadata {name} exceeds usize")))
}

fn positive(content: &Content, name: &str) -> Result<usize> {
    let value = number(content, name)?;
    if value == 0 {
        bail!("qwen35: metadata {name} must be positive")
    }
    Ok(value)
}

fn finite_f32(content: &Content, name: &str) -> Result<f32> {
    let value = required(content, name)?.to_f32()?;
    if !value.is_finite() || value <= 0.0 {
        bail!("qwen35: metadata {name} must be finite and positive")
    }
    Ok(value)
}

fn config(content: &Content, limits: &AdmissionLimits) -> Result<Config> {
    if required(content, "general.architecture")?.to_string()? != "qwen35" {
        bail!("qwen35: general.architecture must be qwen35")
    }
    if required(content, "tokenizer.ggml.model")?.to_string()? != "gpt2"
        || required(content, "tokenizer.ggml.pre")?.to_string()? != "qwen35"
    {
        bail!("qwen35: expected gpt2 tokenizer with qwen35 pre-tokenizer")
    }
    if content.metadata.contains_key("qwen35.nextn_predict_layers") {
        bail!("qwen35: MTP inventory is outside this text-trunk admission")
    }
    if content
        .metadata
        .contains_key("qwen35.attention.recurrent_layers")
    {
        bail!("qwen35: explicit recurrent layer map is not supported")
    }
    let vocab_size = required(content, "tokenizer.ggml.tokens")?.to_vec()?.len();
    if vocab_size == 0 {
        bail!("qwen35: tokenizer vocabulary is empty")
    }
    let context_length = positive(content, "qwen35.context_length")?;
    let block_count = positive(content, "qwen35.block_count")?;
    if block_count > limits.max_layers {
        bail!(
            "qwen35: block count {block_count} exceeds caller limit {}",
            limits.max_layers
        )
    }
    let hidden_size = positive(content, "qwen35.embedding_length")?;
    let intermediate_size = positive(content, "qwen35.feed_forward_length")?;
    let num_attention_heads = positive(content, "qwen35.attention.head_count")?;
    let num_kv_heads = positive(content, "qwen35.attention.head_count_kv")?;
    let attention_head_dim = positive(content, "qwen35.attention.key_length")?;
    let value_head_dim = positive(content, "qwen35.attention.value_length")?;
    if num_attention_heads % num_kv_heads != 0 || value_head_dim != attention_head_dim {
        bail!("qwen35: incompatible attention head counts or K/V widths")
    }
    let rope_dimension_count = positive(content, "qwen35.rope.dimension_count")?;
    let sections = required(content, "qwen35.rope.dimension_sections")?.to_vec()?;
    if sections.len() != 4 {
        bail!("qwen35: rope.dimension_sections must have four entries")
    }
    let mut rope_sections = [0usize; 4];
    for (destination, source) in rope_sections.iter_mut().zip(sections) {
        let value = source.to_i32()?;
        *destination = usize::try_from(value)
            .map_err(|_| candle::Error::Msg("qwen35: negative RoPE section".into()))?;
    }
    let section_pairs = rope_sections
        .iter()
        .try_fold(0usize, |sum, section| sum.checked_add(*section))
        .ok_or_else(|| candle::Error::Msg("qwen35: RoPE section sum overflow".into()))?;
    if section_pairs.checked_mul(2) != Some(rope_dimension_count)
        || rope_dimension_count > attention_head_dim
    {
        bail!("qwen35: RoPE sections do not match rotary dimensions")
    }
    let rope_theta = finite_f32(content, "qwen35.rope.freq_base")?;
    let rms_norm_eps = finite_f32(content, "qwen35.attention.layer_norm_rms_epsilon")?;
    let conv_kernel = positive(content, "qwen35.ssm.conv_kernel")?;
    let state_size = positive(content, "qwen35.ssm.state_size")?;
    let key_groups = positive(content, "qwen35.ssm.group_count")?;
    let value_heads = positive(content, "qwen35.ssm.time_step_rank")?;
    let inner_size = positive(content, "qwen35.ssm.inner_size")?;
    let full_attention_interval = positive(content, "qwen35.full_attention_interval")?;
    if full_attention_interval > block_count
        || key_groups > value_heads
        || value_heads % key_groups != 0
        || state_size.checked_mul(value_heads) != Some(inner_size)
    {
        bail!("qwen35: inconsistent recurrent or layer schedule metadata")
    }
    Ok(Config {
        vocab_size,
        context_length,
        block_count,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        attention_head_dim,
        rope_dimension_count,
        rope_sections,
        rope_theta,
        rms_norm_eps,
        conv_kernel,
        state_size,
        key_groups,
        value_heads,
        inner_size,
        full_attention_interval,
    })
}

type Expected = BTreeMap<String, (Vec<usize>, TensorKind)>;

fn expect(expected: &mut Expected, name: String, shape: Vec<usize>, kind: TensorKind) {
    expected.insert(name, (shape, kind));
}

fn expected_tensors(config: &Config) -> Result<Expected> {
    let mut expected = BTreeMap::new();
    let hidden = config.hidden_size;
    let ffn = config.intermediate_size;
    let key_width = config
        .state_size
        .checked_mul(config.key_groups)
        .ok_or_else(|| candle::Error::Msg("qwen35: key width overflow".into()))?;
    let qkv_width = key_width
        .checked_mul(2)
        .and_then(|n| n.checked_add(config.inner_size))
        .ok_or_else(|| candle::Error::Msg("qwen35: QKV width overflow".into()))?;
    let conv_width = qkv_width;
    let query_width = config
        .num_attention_heads
        .checked_mul(config.attention_head_dim)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| candle::Error::Msg("qwen35: query width overflow".into()))?;
    let kv_width = config
        .num_kv_heads
        .checked_mul(config.attention_head_dim)
        .ok_or_else(|| candle::Error::Msg("qwen35: KV width overflow".into()))?;
    let attention_width = config
        .num_attention_heads
        .checked_mul(config.attention_head_dim)
        .ok_or_else(|| candle::Error::Msg("qwen35: attention width overflow".into()))?;
    expect(
        &mut expected,
        "token_embd.weight".into(),
        vec![config.vocab_size, hidden],
        TensorKind::Matrix,
    );
    expect(
        &mut expected,
        "output_norm.weight".into(),
        vec![hidden],
        TensorKind::Dense,
    );
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        expect(
            &mut expected,
            format!("{prefix}.attn_norm.weight"),
            vec![hidden],
            TensorKind::Dense,
        );
        expect(
            &mut expected,
            format!("{prefix}.post_attention_norm.weight"),
            vec![hidden],
            TensorKind::Dense,
        );
        expect(
            &mut expected,
            format!("{prefix}.ffn_gate.weight"),
            vec![ffn, hidden],
            TensorKind::Matrix,
        );
        expect(
            &mut expected,
            format!("{prefix}.ffn_up.weight"),
            vec![ffn, hidden],
            TensorKind::Matrix,
        );
        expect(
            &mut expected,
            format!("{prefix}.ffn_down.weight"),
            vec![hidden, ffn],
            TensorKind::Matrix,
        );
        if config.is_recurrent(layer) {
            for (suffix, shape, kind) in [
                (
                    "attn_qkv.weight",
                    vec![qkv_width, hidden],
                    TensorKind::Matrix,
                ),
                (
                    "attn_gate.weight",
                    vec![config.inner_size, hidden],
                    TensorKind::Matrix,
                ),
                (
                    "ssm_alpha.weight",
                    vec![config.value_heads, hidden],
                    TensorKind::Matrix,
                ),
                (
                    "ssm_beta.weight",
                    vec![config.value_heads, hidden],
                    TensorKind::Matrix,
                ),
                (
                    "ssm_conv1d.weight",
                    vec![conv_width, config.conv_kernel],
                    TensorKind::Dense,
                ),
                ("ssm_a", vec![config.value_heads], TensorKind::Dense),
                ("ssm_dt.bias", vec![config.value_heads], TensorKind::Dense),
                (
                    "ssm_norm.weight",
                    vec![config.state_size],
                    TensorKind::Dense,
                ),
                (
                    "ssm_out.weight",
                    vec![hidden, config.inner_size],
                    TensorKind::Matrix,
                ),
            ] {
                expect(&mut expected, format!("{prefix}.{suffix}"), shape, kind);
            }
        } else {
            for (suffix, shape, kind) in [
                (
                    "attn_q.weight",
                    vec![query_width, hidden],
                    TensorKind::Matrix,
                ),
                ("attn_k.weight", vec![kv_width, hidden], TensorKind::Matrix),
                ("attn_v.weight", vec![kv_width, hidden], TensorKind::Matrix),
                (
                    "attn_output.weight",
                    vec![hidden, attention_width],
                    TensorKind::Matrix,
                ),
                (
                    "attn_q_norm.weight",
                    vec![config.attention_head_dim],
                    TensorKind::Dense,
                ),
                (
                    "attn_k_norm.weight",
                    vec![config.attention_head_dim],
                    TensorKind::Dense,
                ),
            ] {
                expect(&mut expected, format!("{prefix}.{suffix}"), shape, kind);
            }
        }
    }
    Ok(expected)
}

fn validate(
    content: &Content,
    file_size: u64,
    limits: &AdmissionLimits,
) -> Result<(Config, BTreeMap<String, AdmittedTensor>)> {
    let config = config(content, limits)?;
    let mut expected = expected_tensors(&config)?;
    if content.tensor_infos.contains_key("output.weight") {
        expect(
            &mut expected,
            "output.weight".into(),
            vec![config.vocab_size, config.hidden_size],
            TensorKind::Matrix,
        );
    }
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    let mut total_bytes = 0u64;
    for (name, info) in &content.tensor_infos {
        let (shape, kind) = expected
            .remove(name)
            .ok_or_else(|| candle::Error::Msg(format!("qwen35: unexpected tensor {name}")))?;
        if info.shape.dims() != shape {
            bail!(
                "qwen35: tensor {name} shape {:?}, expected {shape:?}",
                info.shape.dims()
            )
        }
        match (kind, info.ggml_dtype) {
            (TensorKind::Dense, GgmlDType::F32)
            | (TensorKind::Matrix, GgmlDType::F32 | GgmlDType::Q8_0) => (),
            _ => bail!(
                "qwen35: tensor {name} has unsupported dtype {:?}",
                info.ggml_dtype
            ),
        }
        let element_count = shape
            .iter()
            .try_fold(1u64, |n, dimension| {
                u64::try_from(*dimension)
                    .ok()
                    .and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| candle::Error::Msg(format!("qwen35: tensor {name} size overflow")))?;
        let block = info.ggml_dtype.block_size() as u64;
        if element_count == 0 || element_count % block != 0 {
            bail!("qwen35: tensor {name} has invalid element/block count")
        }
        let byte_len = element_count
            .checked_div(block)
            .and_then(|n| n.checked_mul(info.ggml_dtype.type_size() as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!("qwen35: tensor {name} byte length overflow"))
            })?;
        if byte_len > limits.max_tensor_bytes {
            bail!("qwen35: tensor {name} exceeds caller tensor-byte limit")
        }
        total_bytes = total_bytes
            .checked_add(byte_len)
            .ok_or_else(|| candle::Error::Msg("qwen35: tensor byte total overflow".into()))?;
        if total_bytes > limits.max_total_tensor_bytes {
            bail!("qwen35: tensor bytes exceed caller total limit")
        }
        let start = content
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| candle::Error::Msg(format!("qwen35: tensor {name} offset overflow")))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| candle::Error::Msg(format!("qwen35: tensor {name} end overflow")))?;
        if end > file_size || info.offset % gguf_file::DEFAULT_ALIGNMENT != 0 {
            bail!("qwen35: tensor {name} range or alignment is invalid")
        }
        ranges.push((start, end, name.as_str()));
        tensors.insert(
            name.clone(),
            AdmittedTensor {
                shape,
                dtype: info.ggml_dtype,
                byte_start: start,
                byte_len,
            },
        );
    }
    if let Some((name, _)) = expected.into_iter().next() {
        bail!("qwen35: missing tensor {name}")
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            bail!(
                "qwen35: tensor ranges overlap: {} and {}",
                pair[0].2,
                pair[1].2
            )
        }
    }
    Ok((config, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::{TensorInfo, VersionedMagic};
    use candle::Shape;
    use std::collections::HashMap;

    fn limits() -> AdmissionLimits {
        AdmissionLimits {
            max_file_bytes: 1_000_000,
            max_header_bytes: 100_000,
            max_tensors: 100,
            max_metadata: 64,
            max_string_bytes: 10_000,
            max_array_elements: 100,
            max_layers: 4,
            max_tensor_bytes: 1_000_000,
            max_total_tensor_bytes: 1_000_000,
        }
    }

    #[test]
    fn admits_retained_exact_inventory_without_model_bytes() -> Result<()> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/inspection.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("fixture JSON: {error}")))?;
        let metadata_json = fixture["metadata"]
            .as_object()
            .ok_or_else(|| candle::Error::Msg("fixture metadata missing".into()))?;
        let mut metadata = HashMap::new();
        for (key, value) in metadata_json {
            if key.starts_with("qwen35.") {
                let value = if let Some(array) = value.as_array() {
                    Value::Array(
                        array
                            .iter()
                            .map(|item| {
                                item.as_i64()
                                    .and_then(|v| i32::try_from(v).ok())
                                    .map(Value::I32)
                                    .ok_or_else(|| candle::Error::Msg("fixture section".into()))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )
                } else if let Some(integer) = value.as_u64() {
                    Value::U32(
                        u32::try_from(integer)
                            .map_err(|_| candle::Error::Msg("fixture integer".into()))?,
                    )
                } else {
                    Value::F32(
                        value
                            .as_f64()
                            .ok_or_else(|| candle::Error::Msg("fixture float".into()))?
                            as f32,
                    )
                };
                metadata.insert(key.clone(), value);
            }
        }
        metadata.insert(
            "general.architecture".into(),
            Value::String("qwen35".into()),
        );
        metadata.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
        metadata.insert("tokenizer.ggml.pre".into(), Value::String("qwen35".into()));
        let vocab_size = fixture["tokenizer"]["tokenizer.ggml.tokens"]["items"]
            .as_u64()
            .ok_or_else(|| candle::Error::Msg("fixture vocab missing".into()))?;
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); vocab_size as usize]),
        );
        let tensor_json = fixture["tensors"]
            .as_object()
            .ok_or_else(|| candle::Error::Msg("fixture tensors missing".into()))?;
        let data_start = tensor_json
            .values()
            .filter_map(|value| value["data_offset"].as_u64())
            .min()
            .ok_or_else(|| candle::Error::Msg("fixture tensor offsets missing".into()))?;
        let mut tensor_infos = HashMap::new();
        for (name, value) in tensor_json {
            let mut shape = value["shape"]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("fixture shape missing".into()))?
                .iter()
                .map(|dim| {
                    dim.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| candle::Error::Msg("fixture dimension".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            shape.reverse();
            let ggml_dtype = match value["type"].as_str() {
                Some("F32") => GgmlDType::F32,
                Some("Q8_0") => GgmlDType::Q8_0,
                _ => bail!("fixture tensor dtype"),
            };
            let offset = value["data_offset"]
                .as_u64()
                .and_then(|n| n.checked_sub(data_start))
                .ok_or_else(|| candle::Error::Msg("fixture tensor offset".into()))?;
            tensor_infos.insert(
                name.clone(),
                TensorInfo {
                    ggml_dtype,
                    shape: Shape::from(shape),
                    offset,
                },
            );
        }
        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: data_start,
        };
        let limits = AdmissionLimits {
            max_file_bytes: 5_000_000_000,
            max_header_bytes: 16 * 1024 * 1024,
            max_tensors: 500,
            max_metadata: 64,
            max_string_bytes: 1_000_000,
            max_array_elements: 250_000,
            max_layers: 32,
            max_tensor_bytes: 700_000_000,
            max_total_tensor_bytes: 4_500_000_000,
        };
        let (config, tensors) = validate(&content, 4_482_403_200, &limits)?;
        assert_eq!(config.block_count, 32);
        assert_eq!(tensors.len(), 426);
        assert_eq!(
            tensors
                .values()
                .filter(|t| t.dtype == GgmlDType::Q8_0)
                .count(),
            249
        );
        assert_eq!(
            tensors
                .values()
                .filter(|t| t.dtype == GgmlDType::F32)
                .count(),
            177
        );
        Ok(())
    }

    fn synthetic_content() -> Content {
        let mut metadata = HashMap::new();
        for (key, value) in [
            ("general.architecture", Value::String("qwen35".into())),
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            ("tokenizer.ggml.pre", Value::String("qwen35".into())),
            ("qwen35.context_length", Value::U32(128)),
            ("qwen35.block_count", Value::U32(2)),
            ("qwen35.embedding_length", Value::U32(32)),
            ("qwen35.feed_forward_length", Value::U32(64)),
            ("qwen35.attention.head_count", Value::U32(2)),
            ("qwen35.attention.head_count_kv", Value::U32(1)),
            ("qwen35.attention.key_length", Value::U32(16)),
            ("qwen35.attention.value_length", Value::U32(16)),
            ("qwen35.rope.dimension_count", Value::U32(16)),
            ("qwen35.rope.freq_base", Value::F32(1_000_000.0)),
            ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("qwen35.ssm.conv_kernel", Value::U32(4)),
            ("qwen35.ssm.state_size", Value::U32(16)),
            ("qwen35.ssm.group_count", Value::U32(2)),
            ("qwen35.ssm.time_step_rank", Value::U32(4)),
            ("qwen35.ssm.inner_size", Value::U32(64)),
            ("qwen35.full_attention_interval", Value::U32(2)),
        ] {
            metadata.insert(key.into(), value);
        }
        metadata.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array(vec![
                Value::I32(4),
                Value::I32(4),
                Value::I32(0),
                Value::I32(0),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array((0..32).map(|i| Value::String(i.to_string())).collect()),
        );
        let mut content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };
        let config = config(&content, &limits()).expect("valid synthetic config");
        let mut offset = 0u64;
        for (name, (shape, _)) in expected_tensors(&config).expect("valid synthetic shapes") {
            let count = shape.iter().product::<usize>() as u64;
            content.tensor_infos.insert(
                name,
                TensorInfo {
                    ggml_dtype: GgmlDType::F32,
                    shape: Shape::from(shape),
                    offset,
                },
            );
            offset = (offset + count * 4).div_ceil(32) * 32;
        }
        content
    }

    #[test]
    fn admits_checked_synthetic_inventory() -> Result<()> {
        let content = synthetic_content();
        let (_, tensors) = validate(&content, 1_000_000, &limits())?;
        assert_eq!(tensors.len(), content.tensor_infos.len());
        Ok(())
    }

    #[test]
    fn rejects_wrong_architecture_shape_type_and_missing_tensor() {
        let mut content = synthetic_content();
        content
            .metadata
            .insert("general.architecture".into(), Value::String("qwen3".into()));
        assert!(validate(&content, 1_000_000, &limits()).is_err());
        let mut content = synthetic_content();
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .expect("fixture tensor")
            .shape = Shape::from(vec![31]);
        assert!(validate(&content, 1_000_000, &limits()).is_err());
        let mut content = synthetic_content();
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .expect("fixture tensor")
            .ggml_dtype = GgmlDType::Q8_0;
        assert!(validate(&content, 1_000_000, &limits()).is_err());
        let mut content = synthetic_content();
        content.tensor_infos.remove("output_norm.weight");
        assert!(validate(&content, 1_000_000, &limits()).is_err());
    }

    #[test]
    fn rejects_ranges_and_budget() {
        let mut content = synthetic_content();
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .expect("fixture tensor")
            .offset = u64::MAX;
        assert!(validate(&content, 1_000_000, &limits()).is_err());
        let mut content = synthetic_content();
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .expect("fixture tensor")
            .offset = 0;
        assert!(validate(&content, 1_000_000, &limits()).is_err());
        let content = synthetic_content();
        let mut too_small = limits();
        too_small.max_tensor_bytes = 100;
        assert!(validate(&content, 1_000_000, &too_small).is_err());
    }

    #[test]
    #[ignore = "requires the hash-pinned external 4B GGUF"]
    fn exact_external_header_inventory() -> Result<()> {
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        assert_eq!(artifact.config.block_count, 32);
        assert_eq!(artifact.config.vocab_size, 248_320);
        assert_eq!(artifact.tensors.len(), 426);
        assert_eq!(
            artifact
                .tensors
                .values()
                .filter(|t| t.dtype == GgmlDType::Q8_0)
                .count(),
            249
        );
        assert_eq!(
            artifact
                .tensors
                .values()
                .filter(|t| t.dtype == GgmlDType::F32)
                .count(),
            177
        );
        Ok(())
    }
}
