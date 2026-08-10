//! Local-only loading for an unmodified Hugging Face LFM2-VL directory.
//!
//! The resolved checkpoint directory is an immutable input snapshot: no weight
//! file may be replaced or modified from header inspection through the lifetime
//! of the returned model. This is the local-file integrity requirement inherited
//! from memory-mapped safetensors.

use crate::native_checkpoint::{
    inspect_bounded_file, read_bounded_utf8, resolve_checkpoint, ResolvedCheckpoint,
};
use anyhow::{bail, Context, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::lfm2::LayerType;
use candle_transformers::models::lfm2_vl::{Lfm2VlConfig, Lfm2VlModel};
use candle_vlm::lfm2_vl::{
    Lfm2VlProcessor, Lfm2VlProcessorConfig, Lfm2VlPrompt, ProcessorConfigPatch, PromptOptions,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROCESSOR_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_MODEL_LAYERS: usize = 512;
const MAX_EXPECTED_TENSORS: usize = 65_536;

const CONFIG_FILE: &str = "config.json";
const PROCESSOR_FILE: &str = "processor_config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CANONICAL_VISION_ROOT: &str = "model.vision_tower.vision_model";
const DIRECT_VISION_ROOT: &str = "model.vision_tower";
const PROJECTOR_ROOT: &str = "model.multi_modal_projector";
const LANGUAGE_ROOT: &str = "model.language_model";
const LM_HEAD_ROOT: &str = "lm_head";

#[derive(Clone, Copy, Debug)]
pub struct NativeLoadOptions<'a> {
    pub vision_dtype: DType,
    pub text_dtype: DType,
    pub vision_device: &'a Device,
    pub text_device: &'a Device,
}

#[derive(Clone, Debug)]
pub struct NativeLoadReport {
    pub loaded_tensors: Vec<String>,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub shape_or_dtype_mismatches: Vec<String>,
    pub resolved_vision_root: String,
    pub resolved_projector_root: String,
    pub resolved_language_root: String,
    pub tied_output_resolution: String,
    pub shard_count: usize,
    pub indexed: bool,
    pub total_file_bytes: u64,
    pub vision_dtype: String,
    pub text_dtype: String,
    pub vision_device: String,
    pub text_device: String,
}

impl NativeLoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_tensors.is_empty()
            && self.unexpected_tensors.is_empty()
            && self.shape_or_dtype_mismatches.is_empty()
    }

    fn require_clean(&self) -> Result<()> {
        if !self.is_clean() {
            bail!(
                "native LFM2-VL tensor validation failed; missing={:?}, unexpected={:?}, mismatches={:?}",
                self.missing_tensors,
                self.unexpected_tensors,
                self.shape_or_dtype_mismatches
            )
        }
        Ok(())
    }
}

pub struct LoadedNative {
    pub model: Lfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
    pub report: NativeLoadReport,
    pub source_files: Vec<PathBuf>,
}

pub fn load_native(
    model_dir: impl AsRef<Path>,
    processor_override: Option<&Path>,
    options: NativeLoadOptions<'_>,
) -> Result<LoadedNative> {
    let checkpoint = resolve_checkpoint(model_dir)?;
    let config_path =
        checkpoint.required_file(CONFIG_FILE, "native model config", MAX_CONFIG_BYTES)?;
    let processor_path = checkpoint.required_file(
        PROCESSOR_FILE,
        "native processor config",
        MAX_PROCESSOR_CONFIG_BYTES,
    )?;
    let tokenizer_path =
        checkpoint.required_file(TOKENIZER_FILE, "native tokenizer", MAX_TOKENIZER_BYTES)?;

    let config_json = read_bounded_utf8(&config_path, "native model config", MAX_CONFIG_BYTES)?;
    let config = Lfm2VlConfig::from_json(&config_json).map_err(anyhow::Error::msg)?;
    let report = inspect_native_checkpoint(&checkpoint, &config, options)?;
    report.require_clean()?;

    let processor_json = read_bounded_utf8(
        &processor_path,
        "native processor config",
        MAX_PROCESSOR_CONFIG_BYTES,
    )?;
    let processor_patch =
        ProcessorConfigPatch::from_json(&processor_json).map_err(anyhow::Error::msg)?;
    let explicit_processor = processor_override
        .map(|path| {
            let path = inspect_bounded_file(
                path,
                "native processor override",
                MAX_PROCESSOR_CONFIG_BYTES,
            )?;
            let json = read_bounded_utf8(
                &path,
                "native processor override",
                MAX_PROCESSOR_CONFIG_BYTES,
            )?;
            let patch = ProcessorConfigPatch::from_json(&json).map_err(anyhow::Error::msg)?;
            Ok::<_, anyhow::Error>((path, patch))
        })
        .transpose()?;
    let model_patch = ProcessorConfigPatch::from_model_config(&config);
    let processor_config = Lfm2VlProcessorConfig::resolve(
        explicit_processor.as_ref().map(|(_, patch)| patch),
        Some(&processor_patch),
        None,
        Some(&model_patch),
    )?;
    if processor_config.encoder_patch_size != config.vision_config.patch_size
        || processor_config.downsample_factor != config.downsample_factor
    {
        bail!(
            "native processor/model mismatch: processor patch/factor [{}, {}], model [{}, {}]",
            processor_config.encoder_patch_size,
            processor_config.downsample_factor,
            config.vision_config.patch_size,
            config.downsample_factor
        )
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let prompt = Lfm2VlPrompt::from_processor_config(
        tokenizer,
        Some(config.image_token_id),
        &processor_config,
        PromptOptions {
            use_image_special_tokens: config.use_image_special_tokens,
            context_length: Some(config.text_config.max_position_embeddings),
        },
    )?;
    let processor = Lfm2VlProcessor::from_config(&processor_config)?;

    // SAFETY: the resolver canonicalizes regular local files and validates every
    // safetensors header before mapping. The checkpoint files must not be
    // concurrently modified while the returned model is alive.
    let vision_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &checkpoint.weight_files,
            options.vision_dtype,
            options.vision_device,
        )
    }
    .context("mapping native LFM2-VL vision safetensors")?;
    let text_vb = if options.vision_device.same_device(options.text_device)
        && options.vision_dtype == options.text_dtype
    {
        vision_vb.clone()
    } else {
        // SAFETY: this maps the same already-resolved, already-validated files
        // for a distinct target device. They must remain immutable for the
        // lifetime of the returned model.
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &checkpoint.weight_files,
                options.text_dtype,
                options.text_device,
            )
        }
        .context("mapping native LFM2-VL text safetensors")?
    };
    let text_config = config.text_model_config()?;
    let lm_head_vb = (!text_config.tie_embedding).then(|| text_vb.pp(LM_HEAD_ROOT));
    let model = Lfm2VlModel::new_from_parts(
        &config,
        vision_vb.pp(DIRECT_VISION_ROOT),
        vision_vb.pp(PROJECTOR_ROOT),
        text_vb.pp(LANGUAGE_ROOT),
        lm_head_vb,
    )?;

    let mut source_files = Vec::new();
    let source_file_capacity = checkpoint
        .weight_files
        .len()
        .checked_add(5)
        .ok_or_else(|| anyhow::anyhow!("native source-file count overflow"))?;
    source_files
        .try_reserve_exact(source_file_capacity)
        .map_err(|_| anyhow::anyhow!("allocating native source-file evidence"))?;
    source_files.push(config_path);
    source_files.push(processor_path);
    source_files.push(tokenizer_path);
    if let Some(index_file) = &checkpoint.index_file {
        source_files.push(index_file.clone());
    }
    source_files.extend(checkpoint.weight_files.iter().cloned());
    if let Some((path, _)) = explicit_processor {
        if !source_files.contains(&path) {
            source_files.push(path);
        }
    }

    Ok(LoadedNative {
        model,
        processor,
        prompt,
        report,
        source_files,
    })
}

fn inspect_native_checkpoint(
    checkpoint: &ResolvedCheckpoint,
    config: &Lfm2VlConfig,
    options: NativeLoadOptions<'_>,
) -> Result<NativeLoadReport> {
    config.validate()?;
    if config.text_config.num_hidden_layers > MAX_MODEL_LAYERS
        || config.vision_config.num_hidden_layers > MAX_MODEL_LAYERS
    {
        bail!(
            "native LFM2-VL layer counts text={} vision={} exceed {MAX_MODEL_LAYERS}",
            config.text_config.num_hidden_layers,
            config.vision_config.num_hidden_layers
        )
    }
    let vision_root = resolve_vision_root(&checkpoint.tensors)?;
    let expected = expected_tensor_shapes(config, vision_root)?;
    if expected.len() > MAX_EXPECTED_TENSORS {
        bail!(
            "native LFM2-VL expected tensor count {} exceeds {MAX_EXPECTED_TENSORS}",
            expected.len()
        )
    }
    let actual_names: BTreeSet<_> = checkpoint.tensors.keys().cloned().collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();
    let loaded_tensors = expected_names
        .intersection(&actual_names)
        .cloned()
        .collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in expected_names.intersection(&actual_names) {
        let expected_shape = expected
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("expected tensor {name:?} disappeared"))?;
        let actual = checkpoint
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("native tensor {name:?} disappeared"))?;
        if &actual.shape != expected_shape {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected shape {expected_shape:?}, found {} {:?} in {}",
                actual.dtype, actual.shape, actual.shard
            ));
        }
    }
    let text_config = config.text_model_config()?;
    let tied_output_resolution = if text_config.tie_embedding {
        format!("tied:{LANGUAGE_ROOT}.embed_tokens.weight")
    } else {
        format!("explicit:{LM_HEAD_ROOT}.weight")
    };
    Ok(NativeLoadReport {
        loaded_tensors,
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: vision_root.to_owned(),
        resolved_projector_root: PROJECTOR_ROOT.to_owned(),
        resolved_language_root: LANGUAGE_ROOT.to_owned(),
        tied_output_resolution,
        shard_count: checkpoint.weight_files.len(),
        indexed: checkpoint.indexed,
        total_file_bytes: checkpoint.total_file_bytes,
        vision_dtype: format!("{:?}", options.vision_dtype),
        text_dtype: format!("{:?}", options.text_dtype),
        vision_device: format!("{:?}", options.vision_device),
        text_device: format!("{:?}", options.text_device),
    })
}

fn resolve_vision_root(
    tensors: &BTreeMap<String, crate::native_checkpoint::TensorInfo>,
) -> Result<&'static str> {
    let canonical_anchor = format!("{CANONICAL_VISION_ROOT}.embeddings.patch_embedding.weight");
    let direct_anchor = format!("{DIRECT_VISION_ROOT}.embeddings.patch_embedding.weight");
    match (
        tensors.contains_key(&canonical_anchor),
        tensors.contains_key(&direct_anchor),
    ) {
        (true, false) => Ok(CANONICAL_VISION_ROOT),
        (false, true) => Ok(DIRECT_VISION_ROOT),
        (true, true) => bail!("native checkpoint contains both supported vision tensor roots"),
        (false, false) => Ok(CANONICAL_VISION_ROOT),
    }
}

fn expected_tensor_shapes(
    config: &Lfm2VlConfig,
    vision_root: &str,
) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut shapes = BTreeMap::new();
    let vision = &config.vision_config;
    let patch_dimension = vision
        .num_channels
        .checked_mul(vision.patch_size)
        .and_then(|value| value.checked_mul(vision.patch_size))
        .ok_or_else(|| anyhow::anyhow!("native SigLIP2 patch dimension overflow"))?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.patch_embedding.weight"),
        vec![vision.hidden_size, patch_dimension],
    )?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
    )?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
    )?;
    for layer in 0..vision.num_hidden_layers {
        let root = format!("{vision_root}.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.weight"),
                vec![vision.hidden_size],
            )?;
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.bias"),
                vec![vision.hidden_size],
            )?;
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add_shape(
                &mut shapes,
                format!("{root}.self_attn.{projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
            )?;
            add_shape(
                &mut shapes,
                format!("{root}.self_attn.{projection}.bias"),
                vec![vision.hidden_size],
            )?;
        }
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc2.bias"),
            vec![vision.hidden_size],
        )?;
    }
    for parameter in ["weight", "bias"] {
        add_shape(
            &mut shapes,
            format!("{vision_root}.post_layernorm.{parameter}"),
            vec![vision.hidden_size],
        )?;
    }

    let projector_input = config.projector_input_size()?;
    if config.projector_use_layernorm {
        for parameter in ["weight", "bias"] {
            add_shape(
                &mut shapes,
                format!("{PROJECTOR_ROOT}.layer_norm.{parameter}"),
                vec![projector_input],
            )?;
        }
    }
    add_shape(
        &mut shapes,
        format!("{PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
    )?;
    add_shape(
        &mut shapes,
        format!("{PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_config.hidden_size, config.projector_hidden_size],
    )?;
    if config.projector_bias {
        add_shape(
            &mut shapes,
            format!("{PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_config.hidden_size],
        )?;
    }

    let text = config.text_model_config()?;
    add_shape(
        &mut shapes,
        format!("{LANGUAGE_ROOT}.embed_tokens.weight"),
        vec![text.vocab_size, text.hidden_size],
    )?;
    let head_dim = text.head_dim();
    let kv_width = text
        .num_key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("native LFM2 key/value width overflow"))?;
    let conv_projection = text
        .hidden_size
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("native LFM2 convolution projection overflow"))?;
    for (layer, layer_type) in text.layer_types.iter().copied().enumerate() {
        let root = format!("{LANGUAGE_ROOT}.layers.{layer}");
        for norm in ["operator_norm", "ffn_norm"] {
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.weight"),
                vec![text.hidden_size],
            )?;
        }
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w1.weight"),
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w3.weight"),
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w2.weight"),
            vec![text.hidden_size, text.intermediate_size],
        )?;
        match layer_type {
            LayerType::FullAttention => {
                for (name, out_width) in [
                    ("q_proj", text.hidden_size),
                    ("k_proj", kv_width),
                    ("v_proj", kv_width),
                    ("out_proj", text.hidden_size),
                ] {
                    add_shape(
                        &mut shapes,
                        format!("{root}.self_attn.{name}.weight"),
                        vec![out_width, text.hidden_size],
                    )?;
                }
                for norm in ["q_layernorm", "k_layernorm"] {
                    add_shape(
                        &mut shapes,
                        format!("{root}.self_attn.{norm}.weight"),
                        vec![head_dim],
                    )?;
                }
            }
            LayerType::Conv => {
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.in_proj.weight"),
                    vec![conv_projection, text.hidden_size],
                )?;
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.out_proj.weight"),
                    vec![text.hidden_size, text.hidden_size],
                )?;
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.conv.weight"),
                    vec![text.hidden_size, 1, text.conv_l_cache],
                )?;
            }
        }
    }
    add_shape(
        &mut shapes,
        format!("{LANGUAGE_ROOT}.embedding_norm.weight"),
        vec![text.hidden_size],
    )?;
    if !text.tie_embedding {
        add_shape(
            &mut shapes,
            format!("{LM_HEAD_ROOT}.weight"),
            vec![text.vocab_size, text.hidden_size],
        )?;
    }
    Ok(shapes)
}

fn add_shape(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    name: String,
    shape: Vec<usize>,
) -> Result<()> {
    if shape.is_empty() || shape.contains(&0) {
        bail!("native LFM2-VL expected tensor {name:?} has an invalid shape")
    }
    if shapes.insert(name.clone(), shape).is_some() {
        bail!("native LFM2-VL expected tensor {name:?} is duplicated")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Result<Self> {
            let serial = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "candle-lfm2-vl-{label}-{}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&path)
                .with_context(|| format!("creating test directory {}", path.display()))?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_tiny")
    }

    fn source_file_names(files: &[PathBuf]) -> Result<BTreeSet<String>> {
        files
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("test source path has no UTF-8 file name"))
            })
            .collect()
    }

    fn tiny_config_json(tied: bool) -> String {
        json!({
            "model_type": "lfm2-vl",
            "image_token_index": 3,
            "downsample_factor": 2,
            "projector_hidden_size": 24,
            "projector_hidden_act": "gelu",
            "projector_bias": true,
            "projector_use_layer_norm": true,
            "use_image_special_tokens": false,
            "tie_word_embeddings": tied,
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 32,
                "hidden_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "intermediate_size": 32,
                "block_auto_adjust_ff_dim": false,
                "layer_types": ["conv", "full_attention"],
                "rope_theta": 10000.0,
                "max_position_embeddings": 64,
                "conv_L_cache": 3
            },
            "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "patch_size": 2,
                "num_patches": 16,
                "hidden_act": "gelu_pytorch_tanh",
                "vision_use_head": false
            }
        })
        .to_string()
    }

    fn pinned_official_config_json(
        text_hidden_size: usize,
        text_heads: usize,
        text_ffn_size: usize,
        vision_hidden_size: usize,
        vision_ffn_size: usize,
        vision_layers: usize,
        vision_heads: usize,
    ) -> String {
        json!({
            "model_type": "lfm2_vl",
            "image_token_id": 396,
            "downsample_factor": 2,
            "projector_hidden_size": 2048,
            "projector_hidden_act": "gelu",
            "projector_bias": true,
            "projector_use_layernorm": false,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 65536,
                "hidden_size": text_hidden_size,
                "num_hidden_layers": 16,
                "num_attention_heads": text_heads,
                "num_key_value_heads": 8,
                "intermediate_size": text_ffn_size,
                "block_ff_dim": text_ffn_size,
                "block_auto_adjust_ff_dim": true,
                "block_ffn_dim_multiplier": 1.0,
                "block_multiple_of": 256,
                "conv_L_cache": 3,
                "max_position_embeddings": 128000,
                "full_attention_layers": [2, 5, 8, 10, 12, 14]
            },
            "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": vision_hidden_size,
                "intermediate_size": vision_ffn_size,
                "num_hidden_layers": vision_layers,
                "num_attention_heads": vision_heads,
                "num_channels": 3,
                "patch_size": 16,
                "num_patches": 256,
                "vision_use_head": false
            }
        })
        .to_string()
    }

    fn processor_json() -> String {
        json!({
            "encoder_patch_size": 2,
            "downsample_factor": 2,
            "do_image_splitting": false,
            "min_tiles": 1,
            "max_tiles": 2,
            "use_thumbnail": false,
            "tile_size": 8,
            "min_image_tokens": 1,
            "max_image_tokens": 4,
            "max_num_patches": 16
        })
        .to_string()
    }

    fn write_tokenizer(path: &Path) -> Result<()> {
        let tokenizer = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<image>": 3},
                "unk_token": "[UNK]"
            }
        });
        std::fs::write(path, tokenizer.to_string())?;
        Ok(())
    }

    fn native_weights(canonical_vision_root: bool) -> Result<BTreeMap<String, Tensor>> {
        let source =
            candle::safetensors::load(fixture_dir().join("tensors.safetensors"), &Device::Cpu)?;
        let mut weights = BTreeMap::new();
        for (name, tensor) in source {
            let Some(name) = name.strip_prefix("weights.") else {
                continue;
            };
            let name = if canonical_vision_root {
                name.replacen("model.vision_tower.", "model.vision_tower.vision_model.", 1)
            } else {
                name.to_owned()
            };
            weights.insert(name, tensor);
        }
        if weights.is_empty() {
            bail!("tiny fixture contains no native weights")
        }
        Ok(weights)
    }

    fn write_common_files(dir: &Path, tied: bool) -> Result<()> {
        std::fs::write(dir.join(CONFIG_FILE), tiny_config_json(tied))?;
        std::fs::write(dir.join(PROCESSOR_FILE), processor_json())?;
        write_tokenizer(&dir.join(TOKENIZER_FILE))
    }

    fn save_weights(weights: &BTreeMap<String, Tensor>, path: &Path) -> Result<()> {
        let weights: HashMap<_, _> = weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        candle::safetensors::save(&weights, path)?;
        Ok(())
    }

    fn official_inventory_sha256(inventory: &BTreeMap<String, Vec<usize>>) -> String {
        let mut digest = Sha256::new();
        for (name, shape) in inventory {
            digest.update(name.as_bytes());
            digest.update(b"\tBF16\t");
            for (index, dimension) in shape.iter().enumerate() {
                if index > 0 {
                    digest.update(b",");
                }
                digest.update(dimension.to_string().as_bytes());
            }
            digest.update(b"\n");
        }
        format!("{:x}", digest.finalize())
    }

    fn write_single_checkpoint(
        dir: &Path,
        tied: bool,
        canonical_vision_root: bool,
    ) -> Result<BTreeMap<String, Tensor>> {
        write_common_files(dir, tied)?;
        let mut weights = native_weights(canonical_vision_root)?;
        if !tied {
            let embedding = weights
                .get("model.language_model.embed_tokens.weight")
                .ok_or_else(|| anyhow::anyhow!("tiny embedding weight is missing"))?
                .clone();
            weights.insert("lm_head.weight".to_owned(), embedding);
        }
        save_weights(&weights, &dir.join("model.safetensors"))?;
        Ok(weights)
    }

    #[test]
    fn matches_locked_official_header_tensor_inventories() -> Result<()> {
        // These are exact canonical name/dtype/shape digests from bounded
        // header-only reads recorded in tools/lfm2_vl/reference-lock.json.
        for (config_json, expected_ffn, expected_count, expected_digest) in [
            (
                pinned_official_config_json(1024, 16, 6656, 768, 3072, 12, 12),
                4608,
                349,
                "08f544b4495804ed842a37acf0936544ec88aa5d947bef8304a47816fee5b1a7",
            ),
            (
                pinned_official_config_json(2048, 32, 12288, 1152, 4304, 27, 16),
                8192,
                589,
                "24728d0ed10229e788c5b9baf25e0cc6c92c93b9cdb12ebb252a3c140a861703",
            ),
        ] {
            let config = Lfm2VlConfig::from_json(&config_json)?;
            assert_eq!(config.text_model_config()?.intermediate_size, expected_ffn);
            let inventory = expected_tensor_shapes(&config, CANONICAL_VISION_ROOT)?;
            assert_eq!(inventory.len(), expected_count);
            assert_eq!(official_inventory_sha256(&inventory), expected_digest);
        }
        Ok(())
    }

    #[test]
    fn loads_clean_single_file_tied_checkpoint_and_resolves_direct_fixture_root() -> Result<()> {
        let dir = TempDir::new("single")?;
        let weights = write_single_checkpoint(dir.path(), true, false)?;
        let processor_override = dir.path().join("processor-override.json");
        std::fs::write(
            &processor_override,
            json!({"min_image_tokens": 2}).to_string(),
        )?;
        let loaded = load_native(
            dir.path(),
            Some(&processor_override),
            NativeLoadOptions {
                vision_dtype: DType::F32,
                text_dtype: DType::F32,
                vision_device: &Device::Cpu,
                text_device: &Device::Cpu,
            },
        )?;
        assert!(loaded.report.is_clean());
        assert_eq!(loaded.report.loaded_tensors.len(), weights.len());
        assert_eq!(loaded.report.shard_count, 1);
        assert!(!loaded.report.indexed);
        assert_eq!(loaded.report.resolved_vision_root, DIRECT_VISION_ROOT);
        assert_eq!(
            loaded.report.tied_output_resolution,
            "tied:model.language_model.embed_tokens.weight"
        );
        assert_eq!(loaded.prompt.special_tokens().image_token_id, 3);
        assert_eq!(loaded.model.config().text_config.hidden_size, 12);
        assert_eq!(loaded.processor.config().min_image_tokens, 2);
        assert_eq!(
            source_file_names(&loaded.source_files)?,
            BTreeSet::from([
                "config.json".to_owned(),
                "model.safetensors".to_owned(),
                "processor-override.json".to_owned(),
                "processor_config.json".to_owned(),
                "tokenizer.json".to_owned(),
            ])
        );
        Ok(())
    }

    #[test]
    fn loads_indexed_shards_with_explicit_head_and_official_vision_root() -> Result<()> {
        let dir = TempDir::new("indexed")?;
        write_common_files(dir.path(), false)?;
        let mut weights = native_weights(true)?;
        let embedding = weights
            .get("model.language_model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("tiny embedding weight is missing"))?
            .clone();
        weights.insert("lm_head.weight".to_owned(), embedding);
        let split = weights.len().div_ceil(2);
        let first: BTreeMap<_, _> = weights
            .iter()
            .take(split)
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        let second: BTreeMap<_, _> = weights
            .iter()
            .skip(split)
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        let first_name = "model-00001-of-00002.safetensors";
        let second_name = "model-00002-of-00002.safetensors";
        save_weights(&first, &dir.path().join(first_name))?;
        save_weights(&second, &dir.path().join(second_name))?;
        let weight_map: BTreeMap<_, _> = first
            .keys()
            .map(|name| (name.clone(), first_name.to_owned()))
            .chain(
                second
                    .keys()
                    .map(|name| (name.clone(), second_name.to_owned())),
            )
            .collect();
        let total_size = weights.values().try_fold(0u64, |total, tensor| {
            let elements = u64::try_from(tensor.elem_count()).ok()?;
            total.checked_add(elements.checked_mul(4)?)
        });
        let total_size =
            total_size.ok_or_else(|| anyhow::anyhow!("test payload byte total overflow"))?;
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            json!({"metadata": {"total_size": total_size}, "weight_map": weight_map}).to_string(),
        )?;
        let loaded = load_native(
            dir.path(),
            None,
            NativeLoadOptions {
                vision_dtype: DType::F32,
                text_dtype: DType::F32,
                vision_device: &Device::Cpu,
                text_device: &Device::Cpu,
            },
        )?;
        assert!(loaded.report.is_clean());
        assert_eq!(loaded.report.loaded_tensors.len(), weights.len());
        assert_eq!(loaded.report.shard_count, 2);
        assert!(loaded.report.indexed);
        assert_eq!(loaded.report.resolved_vision_root, CANONICAL_VISION_ROOT);
        assert_eq!(
            loaded.report.tied_output_resolution,
            "explicit:lm_head.weight"
        );
        assert_eq!(
            source_file_names(&loaded.source_files)?,
            BTreeSet::from([
                "config.json".to_owned(),
                "model-00001-of-00002.safetensors".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
                "model.safetensors.index.json".to_owned(),
                "processor_config.json".to_owned(),
                "tokenizer.json".to_owned(),
            ])
        );
        Ok(())
    }

    #[test]
    fn rejects_wrong_index_mapping_total_size_and_duplicate_shards() -> Result<()> {
        let wrong_mapping = TempDir::new("wrong-index-map")?;
        let first_name = "model-00001-of-00002.safetensors";
        let second_name = "model-00002-of-00002.safetensors";
        let first = BTreeMap::from([(
            "tensor_a".to_owned(),
            Tensor::zeros(1, DType::F32, &Device::Cpu)?,
        )]);
        let second = BTreeMap::from([(
            "tensor_b".to_owned(),
            Tensor::zeros(1, DType::F32, &Device::Cpu)?,
        )]);
        save_weights(&first, &wrong_mapping.path().join(first_name))?;
        save_weights(&second, &wrong_mapping.path().join(second_name))?;
        std::fs::write(
            wrong_mapping.path().join("model.safetensors.index.json"),
            json!({"weight_map": {
                "tensor_a": second_name,
                "tensor_b": first_name
            }})
            .to_string(),
        )?;
        let error = match resolve_checkpoint(wrong_mapping.path()) {
            Ok(_) => bail!("wrong tensor-to-shard mapping unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("index maps"));

        std::fs::write(
            wrong_mapping.path().join("model.safetensors.index.json"),
            json!({
                "metadata": {"total_size": 1},
                "weight_map": {
                    "tensor_a": first_name,
                    "tensor_b": second_name
                }
            })
            .to_string(),
        )?;
        let error = match resolve_checkpoint(wrong_mapping.path()) {
            Ok(_) => bail!("wrong index total_size unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("total_size"));

        let duplicate = TempDir::new("duplicate-shard-tensor")?;
        save_weights(&first, &duplicate.path().join(first_name))?;
        let second_with_duplicate = BTreeMap::from([
            (
                "tensor_a".to_owned(),
                Tensor::zeros(1, DType::F32, &Device::Cpu)?,
            ),
            (
                "tensor_b".to_owned(),
                Tensor::zeros(1, DType::F32, &Device::Cpu)?,
            ),
        ]);
        save_weights(&second_with_duplicate, &duplicate.path().join(second_name))?;
        std::fs::write(
            duplicate.path().join("model.safetensors.index.json"),
            json!({"weight_map": {
                "tensor_a": first_name,
                "tensor_b": second_name
            }})
            .to_string(),
        )?;
        let error = match resolve_checkpoint(duplicate.path()) {
            Ok(_) => bail!("duplicate shard tensor unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("occurs in multiple shards"));
        Ok(())
    }

    #[test]
    fn loads_independent_native_component_dtypes() -> Result<()> {
        let dir = TempDir::new("component-dtypes")?;
        write_single_checkpoint(dir.path(), true, true)?;
        let loaded = load_native(
            dir.path(),
            None,
            NativeLoadOptions {
                vision_dtype: DType::BF16,
                text_dtype: DType::F32,
                vision_device: &Device::Cpu,
                text_device: &Device::Cpu,
            },
        )?;
        assert_eq!(loaded.model.vision_dtype(), DType::BF16);
        assert_eq!(loaded.model.text_dtype(), DType::F32);
        assert_eq!(loaded.report.vision_dtype, "BF16");
        assert_eq!(loaded.report.text_dtype, "F32");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loads_native_vision_cuda_text_cpu_on_distinct_devices() -> Result<()> {
        let dir = TempDir::new("split-devices")?;
        write_single_checkpoint(dir.path(), true, true)?;
        let vision_device = Device::new_cuda(0)?;
        let loaded = load_native(
            dir.path(),
            None,
            NativeLoadOptions {
                vision_dtype: DType::F32,
                text_dtype: DType::F32,
                vision_device: &vision_device,
                text_device: &Device::Cpu,
            },
        )?;
        assert!(loaded.model.vision_device().same_device(&vision_device));
        assert!(loaded.model.text_device().same_device(&Device::Cpu));
        assert!(!loaded
            .model
            .vision_device()
            .same_device(loaded.model.text_device()));
        Ok(())
    }

    #[test]
    fn reports_inventory_defects_before_model_construction() -> Result<()> {
        let dir = TempDir::new("inventory")?;
        let mut weights = write_single_checkpoint(dir.path(), true, true)?;
        std::fs::remove_file(dir.path().join("model.safetensors"))?;
        let missing_name = "model.multi_modal_projector.linear_2.weight";
        weights.remove(missing_name);
        weights.insert(
            "model.multi_modal_projector.unexpected.weight".to_owned(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        );
        let shape_name = "model.multi_modal_projector.linear_1.weight";
        weights.insert(
            shape_name.to_owned(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        );
        save_weights(&weights, &dir.path().join("model.safetensors"))?;
        let error = match load_native(
            dir.path(),
            None,
            NativeLoadOptions {
                vision_dtype: DType::F32,
                text_dtype: DType::F32,
                vision_device: &Device::Cpu,
                text_device: &Device::Cpu,
            },
        ) {
            Ok(_) => bail!("malformed native inventory unexpectedly loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(missing_name));
        assert!(error.contains("unexpected.weight"));
        assert!(error.contains(shape_name));
        Ok(())
    }

    #[test]
    fn rejects_missing_documents_ambiguous_weights_and_index_traversal() -> Result<()> {
        let missing = TempDir::new("missing-docs")?;
        std::fs::write(missing.path().join("model.safetensors"), b"invalid")?;
        assert!(load_native(
            missing.path(),
            None,
            NativeLoadOptions {
                vision_dtype: DType::F32,
                text_dtype: DType::F32,
                vision_device: &Device::Cpu,
                text_device: &Device::Cpu,
            }
        )
        .is_err());

        let ambiguous = TempDir::new("ambiguous")?;
        write_single_checkpoint(ambiguous.path(), true, true)?;
        std::fs::write(
            ambiguous.path().join("model.safetensors.index.json"),
            json!({"weight_map": {"x": "model.safetensors"}}).to_string(),
        )?;
        let error = match resolve_checkpoint(ambiguous.path()) {
            Ok(_) => bail!("ambiguous native checkpoint unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("both"));

        let traversal = TempDir::new("traversal")?;
        std::fs::write(
            traversal.path().join("model.safetensors.index.json"),
            json!({"weight_map": {"x": "../outside.safetensors"}}).to_string(),
        )?;
        let error = match resolve_checkpoint(traversal.path()) {
            Ok(_) => bail!("traversal index unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("safe local filename"));

        let missing_shard = TempDir::new("missing-shard")?;
        std::fs::write(
            missing_shard.path().join("model.safetensors.index.json"),
            json!({"weight_map": {"tensor": "model-00001-of-00001.safetensors"}}).to_string(),
        )?;
        let error = match resolve_checkpoint(missing_shard.path()) {
            Ok(_) => bail!("missing native shard unexpectedly resolved"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("native safetensors shard"));
        Ok(())
    }

    #[test]
    fn rejects_processor_and_tokenizer_pair_mismatches() -> Result<()> {
        let dir = TempDir::new("pairing")?;
        write_single_checkpoint(dir.path(), true, true)?;
        std::fs::write(
            dir.path().join(PROCESSOR_FILE),
            json!({
                "encoder_patch_size": 4,
                "downsample_factor": 2,
                "do_image_splitting": false,
                "min_tiles": 1,
                "max_tiles": 2,
                "use_thumbnail": false,
                "tile_size": 8,
                "min_image_tokens": 1,
                "max_image_tokens": 4,
                "max_num_patches": 16
            })
            .to_string(),
        )?;
        let options = NativeLoadOptions {
            vision_dtype: DType::F32,
            text_dtype: DType::F32,
            vision_device: &Device::Cpu,
            text_device: &Device::Cpu,
        };
        let error = match load_native(dir.path(), None, options) {
            Ok(_) => bail!("processor/model mismatch unexpectedly loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("processor/model mismatch"));

        std::fs::write(dir.path().join(PROCESSOR_FILE), processor_json())?;
        let tokenizer = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "other": 3, "<image>": 4},
                "unk_token": "[UNK]"
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_FILE), tokenizer.to_string())?;
        let error = match load_native(dir.path(), None, options) {
            Ok(_) => bail!("tokenizer/model mismatch unexpectedly loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("does not match model id"));
        Ok(())
    }
}
