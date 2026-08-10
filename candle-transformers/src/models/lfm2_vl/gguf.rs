//! llama.cpp-compatible GGUF MMProj loading through dense dequantization.

use super::weights::{Mmproj, MmprojLoadReport, MmprojMetadata};
use super::{Lfm2VlMmprojConfig, Lfm2VlProjector};
use crate::models::siglip2;
use candle::quantized::{
    gguf_file::{self, Value},
    GgmlDType,
};
use candle::{DType, Device, Result};
use candle_nn::{Activation, VarBuilder};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_GGUF_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_DENSE_MMPROJ_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_ESTIMATED_MMPROJ_PEAK_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_GGUF_MMPROJ_TENSORS: usize = 16_384;
const MAX_GGUF_MMPROJ_METADATA: u64 = 16_384;
const MAX_GGUF_MMPROJ_ARRAY_ELEMENTS: u64 = 16_384;
const MAX_GGUF_MMPROJ_STRING_BYTES: u64 = 1024 * 1024;
const MAX_GGUF_MMPROJ_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_VISION_LAYERS: usize = 512;
const NATIVE_VISION_ROOT: &str = "model.vision_tower.vision_model";
const NATIVE_PROJECTOR_ROOT: &str = "model.multi_modal_projector";

#[derive(Debug, Clone)]
pub struct GgufMmprojMetadata {
    pub general_architecture: String,
    pub projector_type: String,
    pub general_name: Option<String>,
    pub vision_layer_count: usize,
    pub image_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub source_byte_count: u64,
    pub dense_byte_count: u64,
    pub estimated_peak_byte_count: u64,
    pub tensor_dtypes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct AllocationReport {
    source_byte_count: u64,
    dense_byte_count: u64,
    estimated_peak_byte_count: u64,
}

#[derive(Debug)]
struct ParsedMetadata {
    general_name: Option<String>,
    image_size: usize,
    vision_hidden_size: usize,
    vision_intermediate_size: usize,
    vision_layer_count: usize,
    vision_head_count: usize,
    layer_norm_eps: f64,
    patch_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    downsample_factor: usize,
    text_hidden_size: usize,
    preproc: Option<(usize, usize, usize)>,
}

#[derive(Debug)]
struct ExpectedTensor {
    native_name: String,
    shape: Vec<usize>,
    patch_layout: bool,
}

impl Mmproj {
    /// Load a llama.cpp-compatible LFM2-VL MMProj GGUF from one stable file handle.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| {
            candle::Error::Msg(format!(
                "failed to open GGUF MMProj {}: {error}",
                path.display()
            ))
        })?;
        Self::from_gguf(&mut file, dtype, device, image_token_id)
            .map_err(|error| error.with_path(path))
    }

    /// Load a GGUF MMProj from a seekable reader using the dense compatibility path.
    pub fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        let dense_element_size = match dtype {
            DType::F32 => 4u64,
            DType::F16 | DType::BF16 => 2u64,
            _ => candle::bail!(
                "GGUF MMProj dense compatibility dtype must be F32, F16, or BF16, got {dtype:?}"
            ),
        };
        let file_size = reader.seek(SeekFrom::End(0))?;
        if file_size == 0 || file_size > MAX_GGUF_FILE_BYTES {
            candle::bail!(
                "GGUF MMProj file size {file_size} is outside 1..={MAX_GGUF_FILE_BYTES} bytes"
            )
        }
        reader.seek(SeekFrom::Start(0))?;
        let content = gguf_file::Content::read_with_limits(
            reader,
            gguf_file::ContentReadLimits {
                max_tensor_count: MAX_GGUF_MMPROJ_TENSORS as u64,
                max_metadata_count: MAX_GGUF_MMPROJ_METADATA,
                max_string_length: MAX_GGUF_MMPROJ_STRING_BYTES,
                max_array_elements: MAX_GGUF_MMPROJ_ARRAY_ELEMENTS,
                max_header_bytes: MAX_GGUF_MMPROJ_HEADER_BYTES,
            },
        )?;
        if content.tensor_infos.is_empty() || content.tensor_infos.len() > MAX_GGUF_MMPROJ_TENSORS {
            candle::bail!(
                "GGUF MMProj tensor count {} is outside 1..={MAX_GGUF_MMPROJ_TENSORS}",
                content.tensor_infos.len()
            )
        }

        let parsed = parse_metadata(&content)?;
        let config = reconstruct_config(&content, &parsed, image_token_id)?;
        let expected = expected_tensors(&config)?;
        let report = inspect_inventory(&content, &expected, dtype, device)?;
        report.require_clean()?;
        let allocations =
            validate_ranges_and_sizes(&content, &expected, file_size, dense_element_size)?;

        let mut native_tensors = HashMap::new();
        native_tensors.try_reserve(expected.len()).map_err(|_| {
            candle::Error::Msg("GGUF MMProj dense tensor-map allocation failed".into())
        })?;
        for (gguf_name, expected_tensor) in &expected {
            let quantized = content.tensor(reader, gguf_name, device).map_err(|error| {
                candle::Error::Msg(format!(
                    "failed to read GGUF MMProj tensor {gguf_name:?}: {error}"
                ))
            })?;
            let mut tensor = quantized.dequantize(device)?.to_dtype(dtype)?;
            if expected_tensor.patch_layout {
                let [vision_hidden, channels, patch_rows, patch_cols] =
                    tensor.dims().try_into().map_err(|_| {
                        candle::Error::Msg(format!(
                            "GGUF MMProj patch tensor {gguf_name:?} must have rank 4"
                        ))
                    })?;
                let packed_width = channels
                    .checked_mul(patch_rows)
                    .and_then(|value| value.checked_mul(patch_cols))
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "GGUF MMProj patch packed-width calculation overflowed".into(),
                        )
                    })?;
                tensor = tensor
                    .permute((0, 2, 3, 1))?
                    .contiguous()?
                    .reshape((vision_hidden, packed_width))?;
            }
            if native_tensors
                .insert(expected_tensor.native_name.clone(), tensor)
                .is_some()
            {
                candle::bail!(
                    "GGUF MMProj names normalize to duplicate native tensor {:?}",
                    expected_tensor.native_name
                )
            }
        }

        let vb = VarBuilder::from_tensors(native_tensors, dtype, device);
        let vision_tower =
            siglip2::Siglip2VisionModel::new(&config.vision_config, vb.pp(NATIVE_VISION_ROOT))?;
        let projector = Lfm2VlProjector::from_mmproj_config(&config, vb.pp(NATIVE_PROJECTOR_ROOT))?;
        let tensor_dtypes: BTreeMap<_, _> = content
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), format!("{:?}", info.ggml_dtype)))
            .collect();
        let quantized_tensor_count = content
            .tensor_infos
            .values()
            .filter(|info| {
                !matches!(
                    info.ggml_dtype,
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                )
            })
            .count();
        let gguf_metadata = GgufMmprojMetadata {
            general_architecture: "clip".to_string(),
            projector_type: "lfm2".to_string(),
            general_name: parsed.general_name.clone(),
            vision_layer_count: parsed.vision_layer_count,
            image_size: parsed.image_size,
            image_mean: parsed.image_mean,
            image_std: parsed.image_std,
            tensor_count: content.tensor_infos.len(),
            quantized_tensor_count,
            source_byte_count: allocations.source_byte_count,
            dense_byte_count: allocations.dense_byte_count,
            estimated_peak_byte_count: allocations.estimated_peak_byte_count,
            tensor_dtypes,
        };
        let processor = processor_metadata_json(&parsed);
        let metadata = MmprojMetadata {
            architecture: "lfm2_vl".to_string(),
            vision_hidden_size: config.vision_config.hidden_size,
            text_hidden_size: config.text_hidden_size,
            patch_size: config.vision_config.patch_size,
            downsample_factor: config.downsample_factor,
            image_token_id: config.image_token_id,
            use_image_special_tokens: config.use_image_special_tokens,
            expected_text_layer_count: None,
            processor,
            source_model: parsed.general_name,
            source_revision: None,
            manifest: None,
            gguf: Some(gguf_metadata),
        };
        Ok(Mmproj::from_parts(
            vision_tower,
            projector,
            config,
            metadata,
            report,
            dtype,
            device,
        ))
    }
}

fn parse_metadata(content: &gguf_file::Content) -> Result<ParsedMetadata> {
    let general_architecture = required_string(content, "general.architecture")?;
    if general_architecture != "clip" {
        candle::bail!(
            "GGUF MMProj general.architecture must be \"clip\", got {general_architecture:?}"
        )
    }
    let general_type = required_string(content, "general.type")?;
    if general_type != "mmproj" {
        candle::bail!("GGUF MMProj general.type must be \"mmproj\", got {general_type:?}")
    }
    let projector_type = required_string(content, "clip.projector_type")?;
    if projector_type != "lfm2" {
        candle::bail!("GGUF MMProj clip.projector_type must be \"lfm2\", got {projector_type:?}")
    }
    if !required_bool(content, "clip.has_vision_encoder")? {
        candle::bail!("GGUF MMProj clip.has_vision_encoder must be true")
    }
    if !required_bool(content, "clip.use_gelu")? {
        candle::bail!("GGUF MMProj clip.use_gelu must be true for the LFM2 projector")
    }

    let vision_layer_count = required_positive_usize(content, "clip.vision.block_count")?;
    if vision_layer_count > MAX_GGUF_VISION_LAYERS {
        candle::bail!(
            "GGUF MMProj vision layer count {vision_layer_count} exceeds {MAX_GGUF_VISION_LAYERS}"
        )
    }
    let image_std = required_f32_triplet(content, "clip.vision.image_std")?;
    if image_std.iter().any(|&value| value <= 0.0) {
        candle::bail!("GGUF MMProj clip.vision.image_std values must be positive")
    }
    let preproc_min = optional_positive_usize(content, "clip.vision.preproc_min_tiles")?;
    let preproc_max = optional_positive_usize(content, "clip.vision.preproc_max_tiles")?;
    let preproc_size = optional_positive_usize(content, "clip.vision.preproc_image_size")?;
    let preproc = match (preproc_min, preproc_max, preproc_size) {
        (None, None, None) => None,
        (Some(min_tiles), Some(max_tiles), Some(image_size)) => {
            if min_tiles > max_tiles {
                candle::bail!(
                    "GGUF MMProj preprocessor min tiles {min_tiles} exceeds max tiles {max_tiles}"
                )
            }
            Some((min_tiles, max_tiles, image_size))
        }
        _ => candle::bail!(
            "GGUF MMProj tiling metadata must provide min tiles, max tiles, and image size together"
        ),
    };
    Ok(ParsedMetadata {
        general_name: optional_string(content, "general.name")?.map(str::to_string),
        image_size: required_positive_usize(content, "clip.vision.image_size")?,
        vision_hidden_size: required_positive_usize(content, "clip.vision.embedding_length")?,
        vision_intermediate_size: required_positive_usize(
            content,
            "clip.vision.feed_forward_length",
        )?,
        vision_layer_count,
        vision_head_count: required_positive_usize(content, "clip.vision.attention.head_count")?,
        layer_norm_eps: required_positive_float(
            content,
            "clip.vision.attention.layer_norm_epsilon",
        )?,
        patch_size: required_positive_usize(content, "clip.vision.patch_size")?,
        image_mean: required_f32_triplet(content, "clip.vision.image_mean")?,
        image_std,
        downsample_factor: required_positive_usize(content, "clip.vision.projector.scale_factor")?,
        text_hidden_size: required_positive_usize(content, "clip.vision.projection_dim")?,
        preproc,
    })
}

fn reconstruct_config(
    content: &gguf_file::Content,
    metadata: &ParsedMetadata,
    image_token_id: u32,
) -> Result<Lfm2VlMmprojConfig> {
    if metadata.image_size % metadata.patch_size != 0 {
        candle::bail!(
            "GGUF MMProj image size {} is not divisible by patch size {}",
            metadata.image_size,
            metadata.patch_size
        )
    }
    let base_side = metadata.image_size / metadata.patch_size;
    let num_patches = base_side
        .checked_mul(base_side)
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj base position count overflowed".into()))?;

    let patch_shape = required_tensor_shape(content, "v.patch_embd.weight")?;
    if patch_shape.len() != 4
        || patch_shape[0] != metadata.vision_hidden_size
        || patch_shape[2] != metadata.patch_size
        || patch_shape[3] != metadata.patch_size
    {
        candle::bail!(
            "GGUF MMProj tensor \"v.patch_embd.weight\" has shape {patch_shape:?}, expected [{}, channels, {}, {}]",
            metadata.vision_hidden_size,
            metadata.patch_size,
            metadata.patch_size
        )
    }
    let num_channels = patch_shape[1];
    if num_channels != metadata.image_mean.len() || num_channels != metadata.image_std.len() {
        candle::bail!(
            "GGUF MMProj patch channels {num_channels} do not match normalization metadata"
        )
    }

    let projector_input = metadata
        .vision_hidden_size
        .checked_mul(metadata.downsample_factor)
        .and_then(|value| value.checked_mul(metadata.downsample_factor))
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj projector input overflowed".into()))?;
    let linear_1_shape = required_tensor_shape(content, "mm.1.weight")?;
    if linear_1_shape.len() != 2 || linear_1_shape[1] != projector_input {
        candle::bail!(
            "GGUF MMProj tensor \"mm.1.weight\" has shape {linear_1_shape:?}, expected [hidden, {projector_input}]"
        )
    }
    let projector_hidden_size = linear_1_shape[0];
    let linear_2_shape = required_tensor_shape(content, "mm.2.weight")?;
    if linear_2_shape != [metadata.text_hidden_size, projector_hidden_size] {
        candle::bail!(
            "GGUF MMProj tensor \"mm.2.weight\" has shape {linear_2_shape:?}, expected [{}, {projector_hidden_size}]",
            metadata.text_hidden_size
        )
    }

    let input_norm_weight = content.tensor_infos.contains_key("mm.input_norm.weight");
    let input_norm_bias = content.tensor_infos.contains_key("mm.input_norm.bias");
    if input_norm_weight != input_norm_bias {
        candle::bail!("GGUF MMProj optional input LayerNorm weight and bias must appear together")
    }
    let linear_1_bias = content.tensor_infos.contains_key("mm.1.bias");
    let linear_2_bias = content.tensor_infos.contains_key("mm.2.bias");
    if linear_1_bias != linear_2_bias {
        candle::bail!("GGUF MMProj projector linear biases must appear together")
    }

    let config = Lfm2VlMmprojConfig {
        vision_config: siglip2::Siglip2VisionConfig {
            hidden_size: metadata.vision_hidden_size,
            intermediate_size: metadata.vision_intermediate_size,
            num_hidden_layers: metadata.vision_layer_count,
            num_attention_heads: metadata.vision_head_count,
            num_channels,
            patch_size: metadata.patch_size,
            num_patches,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: metadata.layer_norm_eps,
            attention_dropout: 0.0,
            vision_use_head: false,
        },
        text_hidden_size: metadata.text_hidden_size,
        image_token_id,
        downsample_factor: metadata.downsample_factor,
        projector_hidden_size,
        projector_hidden_act: Activation::Gelu,
        projector_bias: linear_1_bias,
        projector_use_layernorm: input_norm_weight,
        use_image_special_tokens: true,
    };
    config.validate()?;
    Ok(config)
}

fn expected_tensors(config: &Lfm2VlMmprojConfig) -> Result<BTreeMap<String, ExpectedTensor>> {
    let vision = &config.vision_config;
    let projector_input = config.projector_input_size()?;
    let mut expected = BTreeMap::new();
    let mut insert = |gguf: String, native: String, shape: Vec<usize>, patch_layout: bool| {
        expected.insert(
            gguf,
            ExpectedTensor {
                native_name: native,
                shape,
                patch_layout,
            },
        );
    };
    insert(
        "v.patch_embd.weight".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.patch_embedding.weight"),
        vec![
            vision.hidden_size,
            vision.num_channels,
            vision.patch_size,
            vision.patch_size,
        ],
        true,
    );
    insert(
        "v.patch_embd.bias".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
        false,
    );
    insert(
        "v.position_embd.weight".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
        false,
    );
    for layer in 0..vision.num_hidden_layers {
        let gguf_root = format!("v.blk.{layer}");
        let native_root = format!("{NATIVE_VISION_ROOT}.encoder.layers.{layer}");
        for (gguf_norm, native_norm) in [("ln1", "layer_norm1"), ("ln2", "layer_norm2")] {
            for suffix in ["weight", "bias"] {
                insert(
                    format!("{gguf_root}.{gguf_norm}.{suffix}"),
                    format!("{native_root}.{native_norm}.{suffix}"),
                    vec![vision.hidden_size],
                    false,
                );
            }
        }
        for (gguf_projection, native_projection) in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_v", "v_proj"),
            ("attn_out", "out_proj"),
        ] {
            insert(
                format!("{gguf_root}.{gguf_projection}.weight"),
                format!("{native_root}.self_attn.{native_projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
                false,
            );
            insert(
                format!("{gguf_root}.{gguf_projection}.bias"),
                format!("{native_root}.self_attn.{native_projection}.bias"),
                vec![vision.hidden_size],
                false,
            );
        }
        insert(
            format!("{gguf_root}.ffn_up.weight"),
            format!("{native_root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_up.bias"),
            format!("{native_root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_down.weight"),
            format!("{native_root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_down.bias"),
            format!("{native_root}.mlp.fc2.bias"),
            vec![vision.hidden_size],
            false,
        );
    }
    for suffix in ["weight", "bias"] {
        insert(
            format!("v.post_ln.{suffix}"),
            format!("{NATIVE_VISION_ROOT}.post_layernorm.{suffix}"),
            vec![vision.hidden_size],
            false,
        );
    }
    if config.projector_use_layernorm {
        for suffix in ["weight", "bias"] {
            insert(
                format!("mm.input_norm.{suffix}"),
                format!("{NATIVE_PROJECTOR_ROOT}.layer_norm.{suffix}"),
                vec![projector_input],
                false,
            );
        }
    }
    insert(
        "mm.1.weight".into(),
        format!("{NATIVE_PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
        false,
    );
    insert(
        "mm.2.weight".into(),
        format!("{NATIVE_PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_hidden_size, config.projector_hidden_size],
        false,
    );
    if config.projector_bias {
        insert(
            "mm.1.bias".into(),
            format!("{NATIVE_PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
            false,
        );
        insert(
            "mm.2.bias".into(),
            format!("{NATIVE_PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_hidden_size],
            false,
        );
    }
    Ok(expected)
}

fn inspect_inventory(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let actual_names: BTreeSet<_> = content.tensor_infos.keys().cloned().collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in expected_names.intersection(&actual_names) {
        let info = &content.tensor_infos[name];
        let found = info.shape.dims();
        let wanted = &expected[name].shape;
        if found != wanted {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected {wanted:?}, found {:?} ({:?})",
                found, info.ggml_dtype
            ));
        }
    }
    Ok(MmprojLoadReport {
        loaded_tensors: actual_names.into_iter().collect(),
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: NATIVE_VISION_ROOT.to_string(),
        resolved_projector_root: NATIVE_PROJECTOR_ROOT.to_string(),
        target_dtype: format!("{dtype:?}"),
        target_device: format!("{device:?}"),
    })
}

fn validate_ranges_and_sizes(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    file_size: u64,
    dense_element_size: u64,
) -> Result<AllocationReport> {
    let alignment = optional_positive_usize(content, "general.alignment")?.unwrap_or(32) as u64;
    if !alignment.is_power_of_two() {
        candle::bail!("GGUF MMProj alignment {alignment} must be a power of two")
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve(expected.len())
        .map_err(|_| candle::Error::Msg("GGUF MMProj range allocation failed".into()))?;
    let mut dense_bytes = 0u64;
    let mut source_bytes_total = 0u64;
    let mut largest_transient_bytes = 0u64;
    for name in expected.keys() {
        let info = &content.tensor_infos[name];
        let element_count = checked_element_count(info.shape.dims(), name)?;
        let block_size = info.ggml_dtype.block_size() as u64;
        if element_count % block_size != 0 {
            candle::bail!(
                "GGUF MMProj tensor {name:?} element count {element_count} is not divisible by {:?} block size {block_size}",
                info.ggml_dtype
            )
        }
        let source_bytes = element_count
            .checked_div(block_size)
            .and_then(|blocks| blocks.checked_mul(info.ggml_dtype.type_size() as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!("GGUF MMProj tensor {name:?} byte size overflowed"))
            })?;
        source_bytes_total = source_bytes_total
            .checked_add(source_bytes)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj source byte total overflowed".into()))?;
        let target_bytes = element_count
            .checked_mul(dense_element_size)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} dense byte size overflowed"
                ))
            })?;
        dense_bytes = dense_bytes
            .checked_add(target_bytes)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj dense byte total overflowed".into()))?;
        if dense_bytes > MAX_DENSE_MMPROJ_BYTES {
            candle::bail!(
                "GGUF MMProj dense allocation {dense_bytes} exceeds {MAX_DENSE_MMPROJ_BYTES} bytes"
            )
        }
        // Loading can briefly hold both the input byte buffer and quantized
        // storage. F16/BF16 targets also coexist with the F32 dequantization
        // result, and layout conversion can retain one target-dtype scratch
        // tensor. Bound that conservative estimate alongside the retained map.
        let f32_dequant_scratch = if dense_element_size < 4 {
            element_count.checked_mul(4).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} F32 scratch size overflowed"
                ))
            })?
        } else {
            0
        };
        let transient_bytes = source_bytes
            .checked_mul(2)
            .and_then(|value| value.checked_add(f32_dequant_scratch))
            .and_then(|value| value.checked_add(target_bytes))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} transient byte estimate overflowed"
                ))
            })?;
        largest_transient_bytes = largest_transient_bytes.max(transient_bytes);
        if info.offset % alignment != 0 {
            candle::bail!(
                "GGUF MMProj tensor {name:?} relative offset {} is not aligned to {alignment}",
                info.offset
            )
        }
        let start = content
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj tensor offset overflowed".into()))?;
        let end = start.checked_add(source_bytes).ok_or_else(|| {
            candle::Error::Msg(format!("GGUF MMProj tensor {name:?} range overflowed"))
        })?;
        if end > file_size {
            candle::bail!(
                "GGUF MMProj tensor {name:?} ends at byte {end}, beyond file size {file_size}"
            )
        }
        ranges.push((start, end, name));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = content.tensor_data_offset;
    for (start, end, name) in ranges {
        if start < previous_end {
            candle::bail!("GGUF MMProj tensor {name:?} overlaps another tensor")
        }
        previous_end = end;
    }
    let estimated_peak_byte_count = dense_bytes
        .checked_add(largest_transient_bytes)
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj peak byte estimate overflowed".into()))?;
    if estimated_peak_byte_count > MAX_ESTIMATED_MMPROJ_PEAK_BYTES {
        candle::bail!(
            "GGUF MMProj estimated peak allocation {estimated_peak_byte_count} exceeds {MAX_ESTIMATED_MMPROJ_PEAK_BYTES} bytes"
        )
    }
    Ok(AllocationReport {
        source_byte_count: source_bytes_total,
        dense_byte_count: dense_bytes,
        estimated_peak_byte_count,
    })
}

fn processor_metadata_json(metadata: &ParsedMetadata) -> serde_json::Value {
    let mut image_processor = serde_json::Map::new();
    image_processor.insert(
        "encoder_patch_size".into(),
        serde_json::Value::from(metadata.patch_size),
    );
    image_processor.insert(
        "downsample_factor".into(),
        serde_json::Value::from(metadata.downsample_factor),
    );
    image_processor.insert("image_mean".into(), serde_json::json!(metadata.image_mean));
    image_processor.insert("image_std".into(), serde_json::json!(metadata.image_std));
    if let Some((min_tiles, max_tiles, image_size)) = metadata.preproc {
        image_processor.insert("min_tiles".into(), serde_json::Value::from(min_tiles));
        image_processor.insert("max_tiles".into(), serde_json::Value::from(max_tiles));
        image_processor.insert("tile_size".into(), serde_json::Value::from(image_size));
    }
    serde_json::json!({ "image_processor": image_processor })
}

fn metadata_value<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("GGUF MMProj is missing metadata key {key:?}")))
}

fn required_string<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a str> {
    match metadata_value(content, key)? {
        Value::String(value) => Ok(value),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a string, got {:?}",
            value.value_type()
        ),
    }
}

fn optional_string<'a>(content: &'a gguf_file::Content, key: &str) -> Result<Option<&'a str>> {
    match content.metadata.get(key) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(value) => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a string, got {:?}",
            value.value_type()
        ),
    }
}

fn required_bool(content: &gguf_file::Content, key: &str) -> Result<bool> {
    match metadata_value(content, key)? {
        Value::Bool(value) => Ok(*value),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a boolean, got {:?}",
            value.value_type()
        ),
    }
}

fn positive_usize_value(value: &Value, key: &str) -> Result<usize> {
    let raw = match value {
        Value::U8(value) => *value as u64,
        Value::U16(value) => *value as u64,
        Value::U32(value) => *value as u64,
        Value::U64(value) => *value,
        Value::I8(value) if *value > 0 => *value as u64,
        Value::I16(value) if *value > 0 => *value as u64,
        Value::I32(value) if *value > 0 => *value as u64,
        Value::I64(value) if *value > 0 => *value as u64,
        _ => candle::bail!("GGUF MMProj metadata {key:?} must be a positive integer"),
    };
    if raw == 0 {
        candle::bail!("GGUF MMProj metadata {key:?} must be a positive integer")
    }
    usize::try_from(raw).map_err(|_| {
        candle::Error::Msg(format!(
            "GGUF MMProj metadata {key:?} does not fit this platform"
        ))
    })
}

fn required_positive_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    positive_usize_value(metadata_value(content, key)?, key)
}

fn optional_positive_usize(content: &gguf_file::Content, key: &str) -> Result<Option<usize>> {
    content
        .metadata
        .get(key)
        .map(|value| positive_usize_value(value, key))
        .transpose()
}

fn required_positive_float(content: &gguf_file::Content, key: &str) -> Result<f64> {
    let value = match metadata_value(content, key)? {
        Value::F32(value) => *value as f64,
        Value::F64(value) => *value,
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be an f32 or f64, got {:?}",
            value.value_type()
        ),
    };
    if !value.is_finite() || value <= 0.0 {
        candle::bail!("GGUF MMProj metadata {key:?} must be finite and positive")
    }
    Ok(value)
}

fn required_f32_triplet(content: &gguf_file::Content, key: &str) -> Result<[f32; 3]> {
    let values = match metadata_value(content, key)? {
        Value::Array(values) if values.len() == 3 => values,
        Value::Array(values) => candle::bail!(
            "GGUF MMProj metadata {key:?} must contain 3 values, got {}",
            values.len()
        ),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be an array, got {:?}",
            value.value_type()
        ),
    };
    let mut output = [0f32; 3];
    for (index, value) in values.iter().enumerate() {
        let value = match value {
            Value::F32(value) => *value,
            Value::F64(value) => *value as f32,
            value => candle::bail!(
                "GGUF MMProj metadata {key:?}[{index}] must be an f32 or f64, got {:?}",
                value.value_type()
            ),
        };
        if !value.is_finite() {
            candle::bail!("GGUF MMProj metadata {key:?}[{index}] must be finite")
        }
        output[index] = value;
    }
    Ok(output)
}

fn required_tensor_shape(content: &gguf_file::Content, name: &str) -> Result<Vec<usize>> {
    content
        .tensor_infos
        .get(name)
        .map(|info| info.shape.dims().to_vec())
        .ok_or_else(|| candle::Error::Msg(format!("GGUF MMProj is missing tensor {name:?}")))
}

fn checked_element_count(shape: &[usize], name: &str) -> Result<u64> {
    shape.iter().try_fold(1u64, |count, &dimension| {
        let dimension = u64::try_from(dimension).map_err(candle::Error::wrap)?;
        if dimension == 0 {
            candle::bail!("GGUF MMProj tensor {name:?} has a zero dimension")
        }
        count.checked_mul(dimension).ok_or_else(|| {
            candle::Error::Msg(format!(
                "GGUF MMProj tensor {name:?} element count overflowed"
            ))
        })
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::models::lfm2_vl::{
        CropKind, CropMeta, ImageMeta, Lfm2VlConfig, Lfm2VlModel, ProcessedVisionBatch,
    };
    use crate::models::{lfm2, quantized_lfm2};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::Tensor;
    use sha2::{Digest, Sha256};
    use std::io::Cursor;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
    }

    fn fixture_batch(tensors: &HashMap<String, Tensor>) -> Result<ProcessedVisionBatch> {
        Ok(ProcessedVisionBatch {
            pixel_values: fixture_tensor(tensors, "input.pixel_values")?.clone(),
            pixel_attention_mask: fixture_tensor(tensors, "input.pixel_attention_mask")?.clone(),
            spatial_shapes: fixture_tensor(tensors, "input.spatial_shapes")?.clone(),
            crops: vec![CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn max_abs(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("GGUF MMProj parity element count mismatch")
        }
        Ok(actual
            .iter()
            .zip(expected)
            .map(|(&lhs, rhs)| (lhs - rhs).abs())
            .fold(0f32, f32::max))
    }

    fn metadata_entries(config: &Lfm2VlConfig) -> Result<Vec<(String, Value)>> {
        let vision = &config.vision_config;
        let base_side = (vision.num_patches as f64).sqrt() as usize;
        if base_side * base_side != vision.num_patches {
            candle::bail!("tiny GGUF fixture positions must form a square")
        }
        Ok(vec![
            ("general.architecture".into(), Value::String("clip".into())),
            ("general.type".into(), Value::String("mmproj".into())),
            (
                "general.name".into(),
                Value::String("deterministic-tiny-lfm2-vl-mmproj".into()),
            ),
            ("clip.projector_type".into(), Value::String("lfm2".into())),
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            ("clip.use_gelu".into(), Value::Bool(true)),
            (
                "clip.vision.image_size".into(),
                Value::U32((base_side * vision.patch_size) as u32),
            ),
            (
                "clip.vision.patch_size".into(),
                Value::U32(vision.patch_size as u32),
            ),
            (
                "clip.vision.embedding_length".into(),
                Value::U32(vision.hidden_size as u32),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                Value::U32(vision.intermediate_size as u32),
            ),
            (
                "clip.vision.block_count".into(),
                Value::U32(vision.num_hidden_layers as u32),
            ),
            (
                "clip.vision.attention.head_count".into(),
                Value::U32(vision.num_attention_heads as u32),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                Value::F32(vision.layer_norm_eps as f32),
            ),
            (
                "clip.vision.image_mean".into(),
                Value::Array(vec![Value::F32(0.5); 3]),
            ),
            (
                "clip.vision.image_std".into(),
                Value::Array(vec![Value::F32(0.5); 3]),
            ),
            (
                "clip.vision.projection_dim".into(),
                Value::U32(config.text_config.hidden_size as u32),
            ),
            (
                "clip.vision.projector.scale_factor".into(),
                Value::U32(config.downsample_factor as u32),
            ),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tiny_mmproj_gguf(
        tensors: &HashMap<String, Tensor>,
        config: &Lfm2VlConfig,
        quantize_linears: bool,
        omitted: &[&str],
        metadata_override: Option<(&str, Value)>,
        malformed_patch_rank: bool,
    ) -> Result<Vec<u8>> {
        let runtime = Lfm2VlMmprojConfig::from(config);
        let expected = expected_tensors(&runtime)?;
        let mut qtensors = Vec::new();
        for (gguf_name, tensor_info) in expected {
            if omitted.contains(&gguf_name.as_str()) {
                continue;
            }
            let fixture_name = format!("weights.{}", tensor_info.native_name);
            let fixture_name = if tensors.contains_key(&fixture_name) {
                fixture_name
            } else {
                fixture_name.replace(
                    "weights.model.vision_tower.vision_model.",
                    "weights.model.vision_tower.",
                )
            };
            let mut tensor = fixture_tensor(tensors, &fixture_name)?.clone();
            if tensor_info.patch_layout {
                let vision = &config.vision_config;
                tensor = tensor
                    .reshape((
                        vision.hidden_size,
                        vision.patch_size,
                        vision.patch_size,
                        vision.num_channels,
                    ))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?;
                if malformed_patch_rank {
                    tensor = tensor.reshape((
                        vision.hidden_size,
                        vision.num_channels,
                        vision.patch_size * vision.patch_size,
                    ))?;
                }
            }
            let eligible_linear = gguf_name == "mm.1.weight"
                || gguf_name == "mm.2.weight"
                || (gguf_name.contains(".attn_") && gguf_name.ends_with(".weight"))
                || (gguf_name.contains(".ffn_") && gguf_name.ends_with(".weight"));
            let last_dimension = tensor.dim(tensor.rank() - 1)?;
            let dtype = if quantize_linears
                && eligible_linear
                && last_dimension.is_multiple_of(GgmlDType::Q8_0.block_size())
            {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            qtensors.push((gguf_name, QTensor::quantize(&tensor.contiguous()?, dtype)?));
        }

        let mut metadata = metadata_entries(config)?;
        if let Some((name, value)) = metadata_override {
            let entry = metadata
                .iter_mut()
                .find(|(key, _)| key == name)
                .ok_or_else(|| candle::Error::Msg(format!("unknown test metadata key {name}")))?;
            entry.1 = value;
        }
        let metadata_refs: Vec<_> = metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let tensor_refs: Vec<_> = qtensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        let mut output = Cursor::new(Vec::new());
        gguf_file::write(&mut output, &metadata_refs, &tensor_refs)?;
        Ok(output.into_inner())
    }

    fn synthetic_text_metadata(config: &Lfm2VlConfig) -> quantized_lfm2::Lfm2GgufMetadata {
        let text = config.text_model_config().expect("fixed tiny text config");
        quantized_lfm2::Lfm2GgufMetadata {
            architecture: "lfm2".into(),
            embedding_length: text.hidden_size,
            context_length: text.max_position_embeddings,
            block_count: text.num_hidden_layers,
            head_count: text.num_attention_heads,
            head_count_kv: text
                .layer_types
                .iter()
                .map(|kind| match kind {
                    lfm2::LayerType::FullAttention => text.num_key_value_heads,
                    lfm2::LayerType::Conv => 0,
                })
                .collect(),
            rms_norm_eps: text.norm_eps,
            rope_freq_base: text.rope_theta,
            shortconv_l_cache: text.conv_l_cache,
            tied_output: true,
        }
    }

    #[test]
    fn dense_gguf_mmproj_matches_native_image_features() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let native = Lfm2VlModel::new(
            &config,
            VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?.pp("weights"),
        )?;
        let batch = fixture_batch(&tensors)?;
        let native_features = native.encode_images(&batch, 1)?;

        let bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        eprintln!(
            "lfm2_vl deterministic dense MMProj GGUF SHA-256: {:x}",
            Sha256::digest(&bytes)
        );
        let mut reader = Cursor::new(bytes);
        let mmproj = Mmproj::from_gguf(&mut reader, DType::F32, &device, 3)?;
        let gguf = mmproj
            .metadata
            .gguf_metadata()
            .ok_or_else(|| candle::Error::Msg("expected direct GGUF MMProj metadata".into()))?;
        assert_eq!(gguf.tensor_count, 43);
        assert_eq!(gguf.quantized_tensor_count, 0);
        assert_eq!(gguf.projector_type, "lfm2");
        assert_eq!(gguf.general_architecture, "clip");
        assert!(gguf.source_byte_count > 0);
        assert!(gguf.dense_byte_count > 0);
        assert!(gguf.estimated_peak_byte_count > gguf.dense_byte_count);
        assert!(mmproj.metadata.use_image_special_tokens);
        assert!(mmproj.metadata.split_manifest().is_none());
        let loaded_features = mmproj.encode_images(&batch, 1)?;
        let error = max_abs(&loaded_features.embeddings, &native_features.embeddings)?;
        eprintln!("lfm2_vl dense GGUF MMProj image features: max_abs={error:.9e}");
        assert!(error <= 1e-6, "dense GGUF image feature error {error}");
        assert_eq!(
            loaded_features.per_crop_ranges,
            native_features.per_crop_ranges
        );
        assert_eq!(
            loaded_features.per_image_ranges,
            native_features.per_image_ranges
        );
        Ok(())
    }

    #[test]
    fn q8_gguf_mmproj_dequantizes_and_pairs() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let dense_bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let q8_bytes = tiny_mmproj_gguf(&tensors, &config, true, &[], None, false)?;
        let mut dense_reader = Cursor::new(dense_bytes);
        let dense = Mmproj::from_gguf(&mut dense_reader, DType::F32, &device, 3)?;
        let mut q8_reader = Cursor::new(q8_bytes);
        let q8 = Mmproj::from_gguf(&mut q8_reader, DType::F32, &device, 3)?;
        assert!(q8
            .metadata
            .gguf_metadata()
            .is_some_and(|metadata| metadata.quantized_tensor_count > 0));
        let batch = fixture_batch(&tensors)?;
        let dense_features = dense.encode_images(&batch, 1)?;
        let q8_features = q8.encode_images(&batch, 1)?;
        let error = max_abs(&q8_features.embeddings, &dense_features.embeddings)?;
        eprintln!("lfm2_vl Q8_0 dequantized MMProj image features: max_abs={error:.9e}");
        assert!(error <= 2e-2, "Q8_0 GGUF image feature error {error}");

        let text = synthetic_text_metadata(&config);
        let report = q8.metadata.validate_pair(&text, 2, 2, 3)?;
        assert_eq!(report.text_hidden_size, 12);
        assert!(q8.metadata.validate_pair(&text, 4, 2, 3).is_err());
        assert!(q8.metadata.validate_pair(&text, 2, 1, 3).is_err());
        assert!(q8.metadata.validate_pair(&text, 2, 2, 4).is_err());
        let mut wrong_text = text;
        wrong_text.embedding_length += 1;
        assert!(q8.metadata.validate_pair(&wrong_text, 2, 2, 3).is_err());
        Ok(())
    }

    #[test]
    fn gguf_mmproj_rejects_malformed_metadata_inventory_layout_and_payload() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;

        let wrong_projector = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &[],
            Some(("clip.projector_type", Value::String("mlp".into()))),
            false,
        )?;
        let error = Mmproj::from_gguf(&mut Cursor::new(wrong_projector), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("projector_type") && error.contains("lfm2"));

        let mut missing_general_type =
            tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let key = b"general.type";
        let key_offset = missing_general_type
            .windows(key.len())
            .position(|window| window == key)
            .ok_or_else(|| candle::Error::Msg("missing general.type test key".into()))?;
        missing_general_type[key_offset] = b'x';
        let error = Mmproj::from_gguf(
            &mut Cursor::new(missing_general_type),
            DType::F32,
            &device,
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing metadata key") && error.contains("general.type"));

        let wrong_type = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &[],
            Some(("clip.vision.patch_size", Value::String("two".into()))),
            false,
        )?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(wrong_type), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("positive integer")
        );

        let missing = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["v.blk.0.attn_q.bias"],
            None,
            false,
        )?;
        let error = Mmproj::from_gguf(&mut Cursor::new(missing), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing") && error.contains("v.blk.0.attn_q.bias"));

        let mut unexpected = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let original_name = b"v.post_ln.bias";
        let name_offset = unexpected
            .windows(original_name.len())
            .position(|window| window == original_name)
            .ok_or_else(|| candle::Error::Msg("missing fixed GGUF test name".into()))?;
        unexpected[name_offset] = b'x';
        let error = Mmproj::from_gguf(&mut Cursor::new(unexpected), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing") && error.contains("unexpected"));

        let wrong_patch = tiny_mmproj_gguf(&tensors, &config, false, &[], None, true)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(wrong_patch), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("patch_embd.weight")
        );

        let mut truncated = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        truncated.truncate(truncated.len() - 1);
        let error = Mmproj::from_gguf(&mut Cursor::new(truncated), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("beyond file size"));

        let mut oversized_counts = Vec::new();
        oversized_counts.extend_from_slice(b"GGUF");
        oversized_counts.extend_from_slice(&3u32.to_le_bytes());
        oversized_counts.extend_from_slice(&1u64.to_le_bytes());
        oversized_counts.extend_from_slice(&(MAX_GGUF_MMPROJ_METADATA + 1).to_le_bytes());
        let error = Mmproj::from_gguf(&mut Cursor::new(oversized_counts), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("metadata_kv_count") && error.contains("exceeds max"));

        let mut oversized_string = Vec::new();
        oversized_string.extend_from_slice(b"GGUF");
        oversized_string.extend_from_slice(&3u32.to_le_bytes());
        oversized_string.extend_from_slice(&0u64.to_le_bytes());
        oversized_string.extend_from_slice(&1u64.to_le_bytes());
        oversized_string.extend_from_slice(&(MAX_GGUF_MMPROJ_STRING_BYTES + 1).to_le_bytes());
        oversized_string.extend_from_slice(&[0; 9]);
        let error = Mmproj::from_gguf(&mut Cursor::new(oversized_string), DType::F32, &device, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("string length") && error.contains("exceeds max"));

        let valid = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(valid), DType::U8, &device, 3)
                .unwrap_err()
                .to_string()
                .contains("F32, F16, or BF16")
        );
        Ok(())
    }

    #[test]
    fn gguf_mmproj_range_validation_rejects_misalignment_overlap_and_overflow() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let bytes = tiny_mmproj_gguf(&tensors, &config, false, &[], None, false)?;
        let runtime = Lfm2VlMmprojConfig::from(&config);
        let expected = expected_tensors(&runtime)?;
        let file_size = bytes.len() as u64;

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = 1;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not aligned"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.weight")
            .unwrap()
            .offset = 0;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = 0;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlaps another tensor"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes.clone()))?;
        content
            .tensor_infos
            .get_mut("v.patch_embd.bias")
            .unwrap()
            .offset = u64::MAX - 31;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("offset overflowed"));

        let mut content = gguf_file::Content::read(&mut Cursor::new(bytes))?;
        content.tensor_data_offset = file_size + 32;
        let error = validate_ranges_and_sizes(&content, &expected, file_size, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("beyond file size"));
        Ok(())
    }

    #[test]
    fn gguf_mmproj_optional_projector_layer_norm_is_not_synthesized() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let bytes = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.input_norm.weight", "mm.input_norm.bias"],
            None,
            false,
        )?;
        let mmproj = Mmproj::from_gguf(&mut Cursor::new(bytes), DType::F32, &device, 3)?;
        assert_eq!(mmproj.report.loaded_tensors.len(), 41);

        let incomplete = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.input_norm.bias"],
            None,
            false,
        )?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(incomplete), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("appear together")
        );

        let no_biases = tiny_mmproj_gguf(
            &tensors,
            &config,
            false,
            &["mm.1.bias", "mm.2.bias"],
            None,
            false,
        )?;
        let no_biases = Mmproj::from_gguf(&mut Cursor::new(no_biases), DType::F32, &device, 3)?;
        assert_eq!(no_biases.report.loaded_tensors.len(), 41);

        let incomplete_biases =
            tiny_mmproj_gguf(&tensors, &config, false, &["mm.2.bias"], None, false)?;
        assert!(
            Mmproj::from_gguf(&mut Cursor::new(incomplete_biases), DType::F32, &device, 3,)
                .unwrap_err()
                .to_string()
                .contains("biases must appear together")
        );
        Ok(())
    }
}
