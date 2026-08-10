//! Split dense MMProj loading and quantized-text hybrid execution.

use super::model::{encode_images_with_parts, merge_projected_embeddings};
use super::{EncodedImages, ImageTokenSpan, Lfm2VlConfig, Lfm2VlProjector, ProcessedVisionBatch};
use crate::models::{quantized_lfm2, siglip2};
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MMPROJ_FORMAT: &str = "candle-mmproj";
const MMPROJ_VERSION: u32 = 1;
const MMPROJ_NAMESPACE_VERSION: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_MMPROJ_TENSORS: usize = 16_384;
const MAX_VISION_LAYERS: usize = 512;
const VISION_ROOT: &str = "model.vision_tower.vision_model";
const PROJECTOR_ROOT: &str = "model.multi_modal_projector";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct MmprojTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub nbytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MmprojManifest {
    pub format: String,
    pub version: u32,
    pub architecture: String,
    pub source_model: String,
    pub source_revision: String,
    pub source_safetensors: String,
    pub source_safetensors_sha256: String,
    pub source_model_config_sha256: String,
    pub expected_text_hidden_size: usize,
    pub expected_text_layer_count: usize,
    pub vision_hidden_size: usize,
    pub vision_layer_count: usize,
    pub patch_size: usize,
    pub downsample_factor: usize,
    pub image_token_id: u32,
    pub tensor_namespace_version: u32,
    pub tensor_count: usize,
    pub tensor_inventory: BTreeMap<String, MmprojTensorInfo>,
    pub mmproj_safetensors_sha256: String,
    pub processor_config_sha256: String,
    pub model_config: Lfm2VlConfig,
}

impl MmprojManifest {
    pub fn from_json(json: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(json)
            .map_err(|err| candle::Error::Msg(format!("invalid split MMProj manifest: {err}")))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != MMPROJ_FORMAT || self.version != MMPROJ_VERSION {
            candle::bail!(
                "unsupported split MMProj format {:?} version {}; expected {MMPROJ_FORMAT:?} version {MMPROJ_VERSION}",
                self.format,
                self.version
            )
        }
        if self.architecture != "lfm2_vl" && self.architecture != "lfm2-vl" {
            candle::bail!(
                "unsupported split MMProj architecture {:?}",
                self.architecture
            )
        }
        if self.source_model.trim().is_empty() || self.source_model.trim() != self.source_model {
            candle::bail!(
                "split MMProj source model must be a non-empty identifier without outer whitespace"
            )
        }
        validate_lower_hex(
            "source revision",
            &self.source_revision,
            &[40usize, 64usize],
        )?;
        if self.source_safetensors.trim().is_empty() {
            candle::bail!("split MMProj source safetensors name must not be empty")
        }
        if self.tensor_namespace_version != MMPROJ_NAMESPACE_VERSION {
            candle::bail!(
                "unsupported split MMProj tensor namespace version {}; expected {MMPROJ_NAMESPACE_VERSION}",
                self.tensor_namespace_version
            )
        }
        if self.tensor_count == 0
            || self.tensor_count > MAX_MMPROJ_TENSORS
            || self.tensor_count != self.tensor_inventory.len()
        {
            candle::bail!(
                "split MMProj tensor_count {} does not match bounded inventory length {}",
                self.tensor_count,
                self.tensor_inventory.len()
            )
        }
        if self.vision_layer_count == 0 || self.vision_layer_count > MAX_VISION_LAYERS {
            candle::bail!(
                "invalid split MMProj vision layer count {}",
                self.vision_layer_count
            )
        }
        self.model_config.validate()?;
        let model_text_layers = self.model_config.text_config.num_hidden_layers;
        let model_vision = &self.model_config.vision_config;
        for (label, manifest_value, model_value) in [
            (
                "text hidden size",
                self.expected_text_hidden_size,
                self.model_config.text_config.hidden_size,
            ),
            (
                "text layer count",
                self.expected_text_layer_count,
                model_text_layers,
            ),
            (
                "vision hidden size",
                self.vision_hidden_size,
                model_vision.hidden_size,
            ),
            (
                "vision layer count",
                self.vision_layer_count,
                model_vision.num_hidden_layers,
            ),
            ("patch size", self.patch_size, model_vision.patch_size),
            (
                "downsample factor",
                self.downsample_factor,
                self.model_config.downsample_factor,
            ),
        ] {
            if manifest_value != model_value {
                candle::bail!(
                    "split MMProj {label} {manifest_value} does not match embedded model config {model_value}"
                )
            }
        }
        if self.image_token_id != self.model_config.image_token_id {
            candle::bail!(
                "split MMProj image token {} does not match embedded model config {}",
                self.image_token_id,
                self.model_config.image_token_id
            )
        }
        for (label, value) in [
            ("source safetensors", &self.source_safetensors_sha256),
            ("source model config", &self.source_model_config_sha256),
            ("MMProj safetensors", &self.mmproj_safetensors_sha256),
            ("processor config", &self.processor_config_sha256),
        ] {
            validate_sha256(label, value)?;
        }

        let expected = expected_tensor_shapes(&self.model_config)?;
        let manifest_names: BTreeSet<_> = self.tensor_inventory.keys().cloned().collect();
        let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
        if manifest_names != expected_names {
            let missing: Vec<_> = expected_names
                .difference(&manifest_names)
                .cloned()
                .collect();
            let unexpected: Vec<_> = manifest_names
                .difference(&expected_names)
                .cloned()
                .collect();
            candle::bail!(
                "split MMProj manifest inventory disagrees with model config; missing={missing:?}, unexpected={unexpected:?}"
            )
        }
        for (name, expected_shape) in expected {
            let info = &self.tensor_inventory[&name];
            if info.shape != expected_shape {
                candle::bail!(
                    "split MMProj manifest tensor {name:?} has shape {:?}, expected {:?}",
                    info.shape,
                    expected_shape
                )
            }
            let element_size = dense_dtype_size(&info.dtype).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj tensor {name:?} has unsupported dense dtype {:?}",
                    info.dtype
                ))
            })?;
            let element_count = info.shape.iter().try_fold(1usize, |count, &dimension| {
                if dimension == 0 {
                    None
                } else {
                    count.checked_mul(dimension)
                }
            });
            let expected_nbytes = element_count
                .and_then(|count| count.checked_mul(element_size))
                .ok_or_else(|| {
                    candle::Error::Msg(format!("split MMProj tensor {name:?} byte size overflows"))
                })?;
            if info.nbytes != expected_nbytes {
                candle::bail!(
                    "split MMProj tensor {name:?} declares {} bytes, expected {expected_nbytes}",
                    info.nbytes
                )
            }
        }
        Ok(())
    }

    pub fn validate_pair(
        &self,
        text: &quantized_lfm2::Lfm2GgufMetadata,
        processor_patch_size: usize,
        processor_downsample_factor: usize,
        tokenizer_image_token_id: u32,
    ) -> Result<PairingReport> {
        self.validate()?;
        if text.architecture != "lfm2" {
            candle::bail!(
                "split MMProj requires quantized text architecture \"lfm2\", got {:?}",
                text.architecture
            )
        }
        if text.embedding_length != self.expected_text_hidden_size {
            candle::bail!(
                "split MMProj output width {} does not match quantized text hidden size {}",
                self.expected_text_hidden_size,
                text.embedding_length
            )
        }
        if text.block_count != self.expected_text_layer_count {
            candle::bail!(
                "split MMProj expects {} text layers, but GGUF declares {}",
                self.expected_text_layer_count,
                text.block_count
            )
        }
        if processor_patch_size != self.patch_size {
            candle::bail!(
                "processor patch size {processor_patch_size} does not match split MMProj {}",
                self.patch_size
            )
        }
        if processor_downsample_factor != self.downsample_factor {
            candle::bail!(
                "processor downsample factor {processor_downsample_factor} does not match split MMProj {}",
                self.downsample_factor
            )
        }
        if tokenizer_image_token_id != self.image_token_id {
            candle::bail!(
                "tokenizer image token ID {tokenizer_image_token_id} does not match split MMProj {}",
                self.image_token_id
            )
        }
        Ok(PairingReport {
            text_architecture: text.architecture.clone(),
            text_hidden_size: text.embedding_length,
            text_layer_count: text.block_count,
            vision_layer_count: self.vision_layer_count,
            patch_size: self.patch_size,
            downsample_factor: self.downsample_factor,
            image_token_id: self.image_token_id,
            text_output_resolution: if text.tied_output {
                "tied token embeddings".to_string()
            } else {
                "explicit GGUF output tensor".to_string()
            },
            only_projected_features_cross_devices: true,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MmprojMetadata {
    pub architecture: String,
    pub vision_hidden_size: usize,
    pub text_hidden_size: usize,
    pub patch_size: usize,
    pub downsample_factor: usize,
    pub image_token_id: u32,
    /// Kept neutral at this crate boundary because `candle-vlm` depends on
    /// `candle-transformers`; `candle-vlm` provides the typed conversion.
    pub processor: serde_json::Value,
    pub source_model: Option<String>,
    pub source_revision: Option<String>,
    pub manifest: MmprojManifest,
}

#[derive(Debug, Clone)]
pub struct MmprojLoadReport {
    pub loaded_tensors: Vec<String>,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub shape_or_dtype_mismatches: Vec<String>,
    pub resolved_vision_root: String,
    pub resolved_projector_root: String,
    pub target_dtype: String,
    pub target_device: String,
}

impl MmprojLoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_tensors.is_empty()
            && self.unexpected_tensors.is_empty()
            && self.shape_or_dtype_mismatches.is_empty()
    }

    fn require_clean(&self) -> Result<()> {
        if !self.is_clean() {
            candle::bail!(
                "split MMProj tensor validation failed; missing={:?}, unexpected={:?}, mismatches={:?}",
                self.missing_tensors,
                self.unexpected_tensors,
                self.shape_or_dtype_mismatches
            )
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingReport {
    pub text_architecture: String,
    pub text_hidden_size: usize,
    pub text_layer_count: usize,
    pub vision_layer_count: usize,
    pub patch_size: usize,
    pub downsample_factor: usize,
    pub image_token_id: u32,
    pub text_output_resolution: String,
    pub only_projected_features_cross_devices: bool,
}

pub struct Mmproj {
    pub vision_tower: siglip2::Siglip2VisionModel,
    pub projector: Lfm2VlProjector,
    pub metadata: MmprojMetadata,
    pub report: MmprojLoadReport,
    device: Device,
    dtype: DType,
}

impl Mmproj {
    pub fn load(bundle_dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        Self::from_files(
            bundle_dir.join("mmproj.safetensors"),
            bundle_dir.join("mmproj.json"),
            bundle_dir.join("processor_config.json"),
            dtype,
            device,
        )
    }

    pub fn from_files(
        weights_path: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
        processor_path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weights_path = weights_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let processor_path = processor_path.as_ref();
        let manifest_json = read_bounded_text(manifest_path, "split MMProj manifest")?;
        let manifest = MmprojManifest::from_json(&manifest_json)?;
        let weights_bytes = read_weight_bytes(weights_path, "split MMProj safetensors", &manifest)?;
        verify_bytes_sha256(
            &weights_bytes,
            &manifest.mmproj_safetensors_sha256,
            "split MMProj safetensors",
        )?;
        let processor_bytes = read_bounded_bytes(processor_path, "split MMProj processor config")?;
        verify_bytes_sha256(
            &processor_bytes,
            &manifest.processor_config_sha256,
            "split MMProj processor config",
        )?;
        let processor: serde_json::Value =
            serde_json::from_slice(&processor_bytes).map_err(|err| {
                candle::Error::Msg(format!("invalid split MMProj processor config: {err}"))
            })?;
        let (processor_patch_size, processor_downsample_factor) =
            processor_pair_fields(&processor)?;
        if processor_patch_size != manifest.patch_size
            || processor_downsample_factor != manifest.downsample_factor
        {
            candle::bail!(
                "split MMProj processor/model mismatch: processor patch/factor [{processor_patch_size}, {processor_downsample_factor}], manifest [{}, {}]",
                manifest.patch_size,
                manifest.downsample_factor
            )
        }

        let report = inspect_safetensors_bytes(&weights_bytes, &manifest, dtype, device)?;
        report.require_clean()?;
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, dtype, device)?;
        let config = manifest.model_config.clone();
        let vision_tower =
            siglip2::Siglip2VisionModel::new(&config.vision_config, vb.pp(VISION_ROOT))?;
        let projector = Lfm2VlProjector::new(&config, vb.pp(PROJECTOR_ROOT))?;
        let metadata = MmprojMetadata {
            architecture: manifest.architecture.clone(),
            vision_hidden_size: manifest.vision_hidden_size,
            text_hidden_size: manifest.expected_text_hidden_size,
            patch_size: manifest.patch_size,
            downsample_factor: manifest.downsample_factor,
            image_token_id: manifest.image_token_id,
            processor,
            source_model: Some(manifest.source_model.clone()),
            source_revision: Some(manifest.source_revision.clone()),
            manifest,
        };
        Ok(Self {
            vision_tower,
            projector,
            metadata,
            report,
            device: device.clone(),
            dtype,
        })
    }

    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages> {
        let device_inputs = ProcessedVisionBatch {
            pixel_values: inputs
                .pixel_values
                .to_device(&self.device)?
                .to_dtype(self.dtype)?,
            pixel_attention_mask: inputs.pixel_attention_mask.to_device(&self.device)?,
            spatial_shapes: inputs.spatial_shapes.to_device(&self.device)?,
            crops: inputs.crops.clone(),
            images: inputs.images.clone(),
        };
        encode_images_with_parts(
            &self.vision_tower,
            &self.projector,
            &self.metadata.manifest.model_config,
            &device_inputs,
            vision_batch_size,
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Quantized GGUF LFM2 text paired with a split dense vision/projector bundle.
pub struct QuantizedLfm2VlModel {
    text: quantized_lfm2::ModelWeights,
    mmproj: Mmproj,
    pairing: PairingReport,
}

impl QuantizedLfm2VlModel {
    pub fn new(
        text: quantized_lfm2::ModelWeights,
        mmproj: Mmproj,
        processor_patch_size: usize,
        processor_downsample_factor: usize,
        tokenizer_image_token_id: u32,
    ) -> Result<Self> {
        if text.hidden_size() != text.metadata().embedding_length {
            candle::bail!(
                "quantized LFM2 embedding tensor width {} does not match GGUF metadata {}",
                text.hidden_size(),
                text.metadata().embedding_length
            )
        }
        if mmproj.metadata.image_token_id as usize >= text.vocab_size() {
            candle::bail!(
                "split MMProj image token ID {} is outside quantized text vocabulary size {}",
                mmproj.metadata.image_token_id,
                text.vocab_size()
            )
        }
        let pairing = mmproj.metadata.manifest.validate_pair(
            text.metadata(),
            processor_patch_size,
            processor_downsample_factor,
            tokenizer_image_token_id,
        )?;
        Ok(Self {
            text,
            mmproj,
            pairing,
        })
    }

    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages> {
        self.mmproj.encode_images(inputs, vision_batch_size)
    }

    pub fn prefill(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Tensor> {
        // A prefill always starts a new request. Clear every attention,
        // short-convolution, and mask cache even if subsequent validation
        // rejects the supplied multimodal inputs.
        self.text.clear_cache();
        let input_id_values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let image_token_count = input_id_values
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&token_id| token_id == self.pairing.image_token_id)
            .count();
        let input_embeds = self.text.embed_tokens(input_ids)?;
        let input_embeds = if image_token_count == 0 {
            if !image_spans.is_empty() || encoded_images.is_some() {
                candle::bail!("LFM2-VL image spans/features were supplied without image tokens")
            }
            input_embeds
        } else {
            let encoded_images = encoded_images.ok_or_else(|| {
                candle::Error::Msg("LFM2-VL image tokens require encoded image features".into())
            })?;
            if image_spans.is_empty() {
                candle::bail!("LFM2-VL image tokens require explicit image spans")
            }
            merge_projected_embeddings(
                input_ids,
                &input_embeds,
                self.pairing.image_token_id,
                image_spans,
                encoded_images,
            )?
        };
        self.text.forward_embeds(&input_embeds, 0)
    }

    pub fn decode(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        token_ids.dims2()?;
        self.text.forward(token_ids, index_pos)
    }

    pub fn clear_cache(&mut self) {
        self.text.clear_cache();
    }

    pub fn pairing_report(&self) -> &PairingReport {
        &self.pairing
    }

    pub fn mmproj(&self) -> &Mmproj {
        &self.mmproj
    }

    pub fn vision_device(&self) -> &Device {
        self.mmproj.device()
    }

    pub fn text_device(&self) -> &Device {
        self.text.device()
    }
}

pub fn inspect_safetensors(
    weights_path: impl AsRef<Path>,
    manifest: &MmprojManifest,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let weights_path = weights_path.as_ref();
    let weights_bytes = read_weight_bytes(weights_path, "split MMProj safetensors", manifest)?;
    verify_bytes_sha256(
        &weights_bytes,
        &manifest.mmproj_safetensors_sha256,
        "split MMProj safetensors",
    )?;
    inspect_safetensors_bytes(&weights_bytes, manifest, dtype, device)
}

fn inspect_safetensors_bytes(
    weights_bytes: &[u8],
    manifest: &MmprojManifest,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let actual = safetensors_inventory(weights_bytes)?;
    let actual_names: BTreeSet<_> = actual.keys().cloned().collect();
    let manifest_names: BTreeSet<_> = manifest.tensor_inventory.keys().cloned().collect();
    let missing_tensors = manifest_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&manifest_names).cloned().collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in manifest_names.intersection(&actual_names) {
        let expected = &manifest.tensor_inventory[name];
        let found = &actual[name];
        if expected != found {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected {} {:?} ({} bytes), found {} {:?} ({} bytes)",
                expected.dtype,
                expected.shape,
                expected.nbytes,
                found.dtype,
                found.shape,
                found.nbytes
            ));
        }
    }
    Ok(MmprojLoadReport {
        loaded_tensors: actual_names.into_iter().collect(),
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: VISION_ROOT.to_string(),
        resolved_projector_root: PROJECTOR_ROOT.to_string(),
        target_dtype: format!("{dtype:?}"),
        target_device: format!("{device:?}"),
    })
}

fn safetensors_inventory(weights_bytes: &[u8]) -> Result<BTreeMap<String, MmprojTensorInfo>> {
    let prefix = weights_bytes.get(..8).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors is shorter than its header prefix".into())
    })?;
    let header_len = u64::from_le_bytes(prefix.try_into().map_err(candle::Error::wrap)?);
    let header_len = usize::try_from(header_len).map_err(candle::Error::wrap)?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        candle::bail!(
            "split MMProj safetensors header length {header_len} is outside the supported range"
        )
    }
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors header length overflows".into())
    })?;
    let header_bytes = weights_bytes.get(8..header_end).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors header exceeds the file length".into())
    })?;
    let header_value: serde_json::Value = serde_json::from_slice(header_bytes).map_err(|err| {
        candle::Error::Msg(format!("invalid split MMProj safetensors header: {err}"))
    })?;
    let mut header = match header_value {
        serde_json::Value::Object(header) => header,
        _ => {
            candle::bail!("split MMProj safetensors header must be a JSON object")
        }
    };
    if let Some(metadata) = header.remove("__metadata__") {
        let metadata = metadata.as_object().ok_or_else(|| {
            candle::Error::Msg("split MMProj safetensors metadata must be an object".into())
        })?;
        if metadata
            .iter()
            .any(|(key, value)| key.is_empty() || !value.is_string())
        {
            candle::bail!("split MMProj safetensors metadata must contain only strings")
        }
    }
    let tensor_count = header.len();
    if tensor_count == 0 || tensor_count > MAX_MMPROJ_TENSORS {
        candle::bail!(
            "split MMProj safetensors tensor count {tensor_count} is outside the supported range"
        )
    }

    let mut actual = BTreeMap::new();
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(tensor_count).map_err(|_| {
        candle::Error::Msg("split MMProj safetensors range allocation failed".into())
    })?;
    let data_size = weights_bytes.len() - header_end;
    for (name, value) in header {
        let info = value.as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "split MMProj safetensors tensor {name:?} metadata must be an object"
            ))
        })?;
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} lacks a string dtype"
                ))
            })?
            .to_string();
        let element_size = dense_dtype_size(&dtype).ok_or_else(|| {
            candle::Error::Msg(format!(
                "split MMProj safetensors tensor {name:?} has unsupported dense dtype {dtype:?}"
            ))
        })?;
        let raw_shape = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .filter(|shape| !shape.is_empty())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} has an invalid shape"
                ))
            })?;
        let mut shape = Vec::new();
        shape.try_reserve_exact(raw_shape.len()).map_err(|_| {
            candle::Error::Msg("split MMProj safetensors shape allocation failed".into())
        })?;
        for dimension in raw_shape {
            let dimension = dimension
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "split MMProj safetensors tensor {name:?} has an invalid shape"
                    ))
                })?;
            shape.push(dimension);
        }
        let raw_offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} has invalid data offsets"
                ))
            })?;
        let offset = |index: usize| -> Result<usize> {
            raw_offsets[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "split MMProj safetensors tensor {name:?} has invalid data offsets"
                    ))
                })
        };
        let start = offset(0)?;
        let end = offset(1)?;
        if start > end || end > data_size {
            candle::bail!(
                "split MMProj safetensors tensor {name:?} has out-of-bounds data offsets [{start}, {end}]"
            )
        }
        let expected_nbytes = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .and_then(|count| count.checked_mul(element_size))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} byte size overflows"
                ))
            })?;
        let nbytes = end - start;
        if nbytes != expected_nbytes {
            candle::bail!(
                "split MMProj safetensors tensor {name:?} stores {nbytes} bytes, expected {expected_nbytes}"
            )
        }
        ranges.push((start, end, name.clone()));
        actual.insert(
            name,
            MmprojTensorInfo {
                dtype,
                shape,
                nbytes,
            },
        );
    }

    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0usize;
    for (start, end, name) in ranges {
        if start != previous_end {
            let relation = if start < previous_end {
                "overlaps another tensor"
            } else {
                "leaves a payload gap"
            };
            candle::bail!("split MMProj safetensors tensor {name:?} {relation}")
        }
        previous_end = end;
    }
    if previous_end != data_size {
        candle::bail!(
            "split MMProj safetensors has {} unclaimed payload bytes",
            data_size - previous_end
        )
    }
    Ok(actual)
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    validate_lower_hex(label, value, &[64])
}

fn validate_lower_hex(label: &str, value: &str, allowed_lengths: &[usize]) -> Result<()> {
    if !allowed_lengths.contains(&value.len())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        candle::bail!(
            "split MMProj {label} must be lowercase hexadecimal with length in {allowed_lengths:?}"
        )
    }
    Ok(())
}

fn verify_bytes_sha256(bytes: &[u8], expected: &str, label: &str) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        candle::bail!("{label} SHA-256 mismatch: expected {expected}, found {actual}")
    }
    Ok(())
}

fn read_bounded_text(path: &Path, label: &str) -> Result<String> {
    let bytes = read_bounded_bytes(path, label)?;
    String::from_utf8(bytes)
        .map_err(|err| candle::Error::Msg(format!("{label} is not UTF-8: {err}")))
}

fn read_bounded_bytes(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_file_bytes(path, label, MAX_MANIFEST_BYTES)
}

fn read_weight_bytes(path: &Path, label: &str, manifest: &MmprojManifest) -> Result<Vec<u8>> {
    let payload_bytes = manifest
        .tensor_inventory
        .values()
        .try_fold(0u64, |total, info| {
            let nbytes = u64::try_from(info.nbytes).ok()?;
            total.checked_add(nbytes)
        })
        .ok_or_else(|| candle::Error::Msg("split MMProj payload size overflows".into()))?;
    let maximum_file_bytes = payload_bytes
        .checked_add(MAX_SAFETENSORS_HEADER_BYTES as u64)
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| candle::Error::Msg("split MMProj file size limit overflows".into()))?;
    read_file_bytes(path, label, maximum_file_bytes)
}

fn read_file_bytes(path: &Path, label: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|err| {
        candle::Error::Msg(format!("cannot open {label} at {}: {err}", path.display()))
    })?;
    let size = file
        .metadata()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "cannot inspect {label} at {}: {err}",
                path.display()
            ))
        })?
        .len();
    if size == 0 {
        candle::bail!("{label} at {} is empty", path.display())
    }
    if size > max_bytes {
        candle::bail!("{label} size {size} is outside the supported range")
    }
    let size = usize::try_from(size).map_err(|_| {
        candle::Error::Msg(format!(
            "{label} at {} is too large for this platform",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| {
        candle::Error::Msg(format!(
            "cannot allocate {size} bytes for {label} at {}",
            path.display()
        ))
    })?;
    bytes.resize(size, 0);
    file.read_exact(&mut bytes).map_err(|err| {
        candle::Error::Msg(format!("cannot read {label} at {}: {err}", path.display()))
    })?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing).map_err(|err| {
        candle::Error::Msg(format!(
            "cannot finish reading {label} at {}: {err}",
            path.display()
        ))
    })? != 0
    {
        candle::bail!("{label} at {} changed while it was read", path.display())
    }
    Ok(bytes)
}

fn processor_pair_fields(processor: &serde_json::Value) -> Result<(usize, usize)> {
    let values = processor
        .get("image_processor")
        .unwrap_or(processor)
        .as_object()
        .ok_or_else(|| {
            candle::Error::Msg("split MMProj processor config must be a JSON object".into())
        })?;
    let positive_usize = |name: &str, aliases: &[&str]| -> Result<usize> {
        let value = aliases
            .iter()
            .find_map(|alias| values.get(*alias))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj processor config lacks a positive {name}"
                ))
            })?;
        Ok(value)
    };
    Ok((
        positive_usize("encoder patch size", &["encoder_patch_size", "patch_size"])?,
        positive_usize("downsample factor", &["downsample_factor"])?,
    ))
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        "F64" => Some(8),
        _ => None,
    }
}

fn expected_tensor_shapes(config: &Lfm2VlConfig) -> Result<BTreeMap<String, Vec<usize>>> {
    let vision = &config.vision_config;
    if vision.num_hidden_layers > MAX_VISION_LAYERS {
        candle::bail!("split MMProj vision layer count exceeds {MAX_VISION_LAYERS}")
    }
    let patch_dimension = vision.patch_dimension_for_vl()?;
    let mut shapes = BTreeMap::new();
    let mut insert = |name: String, shape: Vec<usize>| {
        shapes.insert(name, shape);
    };
    insert(
        format!("{VISION_ROOT}.embeddings.patch_embedding.weight"),
        vec![vision.hidden_size, patch_dimension],
    );
    insert(
        format!("{VISION_ROOT}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
    );
    insert(
        format!("{VISION_ROOT}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
    );
    for layer in 0..vision.num_hidden_layers {
        let root = format!("{VISION_ROOT}.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            insert(format!("{root}.{norm}.weight"), vec![vision.hidden_size]);
            insert(format!("{root}.{norm}.bias"), vec![vision.hidden_size]);
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            insert(
                format!("{root}.self_attn.{projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
            );
            insert(
                format!("{root}.self_attn.{projection}.bias"),
                vec![vision.hidden_size],
            );
        }
        insert(
            format!("{root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
        );
        insert(
            format!("{root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
        );
        insert(
            format!("{root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
        );
        insert(format!("{root}.mlp.fc2.bias"), vec![vision.hidden_size]);
    }
    insert(
        format!("{VISION_ROOT}.post_layernorm.weight"),
        vec![vision.hidden_size],
    );
    insert(
        format!("{VISION_ROOT}.post_layernorm.bias"),
        vec![vision.hidden_size],
    );

    let projector_input = config.projector_input_size()?;
    if config.projector_use_layernorm {
        insert(
            format!("{PROJECTOR_ROOT}.layer_norm.weight"),
            vec![projector_input],
        );
        insert(
            format!("{PROJECTOR_ROOT}.layer_norm.bias"),
            vec![projector_input],
        );
    }
    insert(
        format!("{PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
    );
    insert(
        format!("{PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_config.hidden_size, config.projector_hidden_size],
    );
    if config.projector_bias {
        insert(
            format!("{PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
        );
        insert(
            format!("{PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_config.hidden_size],
        );
    }
    Ok(shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{lfm2, lfm2_vl::Lfm2VlModel};
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    use candle::{IndexOp, Tensor};
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::path::PathBuf;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

    fn bundle_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_mmproj_tiny")
    }

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
            crops: vec![super::super::CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: super::super::CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![super::super::ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn image_spans(input_ids: &Tensor, image_token_id: u32) -> Result<Vec<ImageTokenSpan>> {
        let values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut spans = Vec::new();
        for (batch_index, row) in values.iter().enumerate() {
            let mut start = None;
            for (position, &token_id) in row.iter().enumerate() {
                if token_id == image_token_id && start.is_none() {
                    start = Some(position);
                } else if token_id != image_token_id {
                    if let Some(start) = start.take() {
                        spans.push(ImageTokenSpan::new(batch_index, start, position));
                    }
                }
            }
            if let Some(start) = start {
                spans.push(ImageTokenSpan::new(batch_index, start, row.len()));
            }
        }
        Ok(spans)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("{label}: element count mismatch")
        }
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0f32, f32::max);
        eprintln!("lfm2_vl hybrid {label}: max_abs={max_abs:.9e}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        Ok(())
    }

    fn synthetic_text_metadata(config: &Lfm2VlConfig) -> quantized_lfm2::Lfm2GgufMetadata {
        let text = config.text_model_config().expect("fixed tiny text config");
        quantized_lfm2::Lfm2GgufMetadata {
            architecture: "lfm2".to_string(),
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

    fn tiny_text_gguf(tensors: &HashMap<String, Tensor>, config: &Lfm2VlConfig) -> Result<Vec<u8>> {
        let text = config.text_model_config()?;
        let root = "weights.model.language_model";
        let mut names = vec![
            (
                "token_embd.weight".to_string(),
                format!("{root}.embed_tokens.weight"),
            ),
            (
                "output_norm.weight".to_string(),
                format!("{root}.embedding_norm.weight"),
            ),
        ];
        for (layer, layer_type) in text.layer_types.iter().enumerate() {
            let native = format!("{root}.layers.{layer}");
            let gguf = format!("blk.{layer}");
            names.extend([
                (
                    format!("{gguf}.attn_norm.weight"),
                    format!("{native}.operator_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_norm.weight"),
                    format!("{native}.ffn_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_gate.weight"),
                    format!("{native}.feed_forward.w1.weight"),
                ),
                (
                    format!("{gguf}.ffn_down.weight"),
                    format!("{native}.feed_forward.w2.weight"),
                ),
                (
                    format!("{gguf}.ffn_up.weight"),
                    format!("{native}.feed_forward.w3.weight"),
                ),
            ]);
            match layer_type {
                lfm2::LayerType::Conv => names.extend([
                    (
                        format!("{gguf}.shortconv.in_proj.weight"),
                        format!("{native}.conv.in_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.out_proj.weight"),
                        format!("{native}.conv.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.conv.weight"),
                        format!("{native}.conv.conv.weight"),
                    ),
                ]),
                lfm2::LayerType::FullAttention => names.extend([
                    (
                        format!("{gguf}.attn_q.weight"),
                        format!("{native}.self_attn.q_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k.weight"),
                        format!("{native}.self_attn.k_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_v.weight"),
                        format!("{native}.self_attn.v_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_output.weight"),
                        format!("{native}.self_attn.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_q_norm.weight"),
                        format!("{native}.self_attn.q_layernorm.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k_norm.weight"),
                        format!("{native}.self_attn.k_layernorm.weight"),
                    ),
                ]),
            }
        }

        let mut qtensors = Vec::new();
        let mut q8_count = 0usize;
        for (gguf_name, native_name) in names {
            let tensor = fixture_tensor(tensors, &native_name)?.contiguous()?;
            let dtype = if tensor.rank() == 2
                && tensor.dim(1)?.is_multiple_of(GgmlDType::Q8_0.block_size())
            {
                q8_count += 1;
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            qtensors.push((gguf_name, QTensor::quantize(&tensor, dtype)?));
        }
        if q8_count == 0 {
            candle::bail!("tiny GGUF fixture contains no Q8_0 matrices")
        }
        let metadata = vec![
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("lfm2".to_string()),
            ),
            (
                "lfm2.attention.head_count".to_string(),
                gguf_file::Value::U32(text.num_attention_heads as u32),
            ),
            (
                "lfm2.attention.head_count_kv".to_string(),
                gguf_file::Value::Array(
                    text.layer_types
                        .iter()
                        .map(|kind| match kind {
                            lfm2::LayerType::FullAttention => {
                                gguf_file::Value::U32(text.num_key_value_heads as u32)
                            }
                            lfm2::LayerType::Conv => gguf_file::Value::U32(0),
                        })
                        .collect(),
                ),
            ),
            (
                "lfm2.embedding_length".to_string(),
                gguf_file::Value::U32(text.hidden_size as u32),
            ),
            (
                "lfm2.context_length".to_string(),
                gguf_file::Value::U32(text.max_position_embeddings as u32),
            ),
            (
                "lfm2.block_count".to_string(),
                gguf_file::Value::U32(text.num_hidden_layers as u32),
            ),
            (
                "lfm2.attention.layer_norm_rms_epsilon".to_string(),
                gguf_file::Value::F32(text.norm_eps as f32),
            ),
            (
                "lfm2.rope.freq_base".to_string(),
                gguf_file::Value::F32(text.rope_theta),
            ),
            (
                "lfm2.shortconv.l_cache".to_string(),
                gguf_file::Value::U32(text.conv_l_cache as u32),
            ),
        ];
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

    #[test]
    fn split_manifest_and_tensor_inventory_are_exact() -> Result<()> {
        let device = Device::Cpu;
        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &device)?;
        assert!(mmproj.report.is_clean());
        assert_eq!(mmproj.report.loaded_tensors.len(), 43);
        assert_eq!(mmproj.metadata.manifest.vision_layer_count, 2);
        assert_eq!(mmproj.metadata.manifest.expected_text_layer_count, 2);
        assert_eq!(mmproj.metadata.manifest.tensor_count, 43);
        assert_eq!(
            mmproj.metadata.source_revision.as_deref(),
            Some("fc6221ca597f3315e4f82fc2df606783267b34ba")
        );
        assert!(mmproj
            .report
            .loaded_tensors
            .iter()
            .all(|name| name.starts_with(VISION_ROOT) || name.starts_with(PROJECTOR_ROOT)));

        let mut bad = mmproj.metadata.manifest.clone();
        let linear_2 = format!("{PROJECTOR_ROOT}.linear_2.weight");
        bad.tensor_inventory
            .get_mut(&linear_2)
            .ok_or_else(|| candle::Error::Msg("missing fixed projector weight".into()))?
            .shape[0] += 1;
        let report = inspect_safetensors(
            bundle_dir().join("mmproj.safetensors"),
            &bad,
            DType::F32,
            &device,
        )?;
        assert!(report
            .shape_or_dtype_mismatches
            .iter()
            .any(|mismatch| mismatch.contains(&linear_2)));
        let error = bad.validate().unwrap_err().to_string();
        assert!(error.contains("linear_2.weight") && error.contains("expected"));

        let mut bad_names = mmproj.metadata.manifest.clone();
        let moved = bad_names
            .tensor_inventory
            .remove(&linear_2)
            .ok_or_else(|| candle::Error::Msg("missing fixed projector weight".into()))?;
        let fake = format!("{PROJECTOR_ROOT}.unexpected.weight");
        bad_names.tensor_inventory.insert(fake.clone(), moved);
        let report = inspect_safetensors(
            bundle_dir().join("mmproj.safetensors"),
            &bad_names,
            DType::F32,
            &device,
        )?;
        assert_eq!(report.missing_tensors, vec![fake]);
        assert_eq!(report.unexpected_tensors, vec![linear_2]);
        Ok(())
    }

    #[test]
    fn manifest_requires_pinned_source_provenance() -> Result<()> {
        let manifest = MmprojManifest::from_json(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)?;
        let mut missing_model = manifest.clone();
        missing_model.source_model.clear();
        assert!(missing_model
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source model"));

        let mut mutable_revision = manifest.clone();
        mutable_revision.source_revision = "main".to_string();
        assert!(mutable_revision
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source revision"));

        let mut json: serde_json::Value = serde_json::from_str(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)
        .map_err(candle::Error::wrap)?;
        json.as_object_mut()
            .ok_or_else(|| candle::Error::Msg("fixed manifest is not an object".into()))?
            .remove("source_revision");
        let json = serde_json::to_string(&json).map_err(candle::Error::wrap)?;
        let error = MmprojManifest::from_json(&json).unwrap_err().to_string();
        assert!(error.contains("source_revision"));
        Ok(())
    }

    #[test]
    fn safetensors_preflight_bounds_header_and_tensor_count() -> Result<()> {
        let oversized_header = (MAX_SAFETENSORS_HEADER_BYTES as u64 + 1)
            .to_le_bytes()
            .to_vec();
        let error = safetensors_inventory(&oversized_header)
            .unwrap_err()
            .to_string();
        assert!(error.contains("header length") && error.contains("outside"));

        let mut header = serde_json::Map::new();
        for index in 0..=MAX_MMPROJ_TENSORS {
            header.insert(
                format!("tensor.{index}"),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }),
            );
        }
        let raw_header =
            serde_json::to_vec(&serde_json::Value::Object(header)).map_err(candle::Error::wrap)?;
        assert!(raw_header.len() < MAX_SAFETENSORS_HEADER_BYTES);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(raw_header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&raw_header);
        let error = safetensors_inventory(&bytes).unwrap_err().to_string();
        assert!(error.contains("tensor count") && error.contains("outside"));
        Ok(())
    }

    #[test]
    fn pairing_rejects_every_cross_artifact_mismatch() -> Result<()> {
        let manifest = MmprojManifest::from_json(&read_bounded_text(
            &bundle_dir().join("mmproj.json"),
            "test manifest",
        )?)?;
        let text = synthetic_text_metadata(&manifest.model_config);
        let report = manifest.validate_pair(&text, 2, 2, 3)?;
        assert!(report.only_projected_features_cross_devices);
        assert_eq!(report.text_output_resolution, "tied token embeddings");

        let mut wrong = text.clone();
        wrong.architecture = "llama".to_string();
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        let mut wrong = text.clone();
        wrong.embedding_length += 1;
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        let mut wrong = text.clone();
        wrong.block_count += 1;
        assert!(manifest.validate_pair(&wrong, 2, 2, 3).is_err());
        assert!(manifest.validate_pair(&text, 4, 2, 3).is_err());
        assert!(manifest.validate_pair(&text, 2, 1, 3).is_err());
        assert!(manifest.validate_pair(&text, 2, 2, 4).is_err());
        Ok(())
    }

    #[test]
    fn split_features_and_hybrid_text_match_unified_native_model() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let unified_vb = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?;
        let native = Lfm2VlModel::new(&config, unified_vb.pp("weights"))?;
        let batch = fixture_batch(&tensors)?;
        let native_encoded = native.encode_images(&batch, 1)?;

        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &device)?;
        let split_encoded = mmproj.encode_images(&batch, 1)?;
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(mmproj.device()));
        assert_close(
            &split_encoded.embeddings,
            &native_encoded.embeddings,
            1e-6,
            "split image features",
        )?;
        assert_eq!(
            split_encoded.per_crop_ranges,
            native_encoded.per_crop_ranges
        );
        assert_eq!(
            split_encoded.per_image_ranges,
            native_encoded.per_image_ranges
        );

        let text_config = config.text_model_config()?;
        let gguf_bytes = tiny_text_gguf(&tensors, &config)?;
        let gguf_hash = format!("{:x}", Sha256::digest(&gguf_bytes));
        eprintln!("lfm2_vl hybrid deterministic text GGUF SHA-256: {gguf_hash}");
        assert_eq!(
            gguf_hash,
            "8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1"
        );
        let mut malformed_reader = Cursor::new(gguf_bytes.clone());
        let mut malformed_gguf = gguf_file::Content::read(&mut malformed_reader)?;
        malformed_gguf.metadata.insert(
            "lfm2.rope.freq_base".to_string(),
            gguf_file::Value::String("not-a-frequency".to_string()),
        );
        assert!(quantized_lfm2::inspect_gguf_metadata(&malformed_gguf)
            .unwrap_err()
            .to_string()
            .contains("f32"));
        let mut gguf_reader = Cursor::new(gguf_bytes);
        let gguf = gguf_file::Content::read(&mut gguf_reader)?;
        let quantized_text =
            quantized_lfm2::ModelWeights::from_gguf(gguf, &mut gguf_reader, &device)?;
        let mut hybrid = QuantizedLfm2VlModel::new(quantized_text, mmproj, 2, 2, 3)?;
        assert!(hybrid.vision_device().same_device(&device));
        assert!(hybrid.text_device().same_device(&device));
        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let spans = image_spans(input_ids, 3)?;
        let mut native_cache = lfm2::Cache::new(true, DType::F32, &text_config, &device)?;
        let native_prefill =
            native.prefill(input_ids, &spans, Some(&native_encoded), &mut native_cache)?;
        let hybrid_prefill = hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        let native_last = native_prefill.i((.., input_ids.dim(1)? - 1, ..))?;
        assert_close(&hybrid_prefill, &native_last, 1e-4, "hybrid prefill logits")?;

        let decode_ids = fixture_tensor(&tensors, "input.decode_token_ids")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let native_logits = native.decode(&token, 5 + step, &mut native_cache)?;
            let hybrid_logits = hybrid.decode(&token, 5 + step)?;
            assert_close(
                &hybrid_logits,
                &native_logits,
                1e-4,
                &format!("hybrid cached decode step {step}"),
            )?;
        }

        let reset = hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        assert_close(&reset, &hybrid_prefill, 1e-6, "hybrid cache reset")?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn split_vision_cuda_text_cpu_transfers_only_projected_features() -> Result<()> {
        let text_device = Device::Cpu;
        let vision_device = Device::new_cuda(0)?;
        assert!(!vision_device.same_device(&text_device));

        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &text_device)?;
        let unified_vb =
            VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &text_device)?;
        let native = Lfm2VlModel::new(&config, unified_vb.pp("weights"))?;
        let batch = fixture_batch(&tensors)?;
        assert!(batch.pixel_values.device().same_device(&text_device));
        let native_encoded = native.encode_images(&batch, 1)?;

        let mmproj = Mmproj::load(bundle_dir(), DType::F32, &vision_device)?;
        let split_encoded = mmproj.encode_images(&batch, 1)?;
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(&vision_device));
        assert!(batch.pixel_values.device().same_device(&text_device));
        assert_close(
            &split_encoded.embeddings.to_device(&text_device)?,
            &native_encoded.embeddings,
            1e-5,
            "CUDA split image features",
        )?;

        let gguf_bytes = tiny_text_gguf(&tensors, &config)?;
        let mut gguf_reader = Cursor::new(gguf_bytes);
        let gguf = gguf_file::Content::read(&mut gguf_reader)?;
        let quantized_text =
            quantized_lfm2::ModelWeights::from_gguf(gguf, &mut gguf_reader, &text_device)?;
        let mut hybrid = QuantizedLfm2VlModel::new(quantized_text, mmproj, 2, 2, 3)?;
        assert!(hybrid.vision_device().same_device(&vision_device));
        assert!(hybrid.text_device().same_device(&text_device));
        assert!(
            hybrid
                .pairing_report()
                .only_projected_features_cross_devices
        );

        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let spans = image_spans(input_ids, 3)?;
        let text_config = config.text_model_config()?;
        let mut native_cache = lfm2::Cache::new(true, DType::F32, &text_config, &text_device)?;
        let native_prefill =
            native.prefill(input_ids, &spans, Some(&native_encoded), &mut native_cache)?;
        let hybrid_prefill = hybrid.prefill(input_ids, &spans, Some(&split_encoded))?;
        assert!(hybrid_prefill.device().same_device(&text_device));
        assert!(split_encoded
            .embeddings
            .device()
            .same_device(&vision_device));
        let native_last = native_prefill.i((.., input_ids.dim(1)? - 1, ..))?;
        assert_close(
            &hybrid_prefill,
            &native_last,
            1e-4,
            "CUDA-vision/CPU-text hybrid prefill logits",
        )?;
        Ok(())
    }
}
