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
        validate_pairing_facts(
            "split MMProj",
            text,
            self.expected_text_hidden_size,
            Some(self.expected_text_layer_count),
            self.vision_layer_count,
            self.patch_size,
            self.downsample_factor,
            self.image_token_id,
            processor_patch_size,
            processor_downsample_factor,
            tokenizer_image_token_id,
        )
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
    pub use_image_special_tokens: bool,
    pub expected_text_layer_count: Option<usize>,
    /// Kept neutral at this crate boundary because `candle-vlm` depends on
    /// `candle-transformers`; `candle-vlm` provides the typed conversion.
    pub processor: serde_json::Value,
    pub source_model: Option<String>,
    pub source_revision: Option<String>,
    pub manifest: Option<MmprojManifest>,
    pub gguf: Option<GgufMmprojMetadata>,
}

impl MmprojMetadata {
    pub fn split_manifest(&self) -> Option<&MmprojManifest> {
        self.manifest.as_ref()
    }

    pub fn gguf_metadata(&self) -> Option<&GgufMmprojMetadata> {
        self.gguf.as_ref()
    }

    pub(super) fn validate_pair(
        &self,
        text: &quantized_lfm2::Lfm2GgufMetadata,
        processor_patch_size: usize,
        processor_downsample_factor: usize,
        tokenizer_image_token_id: u32,
    ) -> Result<PairingReport> {
        let (label, vision_layer_count) = match (&self.manifest, &self.gguf) {
            (Some(manifest), None) => ("split MMProj", manifest.vision_layer_count),
            (None, Some(metadata)) => ("GGUF MMProj", metadata.vision_layer_count),
            _ => candle::bail!("MMProj metadata must contain exactly one source description"),
        };
        validate_pairing_facts(
            label,
            text,
            self.text_hidden_size,
            self.expected_text_layer_count,
            vision_layer_count,
            self.patch_size,
            self.downsample_factor,
            self.image_token_id,
            processor_patch_size,
            processor_downsample_factor,
            tokenizer_image_token_id,
        )
    }
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

    pub(super) fn require_clean(&self) -> Result<()> {
        if !self.is_clean() {
            candle::bail!(
                "MMProj tensor validation failed; missing={:?}, unexpected={:?}, mismatches={:?}",
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

#[allow(clippy::too_many_arguments)]
fn validate_pairing_facts(
    artifact_label: &str,
    text: &quantized_lfm2::Lfm2GgufMetadata,
    text_hidden_size: usize,
    expected_text_layer_count: Option<usize>,
    vision_layer_count: usize,
    patch_size: usize,
    downsample_factor: usize,
    image_token_id: u32,
    processor_patch_size: usize,
    processor_downsample_factor: usize,
    tokenizer_image_token_id: u32,
) -> Result<PairingReport> {
    if text.architecture != "lfm2" {
        candle::bail!(
            "{artifact_label} requires quantized text architecture \"lfm2\", got {:?}",
            text.architecture
        )
    }
    if text.embedding_length != text_hidden_size {
        candle::bail!(
            "{artifact_label} output width {text_hidden_size} does not match quantized text hidden size {}",
            text.embedding_length
        )
    }
    if let Some(expected) = expected_text_layer_count {
        if text.block_count != expected {
            candle::bail!(
                "{artifact_label} expects {expected} text layers, but GGUF declares {}",
                text.block_count
            )
        }
    }
    if processor_patch_size != patch_size {
        candle::bail!(
            "processor patch size {processor_patch_size} does not match {artifact_label} {patch_size}"
        )
    }
    if processor_downsample_factor != downsample_factor {
        candle::bail!(
            "processor downsample factor {processor_downsample_factor} does not match {artifact_label} {downsample_factor}"
        )
    }
    if tokenizer_image_token_id != image_token_id {
        candle::bail!(
            "tokenizer image token ID {tokenizer_image_token_id} does not match {artifact_label} {image_token_id}"
        )
    }
    Ok(PairingReport {
        text_architecture: text.architecture.clone(),
        text_hidden_size: text.embedding_length,
        text_layer_count: text.block_count,
        vision_layer_count,
        patch_size,
        downsample_factor,
        image_token_id,
        text_output_resolution: if text.tied_output {
            "tied token embeddings".to_string()
        } else {
            "explicit GGUF output tensor".to_string()
        },
        only_projected_features_cross_devices: true,
    })
}
