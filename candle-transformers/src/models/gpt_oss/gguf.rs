//! Hash-pinned GPT-OSS GGUF admission and tensor-inventory normalization.
//!
//! This is the hash-pinned GGUF admission and CPU assembly boundary.  It
//! understands the GGUF directory and the MXFP4 wire type used by the selected
//! GPT-OSS artifact, retains tensor ownership and byte ranges, and can assemble
//! one bounded Candle model load without retaining raw tensor payloads.

use super::mxfp4::{Mxfp4ExpertOperation, PackedMxfp4, BYTES_PER_BLOCK, VALUES_PER_BLOCK};
use super::{DenseLinear, GptOssConfig, GptOssLayerWeights, GptOssWeights};
use crate::models::gpt_oss::runtime::{
    GptOssCancellationToken, GptOssLoadRegistry, GptOssLoadedHandle,
};
use candle::{bail, DType, Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

/// SHA-256 of the owner-selected GPT-OSS GGUF artifact.
///
/// The artifact itself is an external runtime input and is intentionally not
/// stored in this repository.  [`GptOssGgufArtifact::open`] admits only this
/// exact byte identity.
pub const SELECTED_GPT_OSS_GGUF_SHA256: &str =
    "aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778";

const GGUF_MAX_FILE_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const GGUF_MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const GGUF_MAX_TENSORS: u64 = 4096;
const GGUF_MAX_METADATA: u64 = 2048;
const GGUF_MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1_000_000;
const GGUF_MAX_VALUE_DEPTH: usize = 64;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_RAW_TENSOR_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const GGUF_READ_CHUNK_BYTES: usize = 1024 * 1024;

fn open_retained_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    options.share_mode(0x0000_0001);
    options.open(path).map_err(|error| {
        candle::Error::Msg(format!(
            "failed to open GPT-OSS GGUF {:?} for retained read-only admission: {error}",
            path
        ))
    })
}

/// GGUF dtypes admitted by the GPT-OSS loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GptOssGgufDType {
    F32,
    F16,
    BF16,
    Q8_0,
    Mxfp4,
}

impl GptOssGgufDType {
    fn from_id(id: u32) -> Result<Self> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            30 => Ok(Self::BF16),
            39 => Ok(Self::Mxfp4),
            _ => bail!("unsupported GPT-OSS GGUF tensor dtype {id}"),
        }
    }

    fn byte_len(self, element_count: u64) -> Result<u64> {
        match self {
            Self::F32 => element_count
                .checked_mul(4)
                .ok_or_else(|| candle::Error::Msg("GGUF F32 byte length overflowed".into())),
            Self::F16 | Self::BF16 => element_count
                .checked_mul(2)
                .ok_or_else(|| candle::Error::Msg("GGUF float16 byte length overflowed".into())),
            Self::Q8_0 => block_byte_len(element_count, 32, 34, "Q8_0"),
            // GGML MXFP4 stores one E8M0 scale byte followed by 16 packed
            // bytes for every 32 logical FP4 values.
            Self::Mxfp4 => block_byte_len(element_count, 32, 17, "MXFP4"),
        }
    }
}

fn block_byte_len(
    element_count: u64,
    block_size: u64,
    bytes_per_block: u64,
    dtype: &str,
) -> Result<u64> {
    if !element_count.is_multiple_of(block_size) {
        bail!(
            "GPT-OSS GGUF {dtype} tensor has {element_count} values, not divisible by block size {block_size}"
        );
    }
    element_count
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(bytes_per_block))
        .ok_or_else(|| candle::Error::Msg(format!("GGUF {dtype} byte length overflowed")))
}

fn checked_usize_product<I>(values: I, label: &str) -> Result<usize>
where
    I: IntoIterator<Item = usize>,
{
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
        .ok_or_else(|| candle::Error::Msg(format!("{label} overflowed")))
}

/// The Candle-side owner of a normalized GGUF tensor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GptOssTensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
    AttentionNorm(usize),
    QkvWeight(usize),
    QkvBias(usize),
    QueryWeight(usize),
    QueryBias(usize),
    KeyWeight(usize),
    KeyBias(usize),
    ValueWeight(usize),
    ValueBias(usize),
    AttentionOutputWeight(usize),
    AttentionOutputBias(usize),
    AttentionSinks(usize),
    MoeNorm(usize),
    RouterWeight(usize),
    RouterBias(usize),
    ExpertGateUpWeight(usize),
    ExpertGateUpBias(usize),
    ExpertGateWeight(usize),
    ExpertGateBias(usize),
    ExpertUpWeight(usize),
    ExpertUpBias(usize),
    ExpertDownWeight(usize),
    ExpertDownBias(usize),
}

/// A normalized GGUF tensor descriptor.  `shape` is logical Candle order,
/// matching the existing Candle GGUF loaders; `offset` is relative to the
/// aligned GGUF tensor-data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptOssGgufTensor {
    pub name: String,
    pub role: GptOssTensorRole,
    pub shape: Vec<usize>,
    pub dtype: GptOssGgufDType,
    pub offset: u64,
    pub byte_len: u64,
}

/// An admitted, normalized GPT-OSS GGUF artifact.
///
/// Loading this value validates the complete tensor inventory and all tensor
/// ranges, but keeps the large payloads on disk.  Use [`Self::read_tensor`] to
/// read one already-owned raw tensor payload for a later executor.
#[derive(Debug, Clone)]
pub struct GptOssGgufArtifact {
    path: PathBuf,
    sha256: String,
    config: GptOssConfig,
    context_length: usize,
    tensor_data_offset: u64,
    file_size: u64,
    tensors: Vec<GptOssGgufTensor>,
    retained_file: Option<Arc<Mutex<File>>>,
}

impl PartialEq for GptOssGgufArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.sha256 == other.sha256
            && self.config == other.config
            && self.context_length == other.context_length
            && self.tensor_data_offset == other.tensor_data_offset
            && self.file_size == other.file_size
            && self.tensors == other.tensors
    }
}

impl GptOssGgufArtifact {
    /// Admit the exact owner-selected artifact from a local path.
    ///
    /// This function never downloads or searches for a model.  The file is
    /// hashed before any GGUF metadata is trusted.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_expected_sha(path, SELECTED_GPT_OSS_GGUF_SHA256, None)
    }

    fn open_with_expected_sha(
        path: impl AsRef<Path>,
        expected_sha256: &str,
        cancellation: Option<&GptOssCancellationToken>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = open_retained_file(&path)?;
        let metadata = file.metadata().map_err(|error| {
            candle::Error::Msg(format!(
                "failed to inspect retained GPT-OSS GGUF {:?}: {error}",
                path
            ))
        })?;
        if !metadata.is_file() {
            bail!("GPT-OSS GGUF path {:?} is not a regular file", path);
        }
        let file_size = metadata.len();
        validate_file_size(file_size)?;
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let sha256 = sha256_reader(&mut file, cancellation)?;
        if sha256 != expected_sha256 {
            bail!(
                "GPT-OSS GGUF artifact identity mismatch: expected {}, got {}",
                expected_sha256,
                sha256
            );
        }
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        file.seek(SeekFrom::Start(0))?;
        let parsed = parse_gguf(&mut file, file_size)?;
        Self::from_parsed_with_file(path, sha256, parsed, Arc::new(Mutex::new(file)))
    }

    /// Admit one local artifact while holding a duplicate-load lease.
    ///
    /// A failed admission drops the in-progress lease, so a later retry can
    /// reopen the same path.  A successful result keeps ownership until the
    /// returned handle is dropped.
    pub fn open_with_registry(
        path: impl AsRef<Path>,
        registry: &GptOssLoadRegistry,
    ) -> Result<(Self, GptOssLoadedHandle)> {
        Self::open_with_registry_inner(path, registry, None)
    }

    /// Admit one local artifact with cooperative cancellation.
    pub fn open_with_registry_with_cancellation(
        path: impl AsRef<Path>,
        registry: &GptOssLoadRegistry,
        cancellation: &GptOssCancellationToken,
    ) -> Result<(Self, GptOssLoadedHandle)> {
        Self::open_with_registry_inner(path, registry, Some(cancellation))
    }

    fn open_with_registry_inner(
        path: impl AsRef<Path>,
        registry: &GptOssLoadRegistry,
        cancellation: Option<&GptOssCancellationToken>,
    ) -> Result<(Self, GptOssLoadedHandle)> {
        let path = path.as_ref();
        let canonical_path = std::fs::canonicalize(path).map_err(|error| {
            candle::Error::Msg(format!(
                "failed to canonicalize GPT-OSS GGUF {:?}: {error}",
                path
            ))
        })?;
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let identity = canonical_path.to_string_lossy().into_owned();
        let lease = registry.begin(identity)?;
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let artifact = Self::open_with_expected_sha(
            &canonical_path,
            SELECTED_GPT_OSS_GGUF_SHA256,
            cancellation,
        )?;
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let handle = lease.commit()?;
        Ok((artifact, handle))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    /// GGUF's maximum context length, distinct from the YaRN initial context.
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn tensors(&self) -> &[GptOssGgufTensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&GptOssGgufTensor> {
        match self
            .tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
        {
            Ok(index) => self.tensors.get(index).ok_or_else(|| {
                candle::Error::Msg(format!("GPT-OSS GGUF tensor {name:?} is not owned"))
            }),
            Err(_) => Err(candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {name:?} is not owned"
            ))),
        }
    }

    /// Read one admitted tensor's raw GGUF payload without dequantizing it.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(name)?;
        let mut session = GptOssGgufLoadSession::open(self)?;
        session.read_tensor(tensor, None)
    }

    /// Return the bytes retained by the assembled CPU model, before any
    /// transient tensor payload is read from the GGUF file.
    pub fn weight_resident_bytes(&self) -> Result<usize> {
        let mut total = 0usize;
        for tensor in &self.tensors {
            let bytes = match &tensor.role {
                GptOssTensorRole::ExpertGateUpWeight(_)
                | GptOssTensorRole::ExpertGateWeight(_)
                | GptOssTensorRole::ExpertUpWeight(_)
                | GptOssTensorRole::ExpertDownWeight(_) => usize::try_from(tensor.byte_len)
                    .map_err(|_| {
                        candle::Error::Msg(format!(
                            "GPT-OSS tensor {:?} resident bytes do not fit usize",
                            tensor.name
                        ))
                    })?,
                _ => usize::try_from(element_count(&tensor.shape, &tensor.name)?)
                    .map_err(|_| {
                        candle::Error::Msg(format!(
                            "GPT-OSS tensor {:?} element count does not fit usize",
                            tensor.name
                        ))
                    })?
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "GPT-OSS tensor {:?} resident bytes overflowed",
                            tensor.name
                        ))
                    })?,
            };
            total = total.checked_add(bytes).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS resident weight bytes overflowed".into())
            })?;
        }
        Ok(total)
    }

    /// Assemble one admitted GGUF artifact into the CPU reference weights.
    ///
    /// `max_resident_bytes` is checked before the first tensor payload is
    /// allocated. The loader then retains one file handle and reads each raw
    /// tensor once, dropping the transient payload after it is converted into
    /// the model-owned representation.
    pub fn load_weights(&self, max_resident_bytes: usize) -> Result<GptOssWeights> {
        let cancellation = GptOssCancellationToken::new();
        self.load_weights_with_cancellation(&cancellation, max_resident_bytes)
    }

    /// Assemble one admitted GGUF artifact with cooperative cancellation.
    pub fn load_weights_with_cancellation(
        &self,
        cancellation: &GptOssCancellationToken,
        max_resident_bytes: usize,
    ) -> Result<GptOssWeights> {
        let mut session = GptOssGgufLoadSession::open(self)?;
        self.assemble_weights_with_reader(
            |tensor| session.read_tensor(tensor, Some(cancellation)),
            Some(cancellation),
            max_resident_bytes,
        )
    }

    fn assemble_weights_with_reader<F>(
        &self,
        mut read_tensor: F,
        cancellation: Option<&GptOssCancellationToken>,
        max_resident_bytes: usize,
    ) -> Result<GptOssWeights>
    where
        F: FnMut(&GptOssGgufTensor) -> Result<Vec<u8>>,
    {
        let resident_bytes = self.weight_resident_bytes()?;
        if resident_bytes > max_resident_bytes {
            bail!(
                "GPT-OSS assembled weights require {resident_bytes} resident bytes, limit is {max_resident_bytes}"
            );
        }

        let token_embedding = read_dense_role(
            self,
            &mut read_tensor,
            &GptOssTensorRole::TokenEmbedding,
            cancellation,
        )?;
        let final_norm = read_dense_role(
            self,
            &mut read_tensor,
            &GptOssTensorRole::OutputNorm,
            cancellation,
        )?;
        let lm_head = read_linear_role(
            self,
            &mut read_tensor,
            &GptOssTensorRole::Output,
            None,
            cancellation,
        )?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.config.num_hidden_layers)
            .map_err(|error| {
                candle::Error::Msg(format!("GPT-OSS layer weight allocation failed: {error}"))
            })?;
        for layer_index in 0..self.config.num_hidden_layers {
            if let Some(cancellation) = cancellation {
                cancellation.checkpoint()?;
            }
            let attention_norm = read_dense_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::AttentionNorm(layer_index),
                cancellation,
            )?;
            let qkv = if self
                .find_role(&GptOssTensorRole::QkvWeight(layer_index))
                .is_some()
            {
                read_linear_role(
                    self,
                    &mut read_tensor,
                    &GptOssTensorRole::QkvWeight(layer_index),
                    Some(GptOssTensorRole::QkvBias(layer_index)),
                    cancellation,
                )?
            } else {
                let mut qkv_weights = Vec::new();
                let mut qkv_bias = Vec::new();
                for (weight_role, bias_role) in [
                    (
                        GptOssTensorRole::QueryWeight(layer_index),
                        GptOssTensorRole::QueryBias(layer_index),
                    ),
                    (
                        GptOssTensorRole::KeyWeight(layer_index),
                        GptOssTensorRole::KeyBias(layer_index),
                    ),
                    (
                        GptOssTensorRole::ValueWeight(layer_index),
                        GptOssTensorRole::ValueBias(layer_index),
                    ),
                ] {
                    let weight =
                        read_dense_role(self, &mut read_tensor, &weight_role, cancellation)?;
                    let bias = read_dense_role(self, &mut read_tensor, &bias_role, cancellation)?;
                    qkv_weights
                        .try_reserve_exact(weight.len())
                        .map_err(|error| {
                            candle::Error::Msg(format!("GPT-OSS QKV allocation failed: {error}"))
                        })?;
                    qkv_weights.extend_from_slice(&weight);
                    qkv_bias.try_reserve_exact(bias.len()).map_err(|error| {
                        candle::Error::Msg(format!("GPT-OSS QKV bias allocation failed: {error}"))
                    })?;
                    qkv_bias.extend_from_slice(&bias);
                }
                let hidden = self.config.hidden_size;
                let qkv_width = qkv_bias.len();
                DenseLinear::new(qkv_width, hidden, qkv_weights, Some(qkv_bias))?
            };
            let attention_out = read_linear_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::AttentionOutputWeight(layer_index),
                Some(GptOssTensorRole::AttentionOutputBias(layer_index)),
                cancellation,
            )?;
            let sinks = read_dense_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::AttentionSinks(layer_index),
                cancellation,
            )?;
            let moe_norm = read_dense_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::MoeNorm(layer_index),
                cancellation,
            )?;
            let gate = read_linear_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::RouterWeight(layer_index),
                Some(GptOssTensorRole::RouterBias(layer_index)),
                cancellation,
            )?;

            let (mlp1, mlp1_bias) = if self
                .find_role(&GptOssTensorRole::ExpertGateUpWeight(layer_index))
                .is_some()
            {
                let fused_intermediate =
                    self.config
                        .intermediate_size
                        .checked_mul(2)
                        .ok_or_else(|| {
                            candle::Error::Msg("GPT-OSS fused intermediate width overflowed".into())
                        })?;
                (
                    read_mxfp4_role(
                        self,
                        &mut read_tensor,
                        &GptOssTensorRole::ExpertGateUpWeight(layer_index),
                        &[
                            self.config.num_experts,
                            fused_intermediate,
                            self.config.hidden_size,
                        ],
                        cancellation,
                    )?,
                    read_dense_role(
                        self,
                        &mut read_tensor,
                        &GptOssTensorRole::ExpertGateUpBias(layer_index),
                        cancellation,
                    )?,
                )
            } else {
                let gate = read_mxfp4_role(
                    self,
                    &mut read_tensor,
                    &GptOssTensorRole::ExpertGateWeight(layer_index),
                    &[
                        self.config.num_experts,
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    ],
                    cancellation,
                )?;
                let up = read_mxfp4_role(
                    self,
                    &mut read_tensor,
                    &GptOssTensorRole::ExpertUpWeight(layer_index),
                    &[
                        self.config.num_experts,
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    ],
                    cancellation,
                )?;
                let gate_bias = read_dense_role(
                    self,
                    &mut read_tensor,
                    &GptOssTensorRole::ExpertGateBias(layer_index),
                    cancellation,
                )?;
                let up_bias = read_dense_role(
                    self,
                    &mut read_tensor,
                    &GptOssTensorRole::ExpertUpBias(layer_index),
                    cancellation,
                )?;
                (
                    interleave_mxfp4_rows(
                        &gate,
                        &up,
                        self.config.num_experts,
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    )?,
                    interleave_biases(
                        &gate_bias,
                        &up_bias,
                        self.config.num_experts,
                        self.config.intermediate_size,
                    )?,
                )
            };
            let mlp2 = read_mxfp4_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::ExpertDownWeight(layer_index),
                &[
                    self.config.num_experts,
                    self.config.hidden_size,
                    self.config.intermediate_size,
                ],
                cancellation,
            )?;
            let mlp2_bias = read_dense_role(
                self,
                &mut read_tensor,
                &GptOssTensorRole::ExpertDownBias(layer_index),
                cancellation,
            )?;
            let experts = Mxfp4ExpertOperation::new(
                mlp1,
                mlp1_bias,
                mlp2,
                mlp2_bias,
                self.config.swiglu_limit,
            )?;
            layers.push(GptOssLayerWeights::new(
                &self.config,
                attention_norm,
                qkv,
                attention_out,
                sinks,
                moe_norm,
                gate,
                experts,
            )?);
        }
        GptOssWeights::new(&self.config, token_embedding, layers, final_norm, lm_head)
    }

    fn find_role(&self, role: &GptOssTensorRole) -> Option<&GptOssGgufTensor> {
        self.tensors.iter().find(|tensor| &tensor.role == role)
    }

    #[cfg(test)]
    fn from_parsed(path: PathBuf, sha256: String, parsed: ParsedGguf) -> Result<Self> {
        let normalized = normalize_config(&parsed.metadata)?;
        let owners = expected_tensor_owners(&normalized.config, &parsed.tensors)?;
        let tensors = validate_and_assign_tensors(&normalized.config, &parsed.tensors, &owners)?;
        Ok(Self {
            path,
            sha256,
            config: normalized.config,
            context_length: normalized.context_length,
            tensor_data_offset: parsed.tensor_data_offset,
            file_size: parsed.file_size,
            tensors,
            retained_file: None,
        })
    }

    fn from_parsed_with_file(
        path: PathBuf,
        sha256: String,
        parsed: ParsedGguf,
        retained_file: Arc<Mutex<File>>,
    ) -> Result<Self> {
        let normalized = normalize_config(&parsed.metadata)?;
        let owners = expected_tensor_owners(&normalized.config, &parsed.tensors)?;
        let tensors = validate_and_assign_tensors(&normalized.config, &parsed.tensors, &owners)?;
        Ok(Self {
            path,
            sha256,
            config: normalized.config,
            context_length: normalized.context_length,
            tensor_data_offset: parsed.tensor_data_offset,
            file_size: parsed.file_size,
            tensors,
            retained_file: Some(retained_file),
        })
    }
}

/// One retained file session used by the GGUF-to-weights assembly path.
///
/// Admission has already verified the artifact identity. Keeping this handle
/// open avoids reopening and rehashing a multi-gigabyte file for every tensor;
/// the size is rechecked before the session starts and each read remains
/// bounded by the admitted tensor descriptor.
struct GptOssGgufLoadSession<'a> {
    artifact: &'a GptOssGgufArtifact,
    file: MutexGuard<'a, File>,
}

impl<'a> GptOssGgufLoadSession<'a> {
    fn open(artifact: &'a GptOssGgufArtifact) -> Result<Self> {
        let retained_file = artifact.retained_file.as_ref().ok_or_else(|| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF {:?} has no retained file session",
                artifact.path
            ))
        })?;
        let file = retained_file.lock().map_err(|_| {
            candle::Error::Msg("GPT-OSS GGUF retained file mutex was poisoned".into())
        })?;
        let current_size = file.metadata()?.len();
        if current_size != artifact.file_size {
            bail!(
                "admitted GPT-OSS GGUF {:?} changed size from {} to {}",
                artifact.path,
                artifact.file_size,
                current_size
            );
        }
        Ok(Self { artifact, file })
    }

    fn read_tensor(
        &mut self,
        tensor: &GptOssGgufTensor,
        cancellation: Option<&GptOssCancellationToken>,
    ) -> Result<Vec<u8>> {
        let owned = self.artifact.tensor(&tensor.name)?;
        if owned != tensor {
            bail!(
                "GPT-OSS GGUF tensor {:?} is not the admitted descriptor",
                tensor.name
            );
        }
        if tensor.byte_len > MAX_RAW_TENSOR_BYTES {
            bail!(
                "GPT-OSS GGUF tensor {:?} payload {} exceeds raw read limit {}",
                tensor.name,
                tensor.byte_len,
                MAX_RAW_TENSOR_BYTES
            );
        }
        let byte_len = usize::try_from(tensor.byte_len).map_err(|_| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {:?} payload does not fit in usize",
                tensor.name
            ))
        })?;
        let absolute_offset = self
            .artifact
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS GGUF tensor offset overflowed".into()))?;
        self.file.seek(SeekFrom::Start(absolute_offset))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|error| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {:?} allocation failed: {error}",
                tensor.name
            ))
        })?;
        bytes.resize(byte_len, 0);
        for chunk in bytes.chunks_mut(GGUF_READ_CHUNK_BYTES) {
            if let Some(cancellation) = cancellation {
                cancellation.checkpoint()?;
            }
            self.file.read_exact(chunk).map_err(|error| {
                candle::Error::Msg(format!(
                    "GPT-OSS GGUF tensor {:?} is truncated while reading: {error}",
                    tensor.name
                ))
            })?;
        }
        Ok(bytes)
    }
}

/// Admit one exact local GGUF and assemble it while retaining one load lease.
pub fn load_gpt_oss_weights(
    path: impl AsRef<Path>,
    registry: &GptOssLoadRegistry,
    max_resident_bytes: usize,
) -> Result<(GptOssGgufArtifact, GptOssWeights, GptOssLoadedHandle)> {
    let cancellation = GptOssCancellationToken::new();
    load_gpt_oss_weights_with_cancellation(path, registry, &cancellation, max_resident_bytes)
}

/// Admit one exact local GGUF, assemble it with cancellation, and return the
/// retained handle so duplicate concurrent loads remain rejected.
pub fn load_gpt_oss_weights_with_cancellation(
    path: impl AsRef<Path>,
    registry: &GptOssLoadRegistry,
    cancellation: &GptOssCancellationToken,
    max_resident_bytes: usize,
) -> Result<(GptOssGgufArtifact, GptOssWeights, GptOssLoadedHandle)> {
    let (artifact, handle) =
        GptOssGgufArtifact::open_with_registry_with_cancellation(path, registry, cancellation)?;
    let mut weights = artifact.load_weights_with_cancellation(cancellation, max_resident_bytes)?;
    weights.attach_load_handle(handle.clone());
    Ok((artifact, weights, handle))
}

fn read_role_bytes<F>(
    artifact: &GptOssGgufArtifact,
    read_tensor: &mut F,
    role: &GptOssTensorRole,
    cancellation: Option<&GptOssCancellationToken>,
) -> Result<Vec<u8>>
where
    F: FnMut(&GptOssGgufTensor) -> Result<Vec<u8>>,
{
    if let Some(cancellation) = cancellation {
        cancellation.checkpoint()?;
    }
    let tensor = artifact
        .find_role(role)
        .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor role {role:?}")))?;
    let bytes = read_tensor(tensor)?;
    if bytes.len()
        != usize::try_from(tensor.byte_len).map_err(|_| {
            candle::Error::Msg(format!(
                "GPT-OSS tensor {:?} length does not fit usize",
                tensor.name
            ))
        })?
    {
        bail!(
            "GPT-OSS tensor {:?} read {} bytes, expected {}",
            tensor.name,
            bytes.len(),
            tensor.byte_len
        );
    }
    if let Some(cancellation) = cancellation {
        cancellation.checkpoint()?;
    }
    Ok(bytes)
}

fn decode_dense_values(tensor: &GptOssGgufTensor, bytes: &[u8]) -> Result<Vec<f32>> {
    let count = usize::try_from(element_count(&tensor.shape, &tensor.name)?).map_err(|_| {
        candle::Error::Msg(format!(
            "GPT-OSS tensor {:?} element count does not fit usize",
            tensor.name
        ))
    })?;
    let dtype = match tensor.dtype {
        GptOssGgufDType::F32 => DType::F32,
        GptOssGgufDType::F16 => DType::F16,
        GptOssGgufDType::BF16 => DType::BF16,
        GptOssGgufDType::Q8_0 => {
            bail!(
                "GPT-OSS GGUF tensor {:?} uses Q8_0; quantized dense text assembly is deferred",
                tensor.name
            )
        }
        GptOssGgufDType::Mxfp4 => {
            bail!(
                "GPT-OSS tensor {:?} is MXFP4, not a dense F32 tensor",
                tensor.name
            )
        }
    };
    Tensor::from_raw_buffer(bytes, dtype, &[count], &Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
}

fn read_dense_role<F>(
    artifact: &GptOssGgufArtifact,
    read_tensor: &mut F,
    role: &GptOssTensorRole,
    cancellation: Option<&GptOssCancellationToken>,
) -> Result<Vec<f32>>
where
    F: FnMut(&GptOssGgufTensor) -> Result<Vec<u8>>,
{
    let tensor = artifact
        .find_role(role)
        .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor role {role:?}")))?;
    let bytes = read_role_bytes(artifact, read_tensor, role, cancellation)?;
    decode_dense_values(tensor, &bytes)
}

fn read_linear_role<F>(
    artifact: &GptOssGgufArtifact,
    read_tensor: &mut F,
    weight_role: &GptOssTensorRole,
    bias_role: Option<GptOssTensorRole>,
    cancellation: Option<&GptOssCancellationToken>,
) -> Result<DenseLinear>
where
    F: FnMut(&GptOssGgufTensor) -> Result<Vec<u8>>,
{
    let weight_tensor = artifact.find_role(weight_role).ok_or_else(|| {
        candle::Error::Msg(format!("missing GPT-OSS tensor role {weight_role:?}"))
    })?;
    if weight_tensor.shape.len() != 2 {
        bail!(
            "GPT-OSS dense weight {:?} must be rank 2, got {:?}",
            weight_tensor.name,
            weight_tensor.shape
        );
    }
    let weight_bytes = read_role_bytes(artifact, read_tensor, weight_role, cancellation)?;
    let weights = decode_dense_values(weight_tensor, &weight_bytes)?;
    let bias = match bias_role {
        None => None,
        Some(role) => Some(read_dense_role(artifact, read_tensor, &role, cancellation)?),
    };
    // The normalized GPT-OSS GGUF inventory uses Candle's output-by-input
    // matrix order, so its contiguous bytes can be retained directly.
    DenseLinear::new(
        weight_tensor.shape[0],
        weight_tensor.shape[1],
        weights,
        bias,
    )
}

fn read_mxfp4_role<F>(
    artifact: &GptOssGgufArtifact,
    read_tensor: &mut F,
    role: &GptOssTensorRole,
    packed_shape: &[usize],
    cancellation: Option<&GptOssCancellationToken>,
) -> Result<PackedMxfp4>
where
    F: FnMut(&GptOssGgufTensor) -> Result<Vec<u8>>,
{
    let tensor = artifact
        .find_role(role)
        .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor role {role:?}")))?;
    if tensor.dtype != GptOssGgufDType::Mxfp4 {
        bail!(
            "GPT-OSS expert tensor {:?} must be MXFP4, got {:?}",
            tensor.name,
            tensor.dtype
        );
    }
    let bytes = read_role_bytes(artifact, read_tensor, role, cancellation)?;
    if !bytes.len().is_multiple_of(BYTES_PER_BLOCK + 1) {
        bail!(
            "GPT-OSS MXFP4 tensor {:?} has {} bytes, not divisible by 17",
            tensor.name,
            bytes.len()
        );
    }
    let block_count = bytes.len() / (BYTES_PER_BLOCK + 1);
    let block_bytes =
        checked_usize_product([block_count, BYTES_PER_BLOCK], "GPT-OSS MXFP4 block bytes")?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(block_bytes).map_err(|error| {
        candle::Error::Msg(format!("GPT-OSS MXFP4 block allocation failed: {error}"))
    })?;
    let mut scales = Vec::new();
    scales.try_reserve_exact(block_count).map_err(|error| {
        candle::Error::Msg(format!("GPT-OSS MXFP4 scale allocation failed: {error}"))
    })?;
    for chunk in bytes.chunks_exact(BYTES_PER_BLOCK + 1) {
        scales.push(chunk[0]);
        blocks.extend_from_slice(&chunk[1..]);
    }
    PackedMxfp4::from_ggml_parts(packed_shape, blocks, scales)
}

fn interleave_mxfp4_rows(
    gate: &PackedMxfp4,
    up: &PackedMxfp4,
    expert_count: usize,
    rows: usize,
    hidden_size: usize,
) -> Result<PackedMxfp4> {
    let expected_shape = [expert_count, rows, hidden_size];
    if gate.shape() != expected_shape || up.shape() != expected_shape {
        bail!(
            "GPT-OSS split expert shapes do not match {:?}: gate={:?}, up={:?}",
            expected_shape,
            gate.shape(),
            up.shape()
        );
    }
    let blocks_per_row = hidden_size / VALUES_PER_BLOCK;
    if !hidden_size.is_multiple_of(VALUES_PER_BLOCK) {
        bail!(
            "GPT-OSS split expert hidden size {hidden_size} is not divisible by {VALUES_PER_BLOCK}"
        );
    }
    let row_block_bytes = checked_usize_product(
        [blocks_per_row, BYTES_PER_BLOCK],
        "GPT-OSS expert row block bytes",
    )?;
    let row_count =
        checked_usize_product([expert_count, rows, 2], "GPT-OSS fused expert row count")?;
    let block_bytes = checked_usize_product(
        [row_count, row_block_bytes],
        "GPT-OSS fused expert block bytes",
    )?;
    let fused_rows = rows
        .checked_mul(2)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS fused expert row count overflowed".into()))?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(block_bytes).map_err(|error| {
        candle::Error::Msg(format!(
            "GPT-OSS fused expert block allocation failed: {error}"
        ))
    })?;
    let mut scales = Vec::new();
    let scale_count = checked_usize_product(
        [row_count, blocks_per_row],
        "GPT-OSS fused expert scale count",
    )?;
    scales.try_reserve_exact(scale_count).map_err(|error| {
        candle::Error::Msg(format!(
            "GPT-OSS fused expert scale allocation failed: {error}"
        ))
    })?;
    for expert in 0..expert_count {
        for row in 0..rows {
            for (source_blocks, source_scales) in
                [(gate.blocks(), gate.scales()), (up.blocks(), up.scales())]
            {
                let row_index = expert
                    .checked_mul(rows)
                    .and_then(|value| value.checked_add(row))
                    .ok_or_else(|| candle::Error::Msg("GPT-OSS expert row overflowed".into()))?;
                let block_start = row_index
                    .checked_mul(blocks_per_row)
                    .and_then(|value| value.checked_mul(BYTES_PER_BLOCK))
                    .ok_or_else(|| {
                        candle::Error::Msg("GPT-OSS expert block offset overflowed".into())
                    })?;
                let block_end = block_start.checked_add(row_block_bytes).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert block range overflowed".into())
                })?;
                blocks.extend_from_slice(source_blocks.get(block_start..block_end).ok_or_else(
                    || candle::Error::Msg("GPT-OSS split expert block row is out of bounds".into()),
                )?);
                let scale_start = row_index.checked_mul(blocks_per_row).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert scale offset overflowed".into())
                })?;
                let scale_end = scale_start.checked_add(blocks_per_row).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert scale range overflowed".into())
                })?;
                scales.extend_from_slice(source_scales.get(scale_start..scale_end).ok_or_else(
                    || candle::Error::Msg("GPT-OSS split expert scale row is out of bounds".into()),
                )?);
            }
        }
    }
    PackedMxfp4::from_parts(&[expert_count, fused_rows, hidden_size], blocks, scales)
}

fn interleave_biases(
    gate: &[f32],
    up: &[f32],
    expert_count: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    let row_count = checked_usize_product([expert_count, rows], "GPT-OSS expert bias rows")?;
    if gate.len() != row_count || up.len() != row_count {
        bail!("GPT-OSS split expert bias shapes do not match the expert configuration");
    }
    let result_len = checked_usize_product([row_count, 2], "GPT-OSS expert bias values")?;
    let mut result = Vec::new();
    result.try_reserve_exact(result_len).map_err(|error| {
        candle::Error::Msg(format!("GPT-OSS expert bias allocation failed: {error}"))
    })?;
    for expert in 0..expert_count {
        for row in 0..rows {
            let index = expert
                .checked_mul(rows)
                .and_then(|value| value.checked_add(row))
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert bias offset overflowed".into())
                })?;
            result.push(
                *gate.get(index).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS gate bias is out of bounds".into())
                })?,
            );
            result
                .push(*up.get(index).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS up bias is out of bounds".into())
                })?);
        }
    }
    Ok(result)
}

#[derive(Debug)]
struct ParsedGguf {
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, RawTensor>,
    tensor_data_offset: u64,
    file_size: u64,
}

#[derive(Debug, Clone)]
struct RawTensor {
    name: String,
    shape: Vec<usize>,
    dtype: GptOssGgufDType,
    offset: u64,
    byte_len: u64,
}

#[derive(Debug, Clone)]
enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool,
    String(String),
    Array(u64),
}

#[derive(Debug, Clone, Copy)]
enum GgufVersion {
    V2,
    V3,
}

fn validate_file_size(file_size: u64) -> Result<()> {
    if file_size == 0 || file_size > GGUF_MAX_FILE_BYTES {
        bail!("GPT-OSS GGUF file size {file_size} is outside 1..={GGUF_MAX_FILE_BYTES} bytes");
    }
    Ok(())
}

fn sha256_reader<R: Read + Seek>(
    reader: &mut R,
    cancellation: Option<&GptOssCancellationToken>,
) -> Result<String> {
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_gguf<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<ParsedGguf> {
    reader.seek(SeekFrom::Start(0))?;
    let magic = read_bytes(reader, file_size, file_size, 4)?;
    if magic.as_slice() != b"GGUF" {
        bail!("GPT-OSS GGUF has invalid magic {:?}", magic);
    }
    let version_number = read_u32(reader, file_size, file_size)?;
    let version = match version_number {
        2 => GgufVersion::V2,
        3 => GgufVersion::V3,
        other => bail!("unsupported GPT-OSS GGUF version {other}"),
    };
    let tensor_count = read_length(reader, file_size, file_size, version)?;
    let metadata_count = read_length(reader, file_size, file_size, version)?;
    if tensor_count > GGUF_MAX_TENSORS {
        bail!("GPT-OSS GGUF tensor count {tensor_count} exceeds {GGUF_MAX_TENSORS}");
    }
    if metadata_count > GGUF_MAX_METADATA {
        bail!("GPT-OSS GGUF metadata count {metadata_count} exceeds {GGUF_MAX_METADATA}");
    }
    let header_limit = file_size.min(GGUF_MAX_HEADER_BYTES);

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = read_string(reader, file_size, header_limit, version)?;
        let value_type = read_u32(reader, file_size, header_limit)?;
        let value = read_value(reader, file_size, header_limit, version, value_type, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            bail!("GPT-OSS GGUF contains duplicate metadata key {key:?}");
        }
    }

    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name = read_string(reader, file_size, header_limit, version)?;
        let dimensions = read_u32(reader, file_size, header_limit)?;
        if dimensions == 0 || dimensions > 4 {
            bail!("GPT-OSS GGUF tensor {name:?} has invalid rank {dimensions}; expected 1..=4");
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(dimensions as usize)
            .map_err(|error| {
                candle::Error::Msg(format!("GGUF shape allocation failed: {error}"))
            })?;
        for _ in 0..dimensions {
            let dimension = read_u64(reader, file_size, header_limit)?;
            if dimension == 0 {
                bail!("GPT-OSS GGUF tensor {name:?} has a zero dimension");
            }
            shape.push(usize::try_from(dimension).map_err(|_| {
                candle::Error::Msg(format!(
                    "GPT-OSS GGUF tensor {name:?} dimension {dimension} does not fit usize"
                ))
            })?);
        }
        shape.reverse();
        let dtype_id = read_u32(reader, file_size, header_limit)?;
        let dtype = GptOssGgufDType::from_id(dtype_id)?;
        let offset = read_u64(reader, file_size, header_limit)?;
        let name_for_error = name.clone();
        let raw = RawTensor {
            name,
            shape,
            dtype,
            offset,
            byte_len: 0,
        };
        if tensors.insert(raw.name.clone(), raw).is_some() {
            bail!("GPT-OSS GGUF contains duplicate tensor {name_for_error:?}");
        }
    }

    let position = reader.stream_position()?;
    if position > header_limit {
        bail!("GPT-OSS GGUF header exceeds {GGUF_MAX_HEADER_BYTES} bytes");
    }
    let alignment = optional_positive_integer(&metadata, "general.alignment")?
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if !alignment.is_power_of_two() {
        bail!("GPT-OSS GGUF alignment {alignment} is not a power of two");
    }
    let tensor_data_offset = align_up(position, alignment)?;
    if tensor_data_offset > file_size {
        bail!("GPT-OSS GGUF tensor data begins beyond the end of the file");
    }

    let mut ranges = Vec::new();
    for raw in tensors.values_mut() {
        if !raw.offset.is_multiple_of(alignment) {
            bail!(
                "GPT-OSS GGUF tensor {:?} offset {} is not aligned to {}",
                raw.name,
                raw.offset,
                alignment
            );
        }
        let elements = element_count(&raw.shape, &raw.name)?;
        raw.byte_len = raw.dtype.byte_len(elements)?;
        let start = tensor_data_offset
            .checked_add(raw.offset)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS GGUF tensor offset overflowed".into()))?;
        let end = start
            .checked_add(raw.byte_len)
            .ok_or_else(|| candle::Error::Msg(format!("tensor {:?} range overflowed", raw.name)))?;
        if end > file_size {
            bail!(
                "GPT-OSS GGUF tensor {:?} ends at byte {end}, beyond file size {file_size}",
                raw.name
            );
        }
        ranges.push((start, end, raw.name.clone()));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = tensor_data_offset;
    for (start, end, name) in ranges {
        if start < previous_end {
            bail!("GPT-OSS GGUF tensor {name:?} overlaps another tensor");
        }
        previous_end = end;
    }

    Ok(ParsedGguf {
        metadata,
        tensors,
        tensor_data_offset,
        file_size,
    })
}

fn read_value<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    header_limit: u64,
    version: GgufVersion,
    value_type: u32,
    depth: usize,
) -> Result<MetadataValue> {
    if depth > GGUF_MAX_VALUE_DEPTH {
        bail!("GPT-OSS GGUF metadata nesting exceeds {GGUF_MAX_VALUE_DEPTH}");
    }
    match value_type {
        0 => Ok(MetadataValue::U8(read_u8(reader, file_size, header_limit)?)),
        1 => Ok(MetadataValue::I8(read_i8(reader, file_size, header_limit)?)),
        2 => Ok(MetadataValue::U16(read_u16(
            reader,
            file_size,
            header_limit,
        )?)),
        3 => Ok(MetadataValue::I16(read_i16(
            reader,
            file_size,
            header_limit,
        )?)),
        4 => Ok(MetadataValue::U32(read_u32(
            reader,
            file_size,
            header_limit,
        )?)),
        5 => Ok(MetadataValue::I32(read_i32(
            reader,
            file_size,
            header_limit,
        )?)),
        6 => Ok(MetadataValue::F32(read_f32(
            reader,
            file_size,
            header_limit,
        )?)),
        7 => {
            let value = read_u8(reader, file_size, header_limit)?;
            match value {
                0 | 1 => Ok(MetadataValue::Bool),
                other => bail!("GPT-OSS GGUF metadata bool has invalid value {other}"),
            }
        }
        8 => Ok(MetadataValue::String(read_string(
            reader,
            file_size,
            header_limit,
            version,
        )?)),
        9 => {
            let element_type = read_u32(reader, file_size, header_limit)?;
            let length = read_length(reader, file_size, header_limit, version)?;
            if length > GGUF_MAX_ARRAY_ELEMENTS {
                bail!(
                    "GPT-OSS GGUF metadata array length {length} exceeds {GGUF_MAX_ARRAY_ELEMENTS}"
                );
            }
            for _ in 0..length {
                let _ = read_value(
                    reader,
                    file_size,
                    header_limit,
                    version,
                    element_type,
                    depth + 1,
                )?;
            }
            Ok(MetadataValue::Array(length))
        }
        10 => Ok(MetadataValue::U64(read_u64(
            reader,
            file_size,
            header_limit,
        )?)),
        11 => Ok(MetadataValue::I64(read_i64(
            reader,
            file_size,
            header_limit,
        )?)),
        12 => Ok(MetadataValue::F64(read_f64(
            reader,
            file_size,
            header_limit,
        )?)),
        other => bail!("GPT-OSS GGUF metadata value type {other} is unsupported"),
    }
}

fn read_string<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    limit: u64,
    version: GgufVersion,
) -> Result<String> {
    let length = read_length(reader, file_size, limit, version)?;
    if length > GGUF_MAX_STRING_BYTES {
        bail!("GPT-OSS GGUF string length {length} exceeds {GGUF_MAX_STRING_BYTES}");
    }
    let length_usize = usize::try_from(length)
        .map_err(|_| candle::Error::Msg("GPT-OSS GGUF string length does not fit usize".into()))?;
    let bytes = read_bytes(reader, file_size, limit, length_usize)?;
    String::from_utf8(bytes)
        .map_err(|error| candle::Error::Msg(format!("GPT-OSS GGUF string is not UTF-8: {error}")))
}

fn read_length<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    limit: u64,
    version: GgufVersion,
) -> Result<u64> {
    match version {
        GgufVersion::V2 | GgufVersion::V3 => read_u64(reader, file_size, limit),
    }
}

fn read_bytes<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    limit: u64,
    length: usize,
) -> Result<Vec<u8>> {
    let position = reader.stream_position()?;
    let length_u64 = u64::try_from(length)
        .map_err(|_| candle::Error::Msg("GGUF read length does not fit u64".into()))?;
    let end = position
        .checked_add(length_u64)
        .ok_or_else(|| candle::Error::Msg("GGUF read range overflowed".into()))?;
    if end > limit || end > file_size {
        bail!("GPT-OSS GGUF read needs bytes through {end}, beyond the bounded header");
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| candle::Error::Msg(format!("GGUF read allocation failed: {error}")))?;
    bytes.resize(length, 0);
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

macro_rules! read_number {
    ($name:ident, $ty:ty, $bytes:expr) => {
        fn $name<R: Read + Seek>(reader: &mut R, file_size: u64, limit: u64) -> Result<$ty> {
            let bytes = read_bytes(reader, file_size, limit, $bytes)?;
            let array: [u8; $bytes] = bytes
                .try_into()
                .map_err(|_| candle::Error::Msg("GGUF scalar width mismatch".into()))?;
            Ok(<$ty>::from_le_bytes(array))
        }
    };
}

fn read_u8<R: Read + Seek>(reader: &mut R, file_size: u64, limit: u64) -> Result<u8> {
    Ok(*read_bytes(reader, file_size, limit, 1)?
        .first()
        .ok_or_else(|| candle::Error::Msg("GGUF u8 read was empty".into()))?)
}

fn read_i8<R: Read + Seek>(reader: &mut R, file_size: u64, limit: u64) -> Result<i8> {
    Ok(read_u8(reader, file_size, limit)? as i8)
}

read_number!(read_u16, u16, 2);
read_number!(read_i16, i16, 2);
read_number!(read_u32, u32, 4);
read_number!(read_i32, i32, 4);
read_number!(read_u64, u64, 8);
read_number!(read_i64, i64, 8);
read_number!(read_f32, f32, 4);
read_number!(read_f64, f64, 8);

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment.saturating_sub(1))
        .and_then(|value| value.checked_div(alignment))
        .and_then(|value| value.checked_mul(alignment))
        .ok_or_else(|| candle::Error::Msg("GPT-OSS GGUF alignment overflowed".into()))
}

fn element_count(shape: &[usize], name: &str) -> Result<u64> {
    shape.iter().try_fold(1u64, |count, &dimension| {
        count
            .checked_mul(u64::try_from(dimension).map_err(|_| {
                candle::Error::Msg(format!("GPT-OSS GGUF tensor {name:?} dimension overflowed"))
            })?)
            .ok_or_else(|| candle::Error::Msg(format!("tensor {name:?} element count overflowed")))
    })
}

fn optional_positive_integer(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<u64>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(value) => {
            let value = match value {
                MetadataValue::U8(value) => u64::from(*value),
                MetadataValue::U16(value) => u64::from(*value),
                MetadataValue::U32(value) => u64::from(*value),
                MetadataValue::U64(value) => *value,
                MetadataValue::I8(value) if *value > 0 => *value as u64,
                MetadataValue::I16(value) if *value > 0 => *value as u64,
                MetadataValue::I32(value) if *value > 0 => *value as u64,
                MetadataValue::I64(value) if *value > 0 => *value as u64,
                _ => bail!("GPT-OSS GGUF metadata {key:?} must be a positive integer"),
            };
            if value == 0 {
                bail!("GPT-OSS GGUF metadata {key:?} must be positive");
            }
            Ok(Some(value))
        }
    }
}

fn required_usize(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<usize> {
    let value = optional_positive_integer(metadata, key)?.ok_or_else(|| {
        candle::Error::Msg(format!(
            "GPT-OSS GGUF metadata is missing required key {key:?}"
        ))
    })?;
    usize::try_from(value).map_err(|_| {
        candle::Error::Msg(format!(
            "GPT-OSS GGUF metadata {key:?} value {value} does not fit usize"
        ))
    })
}

fn optional_f32(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<Option<f32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::F32(value)) if value.is_finite() => Ok(Some(*value)),
        Some(MetadataValue::F64(value)) if value.is_finite() => Ok(Some(*value as f32)),
        Some(value) => bail!("GPT-OSS GGUF metadata {key:?} must be a finite float, got {value:?}"),
    }
}

fn required_f32(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<f32> {
    let value = optional_f32(metadata, key)?.ok_or_else(|| {
        candle::Error::Msg(format!(
            "GPT-OSS GGUF metadata is missing required key {key:?}"
        ))
    })?;
    if value <= 0.0 {
        bail!("GPT-OSS GGUF metadata {key:?} must be positive");
    }
    Ok(value)
}

fn optional_string<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<&'a str>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::String(value)) => Ok(Some(value.as_str())),
        Some(value) => bail!("GPT-OSS GGUF metadata {key:?} must be a string, got {value:?}"),
    }
}

fn required_string<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str> {
    optional_string(metadata, key)?.ok_or_else(|| {
        candle::Error::Msg(format!(
            "GPT-OSS GGUF metadata is missing required key {key:?}"
        ))
    })
}

#[derive(Debug)]
struct NormalizedConfig {
    config: GptOssConfig,
    context_length: usize,
}

fn normalize_config(metadata: &BTreeMap<String, MetadataValue>) -> Result<NormalizedConfig> {
    let architecture = required_string(metadata, "general.architecture")?;
    if architecture != "gpt-oss" {
        bail!("unsupported GPT-OSS GGUF architecture {architecture:?}; expected \"gpt-oss\"");
    }

    let vocab_metadata = optional_positive_integer(metadata, "gpt-oss.vocab_size")?;
    let vocab_tokens = match metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(length)) => Some(*length),
        Some(value) => bail!("GPT-OSS GGUF tokenizer.ggml.tokens must be an array, got {value:?}"),
        None => None,
    };
    let vocab_size = match (vocab_metadata, vocab_tokens) {
        (Some(metadata), Some(tokens)) if metadata != tokens => bail!(
            "GPT-OSS GGUF vocabulary identity mismatch: metadata {metadata}, tokenizer {tokens}"
        ),
        (Some(metadata), _) => usize::try_from(metadata)
            .map_err(|_| candle::Error::Msg("GPT-OSS GGUF vocab_size does not fit usize".into()))?,
        (None, Some(tokens)) => usize::try_from(tokens).map_err(|_| {
            candle::Error::Msg("GPT-OSS GGUF tokenizer vocabulary does not fit usize".into())
        })?,
        (None, None) => {
            bail!("GPT-OSS GGUF must provide gpt-oss.vocab_size or tokenizer.ggml.tokens")
        }
    };

    let expert_feed_forward_length = match (
        optional_positive_integer(metadata, "gpt-oss.expert_feed_forward_length")?,
        optional_positive_integer(metadata, "gpt-oss.feed_forward_length")?,
    ) {
        (Some(expert), Some(feed_forward)) if expert != feed_forward => bail!(
            "GPT-OSS GGUF expert/feed-forward length mismatch: {expert} versus {feed_forward}"
        ),
        (Some(value), _) | (_, Some(value)) => usize::try_from(value).map_err(|_| {
            candle::Error::Msg("GPT-OSS GGUF intermediate size does not fit usize".into())
        })?,
        (None, None) => {
            bail!("GPT-OSS GGUF is missing expert_feed_forward_length/feed_forward_length")
        }
    };

    let initial_context_length = match (
        optional_positive_integer(metadata, "gpt-oss.rope.scaling.original_context_length")?,
        optional_positive_integer(
            metadata,
            "gpt-oss.rope.scaling.original_max_position_embeddings",
        )?,
    ) {
        (Some(first), Some(second)) if first != second => {
            bail!("GPT-OSS GGUF YaRN original-context mismatch: {first} versus {second}")
        }
        (Some(value), _) | (_, Some(value)) => usize::try_from(value).map_err(|_| {
            candle::Error::Msg("GPT-OSS GGUF original context does not fit usize".into())
        })?,
        (None, None) => bail!("GPT-OSS GGUF is missing YaRN original_context_length metadata"),
    };
    let context_length = required_usize(metadata, "gpt-oss.context_length")?;
    if context_length < initial_context_length {
        bail!(
            "GPT-OSS GGUF context_length {context_length} is smaller than YaRN initial context {initial_context_length}"
        );
    }

    let rope_type = optional_string(metadata, "gpt-oss.rope.scaling.type")?;
    if let Some(rope_type) = rope_type {
        if rope_type != "yarn" {
            bail!("unsupported GPT-OSS GGUF rope scaling type {rope_type:?}");
        }
    }
    let rope_ntk_alpha = optional_f32(metadata, "gpt-oss.rope.scaling.beta_slow")?.unwrap_or(1.0);
    let rope_ntk_beta = optional_f32(metadata, "gpt-oss.rope.scaling.beta_fast")?.unwrap_or(32.0);
    let config = GptOssConfig {
        model_type: "gpt_oss".to_string(),
        num_hidden_layers: required_usize(metadata, "gpt-oss.block_count")?,
        num_experts: required_usize(metadata, "gpt-oss.expert_count")?,
        experts_per_token: required_usize(metadata, "gpt-oss.expert_used_count")?,
        vocab_size,
        hidden_size: required_usize(metadata, "gpt-oss.embedding_length")?,
        intermediate_size: expert_feed_forward_length,
        swiglu_limit: optional_f32(metadata, "gpt-oss.swiglu_limit")?.unwrap_or(7.0),
        head_dim: required_usize(metadata, "gpt-oss.attention.key_length")?,
        num_attention_heads: required_usize(metadata, "gpt-oss.attention.head_count")?,
        num_key_value_heads: required_usize(metadata, "gpt-oss.attention.head_count_kv")?,
        sliding_window: required_usize(metadata, "gpt-oss.attention.sliding_window")?,
        initial_context_length,
        rope_theta: required_f32(metadata, "gpt-oss.rope.freq_base")?,
        rope_scaling_factor: required_f32(metadata, "gpt-oss.rope.scaling.factor")?,
        rope_ntk_alpha,
        rope_ntk_beta,
        architectures: vec!["GptOssForCausalLM".to_string()],
    };
    let value_length = required_usize(metadata, "gpt-oss.attention.value_length")?;
    if value_length != config.head_dim {
        bail!(
            "GPT-OSS GGUF key/value head dimensions disagree: {} versus {}",
            config.head_dim,
            value_length
        );
    }
    config.validate()?;
    Ok(NormalizedConfig {
        config,
        context_length,
    })
}

#[derive(Debug, Clone, Copy)]
enum QkvLayout {
    Fused,
    Split,
}

#[derive(Debug, Clone, Copy)]
enum ExpertLayout {
    Fused,
    Split,
}

#[derive(Debug, Clone, Copy)]
struct TensorLayout {
    qkv: QkvLayout,
    attention_output_suffix: &'static str,
    sinks_suffix: &'static str,
    moe_norm_suffix: &'static str,
    experts: ExpertLayout,
}

fn expected_tensor_owners(
    config: &GptOssConfig,
    tensors: &BTreeMap<String, RawTensor>,
) -> Result<BTreeMap<String, GptOssTensorRole>> {
    let names: BTreeSet<&str> = tensors.keys().map(String::as_str).collect();
    let layout = infer_tensor_layout(config, &names)?;
    let mut owners = BTreeMap::new();
    insert_owner(
        &mut owners,
        "token_embd.weight",
        GptOssTensorRole::TokenEmbedding,
    )?;
    insert_owner(
        &mut owners,
        "output_norm.weight",
        GptOssTensorRole::OutputNorm,
    )?;
    insert_owner(&mut owners, "output.weight", GptOssTensorRole::Output)?;

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        insert_owner(
            &mut owners,
            &format!("{prefix}.attn_norm.weight"),
            GptOssTensorRole::AttentionNorm(layer),
        )?;
        match layout.qkv {
            QkvLayout::Fused => {
                insert_owner(
                    &mut owners,
                    &format!("{prefix}.attn_qkv.weight"),
                    GptOssTensorRole::QkvWeight(layer),
                )?;
                insert_owner(
                    &mut owners,
                    &format!("{prefix}.attn_qkv.bias"),
                    GptOssTensorRole::QkvBias(layer),
                )?;
            }
            QkvLayout::Split => {
                for (suffix, weight_role, bias_role) in [
                    (
                        "q",
                        GptOssTensorRole::QueryWeight(layer),
                        GptOssTensorRole::QueryBias(layer),
                    ),
                    (
                        "k",
                        GptOssTensorRole::KeyWeight(layer),
                        GptOssTensorRole::KeyBias(layer),
                    ),
                    (
                        "v",
                        GptOssTensorRole::ValueWeight(layer),
                        GptOssTensorRole::ValueBias(layer),
                    ),
                ] {
                    insert_owner(
                        &mut owners,
                        &format!("{prefix}.attn_{suffix}.weight"),
                        weight_role,
                    )?;
                    insert_owner(
                        &mut owners,
                        &format!("{prefix}.attn_{suffix}.bias"),
                        bias_role,
                    )?;
                }
            }
        }
        insert_owner(
            &mut owners,
            &format!("{prefix}.{}", layout.attention_output_suffix),
            GptOssTensorRole::AttentionOutputWeight(layer),
        )?;
        insert_owner(
            &mut owners,
            &format!(
                "{prefix}.{}",
                layout.attention_output_suffix.replace("weight", "bias")
            ),
            GptOssTensorRole::AttentionOutputBias(layer),
        )?;
        insert_owner(
            &mut owners,
            &format!("{prefix}.{}", layout.sinks_suffix),
            GptOssTensorRole::AttentionSinks(layer),
        )?;
        insert_owner(
            &mut owners,
            &format!("{prefix}.{}", layout.moe_norm_suffix),
            GptOssTensorRole::MoeNorm(layer),
        )?;
        insert_owner(
            &mut owners,
            &format!("{prefix}.ffn_gate_inp.weight"),
            GptOssTensorRole::RouterWeight(layer),
        )?;
        insert_owner(
            &mut owners,
            &format!("{prefix}.ffn_gate_inp.bias"),
            GptOssTensorRole::RouterBias(layer),
        )?;
        match layout.experts {
            ExpertLayout::Fused => {
                for (suffix, weight_role, bias_role) in [
                    (
                        "ffn_gate_up_exps",
                        GptOssTensorRole::ExpertGateUpWeight(layer),
                        GptOssTensorRole::ExpertGateUpBias(layer),
                    ),
                    (
                        "ffn_down_exps",
                        GptOssTensorRole::ExpertDownWeight(layer),
                        GptOssTensorRole::ExpertDownBias(layer),
                    ),
                ] {
                    insert_owner(
                        &mut owners,
                        &format!("{prefix}.{suffix}.weight"),
                        weight_role,
                    )?;
                    insert_owner(&mut owners, &format!("{prefix}.{suffix}.bias"), bias_role)?;
                }
            }
            ExpertLayout::Split => {
                for (suffix, weight_role, bias_role) in [
                    (
                        "ffn_gate_exps",
                        GptOssTensorRole::ExpertGateWeight(layer),
                        GptOssTensorRole::ExpertGateBias(layer),
                    ),
                    (
                        "ffn_up_exps",
                        GptOssTensorRole::ExpertUpWeight(layer),
                        GptOssTensorRole::ExpertUpBias(layer),
                    ),
                    (
                        "ffn_down_exps",
                        GptOssTensorRole::ExpertDownWeight(layer),
                        GptOssTensorRole::ExpertDownBias(layer),
                    ),
                ] {
                    insert_owner(
                        &mut owners,
                        &format!("{prefix}.{suffix}.weight"),
                        weight_role,
                    )?;
                    insert_owner(&mut owners, &format!("{prefix}.{suffix}.bias"), bias_role)?;
                }
            }
        }
    }
    let actual: BTreeSet<&str> = names;
    let expected: BTreeSet<&str> = owners.keys().map(String::as_str).collect();
    let missing: Vec<_> = expected.difference(&actual).copied().collect();
    let unexpected: Vec<_> = actual.difference(&expected).copied().collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        bail!(
            "GPT-OSS GGUF tensor identity mismatch: missing={missing:?}, unexpected={unexpected:?}"
        );
    }
    Ok(owners)
}

fn infer_tensor_layout(config: &GptOssConfig, names: &BTreeSet<&str>) -> Result<TensorLayout> {
    let has = |name: String| names.contains(name.as_str());
    let qkv_fused = has("blk.0.attn_qkv.weight".to_string());
    let qkv_split = ["q", "k", "v"].iter().all(|suffix| {
        has(format!("blk.0.attn_{suffix}.weight")) && has(format!("blk.0.attn_{suffix}.bias"))
    });
    let qkv = match (qkv_fused, qkv_split) {
        (true, false) => QkvLayout::Fused,
        (false, true) => QkvLayout::Split,
        _ => bail!("GPT-OSS GGUF layer 0 has no unique QKV tensor layout"),
    };
    let attention_output_suffix = choose_suffix(
        names,
        ["attn_out.weight", "attn_output.weight"],
        "attention output weight",
    )?;
    let sinks_suffix = choose_suffix(
        names,
        ["attn_sinks", "attn_sinks.weight"],
        "attention sinks",
    )?;
    let moe_norm_suffix = choose_suffix(
        names,
        ["ffn_norm.weight", "post_attention_norm.weight"],
        "MoE norm",
    )?;
    let gate_up = has("blk.0.ffn_gate_up_exps.weight".to_string());
    let split_experts = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .all(|suffix| has(format!("blk.0.{suffix}.weight")));
    let experts = match (gate_up, split_experts) {
        (true, false) => ExpertLayout::Fused,
        (false, true) => ExpertLayout::Split,
        _ => bail!("GPT-OSS GGUF layer 0 has no unique expert tensor layout"),
    };
    for layer in 1..config.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        let expected_qkv = match qkv {
            QkvLayout::Fused => has(format!("{prefix}.attn_qkv.weight")),
            QkvLayout::Split => ["q", "k", "v"].iter().all(|suffix| {
                has(format!("{prefix}.attn_{suffix}.weight"))
                    && has(format!("{prefix}.attn_{suffix}.bias"))
            }),
        };
        if !expected_qkv {
            bail!("GPT-OSS GGUF layer {layer} does not match layer 0 QKV layout");
        }
    }
    Ok(TensorLayout {
        qkv,
        attention_output_suffix: format_suffix(attention_output_suffix),
        sinks_suffix: format_suffix(sinks_suffix),
        moe_norm_suffix: format_suffix(moe_norm_suffix),
        experts,
    })
}

fn choose_suffix<const N: usize>(
    names: &BTreeSet<&str>,
    suffixes: [&'static str; N],
    label: &str,
) -> Result<&'static str> {
    let present: Vec<_> = suffixes
        .iter()
        .filter(|suffix| names.contains(format!("blk.0.{suffix}").as_str()))
        .copied()
        .collect();
    match present.as_slice() {
        [suffix] => Ok(suffix.strip_prefix("blk.0.").unwrap_or(suffix)),
        _ => bail!("GPT-OSS GGUF layer 0 has ambiguous or missing {label}: {present:?}"),
    }
}

fn format_suffix(suffix: &str) -> &'static str {
    match suffix {
        "attn_out.weight" => "attn_out.weight",
        "attn_output.weight" => "attn_output.weight",
        "attn_sinks" => "attn_sinks",
        "attn_sinks.weight" => "attn_sinks.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "post_attention_norm.weight" => "post_attention_norm.weight",
        _ => unreachable!("validated GPT-OSS tensor suffix"),
    }
}

fn insert_owner(
    owners: &mut BTreeMap<String, GptOssTensorRole>,
    name: &str,
    role: GptOssTensorRole,
) -> Result<()> {
    if owners.insert(name.to_string(), role).is_some() {
        bail!("GPT-OSS GGUF tensor owner duplicated {name:?}");
    }
    Ok(())
}

fn validate_and_assign_tensors(
    config: &GptOssConfig,
    tensors: &BTreeMap<String, RawTensor>,
    owners: &BTreeMap<String, GptOssTensorRole>,
) -> Result<Vec<GptOssGgufTensor>> {
    let mut result = Vec::new();
    result.try_reserve_exact(owners.len()).map_err(|error| {
        candle::Error::Msg(format!(
            "GPT-OSS tensor inventory allocation failed: {error}"
        ))
    })?;
    for (name, role) in owners {
        let tensor = tensors.get(name).ok_or_else(|| {
            candle::Error::Msg(format!("GPT-OSS GGUF owned tensor {name:?} is absent"))
        })?;
        let expected_shape = expected_shape(config, role)?;
        if tensor.shape != expected_shape {
            bail!(
                "GPT-OSS GGUF tensor {name:?} shape mismatch: expected {expected_shape:?}, got {:?}",
                tensor.shape
            );
        }
        validate_dtype(role, tensor.dtype, name)?;
        result.push(GptOssGgufTensor {
            name: name.clone(),
            role: role.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            offset: tensor.offset,
            byte_len: tensor.byte_len,
        });
    }
    Ok(result)
}

fn expected_shape(config: &GptOssConfig, role: &GptOssTensorRole) -> Result<Vec<usize>> {
    let q_width = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS Q width overflowed".into()))?;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".into()))?;
    let qkv_width = q_width
        .checked_add(
            kv_width
                .checked_mul(2)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV width overflowed".into()))?,
        )
        .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV width overflowed".into()))?;
    let fused_intermediate = config
        .intermediate_size
        .checked_mul(2)
        .ok_or_else(|| candle::Error::Msg("GPT-OSS fused intermediate width overflowed".into()))?;
    let shape = match role {
        GptOssTensorRole::TokenEmbedding | GptOssTensorRole::Output => {
            vec![config.vocab_size, config.hidden_size]
        }
        GptOssTensorRole::OutputNorm
        | GptOssTensorRole::AttentionNorm(_)
        | GptOssTensorRole::MoeNorm(_) => vec![config.hidden_size],
        GptOssTensorRole::QkvWeight(_) => vec![qkv_width, config.hidden_size],
        GptOssTensorRole::QkvBias(_) => vec![qkv_width],
        GptOssTensorRole::QueryWeight(_) => vec![q_width, config.hidden_size],
        GptOssTensorRole::QueryBias(_) => vec![q_width],
        GptOssTensorRole::KeyWeight(_) | GptOssTensorRole::ValueWeight(_) => {
            vec![kv_width, config.hidden_size]
        }
        GptOssTensorRole::KeyBias(_) | GptOssTensorRole::ValueBias(_) => vec![kv_width],
        GptOssTensorRole::AttentionOutputWeight(_) => vec![config.hidden_size, q_width],
        GptOssTensorRole::AttentionOutputBias(_) => vec![config.hidden_size],
        GptOssTensorRole::AttentionSinks(_) => vec![config.num_attention_heads],
        GptOssTensorRole::RouterWeight(_) => vec![config.num_experts, config.hidden_size],
        GptOssTensorRole::RouterBias(_) => vec![config.num_experts],
        GptOssTensorRole::ExpertGateUpWeight(_) => {
            vec![config.num_experts, fused_intermediate, config.hidden_size]
        }
        GptOssTensorRole::ExpertGateUpBias(_) => {
            vec![config.num_experts, fused_intermediate]
        }
        GptOssTensorRole::ExpertGateWeight(_) | GptOssTensorRole::ExpertUpWeight(_) => {
            vec![
                config.num_experts,
                config.intermediate_size,
                config.hidden_size,
            ]
        }
        GptOssTensorRole::ExpertGateBias(_) | GptOssTensorRole::ExpertUpBias(_) => {
            vec![config.num_experts, config.intermediate_size]
        }
        GptOssTensorRole::ExpertDownWeight(_) => {
            vec![
                config.num_experts,
                config.intermediate_size,
                config.hidden_size,
            ]
        }
        GptOssTensorRole::ExpertDownBias(_) => vec![config.num_experts, config.hidden_size],
    };
    Ok(shape)
}

fn validate_dtype(role: &GptOssTensorRole, dtype: GptOssGgufDType, name: &str) -> Result<()> {
    let is_expert_weight = matches!(
        role,
        GptOssTensorRole::ExpertGateUpWeight(_)
            | GptOssTensorRole::ExpertGateWeight(_)
            | GptOssTensorRole::ExpertUpWeight(_)
            | GptOssTensorRole::ExpertDownWeight(_)
    );
    if is_expert_weight && dtype != GptOssGgufDType::Mxfp4 {
        bail!("GPT-OSS expert weight {name:?} must be MXFP4, got {dtype:?}");
    }
    if !is_expert_weight && dtype == GptOssGgufDType::Mxfp4 {
        bail!("GPT-OSS non-expert tensor {name:?} cannot use MXFP4");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpt_oss::GptOssLoadRegistry;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn normalizes_and_owns_tiny_fused_gguf() -> Result<()> {
        let bytes = tiny_gguf()?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        let artifact = parse_test_bytes(&bytes, &actual)?;
        assert_eq!(artifact.sha256(), actual);
        assert_eq!(artifact.config().num_hidden_layers, 1);
        assert_eq!(artifact.config().num_experts, 2);
        assert_eq!(artifact.config().vocab_size, 8);
        assert_eq!(artifact.context_length(), 16);
        assert_eq!(artifact.tensors().len(), 16);
        let expert = artifact.tensor("blk.0.ffn_gate_up_exps.weight")?;
        assert_eq!(expert.role, GptOssTensorRole::ExpertGateUpWeight(0));
        assert_eq!(expert.dtype, GptOssGgufDType::Mxfp4);
        assert_eq!(expert.shape, vec![2, 64, 32]);
        assert_eq!(expert.byte_len, 2176);
        Ok(())
    }

    #[test]
    fn accepts_converter_style_split_expert_inventory() -> Result<()> {
        let bytes = tiny_converter_style_gguf()?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        let artifact = parse_test_bytes(&bytes, &actual)?;
        assert_eq!(artifact.tensors().len(), 22);
        assert_eq!(
            artifact.tensor("blk.0.attn_k.weight")?.role,
            GptOssTensorRole::KeyWeight(0)
        );
        assert_eq!(
            artifact.tensor("blk.0.attn_output.weight")?.role,
            GptOssTensorRole::AttentionOutputWeight(0)
        );
        assert_eq!(
            artifact.tensor("blk.0.attn_sinks.weight")?.role,
            GptOssTensorRole::AttentionSinks(0)
        );
        assert_eq!(
            artifact.tensor("blk.0.post_attention_norm.weight")?.role,
            GptOssTensorRole::MoeNorm(0)
        );
        assert_eq!(
            artifact.tensor("blk.0.ffn_gate_exps.weight")?.role,
            GptOssTensorRole::ExpertGateWeight(0)
        );
        Ok(())
    }

    #[test]
    fn assembles_fused_and_split_synthetic_gguf_weights() -> Result<()> {
        for bytes in [tiny_gguf()?, tiny_converter_style_gguf()?] {
            let actual = format!("{:x}", Sha256::digest(&bytes));
            let artifact = parse_test_bytes(&bytes, &actual)?;
            let max_resident_bytes = artifact.weight_resident_bytes()?;
            let weights = artifact.assemble_weights_with_reader(
                |tensor| {
                    let start = usize::try_from(
                        artifact
                            .tensor_data_offset()
                            .checked_add(tensor.offset)
                            .ok_or_else(|| {
                                candle::Error::Msg("synthetic GGUF offset overflowed".into())
                            })?,
                    )
                    .map_err(|_| candle::Error::Msg("synthetic GGUF offset is too large".into()))?;
                    let byte_len = usize::try_from(tensor.byte_len).map_err(|_| {
                        candle::Error::Msg("synthetic GGUF tensor length is too large".into())
                    })?;
                    let end = start.checked_add(byte_len).ok_or_else(|| {
                        candle::Error::Msg("synthetic GGUF tensor range overflowed".into())
                    })?;
                    bytes.get(start..end).map(ToOwned::to_owned).ok_or_else(|| {
                        candle::Error::Msg("synthetic GGUF tensor is truncated".into())
                    })
                },
                None,
                max_resident_bytes,
            )?;
            let model =
                crate::models::gpt_oss::GptOssModel::new(artifact.config().clone(), weights)?;
            let logits = model.forward_uncached(&[1])?;
            assert_eq!(logits.len(), artifact.config().vocab_size);
        }
        Ok(())
    }

    #[test]
    fn real_loader_normalizes_nonzero_fused_and_split_gguf_and_matches_reference() -> Result<()> {
        let fixtures = [
            (
                "fused",
                tiny_nonzero_fused_gguf()?,
                "blk.0.ffn_gate_up_exps.weight",
                None,
            ),
            (
                "split",
                tiny_nonzero_split_gguf()?,
                "blk.0.ffn_gate_exps.weight",
                Some("blk.0.ffn_up_exps.weight"),
            ),
        ];
        let mut reference_logits: Option<Vec<f32>> = None;
        let mut reference_contribution: Option<Vec<f32>> = None;
        for (label, bytes, gate_name, up_name) in fixtures {
            let path = temporary_gguf_path(label, &bytes)?;
            let expected_sha = format!("{:x}", Sha256::digest(&bytes));
            let artifact =
                GptOssGgufArtifact::open_with_expected_sha(&path.0, &expected_sha, None)?;
            let gate_up_bytes = artifact.read_tensor(gate_name)?;
            let up_bytes = match up_name {
                Some(name) => Some(artifact.read_tensor(name)?),
                None => None,
            };
            let down_bytes = artifact.read_tensor("blk.0.ffn_down_exps.weight")?;
            let max_resident_bytes = artifact.weight_resident_bytes()?;
            let weights = artifact.load_weights(max_resident_bytes)?;
            let experts = weights
                .layers()
                .first()
                .ok_or_else(|| candle::Error::Msg("test fixture has no layer".into()))?
                .experts();

            for coordinate in 0..VALUES_PER_BLOCK {
                let expected = wire_value(&gate_up_bytes, 0, coordinate)?;
                assert_eq!(
                    experts.mlp1().value_at(0, coordinate)?,
                    expected,
                    "{label} loaded MLP1 coordinate {coordinate}"
                );
                assert_eq!(
                    experts.mlp2().value_at(0, coordinate)?,
                    wire_value(&down_bytes, 0, coordinate)?,
                    "{label} loaded MLP2 coordinate {coordinate}"
                );
            }
            if let Some(up_bytes) = &up_bytes {
                for coordinate in 0..VALUES_PER_BLOCK {
                    assert_eq!(
                        experts.mlp1().value_at(1, coordinate)?,
                        wire_value(up_bytes, 0, coordinate)?,
                        "{label} loaded split-up coordinate {coordinate}"
                    );
                }
            }

            let input: Vec<f32> = (0..32).map(|index| (index as f32 + 0.5) / 17.0).collect();
            let actual_contribution = experts.forward_contribution(&input, 1, &[0], &[1.0])?;
            let expected_contribution = reference_expert_contribution(
                &gate_up_bytes,
                up_bytes.as_deref(),
                &down_bytes,
                &input,
            )?;
            assert_close(&actual_contribution, &expected_contribution, 1e-5, label);
            if let Some(reference) = &reference_contribution {
                assert_close(&actual_contribution, reference, 1e-5, label);
            } else {
                reference_contribution = Some(actual_contribution.clone());
            }

            let config = artifact.config().clone();
            let model = crate::models::gpt_oss::GptOssModel::new(config.clone(), weights.clone())?;
            let logits = model.forward_uncached(&[1])?;
            if let Some(reference) = &reference_logits {
                assert_close(&logits, reference, 1e-5, label);
            } else {
                reference_logits = Some(logits.clone());
            }

            #[cfg(feature = "cuda")]
            {
                let limits = crate::models::gpt_oss::GptOssResourceLimits::for_config(&config)?;
                let cuda_config = crate::models::gpt_oss::GptOssCudaConfig::new(
                    0,
                    DType::F32,
                    usize::MAX,
                    limits,
                )
                .map_err(|error| candle::Error::Msg(error.to_string()))?;
                let cuda =
                    crate::models::gpt_oss::GptOssCudaModel::new(config, weights, cuda_config)
                        .map_err(|error| candle::Error::Msg(error.to_string()))?;
                let cuda_logits = cuda
                    .forward_uncached(&[1], &GptOssCancellationToken::new())
                    .map_err(|error| candle::Error::Msg(error.to_string()))?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_close(&cuda_logits, &logits, 1e-4, label);
            }
            drop(model);
            drop(artifact);
        }
        Ok(())
    }

    #[test]
    fn retained_session_reads_original_after_same_size_path_replacement() -> Result<()> {
        let original = tiny_nonzero_fused_gguf()?;
        let replacement = {
            let mut bytes = original.clone();
            let last = bytes
                .last_mut()
                .ok_or_else(|| candle::Error::Msg("test GGUF is empty".into()))?;
            *last ^= 0x01;
            bytes
        };
        let original_path = temporary_gguf_path("retained-original", &original)?;
        let replacement_path = temporary_gguf_path("retained-replacement", &replacement)?;
        let expected_sha = format!("{:x}", Sha256::digest(&original));
        let artifact =
            GptOssGgufArtifact::open_with_expected_sha(&original_path.0, &expected_sha, None)?;
        let expected_payload = artifact.read_tensor("token_embd.weight")?;
        let rename = std::fs::rename(&replacement_path.0, &original_path.0);
        if let Err(error) = rename {
            assert!(
                matches!(
                    error.kind(),
                    std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::AlreadyExists
                ),
                "unexpected replacement failure: {error}"
            );
        }
        assert_eq!(
            artifact.read_tensor("token_embd.weight")?,
            expected_payload,
            "retained admission handle must not follow a same-size replacement path"
        );
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn retained_session_denies_same_size_mutation_attempts() -> Result<()> {
        let bytes = tiny_nonzero_fused_gguf()?;
        let path = temporary_gguf_path("retained-mutation", &bytes)?;
        let expected_sha = format!("{:x}", Sha256::digest(&bytes));
        let artifact = GptOssGgufArtifact::open_with_expected_sha(&path.0, &expected_sha, None)?;
        let mutation = std::fs::OpenOptions::new().write(true).open(&path.0);
        let error = mutation.expect_err("Windows admission must deny a writer");
        assert!(
            error.kind() == std::io::ErrorKind::PermissionDenied
                || matches!(error.raw_os_error(), Some(5 | 32)),
            "unexpected Windows sharing failure: {error}"
        );
        let _ = artifact.read_tensor("token_embd.weight")?;
        Ok(())
    }

    #[test]
    fn admission_hash_and_chunked_reads_honor_cancellation() -> Result<()> {
        let bytes = tiny_nonzero_fused_gguf()?;
        let path = temporary_gguf_path("cancellable-read", &bytes)?;
        let expected_sha = format!("{:x}", Sha256::digest(&bytes));
        let hash_cancelled = GptOssCancellationToken::cancel_after_checks(0);
        let error = GptOssGgufArtifact::open_with_expected_sha(
            &path.0,
            &expected_sha,
            Some(&hash_cancelled),
        )
        .expect_err("hash admission must be cancellable");
        assert!(error.to_string().contains("cancelled"));

        let artifact = GptOssGgufArtifact::open_with_expected_sha(&path.0, &expected_sha, None)?;
        let read_cancelled = GptOssCancellationToken::cancel_after_checks(1);
        let error = artifact
            .load_weights_with_cancellation(&read_cancelled, artifact.weight_resident_bytes()?)
            .expect_err("chunked tensor reads must be cancellable");
        assert!(error.to_string().contains("cancelled"));
        Ok(())
    }

    #[test]
    fn load_lease_stays_with_model_and_releases_after_teardown_or_construction_failure(
    ) -> Result<()> {
        let bytes = tiny_nonzero_fused_gguf()?;
        let path = temporary_gguf_path("lease-owner", &bytes)?;
        let expected_sha = format!("{:x}", Sha256::digest(&bytes));
        let canonical = std::fs::canonicalize(&path.0).map_err(candle::Error::wrap)?;
        let identity = canonical.to_string_lossy().into_owned();
        let registry = GptOssLoadRegistry::default();

        let artifact = GptOssGgufArtifact::open_with_expected_sha(&path.0, &expected_sha, None)?;
        let handle = registry.begin(identity.clone())?.commit()?;
        let mut weights = artifact.load_weights(artifact.weight_resident_bytes()?)?;
        weights.attach_load_handle(handle.clone());
        let model = crate::models::gpt_oss::GptOssModel::new(artifact.config().clone(), weights)?;
        drop(handle);
        assert_eq!(registry.loaded_count()?, 1);
        assert!(registry
            .begin(identity.clone())
            .expect_err("duplicate load must remain rejected while model lives")
            .to_string()
            .contains("duplicate"));
        drop(model);
        assert_eq!(registry.loaded_count()?, 0);

        let artifact = GptOssGgufArtifact::open_with_expected_sha(&path.0, &expected_sha, None)?;
        let handle = registry.begin(identity)?.commit()?;
        let mut weights = artifact.load_weights(artifact.weight_resident_bytes()?)?;
        weights.attach_load_handle(handle.clone());
        let mut invalid_config = artifact.config().clone();
        invalid_config.num_hidden_layers += 1;
        let error = crate::models::gpt_oss::GptOssModel::new(invalid_config, weights)
            .expect_err("construction failure must drop the resident owner");
        assert!(error.to_string().contains("layer count"));
        drop(handle);
        assert_eq!(registry.loaded_count()?, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires the owner-admitted local GPT-OSS GGUF artifact"]
    fn assembles_owner_selected_product_artifact_when_requested() -> Result<()> {
        let path = std::env::var_os("CANDLE_GPT_OSS_GGUF")
            .ok_or_else(|| candle::Error::Msg("CANDLE_GPT_OSS_GGUF is not set".into()))?;
        let max_resident_bytes = std::env::var("CANDLE_GPT_OSS_MAX_RESIDENT_BYTES")
            .unwrap_or_else(|_| (32usize * 1024 * 1024 * 1024).to_string())
            .parse::<usize>()
            .map_err(|error| {
                candle::Error::Msg(format!(
                    "invalid CANDLE_GPT_OSS_MAX_RESIDENT_BYTES: {error}"
                ))
            })?;
        let registry = GptOssLoadRegistry::default();
        let cancellation = GptOssCancellationToken::new();
        let (artifact, weights, handle) = load_gpt_oss_weights_with_cancellation(
            path,
            &registry,
            &cancellation,
            max_resident_bytes,
        )?;
        assert_eq!(artifact.sha256(), SELECTED_GPT_OSS_GGUF_SHA256);
        assert_eq!(registry.loaded_count()?, 1);
        let model = crate::models::gpt_oss::GptOssModel::new(artifact.config().clone(), weights)?;
        drop(artifact);
        drop(handle);
        assert_eq!(registry.loaded_count()?, 1);
        drop(model);
        assert_eq!(registry.loaded_count()?, 0);
        Ok(())
    }

    #[test]
    fn rejects_wrong_identity_before_parsing() -> Result<()> {
        let bytes = tiny_gguf()?;
        let error =
            parse_test_bytes(&bytes, SELECTED_GPT_OSS_GGUF_SHA256).expect_err("identity must fail");
        assert!(error.to_string().contains("identity mismatch"));
        Ok(())
    }

    #[test]
    fn rejects_truncated_payload_and_malformed_header() -> Result<()> {
        let bytes = tiny_gguf()?;
        let truncated = &bytes[..bytes.len() - 1];
        let truncated_hash = format!("{:x}", Sha256::digest(truncated));
        let error = parse_test_bytes(truncated, &truncated_hash).expect_err("truncation must fail");
        assert!(
            error.to_string().contains("beyond file size")
                || error.to_string().contains("truncated")
        );

        let malformed = b"not-gguf";
        let malformed_hash = format!("{:x}", Sha256::digest(malformed));
        let error = parse_test_bytes(malformed, &malformed_hash).expect_err("bad magic must fail");
        assert!(error.to_string().contains("invalid magic"));
        Ok(())
    }

    #[test]
    fn rejects_tensor_identity_and_config_shape_mismatch() -> Result<()> {
        let shape_mismatch = tiny_gguf_with_qkv_shape(&[32, 32])?;
        let shape_hash = format!("{:x}", Sha256::digest(&shape_mismatch));
        let error = parse_test_bytes(&shape_mismatch, &shape_hash)
            .expect_err("incompatible tensor shape must fail");
        assert!(error.to_string().contains("shape mismatch"));

        let mut renamed = tiny_gguf()?;
        replace_ascii_once(
            &mut renamed,
            b"blk.0.ffn_gate_inp.bias",
            b"blk.0.ffn_gate_inp.biax",
        )?;
        let renamed_hash = format!("{:x}", Sha256::digest(&renamed));
        let error =
            parse_test_bytes(&renamed, &renamed_hash).expect_err("unowned tensor name must fail");
        assert!(error.to_string().contains("tensor identity mismatch"));
        Ok(())
    }

    #[test]
    fn failed_registered_load_releases_lease_for_retry() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "gpt-oss-task2-{}-{}.gguf",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(candle::Error::wrap)?
                .as_nanos()
        ));
        std::fs::write(&path, tiny_gguf()?).map_err(candle::Error::wrap)?;
        let registry = GptOssLoadRegistry::default();

        let error = GptOssGgufArtifact::open_with_registry(&path, &registry)
            .expect_err("wrong-identity fixture must fail registered admission");
        assert!(error.to_string().contains("identity mismatch"));
        assert_eq!(registry.active_count()?, 0);
        assert_eq!(registry.loaded_count()?, 0);

        let retry = GptOssGgufArtifact::open_with_registry(&path, &registry)
            .expect_err("retry should reach the same identity check, not a duplicate-load error");
        assert!(retry.to_string().contains("identity mismatch"));
        assert_eq!(registry.active_count()?, 0);
        assert_eq!(registry.loaded_count()?, 0);
        std::fs::remove_file(&path).map_err(candle::Error::wrap)?;
        Ok(())
    }

    #[test]
    fn cancelled_registered_load_releases_lease_for_retry() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "gpt-oss-task2-cancelled-{}-{}.gguf",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(candle::Error::wrap)?
                .as_nanos()
        ));
        std::fs::write(&path, tiny_gguf()?).map_err(candle::Error::wrap)?;
        let registry = GptOssLoadRegistry::default();
        let cancellation = GptOssCancellationToken::cancel_after_checks(1);

        let error = GptOssGgufArtifact::open_with_registry_with_cancellation(
            &path,
            &registry,
            &cancellation,
        )
        .expect_err("cancellation after lease acquisition must fail the load");
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(registry.active_count()?, 0);
        assert_eq!(registry.loaded_count()?, 0);

        let retry = GptOssGgufArtifact::open_with_registry(&path, &registry)
            .expect_err("retry should reach identity validation after cancellation cleanup");
        assert!(retry.to_string().contains("identity mismatch"));
        assert_eq!(registry.active_count()?, 0);
        assert_eq!(registry.loaded_count()?, 0);
        std::fs::remove_file(&path).map_err(candle::Error::wrap)?;
        Ok(())
    }

    fn parse_test_bytes(bytes: &[u8], expected_sha256: &str) -> Result<GptOssGgufArtifact> {
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != expected_sha256 {
            bail!(
                "GPT-OSS GGUF artifact identity mismatch: expected {expected_sha256}, got {actual}"
            );
        }
        let mut cursor = std::io::Cursor::new(bytes);
        let parsed = parse_gguf(&mut cursor, bytes.len() as u64)?;
        GptOssGgufArtifact::from_parsed(PathBuf::from("<in-memory-gpt-oss-gguf>"), actual, parsed)
    }

    fn tiny_gguf() -> Result<Vec<u8>> {
        tiny_gguf_with_options(&[48, 32], false, false, false, false)
    }

    fn tiny_gguf_with_qkv_shape(qkv_shape: &[usize]) -> Result<Vec<u8>> {
        tiny_gguf_with_options(qkv_shape, false, false, false, false)
    }

    fn tiny_converter_style_gguf() -> Result<Vec<u8>> {
        tiny_gguf_with_options(&[48, 32], true, true, true, false)
    }

    fn tiny_nonzero_fused_gguf() -> Result<Vec<u8>> {
        tiny_gguf_with_options(&[48, 32], false, false, false, true)
    }

    fn tiny_nonzero_split_gguf() -> Result<Vec<u8>> {
        tiny_gguf_with_options(&[48, 32], true, true, true, true)
    }

    fn tiny_gguf_with_options(
        qkv_shape: &[usize],
        split_experts: bool,
        converter_style_names: bool,
        split_qkv: bool,
        nonzero_payloads: bool,
    ) -> Result<Vec<u8>> {
        let config = [
            ("general.architecture", Meta::String("gpt-oss")),
            ("gpt-oss.vocab_size", Meta::U32(8)),
            ("gpt-oss.context_length", Meta::U32(16)),
            ("gpt-oss.embedding_length", Meta::U32(32)),
            ("gpt-oss.block_count", Meta::U32(1)),
            ("gpt-oss.expert_feed_forward_length", Meta::U32(32)),
            ("gpt-oss.expert_count", Meta::U32(2)),
            ("gpt-oss.expert_used_count", Meta::U32(1)),
            ("gpt-oss.attention.head_count", Meta::U32(8)),
            ("gpt-oss.attention.head_count_kv", Meta::U32(2)),
            ("gpt-oss.attention.key_length", Meta::U32(4)),
            ("gpt-oss.attention.value_length", Meta::U32(4)),
            ("gpt-oss.attention.sliding_window", Meta::U32(4)),
            ("gpt-oss.rope.freq_base", Meta::F32(10_000.0)),
            ("gpt-oss.rope.scaling.factor", Meta::F32(1.0)),
            ("gpt-oss.rope.scaling.original_context_length", Meta::U32(4)),
            ("gpt-oss.rope.scaling.type", Meta::String("yarn")),
            ("tokenizer.ggml.tokens", Meta::StringArray(8)),
        ];
        let mut tensors = Vec::new();
        push_tensor(&mut tensors, "token_embd.weight", &[8, 32], Dtype::F32)?;
        push_tensor(&mut tensors, "output_norm.weight", &[32], Dtype::F32)?;
        push_tensor(&mut tensors, "output.weight", &[8, 32], Dtype::F32)?;
        push_tensor(&mut tensors, "blk.0.attn_norm.weight", &[32], Dtype::F32)?;
        if split_qkv {
            for (suffix, width) in [("q", 32usize), ("k", 8usize), ("v", 8usize)] {
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.attn_{suffix}.weight"),
                    &[width, 32],
                    Dtype::F32,
                )?;
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.attn_{suffix}.bias"),
                    &[width],
                    Dtype::F32,
                )?;
            }
        } else {
            push_tensor(&mut tensors, "blk.0.attn_qkv.weight", qkv_shape, Dtype::F32)?;
            push_tensor(&mut tensors, "blk.0.attn_qkv.bias", &[48], Dtype::F32)?;
        }
        let attention_output = if converter_style_names {
            "attn_output"
        } else {
            "attn_out"
        };
        let attention_sinks = if converter_style_names {
            "attn_sinks.weight"
        } else {
            "attn_sinks"
        };
        let moe_norm = if converter_style_names {
            "post_attention_norm"
        } else {
            "ffn_norm"
        };
        push_tensor(
            &mut tensors,
            &format!("blk.0.{attention_output}.weight"),
            &[32, 32],
            Dtype::F32,
        )?;
        push_tensor(
            &mut tensors,
            &format!("blk.0.{attention_output}.bias"),
            &[32],
            Dtype::F32,
        )?;
        push_tensor(
            &mut tensors,
            &format!("blk.0.{attention_sinks}"),
            &[8],
            Dtype::F32,
        )?;
        push_tensor(
            &mut tensors,
            &format!("blk.0.{moe_norm}.weight"),
            &[32],
            Dtype::F32,
        )?;
        push_tensor(
            &mut tensors,
            "blk.0.ffn_gate_inp.weight",
            &[2, 32],
            Dtype::F32,
        )?;
        push_tensor(&mut tensors, "blk.0.ffn_gate_inp.bias", &[2], Dtype::F32)?;
        if split_experts {
            for suffix in ["ffn_gate_exps", "ffn_up_exps"] {
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.{suffix}.weight"),
                    &[2, 32, 32],
                    Dtype::Mxfp4,
                )?;
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.{suffix}.bias"),
                    &[2, 32],
                    Dtype::F32,
                )?;
            }
        } else {
            push_tensor(
                &mut tensors,
                "blk.0.ffn_gate_up_exps.weight",
                &[2, 64, 32],
                Dtype::Mxfp4,
            )?;
            push_tensor(
                &mut tensors,
                "blk.0.ffn_gate_up_exps.bias",
                &[2, 64],
                Dtype::F32,
            )?;
        }
        push_tensor(
            &mut tensors,
            "blk.0.ffn_down_exps.weight",
            &[2, 32, 32],
            Dtype::Mxfp4,
        )?;
        push_tensor(
            &mut tensors,
            "blk.0.ffn_down_exps.bias",
            &[2, 32],
            Dtype::F32,
        )?;
        if nonzero_payloads {
            for tensor in &mut tensors {
                fill_nonzero_test_tensor(tensor)?;
            }
        }
        build_gguf(&config, &tensors)
    }

    fn fill_nonzero_test_tensor(tensor: &mut TestTensor) -> Result<()> {
        match tensor.dtype {
            Dtype::F32 => {
                for (index, chunk) in tensor.bytes.chunks_exact_mut(4).enumerate() {
                    let value = if tensor.name == "token_embd.weight" {
                        (index % 32 + 1) as f32 / 32.0
                    } else if tensor.name == "output_norm.weight"
                        || tensor.name.contains("norm.weight")
                    {
                        1.0
                    } else if tensor.name == "output.weight" {
                        ((index / 32 + 1) * (index % 32 + 1)) as f32 / 64.0
                    } else {
                        0.0
                    };
                    chunk.copy_from_slice(&value.to_le_bytes());
                }
            }
            Dtype::Mxfp4 => {
                for block in tensor.bytes.chunks_exact_mut(BYTES_PER_BLOCK + 1) {
                    block[0] = 127;
                    for (index, byte) in block[1..].iter_mut().enumerate() {
                        let low = ((index * 3 + 1) % 16) as u8;
                        let high = ((index * 5 + 9) % 16) as u8;
                        *byte = low | (high << 4);
                    }
                }
            }
        }
        Ok(())
    }

    fn replace_ascii_once(bytes: &mut [u8], from: &[u8], to: &[u8]) -> Result<()> {
        if from.len() != to.len() {
            bail!("test replacement lengths differ");
        }
        let position = bytes
            .windows(from.len())
            .position(|window| window == from)
            .ok_or_else(|| candle::Error::Msg("test replacement needle is absent".into()))?;
        bytes[position..position + to.len()].copy_from_slice(to);
        Ok(())
    }

    struct TempGgufPath(PathBuf);

    impl Drop for TempGgufPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn temporary_gguf_path(label: &str, bytes: &[u8]) -> Result<TempGgufPath> {
        let path = std::env::temp_dir().join(format!(
            "candle-gpt-oss-nonzero-{label}-{}-{}.gguf",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(candle::Error::wrap)?
                .as_nanos()
        ));
        std::fs::write(&path, bytes).map_err(candle::Error::wrap)?;
        Ok(TempGgufPath(path))
    }

    fn wire_value(bytes: &[u8], block: usize, offset: usize) -> Result<f32> {
        if offset >= VALUES_PER_BLOCK {
            bail!("test MXFP4 offset {offset} is out of range");
        }
        let block_start = block
            .checked_mul(BYTES_PER_BLOCK + 1)
            .ok_or_else(|| candle::Error::Msg("test MXFP4 block offset overflowed".into()))?;
        let byte = *bytes
            .get(block_start + 1 + offset % BYTES_PER_BLOCK)
            .ok_or_else(|| candle::Error::Msg("test MXFP4 wire block is truncated".into()))?;
        let nibble = if offset < BYTES_PER_BLOCK {
            byte & 0x0f
        } else {
            byte >> 4
        } as usize;
        let scale = *bytes
            .get(block_start)
            .ok_or_else(|| candle::Error::Msg("test MXFP4 scale is truncated".into()))?;
        Ok(crate::models::gpt_oss::mxfp4::FP4_VALUES[nibble]
            * 2f32.powi(scale as i32 - crate::models::gpt_oss::mxfp4::SCALE_BIAS))
    }

    fn reference_expert_contribution(
        gate_up_bytes: &[u8],
        split_up_bytes: Option<&[u8]>,
        down_bytes: &[u8],
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != 32 {
            bail!("test expert input width is not 32");
        }
        let mut first = vec![0.0f32; 64];
        for row in 0..64 {
            for coordinate in 0..32 {
                let (bytes, source_row) = match split_up_bytes {
                    Some(up) if row % 2 == 1 => (up, row / 2),
                    Some(_) => (gate_up_bytes, row / 2),
                    None => (gate_up_bytes, row),
                };
                first[row] += input[coordinate] * wire_value(bytes, source_row, coordinate)?;
            }
        }
        let mut activated = vec![0.0f32; 32];
        for index in 0..32 {
            let glu = first[index * 2].min(7.0);
            let linear = first[index * 2 + 1].clamp(-7.0, 7.0);
            let gate = 1.0 / (1.0 + (-1.702 * glu).exp());
            activated[index] = glu * gate * (linear + 1.0);
        }
        let mut output = vec![0.0f32; 32];
        for row in 0..32 {
            for coordinate in 0..32 {
                output[row] += activated[coordinate] * wire_value(down_bytes, row, coordinate)?;
            }
        }
        Ok(output)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label} value {index}: {actual} != {expected}"
            );
        }
    }

    #[derive(Clone, Copy)]
    enum Meta<'a> {
        String(&'a str),
        U32(u32),
        F32(f32),
        StringArray(u64),
    }

    #[derive(Clone, Copy)]
    enum Dtype {
        F32,
        Mxfp4,
    }

    struct TestTensor {
        name: String,
        shape: Vec<usize>,
        dtype: Dtype,
        bytes: Vec<u8>,
    }

    fn push_tensor(
        tensors: &mut Vec<TestTensor>,
        name: &str,
        shape: &[usize],
        dtype: Dtype,
    ) -> Result<()> {
        let elements = shape.iter().try_fold(1usize, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or_else(|| candle::Error::Msg("test tensor size overflowed".into()))
        })?;
        let byte_len = match dtype {
            Dtype::F32 => elements * 4,
            Dtype::Mxfp4 => {
                if !elements.is_multiple_of(32) {
                    bail!("test MXFP4 shape is not block aligned");
                }
                elements / 32 * 17
            }
        };
        tensors.push(TestTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
            bytes: vec![0; byte_len],
        });
        Ok(())
    }

    fn build_gguf(config: &[(&str, Meta<'_>)], tensors: &[TestTensor]) -> Result<Vec<u8>> {
        let offsets = vec![0u64; tensors.len()];
        let mut header = Vec::new();
        write_header(&mut header, config, tensors, &offsets)?;
        let data_offset = align_up(header.len() as u64, 32)? as usize;
        let mut next_offset = data_offset as u64;
        let mut actual_offsets = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            next_offset = align_up(next_offset, 32)?;
            actual_offsets.push(next_offset - data_offset as u64);
            next_offset = next_offset
                .checked_add(tensor.bytes.len() as u64)
                .ok_or_else(|| candle::Error::Msg("test GGUF data overflowed".into()))?;
        }
        header.clear();
        write_header(&mut header, config, tensors, &actual_offsets)?;
        let mut bytes = header;
        bytes.resize(data_offset, 0);
        for (tensor, offset) in tensors.iter().zip(actual_offsets) {
            let absolute = data_offset
                .checked_add(offset as usize)
                .ok_or_else(|| candle::Error::Msg("test GGUF offset overflowed".into()))?;
            bytes.resize(absolute, 0);
            bytes.extend_from_slice(&tensor.bytes);
        }
        Ok(bytes)
    }

    fn write_header(
        bytes: &mut Vec<u8>,
        metadata: &[(&str, Meta<'_>)],
        tensors: &[TestTensor],
        offsets: &[u64],
    ) -> Result<()> {
        bytes.write_all(b"GGUF")?;
        bytes.write_all(&3u32.to_le_bytes())?;
        bytes.write_all(&(tensors.len() as u64).to_le_bytes())?;
        bytes.write_all(&(metadata.len() as u64).to_le_bytes())?;
        for (key, value) in metadata {
            write_string(bytes, key)?;
            match value {
                Meta::String(value) => {
                    bytes.write_all(&8u32.to_le_bytes())?;
                    write_string(bytes, value)?;
                }
                Meta::U32(value) => {
                    bytes.write_all(&4u32.to_le_bytes())?;
                    bytes.write_all(&value.to_le_bytes())?;
                }
                Meta::F32(value) => {
                    bytes.write_all(&6u32.to_le_bytes())?;
                    bytes.write_all(&value.to_le_bytes())?;
                }
                Meta::StringArray(length) => {
                    bytes.write_all(&9u32.to_le_bytes())?;
                    bytes.write_all(&8u32.to_le_bytes())?;
                    bytes.write_all(&length.to_le_bytes())?;
                    for index in 0..*length {
                        write_string(bytes, &format!("token-{index}"))?;
                    }
                }
            }
        }
        for (tensor, offset) in tensors.iter().zip(offsets) {
            write_string(bytes, &tensor.name)?;
            bytes.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
            for dimension in tensor.shape.iter().rev() {
                bytes.write_all(&(*dimension as u64).to_le_bytes())?;
            }
            let dtype: u32 = match tensor.dtype {
                Dtype::F32 => 0,
                Dtype::Mxfp4 => 39,
            };
            bytes.write_all(&dtype.to_le_bytes())?;
            bytes.write_all(&offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
        bytes.write_all(&(value.len() as u64).to_le_bytes())?;
        bytes.write_all(value.as_bytes())?;
        Ok(())
    }
}
