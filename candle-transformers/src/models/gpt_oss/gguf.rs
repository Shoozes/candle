//! Hash-pinned GPT-OSS GGUF admission and tensor-inventory normalization.
//!
//! This is an admission boundary, not a runtime executor.  It understands the
//! GGUF directory and the MXFP4 wire type used by the selected GPT-OSS
//! artifact, retains tensor ownership and byte ranges, and reads raw tensor
//! bytes on demand.  It deliberately does not dequantize or construct a
//! Candle model.

use super::GptOssConfig;
use candle::{bail, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssGgufArtifact {
    path: PathBuf,
    sha256: String,
    config: GptOssConfig,
    context_length: usize,
    tensor_data_offset: u64,
    file_size: u64,
    tensors: Vec<GptOssGgufTensor>,
}

impl GptOssGgufArtifact {
    /// Admit the exact owner-selected artifact from a local path.
    ///
    /// This function never downloads or searches for a model.  The file is
    /// hashed before any GGUF metadata is trusted.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|error| {
            candle::Error::Msg(format!("failed to stat GPT-OSS GGUF {:?}: {error}", path))
        })?;
        if !metadata.is_file() {
            bail!("GPT-OSS GGUF path {:?} is not a regular file", path);
        }
        let file_size = metadata.len();
        validate_file_size(file_size)?;

        let mut file = File::open(path).map_err(|error| {
            candle::Error::Msg(format!("failed to open GPT-OSS GGUF {:?}: {error}", path))
        })?;
        let sha256 = sha256_reader(&mut file)?;
        if sha256 != SELECTED_GPT_OSS_GGUF_SHA256 {
            bail!(
                "GPT-OSS GGUF artifact identity mismatch: expected {}, got {}",
                SELECTED_GPT_OSS_GGUF_SHA256,
                sha256
            );
        }
        file.seek(SeekFrom::Start(0))?;
        let parsed = parse_gguf(&mut file, file_size)?;
        Self::from_parsed(path.to_path_buf(), sha256, parsed)
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
        if tensor.byte_len > MAX_RAW_TENSOR_BYTES {
            bail!(
                "GPT-OSS GGUF tensor {name:?} payload {} exceeds raw read limit {}",
                tensor.byte_len,
                MAX_RAW_TENSOR_BYTES
            );
        }
        let current_size = std::fs::metadata(&self.path)
            .map_err(|error| {
                candle::Error::Msg(format!(
                    "failed to stat admitted GPT-OSS GGUF {:?}: {error}",
                    self.path
                ))
            })?
            .len();
        if current_size != self.file_size {
            bail!(
                "admitted GPT-OSS GGUF {:?} changed size from {} to {}",
                self.path,
                self.file_size,
                current_size
            );
        }
        let byte_len = usize::try_from(tensor.byte_len).map_err(|_| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {name:?} payload does not fit in usize"
            ))
        })?;
        let absolute_offset = self
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS GGUF tensor offset overflowed".into()))?;
        let mut file = File::open(&self.path).map_err(|error| {
            candle::Error::Msg(format!(
                "failed to reopen admitted GPT-OSS GGUF {:?}: {error}",
                self.path
            ))
        })?;
        let current_sha256 = sha256_reader(&mut file)?;
        if current_sha256 != self.sha256 {
            bail!(
                "admitted GPT-OSS GGUF {:?} changed identity from {} to {}",
                self.path,
                self.sha256,
                current_sha256
            );
        }
        file.seek(SeekFrom::Start(absolute_offset))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|error| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {name:?} allocation failed: {error}"
            ))
        })?;
        bytes.resize(byte_len, 0);
        file.read_exact(&mut bytes).map_err(|error| {
            candle::Error::Msg(format!(
                "GPT-OSS GGUF tensor {name:?} is truncated while reading: {error}"
            ))
        })?;
        Ok(bytes)
    }

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
        })
    }
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

fn sha256_reader<R: Read + Seek>(reader: &mut R) -> Result<String> {
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
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
            vec![config.hidden_size, config.vocab_size]
        }
        GptOssTensorRole::OutputNorm
        | GptOssTensorRole::AttentionNorm(_)
        | GptOssTensorRole::MoeNorm(_) => vec![config.hidden_size],
        GptOssTensorRole::QkvWeight(_) => vec![config.hidden_size, qkv_width],
        GptOssTensorRole::QkvBias(_) => vec![qkv_width],
        GptOssTensorRole::QueryWeight(_) => vec![config.hidden_size, q_width],
        GptOssTensorRole::QueryBias(_) => vec![q_width],
        GptOssTensorRole::KeyWeight(_) | GptOssTensorRole::ValueWeight(_) => {
            vec![config.hidden_size, kv_width]
        }
        GptOssTensorRole::KeyBias(_) | GptOssTensorRole::ValueBias(_) => vec![kv_width],
        GptOssTensorRole::AttentionOutputWeight(_) => vec![q_width, config.hidden_size],
        GptOssTensorRole::AttentionOutputBias(_) => vec![config.hidden_size],
        GptOssTensorRole::AttentionSinks(_) => vec![config.num_attention_heads],
        GptOssTensorRole::RouterWeight(_) => vec![config.hidden_size, config.num_experts],
        GptOssTensorRole::RouterBias(_) => vec![config.num_experts],
        GptOssTensorRole::ExpertGateUpWeight(_) => {
            vec![config.hidden_size, fused_intermediate, config.num_experts]
        }
        GptOssTensorRole::ExpertGateUpBias(_) => {
            vec![fused_intermediate, config.num_experts]
        }
        GptOssTensorRole::ExpertGateWeight(_) | GptOssTensorRole::ExpertUpWeight(_) => {
            vec![
                config.hidden_size,
                config.intermediate_size,
                config.num_experts,
            ]
        }
        GptOssTensorRole::ExpertGateBias(_) | GptOssTensorRole::ExpertUpBias(_) => {
            vec![config.intermediate_size, config.num_experts]
        }
        GptOssTensorRole::ExpertDownWeight(_) => {
            vec![
                config.hidden_size,
                config.intermediate_size,
                config.num_experts,
            ]
        }
        GptOssTensorRole::ExpertDownBias(_) => vec![config.hidden_size, config.num_experts],
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
    use std::io::Write;

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
        assert_eq!(expert.shape, vec![32, 64, 2]);
        assert_eq!(expert.byte_len, 2176);
        Ok(())
    }

    #[test]
    fn accepts_converter_style_split_expert_inventory() -> Result<()> {
        let bytes = tiny_converter_style_gguf()?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        let artifact = parse_test_bytes(&bytes, &actual)?;
        assert_eq!(artifact.tensors().len(), 18);
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
        tiny_gguf_with_options(&[32, 48], false, false)
    }

    fn tiny_gguf_with_qkv_shape(qkv_shape: &[usize]) -> Result<Vec<u8>> {
        tiny_gguf_with_options(qkv_shape, false, false)
    }

    fn tiny_converter_style_gguf() -> Result<Vec<u8>> {
        tiny_gguf_with_options(&[32, 48], true, true)
    }

    fn tiny_gguf_with_options(
        qkv_shape: &[usize],
        split_experts: bool,
        converter_style_names: bool,
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
        push_tensor(&mut tensors, "token_embd.weight", &[32, 8], Dtype::F32)?;
        push_tensor(&mut tensors, "output_norm.weight", &[32], Dtype::F32)?;
        push_tensor(&mut tensors, "output.weight", &[32, 8], Dtype::F32)?;
        push_tensor(&mut tensors, "blk.0.attn_norm.weight", &[32], Dtype::F32)?;
        push_tensor(&mut tensors, "blk.0.attn_qkv.weight", qkv_shape, Dtype::F32)?;
        push_tensor(&mut tensors, "blk.0.attn_qkv.bias", &[48], Dtype::F32)?;
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
            &[32, 2],
            Dtype::F32,
        )?;
        push_tensor(&mut tensors, "blk.0.ffn_gate_inp.bias", &[2], Dtype::F32)?;
        if split_experts {
            for suffix in ["ffn_gate_exps", "ffn_up_exps"] {
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.{suffix}.weight"),
                    &[32, 32, 2],
                    Dtype::Mxfp4,
                )?;
                push_tensor(
                    &mut tensors,
                    &format!("blk.0.{suffix}.bias"),
                    &[32, 2],
                    Dtype::F32,
                )?;
            }
        } else {
            push_tensor(
                &mut tensors,
                "blk.0.ffn_gate_up_exps.weight",
                &[32, 64, 2],
                Dtype::Mxfp4,
            )?;
            push_tensor(
                &mut tensors,
                "blk.0.ffn_gate_up_exps.bias",
                &[64, 2],
                Dtype::F32,
            )?;
        }
        push_tensor(
            &mut tensors,
            "blk.0.ffn_down_exps.weight",
            &[32, 32, 2],
            Dtype::Mxfp4,
        )?;
        push_tensor(
            &mut tensors,
            "blk.0.ffn_down_exps.bias",
            &[32, 2],
            Dtype::F32,
        )?;
        build_gguf(&config, &tensors)
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
