//! Bounded local Hugging Face safetensors discovery and header inspection.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const SINGLE_WEIGHTS: &str = "model.safetensors";
const WEIGHTS_INDEX: &str = "model.safetensors.index.json";
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SAFETENSORS_FILE_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MAX_TOTAL_SAFETENSORS_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_SAFETENSORS_SHARDS: usize = 1_024;
const MAX_SAFETENSORS_TENSORS: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub nbytes: u64,
    pub shard: String,
}

#[derive(Clone, Debug)]
pub struct ResolvedCheckpoint {
    pub root: PathBuf,
    pub index_file: Option<PathBuf>,
    pub weight_files: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub indexed: bool,
    pub total_file_bytes: u64,
}

impl ResolvedCheckpoint {
    pub fn required_file(&self, name: &str, label: &str, max_bytes: u64) -> Result<PathBuf> {
        canonical_bounded_file(&self.root, name, label, max_bytes)
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    #[serde(default)]
    metadata: Option<SafetensorsIndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

pub fn resolve_checkpoint(model_dir: impl AsRef<Path>) -> Result<ResolvedCheckpoint> {
    let requested_root = model_dir.as_ref();
    let root = std::fs::canonicalize(requested_root).with_context(|| {
        format!(
            "resolving native LFM2-VL model directory {}",
            requested_root.display()
        )
    })?;
    if !root.is_dir() {
        bail!(
            "native LFM2-VL model path {} is not a directory",
            root.display()
        )
    }

    let single_path = root.join(SINGLE_WEIGHTS);
    let index_path = root.join(WEIGHTS_INDEX);
    let has_single = single_path
        .try_exists()
        .with_context(|| format!("checking {}", single_path.display()))?;
    let has_index = index_path
        .try_exists()
        .with_context(|| format!("checking {}", index_path.display()))?;
    match (has_single, has_index) {
        (true, true) => bail!(
            "native checkpoint contains both {SINGLE_WEIGHTS} and {WEIGHTS_INDEX}; remove the ambiguity"
        ),
        (false, false) => bail!(
            "native checkpoint requires either {SINGLE_WEIGHTS} or {WEIGHTS_INDEX}"
        ),
        (true, false) => resolve_single(&root),
        (false, true) => resolve_indexed(&root),
    }
}

fn resolve_single(root: &Path) -> Result<ResolvedCheckpoint> {
    let path = canonical_bounded_file(
        root,
        SINGLE_WEIGHTS,
        "native safetensors weights",
        MAX_SAFETENSORS_FILE_BYTES,
    )?;
    let (tensors, file_bytes) = inspect_safetensors_header(&path)?;
    Ok(ResolvedCheckpoint {
        root: root.to_path_buf(),
        index_file: None,
        weight_files: vec![path],
        tensors,
        indexed: false,
        total_file_bytes: file_bytes,
    })
}

fn resolve_indexed(root: &Path) -> Result<ResolvedCheckpoint> {
    let index_path = canonical_bounded_file(
        root,
        WEIGHTS_INDEX,
        "native safetensors index",
        MAX_INDEX_BYTES,
    )?;
    let index_json = read_bounded_utf8(&index_path, "native safetensors index", MAX_INDEX_BYTES)?;
    let index: SafetensorsIndex =
        serde_json::from_str(&index_json).context("parsing native safetensors index")?;
    if index.weight_map.is_empty() || index.weight_map.len() > MAX_SAFETENSORS_TENSORS {
        bail!(
            "native safetensors index tensor count {} is outside 1..={MAX_SAFETENSORS_TENSORS}",
            index.weight_map.len()
        )
    }

    let mut shard_names = BTreeSet::new();
    for (tensor_name, shard_name) in &index.weight_map {
        validate_tensor_name(tensor_name)?;
        validate_shard_name(shard_name)?;
        shard_names.insert(shard_name.clone());
    }
    if shard_names.is_empty() || shard_names.len() > MAX_SAFETENSORS_SHARDS {
        bail!(
            "native safetensors shard count {} is outside 1..={MAX_SAFETENSORS_SHARDS}",
            shard_names.len()
        )
    }

    let mut tensors = BTreeMap::new();
    let mut weight_files = Vec::new();
    weight_files
        .try_reserve_exact(shard_names.len())
        .map_err(|_| anyhow::anyhow!("native safetensors shard-path allocation failed"))?;
    let mut total_file_bytes = 0u64;
    for shard_name in shard_names {
        let path = canonical_bounded_file(
            root,
            &shard_name,
            "native safetensors shard",
            MAX_SAFETENSORS_FILE_BYTES,
        )?;
        let (shard_tensors, file_bytes) = inspect_safetensors_header(&path)?;
        total_file_bytes = total_file_bytes
            .checked_add(file_bytes)
            .ok_or_else(|| anyhow::anyhow!("native safetensors total byte size overflow"))?;
        if total_file_bytes > MAX_TOTAL_SAFETENSORS_BYTES {
            bail!(
                "native safetensors total size {total_file_bytes} exceeds {MAX_TOTAL_SAFETENSORS_BYTES} bytes"
            )
        }
        for (tensor_name, info) in shard_tensors {
            if tensors.insert(tensor_name.clone(), info).is_some() {
                bail!("native safetensors tensor {tensor_name:?} occurs in multiple shards")
            }
        }
        weight_files.push(path);
    }

    if tensors.len() > MAX_SAFETENSORS_TENSORS {
        bail!(
            "native safetensors tensor count {} exceeds {MAX_SAFETENSORS_TENSORS}",
            tensors.len()
        )
    }
    let actual_names: BTreeSet<_> = tensors.keys().cloned().collect();
    let indexed_names: BTreeSet<_> = index.weight_map.keys().cloned().collect();
    if actual_names != indexed_names {
        let missing: Vec<_> = indexed_names.difference(&actual_names).cloned().collect();
        let unindexed: Vec<_> = actual_names.difference(&indexed_names).cloned().collect();
        bail!(
            "native safetensors index inventory mismatch; missing={missing:?}, unindexed={unindexed:?}"
        )
    }
    for (tensor_name, expected_shard) in &index.weight_map {
        let actual_shard = &tensors
            .get(tensor_name)
            .ok_or_else(|| anyhow::anyhow!("indexed tensor {tensor_name:?} disappeared"))?
            .shard;
        if actual_shard != expected_shard {
            bail!(
                "native safetensors index maps {tensor_name:?} to {expected_shard:?}, but it is stored in {actual_shard:?}"
            )
        }
    }
    if let Some(declared_total_size) = index.metadata.and_then(|metadata| metadata.total_size) {
        let actual_total_size = tensors
            .values()
            .try_fold(0u64, |total, info| total.checked_add(info.nbytes));
        let actual_total_size = actual_total_size
            .ok_or_else(|| anyhow::anyhow!("native safetensors payload byte total overflow"))?;
        if declared_total_size != actual_total_size {
            bail!(
                "native safetensors index total_size {declared_total_size} does not match payload bytes {actual_total_size}"
            )
        }
    }

    Ok(ResolvedCheckpoint {
        root: root.to_path_buf(),
        index_file: Some(index_path),
        weight_files,
        tensors,
        indexed: true,
        total_file_bytes,
    })
}

fn inspect_safetensors_header(path: &Path) -> Result<(BTreeMap<String, TensorInfo>, u64)> {
    let mut file = File::open(path)
        .with_context(|| format!("opening native safetensors file {}", path.display()))?;
    let file_bytes = file
        .metadata()
        .with_context(|| format!("inspecting native safetensors file {}", path.display()))?
        .len();
    if !(8..=MAX_SAFETENSORS_FILE_BYTES).contains(&file_bytes) {
        bail!(
            "native safetensors file {} size {file_bytes} is outside 8..={MAX_SAFETENSORS_FILE_BYTES}",
            path.display()
        )
    }
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("reading native safetensors prefix {}", path.display()))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        bail!(
            "native safetensors header length {header_len} is outside 1..={MAX_SAFETENSORS_HEADER_BYTES}"
        )
    }
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("native safetensors header length overflow"))?;
    if header_end > file_bytes {
        bail!("native safetensors header ends at {header_end}, beyond file size {file_bytes}")
    }
    let header_capacity = usize::try_from(header_len)
        .context("native safetensors header length does not fit usize")?;
    let mut header_bytes = Vec::new();
    header_bytes
        .try_reserve_exact(header_capacity)
        .map_err(|_| anyhow::anyhow!("native safetensors header allocation failed"))?;
    header_bytes.resize(header_capacity, 0);
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("reading native safetensors header {}", path.display()))?;
    let header_value: serde_json::Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("parsing native safetensors header {}", path.display()))?;
    let mut header = header_value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("native safetensors header must be a JSON object"))?;
    if let Some(metadata) = header.remove("__metadata__") {
        let metadata = metadata
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("native safetensors metadata must be an object"))?;
        if metadata
            .iter()
            .any(|(key, value)| key.is_empty() || !value.is_string())
        {
            bail!("native safetensors metadata must contain only non-empty string entries")
        }
    }
    if header.is_empty() || header.len() > MAX_SAFETENSORS_TENSORS {
        bail!(
            "native safetensors tensor count {} is outside 1..={MAX_SAFETENSORS_TENSORS}",
            header.len()
        )
    }

    let shard = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("native safetensors shard name is not UTF-8"))?
        .to_owned();
    let data_size = file_bytes - header_end;
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(header.len())
        .map_err(|_| anyhow::anyhow!("native safetensors range allocation failed"))?;
    let mut tensors = BTreeMap::new();
    for (name, value) in header {
        validate_tensor_name(&name)?;
        let info = value.as_object().ok_or_else(|| {
            anyhow::anyhow!("native safetensors tensor {name:?} metadata must be an object")
        })?;
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("native safetensors tensor {name:?} lacks dtype"))?;
        let element_size = dense_float_size(dtype).ok_or_else(|| {
            anyhow::anyhow!(
                "native safetensors tensor {name:?} has unsupported model dtype {dtype:?}; expected F32, F16, or BF16"
            )
        })?;
        let raw_shape = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .filter(|shape| !shape.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("native safetensors tensor {name:?} has an invalid shape")
            })?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(raw_shape.len())
            .map_err(|_| anyhow::anyhow!("native safetensors shape allocation failed"))?;
        let mut element_count = 1u64;
        for dimension in raw_shape {
            let dimension = dimension
                .as_u64()
                .filter(|&dimension| dimension > 0)
                .ok_or_else(|| {
                    anyhow::anyhow!("native safetensors tensor {name:?} has an invalid shape")
                })?;
            element_count = element_count.checked_mul(dimension).ok_or_else(|| {
                anyhow::anyhow!("native safetensors tensor {name:?} shape overflows")
            })?;
            shape.push(
                usize::try_from(dimension)
                    .context("native safetensors shape does not fit usize")?,
            );
        }
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                anyhow::anyhow!("native safetensors tensor {name:?} has invalid data offsets")
            })?;
        let start = offsets
            .first()
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                anyhow::anyhow!("native safetensors tensor {name:?} has invalid start offset")
            })?;
        let end = offsets
            .get(1)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                anyhow::anyhow!("native safetensors tensor {name:?} has invalid end offset")
            })?;
        if start > end || end > data_size {
            bail!(
                "native safetensors tensor {name:?} offsets [{start}, {end}] exceed payload size {data_size}"
            )
        }
        let nbytes = end - start;
        let expected_nbytes = element_count
            .checked_mul(element_size)
            .ok_or_else(|| anyhow::anyhow!("native safetensors tensor {name:?} size overflows"))?;
        if nbytes != expected_nbytes {
            bail!(
                "native safetensors tensor {name:?} stores {nbytes} bytes, expected {expected_nbytes}"
            )
        }
        ranges.push((start, end, name.clone()));
        tensors.insert(
            name,
            TensorInfo {
                dtype: dtype.to_owned(),
                shape,
                nbytes,
                shard: shard.clone(),
            },
        );
    }

    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0u64;
    for (start, end, name) in ranges {
        if start != previous_end {
            let relation = if start < previous_end {
                "overlaps another tensor"
            } else {
                "leaves a payload gap"
            };
            bail!("native safetensors tensor {name:?} {relation}")
        }
        previous_end = end;
    }
    if previous_end != data_size {
        bail!(
            "native safetensors file {} has {} unclaimed payload bytes",
            path.display(),
            data_size - previous_end
        )
    }
    Ok((tensors, file_bytes))
}

pub fn read_bounded_utf8(path: &Path, label: &str, max_bytes: u64) -> Result<String> {
    let bytes = read_bounded_bytes(path, label, max_bytes)?;
    String::from_utf8(bytes).with_context(|| format!("{label} is not UTF-8"))
}

pub fn inspect_bounded_file(path: &Path, label: &str, max_bytes: u64) -> Result<PathBuf> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolving {label} at {}", path.display()))?;
    validate_bounded_file(&canonical, label, max_bytes)?;
    Ok(canonical)
}

fn canonical_bounded_file(root: &Path, name: &str, label: &str, max_bytes: u64) -> Result<PathBuf> {
    validate_relative_name(name, label)?;
    let requested = root.join(name);
    let canonical = std::fs::canonicalize(&requested)
        .with_context(|| format!("resolving {label} at {}", requested.display()))?;
    if !canonical.starts_with(root) {
        bail!("{label} resolves outside the native model directory")
    }
    validate_bounded_file(&canonical, label, max_bytes)?;
    Ok(canonical)
}

fn validate_bounded_file(path: &Path, label: &str, max_bytes: u64) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("inspecting {label} at {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{label} at {} is not a regular file", path.display())
    }
    let size = metadata.len();
    if size == 0 || size > max_bytes {
        bail!(
            "{label} at {} has {size} bytes, outside 1..={max_bytes}",
            path.display()
        )
    }
    Ok(size)
}

fn read_bounded_bytes(path: &Path, label: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let size = validate_bounded_file(path, label, max_bytes)?;
    let capacity =
        usize::try_from(size).with_context(|| format!("{label} size does not fit usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| anyhow::anyhow!("{label} allocation failed"))?;
    let file = File::open(path).with_context(|| format!("opening {label} {}", path.display()))?;
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {label} {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!("{label} grew beyond {max_bytes} bytes while reading")
    }
    Ok(bytes)
}

fn validate_shard_name(name: &str) -> Result<()> {
    validate_relative_name(name, "native safetensors shard name")?;
    if !name.ends_with(".safetensors") {
        bail!("native safetensors shard {name:?} must end in .safetensors")
    }
    Ok(())
}

fn validate_relative_name(name: &str, label: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains(':')
        || Path::new(name).file_name().and_then(|part| part.to_str()) != Some(name)
    {
        bail!("{label} {name:?} is not a safe local filename")
    }
    Ok(())
}

fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() || name.trim() != name || name.contains('\0') {
        bail!("native safetensors tensor name {name:?} is invalid")
    }
    Ok(())
}

fn dense_float_size(dtype: &str) -> Option<u64> {
    match dtype {
        "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        _ => None,
    }
}
