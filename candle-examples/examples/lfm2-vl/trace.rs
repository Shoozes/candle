//! Bounded native tensor-trace export for the official CPU/F32 parity lane.

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::lfm2_vl::{
    Lfm2VlDecodeTrace, Lfm2VlImageTrace, Lfm2VlPrefillTrace, ProcessedVisionBatch,
};
use image::DynamicImage;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use super::runner::InferenceReport;

const TRACE_FORMAT: &str = "lfm2-vl-reference-bundle";
const TRACE_MODE: &str = "native-trace";
const TRACE_SCHEMA_VERSION: u32 = 1;

#[cfg(target_os = "linux")]
fn rename_directory_no_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_uint};
    use std::os::unix::ffi::OsStrExt;

    unsafe extern "C" {
        fn renameat2(
            olddirfd: c_int,
            oldpath: *const c_char,
            newdirfd: c_int,
            newpath: *const c_char,
            flags: c_uint,
        ) -> c_int;
    }

    const AT_FDCWD: c_int = -100;
    const RENAME_NOREPLACE: c_uint = 1;
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
    })?;
    let result = unsafe {
        renameat2(
            AT_FDCWD,
            source.as_ptr(),
            AT_FDCWD,
            destination.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_os = "windows")]
fn rename_directory_no_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn rename_directory_no_replace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "native trace no-clobber publication is supported only on Windows and Linux",
    ))
}

pub struct NativeTraceCapture {
    pub image: Lfm2VlImageTrace,
    pub prefill: Option<Lfm2VlPrefillTrace>,
    pub decode_input_ids: Vec<Tensor>,
    pub decode: Vec<Lfm2VlDecodeTrace>,
}

impl NativeTraceCapture {
    pub fn new(image: Lfm2VlImageTrace) -> Self {
        Self {
            image,
            prefill: None,
            decode_input_ids: Vec::new(),
            decode: Vec::new(),
        }
    }
}

pub fn write_native_trace(
    output: &Path,
    report: &InferenceReport,
    processed: &ProcessedVisionBatch,
    input_ids: &Tensor,
    image: Option<&DynamicImage>,
    capture: NativeTraceCapture,
) -> Result<()> {
    let output = resolve_output(output)?;
    let image = image.ok_or_else(|| anyhow::anyhow!("native trace requires a decoded image"))?;
    let prefill = capture
        .prefill
        .ok_or_else(|| anyhow::anyhow!("native trace did not capture prefill tensors"))?;
    if capture.decode_input_ids.len() != capture.decode.len() {
        bail!(
            "native trace decode input/trace count mismatch: {} versus {}",
            capture.decode_input_ids.len(),
            capture.decode.len()
        )
    }

    let cpu = Device::Cpu;
    let mut tensors = BTreeMap::<String, Tensor>::new();
    tensors.insert(
        "input.pixel_values".to_owned(),
        cpu_f32(&processed.pixel_values)?,
    );
    tensors.insert(
        "input.pixel_attention_mask".to_owned(),
        cpu_i32(&processed.pixel_attention_mask)?,
    );
    tensors.insert(
        "input.spatial_shapes".to_owned(),
        cpu_i64(&processed.spatial_shapes)?,
    );
    tensors.insert("input.input_ids".to_owned(), cpu_i64(input_ids)?);
    let (batch_size, sequence_length) = input_ids.dims2()?;
    tensors.insert(
        "input.attention_mask".to_owned(),
        Tensor::ones((batch_size, sequence_length), DType::I64, &cpu)?,
    );
    tensors.insert(
        "input.projector_crop_ranges".to_owned(),
        projector_crop_ranges(&capture.image.projector.input, &cpu)?,
    );
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    tensors.insert(
        "input.image_rgb_u8".to_owned(),
        Tensor::from_vec(
            rgb.into_raw(),
            (height as usize, width as usize, 3usize),
            &cpu,
        )?,
    );

    insert_image_trace(&mut tensors, capture.image)?;
    tensors.insert(
        "stage.text.embeddings".to_owned(),
        cpu_f32(&prefill.input_embeddings)?,
    );
    tensors.insert(
        "stage.multimodal.merged_embeddings".to_owned(),
        cpu_f32(&prefill.merged_embeddings)?,
    );
    tensors.insert(
        "stage.language.hidden_states".to_owned(),
        cpu_f32(&prefill.hidden_states)?,
    );
    tensors.insert(
        "stage.language.prefill_logits".to_owned(),
        cpu_f32(&prefill.logits)?,
    );

    let decode_token_ids = stack_decode_ids(&capture.decode_input_ids, &cpu)?;
    let decode_logits = stack_decode_logits(&capture.decode, &cpu, prefill.logits.dim(2)?)?;
    tensors.insert("input.decode_token_ids".to_owned(), decode_token_ids);
    tensors.insert("stage.language.decode_logits".to_owned(), decode_logits);

    let metadata = json!({
        "schema_version": TRACE_SCHEMA_VERSION,
        "mode": TRACE_MODE,
        "device": "cpu",
        "dtype": "float32",
        "seed": 0,
        "weights_serialized": false,
        "model_inputs_reverified": true,
        "backend": report.backend,
        "model_inputs": report.model_inputs,
        "prompt": report.prompt,
        "expanded_prompt": report.expanded_prompt,
        "input_ids": report.input_ids,
        "image_files": report.image_files,
        "processed_images": report.processed_images,
        "processed_crops": report.processed_crops,
        "image_spans": report.image_spans,
        "context_length": report.context_length,
        "vision_batch_size": report.vision_batch_size,
        "max_new_tokens": report.max_new_tokens,
        "generation": report.generation,
        "cache_reset_exact": report.cache_reset_exact,
        "trace_contract": {
            "cpu_f32_only": true,
            "single_crop": true,
            "max_input_tokens": 4096,
            "max_image_patches": 1024,
            "max_new_tokens": 32,
        },
    });
    let metadata_bytes = json_bytes(&metadata)?;

    let tensor_inventory = tensor_inventory(&tensors)?;
    let staging = create_staging_dir(&output)?;
    let result = write_staging(
        &staging,
        &tensors,
        &metadata_bytes,
        tensor_inventory,
        report,
    );
    match result {
        Ok(()) => {
            if let Err(error) = rename_directory_no_replace(&staging, &output) {
                let _ = fs::remove_dir_all(&staging);
                return Err(error).with_context(|| {
                    format!("publishing native trace directory {}", output.display())
                });
            }
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn insert_image_trace(
    tensors: &mut BTreeMap<String, Tensor>,
    trace: Lfm2VlImageTrace,
) -> Result<()> {
    tensors.insert(
        "stage.vision.patch_embedding".to_owned(),
        cpu_f32(&trace.vision_patch_embedding)?,
    );
    tensors.insert(
        "stage.vision.resized_position_embedding".to_owned(),
        cpu_f32(&trace.vision_resized_position_embedding)?,
    );
    tensors.insert(
        "stage.vision.embeddings_with_resized_position".to_owned(),
        cpu_f32(&trace.vision_embeddings_with_position)?,
    );
    for (index, layer) in trace.vision_encoder_layers.iter().enumerate() {
        tensors.insert(
            format!("stage.vision.encoder_layer.{index}"),
            cpu_f32(layer)?,
        );
    }
    tensors.insert(
        "stage.vision.post_layernorm".to_owned(),
        cpu_f32(&trace.vision_last_hidden_state)?,
    );
    tensors.insert(
        "stage.vision.last_hidden_state".to_owned(),
        cpu_f32(&trace.vision_last_hidden_state)?,
    );

    let projector = trace.projector;
    tensors.insert(
        "stage.projector.input".to_owned(),
        cpu_f32(&projector.input)?,
    );
    tensors.insert(
        "stage.projector.pixel_unshuffle".to_owned(),
        cpu_f32(&projector.pixel_unshuffle)?,
    );
    if let Some(layer_norm) = projector.layer_norm {
        tensors.insert(
            "stage.projector.layer_norm".to_owned(),
            cpu_f32(&layer_norm)?,
        );
    }
    tensors.insert(
        "stage.projector.linear_1".to_owned(),
        cpu_f32(&projector.linear_1)?,
    );
    tensors.insert(
        "stage.projector.activation".to_owned(),
        cpu_f32(&projector.activation)?,
    );
    tensors.insert(
        "stage.projector.linear_2".to_owned(),
        cpu_f32(&projector.linear_2)?,
    );
    tensors.insert(
        "stage.projector.output".to_owned(),
        cpu_f32(&projector.output)?,
    );
    Ok(())
}

fn projector_crop_ranges(projector_input: &Tensor, device: &Device) -> Result<Tensor> {
    let (valid_patches, _) = projector_input
        .dims2()
        .context("native trace projector input must be rank two")?;
    let valid_patches = i64::try_from(valid_patches)
        .context("native trace projector input patch count exceeds i64")?;
    Tensor::from_vec(vec![0i64, valid_patches], (1usize, 2usize), device)
        .map_err(anyhow::Error::from)
}

fn stack_decode_ids(input_ids: &[Tensor], device: &Device) -> Result<Tensor> {
    if input_ids.is_empty() {
        return Tensor::from_vec(Vec::<i64>::new(), (1usize, 0usize), device)
            .map_err(anyhow::Error::from);
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(input_ids.len())
        .map_err(|_| anyhow::anyhow!("native trace decode-id allocation failed"))?;
    for input in input_ids {
        rows.push(cpu_i64(input)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&refs, 1).map_err(anyhow::Error::from)
}

fn stack_decode_logits(
    traces: &[Lfm2VlDecodeTrace],
    device: &Device,
    vocab_size: usize,
) -> Result<Tensor> {
    if traces.is_empty() {
        return Tensor::zeros((1usize, 0usize, vocab_size), DType::F32, device)
            .map_err(anyhow::Error::from);
    }
    let mut logits = Vec::new();
    logits
        .try_reserve_exact(traces.len())
        .map_err(|_| anyhow::anyhow!("native trace decode-logit allocation failed"))?;
    for trace in traces {
        let value = cpu_f32(&trace.logits)?;
        if value.rank() != 3 {
            bail!("native trace decode logits must have rank three")
        }
        logits.push(value);
    }
    let refs: Vec<&Tensor> = logits.iter().collect();
    Tensor::cat(&refs, 1).map_err(anyhow::Error::from)
}

fn cpu_f32(tensor: &Tensor) -> Result<Tensor> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)
        .map_err(anyhow::Error::from)
}

fn cpu_i64(tensor: &Tensor) -> Result<Tensor> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)
        .map_err(anyhow::Error::from)
}

fn cpu_i32(tensor: &Tensor) -> Result<Tensor> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I32)
        .map_err(anyhow::Error::from)
}

fn resolve_output(path: &Path) -> Result<PathBuf> {
    let current = std::env::current_dir().context("resolving current directory")?;
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        current.join(path)
    };
    let path = path
        .canonicalize()
        .or_else(|_| Ok::<PathBuf, std::io::Error>(path.clone()))?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow::anyhow!("candle-examples has no repository parent"))?
        .canonicalize()
        .context("resolving repository root")?;
    if path == repo || path.starts_with(&repo) {
        bail!(
            "native trace output must be outside the repository: {}",
            path.display()
        )
    }
    if path.exists() {
        bail!("native trace output already exists: {}", path.display())
    }
    Ok(path)
}

fn create_staging_dir(output: &Path) -> Result<PathBuf> {
    let parent = output
        .parent()
        .ok_or_else(|| anyhow::anyhow!("native trace output has no parent directory"))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating native trace parent {}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("native trace output name is not valid UTF-8"))?;
    let staging = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if staging.exists() {
        bail!(
            "native trace staging path already exists: {}",
            staging.display()
        )
    }
    fs::create_dir(&staging)
        .with_context(|| format!("creating native trace staging {}", staging.display()))?;
    Ok(staging)
}

fn write_staging(
    staging: &Path,
    tensors: &BTreeMap<String, Tensor>,
    metadata_bytes: &[u8],
    tensor_inventory: Map<String, Value>,
    report: &InferenceReport,
) -> Result<()> {
    let tensor_path = staging.join("tensors.safetensors");
    let metadata_path = staging.join("metadata.json");
    let manifest_path = staging.join("manifest.json");
    let tensor_map: HashMap<_, _> = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    candle::safetensors::save(&tensor_map, &tensor_path)
        .map_err(anyhow::Error::from)
        .context("writing native trace tensors")?;
    fs::write(&metadata_path, metadata_bytes)
        .with_context(|| format!("writing native trace metadata {}", metadata_path.display()))?;
    let tensor_sha256 = sha256_file(&tensor_path)?;
    let metadata_sha256 = sha256_file(&metadata_path)?;
    let manifest = json!({
        "format": TRACE_FORMAT,
        "mode": TRACE_MODE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "device": "cpu",
        "dtype": "float32",
        "weights_serialized": false,
        "model_inputs_reverified": true,
        "tensor_file": "tensors.safetensors",
        "metadata_file": "metadata.json",
        "tensor_sha256": tensor_sha256,
        "metadata_sha256": metadata_sha256,
        "tensor_count": tensors.len(),
        "tensor_inventory": tensor_inventory,
        "backend": report.backend,
    });
    fs::write(&manifest_path, json_bytes(&manifest)?)
        .with_context(|| format!("writing native trace manifest {}", manifest_path.display()))?;
    Ok(())
}

fn trace_dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::U8 => Ok("uint8"),
        DType::I32 => Ok("int32"),
        DType::I64 => Ok("int64"),
        DType::F32 => Ok("float32"),
        unsupported => bail!("native trace has unsupported tensor dtype {unsupported:?}"),
    }
}

fn tensor_inventory(tensors: &BTreeMap<String, Tensor>) -> Result<Map<String, Value>> {
    tensors
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.clone(),
                json!({
                    "dtype": trace_dtype_name(tensor.dtype())?,
                    "shape": tensor.dims(),
                }),
            ))
        })
        .collect()
}

fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(anyhow::Error::from)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("opening trace file for hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing trace file {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let number = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "candle-lfm2-vl-trace-{}-{number}",
                std::process::id()
            ));
            fs::create_dir(&path)
                .with_context(|| format!("creating trace test directory {}", path.display()))?;
            Ok(Self(path))
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn trace_publication_does_not_replace_a_racing_directory() -> Result<()> {
        let root = TestDir::new()?;
        let source = root.0.join("staging");
        let destination = root.0.join("trace");
        fs::create_dir(&source)?;
        fs::write(source.join("incoming.txt"), b"incoming")?;
        fs::create_dir(&destination)?;
        fs::write(destination.join("owner.txt"), b"owner")?;

        let error = rename_directory_no_replace(&source, &destination).unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
        ));
        assert!(source.join("incoming.txt").is_file());
        assert_eq!(fs::read(destination.join("owner.txt"))?, b"owner");
        Ok(())
    }
}
