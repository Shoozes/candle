//! Local-only loading for quantized LFM2 text with split or GGUF MMProj.

use super::{
    Lfm2VlProcessor, Lfm2VlProcessorConfig, Lfm2VlPrompt, Lfm2VlSpecialTokens,
    ProcessorConfigPatch, PromptOptions,
};
use candle::quantized::gguf_file;
use candle::{DType, Device, Result};
use candle_transformers::models::lfm2_vl::{Mmproj, QuantizedLfm2VlModel};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokenizers::Tokenizer;

const MAX_PROCESSOR_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 512 * 1024 * 1024;

/// Local source for the vision tower and multimodal projector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lfm2VlMmprojSource<'a> {
    /// A Candle split bundle containing `mmproj.safetensors`, `mmproj.json`,
    /// and `processor_config.json`.
    SplitDirectory(&'a Path),
    /// A llama.cpp-compatible MMProj GGUF file.
    GgufFile(&'a Path),
}

/// Requested execution policy for a local MMProj source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lfm2VlMmprojExecution {
    /// Select native Q8 execution when the GGUF supports it, otherwise dense.
    #[default]
    Auto,
    /// Dequantize GGUF tensors and execute with dense operators.
    Dense,
    /// Require native Q8_0 execution with F32 activations.
    Q8,
}

impl FromStr for Lfm2VlMmprojExecution {
    type Err = candle::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "dense" | "dequantize" => Ok(Self::Dense),
            "q8" | "q8_0" | "native-q8" => Ok(Self::Q8),
            _ => {
                candle::bail!("unsupported MMProj execution {value:?}; expected auto, dense, or q8")
            }
        }
    }
}

impl fmt::Display for Lfm2VlMmprojExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Auto => "auto",
            Self::Dense => "dense",
            Self::Q8 => "q8",
        })
    }
}

/// Complete, explicit inputs for one local hybrid model load.
#[derive(Clone, Copy, Debug)]
pub struct Lfm2VlHybridLoadOptions<'a> {
    pub text_gguf: &'a Path,
    pub mmproj: Lfm2VlMmprojSource<'a>,
    pub tokenizer: &'a Path,
    pub processor_config: Option<&'a Path>,
    pub mmproj_execution: Lfm2VlMmprojExecution,
    pub vision_dtype: DType,
    pub vision_device: &'a Device,
    pub text_device: &'a Device,
}

/// Loaded hybrid runtime plus the exact local files consumed to build it.
pub struct LoadedLfm2VlHybrid {
    pub model: QuantizedLfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
    pub consumed_files: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmprojLoadPlan {
    SplitDense,
    GgufAuto,
    GgufDense,
    GgufQ8,
}

/// Load a complete hybrid LFM2-VL runtime from explicit local files.
///
/// This function performs no network access, model discovery, or fallback.
/// Pairing, tokenizer, processor, dtype, and execution-policy mismatches fail
/// before a runtime is returned.
pub fn load_lfm2_vl_hybrid(options: Lfm2VlHybridLoadOptions<'_>) -> Result<LoadedLfm2VlHybrid> {
    let mmproj_plan = mmproj_load_plan(
        options.mmproj,
        options.mmproj_execution,
        options.vision_dtype,
    )?;

    // Parse small public metadata before opening either large model payload.
    let tokenizer = load_tokenizer(options.tokenizer)?;
    let tokenizer_image_token_id =
        Lfm2VlSpecialTokens::resolve(&tokenizer, None, 0, false)?.image_token_id;
    let processor_patch = options
        .processor_config
        .map(|path| {
            let json = read_processor_config(path)?;
            ProcessorConfigPatch::from_json(&json)
        })
        .transpose()?;

    let mut text_file = File::open(options.text_gguf).map_err(|error| {
        candle::Error::Msg(format!(
            "cannot open text GGUF {}: {error}",
            options.text_gguf.display()
        ))
    })?;
    let content =
        gguf_file::Content::read(&mut text_file).map_err(|err| err.with_path(options.text_gguf))?;
    let text = ModelWeights::from_gguf(content, &mut text_file, options.text_device)?;

    let mmproj = match options.mmproj {
        Lfm2VlMmprojSource::SplitDirectory(path) => {
            if mmproj_plan != MmprojLoadPlan::SplitDense {
                candle::bail!("internal MMProj load plan does not match split input")
            }
            Mmproj::load(path, options.vision_dtype, options.vision_device)?
        }
        Lfm2VlMmprojSource::GgufFile(path) => match mmproj_plan {
            MmprojLoadPlan::GgufAuto => Mmproj::load_gguf_auto(
                path,
                options.vision_dtype,
                options.vision_device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::GgufDense => Mmproj::load_gguf(
                path,
                options.vision_dtype,
                options.vision_device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::GgufQ8 => Mmproj::load_gguf_q8(
                path,
                options.vision_dtype,
                options.vision_device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::SplitDense => {
                candle::bail!("internal MMProj load plan does not match GGUF input")
            }
        },
    };

    let processor_config = Lfm2VlProcessorConfig::from_mmproj_metadata_with_processor(
        &mmproj.metadata,
        processor_patch.as_ref(),
    )?;
    let prompt = Lfm2VlPrompt::from_processor_config(
        tokenizer,
        Some(mmproj.metadata.image_token_id),
        &processor_config,
        PromptOptions {
            use_image_special_tokens: mmproj.metadata.use_image_special_tokens,
            context_length: Some(text.metadata().context_length),
        },
    )?;
    let image_token_id = prompt.special_tokens().image_token_id;
    let model = QuantizedLfm2VlModel::new(
        text,
        mmproj,
        processor_config.encoder_patch_size,
        processor_config.downsample_factor,
        image_token_id,
    )?;
    let processor = Lfm2VlProcessor::from_config(&processor_config)?;
    let consumed_files = hybrid_consumed_files(
        options.text_gguf,
        options.mmproj,
        options.tokenizer,
        options.processor_config,
    )?;

    Ok(LoadedLfm2VlHybrid {
        model,
        processor,
        prompt,
        consumed_files,
    })
}

fn hybrid_consumed_files(
    text_gguf: &Path,
    mmproj: Lfm2VlMmprojSource<'_>,
    tokenizer: &Path,
    processor_config: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    files
        .try_reserve_exact(6)
        .map_err(|_| candle::Error::Msg("allocating hybrid consumed-file inventory".into()))?;
    files.push(text_gguf.to_path_buf());
    match mmproj {
        Lfm2VlMmprojSource::SplitDirectory(path) => {
            files.push(path.join("mmproj.safetensors"));
            files.push(path.join("mmproj.json"));
            files.push(path.join("processor_config.json"));
        }
        Lfm2VlMmprojSource::GgufFile(path) => files.push(path.to_path_buf()),
    }
    files.push(tokenizer.to_path_buf());
    if let Some(path) = processor_config {
        let path = path.to_path_buf();
        if !files.contains(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

fn mmproj_load_plan(
    source: Lfm2VlMmprojSource<'_>,
    execution: Lfm2VlMmprojExecution,
    vision_dtype: DType,
) -> Result<MmprojLoadPlan> {
    if execution == Lfm2VlMmprojExecution::Q8 && vision_dtype != DType::F32 {
        candle::bail!(
            "native Q8 MMProj execution requires F32 activations, received {vision_dtype:?}"
        )
    }
    match (source, execution) {
        (
            Lfm2VlMmprojSource::SplitDirectory(_),
            Lfm2VlMmprojExecution::Auto | Lfm2VlMmprojExecution::Dense,
        ) => Ok(MmprojLoadPlan::SplitDense),
        (Lfm2VlMmprojSource::SplitDirectory(_), Lfm2VlMmprojExecution::Q8) => {
            candle::bail!("native Q8 MMProj execution requires a GGUF MMProj file")
        }
        (Lfm2VlMmprojSource::GgufFile(_), Lfm2VlMmprojExecution::Auto) => {
            Ok(MmprojLoadPlan::GgufAuto)
        }
        (Lfm2VlMmprojSource::GgufFile(_), Lfm2VlMmprojExecution::Dense) => {
            Ok(MmprojLoadPlan::GgufDense)
        }
        (Lfm2VlMmprojSource::GgufFile(_), Lfm2VlMmprojExecution::Q8) => Ok(MmprojLoadPlan::GgufQ8),
    }
}

fn read_processor_config(path: &Path) -> Result<String> {
    let bytes = read_bounded_bytes(path, "processor config", MAX_PROCESSOR_CONFIG_BYTES)?;
    String::from_utf8(bytes).map_err(|error| {
        candle::Error::Msg(format!(
            "processor config {} is not UTF-8: {error}",
            path.display()
        ))
    })
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let bytes = read_bounded_bytes(path, "tokenizer", MAX_TOKENIZER_BYTES)?;
    Tokenizer::from_bytes(&bytes).map_err(|error| {
        candle::Error::Msg(format!("cannot load tokenizer {}: {error}", path.display()))
    })
}

fn read_bounded_bytes(path: &Path, label: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|error| {
        candle::Error::Msg(format!("cannot open {label} {}: {error}", path.display()))
    })?;
    let declared = file
        .metadata()
        .map_err(|error| {
            candle::Error::Msg(format!(
                "cannot read {label} metadata {}: {error}",
                path.display()
            ))
        })?
        .len();
    if declared == 0 || declared > max_bytes {
        candle::bail!(
            "{label} {} is {declared} bytes, outside 1..={max_bytes}",
            path.display()
        )
    }
    let capacity = usize::try_from(declared)
        .map_err(|_| candle::Error::Msg(format!("{label} length does not fit usize")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| candle::Error::Msg(format!("{label} allocation failed")))?;
    bytes.resize(capacity, 0);
    let mut file = file;
    file.read_exact(&mut bytes).map_err(|error| {
        candle::Error::Msg(format!("cannot read {label} {}: {error}", path.display()))
    })?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing).map_err(|error| {
        candle::Error::Msg(format!(
            "cannot finish reading {label} {}: {error}",
            path.display()
        ))
    })? != 0
    {
        candle::bail!("{label} {} changed while it was read", path.display())
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::lfm2_vl::GgufMmprojExecution;
    use sha2::{Digest, Sha256};

    fn repository_fixture(path: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures")
            .join(path)
    }

    #[test]
    fn execution_policy_parses_and_displays_stably() -> Result<()> {
        assert_eq!(
            "auto".parse::<Lfm2VlMmprojExecution>()?,
            Lfm2VlMmprojExecution::Auto
        );
        assert_eq!(
            "dequantize".parse::<Lfm2VlMmprojExecution>()?,
            Lfm2VlMmprojExecution::Dense
        );
        assert_eq!(
            "native-q8".parse::<Lfm2VlMmprojExecution>()?,
            Lfm2VlMmprojExecution::Q8
        );
        assert_eq!(Lfm2VlMmprojExecution::Q8.to_string(), "q8");
        assert!("other".parse::<Lfm2VlMmprojExecution>().is_err());
        Ok(())
    }

    #[test]
    fn load_plan_maps_supported_modes_and_rejects_q8_early() -> Result<()> {
        let split = Lfm2VlMmprojSource::SplitDirectory(Path::new("split"));
        let gguf = Lfm2VlMmprojSource::GgufFile(Path::new("mmproj.gguf"));
        assert_eq!(
            mmproj_load_plan(split, Lfm2VlMmprojExecution::Auto, DType::F32)?,
            MmprojLoadPlan::SplitDense
        );
        assert_eq!(
            mmproj_load_plan(split, Lfm2VlMmprojExecution::Dense, DType::F16)?,
            MmprojLoadPlan::SplitDense
        );
        assert_eq!(
            mmproj_load_plan(gguf, Lfm2VlMmprojExecution::Auto, DType::BF16)?,
            MmprojLoadPlan::GgufAuto
        );
        assert_eq!(
            mmproj_load_plan(gguf, Lfm2VlMmprojExecution::Dense, DType::F16)?,
            MmprojLoadPlan::GgufDense
        );
        assert_eq!(
            mmproj_load_plan(gguf, Lfm2VlMmprojExecution::Q8, DType::F32)?,
            MmprojLoadPlan::GgufQ8
        );
        assert!(mmproj_load_plan(split, Lfm2VlMmprojExecution::Q8, DType::F32).is_err());
        assert!(mmproj_load_plan(gguf, Lfm2VlMmprojExecution::Q8, DType::BF16).is_err());
        Ok(())
    }

    #[test]
    fn consumed_files_cover_split_direct_and_override_inputs() -> Result<()> {
        let split_dir = Path::new("split");
        let bundled_processor = split_dir.join("processor_config.json");
        assert_eq!(
            hybrid_consumed_files(
                Path::new("text.gguf"),
                Lfm2VlMmprojSource::SplitDirectory(split_dir),
                Path::new("tokenizer.json"),
                Some(&bundled_processor),
            )?,
            vec![
                PathBuf::from("text.gguf"),
                split_dir.join("mmproj.safetensors"),
                split_dir.join("mmproj.json"),
                bundled_processor,
                PathBuf::from("tokenizer.json"),
            ]
        );

        assert_eq!(
            hybrid_consumed_files(
                Path::new("text.gguf"),
                Lfm2VlMmprojSource::GgufFile(Path::new("mmproj.gguf")),
                Path::new("tokenizer.json"),
                Some(Path::new("processor-override.json")),
            )?,
            vec![
                PathBuf::from("text.gguf"),
                PathBuf::from("mmproj.gguf"),
                PathBuf::from("tokenizer.json"),
                PathBuf::from("processor-override.json"),
            ]
        );
        Ok(())
    }

    #[test]
    fn invalid_q8_policy_fails_before_any_file_is_opened() {
        let device = Device::Cpu;
        let options = Lfm2VlHybridLoadOptions {
            text_gguf: Path::new("missing-text.gguf"),
            mmproj: Lfm2VlMmprojSource::GgufFile(Path::new("missing-mmproj.gguf")),
            tokenizer: Path::new("missing-tokenizer.json"),
            processor_config: None,
            mmproj_execution: Lfm2VlMmprojExecution::Q8,
            vision_dtype: DType::BF16,
            vision_device: &device,
            text_device: &device,
        };
        let error = load_lfm2_vl_hybrid(options)
            .err()
            .expect("invalid Q8 dtype must fail");
        assert!(error.to_string().contains("requires F32 activations"));
    }

    #[test]
    fn malformed_metadata_fails_before_the_text_model_is_opened() {
        let device = Device::Cpu;
        let loader = repository_fixture("lfm2_vl_loader_tiny");
        let tokenizer = loader.join("tokenizer.json");
        let non_json = loader.join("text.gguf");
        let missing_text = loader.join("missing-text.gguf");
        let missing_mmproj = loader.join("missing-mmproj.gguf");

        let bad_tokenizer = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
            text_gguf: &missing_text,
            mmproj: Lfm2VlMmprojSource::GgufFile(&missing_mmproj),
            tokenizer: &non_json,
            processor_config: None,
            mmproj_execution: Lfm2VlMmprojExecution::Dense,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })
        .err()
        .expect("malformed tokenizer must fail");
        assert!(bad_tokenizer.to_string().contains("cannot load tokenizer"));
        assert!(!bad_tokenizer.to_string().contains("text GGUF"));

        let bad_processor = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
            text_gguf: &missing_text,
            mmproj: Lfm2VlMmprojSource::GgufFile(&missing_mmproj),
            tokenizer: &tokenizer,
            processor_config: Some(&non_json),
            mmproj_execution: Lfm2VlMmprojExecution::Dense,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })
        .err()
        .expect("malformed processor config must fail");
        assert!(bad_processor.to_string().contains("processor config"));
        assert!(!bad_processor.to_string().contains("text GGUF"));
    }

    #[test]
    fn metadata_reader_accepts_the_exact_bound_and_rejects_one_byte_over() -> Result<()> {
        let tokenizer = repository_fixture("lfm2_vl_loader_tiny/tokenizer.json");
        let bytes = std::fs::metadata(&tokenizer)
            .map_err(|error| candle::Error::Msg(error.to_string()))?
            .len();
        assert_eq!(
            read_bounded_bytes(&tokenizer, "tokenizer", bytes)?.len() as u64,
            bytes
        );
        let error = read_bounded_bytes(&tokenizer, "tokenizer", bytes - 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside 1..="));
        Ok(())
    }

    #[test]
    fn public_loader_covers_split_direct_dense_and_direct_q8_fixtures() -> Result<()> {
        let device = Device::Cpu;
        let loader = repository_fixture("lfm2_vl_loader_tiny");
        let split = repository_fixture("lfm2_vl_mmproj_tiny");
        let processor = split.join("processor_config.json");
        let text = loader.join("text.gguf");
        let tokenizer = loader.join("tokenizer.json");

        let split_loaded = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
            text_gguf: &text,
            mmproj: Lfm2VlMmprojSource::SplitDirectory(&split),
            tokenizer: &tokenizer,
            processor_config: None,
            mmproj_execution: Lfm2VlMmprojExecution::Dense,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })?;
        assert_eq!(split_loaded.consumed_files.len(), 5);
        assert_eq!(split_loaded.model.mmproj().gguf_execution(), None);

        let dense_mmproj = loader.join("mmproj-dense.gguf");
        let dense_loaded = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
            text_gguf: &text,
            mmproj: Lfm2VlMmprojSource::GgufFile(&dense_mmproj),
            tokenizer: &tokenizer,
            processor_config: Some(&processor),
            mmproj_execution: Lfm2VlMmprojExecution::Dense,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })?;
        assert_eq!(dense_loaded.consumed_files.len(), 4);
        assert_eq!(
            dense_loaded.model.mmproj().gguf_execution(),
            Some(GgufMmprojExecution::DenseCompatibility)
        );
        assert_eq!(
            dense_loaded.model.mmproj().native_quantized_tensor_count(),
            0
        );

        let q8_mmproj = loader.join("mmproj-q8.gguf");
        let q8_loaded = load_lfm2_vl_hybrid(Lfm2VlHybridLoadOptions {
            text_gguf: &text,
            mmproj: Lfm2VlMmprojSource::GgufFile(&q8_mmproj),
            tokenizer: &tokenizer,
            processor_config: Some(&processor),
            mmproj_execution: Lfm2VlMmprojExecution::Q8,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        })?;
        assert_eq!(q8_loaded.consumed_files.len(), 4);
        assert_eq!(
            q8_loaded.model.mmproj().gguf_execution(),
            Some(GgufMmprojExecution::Q8_0)
        );
        assert!(q8_loaded.model.mmproj().native_quantized_tensor_count() > 0);
        Ok(())
    }

    #[test]
    fn loader_fixture_manifest_pins_every_generated_input() -> Result<()> {
        let root = repository_fixture("lfm2_vl_loader_tiny");
        let manifest_path = root.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
            candle::Error::Msg(format!(
                "cannot read loader fixture manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                candle::Error::Msg(format!("invalid loader fixture manifest: {error}"))
            })?;
        let files = manifest["files"]
            .as_object()
            .ok_or_else(|| candle::Error::Msg("loader fixture manifest has no files".into()))?;
        assert_eq!(files.len(), 4);
        for (name, record) in files {
            let expected_bytes = record["bytes"].as_u64().ok_or_else(|| {
                candle::Error::Msg(format!("loader fixture {name} has no byte count"))
            })?;
            let expected_hash = record["sha256"].as_str().ok_or_else(|| {
                candle::Error::Msg(format!("loader fixture {name} has no SHA-256"))
            })?;
            let bytes = std::fs::read(root.join(name)).map_err(|error| {
                candle::Error::Msg(format!("cannot read loader fixture {name}: {error}"))
            })?;
            assert_eq!(bytes.len() as u64, expected_bytes, "fixture {name} size");
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                expected_hash,
                "fixture {name} SHA-256"
            );
        }
        Ok(())
    }
}
