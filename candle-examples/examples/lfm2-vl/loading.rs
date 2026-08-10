//! Local-only loader for quantized LFM2 GGUF text plus split or GGUF MMProj.

use crate::args::MmprojExecutionArg;
use anyhow::{bail, Context, Result};
use candle::quantized::gguf_file;
use candle::{DType, Device};
use candle_transformers::models::lfm2_vl::{Mmproj, QuantizedLfm2VlModel};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_vlm::lfm2_vl::{
    Lfm2VlProcessor, Lfm2VlProcessorConfig, Lfm2VlPrompt, Lfm2VlSpecialTokens,
    ProcessorConfigPatch, PromptOptions,
};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const MAX_PROCESSOR_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub enum MmprojInput<'a> {
    SplitDirectory(&'a Path),
    GgufFile(&'a Path),
}

#[derive(Clone, Copy, Debug)]
pub struct MmprojLoadOptions<'a> {
    pub execution: MmprojExecutionArg,
    pub dtype: DType,
    pub device: &'a Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmprojLoadPlan {
    SplitDense,
    GgufAuto,
    GgufDense,
    GgufQ8,
}

pub struct LoadedHybrid {
    pub model: QuantizedLfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
    pub source_files: Vec<PathBuf>,
}

pub fn load_hybrid(
    text_gguf: impl AsRef<Path>,
    mmproj_input: MmprojInput<'_>,
    tokenizer_path: impl AsRef<Path>,
    processor_config_path: Option<&Path>,
    mmproj_options: MmprojLoadOptions<'_>,
    text_device: &Device,
) -> Result<LoadedHybrid> {
    let mmproj_plan =
        mmproj_load_plan(mmproj_input, mmproj_options.execution, mmproj_options.dtype)?;
    let text_gguf = text_gguf.as_ref();
    let mut text_file = std::fs::File::open(text_gguf)
        .with_context(|| format!("opening text GGUF {}", text_gguf.display()))?;
    let content =
        gguf_file::Content::read(&mut text_file).map_err(|err| err.with_path(text_gguf))?;
    let text = ModelWeights::from_gguf(content, &mut text_file, text_device)?;

    let tokenizer_path = tokenizer_path.as_ref();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let tokenizer_image_token_id =
        Lfm2VlSpecialTokens::resolve(&tokenizer, None, 0, false)?.image_token_id;

    let mmproj = match mmproj_input {
        MmprojInput::SplitDirectory(path) => {
            if mmproj_plan != MmprojLoadPlan::SplitDense {
                bail!("internal MMProj load plan does not match split input")
            }
            Mmproj::load(path, mmproj_options.dtype, mmproj_options.device)?
        }
        MmprojInput::GgufFile(path) => match mmproj_plan {
            MmprojLoadPlan::GgufAuto => Mmproj::load_gguf_auto(
                path,
                mmproj_options.dtype,
                mmproj_options.device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::GgufDense => Mmproj::load_gguf(
                path,
                mmproj_options.dtype,
                mmproj_options.device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::GgufQ8 => Mmproj::load_gguf_q8(
                path,
                mmproj_options.dtype,
                mmproj_options.device,
                tokenizer_image_token_id,
            )?,
            MmprojLoadPlan::SplitDense => {
                bail!("internal MMProj load plan does not match GGUF input")
            }
        },
    };
    let processor_patch = processor_config_path
        .map(|path| {
            let json = read_processor_config(path)?;
            ProcessorConfigPatch::from_json(&json).map_err(anyhow::Error::msg)
        })
        .transpose()?;
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
    let source_files = hybrid_source_files(
        text_gguf,
        mmproj_input,
        tokenizer_path,
        processor_config_path,
    )?;
    Ok(LoadedHybrid {
        model,
        processor,
        prompt,
        source_files,
    })
}

fn hybrid_source_files(
    text_gguf: &Path,
    mmproj_input: MmprojInput<'_>,
    tokenizer_path: &Path,
    processor_config_path: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    let mut source_files = Vec::new();
    source_files
        .try_reserve_exact(6)
        .map_err(|_| anyhow::anyhow!("allocating hybrid source-file evidence"))?;
    source_files.push(text_gguf.to_path_buf());
    match mmproj_input {
        MmprojInput::SplitDirectory(path) => {
            source_files.push(path.join("mmproj.safetensors"));
            source_files.push(path.join("mmproj.json"));
            source_files.push(path.join("processor_config.json"));
        }
        MmprojInput::GgufFile(path) => source_files.push(path.to_path_buf()),
    }
    source_files.push(tokenizer_path.to_path_buf());
    if let Some(path) = processor_config_path {
        let path = path.to_path_buf();
        if !source_files.contains(&path) {
            source_files.push(path);
        }
    }
    Ok(source_files)
}

fn mmproj_load_plan(
    input: MmprojInput<'_>,
    execution: MmprojExecutionArg,
    vision_dtype: DType,
) -> Result<MmprojLoadPlan> {
    if execution == MmprojExecutionArg::Q8 && vision_dtype != DType::F32 {
        bail!("native Q8 MMProj execution requires F32 activations, received {vision_dtype:?}")
    }
    match (input, execution) {
        (MmprojInput::SplitDirectory(_), MmprojExecutionArg::Auto | MmprojExecutionArg::Dense) => {
            Ok(MmprojLoadPlan::SplitDense)
        }
        (MmprojInput::SplitDirectory(_), MmprojExecutionArg::Q8) => {
            bail!("native Q8 MMProj execution requires a GGUF --mmproj-file")
        }
        (MmprojInput::GgufFile(_), MmprojExecutionArg::Auto) => Ok(MmprojLoadPlan::GgufAuto),
        (MmprojInput::GgufFile(_), MmprojExecutionArg::Dense) => Ok(MmprojLoadPlan::GgufDense),
        (MmprojInput::GgufFile(_), MmprojExecutionArg::Q8) => Ok(MmprojLoadPlan::GgufQ8),
    }
}

fn read_processor_config(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening processor config {}", path.display()))?;
    let declared = file
        .metadata()
        .with_context(|| format!("reading processor config metadata {}", path.display()))?
        .len();
    if declared > MAX_PROCESSOR_CONFIG_BYTES {
        bail!(
            "processor config {} is {declared} bytes, exceeding {MAX_PROCESSOR_CONFIG_BYTES}",
            path.display()
        )
    }
    let capacity =
        usize::try_from(declared).context("processor config length does not fit usize")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| anyhow::anyhow!("processor config allocation failed"))?;
    file.take(MAX_PROCESSOR_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading processor config {}", path.display()))?;
    if bytes.len() as u64 > MAX_PROCESSOR_CONFIG_BYTES {
        bail!(
            "processor config {} grew beyond {MAX_PROCESSOR_CONFIG_BYTES} bytes while reading",
            path.display()
        )
    }
    String::from_utf8(bytes)
        .with_context(|| format!("processor config {} is not UTF-8", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmproj_load_plan_maps_every_supported_execution_mode() -> Result<()> {
        let split = MmprojInput::SplitDirectory(Path::new("split"));
        let gguf = MmprojInput::GgufFile(Path::new("mmproj.gguf"));
        assert_eq!(
            mmproj_load_plan(split, MmprojExecutionArg::Auto, DType::F32)?,
            MmprojLoadPlan::SplitDense
        );
        assert_eq!(
            mmproj_load_plan(split, MmprojExecutionArg::Dense, DType::F16)?,
            MmprojLoadPlan::SplitDense
        );
        assert_eq!(
            mmproj_load_plan(gguf, MmprojExecutionArg::Auto, DType::BF16)?,
            MmprojLoadPlan::GgufAuto
        );
        assert_eq!(
            mmproj_load_plan(gguf, MmprojExecutionArg::Dense, DType::F16)?,
            MmprojLoadPlan::GgufDense
        );
        assert_eq!(
            mmproj_load_plan(gguf, MmprojExecutionArg::Q8, DType::F32)?,
            MmprojLoadPlan::GgufQ8
        );
        Ok(())
    }

    #[test]
    fn mmproj_load_plan_rejects_q8_before_file_loading() {
        let split = MmprojInput::SplitDirectory(Path::new("missing-split"));
        let gguf = MmprojInput::GgufFile(Path::new("missing.gguf"));
        assert!(mmproj_load_plan(split, MmprojExecutionArg::Q8, DType::F32).is_err());
        assert!(mmproj_load_plan(gguf, MmprojExecutionArg::Q8, DType::BF16).is_err());
        assert!(mmproj_load_plan(gguf, MmprojExecutionArg::Q8, DType::F16).is_err());
    }

    #[test]
    fn hybrid_source_files_cover_split_direct_and_override_inputs() -> Result<()> {
        let split_dir = Path::new("split");
        let bundled_processor = split_dir.join("processor_config.json");
        assert_eq!(
            hybrid_source_files(
                Path::new("text.gguf"),
                MmprojInput::SplitDirectory(split_dir),
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
            hybrid_source_files(
                Path::new("text.gguf"),
                MmprojInput::GgufFile(Path::new("mmproj.gguf")),
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
    fn hybrid_loader_rejects_invalid_q8_policy_before_opening_paths() -> Result<()> {
        let missing_text = Path::new("definitely-missing-text.gguf");
        let missing_tokenizer = Path::new("definitely-missing-tokenizer.json");
        let device = Device::Cpu;
        let error = match load_hybrid(
            missing_text,
            MmprojInput::GgufFile(Path::new("definitely-missing-mmproj.gguf")),
            missing_tokenizer,
            None,
            MmprojLoadOptions {
                execution: MmprojExecutionArg::Q8,
                dtype: DType::BF16,
                device: &device,
            },
            &device,
        ) {
            Ok(_) => bail!("invalid Q8 dtype unexpectedly reached file loading"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires F32 activations"));

        let error = match load_hybrid(
            missing_text,
            MmprojInput::SplitDirectory(Path::new("definitely-missing-split")),
            missing_tokenizer,
            None,
            MmprojLoadOptions {
                execution: MmprojExecutionArg::Q8,
                dtype: DType::F32,
                device: &device,
            },
            &device,
        ) {
            Ok(_) => bail!("invalid split Q8 policy unexpectedly reached file loading"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires a GGUF --mmproj-file"));
        Ok(())
    }
}
