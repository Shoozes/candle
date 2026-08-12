//! Argument parsing and execution-policy validation for the LFM2-VL example.

use anyhow::{bail, Result};
use candle::{DType, Device};
use candle_transformers::models::lfm2_vl::VisionLimits;
use std::fmt;
use std::path::PathBuf;

const DEFAULT_MAX_NEW_TOKENS: usize = 32;
const MAX_NEW_TOKENS: usize = 1_024;
const DEFAULT_VISION_BATCH_SIZE: usize = 1;
const MAX_VISION_BATCH_SIZE: usize = 64;
const MAX_PROMPT_BYTES: usize = 1024 * 1024;
const MAX_TRACE_NEW_TOKENS: usize = 32;

pub const USAGE: &str = "usage: lfm2-vl --model-dir <hf-checkpoint-dir> [--processor-config <override.json>] [--dtype <f32|bf16|f16>] [--mmproj-execution <auto|dense>] [--cpu] [--text-cpu] [--vision-cpu] [inference]\n       lfm2-vl --model-file <text.gguf> (--mmproj-file <mmproj.gguf> | --mmproj-dir <split-dir>) --tokenizer <tokenizer.json> [--processor-config <processor_config.json>] [--dtype <f32|bf16|f16>] [--mmproj-execution <auto|dense|q8>] [--cpu] [--text-cpu] [--vision-cpu] [inference]\n       lfm2-vl <text.gguf> <split-mmproj-dir> <tokenizer.json> [--processor-config <processor_config.json>] [--dtype <f32|bf16|f16>] [--mmproj-execution <auto|dense>] [--cpu] [--text-cpu] [--vision-cpu] [inference]\n\ninference: --prompt <text> [--image <path>]... [--max-new-tokens <0..1024>] [--vision-batch-size <1..64>] [--eos-token-id <u32>] [--json] [--timings] [--trace-output <external-dir>]\nEach image requires one literal <image> sentinel in the prompt. Without --prompt the command remains load-and-report only. --timings writes stage durations to stderr without changing JSON evidence. --trace-output is a native CPU/F32, single-crop parity lane and requires --cpu, one image, and at most 32 generated tokens.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MmprojArg {
    SplitDirectory(PathBuf),
    GgufFile(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    NativeDirectory(PathBuf),
    Hybrid {
        text_gguf: PathBuf,
        mmproj: MmprojArg,
        tokenizer: PathBuf,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceArgs {
    pub prompt: String,
    pub images: Vec<PathBuf>,
    pub max_new_tokens: usize,
    pub vision_batch_size: usize,
    pub eos_token_id: Option<u32>,
    pub json: bool,
    pub timings: bool,
    pub trace_output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeArg {
    F32,
    Bf16,
    F16,
}

impl DTypeArg {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "f16" | "float16" => Ok(Self::F16),
            _ => bail!("unsupported --dtype {value:?}; expected f32, bf16, or f16"),
        }
    }

    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
        }
    }
}

impl fmt::Display for DTypeArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MmprojExecutionArg {
    #[default]
    Auto,
    Dense,
    Q8,
}

impl MmprojExecutionArg {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "dense" | "dequantize" => Ok(Self::Dense),
            "q8" | "q8_0" | "native-q8" => Ok(Self::Q8),
            _ => bail!("unsupported --mmproj-execution {value:?}; expected auto, dense, or q8"),
        }
    }
}

impl fmt::Display for MmprojExecutionArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Auto => "auto",
            Self::Dense => "dense",
            Self::Q8 => "q8",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    pub source: ModelSource,
    pub processor_config: Option<PathBuf>,
    pub dtype: Option<DTypeArg>,
    pub mmproj_execution: MmprojExecutionArg,
    pub cpu: bool,
    pub text_cpu: bool,
    pub vision_cpu: bool,
    pub inference: Option<InferenceArgs>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePolicy {
    pub text_cpu: bool,
    pub vision_cpu: bool,
}

impl Args {
    pub fn device_policy(&self) -> DevicePolicy {
        DevicePolicy {
            text_cpu: self.cpu || self.text_cpu,
            vision_cpu: self.cpu || self.vision_cpu,
        }
    }

    pub fn resolved_vision_dtype(&self, vision_device: &Device) -> DType {
        self.resolved_dtype_for(vision_device.is_cuda())
    }

    pub fn resolved_text_dtype(&self, text_device: &Device) -> DType {
        self.resolved_dtype_for(text_device.is_cuda())
    }

    pub fn requested_dtype_label(&self) -> &'static str {
        self.dtype.map_or("default", |dtype| match dtype {
            DTypeArg::F32 => "f32",
            DTypeArg::Bf16 => "bf16",
            DTypeArg::F16 => "f16",
        })
    }

    fn resolved_dtype_for(&self, device_is_cuda: bool) -> DType {
        self.dtype.map(DTypeArg::dtype).unwrap_or_else(|| {
            if device_is_cuda {
                DType::BF16
            } else {
                DType::F32
            }
        })
    }

    pub fn validate_execution(&self, vision_dtype: DType) -> Result<()> {
        if self.mmproj_execution == MmprojExecutionArg::Q8 && vision_dtype != DType::F32 {
            bail!(
                "--mmproj-execution q8 requires --dtype f32; resolved vision dtype is {vision_dtype:?}"
            )
        }
        if self.mmproj_execution == MmprojExecutionArg::Q8 {
            match &self.source {
                ModelSource::NativeDirectory(_) => {
                    bail!("--mmproj-execution q8 is unavailable for native safetensors")
                }
                ModelSource::Hybrid {
                    mmproj: MmprojArg::SplitDirectory(_),
                    ..
                } => {
                    bail!(
                        "--mmproj-execution q8 requires --mmproj-file; split MMProj bundles are dense"
                    )
                }
                ModelSource::Hybrid {
                    mmproj: MmprojArg::GgufFile(_),
                    ..
                } => {}
            }
        }
        Ok(())
    }

    pub fn validate_device_dtypes(
        &self,
        policy: DevicePolicy,
        vision_dtype: DType,
        text_dtype: DType,
    ) -> Result<()> {
        if policy.vision_cpu && vision_dtype == DType::BF16 {
            bail!(
                "--dtype bf16 is unsupported for CPU vision matmul; use --dtype f32 or keep vision on CUDA"
            )
        }
        if policy.text_cpu && text_dtype == DType::BF16 {
            bail!(
                "--dtype bf16 is unsupported for CPU text matmul; use --dtype f32 or keep text on CUDA"
            )
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseOutcome {
    Run(Box<Args>),
    Help,
}

pub fn parse_env() -> Result<ParseOutcome> {
    parse_args(std::env::args().skip(1))
}

pub fn parse_args<I>(arguments: I) -> Result<ParseOutcome>
where
    I: IntoIterator<Item = String>,
{
    let mut positional = Vec::new();
    let mut model_dir = None;
    let mut text_gguf = None;
    let mut mmproj_file = None;
    let mut mmproj_dir = None;
    let mut tokenizer = None;
    let mut processor_config = None;
    let mut dtype = None;
    let mut mmproj_execution = None;
    let mut cpu = false;
    let mut text_cpu = false;
    let mut vision_cpu = false;
    let mut prompt = None;
    let mut images = Vec::new();
    let mut max_new_tokens = None;
    let mut vision_batch_size = None;
    let mut eos_token_id = None;
    let mut json = false;
    let mut timings = false;
    let mut trace_output = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cpu" => cpu = true,
            "--text-cpu" => text_cpu = true,
            "--vision-cpu" => vision_cpu = true,
            "--json" => json = true,
            "--timings" => timings = true,
            "--trace-output" => set_once(
                &mut trace_output,
                PathBuf::from(next_value(&mut arguments, "--trace-output")?),
                "--trace-output",
            )?,
            "--model-dir" => set_once(
                &mut model_dir,
                PathBuf::from(next_value(&mut arguments, "--model-dir")?),
                "--model-dir",
            )?,
            "--model-file" => set_once(
                &mut text_gguf,
                PathBuf::from(next_value(&mut arguments, "--model-file")?),
                "--model-file",
            )?,
            "--mmproj-file" => set_once(
                &mut mmproj_file,
                PathBuf::from(next_value(&mut arguments, "--mmproj-file")?),
                "--mmproj-file",
            )?,
            "--mmproj-dir" => set_once(
                &mut mmproj_dir,
                PathBuf::from(next_value(&mut arguments, "--mmproj-dir")?),
                "--mmproj-dir",
            )?,
            "--tokenizer" => set_once(
                &mut tokenizer,
                PathBuf::from(next_value(&mut arguments, "--tokenizer")?),
                "--tokenizer",
            )?,
            "--processor-config" => set_once(
                &mut processor_config,
                PathBuf::from(next_value(&mut arguments, "--processor-config")?),
                "--processor-config",
            )?,
            "--prompt" => set_once(
                &mut prompt,
                next_value(&mut arguments, "--prompt")?,
                "--prompt",
            )?,
            "--image" => {
                if images.len() >= VisionLimits::default().max_images {
                    bail!(
                        "--image may appear at most {} times",
                        VisionLimits::default().max_images
                    )
                }
                images.push(PathBuf::from(next_value(&mut arguments, "--image")?));
            }
            "--max-new-tokens" => {
                let value = next_value(&mut arguments, "--max-new-tokens")?;
                set_once(
                    &mut max_new_tokens,
                    parse_usize(&value, "--max-new-tokens")?,
                    "--max-new-tokens",
                )?;
            }
            "--vision-batch-size" => {
                let value = next_value(&mut arguments, "--vision-batch-size")?;
                set_once(
                    &mut vision_batch_size,
                    parse_usize(&value, "--vision-batch-size")?,
                    "--vision-batch-size",
                )?;
            }
            "--eos-token-id" => {
                let value = next_value(&mut arguments, "--eos-token-id")?;
                set_once(
                    &mut eos_token_id,
                    value.parse::<u32>().map_err(|_| {
                        anyhow::anyhow!("--eos-token-id requires an unsigned 32-bit integer")
                    })?,
                    "--eos-token-id",
                )?;
            }
            "--dtype" => {
                let value = next_value(&mut arguments, "--dtype")?;
                set_once(&mut dtype, DTypeArg::parse(&value)?, "--dtype")?;
            }
            "--mmproj-execution" => {
                let value = next_value(&mut arguments, "--mmproj-execution")?;
                set_once(
                    &mut mmproj_execution,
                    MmprojExecutionArg::parse(&value)?,
                    "--mmproj-execution",
                )?;
            }
            "-h" | "--help" => return Ok(ParseOutcome::Help),
            _ if argument.starts_with('-') => bail!("unknown option {argument}\n{USAGE}"),
            _ => positional.push(PathBuf::from(argument)),
        }
    }

    let explicit_paths = model_dir.is_some()
        || text_gguf.is_some()
        || mmproj_file.is_some()
        || mmproj_dir.is_some()
        || tokenizer.is_some();
    let source = if let Some(model_dir) = model_dir {
        if !positional.is_empty()
            || text_gguf.is_some()
            || mmproj_file.is_some()
            || mmproj_dir.is_some()
            || tokenizer.is_some()
        {
            bail!("--model-dir cannot be mixed with positional or hybrid loading paths\n{USAGE}")
        }
        ModelSource::NativeDirectory(model_dir)
    } else if explicit_paths {
        if !positional.is_empty() {
            bail!("positional model paths cannot be mixed with explicit loading flags\n{USAGE}")
        }
        let text_gguf = text_gguf.ok_or_else(|| anyhow::anyhow!("--model-file is required"))?;
        let tokenizer = tokenizer.ok_or_else(|| anyhow::anyhow!("--tokenizer is required"))?;
        let mmproj = match (mmproj_file, mmproj_dir) {
            (Some(path), None) => MmprojArg::GgufFile(path),
            (None, Some(path)) => MmprojArg::SplitDirectory(path),
            (None, None) => bail!("one of --mmproj-file or --mmproj-dir is required"),
            (Some(_), Some(_)) => {
                bail!("--mmproj-file and --mmproj-dir are mutually exclusive")
            }
        };
        ModelSource::Hybrid {
            text_gguf,
            mmproj,
            tokenizer,
        }
    } else {
        if positional.len() != 3 {
            bail!(USAGE)
        }
        let text_gguf = positional.remove(0);
        let mmproj = MmprojArg::SplitDirectory(positional.remove(0));
        let tokenizer = positional.remove(0);
        ModelSource::Hybrid {
            text_gguf,
            mmproj,
            tokenizer,
        }
    };
    let inference_requested = prompt.is_some()
        || !images.is_empty()
        || max_new_tokens.is_some()
        || vision_batch_size.is_some()
        || eos_token_id.is_some()
        || timings
        || trace_output.is_some()
        || json;
    let inference = match prompt {
        Some(prompt) => {
            if prompt.len() > MAX_PROMPT_BYTES {
                bail!("--prompt exceeds {MAX_PROMPT_BYTES} UTF-8 bytes")
            }
            let sentinel_count = prompt.match_indices("<image>").count();
            if sentinel_count != images.len() {
                bail!(
                    "--prompt contains {sentinel_count} <image> sentinels for {} --image inputs",
                    images.len()
                )
            }
            let max_new_tokens = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
            if max_new_tokens > MAX_NEW_TOKENS {
                bail!("--max-new-tokens {max_new_tokens} exceeds {MAX_NEW_TOKENS}")
            }
            if trace_output.is_some() {
                if !cpu {
                    bail!("--trace-output requires --cpu for the bounded CPU/F32 parity lane")
                }
                if dtype.is_some_and(|value| value != DTypeArg::F32) {
                    bail!("--trace-output requires --dtype f32 when dtype is explicit")
                }
                if images.len() != 1 {
                    bail!("--trace-output requires exactly one --image input")
                }
                if max_new_tokens > MAX_TRACE_NEW_TOKENS {
                    bail!("--trace-output allows at most {MAX_TRACE_NEW_TOKENS} generated tokens")
                }
            }
            let vision_batch_size = vision_batch_size.unwrap_or(DEFAULT_VISION_BATCH_SIZE);
            if !(1..=MAX_VISION_BATCH_SIZE).contains(&vision_batch_size) {
                bail!(
                    "--vision-batch-size {vision_batch_size} is outside 1..={MAX_VISION_BATCH_SIZE}"
                )
            }
            Some(InferenceArgs {
                prompt,
                images,
                max_new_tokens,
                vision_batch_size,
                eos_token_id,
                json,
                timings,
                trace_output,
            })
        }
        None if inference_requested => {
            bail!("inference options require --prompt\n{USAGE}")
        }
        None => None,
    };
    let args = Args {
        source,
        processor_config,
        dtype,
        mmproj_execution: mmproj_execution.unwrap_or_default(),
        cpu,
        text_cpu,
        vision_cpu,
        inference,
    };
    if args.mmproj_execution == MmprojExecutionArg::Q8 {
        match &args.source {
            ModelSource::NativeDirectory(_) => {
                bail!("--mmproj-execution q8 is unavailable for native safetensors")
            }
            ModelSource::Hybrid {
                mmproj: MmprojArg::SplitDirectory(_),
                ..
            } => {
                bail!(
                    "--mmproj-execution q8 requires --mmproj-file; split MMProj bundles are dense"
                )
            }
            ModelSource::Hybrid {
                mmproj: MmprojArg::GgufFile(_),
                ..
            } => {}
        }
    }
    if let Some(inference) = &args.inference {
        if inference.trace_output.is_some()
            && !matches!(args.source, ModelSource::NativeDirectory(_))
        {
            bail!("--trace-output is supported only with native safetensors loading")
        }
    }
    Ok(ParseOutcome::Run(Box::new(args)))
}

fn next_value(arguments: &mut impl Iterator<Item = String>, option: &str) -> Result<String> {
    arguments
        .next()
        .ok_or_else(|| anyhow::anyhow!("{option} requires a value"))
}

fn parse_usize(value: &str, option: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{option} requires a non-negative integer"))
}

fn set_once<T>(target: &mut Option<T>, value: T, option: &str) -> Result<()> {
    if target.is_some() {
        bail!("{option} may only be specified once")
    }
    *target = Some(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Result<ParseOutcome> {
        parse_args(values.iter().map(|value| (*value).to_owned()))
    }

    fn run(values: &[&str]) -> Result<Args> {
        match parse(values)? {
            ParseOutcome::Run(args) => Ok(*args),
            ParseOutcome::Help => bail!("unexpected help result"),
        }
    }

    #[test]
    fn positional_defaults_preserve_device_dependent_dtype_and_auto_execution() -> Result<()> {
        let args = run(&["text.gguf", "mmproj", "tokenizer.json"])?;
        assert_eq!(
            args.source,
            ModelSource::Hybrid {
                text_gguf: PathBuf::from("text.gguf"),
                mmproj: MmprojArg::SplitDirectory(PathBuf::from("mmproj")),
                tokenizer: PathBuf::from("tokenizer.json"),
            }
        );
        assert_eq!(args.dtype, None);
        assert_eq!(args.mmproj_execution, MmprojExecutionArg::Auto);
        assert_eq!(args.inference, None);
        assert_eq!(args.resolved_dtype_for(false), DType::F32);
        assert_eq!(args.resolved_dtype_for(true), DType::BF16);
        Ok(())
    }

    #[test]
    fn explicit_paths_and_value_aliases_parse() -> Result<()> {
        let args = run(&[
            "--model-file",
            "text.gguf",
            "--mmproj-file",
            "mmproj.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--processor-config",
            "processor.json",
            "--dtype",
            "bfloat16",
            "--mmproj-execution",
            "dequantize",
            "--cpu",
            "--text-cpu",
            "--vision-cpu",
        ])?;
        assert_eq!(
            args.source,
            ModelSource::Hybrid {
                text_gguf: PathBuf::from("text.gguf"),
                mmproj: MmprojArg::GgufFile(PathBuf::from("mmproj.gguf")),
                tokenizer: PathBuf::from("tokenizer.json"),
            }
        );
        assert_eq!(args.processor_config, Some(PathBuf::from("processor.json")));
        assert_eq!(args.dtype, Some(DTypeArg::Bf16));
        assert_eq!(args.mmproj_execution, MmprojExecutionArg::Dense);
        assert!(args.cpu);
        assert!(args.text_cpu);
        assert!(args.vision_cpu);
        Ok(())
    }

    #[test]
    fn native_directory_preserves_common_policy_and_rejects_hybrid_conflicts() -> Result<()> {
        let args = run(&[
            "--model-dir",
            "checkpoint",
            "--processor-config",
            "override.json",
            "--dtype",
            "f16",
            "--mmproj-execution",
            "dense",
            "--cpu",
        ])?;
        assert_eq!(
            args.source,
            ModelSource::NativeDirectory(PathBuf::from("checkpoint"))
        );
        assert_eq!(args.processor_config, Some(PathBuf::from("override.json")));
        assert_eq!(args.dtype, Some(DTypeArg::F16));
        assert_eq!(args.mmproj_execution, MmprojExecutionArg::Dense);
        assert!(args.cpu);
        args.validate_execution(DType::F16)?;

        assert!(parse(&["--model-dir", "checkpoint", "--model-file", "text.gguf"]).is_err());
        assert!(parse(&["--model-dir", "checkpoint", "--mmproj-execution", "q8"]).is_err());
        Ok(())
    }

    #[test]
    fn every_documented_dtype_spelling_parses() -> Result<()> {
        for (value, expected) in [
            ("f32", DTypeArg::F32),
            ("float32", DTypeArg::F32),
            ("bf16", DTypeArg::Bf16),
            ("bfloat16", DTypeArg::Bf16),
            ("f16", DTypeArg::F16),
            ("float16", DTypeArg::F16),
        ] {
            let args = run(&["text.gguf", "mmproj", "tokenizer.json", "--dtype", value])?;
            assert_eq!(args.dtype, Some(expected), "dtype spelling {value}");
        }
        Ok(())
    }

    #[test]
    fn device_policy_covers_cpu_flag_combinations() -> Result<()> {
        let defaults = run(&["text.gguf", "mmproj", "tokenizer.json"])?;
        assert_eq!(
            defaults.device_policy(),
            DevicePolicy {
                text_cpu: false,
                vision_cpu: false,
            }
        );

        let vision_only = run(&["text.gguf", "mmproj", "tokenizer.json", "--vision-cpu"])?;
        assert_eq!(
            vision_only.device_policy(),
            DevicePolicy {
                text_cpu: false,
                vision_cpu: true,
            }
        );

        let text_only = run(&["text.gguf", "mmproj", "tokenizer.json", "--text-cpu"])?;
        assert_eq!(
            text_only.device_policy(),
            DevicePolicy {
                text_cpu: true,
                vision_cpu: false,
            }
        );

        let all_cpu = run(&["text.gguf", "mmproj", "tokenizer.json", "--cpu"])?;
        assert_eq!(
            all_cpu.device_policy(),
            DevicePolicy {
                text_cpu: true,
                vision_cpu: true,
            }
        );

        let cpu_is_authoritative = run(&[
            "text.gguf",
            "mmproj",
            "tokenizer.json",
            "--cpu",
            "--text-cpu",
            "--vision-cpu",
        ])?;
        assert_eq!(
            cpu_is_authoritative.device_policy(),
            all_cpu.device_policy()
        );
        Ok(())
    }

    #[test]
    fn text_cpu_help_and_trace_restriction_are_explicit() -> Result<()> {
        assert!(USAGE.contains("--text-cpu"));
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--text-cpu",
            "--prompt",
            "describe <image>",
            "--image",
            "image.png",
            "--trace-output",
            "trace",
        ])
        .is_err());
        Ok(())
    }

    #[test]
    fn q8_aliases_and_policy_validation_are_controlled() -> Result<()> {
        let args = run(&[
            "--model-file",
            "text.gguf",
            "--mmproj-file",
            "mmproj.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--dtype",
            "float32",
            "--mmproj-execution",
            "q8_0",
        ])?;
        assert_eq!(args.dtype, Some(DTypeArg::F32));
        assert_eq!(args.mmproj_execution, MmprojExecutionArg::Q8);
        args.validate_execution(DType::F32)?;
        assert!(args.validate_execution(DType::BF16).is_err());
        assert!(args.validate_execution(DType::F16).is_err());
        Ok(())
    }

    #[test]
    fn cpu_bf16_component_is_rejected_before_model_load() -> Result<()> {
        let args = run(&["--model-dir", "checkpoint", "--dtype", "bf16", "--text-cpu"])?;
        let policy = args.device_policy();
        assert!(args
            .validate_device_dtypes(policy, DType::BF16, DType::BF16)
            .is_err());
        Ok(())
    }

    #[test]
    fn path_forms_and_execution_modes_reject_conflicts() {
        assert!(parse(&[
            "--model-file",
            "text.gguf",
            "--mmproj-file",
            "mmproj.gguf",
            "--mmproj-dir",
            "mmproj",
            "--tokenizer",
            "tokenizer.json",
        ])
        .is_err());
        assert!(parse(&[
            "text.gguf",
            "mmproj",
            "tokenizer.json",
            "--model-file",
            "other.gguf",
        ])
        .is_err());
        assert!(parse(&[
            "text.gguf",
            "mmproj",
            "tokenizer.json",
            "--mmproj-execution",
            "q8",
        ])
        .is_err());
        assert!(parse(&[
            "--model-file",
            "text.gguf",
            "--mmproj-file",
            "mmproj.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--dtype",
            "f32",
            "--dtype",
            "f16",
        ])
        .is_err());
    }

    #[test]
    fn help_missing_values_and_unknown_values_are_controlled() -> Result<()> {
        assert_eq!(parse(&["--help"])?, ParseOutcome::Help);
        assert!(parse(&["--model-file"]).is_err());
        assert!(parse(&["--dtype", "f64"]).is_err());
        assert!(parse(&["--mmproj-execution", "q4"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
        Ok(())
    }

    #[test]
    fn timings_without_prompt_are_rejected_before_loading() {
        assert!(parse(&["--model-dir", "checkpoint", "--timings"]).is_err());
    }

    #[test]
    fn inference_options_parse_with_exact_image_sentinel_pairing() -> Result<()> {
        let args = run(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "describe <image>",
            "--image",
            "image.png",
            "--max-new-tokens",
            "0",
            "--vision-batch-size",
            "4",
            "--eos-token-id",
            "7",
            "--json",
            "--timings",
        ])?;
        assert_eq!(
            args.inference,
            Some(InferenceArgs {
                prompt: "describe <image>".to_owned(),
                images: vec![PathBuf::from("image.png")],
                max_new_tokens: 0,
                vision_batch_size: 4,
                eos_token_id: Some(7),
                json: true,
                timings: true,
                trace_output: None,
            })
        );
        Ok(())
    }

    #[test]
    fn trace_output_is_bounded_and_native_cpu_only() -> Result<()> {
        let args = run(&[
            "--model-dir",
            "checkpoint",
            "--cpu",
            "--prompt",
            "describe <image>",
            "--image",
            "image.png",
            "--trace-output",
            "trace",
            "--max-new-tokens",
            "8",
        ])?;
        assert_eq!(
            args.inference
                .as_ref()
                .and_then(|inference| inference.trace_output.as_deref()),
            Some(PathBuf::from("trace").as_path())
        );
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "<image>",
            "--image",
            "image.png",
            "--trace-output",
            "trace",
        ])
        .is_err());
        assert!(parse(&[
            "--model-file",
            "text.gguf",
            "--mmproj-file",
            "mmproj.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--cpu",
            "--prompt",
            "<image>",
            "--image",
            "image.png",
            "--trace-output",
            "trace",
        ])
        .is_err());
        Ok(())
    }

    #[test]
    fn inference_options_reject_missing_prompt_mismatch_and_unsafe_bounds() {
        assert!(parse(&["--model-dir", "checkpoint", "--image", "image.png"]).is_err());
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "no image sentinel",
            "--image",
            "image.png",
        ])
        .is_err());
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "<image>",
            "--image",
            "image.png",
            "--max-new-tokens",
            "1025",
        ])
        .is_err());
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "text only",
            "--vision-batch-size",
            "0",
        ])
        .is_err());
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "text only",
            "--vision-batch-size",
            "65",
        ])
        .is_err());
        assert!(parse(&[
            "--model-dir",
            "checkpoint",
            "--prompt",
            "text only",
            "--max-new-tokens",
            "not-a-number",
        ])
        .is_err());
    }
}
