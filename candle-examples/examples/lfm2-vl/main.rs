#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod loading;

use anyhow::{bail, Result};
use candle::{DType, Device};
use std::path::PathBuf;

enum MmprojArg {
    SplitDirectory(PathBuf),
    GgufFile(PathBuf),
}

struct Args {
    text_gguf: PathBuf,
    mmproj: MmprojArg,
    tokenizer: PathBuf,
    processor_config: Option<PathBuf>,
    cpu: bool,
    vision_cpu: bool,
}

const USAGE: &str = "usage: lfm2-vl --model-file <text.gguf> (--mmproj-file <mmproj.gguf> | --mmproj-dir <split-dir>) --tokenizer <tokenizer.json> [--processor-config <processor_config.json>] [--cpu] [--vision-cpu]\n       lfm2-vl <text.gguf> <split-mmproj-dir> <tokenizer.json> [--cpu] [--vision-cpu]";

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut text_gguf = None;
    let mut mmproj_file = None;
    let mut mmproj_dir = None;
    let mut tokenizer = None;
    let mut processor_config = None;
    let mut cpu = false;
    let mut vision_cpu = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cpu" => cpu = true,
            "--vision-cpu" => vision_cpu = true,
            "--model-file" => {
                text_gguf =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--model-file requires a path")
                    })?));
            }
            "--mmproj-file" => {
                mmproj_file =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--mmproj-file requires a path")
                    })?));
            }
            "--mmproj-dir" => {
                mmproj_dir =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--mmproj-dir requires a path")
                    })?));
            }
            "--tokenizer" => {
                tokenizer =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--tokenizer requires a path")
                    })?));
            }
            "--processor-config" => {
                processor_config =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--processor-config requires a path")
                    })?));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ if argument.starts_with('-') => bail!("unknown option {argument}"),
            _ => positional.push(PathBuf::from(argument)),
        }
    }

    let explicit_paths =
        text_gguf.is_some() || mmproj_file.is_some() || mmproj_dir.is_some() || tokenizer.is_some();
    let (text_gguf, mmproj, tokenizer) = if explicit_paths {
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
        (text_gguf, mmproj, tokenizer)
    } else {
        if positional.len() != 3 {
            bail!(USAGE)
        }
        let text_gguf = positional.remove(0);
        let mmproj = MmprojArg::SplitDirectory(positional.remove(0));
        let tokenizer = positional.remove(0);
        (text_gguf, mmproj, tokenizer)
    };
    Ok(Args {
        text_gguf,
        mmproj,
        tokenizer,
        processor_config,
        cpu,
        vision_cpu,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let text_device = candle_examples::device(args.cpu)?;
    let vision_device = if args.cpu || args.vision_cpu {
        Device::Cpu
    } else {
        text_device.clone()
    };
    let vision_dtype = if vision_device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let mmproj_input = match &args.mmproj {
        MmprojArg::SplitDirectory(path) => loading::MmprojInput::SplitDirectory(path),
        MmprojArg::GgufFile(path) => loading::MmprojInput::GgufFile(path),
    };
    let loaded = loading::load_hybrid(
        &args.text_gguf,
        mmproj_input,
        &args.tokenizer,
        args.processor_config.as_deref(),
        vision_dtype,
        &vision_device,
        &text_device,
    )?;
    let pairing = loaded.model.pairing_report();
    println!(
        "loaded hybrid LFM2-VL: text={}x{} vision_layers={} patch={} factor={} image_token={} vision_device={:?} text_device={:?}",
        pairing.text_layer_count,
        pairing.text_hidden_size,
        pairing.vision_layer_count,
        pairing.patch_size,
        pairing.downsample_factor,
        pairing.image_token_id,
        loaded.model.vision_device(),
        loaded.model.text_device(),
    );
    println!(
        "MMProj tensors={} execution={:?} native_q8_tensors={} processor_max_patches={:?} output={}",
        loaded.model.mmproj().report.loaded_tensors.len(),
        loaded.model.mmproj().gguf_execution(),
        loaded.model.mmproj().native_quantized_tensor_count(),
        loaded.processor.config().max_num_patches,
        pairing.text_output_resolution,
    );
    println!(
        "tokenizer image token validated as {}",
        loaded.prompt.special_tokens().image_token_id
    );
    Ok(())
}
