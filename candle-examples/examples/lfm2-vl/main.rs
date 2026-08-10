#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod loading;

use anyhow::{bail, Result};
use candle::{DType, Device};
use std::path::PathBuf;

struct Args {
    text_gguf: PathBuf,
    mmproj_dir: PathBuf,
    tokenizer: PathBuf,
    cpu: bool,
    vision_cpu: bool,
}

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut cpu = false;
    let mut vision_cpu = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--cpu" => cpu = true,
            "--vision-cpu" => vision_cpu = true,
            "-h" | "--help" => {
                println!(
                    "usage: lfm2-vl <text.gguf> <mmproj-dir> <tokenizer.json> [--cpu] [--vision-cpu]"
                );
                std::process::exit(0);
            }
            _ if argument.starts_with('-') => bail!("unknown option {argument}"),
            _ => positional.push(PathBuf::from(argument)),
        }
    }
    if positional.len() != 3 {
        bail!("usage: lfm2-vl <text.gguf> <mmproj-dir> <tokenizer.json> [--cpu] [--vision-cpu]")
    }
    Ok(Args {
        text_gguf: positional.remove(0),
        mmproj_dir: positional.remove(0),
        tokenizer: positional.remove(0),
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
    let loaded = loading::load_hybrid(
        args.text_gguf,
        args.mmproj_dir,
        args.tokenizer,
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
        "MMProj tensors={} processor_max_patches={:?} output={}",
        loaded.model.mmproj().report.loaded_tensors.len(),
        loaded.processor.config().max_num_patches,
        pairing.text_output_resolution,
    );
    println!(
        "tokenizer image token validated as {}",
        loaded.prompt.special_tokens().image_token_id
    );
    Ok(())
}
