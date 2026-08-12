#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod args;
mod loading;
mod native_checkpoint;
mod native_loading;
mod runner;
mod trace;

use anyhow::Result;
use args::{Args, InferenceArgs, MmprojArg, ModelSource, ParseOutcome};
use candle::{DType, Device};
use candle_transformers::models::lfm2_vl::GgufMmprojExecution;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args = match args::parse_env()? {
        ParseOutcome::Run(args) => args,
        ParseOutcome::Help => {
            println!("{}", args::USAGE);
            return Ok(());
        }
    };
    let device_policy = args.device_policy();
    let text_device = candle_examples::device(device_policy.text_cpu)?;
    let vision_device = match (device_policy.vision_cpu, device_policy.text_cpu) {
        (true, _) => Device::Cpu,
        (false, true) => candle_examples::device(false)?,
        (false, false) => text_device.clone(),
    };
    let vision_dtype = args.resolved_vision_dtype(&vision_device);
    let text_dtype = args.resolved_text_dtype(&text_device);
    args.validate_device_dtypes(&vision_device, &text_device, vision_dtype, text_dtype)?;
    args.validate_execution(vision_dtype)?;
    match &args.source {
        ModelSource::NativeDirectory(model_dir) => run_native(
            &args,
            model_dir,
            vision_dtype,
            text_dtype,
            &vision_device,
            &text_device,
        ),
        ModelSource::Hybrid {
            text_gguf,
            mmproj,
            tokenizer,
        } => run_hybrid(
            &args,
            text_gguf,
            mmproj,
            tokenizer,
            vision_dtype,
            &vision_device,
            &text_device,
        ),
    }
}

fn run_hybrid(
    args: &Args,
    text_gguf: &Path,
    mmproj: &MmprojArg,
    tokenizer: &Path,
    vision_dtype: DType,
    vision_device: &Device,
    text_device: &Device,
) -> Result<()> {
    let mmproj_input = match mmproj {
        MmprojArg::SplitDirectory(path) => loading::MmprojInput::SplitDirectory(path),
        MmprojArg::GgufFile(path) => loading::MmprojInput::GgufFile(path),
    };
    let profile = args
        .inference
        .as_ref()
        .is_some_and(|inference| inference.timings);
    let load_started = Instant::now();
    let mut loaded = loading::load_hybrid(loading::HybridLoadOptions {
        text_gguf,
        mmproj: mmproj_input,
        tokenizer,
        processor_config: args.processor_config.as_deref(),
        mmproj_execution: args.mmproj_execution,
        vision_dtype,
        vision_device,
        text_device,
    })?;
    if profile {
        synchronize_timing_devices(loaded.model.vision_device(), loaded.model.text_device())?;
        eprintln!(
            "lfm2-vl timings_ms model_load={:.3} sync=cuda-device-complete",
            load_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let json = args
        .inference
        .as_ref()
        .is_some_and(|inference| inference.json);
    if !json {
        let pairing = loaded.model.pairing_report();
        let resolved_mmproj_execution = loaded
            .model
            .mmproj()
            .gguf_execution()
            .unwrap_or(GgufMmprojExecution::DenseCompatibility);
        println!(
            "loaded hybrid LFM2-VL: text={}x{} vision_layers={} patch={} factor={} image_token={} requested_dtype={} resolved_vision_dtype={vision_dtype:?} vision_device={:?} text_device={:?}",
            pairing.text_layer_count,
            pairing.text_hidden_size,
            pairing.vision_layer_count,
            pairing.patch_size,
            pairing.downsample_factor,
            pairing.image_token_id,
            args.requested_dtype_label(),
            loaded.model.vision_device(),
            loaded.model.text_device(),
        );
        println!(
            "MMProj tensors={} requested_execution={} resolved_execution={resolved_mmproj_execution:?} native_q8_tensors={} processor_max_patches={:?} output={}",
            loaded.model.mmproj().report.loaded_tensors.len(),
            args.mmproj_execution,
            loaded.model.mmproj().native_quantized_tensor_count(),
            loaded.processor.config().max_num_patches,
            pairing.text_output_resolution,
        );
        println!(
            "tokenizer image token validated as {}",
            loaded.prompt.special_tokens().image_token_id
        );
    }
    if let Some(inference) = &args.inference {
        let backend = match mmproj {
            MmprojArg::SplitDirectory(_) => "hybrid-split-mmproj",
            MmprojArg::GgufFile(_) => "hybrid-gguf-mmproj",
        };
        let report = runner::run_hybrid(
            &mut loaded.model,
            &loaded.processor,
            &loaded.prompt,
            inference_request(backend, &loaded.consumed_files, inference),
        )?;
        emit_report(&report, inference.json)?;
    }
    Ok(())
}

fn run_native(
    args: &Args,
    model_dir: &Path,
    vision_dtype: DType,
    text_dtype: DType,
    vision_device: &Device,
    text_device: &Device,
) -> Result<()> {
    let profile = args
        .inference
        .as_ref()
        .is_some_and(|inference| inference.timings);
    let load_started = Instant::now();
    let loaded = native_loading::load_native(
        model_dir,
        args.processor_config.as_deref(),
        native_loading::NativeLoadOptions {
            vision_dtype,
            text_dtype,
            vision_device,
            text_device,
        },
    )?;
    if profile {
        synchronize_timing_devices(loaded.model.vision_device(), loaded.model.text_device())?;
        eprintln!(
            "lfm2-vl timings_ms model_load={:.3} sync=cuda-device-complete",
            load_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let json = args
        .inference
        .as_ref()
        .is_some_and(|inference| inference.json);
    if !json {
        let config = loaded.model.config();
        println!(
            "loaded native LFM2-VL: text={}x{} vision_layers={} patch={} factor={} image_token={} requested_dtype={} resolved_vision_dtype={vision_dtype:?} resolved_text_dtype={text_dtype:?} vision_device={:?} text_device={:?}",
            config.text_config.num_hidden_layers,
            config.text_config.hidden_size,
            config.vision_config.num_hidden_layers,
            config.vision_config.patch_size,
            config.downsample_factor,
            config.image_token_id,
            args.requested_dtype_label(),
            loaded.model.vision_device(),
            loaded.model.text_device(),
        );
        println!(
            "native tensors={} shards={} indexed={} bytes={} vision_root={} projector_root={} language_root={} tied_output={} requested_execution={} resolved_execution=native-dense processor_max_patches={:?}",
            loaded.report.loaded_tensors.len(),
            loaded.report.shard_count,
            loaded.report.indexed,
            loaded.report.total_file_bytes,
            loaded.report.resolved_vision_root,
            loaded.report.resolved_projector_root,
            loaded.report.resolved_language_root,
            loaded.report.tied_output_resolution,
            args.mmproj_execution,
            loaded.processor.config().max_num_patches,
        );
        println!(
            "native report_vision_dtype={} report_text_dtype={} reported_vision_device={} reported_text_device={} tokenizer image token validated as {}",
            loaded.report.vision_dtype,
            loaded.report.text_dtype,
            loaded.report.vision_device,
            loaded.report.text_device,
            loaded.prompt.special_tokens().image_token_id
        );
    }
    if let Some(inference) = &args.inference {
        let report = runner::run_native(
            &loaded.model,
            &loaded.processor,
            &loaded.prompt,
            inference_request("native-safetensors", &loaded.source_files, inference),
        )?;
        emit_report(&report, inference.json)?;
    }
    Ok(())
}

fn inference_request<'a>(
    backend: &'a str,
    model_inputs: &'a [PathBuf],
    inference: &'a InferenceArgs,
) -> runner::InferenceRequest<'a> {
    runner::InferenceRequest {
        backend,
        model_inputs,
        prompt: &inference.prompt,
        image_paths: &inference.images,
        max_new_tokens: inference.max_new_tokens,
        vision_batch_size: inference.vision_batch_size,
        eos_token_id: inference.eos_token_id,
        timings: inference.timings,
        benchmark_generation: inference.benchmark_generation,
        trace_output: inference.trace_output.as_deref(),
    }
}

fn synchronize_timing_devices(vision_device: &Device, text_device: &Device) -> Result<()> {
    vision_device.synchronize()?;
    if !vision_device.same_device(text_device) {
        text_device.synchronize()?;
    }
    Ok(())
}

fn emit_report(report: &runner::InferenceReport, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
    } else {
        println!("generated token ids: {:?}", report.generation.generated_ids);
        println!(
            "generated text: {}",
            report.generation.decoded_skip_special_tokens
        );
        println!(
            "stop={} cache_reset_exact={}",
            report.generation.stop_reason, report.cache_reset_exact
        );
    }
    Ok(())
}
