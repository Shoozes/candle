//! Fixed-token capacity stress; execution requires its own bounded allowance.
#[path = "gpt_oss_capacity/mod.rs"]
mod capacity;

use candle::{bail, Error, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, conflicts_with_all = ["check_input", "verify_progress"])]
    output: Option<PathBuf>,
    /// Verify the bounded input and token file without opening model weights.
    #[arg(long, conflicts_with = "verify_progress")]
    check_input: bool,
    /// Revalidate a completed capacity receipt without model execution.
    #[arg(long)]
    verify_progress: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (input, input_hash, tokens) = capacity::read_input(&args.input)?;
    if args.check_input {
        println!("{{\"status\":\"verified-capacity-input\",\"token_steps\":8192,\"public_forwards\":2096,\"sampled_outputs\":0}}");
        return Ok(());
    }
    if let Some(path) = args.verify_progress {
        capacity::verify_progress(&path, &input, &input_hash)?;
        println!("{{\"status\":\"verified-capacity-progress\",\"token_steps\":8192,\"public_forwards\":2096,\"sampled_outputs\":0}}");
        return Ok(());
    }
    if cfg!(debug_assertions) {
        bail!("capacity stress requires an optimized release build and a fresh allowance");
    }
    require_execution_backend(input.backend)?;
    let output = args
        .output
        .ok_or_else(|| Error::Msg("--output is required for execution".into()))?;
    capacity::execute(&input, input_hash, &tokens, &output)
}

fn require_execution_backend(_backend: capacity::Backend) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    if matches!(_backend, capacity::Backend::Cuda) {
        bail!("CUDA capacity execution requires the cuda feature");
    }
    Ok(())
}
