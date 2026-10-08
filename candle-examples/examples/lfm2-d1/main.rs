//! Local d1 decisions. All inputs and model identities are supplied by the caller.

mod files;

use anyhow::{Context, Result};
use candle::{DType, Device};
use candle_vlm::lfm2_d1::{
    load_d1_q8, D1Limits, D1LoadOptions, D1TraceEvent, DecisionFailure, DecisionRequest,
    DecisionResponse,
};
use candle_vlm::lfm2_vl::{Lfm2VlHybridLoadOptions, Lfm2VlMmprojExecution, Lfm2VlMmprojSource};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    mmproj: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    processor: PathBuf,
    #[arg(long)]
    template: PathBuf,
    #[arg(long)]
    artifact_manifest: PathBuf,
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 4096)]
    context: usize,
    #[arg(long)]
    trace: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    request_id: String,
    request: DecisionRequest,
    #[serde(default)]
    images: Vec<PathBuf>,
}

#[derive(Serialize)]
struct CaseResult {
    request_id: String,
    status: &'static str,
    elapsed_ms: f64,
    had_image: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    question_index: Option<usize>,
    response: DecisionResponse,
}

#[derive(Serialize)]
struct RunReport<'a> {
    schema: &'static str,
    device: &'a str,
    state: &'static str,
    planned_cases: usize,
    planned_language_forwards: usize,
    unattempted_cases: Vec<&'a str>,
    language_forwards: u64,
    output_tokens: u64,
    results: &'a [CaseResult],
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir(&args.out)
        .context("output must be a fresh directory with an existing parent")?;
    let outcome = run(&args);
    if let Err(error) = &outcome {
        files::write_json(
            &args.out.join("failure.json"),
            &json!({"schema":"candle.d1_failure.v1","error":error.to_string()}),
        )?;
    }
    outcome
}

fn run(args: &Args) -> Result<()> {
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0).context("opening CUDA device 0")?,
        other => anyhow::bail!("unsupported device {other:?}; expected cpu or cuda"),
    };
    anyhow::ensure!(args.context > 0, "context must be positive");
    let inputs = files::read_bounded(&args.requests, 1024 * 1024)?;
    let text = std::str::from_utf8(&inputs)?;
    let cases: Vec<Input> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(
        !cases.is_empty() && cases.len() <= 64,
        "expected 1..=64 explicit cases"
    );
    let mut names = std::collections::HashSet::new();
    anyhow::ensure!(
        cases
            .iter()
            .all(|input| !input.request_id.trim().is_empty()
                && names.insert(input.request_id.as_str())),
        "case IDs must be nonempty and unique"
    );
    let planned: Vec<_> = cases
        .iter()
        .map(|input| (input.request_id.clone(), input.request.questions.0.len()))
        .collect();
    let mut retained = files::retain_artifacts(
        &args.artifact_manifest,
        &[
            &args.model,
            &args.mmproj,
            &args.tokenizer,
            &args.processor,
            &args.template,
        ],
    )?;
    files::write_json(
        &args.out.join("admission.json"),
        &json!({"schema":"candle.d1_artifact_admission.v1","artifacts":retained.identities,"device":args.device,"context":args.context,"cases":cases.len(),"requests_sha256":files::sha256(&inputs),"policy":candle_vlm::lfm2_d1::D1PolicyV1::ID}),
    )?;
    let load_started = Instant::now();
    let mut session = load_d1_q8(D1LoadOptions {
        hybrid: Lfm2VlHybridLoadOptions {
            text_gguf: &args.model,
            mmproj: Lfm2VlMmprojSource::GgufFile(&args.mmproj),
            tokenizer: &args.tokenizer,
            processor_config: Some(&args.processor),
            mmproj_execution: Lfm2VlMmprojExecution::Q8,
            vision_dtype: DType::F32,
            vision_device: &device,
            text_device: &device,
        },
        limits: D1Limits {
            max_context_tokens: Some(args.context),
            ..Default::default()
        },
    })?;
    files::write_json(
        &args.out.join("loaded.json"),
        &json!({"load_ms":load_started.elapsed().as_secs_f64()*1000.,"consumed_files":session.consumed_files(),"template":args.template,"policy_revision":candle_vlm::lfm2_d1::D1PolicyV1::REFERENCE_REVISION}),
    )?;
    let mut results = Vec::new();
    let mut total_forwards = 0u64;
    let mut failed = false;
    for (index, input) in cases.into_iter().enumerate() {
        let case_root = args.out.join(format!("case-{index:03}"));
        fs::create_dir(&case_root)?;
        let started = Instant::now();
        let images = input
            .images
            .iter()
            .map(|path| files::load_image(path))
            .collect::<Result<Vec<_>>>();
        let result = match images {
            Ok(images) => run_case(
                &mut session,
                &input,
                &images,
                &case_root,
                args.trace,
                !results.iter().any(|value: &CaseResult| value.had_image),
            ),
            Err(error) => Err(DecisionFailure {
                message: error.to_string(),
                question_index: None,
                partial: Box::new(DecisionResponse {
                    planned_questions: input.request.questions.0.len(),
                    unattempted_questions: input.request.questions.0.len(),
                    execution: session.execution_details(),
                    ..Default::default()
                }),
            }),
        };
        let (response, error, question_index) = match result {
            Ok(response) => (response, None, None),
            Err(error) => {
                failed = true;
                (*error.partial, Some(error.message), error.question_index)
            }
        };
        total_forwards += response.usage.language_forwards;
        let record = CaseResult {
            request_id: input.request_id,
            status: if failed { "failed" } else { "complete" },
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
            had_image: !input.images.is_empty(),
            error,
            question_index,
            response,
        };
        files::write_json(&case_root.join("result.json"), &record)?;
        println!("{}", serde_json::to_string(&record)?);
        results.push(record);
        if failed {
            break;
        }
    }
    files::write_json(
        &args.out.join("report.json"),
        &RunReport {
            schema: "candle.d1_run.v1",
            device: &args.device,
            state: if failed { "failed" } else { "complete" },
            planned_cases: planned.len(),
            planned_language_forwards: planned.iter().map(|(_, count)| *count).sum(),
            unattempted_cases: planned
                .iter()
                .skip(results.len())
                .map(|(id, _)| id.as_str())
                .collect(),
            language_forwards: total_forwards,
            output_tokens: 0,
            results: &results,
        },
    )?;
    drop(session);
    retained.recheck()?;
    if failed {
        anyhow::bail!("d1 run failed; partial results retained")
    }
    Ok(())
}

fn run_case(
    session: &mut candle_vlm::lfm2_d1::D1Session,
    input: &Input,
    images: &[image::DynamicImage],
    case_root: &std::path::Path,
    trace: bool,
    trace_images: bool,
) -> std::result::Result<DecisionResponse, DecisionFailure> {
    let preparation = (|| -> Result<()> {
        let plans = session.prepare(&input.request, images.len())?;
        files::write_json(
            &case_root.join("plans.json"),
            &json!({"request_id":input.request_id,"plans":plans.iter().map(|(name, plan, text)| json!({"name":name,"text":text,"groups":plan.token_groups()})).collect::<Vec<_>>()}),
        )?;
        Ok(())
    })();
    if let Err(error) = preparation {
        return Err(DecisionFailure {
            message: error.to_string(),
            question_index: None,
            partial: Box::new(DecisionResponse {
                planned_questions: input.request.questions.0.len(),
                unattempted_questions: input.request.questions.0.len(),
                execution: session.execution_details(),
                ..Default::default()
            }),
        });
    }
    session.decide_traced(
        &input.request,
        images,
        || false,
        |event| {
            if !trace {
                return Ok(());
            }
            let outcome = match event {
                D1TraceEvent::QuestionInput { index, text, ids } => files::write_json(
                    &case_root.join(format!("question-{index}.json")),
                    &json!({"text":text,"input_ids":ids}),
                ),
                D1TraceEvent::AnswerLogits { index, logits } => files::write_tensor(
                    &case_root.join(format!("logits-{index}.safetensors")),
                    "logits",
                    logits,
                ),
                D1TraceEvent::ProcessedImages(processed) if trace_images => files::write_tensors(
                    &case_root.join("processor.safetensors"),
                    &[
                        ("pixels", &processed.pixel_values),
                        ("mask", &processed.pixel_attention_mask),
                        ("spatial", &processed.spatial_shapes),
                    ],
                ),
                D1TraceEvent::ImageFeatures(encoded) if trace_images => files::write_tensor(
                    &case_root.join("features.safetensors"),
                    "features",
                    &encoded.embeddings,
                ),
                _ => Ok(()),
            };
            outcome.map_err(|error| candle::Error::Msg(error.to_string()))
        },
    )
}
