use super::*;
use candle::{Device, Tensor};
use sha2::{Digest, Sha256};
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
};

fn tokenizer() -> Tokenizer {
    let tokens = [
        "<unk>",
        "<|startoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "A",
        "B",
        "C",
        "yes",
        "Yes",
        "YES",
        "no",
        "No",
        "NO",
        "0",
        "1",
        "2",
    ];
    let vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".into())
        .build()
        .expect("authored vocabulary");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<|startoftext|>", true),
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ])
        .expect("authored markers");
    tokenizer
}

#[test]
fn d1_load_rejects_invalid_limits_before_opening_inputs() -> candle::Result<()> {
    use crate::lfm2_vl::{Lfm2VlHybridLoadOptions, Lfm2VlMmprojExecution, Lfm2VlMmprojSource};
    let device = Device::Cpu;
    let absent = std::path::Path::new("");
    let hybrid = Lfm2VlHybridLoadOptions {
        text_gguf: absent,
        mmproj: Lfm2VlMmprojSource::GgufFile(absent),
        tokenizer: absent,
        processor_config: None,
        mmproj_execution: Lfm2VlMmprojExecution::Q8,
        vision_dtype: candle::DType::F32,
        vision_device: &device,
        text_device: &device,
    };
    for limits in [
        D1Limits {
            max_questions: 0,
            ..Default::default()
        },
        D1Limits {
            max_options: 0,
            ..Default::default()
        },
        D1Limits {
            max_prompt_bytes: 0,
            ..Default::default()
        },
        D1Limits {
            max_context_tokens: Some(0),
            ..Default::default()
        },
    ] {
        let error = load_d1_q8(D1LoadOptions { hybrid, limits })
            .err()
            .ok_or_else(|| candle::Error::Msg("invalid limits were accepted".into()))?;
        assert!(
            error.to_string().contains("d1 limits must be positive"),
            "{error}"
        );
    }
    // A zero image allowance is valid for callers that only accept text.
    let error = load_d1_q8(D1LoadOptions {
        hybrid,
        limits: D1Limits {
            max_images: 0,
            ..Default::default()
        },
    })
    .err()
    .ok_or_else(|| candle::Error::Msg("absent tokenizer was accepted".into()))?;
    assert!(
        error.to_string().contains("cannot open tokenizer"),
        "{error}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CANDLE_D1_READOUT_ROOT containing the retained CPU/CUDA study logits"]
fn d1_retained_readout_matches_published_answers() -> candle::Result<()> {
    #[derive(serde::Deserialize)]
    struct Plan {
        name: String,
        groups: Vec<Vec<u32>>,
    }
    #[derive(serde::Deserialize)]
    struct Plans {
        plans: Vec<Plan>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        response: DecisionResponse,
    }
    #[derive(serde::Deserialize)]
    struct Run {
        state: String,
        language_forwards: u64,
        output_tokens: u64,
        results: Vec<Case>,
    }
    fn read<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> candle::Result<T> {
        serde_json::from_slice(&std::fs::read(path)?).map_err(candle::Error::wrap)
    }
    let root = std::path::PathBuf::from(
        std::env::var("CANDLE_D1_READOUT_ROOT").map_err(candle::Error::wrap)?,
    );
    if !root.is_absolute() {
        candle::bail!("retained readout root must be absolute")
    }
    let mut maximum = 0f64;
    let mut checked = 0;
    for name in ["cpu-study-final", "cuda-study"] {
        let run_root = root.join(name);
        let run: Run = read(&run_root.join("report.json"))?;
        assert_eq!(run.state, "complete");
        assert_eq!(run.results.len(), 20);
        assert_eq!(run.language_forwards, 60);
        assert_eq!(run.output_tokens, 0);
        for (case_index, case) in run.results.iter().enumerate() {
            let case_root = run_root.join(format!("case-{case_index:03}"));
            let plans: Plans = read(&case_root.join("plans.json"))?;
            assert_eq!(plans.plans.len(), case.response.answers.0.len());
            for (index, (plan, (name, answer))) in
                plans.plans.iter().zip(&case.response.answers.0).enumerate()
            {
                assert_eq!(&plan.name, name);
                let tensors = candle::safetensors::load(
                    case_root.join(format!("logits-{index}.safetensors")),
                    &Device::Cpu,
                )?;
                let logits = tensors
                    .get("logits")
                    .ok_or_else(|| candle::Error::Msg("retained logits are missing".into()))?;
                let actual = candle_transformers::models::lfm2_d1::option_probabilities(
                    logits,
                    &plan.groups,
                )?;
                let expected = match answer {
                    Answer::Noul { noul } => {
                        assert_eq!(actual[0] >= 0.5, *noul >= 0.5);
                        vec![*noul, 1. - noul]
                    }
                    Answer::Choice {
                        choice,
                        confidence,
                        probabilities,
                        ..
                    } => {
                        let best = actual.iter().enumerate().fold(0, |best, (i, p)| {
                            if *p > actual[best] {
                                i
                            } else {
                                best
                            }
                        });
                        assert_eq!(&probabilities.0[best].0, choice);
                        maximum = maximum.max((actual[best] - confidence).abs());
                        probabilities.0.iter().map(|(_, p)| *p).collect()
                    }
                    Answer::Score {
                        score,
                        confidence,
                        probabilities,
                        ..
                    } => {
                        let expected_score: f64 =
                            actual.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
                        maximum = maximum.max((expected_score - score).abs());
                        let best = actual.iter().copied().fold(0f64, f64::max);
                        maximum = maximum.max((best - confidence).abs());
                        probabilities.0.iter().map(|(_, p)| *p).collect()
                    }
                };
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(expected) {
                    maximum = maximum.max((a - b).abs());
                }
                assert!(maximum <= 1e-6, "retained readout error {maximum}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 120);
    println!(
        "d1-readout replay_questions={checked} max_abs={maximum:e} bound=1e-6 model_forwards=0"
    );
    Ok(())
}

#[test]
fn d1_fixture_checkout_identity() -> candle::Result<()> {
    #[derive(serde::Deserialize)]
    struct FileIdentity {
        bytes: usize,
        sha256: String,
    }
    #[derive(serde::Deserialize)]
    struct Manifest {
        schema: String,
        files: std::collections::HashMap<String, FileIdentity>,
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_d1_tiny");
    let manifest_bytes = std::fs::read(root.join("manifest.json"))?;
    assert!(
        !manifest_bytes.contains(&b'\r'),
        "manifest must use LF bytes"
    );
    let manifest: Manifest =
        serde_json::from_slice(&manifest_bytes).map_err(candle::Error::wrap)?;
    assert_eq!(manifest.schema, "candle.d1_fixture.v1");
    let files = [
        "README.md",
        "bicubic.json",
        "policy.json",
        "tokenizer.json",
        "torchvision-bicubic.json",
    ];
    assert_eq!(manifest.files.len(), files.len());
    for name in files {
        let bytes = std::fs::read(root.join(name))?;
        assert!(!bytes.contains(&b'\r'), "{name} must use LF bytes");
        let expected = manifest
            .files
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("fixture identity missing for {name}")))?;
        assert_eq!(bytes.len(), expected.bytes, "{name} size");
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected.sha256,
            "{name} SHA-256"
        );
    }
    Ok(())
}

#[test]
fn d1_pinned_reference_goldens() -> candle::Result<()> {
    #[derive(serde::Deserialize)]
    struct Golden {
        name: String,
        state: State,
        question: Question,
        images: usize,
        text: String,
        input_ids: Vec<u32>,
        groups: Vec<Vec<u32>>,
        logits: Vec<f32>,
        probabilities: Vec<f64>,
    }
    #[derive(serde::Deserialize)]
    struct Suite {
        tokenizer_sha256: String,
        cases: Vec<Golden>,
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_d1_tiny");
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    let suite: Suite = serde_json::from_slice(&std::fs::read(root.join("policy.json"))?)
        .map_err(candle::Error::wrap)?;
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(std::fs::read(root.join("tokenizer.json"))?)
        ),
        suite.tokenizer_sha256,
        "pinned fixture tokenizer SHA-256"
    );
    for case in suite.cases {
        let plan = D1PolicyV1.prepare(&tokenizer, &case.question, &D1Limits::default())?;
        assert_eq!(plan.token_groups(), case.groups, "{} aliases", case.name);
        let text = D1PolicyV1.render(&tokenizer, &case.state, &plan, case.images, 4096)?;
        assert_eq!(text, case.text, "{} prompt", case.name);
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert_eq!(encoding.get_ids(), case.input_ids, "{} tokens", case.name);
        let logits = Tensor::new(case.logits.as_slice(), &Device::Cpu)?;
        let answer = plan.readout(&logits)?;
        let actual = match answer {
            Answer::Noul { noul } => vec![noul, 1. - noul],
            Answer::Choice { probabilities, .. } => {
                probabilities.0.into_iter().map(|(_, p)| p).collect()
            }
            Answer::Score {
                score,
                probabilities,
                ..
            } => {
                let expected = case
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum::<f64>();
                assert!((score - expected).abs() <= 1e-6);
                probabilities.0.into_iter().map(|(_, p)| p).collect()
            }
        };
        assert_eq!(actual.len(), case.probabilities.len());
        for (actual, expected) in actual.iter().zip(case.probabilities) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "{} probability",
                case.name
            );
        }
    }
    Ok(())
}

#[test]
fn d1_malformed_questions_and_byte_limits_fail_before_execution() -> candle::Result<()> {
    let tokenizer = tokenizer();
    let malformed = [
        r#"{"type":"score","instructions":"Rate","criteria":["only"]}"#,
        r#"{"type":"choice","instructions":"Pick","criteria":{}}"#,
        r#"{"type":"noul","instructions":"Ready?","criteria":{"invalid":"text"}}"#,
        r#"{"type":"noul","instructions":" "}"#,
    ];
    for raw in malformed {
        let question: Question = serde_json::from_str(raw).map_err(candle::Error::wrap)?;
        assert!(D1PolicyV1
            .prepare(&tokenizer, &question, &D1Limits::default())
            .is_err());
    }
    let plan = D1PolicyV1.prepare(
        &tokenizer,
        &request().questions.0[0].1,
        &D1Limits::default(),
    )?;
    let state = State::Object(OrderedMap(vec![(
        "escaped".into(),
        State::Text("\0".repeat(200)),
    )]));
    assert!(D1PolicyV1
        .render(&tokenizer, &state, &plan, 0, 1024)
        .is_err());
    assert!(D1PolicyV1
        .render(&tokenizer, &State::Null, &plan, usize::MAX, 1024)
        .is_err());
    assert!(DecisionRequest::from_json(br#"{"state":null,"questions":{"q":{"type":"noul","instructions":"Ready?"},"q":{"type":"noul","instructions":"Ready?"}}}"#, 4096).is_err());
    Ok(())
}

#[test]
fn d1_typed_noul_duplicates_are_rejected_before_any_forward() -> candle::Result<()> {
    let mut session = tiny_session(&Device::Cpu, D1Limits::default())?;
    let question = Question::Noul {
        instructions: "Ready?".into(),
        criteria: Some(OrderedMap(vec![
            ("true".into(), Some("first".into())),
            ("true".into(), Some("contradictory duplicate".into())),
        ])),
    };
    let request = DecisionRequest {
        state: State::Null,
        questions: OrderedMap(vec![("ready".into(), question)]),
    };
    let failed = session
        .decide(&request, &[])
        .expect_err("typed duplicate must fail");
    assert!(failed.message.contains("unique"));
    assert_eq!(failed.partial.usage.language_forwards, 0);
    assert_eq!(failed.partial.usage.vision_forwards, 0);
    assert_eq!(failed.partial.unattempted_questions, 1);
    Ok(())
}

#[test]
fn d1_question_byte_limits_precede_tokenizer_alias_work() {
    let tokenizer = Tokenizer::new(WordLevel::default());
    let limits = D1Limits {
        max_prompt_bytes: 64,
        ..Default::default()
    };
    let questions = [
        Question::Noul {
            instructions: "x".repeat(65),
            criteria: None,
        },
        Question::Choice {
            instructions: "Pick".into(),
            criteria: OrderedMap(vec![("a".into(), Some("x".repeat(65)))]),
        },
        Question::Score {
            instructions: "Rate".into(),
            criteria: vec!["x".repeat(65), "high".into()],
        },
    ];
    for question in questions {
        let error = D1PolicyV1
            .prepare(&tokenizer, &question, &limits)
            .expect_err("payload limit must precede an unusable tokenizer");
        assert!(error.to_string().contains("prompt byte limit"), "{error}");
    }
}

#[test]
fn d1_choice_prompt_limit_excludes_unrendered_explicit_label() -> candle::Result<()> {
    let question = Question::Choice {
        instructions: "Pick".into(),
        criteria: OrderedMap(vec![("x".repeat(4096), Some("brief".into()))]),
    };
    let limits = D1Limits {
        max_prompt_bytes: 128,
        ..Default::default()
    };
    let plan = D1PolicyV1.prepare(&tokenizer(), &question, &limits)?;
    assert!(plan.question_text().contains("A brief"));
    Ok(())
}

#[test]
fn d1_pixel_cap_matches_pillow_bicubic_bytes() -> candle::Result<()> {
    #[derive(serde::Deserialize)]
    struct Case {
        width: u32,
        height: u32,
        cap: u64,
        output_width: u32,
        output_height: u32,
        rgb: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Suite {
        cases: Vec<Case>,
    }
    let suite: Suite = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/lfm2_d1_tiny/bicubic.json"
    ))
    .map_err(candle::Error::wrap)?;
    for case in suite.cases {
        let image = image::RgbImage::from_fn(case.width, case.height, |x, y| {
            image::Rgb([
                ((x * 37 + y * 19) % 256) as u8,
                ((x * 17 + y * 71) % 256) as u8,
                ((x * 91 + y * 13) % 256) as u8,
            ])
        });
        let output =
            super::image_cap::cap_pixels(&image::DynamicImage::ImageRgb8(image), case.cap)?
                .into_rgb8();
        assert_eq!(output.dimensions(), (case.output_width, case.output_height));
        assert_eq!(output.as_raw(), &case.rgb);
    }
    let image = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([32, 64, 96, 0]),
    ));
    assert_eq!(
        super::image_cap::cap_pixels(&image, 4)?
            .into_rgb8()
            .get_pixel(0, 0)
            .0,
        [32, 64, 96]
    );
    assert!(super::image_cap::cap_pixels(&image, 0).is_err());
    Ok(())
}

#[test]
fn d1_processor_resize_matches_pinned_torchvision_bicubic() -> candle::Result<()> {
    #[derive(serde::Deserialize)]
    struct Case {
        width: u32,
        height: u32,
        output_width: usize,
        output_height: usize,
        indices: Vec<usize>,
        rgb: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Suite {
        cases: Vec<Case>,
    }
    let suite: Suite = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/lfm2_d1_tiny/torchvision-bicubic.json"
    ))
    .map_err(candle::Error::wrap)?;
    for case in suite.cases {
        let image = image::RgbImage::from_fn(case.width, case.height, |x, y| {
            image::Rgb([
                ((x * 37 + y * 19) % 256) as u8,
                ((x * 17 + y * 71) % 256) as u8,
                ((x * 91 + y * 13) % 256) as u8,
            ])
        });
        let output =
            crate::image::resize_bicubic_antialias(&image, case.output_width, case.output_height)?;
        assert_eq!(
            case.indices
                .iter()
                .map(|&i| output.as_raw()[i])
                .collect::<Vec<_>>(),
            case.rgb,
            "{}x{} resize",
            case.width,
            case.height
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit retained metadata, frozen image inputs and fresh ignored output"]
fn d1_retained_processor_slice() -> candle::Result<()> {
    use std::{collections::HashSet, fs, path::PathBuf};
    let variable = |name| {
        std::env::var(name)
            .map(PathBuf::from)
            .map_err(candle::Error::wrap)
    };
    let metadata = variable("CANDLE_D1_PROCESSOR")?;
    let requests = variable("CANDLE_D1_PROCESSOR_REQUESTS")?;
    let out = variable("CANDLE_D1_PROCESSOR_OUT")?;
    let processor = crate::lfm2_vl::Lfm2VlProcessor::from_json(
        &fs::read_to_string(metadata).map_err(candle::Error::wrap)?,
    )?
    .with_bicubic_resize();
    fs::create_dir(&out).map_err(candle::Error::wrap)?;
    let mut seen = HashSet::new();
    for line in fs::read_to_string(requests)
        .map_err(candle::Error::wrap)?
        .lines()
    {
        let case: serde_json::Value = serde_json::from_str(line).map_err(candle::Error::wrap)?;
        for path in case["images"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("missing frozen image array".into()))?
        {
            let path = path
                .as_str()
                .ok_or_else(|| candle::Error::Msg("invalid frozen image path".into()))?;
            if !seen.insert(path.to_owned()) {
                continue;
            }
            let image = image::open(path).map_err(candle::Error::wrap)?;
            let capped = super::image_cap::cap_pixels(&image, super::image_cap::MAX_PIXELS)?;
            let processed = processor.process(&[capped], &Device::Cpu)?;
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| candle::Error::Msg("invalid frozen image filename".into()))?;
            let tensors = std::collections::HashMap::from([
                ("pixels".to_owned(), processed.pixel_values),
                ("mask".to_owned(), processed.pixel_attention_mask),
                ("spatial".to_owned(), processed.spatial_shapes),
            ]);
            candle::safetensors::save(&tensors, out.join(format!("{stem}.safetensors")))?;
        }
    }
    assert_eq!(
        seen.len(),
        4,
        "corrected study requires four retained images"
    );
    Ok(())
}

#[test]
fn d1_ordered_state_and_questions_keep_reference_prompt() -> candle::Result<()> {
    let request = DecisionRequest::from_json(
        br#"{"state":{"z":1,"a":2},"questions":{"q":{"type":"noul","instructions":"Ready?"}}}"#,
        4096,
    )?;
    let tok = tokenizer();
    let plan = D1PolicyV1.prepare(&tok, &request.questions.0[0].1, &D1Limits::default())?;
    let text = D1PolicyV1.render(&tok, &request.state, &plan, 0, 4096)?;
    assert_eq!(text, "<|startoftext|><|im_start|>user\n{\n  \"z\": 1,\n  \"a\": 2\n}\n\n\nQUESTION:\nReady?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n");
    assert!(
        DecisionRequest::from_json(br#"{"state":{"a":1,"a":2},"questions":{}}"#, 4096).is_err()
    );
    Ok(())
}

#[test]
fn d1_typed_answers_use_yes_probability_and_expected_score() -> candle::Result<()> {
    let tok = tokenizer();
    let limits = D1Limits::default();
    let logits = Tensor::zeros(16, candle::DType::F32, &Device::Cpu)?;
    let choice = Question::Choice {
        instructions: "Pick".into(),
        criteria: OrderedMap(vec![
            ("second".into(), Some("second option".into())),
            ("first".into(), None),
        ]),
    };
    let plan = D1PolicyV1.prepare(&tok, &choice, &limits)?;
    match plan.readout(&logits)? {
        Answer::Choice {
            choice, confidence, ..
        } => {
            assert_eq!(choice, "second");
            assert_eq!(confidence, 0.5);
        }
        other => panic!("unexpected answer {other:?}"),
    }
    let score = D1PolicyV1.prepare(
        &tok,
        &Question::Score {
            instructions: "Rate".into(),
            criteria: vec!["low".into(), "middle".into(), "high".into()],
        },
        &limits,
    )?;
    match score.readout(&logits)? {
        Answer::Score { score, .. } => assert!((score - 1.).abs() < 1e-12),
        other => panic!("unexpected answer {other:?}"),
    }
    Ok(())
}

fn tiny_session(device: &Device, limits: D1Limits) -> candle::Result<D1Session> {
    use std::path::Path;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");
    use crate::lfm2_vl::{Lfm2VlProcessor, Lfm2VlPrompt, LoadedLfm2VlHybrid, PromptOptions};
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_transformers::models::{
        lfm2_vl::{Mmproj, QuantizedLfm2VlModel},
        quantized_lfm2,
    };
    let mut reader =
        std::io::Cursor::new(std::fs::read(root.join("lfm2_vl_loader_tiny/text.gguf"))?);
    let mut content = gguf_file::Content::read(&mut reader)?;
    // The older hybrid fixture has width 12, which cannot store Q8 rows.
    // Author a block-aligned text tower; preserve its conv/attention topology.
    content
        .metadata
        .insert("lfm2.embedding_length".into(), gguf_file::Value::U32(32));
    content
        .metadata
        .insert("lfm2.attention.head_count".into(), gguf_file::Value::U32(4));
    content
        .metadata
        .insert("lfm2.context_length".into(), gguf_file::Value::U32(512));
    let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut tensors = Vec::new();
    for name in names {
        let tensor = content.tensor(&mut reader, &name, &Device::Cpu)?;
        let dims: Vec<_> = tensor
            .shape()
            .dims()
            .iter()
            .map(|&dim| match dim {
                12 => 32,
                36 => 96,
                4 => 8,
                other => other,
            })
            .collect();
        let count = dims.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| {
                if name.contains("norm") {
                    1.
                } else {
                    ((i % 19) as f32 - 9.) / 100.
                }
            })
            .collect();
        let dense = Tensor::from_vec(values, dims, &Device::Cpu)?;
        let dtype = if dense.rank() == 2 {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        let tensor = QTensor::quantize(&dense, dtype)?;
        tensors.push((name, tensor));
    }
    let tensor_refs: Vec<_> = tensors
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    let mut metadata: Vec<_> = content
        .metadata
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    metadata.sort_by_key(|(name, _)| *name);
    let mut bytes = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, &metadata, &tensor_refs)?;
    bytes.set_position(0);
    let content = gguf_file::Content::read(&mut bytes)?;
    let text = quantized_lfm2::ModelWeights::from_gguf(content, &mut bytes, device)?;
    let processor = Lfm2VlProcessor::from_json(&std::fs::read_to_string(
        root.join("lfm2_vl_mmproj_tiny/processor_config.json"),
    )?)?;
    let config = processor.config();
    let mut reader = std::io::Cursor::new(std::fs::read(
        root.join("lfm2_vl_loader_tiny/mmproj-q8.gguf"),
    )?);
    let mut content = gguf_file::Content::read(&mut reader)?;
    content.metadata.insert(
        "clip.vision.projection_dim".into(),
        gguf_file::Value::U32(32),
    );
    let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut tensors = Vec::new();
    for name in names {
        let tensor = match name.as_str() {
            "mm.2.weight" => QTensor::quantize(
                &Tensor::zeros((32, 24), candle::DType::F32, &Device::Cpu)?,
                GgmlDType::F32,
            )?,
            "mm.2.bias" => QTensor::quantize(
                &Tensor::zeros(32, candle::DType::F32, &Device::Cpu)?,
                GgmlDType::F32,
            )?,
            _ => content.tensor(&mut reader, &name, &Device::Cpu)?,
        };
        tensors.push((name, tensor));
    }
    let tensor_refs: Vec<_> = tensors
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    let mut metadata: Vec<_> = content
        .metadata
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    metadata.sort_by_key(|(name, _)| *name);
    let mut bytes = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, &metadata, &tensor_refs)?;
    bytes.set_position(0);
    let mmproj = Mmproj::from_gguf_q8(&mut bytes, candle::DType::F32, device, 3)?;
    let model = QuantizedLfm2VlModel::new(
        text,
        mmproj,
        config.encoder_patch_size,
        config.downsample_factor,
        3,
    )?;
    let mut tokenizer = Tokenizer::from_file(root.join("lfm2_d1_tiny/tokenizer.json"))
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 4,
            ..Default::default()
        }))
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    let prompt =
        Lfm2VlPrompt::from_processor_config(tokenizer, Some(3), config, PromptOptions::default())?;
    D1Session::from_hybrid(
        LoadedLfm2VlHybrid {
            model,
            processor,
            prompt,
            consumed_files: Vec::new(),
        },
        D1PolicyV1,
        limits,
    )
}

fn request() -> DecisionRequest {
    DecisionRequest {
        state: State::Text("hello".into()),
        questions: OrderedMap(vec![
            (
                "ready".into(),
                Question::Noul {
                    instructions: "Ready?".into(),
                    criteria: None,
                },
            ),
            (
                "pick".into(),
                Question::Choice {
                    instructions: "Pick".into(),
                    criteria: OrderedMap(vec![
                        ("a".into(), Some("hello".into())),
                        ("b".into(), Some("other".into())),
                    ]),
                },
            ),
        ]),
    }
}

#[test]
fn d1_independent_questions_reset_and_replay_exactly() -> candle::Result<()> {
    let mut session = tiny_session(&Device::Cpu, D1Limits::default())?;
    assert!(session.tokenizer().get_padding().is_none());
    assert!(session.tokenizer().get_truncation().is_none());
    let request = request();
    let response = session
        .decide(&request, &[])
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(response.usage.language_forwards, 2);
    assert_eq!(response.usage.output_tokens, 0);
    assert!(response.execution.cpu_q8_matmuls > 0);
    for (name, question) in &request.questions.0 {
        let single = DecisionRequest {
            state: request.state.clone(),
            questions: OrderedMap(vec![(name.clone(), question.clone())]),
        };
        let answer = session
            .decide(&single, &[])
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert_eq!(
            response.answers.0.iter().find(|entry| &entry.0 == name),
            answer.answers.0.first()
        );
    }
    assert_eq!(
        response,
        session
            .decide(&request, &[])
            .map_err(|e| candle::Error::Msg(e.to_string()))?
    );
    Ok(())
}

#[test]
fn d1_partial_cancel_preserves_answer_counters_and_releases_cache() -> candle::Result<()> {
    use std::cell::Cell;
    let mut session = tiny_session(&Device::Cpu, D1Limits::default())?;
    let request = request();
    let completed = Cell::new(false);
    let failure = session
        .decide_traced(
            &request,
            &[],
            || completed.get(),
            |event| {
                if matches!(event, D1TraceEvent::AnswerLogits { .. }) {
                    completed.set(true);
                }
                Ok(())
            },
        )
        .expect_err("authored cancellation before second question");
    assert_eq!(failure.partial.answers.0.len(), 1);
    assert_eq!(failure.partial.usage.language_forwards, 1);
    assert_eq!(failure.partial.unattempted_questions, 1);
    let (_, dispatch) = candle::quantized::with_native_q8_0(|| Ok(()));
    assert_eq!(dispatch, candle::quantized::NativeQ8Execution::default());
    let recovered = session
        .decide(&request, &[])
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(
        recovered.answers.0.first(),
        failure.partial.answers.0.first()
    );
    Ok(())
}

#[test]
fn d1_context_rejection_precedes_all_model_forwards() -> candle::Result<()> {
    let mut session = tiny_session(
        &Device::Cpu,
        D1Limits {
            max_context_tokens: Some(1),
            ..Default::default()
        },
    )?;
    let failure = session
        .decide(&request(), &[])
        .expect_err("bounded context rejection");
    assert_eq!(failure.partial.usage.language_forwards, 0);
    assert_eq!(failure.partial.usage.vision_forwards, 0);
    assert_eq!(failure.partial.unattempted_questions, 2);
    Ok(())
}

#[test]
fn d1_profiling_preserves_answers_failure_counts_and_recovery() -> candle::Result<()> {
    check_profiling(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn d1_cuda_profiling_preserves_answers_failure_counts_and_recovery() -> candle::Result<()> {
    check_profiling(&Device::new_cuda(0)?)
}

fn check_profiling(device: &Device) -> candle::Result<()> {
    let mut session = tiny_session(device, D1Limits::default())?;
    let request = request();
    let expected = session
        .decide(&request, &[])
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    let mut timings = super::D1Timings::default();
    let actual = session
        .decide_traced_profiled(&request, &[], || false, |_| Ok(()), &mut timings)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(actual.answers, expected.answers);
    assert_eq!(actual.usage, expected.usage);
    assert_eq!(timings.questions.len(), request.questions.0.len());
    assert_eq!(timings.vision_ms, 0.);
    assert!(timings.total_ms.is_finite() && timings.total_ms >= timings.observation_ms);
    let failure = session
        .decide_traced_profiled(
            &request,
            &[],
            || false,
            |event| {
                if matches!(event, super::D1TraceEvent::QuestionInput { index: 1, .. }) {
                    candle::bail!("authored observer failure")
                }
                Ok(())
            },
            &mut timings,
        )
        .expect_err("profiled observer failure");
    assert_eq!(failure.partial.usage.language_forwards, 1);
    assert_eq!(failure.partial.answers.0.len(), 1);
    assert_eq!(timings.questions.len(), 2);
    assert_eq!(timings.questions[1].prefill_ms, 0.);
    let recovered = session
        .decide_traced_profiled(&request, &[], || false, |_| Ok(()), &mut timings)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(recovered.answers, expected.answers);
    assert_eq!(timings.questions.len(), request.questions.0.len());
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        8,
        8,
        image::Rgb([64, 128, 192]),
    ));
    let expected = session
        .decide(&request, std::slice::from_ref(&image))
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    let actual = session
        .decide_traced_profiled(&request, &[image], || false, |_| Ok(()), &mut timings)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(actual.answers, expected.answers);
    assert_eq!(actual.usage, expected.usage);
    assert_eq!(actual.usage.vision_forwards, 1);
    assert!(timings.vision_ms.is_finite());
    let cancelled = session
        .decide_traced_profiled(&request, &[], || true, |_| Ok(()), &mut timings)
        .expect_err("profiled cancellation");
    assert_eq!(cancelled.partial.usage.language_forwards, 0);
    assert_eq!(cancelled.partial.usage.vision_forwards, 0);
    assert!(timings.questions.is_empty());
    assert_eq!(timings.vision_ms, 0.);
    Ok(())
}

#[test]
fn d1_image_features_are_encoded_once_for_independent_questions() -> candle::Result<()> {
    let mut session = tiny_session(&Device::Cpu, D1Limits::default())?;
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        8,
        8,
        image::Rgb([64, 128, 192]),
    ));
    let response = session
        .decide(&request(), &[image])
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(response.usage.vision_forwards, 1);
    assert_eq!(response.usage.language_forwards, 2);
    assert_eq!(response.answers.0.len(), 2);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn d1_cuda_question_isolation_image_reuse_and_partial_failure() -> candle::Result<()> {
    let mut session = tiny_session(&Device::new_cuda(0)?, D1Limits::default())?;
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        8,
        8,
        image::Rgb([64, 128, 192]),
    ));
    let request = request();
    let response = session
        .decide(&request, &[image.clone()])
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    assert_eq!(response.usage.vision_forwards, 1);
    assert_eq!(response.usage.language_forwards, 2);
    assert!(response.execution.cuda_f32_q8 > 0);
    assert_eq!(response.execution.cpu_q8_matmuls, 0);
    for (name, question) in &request.questions.0 {
        let single = DecisionRequest {
            state: request.state.clone(),
            questions: OrderedMap(vec![(name.clone(), question.clone())]),
        };
        let result = session
            .decide(&single, &[image.clone()])
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert_eq!(
            response.answers.0.iter().find(|entry| &entry.0 == name),
            result.answers.0.first()
        );
    }
    let failed = session
        .decide_traced(
            &request,
            &[],
            || false,
            |event| {
                if matches!(event, D1TraceEvent::QuestionInput { index: 1, .. }) {
                    candle::bail!("authored second-question trace failure")
                }
                Ok(())
            },
        )
        .expect_err("authored partial failure");
    assert_eq!(failed.partial.usage.language_forwards, 1);
    assert_eq!(failed.partial.answers.0.len(), 1);
    assert_eq!(failed.partial.unattempted_questions, 1);
    assert!(session.decide(&request, &[]).is_ok());
    Ok(())
}
