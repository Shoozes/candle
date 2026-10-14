use super::{
    input::*,
    runner::{walk, Progress},
};
use candle::{bail, Result};
use candle_transformers::models::gpt_oss::{
    mxfp4::{Mxfp4ExpertOperation, PackedMxfp4},
    DenseLinear, GptOssCancellationToken, GptOssConfig, GptOssForwardTimings, GptOssLayerWeights,
    GptOssModel, GptOssResourceUsage, GptOssWeights,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct OwnedFixture(PathBuf);
impl OwnedFixture {
    fn new() -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "candle-capacity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
    fn input(&self) -> Result<PathBuf> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/gpt_oss_task3_short_parity/reference.json"
        ))
        .map_err(candle::Error::wrap)?;
        let prefix: Vec<u32> =
            serde_json::from_value(fixture["reference_prefix_token_ids"].clone())
                .map_err(candle::Error::wrap)?;
        let bytes: Vec<u8> = prefix
            .iter()
            .cycle()
            .take(TOTAL)
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let token_file = self.0.join("tokens.u32");
        fs::write(&token_file, &bytes)?;
        let input = Input {
            schema: "candle.gpt-oss.capacity-input.v1".into(),
            source_commit: "a".repeat(40),
            backend: Backend::Cpu,
            device_index: 0,
            model: Identity {
                path: self.0.join("absent-model.gguf"),
                size_bytes: MODEL_BYTES,
                sha256: MODEL_SHA.into(),
            },
            token_file: Identity {
                path: token_file,
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            },
            prefill_tokens: PREFILL,
            prefill_chunk_tokens: CHUNK,
            fixed_decode_tokens: DECODES,
            sampled_outputs: 0,
        };
        let path = self.0.join("input.json");
        fs::write(
            &path,
            serde_json::to_vec(&input).map_err(candle::Error::wrap)?,
        )?;
        Ok(path)
    }
}
impl Drop for OwnedFixture {
    fn drop(&mut self) {
        for name in ["input.json", "tokens.u32", "progress.json"] {
            let _ = fs::remove_file(self.0.join(name));
        }
        let _ = fs::remove_dir(&self.0); // Nonrecursive: unknown contents are preserved.
    }
}

fn timing() -> GptOssForwardTimings {
    GptOssForwardTimings {
        forward_ms: 1.,
        cuda_final_validation_ms: None,
    }
}

#[test]
fn input_admission_is_model_free_and_rejects_changed_work_and_wire_bytes() -> Result<()> {
    let fixture = OwnedFixture::new()?;
    let path = fixture.input()?;
    let (mut input, _, tokens) = read_input(&path)?;
    assert_eq!(tokens.len(), TOTAL);
    assert!(!input.model.path.exists());
    input.sampled_outputs = 1;
    assert!(input.validate().is_err());
    input.sampled_outputs = 0;
    input.prefill_chunk_tokens = 0;
    assert!(input.validate().is_err());
    input.prefill_chunk_tokens = CHUNK;
    input.source_commit = "x".repeat(40);
    assert!(input.validate().is_err());
    input.source_commit = "a".repeat(40);
    input.model.size_bytes += 1;
    assert!(input.validate().is_err());
    input.model.size_bytes -= 1;
    let mut bytes = fs::read(&input.token_file.path)?;
    bytes[0] ^= 1;
    fs::write(&input.token_file.path, &bytes)?;
    assert!(read_input(&path).is_err());
    input.token_file.sha256 = format!("{:x}", Sha256::digest(&bytes));
    fs::write(
        &path,
        serde_json::to_vec(&input).map_err(candle::Error::wrap)?,
    )?;
    assert!(read_input(&path).is_err()); // Matching digest cannot admit a different prefix.
    fs::write(&input.token_file.path, &bytes[..bytes.len() - 1])?;
    assert!(read_input(&path).is_err());
    bytes.push(0);
    fs::write(&input.token_file.path, &bytes)?;
    assert!(read_input(&path).is_err());
    assert!(read_bounded(&path, usize::MAX).is_err());
    let mut value = serde_json::to_value(&input).map_err(candle::Error::wrap)?;
    value["unknown"] = serde_json::json!(1);
    assert!(serde_json::from_value::<Input>(value).is_err());
    Ok(())
}

#[test]
fn relative_paths_are_rejected_before_token_or_model_io() -> Result<()> {
    let fixture = OwnedFixture::new()?;
    let path = fixture.input()?;
    let (mut input, _, _) = read_input(&path)?;
    let model_path = input.model.path.clone();
    for relative_model in [true, false] {
        input.model.path = if relative_model {
            PathBuf::from("relative-model.gguf")
        } else {
            model_path.clone()
        };
        input.token_file.path = if relative_model {
            fixture.0.join("absent-tokens.u32")
        } else {
            PathBuf::from("relative-tokens.u32")
        };
        fs::write(
            &path,
            serde_json::to_vec(&input).map_err(candle::Error::wrap)?,
        )?;
        let error = read_input(&path).expect_err("relative paths must fail admission");
        assert!(error
            .to_string()
            .contains("explicit absolute model and token paths"));
    }
    Ok(())
}

#[test]
fn frozen_trace_has_exact_prefill_decode_positions_and_no_sampling() -> Result<()> {
    let tokens: Vec<u32> = (0..TOTAL as u32).collect();
    let mut p = Progress::default();
    let mut position = 0;
    let mut sizes = Vec::new();
    walk(
        &tokens,
        PREFILL,
        CHUNK,
        &GptOssCancellationToken::new(),
        &mut p,
        |span, prefill| {
            assert_eq!(span.first().copied(), Some(position as u32));
            assert_eq!(prefill, position < PREFILL);
            sizes.push(span.len());
            position += span.len();
            Ok((
                GptOssResourceUsage {
                    sequence_tokens: position,
                    cache_bytes: position,
                    cache_capacity_bytes: position,
                },
                timing(),
            ))
        },
        |_| Ok(()),
    )?;
    assert_eq!(sizes[..48], [CHUNK; 48]);
    assert!(sizes[48..].iter().all(|&n| n == 1));
    assert_eq!(p.completed_forward_calls, FORWARDS);
    assert_eq!(p.completed_token_steps, TOTAL);
    assert_eq!(p.completed_prefill_calls, 48);
    assert_eq!(p.completed_decode_calls, DECODES);
    assert_eq!(p.attempted_forward_calls, FORWARDS);
    assert_eq!(p.sampled_output_tokens, 0);
    Ok(())
}

fn tiny_model() -> Result<GptOssModel> {
    let (config, weights) = tiny_weights(3)?;
    GptOssModel::new(config, weights)
}

fn tiny_weights(window: usize) -> Result<(GptOssConfig, GptOssWeights)> {
    tiny_weights_with_vocab(window, 8)
}

fn tiny_weights_with_vocab(
    window: usize,
    vocab_size: usize,
) -> Result<(GptOssConfig, GptOssWeights)> {
    let mut config = GptOssConfig::from_json_str(
        r#"{"hidden_size":32,"intermediate_size":32,"num_hidden_layers":2,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"num_experts":2,"experts_per_token":2,"vocab_size":8,"sliding_window":3}"#,
    )?;
    config.sliding_window = window;
    config.vocab_size = vocab_size;
    let linear = |rows, cols| {
        DenseLinear::new(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| ((i % 11) as f32 - 5.) * 0.015625)
                .collect(),
            None,
        )
    };
    let packed =
        |rows| PackedMxfp4::from_parts(&[2, rows, 32], vec![0; 2 * rows * 16], vec![127; 2 * rows]);
    let mut layers = Vec::new();
    for _ in 0..2 {
        let experts =
            Mxfp4ExpertOperation::new(packed(64)?, vec![0.; 128], packed(32)?, vec![0.; 64], 7.)?;
        layers.push(GptOssLayerWeights::new(
            &config,
            vec![1.; 32],
            linear(64, 32)?,
            linear(32, 32)?,
            vec![0.; 4],
            vec![1.; 32],
            linear(2, 32)?,
            experts,
        )?);
    }
    let weights = GptOssWeights::new(
        &config,
        (0..vocab_size * 32)
            .map(|i| ((i % 7) as f32 - 3.) * 0.03125)
            .collect(),
        layers,
        vec![1.; 32],
        linear(vocab_size, 32)?,
    )?;
    Ok((config, weights))
}

// Frozen authored workload: no files, tokenizer, retained weights or sampling.
fn profile_synthetic_prefill(
    backend: &str,
    mut run: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<()> {
    let tokens: Vec<u32> = (0..128).map(|i| i % 8).collect();
    for length in [1, 8, 128] {
        let mut expected = None;
        let mut samples = Vec::new();
        for iteration in 0..9 {
            let started = std::time::Instant::now();
            let logits = run(&tokens[..length])?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
            let bits: Vec<_> = logits.iter().map(|value| value.to_bits()).collect();
            if let Some(expected) = &expected {
                assert_eq!(&bits, expected);
            } else {
                expected = Some(bits);
            }
            std::hint::black_box(logits);
            if iteration >= 2 {
                samples.push(elapsed_ms);
            }
        }
        println!(
            "synthetic-prefill {}",
            serde_json::json!({"backend": backend, "tokens": length,
                "vocab": 8192, "hidden": 32, "layers": 2,
                "warmups": 2, "samples_ms": samples})
        );
    }
    Ok(())
}

#[test]
#[ignore = "explicit synthetic release profiling; never loads retained model files"]
fn profile_cpu_prefill_head_work() -> Result<()> {
    let (config, weights) = tiny_weights_with_vocab(128, 8192)?;
    let mut model = GptOssModel::new(config, weights)?;
    let cancel = GptOssCancellationToken::new();
    profile_synthetic_prefill("cpu", |tokens| {
        model.reset_cache();
        model.prefill(tokens, &cancel)
    })
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit synthetic release profiling; never loads retained model files"]
fn profile_cuda_prefill_head_work() -> Result<()> {
    use candle_transformers::models::gpt_oss::{
        GptOssCudaConfig, GptOssCudaModel, GptOssResourceLimits,
    };
    let (config, weights) = tiny_weights_with_vocab(128, 8192)?;
    let limits = GptOssResourceLimits {
        max_sequence_tokens: 256,
        max_cache_bytes: 8 * 1024 * 1024,
        max_total_device_bytes: 8 * 1024 * 1024,
    };
    let cuda = GptOssCudaConfig::new(0, candle::DType::F32, 8 * 1024 * 1024, limits)
        .map_err(candle::Error::wrap)?;
    let mut model = GptOssCudaModel::new(config, weights, cuda).map_err(candle::Error::wrap)?;
    let cancel = GptOssCancellationToken::new();
    profile_synthetic_prefill("cuda", |tokens| {
        model.reset_cache().map_err(candle::Error::wrap)?;
        model
            .prefill(tokens, &cancel)
            .map_err(candle::Error::wrap)?
            .flatten_all()?
            .to_vec1::<f32>()
    })
}

#[test]
fn cpu_chunk128_and_single_tokens_agree_across_page_and_window_boundaries() -> Result<()> {
    let (config, weights) = tiny_weights(128)?;
    let mut chunked = GptOssModel::new(config.clone(), weights.clone())?;
    let mut singles = GptOssModel::new(config, weights)?;
    let tokens: Vec<u32> = (0..272).map(|i| i % 8).collect();
    let cancel = GptOssCancellationToken::new();
    for chunk in tokens[..256].chunks(128) {
        let actual = chunked.prefill(chunk, &cancel)?;
        let mut expected = Vec::new();
        for &token in chunk {
            expected = singles.decode(token, &cancel)?;
        }
        assert_eq!(actual.len(), chunked.config().vocab_size);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(chunked.resource_usage()?, singles.resource_usage()?);
    }
    for &token in &tokens[256..] {
        assert_eq!(
            chunked.decode(token, &cancel)?,
            singles.decode(token, &cancel)?
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_chunk128_preserves_causal_window_and_cancelled_snapshot() -> Result<()> {
    use candle_transformers::models::gpt_oss::{
        GptOssCudaConfig, GptOssCudaModel, GptOssResourceLimits,
    };
    let (config, weights) = tiny_weights(128)?;
    let limits = GptOssResourceLimits {
        max_sequence_tokens: 512,
        max_cache_bytes: 8 * 1024 * 1024,
        max_total_device_bytes: 8 * 1024 * 1024,
    };
    let cuda = GptOssCudaConfig::new(0, candle::DType::F32, 8 * 1024 * 1024, limits)
        .map_err(candle::Error::wrap)?;
    let mut chunked = GptOssCudaModel::new(config.clone(), weights.clone(), cuda.clone())
        .map_err(candle::Error::wrap)?;
    let mut singles =
        GptOssCudaModel::new(config.clone(), weights.clone(), cuda).map_err(candle::Error::wrap)?;
    let mut cpu = GptOssModel::new(config, weights)?;
    let tokens: Vec<u32> = (0..272).map(|i| i % 8).collect();
    let cancel = GptOssCancellationToken::new();
    for chunk in tokens[..256].chunks(128) {
        let output = chunked
            .prefill(chunk, &cancel)
            .map_err(candle::Error::wrap)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let mut expected = Vec::new();
        for &token in chunk {
            expected = singles
                .decode(token, &cancel)
                .map_err(candle::Error::wrap)?
                .flatten_all()?
                .to_vec1::<f32>()?;
        }
        assert_eq!(actual.len(), chunked.config().vocab_size);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let reference = cpu.prefill(chunk, &cancel)?;
        assert!(actual
            .iter()
            .zip(reference)
            .all(|(&a, b)| (a - b).abs() <= 1e-4));
        assert_eq!(
            chunked.resource_usage().map_err(candle::Error::wrap)?,
            singles.resource_usage().map_err(candle::Error::wrap)?
        );
    }
    let before = chunked.resource_usage().map_err(candle::Error::wrap)?;
    let cancelled = GptOssCancellationToken::cancel_after_checks(12);
    assert!(chunked.prefill(&tokens[..128], &cancelled).is_err());
    assert_eq!(
        chunked.resource_usage().map_err(candle::Error::wrap)?,
        before
    );
    for &token in &tokens[256..] {
        let a = chunked
            .decode(token, &cancel)
            .map_err(candle::Error::wrap)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let b = singles
            .decode(token, &cancel)
            .map_err(candle::Error::wrap)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(a, b);
    }
    Ok(())
}

#[test]
fn actual_tiny_cpu_cache_replays_after_partial_failure_and_cancellation() -> Result<()> {
    let tokens = [1, 2, 3, 4, 5, 6];
    let mut model = tiny_model()?;
    let mut p = Progress::default();
    let cancel = GptOssCancellationToken::new();
    let failed = walk(
        &tokens,
        4,
        2,
        &cancel,
        &mut p,
        |span, prefill| {
            if model.cache_len() == 4 {
                bail!("injected pre-dispatch failure");
            }
            if prefill {
                model.prefill(span, &cancel)?;
            } else {
                model.decode(span[0], &cancel)?;
            }
            Ok((model.resource_usage()?, timing()))
        },
        |_| Ok(()),
    );
    assert!(failed.is_err());
    assert_eq!(p.completed_token_steps, 4);
    assert_eq!(p.attempted_token_steps_upper_bound, 5);
    assert_eq!(p.completed_forward_calls, 2);
    assert_eq!(p.attempted_forward_calls, 3);
    assert_eq!(model.cache_len(), 4);
    let baseline = model.clone().decode(5, &GptOssCancellationToken::new())?;
    let cancelled = GptOssCancellationToken::cancel_after_checks(3);
    assert!(model.decode(5, &cancelled).is_err());
    assert_eq!(model.cache_len(), 4);
    assert_eq!(model.decode(5, &GptOssCancellationToken::new())?, baseline);
    model.reset_cache();
    let mut recovered = Progress::default();
    walk(
        &tokens,
        4,
        2,
        &GptOssCancellationToken::new(),
        &mut recovered,
        |span, prefill| {
            if prefill {
                model.prefill(span, &GptOssCancellationToken::new())?;
            } else {
                model.decode(span[0], &GptOssCancellationToken::new())?;
            }
            Ok((model.resource_usage()?, timing()))
        },
        |_| Ok(()),
    )?;
    assert_eq!(recovered.committed_sequence_tokens, 6);
    let mut empty = Progress::default();
    let cancelled = GptOssCancellationToken::new();
    cancelled.cancel();
    assert!(walk(
        &tokens,
        4,
        2,
        &cancelled,
        &mut empty,
        |_, _| bail!("must not forward"),
        |_| Ok(())
    )
    .is_err());
    assert_eq!(empty.attempted_forward_calls, 0);
    Ok(())
}

#[test]
fn malformed_bounds_counts_and_observation_fail_before_unaccounted_forward() -> Result<()> {
    let cancel = GptOssCancellationToken::new();
    for (prefill, chunk) in [(0, 1), (1, 0), (3, 2), (usize::MAX, 1)] {
        assert!(walk(
            &[1, 2, 3, 4],
            prefill,
            chunk,
            &cancel,
            &mut Progress::default(),
            |_, _| bail!("must not forward"),
            |_| Ok(())
        )
        .is_err());
    }
    let mut p = Progress {
        attempted_forward_calls: usize::MAX,
        ..Progress::default()
    };
    assert!(walk(
        &[1, 2],
        1,
        1,
        &cancel,
        &mut p,
        |_, _| bail!("must not forward"),
        |_| Ok(())
    )
    .is_err());
    let mut p = Progress::default();
    assert!(walk(
        &[1, 2],
        1,
        1,
        &cancel,
        &mut p,
        |_, _| bail!("must not forward"),
        |_| bail!("observer failure")
    )
    .is_err());
    assert_eq!(p.completed_forward_calls, 0);
    let mut p = Progress::default();
    assert!(walk(
        &[1, 2],
        1,
        1,
        &cancel,
        &mut p,
        |_, _| Ok((
            GptOssResourceUsage {
                sequence_tokens: 2,
                cache_bytes: 1,
                cache_capacity_bytes: 1
            },
            timing()
        )),
        |_| Ok(())
    )
    .is_err());
    assert_eq!(p.completed_forward_calls, 1);
    Ok(())
}

#[test]
fn host_verifier_accepts_cuda_metadata_and_rejects_tampered_completion() -> Result<()> {
    let fixture = OwnedFixture::new()?;
    let input_path = fixture.input()?;
    let (mut input, hash, _) = read_input(&input_path)?;
    input.backend = Backend::Cuda;
    let mut p = Progress {
        schema: "candle.gpt-oss.capacity-progress.v1".into(),
        status: "completed".into(),
        phase: "unloading".into(),
        source_commit: input.source_commit.clone(),
        input_sha256: hash.clone(),
        pid: 1,
        executable: "test-not-executed.exe".into(),
        consumed_files: vec![input.model.clone()],
        token_file: Some(input.token_file.clone()),
        loads_attempted: 1,
        requests_attempted: 1,
        requests_completed: 1,
        attempted_forward_calls: FORWARDS,
        completed_forward_calls: FORWARDS,
        attempted_token_steps_upper_bound: TOTAL,
        completed_token_steps: TOTAL,
        completed_prefill_calls: 48,
        completed_decode_calls: DECODES,
        sampled_output_tokens: 0,
        committed_sequence_tokens: TOTAL,
        cache_bytes: 1024,
        cache_capacity_bytes: 2048,
        peak_cache_capacity_bytes: 2048,
        static_weight_bytes: Some(15_053_650_176),
        load_ms: Some(10.),
        unload_ms: Some(10.),
        prefill_forward_ms: 48.,
        decode_forward_ms: 2048.,
        prefill_validation_ms: Some(4.),
        decode_validation_ms: Some(50.),
        registry_entries_after_release: Some(0),
        total_ms: 2200.,
        error: None,
    };
    let path = fixture.0.join("progress.json");
    let write = |p: &Progress| -> Result<()> {
        fs::write(&path, serde_json::to_vec(p).map_err(candle::Error::wrap)?)?;
        Ok(())
    };
    write(&p)?;
    super::runner::verify_progress(&path, &input, &hash)?;
    p.completed_token_steps -= 1;
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    p.completed_token_steps += 1;
    p.sampled_output_tokens = 1;
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    p.sampled_output_tokens = 0;
    p.registry_entries_after_release = Some(1);
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    p.registry_entries_after_release = Some(0);
    p.total_ms = 1.;
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    p.total_ms = 2200.;
    p.prefill_validation_ms = None;
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    p.prefill_validation_ms = Some(4.);
    p.consumed_files[0].path = fixture.0.join("different-model.gguf");
    write(&p)?;
    assert!(super::runner::verify_progress(&path, &input, &hash).is_err());
    #[cfg(windows)]
    {
        p.consumed_files[0].path = PathBuf::from(format!(r"\\?\{}", input.model.path.display()));
        write(&p)?;
        super::runner::verify_progress(&path, &input, &hash)?;
    }
    Ok(())
}
