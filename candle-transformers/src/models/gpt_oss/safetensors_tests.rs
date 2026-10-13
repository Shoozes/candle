use super::*;
use crate::models::gpt_oss::GptOssModel;
use crate::models::gpt_oss::{
    GptOssDevice, GptOssLoadOptions, GptOssResourceLimits, GptOssSession, GptOssSource,
};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    source: GptOssSafetensorsSource,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

struct Payload {
    shape: Vec<usize>,
    dtype: &'static str,
    bytes: Vec<u8>,
}

fn identity(path: &Path) -> Result<GptOssFileIdentity> {
    let bytes = std::fs::read(path)?;
    Ok(GptOssFileIdentity {
        path: path.to_path_buf(),
        size_bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

fn f32_payload(shape: &[usize], values: impl Iterator<Item = f32>) -> Payload {
    Payload {
        shape: shape.to_vec(),
        dtype: "F32",
        bytes: values.flat_map(f32::to_le_bytes).collect(),
    }
}

fn linear(shape: &[usize], seed: f32, start: usize) -> Payload {
    f32_payload(
        shape,
        (start..start + shape.iter().product::<usize>())
            .map(|index| ((index % 11) as f32 - 5.0) * seed),
    )
}

fn bias(rows: usize, seed: f32, start: usize) -> Payload {
    f32_payload(
        &[rows],
        (start..start + rows).map(|index| index as f32 * seed),
    )
}

fn serialize_shard(path: &Path, entries: &BTreeMap<String, Payload>) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, entry) in entries {
        let start = data.len();
        data.extend_from_slice(&entry.bytes);
        header.insert(
            name.clone(),
            serde_json::json!({
                "shape": entry.shape, "dtype": entry.dtype, "data_offsets": [start, data.len()]
            }),
        );
    }
    let header = serde_json::to_vec(&header).map_err(candle::Error::wrap)?;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    std::fs::write(path, bytes)?;
    Ok(())
}

fn fixture(layout: GptOssSafetensorsLayout, indexed: bool) -> Result<Fixture> {
    let root = std::env::temp_dir().join(format!(
        "candle-gpt-oss-native-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| candle::Error::Msg(e.to_string()))?
            .as_nanos(),
        NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root)?;
    let config_path = root.join("config.json");
    let tokenizer_path = root.join("tokenizer.json");
    let mut config = serde_json::json!({
        "num_hidden_layers":2,"num_experts":2,"experts_per_token":2,"vocab_size":8,
        "hidden_size":32,"intermediate_size":32,"head_dim":8,"num_attention_heads":4,
        "num_key_value_heads":2,"sliding_window":3,"initial_context_length":8,
        "rope_theta":10000.0,"rope_scaling_factor":1.0,"rope_ntk_alpha":1.0,"rope_ntk_beta":32.0
    });
    let original = layout == GptOssSafetensorsLayout::Original;
    if !original {
        let object = config.as_object_mut().expect("authored object");
        object.remove("num_experts");
        object.insert("num_local_experts".into(), 2.into());
        object.insert("num_experts_per_tok".into(), 2.into());
    }
    std::fs::write(
        &config_path,
        serde_json::to_vec(&config).map_err(candle::Error::wrap)?,
    )?;
    std::fs::write(&tokenizer_path, b"{}")?;
    let mut tensors = BTreeMap::new();
    tensors.insert(
        if original {
            "embedding.weight"
        } else {
            "model.embed_tokens.weight"
        }
        .into(),
        f32_payload(&[8, 32], (0..256).map(|i| (i % 17) as f32 * 0.001)),
    );
    tensors.insert(
        if original {
            "norm.scale"
        } else {
            "model.norm.weight"
        }
        .into(),
        f32_payload(&[32], std::iter::repeat_n(1.0, 32)),
    );
    tensors.insert(
        if original {
            "unembedding.weight"
        } else {
            "lm_head.weight"
        }
        .into(),
        linear(&[8, 32], 0.0001, 0),
    );
    for layer in 0..2 {
        let prefix = if original {
            format!("block.{layer}")
        } else {
            format!("model.layers.{layer}")
        };
        let name = |a: &str, b: &str| format!("{prefix}.{}", if original { a } else { b });
        tensors.insert(
            name("attn.norm.scale", "input_layernorm.weight"),
            f32_payload(&[32], std::iter::repeat_n(1.0, 32)),
        );
        tensors.insert(
            name("mlp.norm.scale", "post_attention_layernorm.weight"),
            f32_payload(&[32], std::iter::repeat_n(1.0, 32)),
        );
        let seed = 0.0003 + layer as f32 * 0.00007;
        if original {
            tensors.insert(name("attn.qkv.weight", ""), linear(&[64, 32], seed, 0));
            tensors.insert(name("attn.qkv.bias", ""), bias(64, seed, 0));
        } else {
            for (projection, rows, offset) in [("q", 32, 0), ("k", 16, 32), ("v", 16, 48)] {
                tensors.insert(
                    name("", &format!("self_attn.{projection}_proj.weight")),
                    linear(&[rows, 32], seed, offset * 32),
                );
                tensors.insert(
                    name("", &format!("self_attn.{projection}_proj.bias")),
                    bias(rows, seed, offset),
                );
            }
        }
        let seed = 0.0002 + layer as f32 * 0.00005;
        tensors.insert(
            name("attn.out.weight", "self_attn.o_proj.weight"),
            linear(&[32, 32], seed, 0),
        );
        tensors.insert(
            name("attn.out.bias", "self_attn.o_proj.bias"),
            bias(32, seed, 0),
        );
        tensors.insert(
            name("attn.sinks", "self_attn.sinks"),
            f32_payload(
                &[4],
                (0..4).map(|h| (h as f32 - 1.5) * 0.125 + layer as f32 * 0.25),
            ),
        );
        let seed = 0.0004 + layer as f32 * 0.0003;
        tensors.insert(
            name("mlp.gate.weight", "mlp.router.weight"),
            linear(&[2, 32], seed, 0),
        );
        tensors.insert(name("mlp.gate.bias", "mlp.router.bias"), bias(2, seed, 0));
        for (projection, rows, seed) in [(1, 64, 1 + layer * 17), (2, 32, 7 + layer * 19)] {
            let (blocks, scales, biases) = if projection == 1 {
                (
                    name("mlp.mlp1_weight.blocks", "mlp.experts.gate_up_proj_blocks"),
                    name("mlp.mlp1_weight.scales", "mlp.experts.gate_up_proj_scales"),
                    name("mlp.mlp1_bias", "mlp.experts.gate_up_proj_bias"),
                )
            } else {
                (
                    name("mlp.mlp2_weight.blocks", "mlp.experts.down_proj_blocks"),
                    name("mlp.mlp2_weight.scales", "mlp.experts.down_proj_scales"),
                    name("mlp.mlp2_bias", "mlp.experts.down_proj_bias"),
                )
            };
            tensors.insert(
                blocks,
                Payload {
                    shape: vec![2, rows, 1, 16],
                    dtype: "U8",
                    bytes: (0..2 * rows * 16)
                        .map(|i| (seed as u8).wrapping_add(i as u8).rotate_left(1))
                        .collect(),
                },
            );
            tensors.insert(
                scales,
                Payload {
                    shape: vec![2, rows, 1],
                    dtype: "U8",
                    bytes: vec![123; 2 * rows],
                },
            );
            tensors.insert(
                biases,
                f32_payload(&[2, rows], std::iter::repeat_n(0.0, 2 * rows)),
            );
        }
    }
    let mut weights = Vec::new();
    let mut weight_map = BTreeMap::new();
    if indexed {
        let mut shards = [BTreeMap::new(), BTreeMap::new()];
        for (i, (name, entry)) in tensors.into_iter().enumerate() {
            let filename = format!("weights-{}-of-17.safetensors", i % 2);
            weight_map.insert(name.clone(), filename);
            shards[i % 2].insert(name, entry);
        }
        for (i, shard) in shards.iter().enumerate() {
            let path = root.join(format!("weights-{i}-of-17.safetensors"));
            serialize_shard(&path, shard)?;
            weights.push(identity(&path)?);
        }
    } else {
        let path = root.join("model.safetensors");
        serialize_shard(&path, &tensors)?;
        weights.push(identity(&path)?);
    }
    let index = if indexed {
        let path = root.join("model.safetensors.index.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"weight_map":weight_map}))
                .map_err(candle::Error::wrap)?,
        )?;
        Some(identity(&path)?)
    } else {
        None
    };
    Ok(Fixture {
        root,
        source: GptOssSafetensorsSource {
            layout,
            config: identity(&config_path)?,
            tokenizer: identity(&tokenizer_path)?,
            index,
            weights,
        },
    })
}

#[test]
fn both_layouts_and_indexed_shards_match_independent_two_layer_oracle() -> Result<()> {
    let bytes = include_bytes!("../../../../tests/fixtures/gpt_oss_task3_two_layer/oracle.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "0530081353cc7268d796ae070c4168ec4e6bb036f62d54670b975a43556bc173"
    );
    let oracle: serde_json::Value = serde_json::from_slice(bytes).map_err(candle::Error::wrap)?;
    let prompt: Vec<u32> =
        serde_json::from_value(oracle["forward"]["prompt"].clone()).map_err(candle::Error::wrap)?;
    let expected: Vec<f32> = serde_json::from_value(oracle["forward"]["uncached_logits"].clone())
        .map_err(candle::Error::wrap)?;
    for layout in [
        GptOssSafetensorsLayout::Original,
        GptOssSafetensorsLayout::Transformers,
    ] {
        for indexed in [false, true] {
            let fixture = fixture(layout, indexed)?;
            let registry = GptOssLoadRegistry::default();
            let (artifact, weights, handle) = load_gpt_oss_safetensors(
                &fixture.source,
                &registry,
                &GptOssCancellationToken::new(),
                1024 * 1024,
            )?;
            assert_eq!(artifact.inventory().len(), if indexed { 5 } else { 3 });
            let mut model = GptOssModel::new(artifact.config().clone(), weights)?;
            let logits = model.forward(&prompt)?;
            assert_eq!(logits.len(), expected.len());
            for (actual, expected) in logits.iter().zip(&expected) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
            model.reset_cache();
            assert_eq!(model.forward(&prompt)?, logits);
            drop(model);
            drop(handle);
            let _lease = registry.begin(
                std::fs::canonicalize(&fixture.source.config.path)?
                    .to_string_lossy()
                    .into_owned(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn wrong_identity_budget_index_and_cancelled_load_release_ownership() -> Result<()> {
    let mut fixture = fixture(GptOssSafetensorsLayout::Transformers, true)?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let original = fixture.source.config.sha256.clone();
    fixture.source.config.sha256 = "00".repeat(32);
    assert!(load_gpt_oss_safetensors(&fixture.source, &registry, &cancel, 1024 * 1024).is_err());
    fixture.source.config.sha256 = original;
    assert!(load_gpt_oss_safetensors(&fixture.source, &registry, &cancel, 1).is_err());
    let index = fixture.source.index.take();
    assert!(load_gpt_oss_safetensors(&fixture.source, &registry, &cancel, 1024 * 1024).is_err());
    fixture.source.index = index;
    cancel.cancel();
    assert!(load_gpt_oss_safetensors(&fixture.source, &registry, &cancel, 1024 * 1024).is_err());
    let (_, weights, handle) = load_gpt_oss_safetensors(
        &fixture.source,
        &registry,
        &GptOssCancellationToken::new(),
        1024 * 1024,
    )?;
    assert!(load_gpt_oss_safetensors(
        &fixture.source,
        &registry,
        &GptOssCancellationToken::new(),
        1024 * 1024
    )
    .is_err());
    drop(weights);
    drop(handle);
    assert!(load_gpt_oss_safetensors(
        &fixture.source,
        &registry,
        &GptOssCancellationToken::new(),
        1024 * 1024
    )
    .is_ok());
    Ok(())
}

#[test]
fn rejects_malformed_header_shapes_and_payload_ranges() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Original, false)?;
    for header in [
        serde_json::json!({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}),
        serde_json::json!({"x":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}),
        serde_json::json!({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}),
        serde_json::json!({"x":{"dtype":"U8","shape":[usize::MAX,2],"data_offsets":[0,2]}}),
        serde_json::json!({"x":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}}),
    ] {
        let bytes = serde_json::to_vec(&header).map_err(candle::Error::wrap)?;
        let path = fixture.root.join("malformed.safetensors");
        let mut file = (bytes.len() as u64).to_le_bytes().to_vec();
        file.extend(bytes);
        file.extend([0u8; 2]);
        std::fs::write(&path, file)?;
        assert!(Shard::open(&identity(&path)?, &GptOssCancellationToken::new()).is_err());
    }
    Ok(())
}

#[test]
fn header_parser_rejects_duplicates_and_aggregate_exhaustion_before_payload() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Original, false)?;
    let identity = &fixture.source.weights[0];
    let cancel = GptOssCancellationToken::new();
    assert!(Shard::open_bounded(identity, &cancel, 1, MAX_TOTAL_TENSORS).is_err());
    assert!(Shard::open_bounded(identity, &cancel, MAX_TOTAL_HEADER_BYTES, 1).is_err());
    let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
    let path = fixture.root.join("duplicate.safetensors");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0u8; 2]);
    std::fs::write(&path, bytes)?;
    assert!(Shard::open(&super::tests::identity(&path)?, &cancel).is_err());
    Ok(())
}

fn options(fixture: &Fixture, device: GptOssDevice) -> Result<GptOssLoadOptions> {
    let config = GptOssConfig::from_json_file(&fixture.source.config.path)?;
    let mut limits = GptOssResourceLimits::for_config(&config)?;
    limits.max_total_device_bytes = 16 * 1024 * 1024;
    Ok(GptOssLoadOptions {
        source: GptOssSource::PackedSafetensors(fixture.source.clone()),
        device,
        limits,
        max_weight_bytes: 1024 * 1024,
    })
}

#[test]
fn common_session_preserves_prefix_and_releases_after_failures() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Transformers, true)?;
    let options = options(&fixture, GptOssDevice::Cpu)?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let mut session = GptOssSession::load(&options, &registry, &cancel)?;
    assert_eq!(session.device(), GptOssDevice::Cpu);
    assert_eq!(session.activation_dtype(), DType::F32);
    assert_eq!(session.consumed_files().len(), 5);
    session.prefill(&[1, 2], &cancel)?;
    let prefix = session.resource_usage()?;
    assert!(session
        .prefill_profiled(&[3, 4, 5], &GptOssCancellationToken::cancel_after_checks(4))
        .is_err());
    assert_eq!(
        session.resource_usage()?.sequence_tokens,
        prefix.sequence_tokens
    );
    assert!(session.decode_profiled(99, &cancel).is_err());
    assert_eq!(
        session.resource_usage()?.sequence_tokens,
        prefix.sequence_tokens
    );
    let (recovered, timing) = session.decode_profiled(3, &cancel)?;
    assert!(timing.forward_ms.is_finite() && timing.forward_ms >= 0.);
    assert!(timing.cuda_final_validation_ms.is_none());
    let recovered = recovered.into_cpu()?;
    session.reset_cache()?;
    assert_eq!(session.resource_usage()?.cache_bytes, 0);
    assert_eq!(session.prefill(&[1, 2, 3], &cancel)?.into_cpu()?, recovered);
    drop(session);
    let mut invalid = options.clone();
    invalid.limits.max_sequence_tokens = 9;
    assert!(GptOssSession::load(&invalid, &registry, &cancel).is_err());
    assert!(GptOssSession::load(&options, &registry, &cancel).is_ok());
    Ok(())
}

#[test]
fn session_combined_static_cache_workspace_limit_rejects_before_mutation() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Original, false)?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let (artifact, handle) =
        GptOssSafetensorsArtifact::open_with_registry(&fixture.source, &registry, &cancel)?;
    let static_bytes = artifact.weight_resident_bytes()?;
    let one_token =
        crate::models::gpt_oss::total_device_bytes_for_tokens(artifact.config(), static_bytes, 1)?;
    drop(artifact);
    drop(handle);
    let mut options = options(&fixture, GptOssDevice::Cpu)?;
    options.limits.max_total_device_bytes = one_token;
    let mut session = GptOssSession::load(&options, &registry, &cancel)?;
    assert_eq!(session.static_weight_bytes(), static_bytes);
    session.prefill(&[1], &cancel)?;
    assert!(session.decode(2, &cancel).is_err());
    assert_eq!(session.resource_usage()?.sequence_tokens, 1);
    drop(session);
    options.limits.max_total_device_bytes -= 1;
    assert!(GptOssSession::load(&options, &registry, &cancel).is_err());
    Ok(())
}

#[test]
#[cfg(not(feature = "cuda"))]
fn cuda_selection_rejects_before_opening_any_artifact() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Original, false)?;
    let mut options = options(&fixture, GptOssDevice::Cuda { device_index: 0 })?;
    let mut source = fixture.source.clone();
    source.config.path = fixture.root.join("absent-config.json");
    options.source = GptOssSource::PackedSafetensors(source);
    let error = match GptOssSession::load(
        &options,
        &GptOssLoadRegistry::default(),
        &GptOssCancellationToken::new(),
    ) {
        Ok(_) => panic!("CPU build must reject CUDA"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("cuda feature"));
    Ok(())
}

#[test]
fn float16_and_bfloat16_dense_tensors_keep_native_forward_parity() -> Result<()> {
    for (dtype, one) in [("F16", [0u8, 0x3c]), ("BF16", [0x80u8, 0x3f])] {
        let mut fixture = fixture(GptOssSafetensorsLayout::Original, false)?;
        let path = fixture.source.weights[0].path.clone();
        let bytes = std::fs::read(&path)?;
        let len = u64::from_le_bytes(bytes[..8].try_into().expect("authored header")) as usize;
        let header: BTreeMap<String, TensorInfo> =
            serde_json::from_slice(&bytes[8..8 + len]).map_err(candle::Error::wrap)?;
        let mut payloads = BTreeMap::new();
        for (name, info) in header {
            let start = 8 + len + info.data_offsets[0] as usize;
            let end = 8 + len + info.data_offsets[1] as usize;
            if name == "norm.scale" {
                payloads.insert(
                    name,
                    Payload {
                        shape: info.shape,
                        dtype,
                        bytes: std::iter::repeat_n(one, 32).flatten().collect(),
                    },
                );
            } else {
                let dtype = if info.dtype == "U8" { "U8" } else { "F32" };
                payloads.insert(
                    name,
                    Payload {
                        shape: info.shape,
                        dtype,
                        bytes: bytes[start..end].to_vec(),
                    },
                );
            }
        }
        serialize_shard(&path, &payloads)?;
        fixture.source.weights[0] = identity(&path)?;
        let (_, weights, _) = load_gpt_oss_safetensors(
            &fixture.source,
            &GptOssLoadRegistry::default(),
            &GptOssCancellationToken::new(),
            1024 * 1024,
        )?;
        let config = GptOssConfig::from_json_file(&fixture.source.config.path)?;
        let mut model = GptOssModel::new(config, weights)?;
        let oracle: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../../tests/fixtures/gpt_oss_task3_two_layer/oracle.json"
        ))
        .map_err(candle::Error::wrap)?;
        let prompt: Vec<u32> = serde_json::from_value(oracle["forward"]["prompt"].clone())
            .map_err(candle::Error::wrap)?;
        let expected: Vec<f32> =
            serde_json::from_value(oracle["forward"]["uncached_logits"].clone())
                .map_err(candle::Error::wrap)?;
        for (actual, expected) in model.forward(&prompt)?.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1e-4);
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn common_cuda_session_matches_cpu_and_releases() -> Result<()> {
    let fixture = fixture(GptOssSafetensorsLayout::Transformers, true)?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let mut cpu = GptOssSession::load(&options(&fixture, GptOssDevice::Cpu)?, &registry, &cancel)?;
    let expected = cpu.prefill(&[1, 2, 3, 4], &cancel)?.into_cpu()?;
    drop(cpu);
    let mut cuda = GptOssSession::load(
        &options(&fixture, GptOssDevice::Cuda { device_index: 0 })?,
        &registry,
        &cancel,
    )?;
    let (actual, timing) = cuda.prefill_profiled(&[1, 2, 3, 4], &cancel)?;
    let validation = timing.cuda_final_validation_ms.ok_or_else(|| {
        candle::Error::Msg("profiled CUDA forward omitted validation timing".into())
    })?;
    assert!(validation.is_finite() && validation >= 0. && validation <= timing.forward_ms);
    let actual = actual.into_cpu()?;
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
    let prefix = cuda.resource_usage()?.sequence_tokens;
    assert!(cuda.decode_profiled(99, &cancel).is_err());
    assert_eq!(cuda.resource_usage()?.sequence_tokens, prefix);
    cuda.reset_cache()?;
    assert_eq!(cuda.resource_usage()?.cache_capacity_bytes, 0);
    drop(cuda);
    assert!(
        GptOssSession::load(&options(&fixture, GptOssDevice::Cpu)?, &registry, &cancel).is_ok()
    );
    Ok(())
}
