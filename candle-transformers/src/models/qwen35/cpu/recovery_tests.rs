use super::*;
use candle::quantized::GgmlDType;

pub(super) fn matrix(rows: usize, columns: usize, seed: usize, device: &Device) -> Result<QMatMul> {
    QMatMul::from_qtensor(weight(rows, columns, seed, device)?)
}

fn weight(rows: usize, columns: usize, seed: usize, device: &Device) -> Result<QTensor> {
    let values = (0..rows * columns)
        .map(|i| ((i * 17 + seed * 13) % 101) as f32 / 505.0 - 0.1)
        .collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (rows, columns), &Device::Cpu)?;
    QTensor::quantize_onto(&tensor, GgmlDType::Q8_0, device)
}

pub(super) fn tiny(device: &Device) -> Result<CpuModel> {
    let config = Config {
        vocab_size: 32,
        context_length: 128,
        block_count: 2,
        hidden_size: 32,
        intermediate_size: 64,
        num_attention_heads: 2,
        num_kv_heads: 1,
        attention_head_dim: 16,
        rope_dimension_count: 16,
        rope_sections: [4, 4, 0, 0],
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        conv_kernel: 4,
        state_size: 16,
        key_groups: 2,
        value_heads: 4,
        inner_size: 64,
        full_attention_interval: 2,
    };
    let embedding = Arc::new(weight(32, 32, 1, device)?);
    let output = QMatMul::from_arc(embedding.clone())?;
    let mut layers = Vec::new();
    for index in 0..2 {
        let mixer = if index == 0 {
            Mixer::Recurrent(Recurrent {
                qkv: matrix(128, 32, 2, device)?,
                z: matrix(64, 32, 3, device)?,
                alpha: matrix(4, 32, 4, device)?,
                beta: matrix(4, 32, 5, device)?,
                out: matrix(32, 64, 6, device)?,
                conv: vec![0.2; 128 * 4],
                a: vec![-0.1; 4],
                dt: vec![0.02; 4],
                norm: vec![1.0; 16],
            })
        } else {
            Mixer::Attention(Attention {
                q: matrix(64, 32, 7, device)?,
                k: matrix(16, 32, 8, device)?,
                v: matrix(16, 32, 9, device)?,
                out: matrix(32, 32, 10, device)?,
                q_norm: vec![1.0; 16],
                k_norm: vec![1.0; 16],
            })
        };
        layers.push(Layer {
            attn_norm: vec![1.0; 32],
            post_norm: vec![1.0; 32],
            gate: matrix(64, 32, 11 + index, device)?,
            up: matrix(64, 32, 13 + index, device)?,
            down: matrix(32, 64, 15 + index, device)?,
            mixer,
        });
    }
    Ok(CpuModel {
        projection_mode: HybridProjectionMode::Sequential,
        config,
        max_positions: 4,
        position: 0,
        embedding,
        output_norm: vec![1.0; 32],
        output,
        layers,
        cache: vec![
            Cache::Recurrent {
                fifo: vec![0.0; 384],
                state: vec![0.0; 1024],
            },
            Cache::Attention {
                keys: Vec::new(),
                values: Vec::new(),
            },
        ],
    })
}

fn snapshot(model: &CpuModel) -> Vec<u32> {
    let mut bits = vec![model.position as u32];
    for cache in &model.cache {
        match cache {
            Cache::Recurrent { fifo, state } => {
                bits.extend(fifo.iter().chain(state).map(|v| v.to_bits()));
            }
            Cache::Attention { keys, values } => {
                bits.push(keys.len() as u32);
                bits.push(values.len() as u32);
                bits.extend(keys.iter().chain(values).flatten().map(|v| v.to_bits()));
            }
        }
    }
    bits
}

#[test]
fn cpu_recovery_and_admission_preserve_contract() -> Result<()> {
    check(&Device::Cpu, HybridProjectionMode::Sequential)
}

#[test]
fn cpu_failed_projection_keeps_completed_observations() -> Result<()> {
    check_failed_projection(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_failed_projection_keeps_completed_observations() -> Result<()> {
    let device = Device::new_cuda(0)?;
    check_failed_projection(&device)?;
    device.synchronize()
}

fn check_failed_projection(device: &Device) -> Result<()> {
    for mode in [
        HybridProjectionMode::Sequential,
        HybridProjectionMode::Grouped,
    ] {
        let mut model = tiny(device)?;
        model.projection_mode = mode;
        let empty = snapshot(&model);
        let good = std::mem::replace(&mut model.layers[1].up, matrix(64, 64, 24, device)?);
        let mut events = Vec::new();
        assert!(model
            .decode_observed(
                1,
                Some(&mut |layer, name, _| {
                    events.push((layer, name.to_owned()));
                })
            )
            .is_err());
        assert_eq!(
            events.last(),
            Some(&(1, "ffn_gate".to_owned())),
            "mode {mode:?}"
        );
        assert_eq!(snapshot(&model), empty);
        model.layers[1].up = good;
        assert_eq!(
            model.prefill(&[1, 2, 3])?,
            tiny(device)?.prefill(&[1, 2, 3])?
        );

        model.reset();
        model.prefill(&[1])?;
        let Mixer::Recurrent(recurrent) = &mut model.layers[0].mixer else {
            panic!("authored mixer")
        };
        let good = std::mem::replace(&mut recurrent.z, matrix(64, 64, 25, device)?);
        let mut events = Vec::new();
        assert!(model
            .decode_observed(
                2,
                Some(&mut |layer, name, _| {
                    events.push((layer, name.to_owned()));
                })
            )
            .is_err());
        assert_eq!(
            events.last(),
            Some(&(0, "linear_attn_qkv_mixed".to_owned())),
            "mode {mode:?}"
        );
        assert_eq!(snapshot(&model), empty);
        let Mixer::Recurrent(recurrent) = &mut model.layers[0].mixer else {
            panic!("authored mixer")
        };
        recurrent.z = good;
        assert_eq!(
            model.prefill(&[1, 2, 3])?,
            tiny(device)?.prefill(&[1, 2, 3])?
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn hybrid_recovery_and_admission_preserve_contract() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for mode in [
        HybridProjectionMode::Sequential,
        HybridProjectionMode::Grouped,
    ] {
        check(&device, mode)?;
    }
    let mut hybrid = HybridGpuModel {
        inner: tiny(&device)?,
    };
    let expected = hybrid.prefill(&[1, 2, 3])?;
    hybrid.reset();
    let mut calls = 0;
    assert!(hybrid
        .prefill_with_cancel(&[1, 2, 3], &mut || {
            calls += 1;
            calls == 2
        })
        .is_err());
    assert_eq!(hybrid.position(), 0);
    assert_eq!(hybrid.prefill(&[1, 2, 3])?, expected);
    device.synchronize()
}

fn check(device: &Device, mode: HybridProjectionMode) -> Result<()> {
    let mut model = tiny(device)?;
    model.projection_mode = mode;
    let empty = snapshot(&model);
    for tokens in [&[][..], &[1, 32][..], &[1, 2, 3, 4, 5][..]] {
        assert!(model
            .prefill_with_cancel(tokens, &mut || panic!("admission precedes cancellation"))
            .is_err());
        assert_eq!(snapshot(&model), empty);
    }
    let expected = model.prefill(&[1, 2, 3])?;
    let admitted = snapshot(&model);
    assert!(model.decode(32).is_err());
    assert!(model.prefill(&[1]).is_err());
    assert_eq!(snapshot(&model), admitted);
    model.decode(4)?;
    let full = snapshot(&model);
    assert!(model.decode(5).is_err());
    assert_eq!(snapshot(&model), full);
    model.reset();
    let mut calls = 0;
    assert!(model
        .prefill_with_cancel(&[1, 2, 3], &mut || {
            calls += 1;
            calls == 2
        })
        .is_err());
    assert_eq!(calls, 2);
    assert_eq!(snapshot(&model), empty);
    assert_eq!(model.prefill(&[1, 2, 3])?, expected);

    // A bad final-layer FFN shape fails after both mixer caches have mutated.
    let bad = matrix(64, 64, 21, device)?;
    let good = std::mem::replace(&mut model.layers[1].gate, bad);
    let mut reached_mutation = false;
    assert!(model
        .decode_observed(
            4,
            Some(&mut |layer, name, _| {
                reached_mutation |= layer == 1 && name == "attn_post_norm";
            })
        )
        .is_err());
    assert!(reached_mutation);
    assert_eq!(snapshot(&model), empty);
    model.layers[1].gate = good;
    assert_eq!(model.prefill(&[1, 2, 3])?, expected);
    model.reset();

    // The output projection fails only after every layer's token state changes.
    let good = std::mem::replace(&mut model.output, matrix(32, 64, 22, device)?);
    assert!(model.prefill(&[1, 2, 3]).is_err());
    assert_eq!(snapshot(&model), empty);
    model.output = good;
    for _ in 0..3 {
        assert_eq!(model.prefill(&[1, 2, 3])?, expected);
        model.reset();
    }
    Ok(())
}
