use candle_core::{
    quantized::{
        k_quants::{self, GgmlType},
        with_cpu_quantized_matmul_mode as with_mode, with_native_q8_0,
        CpuQuantizedMatmulMode as Mode, GgmlDType, NativeQ8Execution, QMatMul, QTensor,
    },
    DType, Device, Module, Result, Tensor,
};

fn values(count: usize, modulus: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 17 % modulus) as f32 - (modulus / 2) as f32) / 16.)
        .collect()
}

#[test]
fn tiled_and_rowwise_are_exact_across_types_tiles_tails_and_offsets() -> Result<()> {
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        for n in [1, 2, 3, 23] {
            let k = 256;
            let weights = Tensor::from_vec(values(n * k, 101), (n, k), &Device::Cpu)?;
            let q = QTensor::quantize(&weights, dtype)?;
            for m in [1, 15, 16, 17, 33] {
                let input = Tensor::from_vec(values((m + 1) * k, 97), (m + 1, k), &Device::Cpu)?
                    .narrow(0, 1, m)?
                    .unsqueeze(0)?;
                for activation in [DType::F32, DType::BF16] {
                    let input = input.to_dtype(activation)?;
                    let tiled = with_mode(Mode::GenericTiled, || input.apply_op1_no_bwd(&q))?;
                    let legacy = with_mode(Mode::GenericRowwise, || input.apply_op1_no_bwd(&q))?;
                    assert_eq!(tiled.dtype(), activation);
                    assert_eq!(tiled.dims(), [1, m, n]);
                    let tiled = tiled
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        tiled,
                        legacy
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        "{dtype:?}/{activation:?}/{m}/{n}"
                    );
                    for row in 0..m {
                        let single = with_mode(Mode::GenericTiled, || {
                            input.narrow(1, row, 1)?.apply_op1_no_bwd(&q)
                        })?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        assert_eq!(&tiled[row * n..(row + 1) * n], single);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn native_q8_and_dense_contracts_survive_explicit_selection() -> Result<()> {
    let weight = Tensor::from_vec(values(32 * 32, 101), (32, 32), &Device::Cpu)?;
    let input = Tensor::from_vec(values(2 * 32, 97), (2, 32), &Device::Cpu)?;
    let q = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
        &weight,
        GgmlDType::Q8_0,
    )?));
    for mode in [Mode::GenericTiled, Mode::GenericRowwise] {
        with_mode(mode, || {
            let (result, report) = with_native_q8_0(|| q.forward(&input));
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("conflicts with native Q8"));
            assert_eq!(report, NativeQ8Execution::default());
            assert!(q
                .forward(&input.to_dtype(DType::F16)?)
                .unwrap_err()
                .to_string()
                .contains("F32 or BF16"));
            // The selector does not turn a dense linear into a quantized operation.
            assert_eq!(
                QMatMul::Tensor(weight.clone()).forward(&input)?.dims(),
                [2, 32]
            );
            Ok(())
        })?;
    }
    let (actual, report) = with_mode(Mode::Auto, || Ok(with_native_q8_0(|| q.forward(&input))))?;
    assert_eq!(actual?.dims(), [2, 32]);
    assert_eq!(report.cpu_matmuls, 1);
    assert_eq!(q.forward(&input.to_dtype(DType::F16)?)?.dtype(), DType::F16);
    Ok(())
}

#[test]
fn generic_modes_bypass_repacked_shapes_and_preserve_ggml_scope() -> Result<()> {
    let (m, k, n) = (17, 512, 64);
    let weights = values(n * k, 101);
    let mut blocks = vec![k_quants::BlockQ8_0::zeros(); n * k / 32];
    k_quants::BlockQ8_0::from_float(&weights, &mut blocks);
    let q = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
        &Tensor::from_vec(weights, (n, k), &Device::Cpu)?,
        GgmlDType::Q8_0,
    )?));
    let lhs = values(m * k, 97);
    let input = Tensor::from_vec(lhs.clone(), (m, k), &Device::Cpu)?;
    for mode in [Mode::GenericTiled, Mode::GenericRowwise] {
        with_mode(mode, || {
            let mut direct = vec![0.; m * n];
            k_quants::matmul((m, k, n), &lhs, &blocks, &mut direct)?;
            let ordinary = q.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(ordinary, direct);
            let ggml = q
                .forward_ggml_q8_0(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                q.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
                ordinary
            );
            let legacy_ggml = with_mode(Mode::GenericRowwise, || q.forward_ggml_q8_0(&input))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(ggml, legacy_ggml);
            Ok(())
        })?;
    }
    // Existing optimized repack tests separately cover Auto's architecture-dependent kernels.
    assert_eq!(q.forward(&input)?.dims(), [m, n]);
    Ok(())
}

#[test]
fn malformed_buffers_overflow_and_zero_dimensions_are_controlled() -> Result<()> {
    let blocks = vec![k_quants::BlockQ4_0::zeros(); 4];
    for (dims, lhs, rhs, out) in [
        ((2, 32, 4), vec![1.; 63], blocks.clone(), 8),
        ((2, 32, 4), vec![1.; 64], blocks[..3].to_vec(), 8),
        ((2, 32, 4), vec![1.; 64], blocks.clone(), 7),
        ((1, 31, 4), vec![1.; 31], blocks.clone(), 4),
        ((usize::MAX, 32, 1), vec![], vec![], 0),
        ((usize::MAX, 0, 2), vec![], vec![], 0),
        ((0, 32, usize::MAX), vec![], vec![], 0),
    ] {
        let mut dst = vec![42.; out];
        assert!(k_quants::matmul(dims, &lhs, &rhs, &mut dst).is_err());
        assert_eq!(dst, vec![42.; out]);
        let lhs: Vec<_> = lhs.into_iter().map(half::f16::from_f32).collect();
        let mut dst = vec![half::f16::ONE; out];
        assert!(k_quants::matmul_f16(dims, &lhs, &rhs, &mut dst).is_err());
        assert_eq!(dst, vec![half::f16::ONE; out]);
    }
    k_quants::matmul::<k_quants::BlockQ4_0>((0, 32, 4), &[], &blocks, &mut [])?;
    k_quants::matmul::<k_quants::BlockQ4_0>((2, 32, 0), &[1.; 64], &[], &mut [])?;
    let mut dst = [42.; 7];
    k_quants::matmul::<k_quants::BlockQ4_0>((2, 0, 3), &[], &[], &mut dst)?;
    assert_eq!(dst, [0., 0., 0., 0., 0., 0., 42.]);
    for (m, k, n) in [(0, 32, 4), (2, 32, 0), (2, 0, 3)] {
        let q = QTensor::quantize(
            &Tensor::zeros((n, k), DType::F32, &Device::Cpu)?,
            GgmlDType::Q4_0,
        )?;
        let x = Tensor::zeros((m, k), DType::F32, &Device::Cpu)?;
        assert_eq!(x.apply_op1_no_bwd(&q)?.dims(), [m, n]);
    }
    Ok(())
}

/// Frozen deterministic component timing: two dtypes, fixed shapes, three warmups,
/// five alternating-order samples per mode, ten calls per sample; no model weights.
#[test]
#[ignore = "Explicit CPU component timing, no production model forwards"]
fn benchmark_cpu_generic_modes() -> Result<()> {
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        let (n, k) = (2040, 2048);
        let q = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
            &Tensor::from_vec(values(n * k, 101), (n, k), &Device::Cpu)?,
            dtype,
        )?));
        for m in [1, 32, 128] {
            let input = Tensor::from_vec(values(m * k, 97), (m, k), &Device::Cpu)?;
            let modes = [Mode::GenericTiled, Mode::GenericRowwise];
            for mode in modes {
                with_mode(mode, || {
                    for _ in 0..3 {
                        std::hint::black_box(q.forward(&input)?);
                    }
                    Ok(())
                })?;
            }
            let mut samples = [Vec::new(), Vec::new()];
            for sample in 0..5 {
                for index in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let started = std::time::Instant::now();
                    with_mode(modes[index], || {
                        for _ in 0..10 {
                            std::hint::black_box(q.forward(&input)?);
                        }
                        Ok(())
                    })?;
                    samples[index].push(started.elapsed().as_secs_f64() * 1000. / 10.);
                }
            }
            for (index, mode) in modes.into_iter().enumerate() {
                samples[index].sort_by(f64::total_cmp);
                println!("{dtype:?} m={m} n={n} k={k} {mode:?} median_ms={:.6} min_ms={:.6} max_ms={:.6}", samples[index][2], samples[index][0], samples[index][4]);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cpu_selection_leaves_cuda_and_native_cuda_unchanged() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let weights = Tensor::from_vec(values(5 * 64, 101), (5, 64), &Device::Cpu)?;
    let q = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize_onto(
        &weights,
        GgmlDType::Q8_0,
        &device,
    )?));
    let input = Tensor::from_vec(values(4 * 64, 97), (4, 64), &device)?;
    let expected = q.forward(&input)?.to_vec2::<f32>()?;
    let (native, baseline_report) = with_native_q8_0(|| q.forward(&input));
    let native = native?.to_vec2::<f32>()?;
    assert_eq!(baseline_report.cuda_f32_q8, 1);
    for mode in [Mode::GenericTiled, Mode::GenericRowwise] {
        with_mode(mode, || {
            assert_eq!(q.forward(&input)?.to_vec2::<f32>()?, expected);
            let (actual, report) = with_native_q8_0(|| q.forward(&input));
            assert_eq!(actual?.to_vec2::<f32>()?, native);
            assert_eq!(report, baseline_report);
            Ok(())
        })?;
    }
    Ok(())
}
