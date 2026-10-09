use super::*;
use crate::quantized::k_quants::BlockQ8_0;

fn weights(k: usize, n: usize) -> Vec<BlockQ8_0> {
    (0..n * (k / 32))
        .map(|block| BlockQ8_0 {
            d: half::f16::from_f32(((block % 7) + 1) as f32 / 128.),
            qs: std::array::from_fn(|i| ((block * 17 + i * 13) % 101) as i8 - 50),
        })
        .collect()
}

fn inputs(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i * 17 % 97) as f32 - 48.) / 64.)
        .collect()
}

// Frozen pre-optimization scheduler and dot math from 7425c843.
fn legacy_rows(
    storage: &dyn QuantizedType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    dst: &mut [f32],
) {
    assert_eq!(storage.dtype(), GgmlDType::Q8_0);
    assert!(k > 0 && n > 0 && k.is_multiple_of(32));
    assert_eq!(lhs.len(), m * k);
    assert_eq!(dst.len(), m * n);
    assert_eq!(storage.storage_size_in_bytes(), n * (k / 32) * 34);
    let weights =
        unsafe { std::slice::from_raw_parts(storage.as_ptr(), storage.storage_size_in_bytes()) };
    dst.par_chunks_mut(n * 4)
        .enumerate()
        .for_each(|(index, output)| {
            let rows = output.len() / n;
            let input = &lhs[index * 4 * k..index * 4 * k + rows * k];
            for column in 0..n {
                let weight = &weights[column * (k / 32) * 34..(column + 1) * (k / 32) * 34];
                let values = legacy_dot(weight, input, rows, k);
                for r in 0..rows {
                    output[r * n + column] = values[r];
                }
            }
        });
}

fn legacy_dot(weights: &[u8], input: &[f32], rows: usize, k: usize) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { legacy_avx_dot(weights, input, rows, k) };
    }
    let mut result = [0f32; 4];
    for (block, values) in weights.chunks_exact(34).enumerate() {
        let scale = half::f16::from_bits(u16::from_le_bytes([values[0], values[1]])).to_f32();
        for i in 0..32 {
            let weight = f32::from(values[i + 2] as i8) * scale;
            for row in 0..rows {
                result[row] += input[row * k + block * 32 + i] * weight;
            }
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn legacy_avx_dot(weights: &[u8], input: &[f32], rows: usize, k: usize) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    for (block, values) in weights.chunks_exact(34).enumerate() {
        let scale = _mm256_set1_ps(
            half::f16::from_bits(u16::from_le_bytes([values[0], values[1]])).to_f32(),
        );
        for group in 0..4 {
            let packed = _mm_loadl_epi64(values.as_ptr().add(2 + group * 8).cast());
            let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), scale);
            for (row, sum) in sums.iter_mut().take(rows).enumerate() {
                let x = _mm256_loadu_ps(input.as_ptr().add(row * k + block * 32 + group * 8));
                *sum = _mm256_fmadd_ps(weight, x, *sum);
            }
        }
    }
    let mut result = [0f32; 4];
    for row in 0..rows {
        let mut values = [0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), sums[row]);
        result[row] = values.iter().sum();
    }
    result
}

#[test]
fn column_schedule_is_exact_and_matches_independent_scalar_math() -> Result<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .map_err(crate::Error::msg)?;
    for (m, k, n) in [
        (1, 32, 1),
        (1, 96, 257),
        (1, 2048, 255),
        (1, 2048, 256),
        (1, 2048, 257),
        (1, 2048, 511),
        (1, 2048, 512),
        (1, 2048, 513),
        (2, 256, 511),
        (3, 2048, 513),
        (4, 32, 1024),
        (5, 64, 2048),
        (9, 2048, 257),
    ] {
        let mut storage = weights(k, n);
        for block in &mut storage {
            block.d = half::f16::from_f32(block.d.to_f32() * 0.371);
        }
        let input: Vec<_> = inputs(m, k)
            .into_iter()
            .map(|x| x * 0.137 + 0.017)
            .collect();
        let mut reference = vec![0.; m * n];
        legacy_rows(&storage, (m, k, n), &input, &mut reference);
        let mut output = vec![42.; m * n];
        pool.install(|| matmul(&storage, (m, k, n), &input, &mut output))?;
        assert_eq!(
            output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "exact {m}/{k}/{n}"
        );
        for row in 0..m {
            for column in 0..n {
                let mut scalar = 0f64;
                for block in 0..k / 32 {
                    let weight = &storage[column * (k / 32) + block];
                    for i in 0..32 {
                        scalar += f64::from(weight.d.to_f32())
                            * f64::from(weight.qs[i])
                            * f64::from(input[row * k + block * 32 + i]);
                    }
                }
                assert!(
                    (f64::from(output[row * n + column]) - scalar).abs() <= 0.0003,
                    "scalar {m}/{k}/{n} at {row}/{column}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn public_native_dispatch_preserves_offsets_batches_and_q8_storage() -> Result<()> {
    use crate::quantized::{with_native_q8_0, QMatMul, QStorage, QTensor};
    use crate::{Device, Module, Tensor};
    for (m, k, n) in [(1, 2048, 513), (4, 256, 513), (9, 64, 257)] {
        let storage = weights(k, n);
        let raw = inputs(m + 1, k);
        let lhs = &raw[k..];
        let mut expected = vec![0.; m * n];
        legacy_rows(&storage, (m, k, n), lhs, &mut expected);
        let q = QTensor::new(QStorage::Cpu(Box::new(storage)), (n, k))?;
        let before = q.data()?.into_owned();
        let model = QMatMul::QTensor(std::sync::Arc::new(q));
        let input = Tensor::from_vec(raw, (m + 1, k), &Device::Cpu)?.narrow(0, 1, m)?;
        for input in [input.clone(), input.unsqueeze(0)?] {
            let (result, report) = with_native_q8_0(|| model.forward(&input));
            assert_eq!(report.cpu_matmuls, 1);
            assert_eq!(result?.flatten_all()?.to_vec1::<f32>()?, expected);
        }
        let QMatMul::QTensor(q) = model else {
            unreachable!()
        };
        assert_eq!(q.dtype(), GgmlDType::Q8_0);
        assert_eq!(q.data()?.as_ref(), before);
    }
    Ok(())
}

#[test]
fn malformed_native_inputs_fail_without_writing_output() {
    let storage = weights(32, 4);
    for (dims, lhs_len, dst_len) in [
        ((1, 32, 4), 31, 4),
        ((1, 32, 4), 32, 3),
        ((1, 31, 4), 31, 4),
        ((1, 32, 5), 32, 5),
        ((1, 0, 4), 0, 4),
        ((1, 32, 0), 32, 0),
        ((usize::MAX, 32, 4), 0, 0),
        ((1, usize::MAX, 4), 0, 4),
    ] {
        let mut output = vec![42.; dst_len];
        assert!(matmul(&storage, dims, &vec![1.; lhs_len], &mut output).is_err());
        assert_eq!(output, vec![42.; dst_len]);
    }
}

/// Frozen authored profile: 1/16 workers, three warmups, five alternating-order
/// samples, three calls per sample. No model weights or model forwards.
#[test]
#[ignore = "Explicit bounded native Q8/F32 CPU profiling"]
fn profile_single_row_projection() -> Result<()> {
    for (m, k, n) in [
        (1, 32, 17),
        (1, 2048, 256),
        (1, 2048, 257),
        (1, 2048, 2048),
        (4, 2048, 2048),
        (9, 2048, 2048),
        (1, 2048, 128000),
        (4, 2048, 128000),
    ] {
        let storage = weights(k, n);
        let input = inputs(m, k);
        let mut legacy = vec![0.; m * n];
        let mut selected = vec![0.; m * n];
        for threads in [1, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(crate::Error::msg)?;
            let mut samples = [Vec::new(), Vec::new()];
            pool.install(|| -> Result<()> {
                for _ in 0..3 {
                    legacy_rows(&storage, (m, k, n), &input, &mut legacy);
                    matmul(&storage, (m, k, n), &input, &mut selected)?;
                }
                assert_eq!(selected, legacy, "warm parity {m}/{k}/{n}");
                for sample in 0..5 {
                    for mode in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let started = std::time::Instant::now();
                        for _ in 0..3 {
                            if mode == 0 {
                                legacy_rows(&storage, (m, k, n), &input, &mut legacy);
                            } else {
                                matmul(&storage, (m, k, n), &input, &mut selected)?;
                            }
                            std::hint::black_box(if mode == 0 { &legacy } else { &selected });
                        }
                        samples[mode].push(started.elapsed().as_secs_f64() * 1000. / 3.);
                    }
                    assert_eq!(selected, legacy, "sample parity {m}/{k}/{n}");
                }
                Ok(())
            })?;
            for (mode, label) in ["legacy", "selected"].into_iter().enumerate() {
                samples[mode].sort_by(f64::total_cmp);
                println!("native-q8-cpu rows={m} width={k} columns={n} threads={threads} scheduler={label} median_ms={:.6} min_ms={:.6} max_ms={:.6}", samples[mode][2], samples[mode][0], samples[mode][4]);
            }
        }
    }
    Ok(())
}
