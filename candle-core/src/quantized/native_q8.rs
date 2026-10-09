//! Call-scoped native Q8 admission and dispatch accounting.

use crate::Result;
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeQ8Execution {
    pub cpu_matmuls: u64,
    pub cuda_mmvq: u64,
    pub cuda_mmq: u64,
    pub cuda_other_q8: u64,
    pub cuda_f32_q8: u64,
}

thread_local! {
    static EXECUTION: Cell<Option<NativeQ8Execution>> = const { Cell::new(None) };
}

struct Restore(Option<NativeQ8Execution>);

impl Drop for Restore {
    fn drop(&mut self) {
        EXECUTION.with(|state| state.set(self.0));
    }
}

/// Require retained Q8_0 linear weights and native execution for this call.
/// The dispatch report survives an error; the prior policy is restored on unwind.
pub fn with_native_q8_0<T>(f: impl FnOnce() -> Result<T>) -> (Result<T>, NativeQ8Execution) {
    let previous = EXECUTION.with(|state| state.replace(Some(NativeQ8Execution::default())));
    let restore = Restore(previous);
    let result = f();
    let report = EXECUTION.with(|state| state.get().unwrap_or_default());
    drop(restore);
    (result, report)
}

pub(crate) fn required() -> bool {
    EXECUTION.with(|state| state.get().is_some())
}

pub(crate) fn record(kind: u8) -> Result<()> {
    EXECUTION.with(|state| {
        if let Some(mut report) = state.get() {
            let count = match kind {
                0 => &mut report.cpu_matmuls,
                1 => &mut report.cuda_mmvq,
                2 => &mut report.cuda_mmq,
                4 => &mut report.cuda_f32_q8,
                _ => &mut report.cuda_other_q8,
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| crate::Error::Msg("native Q8 dispatch counter overflow".into()))?;
            state.set(Some(report));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_q8_policy_restores_after_failure_and_nested_calls() {
        assert!(!required());
        let (result, report) = with_native_q8_0(|| -> Result<()> {
            record(0)?;
            let (_, nested) = with_native_q8_0(|| record(2));
            assert_eq!(nested.cuda_mmq, 1);
            assert!(required());
            crate::bail!("authored failure")
        });
        assert!(result.is_err());
        assert_eq!(report.cpu_matmuls, 1);
        assert_eq!(report.cuda_mmq, 0);
        assert!(!required());
    }

    #[test]
    fn native_q8_rejects_dense_and_non_f32_without_dispatch() -> Result<()> {
        use crate::{
            quantized::{GgmlDType, QMatMul, QTensor},
            DType, Device, Module, Tensor,
        };
        let device = Device::Cpu;
        let input = Tensor::ones((2, 32), DType::F32, &device)?;
        let weight = Tensor::ones((32, 32), DType::F32, &device)?;
        let dense = QMatMul::Tensor(weight.clone());
        let (result, report) = with_native_q8_0(|| dense.forward(&input));
        assert!(result.is_err());
        assert_eq!(report, NativeQ8Execution::default());
        let quantized = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
            &weight,
            GgmlDType::Q8_0,
        )?));
        let (result, report) = with_native_q8_0(|| quantized.forward(&input.to_dtype(DType::F16)?));
        assert!(result.is_err());
        assert_eq!(report, NativeQ8Execution::default());
        let (result, report) = with_native_q8_0(|| quantized.forward(&input));
        assert_eq!(result?.dims(), [2, 32]);
        assert_eq!(report.cpu_matmuls, 1);
        assert_eq!(dense.forward(&input)?.dims(), [2, 32]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn authored_values(count: usize, modulus: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 17 % modulus) as f32 - (modulus / 2) as f32) * 0.01)
            .collect()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_q8_cuda_rows_tails_offsets_and_batches() -> Result<()> {
        use crate::{
            quantized::{GgmlDType, QMatMul, QTensor},
            Device, Module, Tensor,
        };
        let device = Device::new_cuda(0)?;
        for (rows, width, columns) in [
            (1, 32, 1),
            (2, 64, 3),
            (3, 96, 7),
            (4, 128, 4),
            (5, 160, 5),
            (9, 256, 9),
            (64, 64, 64),
        ] {
            let weights = Tensor::from_vec(
                authored_values(width * columns, 101),
                (columns, width),
                &Device::Cpu,
            )?;
            let cpu_q = QTensor::quantize(&weights, GgmlDType::Q8_0)?;
            let gpu_q = QTensor::quantize_onto(&weights, GgmlDType::Q8_0, &device)?;
            assert_eq!(cpu_q.data()?.as_ref(), gpu_q.data()?.as_ref());
            let cpu = QMatMul::QTensor(std::sync::Arc::new(cpu_q));
            let gpu = QMatMul::QTensor(std::sync::Arc::new(gpu_q));
            // A leading row makes the tested contiguous view's offset nonzero.
            let input = Tensor::from_vec(
                authored_values((rows + 1) * width, 97),
                (rows + 1, width),
                &Device::Cpu,
            )?
            .narrow(0, 1, rows)?;
            let (expected, cpu_report) = with_native_q8_0(|| cpu.forward(&input));
            assert_eq!(cpu_report.cpu_matmuls, 1);
            let expected = expected?.flatten_all()?.to_vec1::<f32>()?;
            let gpu_input = Tensor::from_vec(
                authored_values((rows + 1) * width, 97),
                (rows + 1, width),
                &device,
            )?
            .narrow(0, 1, rows)?;
            for input in [gpu_input.clone(), gpu_input.unsqueeze(0)?] {
                let (actual, report) = with_native_q8_0(|| gpu.forward(&input));
                assert_eq!(report.cuda_f32_q8, 1);
                assert_eq!(report.cuda_mmq + report.cuda_mmvq + report.cuda_other_q8, 0);
                let actual = actual?.flatten_all()?.to_vec1::<f32>()?;
                let error = expected
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(error <= 0.0003, "shape {rows}/{width}/{columns}: {error}");
            }
            let permuted = gpu_input.t()?;
            if rows == width {
                let (result, report) = with_native_q8_0(|| gpu.forward(&permuted));
                assert!(result.is_err());
                assert_eq!(report, NativeQ8Execution::default());
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Explicit authored CUDA timing; no production model forwards"]
    fn native_q8_cuda_benchmark() -> Result<()> {
        use crate::{
            quantized::{GgmlDType, QMatMul, QTensor},
            Device, Module, Tensor,
        };
        let device = Device::new_cuda(0)?;
        for (rows, width, columns) in [
            (256, 2048, 2048),
            (256, 2048, 6144),
            (256, 1152, 1152),
            (1, 2048, 128000),
        ] {
            let weights = Tensor::from_vec(
                authored_values(width * columns, 101),
                (columns, width),
                &Device::Cpu,
            )?;
            let quantized = QTensor::quantize_onto(&weights, GgmlDType::Q8_0, &device)?;
            drop(weights);
            let matmul = QMatMul::QTensor(std::sync::Arc::new(quantized));
            let input =
                Tensor::from_vec(authored_values(rows * width, 97), (rows, width), &device)?;
            for _ in 0..3 {
                let (result, _) = with_native_q8_0(|| matmul.forward(&input));
                std::hint::black_box(result?);
                device.synchronize()?;
            }
            let mut samples = Vec::new();
            for _ in 0..5 {
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    let (result, report) = with_native_q8_0(|| matmul.forward(&input));
                    std::hint::black_box(result?);
                    assert_eq!(report.cuda_f32_q8, 1);
                    device.synchronize()?;
                }
                samples.push(started.elapsed().as_secs_f64() * 1000. / 10.);
            }
            samples.sort_by(f64::total_cmp);
            println!("native-q8-cuda rows={rows} width={width} columns={columns} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                samples[2], samples[0], samples[4]);
        }
        Ok(())
    }
}
