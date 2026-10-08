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
}
