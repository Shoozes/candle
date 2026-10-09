//! Call-scoped selection of CPU quantized matrix multiplication.

use crate::Result;
use std::cell::Cell;

/// CPU execution choice for quantized linears with F32 or BF16 activations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuQuantizedMatmulMode {
    /// Preserve native Q8 admission and optimized repacks, then use generic tiling.
    #[default]
    Auto,
    /// Bypass repacks and stream weight quads across up to sixteen activation rows.
    GenericTiled,
    /// Bypass repacks and run the legacy generic scheduler one activation row at a time.
    GenericRowwise,
}

thread_local! {
    static MODE: Cell<CpuQuantizedMatmulMode> = const { Cell::new(CpuQuantizedMatmulMode::Auto) };
}

/// Select CPU quantized execution for a synchronous operation on the calling thread.
/// Nested calls restore the previous selection on return, error, or unwind. Other
/// threads, dense linears, CUDA and Metal retain their existing behavior. Explicit
/// generic modes reject F16 activations and conflict with strict native CPU Q8.
pub fn with_cpu_quantized_matmul_mode<T>(
    mode: CpuQuantizedMatmulMode,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    struct Restore(CpuQuantizedMatmulMode);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODE.with(|state| state.set(self.0));
        }
    }
    let restore = Restore(MODE.with(|state| state.replace(mode)));
    let result = operation();
    drop(restore);
    result
}

pub(super) fn current() -> CpuQuantizedMatmulMode {
    MODE.with(Cell::get)
}

pub(super) fn validate_native() -> Result<()> {
    if current() != CpuQuantizedMatmulMode::Auto && super::native_q8::required() {
        crate::bail!("explicit generic CPU quantized matmul conflicts with native Q8/F32; use Auto")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_error_unwind_and_thread_isolation() -> Result<()> {
        assert_eq!(current(), CpuQuantizedMatmulMode::Auto);
        with_cpu_quantized_matmul_mode(CpuQuantizedMatmulMode::GenericTiled, || {
            assert_eq!(current(), CpuQuantizedMatmulMode::GenericTiled);
            let error = with_cpu_quantized_matmul_mode(
                CpuQuantizedMatmulMode::GenericRowwise,
                || -> Result<()> { crate::bail!("authored error") },
            );
            assert!(error.is_err());
            assert_eq!(current(), CpuQuantizedMatmulMode::GenericTiled);
            assert!(std::panic::catch_unwind(|| {
                let _ = with_cpu_quantized_matmul_mode(
                    CpuQuantizedMatmulMode::Auto,
                    || -> Result<()> { panic!("authored unwind") },
                );
            })
            .is_err());
            assert_eq!(current(), CpuQuantizedMatmulMode::GenericTiled);
            assert_eq!(
                std::thread::spawn(current).join().unwrap(),
                CpuQuantizedMatmulMode::Auto
            );
            Ok(())
        })?;
        assert_eq!(current(), CpuQuantizedMatmulMode::Auto);
        Ok(())
    }
}
