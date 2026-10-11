//! Opt-in synchronized phase timing, independent of numerical tracing.

use candle::{Device, Result};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize)]
pub struct D1QuestionTiming {
    pub index: usize,
    pub prefill_ms: f64,
    pub readout_ms: f64,
}

/// Wall-clock milliseconds. GPU work is synchronized only when profiling.
/// Observation includes the caller's trace callback; total also includes overhead.
#[derive(Clone, Debug, Default, Serialize)]
pub struct D1Timings {
    pub preparation_ms: f64,
    pub image_processing_ms: f64,
    pub tokenization_ms: f64,
    pub vision_ms: f64,
    pub observation_ms: f64,
    pub questions: Vec<D1QuestionTiming>,
    pub total_ms: f64,
}

#[derive(Clone, Copy)]
pub(super) enum Phase {
    Preparation,
    ImageProcessing,
    Tokenization,
    Vision,
    Observation,
    Prefill(usize),
    Readout(usize),
}

#[cfg(test)]
thread_local! {
    static FAIL_VISION_SYNC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_vision_sync_failure<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FAIL_VISION_SYNC.with(|state| state.set(self.0));
        }
    }
    let restore = Restore(FAIL_VISION_SYNC.with(|state| state.replace(true)));
    let result = run();
    drop(restore);
    result
}

pub(super) fn measure<T>(
    timings: &mut Option<&mut D1Timings>,
    phase: Phase,
    device: Option<&Device>,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(timings) = timings.as_deref_mut() else {
        return run();
    };
    if let Some(device) = device {
        #[cfg(test)]
        if matches!(phase, Phase::Vision) && FAIL_VISION_SYNC.with(|state| state.replace(false)) {
            candle::bail!("authored vision synchronization failure")
        }
        device.synchronize()?;
    }
    let started = Instant::now();
    let result = run();
    let synchronized = device.map_or(Ok(()), Device::synchronize);
    let elapsed = started.elapsed().as_secs_f64() * 1000.;
    match phase {
        Phase::Preparation => timings.preparation_ms += elapsed,
        Phase::ImageProcessing => timings.image_processing_ms += elapsed,
        Phase::Tokenization => timings.tokenization_ms += elapsed,
        Phase::Vision => timings.vision_ms += elapsed,
        Phase::Observation => timings.observation_ms += elapsed,
        Phase::Prefill(index) | Phase::Readout(index) => {
            if let Some(question) = timings.questions.iter_mut().find(|q| q.index == index) {
                match phase {
                    Phase::Prefill(_) => question.prefill_ms += elapsed,
                    _ => question.readout_ms += elapsed,
                }
            }
        }
    }
    let value = result?;
    synchronized?;
    Ok(value)
}
