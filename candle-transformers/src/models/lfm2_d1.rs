//! System One option probabilities from one answer-slot vocabulary tensor.

use candle::{DType, Result, Tensor};

/// Max-pool each option's single-token aliases, then normalize over options.
/// This reads logits; it never samples or generates an answer token.
pub fn option_probabilities(logits: &Tensor, groups: &[Vec<u32>]) -> Result<Vec<f64>> {
    let vocabulary = logits.dim(0)?;
    if logits.rank() != 1 || vocabulary == 0 || logits.dtype() != DType::F32 {
        candle::bail!("d1 readout requires a nonempty rank-one F32 vocabulary tensor")
    }
    if groups.is_empty() {
        candle::bail!("d1 readout requires at least one option")
    }
    let values = logits.to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        candle::bail!("d1 vocabulary logits must be finite")
    }
    // The shared vocabulary normalizer cancels when option scores are normalized.
    let mut scores = Vec::with_capacity(groups.len());
    for group in groups {
        if group.is_empty() {
            candle::bail!("d1 option token group cannot be empty")
        }
        let mut score = f64::NEG_INFINITY;
        for &id in group {
            let index = usize::try_from(id).map_err(candle::Error::wrap)?;
            let value = values.get(index).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "d1 option token {id} is outside vocabulary {vocabulary}"
                ))
            })?;
            score = score.max(f64::from(*value));
        }
        if !score.is_finite() {
            candle::bail!("d1 option logits must be finite")
        }
        scores.push(score);
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<_> = scores
        .into_iter()
        .map(|score| (score - maximum).exp())
        .collect();
    let denominator: f64 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= denominator;
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn d1_max_pools_aliases_and_preserves_ties() -> Result<()> {
        let logits = Tensor::new(&[0f32, 1., 2., 1.], &Device::Cpu)?;
        let actual = option_probabilities(&logits, &[vec![0, 2], vec![1, 3]])?;
        assert!((actual[0] - 1.0 / (1.0 + (-1f64).exp())).abs() < 1e-7);
        let tied = option_probabilities(&logits, &[vec![1], vec![3]])?;
        assert_eq!(tied, vec![0.5, 0.5]);
        Ok(())
    }

    #[test]
    fn d1_rejects_bad_groups_and_nonfinite_logits() -> Result<()> {
        let logits = Tensor::new(&[0f32, 1.], &Device::Cpu)?;
        for groups in [vec![], vec![vec![]], vec![vec![2]]] {
            assert!(option_probabilities(&logits, &groups).is_err());
        }
        let invalid = Tensor::new(&[f32::NAN, 0.], &Device::Cpu)?;
        assert!(option_probabilities(&invalid, &[vec![0], vec![1]]).is_err());
        let extreme = Tensor::new(&[1000f32, -1000.], &Device::Cpu)?;
        assert_eq!(
            option_probabilities(&extreme, &[vec![0], vec![1]])?,
            vec![1., 0.]
        );
        Ok(())
    }

    #[test]
    fn d1_finite_extreme_aliases_normalize_without_f32_overflow() -> Result<()> {
        let logits = Tensor::new(&[f32::MAX, -f32::MAX, f32::MAX], &Device::Cpu)?;
        let probabilities = option_probabilities(&logits, &[vec![0], vec![1], vec![2]])?;
        assert_eq!(probabilities, vec![0.5, 0., 0.5]);
        Ok(())
    }

    #[test]
    fn d1_nonfinite_unselected_logits_remain_rejected() -> Result<()> {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let logits = Tensor::new(&[1f32, 0., value], &Device::Cpu)?;
            assert!(option_probabilities(&logits, &[vec![0], vec![1]]).is_err());
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn d1_cuda_readout_matches_cpu_and_checks_unselected_logits() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let groups = [vec![0, 2], vec![1]];
        for values in [[0f32, 2., 2.], [f32::MAX, -f32::MAX, f32::MAX]] {
            let cpu = Tensor::new(&values, &Device::Cpu)?;
            let cuda = cpu.to_device(&device)?;
            assert_eq!(
                option_probabilities(&cpu, &groups)?,
                option_probabilities(&cuda, &groups)?,
            );
        }
        let invalid = Tensor::new(&[0f32, 1., f32::NAN], &device)?;
        assert!(option_probabilities(&invalid, &[vec![0], vec![1]]).is_err());
        Ok(())
    }

    fn benchmark_readout(device: &Device) -> Result<()> {
        let groups: Vec<Vec<u32>> = (0..6)
            .map(|index| vec![index * 31 + 1, index * 31 + 2, index * 31 + 3])
            .collect();
        for vocabulary in [4096, 128_000] {
            let values: Vec<f32> = (0..vocabulary)
                .map(|index| (index % 97) as f32 * 0.17 - 8.)
                .collect();
            let logits = Tensor::from_vec(values, vocabulary, device)?;
            for _ in 0..3 {
                std::hint::black_box(option_probabilities(&logits, &groups)?);
            }
            let mut samples = Vec::new();
            for _ in 0..5 {
                let started = std::time::Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(option_probabilities(
                        std::hint::black_box(&logits),
                        std::hint::black_box(&groups),
                    )?);
                }
                samples.push(started.elapsed().as_secs_f64() * 10_000.);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "d1-readout device={} release={} vocabulary={vocabulary} iterations=100 samples=5 median_us={:.3} min_us={:.3} max_us={:.3} model_forwards=0",
                if device.is_cpu() { "cpu" } else { "cuda" },
                !cfg!(debug_assertions),
                samples[2], samples[0], samples[4],
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "explicit readout microbenchmark; zero model forwards"]
    fn d1_readout_benchmark_cpu() -> Result<()> {
        benchmark_readout(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit CUDA readout microbenchmark; zero model forwards"]
    fn d1_readout_benchmark_cuda() -> Result<()> {
        benchmark_readout(&Device::new_cuda(0)?)
    }
}
