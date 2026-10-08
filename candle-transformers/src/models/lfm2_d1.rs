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
    let logz = candle_nn::ops::log_softmax(logits, 0)?.to_vec1::<f32>()?;
    let mut scores = Vec::with_capacity(groups.len());
    for group in groups {
        if group.is_empty() {
            candle::bail!("d1 option token group cannot be empty")
        }
        let mut score = f64::NEG_INFINITY;
        for &id in group {
            let index = usize::try_from(id).map_err(candle::Error::wrap)?;
            let value = logz.get(index).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "d1 option token {id} is outside vocabulary {vocabulary}"
                ))
            })?;
            score = score.max(f64::from(*value));
        }
        if !score.is_finite() {
            candle::bail!("d1 normalized option logits must be finite")
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
}
