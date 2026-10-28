use candle::Result;
use rustfft::{num_complex::Complex32, FftPlanner};

/// Predicted [log magnitudes | angles] to symmetrically trimmed overlap-add PCM.
pub(crate) fn synthesize(features: &[Vec<f32>], window: &[f32], hop: usize) -> Result<Vec<f32>> {
    let (audio, envelope) = overlap(features, window, hop)?;
    let trim = (window.len() - hop) / 2;
    normalize(
        &audio[trim..audio.len() - trim],
        &envelope[trim..envelope.len() - trim],
    )
}

pub(super) fn overlap(
    features: &[Vec<f32>],
    window: &[f32],
    hop: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = window.len();
    if features.is_empty()
        || n == 0
        || !n.is_multiple_of(2)
        || hop == 0
        || hop >= n
        || !(n - hop).is_multiple_of(2)
        || window.iter().any(|v| !v.is_finite())
    {
        candle::bail!("invalid LiquidCodec inverse STFT dimensions")
    }
    let length = (features.len() - 1)
        .checked_mul(hop)
        .and_then(|l| l.checked_add(n))
        .filter(|l| l.checked_mul(4).is_some())
        .ok_or_else(|| candle::Error::Msg("LiquidCodec inverse STFT allocation overflow".into()))?;
    let bins = n / 2 + 1;
    let mut audio = vec![0f32; length];
    let mut envelope = vec![0f32; length];
    let fft = FftPlanner::new().plan_fft_inverse(n);
    let mut spectrum = vec![Complex32::default(); n];
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    for (frame, predicted) in features.iter().enumerate() {
        if predicted.len() != 2 * bins || predicted.iter().any(|v| !v.is_finite()) {
            candle::bail!("LiquidCodec predicted spectrum has invalid shape or nonfinite values")
        }
        for k in 0..bins {
            let magnitude = predicted[k].exp();
            if !magnitude.is_finite() {
                candle::bail!("LiquidCodec magnitude overflow")
            }
            spectrum[k] = Complex32::from_polar(magnitude, predicted[bins + k]);
        }
        // Real inverse FFT ignores imaginary DC and Nyquist components.
        spectrum[0].im = 0.;
        spectrum[n / 2].im = 0.;
        for k in 1..n / 2 {
            spectrum[n - k] = spectrum[k].conj();
        }
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        let start = frame * hop;
        for i in 0..n {
            audio[start + i] += spectrum[i].re / n as f32 * window[i];
            envelope[start + i] += window[i] * window[i];
        }
    }
    Ok((audio, envelope))
}

pub(super) fn normalize(audio: &[f32], envelope: &[f32]) -> Result<Vec<f32>> {
    if audio.len() != envelope.len() {
        candle::bail!("LiquidCodec overlap envelope length mismatch")
    }
    let mut output = Vec::with_capacity(audio.len());
    for i in 0..audio.len() {
        if !envelope[i].is_finite() || envelope[i] <= 0.0 {
            candle::bail!("LiquidCodec inverse STFT has a zero or invalid window envelope")
        }
        let value = audio[i] / envelope[i];
        if !value.is_finite() {
            candle::bail!("LiquidCodec inverse STFT produced nonfinite PCM")
        }
        output.push(value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_fft_matches_independent_direct_inverse_dft_and_overlap_envelope() -> Result<()> {
        let n = 8;
        let hop = 2;
        let bins = n / 2 + 1;
        let window = (0..n)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
            .collect::<Vec<_>>();
        let features = (0..3)
            .map(|j| {
                (0..2 * bins)
                    .map(|i| 0.13 * (j * bins + i) as f32 - 0.4)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let output = synthesize(&features, &window, hop)?;
        let mut expected = [0f64; 12];
        let mut envelope = [0f64; 12];
        for (j, f) in features.iter().enumerate() {
            for t in 0..n {
                let mut value = f[0].exp() as f64 * (f[bins] as f64).cos();
                value += f[n / 2].exp() as f64
                    * (f[bins + n / 2] as f64).cos()
                    * if t % 2 == 0 { 1. } else { -1. };
                for k in 1..n / 2 {
                    let angle =
                        std::f64::consts::TAU * (t * k) as f64 / n as f64 + f[bins + k] as f64;
                    value += 2. * f[k].exp() as f64 * angle.cos();
                }
                expected[j * hop + t] += value / n as f64 * window[t] as f64;
                envelope[j * hop + t] += (window[t] as f64).powi(2);
            }
        }
        assert_eq!(output.len(), 6);
        for (i, a) in output.iter().enumerate() {
            let p = i + 3;
            assert!((*a as f64 - expected[p] / envelope[p]).abs() < 2e-6);
        }
        assert!(synthesize(&[vec![f32::NAN; 10]], &window, hop).is_err());
        Ok(())
    }
}
