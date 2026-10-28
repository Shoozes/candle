//! Released 16 kHz log-mel frontend, without implicit resampling or Python.
use candle::{Device, Result, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub r#type: String,
    pub offset: f32,
    pub divisor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Padding {
    pub side: String,
    pub mode: String,
    pub value: f32,
    pub minimum_samples: usize,
    pub length_alignment_offset: usize,
    pub length_alignment_multiple: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub feature_type: String,
    pub sample_rate: u32,
    pub waveform_dtype: String,
    pub channel_reduction: String,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub window: String,
    pub window_periodic: bool,
    pub center: bool,
    pub normalized: bool,
    pub onesided: bool,
    pub power: f32,
    pub f_min: f64,
    pub f_max: f64,
    pub mel_scale: String,
    pub mel_norm: String,
    pub log_base: u32,
    pub logmel_floor: f32,
    pub normalization: Normalization,
    pub padding: Padding,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.feature_type != "log_mel"
            || self.waveform_dtype != "float32"
            || self.channel_reduction != "mean"
            || self.window != "hann"
            || !self.window_periodic
            || self.center
            || self.normalized
            || !self.onesided
            || self.power != 2.
            || self.mel_scale != "slaney"
            || self.mel_norm != "slaney"
            || self.log_base != 10
            || self.normalization.r#type != "affine"
            || self.f_min != 0.
            || self.f_max != self.sample_rate as f64 / 2.
            || self.padding.side != "right"
            || self.padding.mode != "constant"
            || self.padding.value != 0.
        {
            candle::bail!("unsupported LFM3-Audio frontend: expected released Hann/Slaney power-log-mel settings")
        }
        if self.sample_rate == 0
            || self.n_fft < 2
            || self.n_fft > 16_384
            || !self.n_fft.is_multiple_of(2)
            || self.win_length != self.n_fft
            || self.hop_length == 0
            || self.hop_length > self.n_fft
            || self.n_mels == 0
            || self.n_mels > 512
            || !self.logmel_floor.is_finite()
            || !self.normalization.offset.is_finite()
            || !self.normalization.divisor.is_finite()
            || self.normalization.divisor <= 0.
            || self.padding.minimum_samples != self.n_fft
            || self.padding.length_alignment_offset != self.n_fft
            || self.padding.length_alignment_multiple != self.hop_length
        {
            candle::bail!("invalid or unsupported LFM3-Audio frontend dimensions/normalization")
        }
        Ok(())
    }

    pub fn frame_count(&self, samples: usize) -> Result<usize> {
        self.validate()?;
        let tail = samples.saturating_sub(self.n_fft);
        tail.div_ceil(self.hop_length)
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("LFM3-Audio feature frame count overflow".into()))
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000. {
        hz / (200. / 3.)
    } else {
        15. + (hz / 1000.).ln() / (6.4f64.ln() / 27.)
    }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15. {
        mel * (200. / 3.)
    } else {
        1000. * ((mel - 15.) * (6.4f64.ln() / 27.)).exp()
    }
}

/// Mono features [1, mel, frames]. Multi-channel hosts should mean-reduce first.
pub fn extract(
    config: &Config,
    samples: &[f32],
    sample_rate: u32,
    max_samples: usize,
    device: &Device,
) -> Result<Tensor> {
    config.validate()?;
    if samples.is_empty()
        || max_samples == 0
        || samples.len() > max_samples
        || sample_rate != config.sample_rate
    {
        candle::bail!(
            "LFM3-Audio expects a nonempty bounded mono clip sampled at {} Hz",
            config.sample_rate
        )
    }
    if samples.iter().any(|v| !v.is_finite() || v.abs() > 1.) {
        candle::bail!("LFM3-Audio samples must be finite and within [-1, 1]")
    }
    let frames = config.frame_count(samples.len())?;
    let output_count = frames
        .checked_mul(config.n_mels)
        .filter(|n| n.checked_mul(4).is_some())
        .ok_or_else(|| candle::Error::Msg("LFM3-Audio feature allocation overflow".into()))?;
    let bins = config.n_fft / 2 + 1;
    let max_mel = hz_to_mel(config.f_max);
    let points = (0..config.n_mels + 2)
        .map(|i| mel_to_hz(max_mel * i as f64 / (config.n_mels + 1) as f64))
        .collect::<Vec<_>>();
    let filters = (0..config.n_mels)
        .map(|m| {
            (0..bins)
                .map(|k| {
                    let hz = k as f64 * config.sample_rate as f64 / config.n_fft as f64;
                    let lower = (hz - points[m]) / (points[m + 1] - points[m]);
                    let upper = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
                    (lower.min(upper).max(0.) * 2. / (points[m + 2] - points[m])) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let window = (0..config.n_fft)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / config.n_fft as f32).cos())
        .collect::<Vec<_>>();
    let fft = FftPlanner::new().plan_fft_forward(config.n_fft);
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let mut spectrum = vec![Complex32::default(); config.n_fft];
    let mut output = vec![0.; output_count];
    for frame in 0..frames {
        let start = frame * config.hop_length;
        for i in 0..config.n_fft {
            spectrum[i] = Complex32::new(
                samples.get(start + i).copied().unwrap_or(0.) * window[i],
                0.,
            );
        }
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        let power = spectrum[..bins]
            .iter()
            .map(|v| v.norm().powi(2))
            .collect::<Vec<_>>();
        for (m, filter) in filters.iter().enumerate() {
            let energy: f32 = power.iter().zip(filter).map(|(a, b)| a * b).sum();
            let normalized = (energy.log10().max(config.logmel_floor)
                + config.normalization.offset)
                / config.normalization.divisor;
            if !normalized.is_finite() {
                candle::bail!("LFM3-Audio feature computation overflow")
            }
            output[m * frames + frame] = normalized;
        }
    }
    Tensor::from_vec(output, (1, config.n_mels, frames), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Result<Config> {
        serde_json::from_str(r#"{"feature_type":"log_mel","sample_rate":16000,"waveform_dtype":"float32","channel_reduction":"mean","n_mels":128,"n_fft":400,"win_length":400,"hop_length":160,"window":"hann","window_periodic":true,"center":false,"normalized":false,"onesided":true,"power":2.0,"f_min":0.0,"f_max":8000.0,"mel_scale":"slaney","mel_norm":"slaney","log_base":10,"logmel_floor":-12.0,"normalization":{"type":"affine","offset":4.0,"divisor":4.0},"padding":{"side":"right","mode":"constant","value":0.0,"minimum_samples":400,"length_alignment_offset":400,"length_alignment_multiple":160}}"#).map_err(candle::Error::wrap)
    }
    #[test]
    fn silence_floor_alignment_and_input_admission() -> Result<()> {
        let c = config()?;
        for (length, frames) in [
            (1, 1),
            (399, 1),
            (400, 1),
            (401, 2),
            (560, 2),
            (561, 3),
            (16000, 99),
        ] {
            assert_eq!(c.frame_count(length)?, frames);
            let output = extract(&c, &vec![0.; length], 16000, 16000, &Device::Cpu)?;
            assert_eq!(output.dims(), [1, 128, frames]);
            assert!(output
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| *v == -2.));
        }
        assert!(extract(&c, &[0.], 24000, 16000, &Device::Cpu).is_err());
        assert!(extract(&c, &[f32::NAN], 16000, 16000, &Device::Cpu).is_err());
        assert!(extract(&c, &[1.01], 16000, 16000, &Device::Cpu).is_err());
        assert!(extract(&c, &[0.; 2], 16000, 1, &Device::Cpu).is_err());
        Ok(())
    }
    #[test]
    fn slaney_transition_and_hann_power_spectrum_against_scalar_reference() -> Result<()> {
        assert!((hz_to_mel(1000.) - 15.).abs() < 1e-12);
        for hz in [0., 200., 1000., 3500., 8000.] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9);
        }
        let c = config()?;
        // One-bin cosine has exactly three nonzero Hann-window DFT bins. Compare
        // the resulting mel energies with a separate analytic power calculation.
        let bin = 25;
        let samples = (0..400)
            .map(|i| (std::f32::consts::TAU * (i * bin) as f32 / 400.).cos() * 0.5)
            .collect::<Vec<_>>();
        let actual = extract(&c, &samples, 16000, 400, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let points = (0..130)
            .map(|i| mel_to_hz(hz_to_mel(8000.) * i as f64 / 129.))
            .collect::<Vec<_>>();
        for m in 0..128 {
            let mut energy = 0.;
            for (k, power) in [(bin - 1, 625.), (bin, 2500.), (bin + 1, 625.)] {
                let hz = k as f64 * 40.;
                let triangle = ((hz - points[m]) / (points[m + 1] - points[m]))
                    .min((points[m + 2] - hz) / (points[m + 2] - points[m + 1]))
                    .max(0.);
                energy += power * triangle * 2. / (points[m + 2] - points[m]);
            }
            if energy > 0. {
                let expected = (energy.log10() + 4.) / 4.;
                assert!((actual[m] as f64 - expected).abs() < 2e-5);
            }
        }
        Ok(())
    }
}
