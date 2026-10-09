//! Causal depthwise convolution without one backend launch per channel.

use candle::{DType, Result, Tensor};

pub(super) fn causal_depthwise(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, channels, length) = input.dims3()?;
    let (weight_channels, taps) = weights.dims2()?;
    if channels != weight_channels
        || batch == 0
        || channels == 0
        || length == 0
        || taps == 0
        || input.dtype() != DType::F32
        || weights.dtype() != DType::F32
    {
        candle::bail!("causal depthwise convolution requires nonempty matching F32 tensors")
    }
    let padded_length = length
        .checked_add(taps - 1)
        .ok_or_else(|| candle::Error::Msg("causal convolution length overflow".into()))?;
    batch
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(padded_length))
        .ok_or_else(|| candle::Error::Msg("causal convolution allocation overflow".into()))?;
    let padded = input.pad_with_zeros(2, taps - 1, 0)?;
    let mut output = None;
    for tap in 0..taps {
        let value = padded
            .narrow(2, tap, length)?
            .broadcast_mul(&weights.narrow(1, tap, 1)?.unsqueeze(0)?)?;
        output = Some(match output {
            None => value,
            Some(previous) => (previous + value)?,
        });
    }
    output.ok_or_else(|| candle::Error::Msg("causal convolution has no taps".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 13 % 97) as f32 - 48.) * 0.01)
            .collect()
    }

    fn compare(device: &Device) -> Result<()> {
        for (batch, channels, length, taps) in [(1, 1, 2, 1), (2, 3, 5, 3), (1, 31, 17, 7)] {
            // Non-contiguous input with a nonzero backing offset.
            let input = Tensor::from_vec(
                values((batch + 1) * length * channels),
                (batch + 1, length, channels),
                &Device::Cpu,
            )?
            .narrow(0, 1, batch)?
            .transpose(1, 2)?;
            let weights =
                Tensor::from_vec(values(channels * taps), (channels, taps), &Device::Cpu)?;
            let expected = input
                .conv1d(&weights.unsqueeze(1)?, taps - 1, 1, 1, channels)?
                .narrow(2, 0, length)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = causal_depthwise(&input.to_device(device)?, &weights.to_device(device)?)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(expected.len(), actual.len());
            let error = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(error <= 0.00001, "causal convolution error {error}");
        }
        let empty = Tensor::zeros((1, 2, 0), DType::F32, device)?;
        let weights = Tensor::ones((2, 3), DType::F32, device)?;
        assert!(causal_depthwise(&empty, &weights).is_err());
        let input = Tensor::ones((1, 3, 5), DType::F32, device)?;
        assert!(causal_depthwise(&input, &weights).is_err());
        assert!(causal_depthwise(
            &input.to_dtype(DType::F16)?,
            &Tensor::ones((3, 3), DType::F16, device)?
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn causal_depthwise_cpu_matches_grouped_reference() -> Result<()> {
        compare(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn causal_depthwise_cuda_matches_cpu_grouped_reference() -> Result<()> {
        compare(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Explicit authored CUDA convolution timing; no production model forwards"]
    fn causal_depthwise_cuda_benchmark() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(values(2048 * 256), (1, 2048, 256), &device)?;
        let weights = Tensor::from_vec(values(2048 * 3), (2048, 3), &device)?;
        for grouped in [true, false] {
            let run = || -> Result<Tensor> {
                if grouped {
                    input
                        .conv1d(&weights.unsqueeze(1)?, 2, 1, 1, 2048)?
                        .narrow(2, 0, 256)
                } else {
                    causal_depthwise(&input, &weights)
                }
            };
            for _ in 0..3 {
                std::hint::black_box(run()?);
                device.synchronize()?;
            }
            let mut samples = Vec::new();
            for _ in 0..5 {
                let started = std::time::Instant::now();
                std::hint::black_box(run()?);
                device.synchronize()?;
                samples.push(started.elapsed().as_secs_f64() * 1000.);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "causal-depthwise-cuda grouped={grouped} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                samples[2], samples[0], samples[4]
            );
        }
        Ok(())
    }
}
