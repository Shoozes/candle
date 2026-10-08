// Numerical contract: Pillow 11.3 Resample.c (MIT-CMU).
// Copyright 1997-2011 Secret Labs AB; 1995-2011 Fredrik Lundh and contributors;
// 2010 Jeffrey A. Clark and contributors. See docs/lfm2-vl/licenses/PILLOW_LICENSE.txt.
use candle::Result;
use image::{DynamicImage, RgbImage};

// The d1 runner caps pixels before the ordinary LFM2-VL processor.
pub(super) const MAX_PIXELS: u64 = 1024 * 1024;

pub(super) fn cap_pixels(image: &DynamicImage, maximum: u64) -> Result<DynamicImage> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 || maximum == 0 {
        candle::bail!("d1 image dimensions and pixel limit must be positive")
    }
    let area = u64::from(width) * u64::from(height);
    let rgb = image.to_rgb8();
    if area <= maximum {
        return Ok(DynamicImage::ImageRgb8(rgb));
    }
    let scale = (maximum as f64 / area as f64).sqrt();
    let width = (f64::from(width) * scale).floor().max(1.) as u32;
    let height = (f64::from(height) * scale).floor().max(1.) as u32;
    Ok(DynamicImage::ImageRgb8(resize(&rgb, width, height)?))
}

// Candle-native implementation of the Pillow 11.3 RGB bicubic contract:
// a=-0.5, antialias support, 22-bit coefficients, byte rounding per pass.
fn axis(input: u32, output: u32) -> Result<Vec<(usize, Vec<i64>)>> {
    let scale = f64::from(input) / f64::from(output);
    let filter_scale = scale.max(1.);
    let support = 2. * filter_scale;
    let mut result = Vec::new();
    result
        .try_reserve_exact(output as usize)
        .map_err(candle::Error::wrap)?;
    for index in 0..output {
        let center = (f64::from(index) + 0.5) * scale;
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(i64::from(input)) as usize;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(end - start)
            .map_err(candle::Error::wrap)?;
        for source in start..end {
            let x = ((source as f64 - center + 0.5) / filter_scale).abs();
            weights.push(if x < 1. {
                (1.5 * x - 2.5) * x * x + 1.
            } else if x < 2. {
                (((x - 5.) * x + 8.) * x - 4.) * -0.5
            } else {
                0.
            });
        }
        let sum: f64 = weights.iter().sum();
        if sum == 0. || !sum.is_finite() {
            candle::bail!("invalid d1 bicubic coefficients")
        }
        result.push((
            start,
            weights
                .iter()
                .map(|weight| (weight / sum * ((1 << 22) as f64)).round() as i64)
                .collect(),
        ));
    }
    Ok(result)
}

fn buffer(width: u32, height: u32) -> Result<Vec<u8>> {
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| candle::Error::Msg("d1 image allocation overflow".into()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(candle::Error::wrap)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn resize(input: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let horizontal = axis(input.width(), width)?;
    let vertical = axis(input.height(), height)?;
    let mut temp = buffer(width, input.height())?;
    for y in 0..input.height() as usize {
        for (x, (start, weights)) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let sum = weights
                    .iter()
                    .enumerate()
                    .fold(1i64 << 21, |sum, (i, weight)| {
                        sum + i64::from(
                            input.as_raw()[(y * input.width() as usize + start + i) * 3 + channel],
                        ) * weight
                    });
                temp[(y * width as usize + x) * 3 + channel] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    let mut bytes = buffer(width, height)?;
    for (y, (start, weights)) in vertical.iter().enumerate() {
        for x in 0..width as usize {
            for channel in 0..3 {
                let sum = weights
                    .iter()
                    .enumerate()
                    .fold(1i64 << 21, |sum, (i, weight)| {
                        sum + i64::from(temp[((start + i) * width as usize + x) * 3 + channel])
                            * weight
                    });
                bytes[(y * width as usize + x) * 3 + channel] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    RgbImage::from_raw(width, height, bytes)
        .ok_or_else(|| candle::Error::Msg("invalid d1 resized image storage".into()))
}
