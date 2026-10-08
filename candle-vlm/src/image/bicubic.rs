//! TorchVision v2 / Torch 2.8 CPU uint8 bicubic antialias contract.
//! Uses F64 coefficient construction, signed fixed-point weights, and byte
//! rounding/clamping between passes. See the pinned sources in SOURCES.md.

use super::{clone_rgb_image, try_filled_vec, try_vec_with_capacity};
use candle::Result;
use image::RgbImage;

struct Axis {
    rows: Vec<(usize, Vec<i64>)>,
    precision: u32,
}

fn axis(input: usize, output: usize) -> Result<Axis> {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.) * 2.;
    let invscale = if scale >= 1. { 1. / scale } else { 1. };
    let mut rows = try_vec_with_capacity(output, "bicubic coefficient rows")?;
    let mut maximum = 0f64;
    for index in 0..output {
        let center = scale * (index as f64 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let mut weights = try_vec_with_capacity(end - start, "bicubic coefficients")?;
        for source in start..end {
            let x = ((source as f64 - center + 0.5) * invscale).abs();
            weights.push(if x < 1. {
                ((1.5 * x - 2.5) * x) * x + 1.
            } else if x < 2. {
                ((-0.5 * x + 2.5) * x - 4.) * x + 2.
            } else {
                0.
            });
        }
        let total: f64 = weights.iter().sum();
        if total == 0. || !total.is_finite() {
            candle::bail!("invalid bicubic coefficient normalization")
        }
        for weight in &mut weights {
            *weight /= total;
            maximum = maximum.max(*weight);
        }
        rows.push((start, weights));
    }
    let mut precision = 0;
    while precision < 22 && (0.5 + maximum * f64::from(1u32 << (precision + 1))) < 32768. {
        precision += 1;
    }
    if precision == 0 {
        candle::bail!("invalid bicubic fixed-point precision")
    }
    let mut quantized = try_vec_with_capacity(output, "bicubic fixed-point rows")?;
    for (start, weights) in rows {
        let mut values = try_vec_with_capacity(weights.len(), "bicubic fixed-point coefficients")?;
        for weight in weights {
            values.push((weight * f64::from(1u32 << precision)).round() as i64);
        }
        quantized.push((start, values));
    }
    Ok(Axis {
        rows: quantized,
        precision,
    })
}

/// Explicit CPU uint8 bicubic policy used by the pinned d1 processor.
pub fn resize_bicubic_antialias(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    if iw == 0 || ih == 0 || width == 0 || height == 0 {
        candle::bail!("bicubic resize dimensions must be positive")
    }
    let out_width = u32::try_from(width).map_err(candle::Error::wrap)?;
    let out_height = u32::try_from(height).map_err(candle::Error::wrap)?;
    if iw == width && ih == height {
        return clone_rgb_image(image);
    }
    let size = |w: usize, h: usize| {
        w.checked_mul(h)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| candle::Error::Msg("bicubic output size overflow".into()))
    };
    let horizontal = axis(iw, width)?;
    let vertical = axis(ih, height)?;
    let mut temp = try_filled_vec(size(width, ih)?, 0u8, "bicubic horizontal buffer")?;
    for y in 0..ih {
        for (x, (start, weights)) in horizontal.rows.iter().enumerate() {
            for channel in 0..3 {
                let mut sum = 1i64 << (horizontal.precision - 1);
                for (i, weight) in weights.iter().enumerate() {
                    sum += i64::from(image.as_raw()[(y * iw + start + i) * 3 + channel]) * weight;
                }
                temp[(y * width + x) * 3 + channel] =
                    (sum >> horizontal.precision).clamp(0, 255) as u8;
            }
        }
    }
    let mut output = try_filled_vec(size(width, height)?, 0u8, "bicubic output buffer")?;
    for (y, (start, weights)) in vertical.rows.iter().enumerate() {
        for x in 0..width {
            for channel in 0..3 {
                let mut sum = 1i64 << (vertical.precision - 1);
                for (i, weight) in weights.iter().enumerate() {
                    sum += i64::from(temp[((start + i) * width + x) * 3 + channel]) * weight;
                }
                output[(y * width + x) * 3 + channel] =
                    (sum >> vertical.precision).clamp(0, 255) as u8;
            }
        }
    }
    RgbImage::from_raw(out_width, out_height, output)
        .ok_or_else(|| candle::Error::Msg("invalid bicubic image storage".into()))
}
