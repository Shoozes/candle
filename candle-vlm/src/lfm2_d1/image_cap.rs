// Numerical contract: Pillow 11.3 Resample.c (MIT-CMU).
// Copyright 1997-2011 Secret Labs AB; 1995-2011 Fredrik Lundh and contributors;
// 2010 Jeffrey A. Clark and contributors. See docs/lfm2-vl/licenses/PILLOW_LICENSE.txt.
use candle::Result;
use image::{DynamicImage, RgbImage};

// The d1 runner caps pixels before the ordinary LFM2-VL processor.
pub(super) const MAX_PIXELS: u64 = 1024 * 1024;

fn cap_dimensions(width: u32, height: u32, maximum: u64) -> Result<(u32, u32)> {
    if width == 0 || height == 0 || maximum == 0 {
        candle::bail!("d1 image dimensions and pixel limit must be positive")
    }
    let pixels = |width: u32, height: u32| {
        u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| candle::Error::Msg("d1 image area overflow".into()))
    };
    let area = pixels(width, height)?;
    if area <= maximum {
        return Ok((width, height));
    }
    let scale = (maximum as f64 / area as f64).sqrt();
    let mut output_width = (f64::from(width) * scale).floor().max(1.) as u32;
    let mut output_height = (f64::from(height) * scale).floor().max(1.) as u32;
    if pixels(output_width, output_height)? > maximum {
        // Preserve valid legacy sizes; trim the longer axis only when needed.
        let (long, short) = if output_width >= output_height {
            (&mut output_width, output_height)
        } else {
            (&mut output_height, output_width)
        };
        *long = maximum
            .checked_div(u64::from(short))
            .and_then(|bound| u32::try_from(bound).ok())
            .ok_or_else(|| candle::Error::Msg("d1 capped dimension exceeds bounds".into()))?;
    }
    if output_width == 0
        || output_height == 0
        || output_width > width
        || output_height > height
        || pixels(output_width, output_height)? > maximum
    {
        candle::bail!("invalid d1 capped dimensions for {width}x{height} at pixel limit {maximum}")
    }
    Ok((output_width, output_height))
}

pub(super) fn cap_pixels(image: &DynamicImage, maximum: u64) -> Result<DynamicImage> {
    let (width, height) = cap_dimensions(image.width(), image.height(), maximum)?;
    let rgb = image.to_rgb8();
    if (width, height) == (image.width(), image.height()) {
        return Ok(DynamicImage::ImageRgb8(rgb));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([
                ((x * 37 + y * 19) % 256) as u8,
                ((x * 17 + y * 71) % 256) as u8,
                ((x * 91 + y * 13) % 256) as u8,
            ])
        })
    }

    fn assert_cap(
        width: u32,
        height: u32,
        maximum: u64,
        dimensions: (u32, u32),
        pixels: &[u8],
    ) -> Result<()> {
        let output =
            cap_pixels(&DynamicImage::ImageRgb8(pattern(width, height)), maximum)?.into_rgb8();
        assert_eq!(
            output.dimensions(),
            dimensions,
            "{width}x{height}, cap {maximum}"
        );
        assert_eq!(output.as_raw(), pixels);
        Ok(())
    }

    #[test]
    fn d1_pixel_cap_matches_pillow_bicubic_bytes() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Case {
            width: u32,
            height: u32,
            cap: u64,
            output_width: u32,
            output_height: u32,
            rgb: Vec<u8>,
        }
        #[derive(serde::Deserialize)]
        struct Suite {
            cases: Vec<Case>,
        }
        let suite: Suite = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/lfm2_d1_tiny/bicubic.json"
        ))
        .map_err(candle::Error::wrap)?;
        for case in suite.cases {
            let image = pattern(case.width, case.height);
            let dimensions = (case.output_width, case.output_height);
            assert_eq!(
                resize(&image, dimensions.0, dimensions.1)?.as_raw(),
                &case.rgb
            );
            if u64::from(dimensions.0) * u64::from(dimensions.1) <= case.cap {
                assert_cap(case.width, case.height, case.cap, dimensions, &case.rgb)?;
            } else {
                // Preserve the frozen resampler golden, not its invalid legacy cap.
                assert_eq!(
                    (case.width, case.height, case.cap, dimensions),
                    (9, 1, 6, (7, 1))
                );
            }
        }
        Ok(())
    }

    // Pillow 11.3.0 RGB BICUBIC at the corrected sizes, generated independently.
    #[test]
    fn d1_pixel_cap_wide_images_fit() -> Result<()> {
        assert_cap(
            16,
            1,
            4,
            (4, 1),
            &[67, 27, 85, 135, 92, 117, 101, 163, 140, 122, 228, 159],
        )?;
        assert_cap(
            9,
            1,
            6,
            (6, 1),
            &[
                11, 5, 28, 63, 29, 144, 121, 56, 55, 186, 80, 161, 147, 107, 72, 17, 131, 188,
            ],
        )?;
        assert_cap(16, 1, 1, (1, 1), &[106, 128, 124])
    }

    #[test]
    fn d1_pixel_cap_tall_images_fit() -> Result<()> {
        assert_cap(
            1,
            16,
            4,
            (1, 4),
            &[30, 95, 21, 103, 140, 70, 190, 121, 125, 137, 124, 174],
        )?;
        assert_cap(1, 16, 1, (1, 1), &[118, 120, 98])
    }

    #[test]
    fn d1_pixel_cap_normal_resizes_match_pillow() -> Result<()> {
        assert_cap(
            8,
            4,
            12,
            (4, 2),
            &[
                31, 50, 70, 102, 88, 108, 186, 130, 134, 121, 147, 102, 66, 186, 93, 140, 155, 131,
                196, 130, 157, 57, 109, 125,
            ],
        )?;
        assert_cap(
            4,
            8,
            12,
            (2, 4),
            &[
                32, 51, 70, 99, 87, 109, 69, 172, 95, 136, 149, 135, 108, 99, 120, 175, 120, 152,
                145, 144, 118, 212, 119, 77,
            ],
        )
    }

    #[test]
    fn d1_pixel_cap_preserves_images_within_limit() -> Result<()> {
        let image = pattern(4, 3);
        for maximum in [12, 24, u64::MAX] {
            let output = cap_pixels(&DynamicImage::ImageRgb8(image.clone()), maximum)?.into_rgb8();
            assert_eq!(output, image);
        }
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            2,
            2,
            image::Rgba([32, 64, 96, 0]),
        ));
        assert_eq!(
            cap_pixels(&rgba, 4)?.into_rgb8(),
            RgbImage::from_pixel(2, 2, image::Rgb([32, 64, 96]))
        );
        Ok(())
    }

    #[test]
    fn d1_pixel_cap_dimensions_cover_integer_limits() -> Result<()> {
        let largest = u32::MAX;
        let area = u64::from(largest) * u64::from(largest);
        for (width, height, maximum, expected) in [
            (2_097_152, 1, MAX_PIXELS, (1_048_576, 1)),
            (1, 2_097_152, MAX_PIXELS, (1, 1_048_576)),
            (largest, 1, 1, (1, 1)),
            (1, largest, 1, (1, 1)),
            (largest, largest, u64::MAX, (largest, largest)),
            (largest, largest, area - 1, (largest - 1, largest)),
        ] {
            assert_eq!(
                cap_dimensions(width, height, maximum)?,
                expected,
                "{width}x{height}, cap {maximum}"
            );
        }
        Ok(())
    }

    #[test]
    fn d1_pixel_cap_rejects_invalid_dimensions_and_limit() {
        for (width, height, maximum) in [(0, 1, 1), (1, 0, 1), (0, 0, 1), (1, 1, 0)] {
            assert!(cap_dimensions(width, height, maximum).is_err());
        }
        assert!(cap_pixels(&DynamicImage::ImageRgb8(pattern(1, 1)), 0).is_err());
    }

    #[test]
    fn d1_pixel_cap_dimension_matrix_stays_bounded() -> Result<()> {
        let dimensions = [1, 2, 3, 16, 255, 4096, 2_097_152, u32::MAX / 2, u32::MAX];
        for width in dimensions {
            for height in dimensions {
                for maximum in [1, 2, 4, 16, 1024, u64::from(u32::MAX), u64::MAX] {
                    let output = cap_dimensions(width, height, maximum)?;
                    assert!(
                        output.0 > 0 && output.1 > 0 && output.0 <= width && output.1 <= height
                    );
                    assert!(
                        u64::from(output.0)
                            .checked_mul(u64::from(output.1))
                            .is_some_and(|area| area <= maximum),
                        "{width}x{height}, cap {maximum}: {output:?}"
                    );
                }
            }
        }
        Ok(())
    }
}
