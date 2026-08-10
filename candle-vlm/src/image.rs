//! Small image primitives used by the LFM2-VL processor.
//!
//! The official processor gives TorchVision the `uint8` tensor produced from a
//! PIL RGB image. TorchVision performs antialiased interpolation through an F32
//! intermediate and rounds back to bytes. Keeping that operation here, before
//! normalization, preserves the pinned oracle's rounding behavior and avoids
//! an image resizing convenience function whose backend may differ.

use candle::Result;
use image::{DynamicImage, RgbImage};

#[derive(Clone, Debug)]
struct ResizeWeights {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ResizeAxis {
    outputs: Vec<ResizeWeights>,
}

/// Convert any supported image variant to the RGB byte representation used by
/// the reference processor.  The `image` crate performs the channel-aware
/// grayscale and alpha conversion here.
pub fn to_rgb8(image: &DynamicImage) -> RgbImage {
    image.to_rgb8()
}

/// Resize an RGB byte image with the separable TorchVision bilinear antialias
/// kernel used by the pinned LFM2-VL processor.
pub fn resize_bilinear_antialias(
    image: &RgbImage,
    output_width: usize,
    output_height: usize,
) -> Result<RgbImage> {
    if output_width == 0 || output_height == 0 {
        candle::bail!("image resize dimensions must be positive")
    }
    let input_width = usize::try_from(image.width())
        .map_err(|_| candle::Error::Msg("image width does not fit usize".into()))?;
    let input_height = usize::try_from(image.height())
        .map_err(|_| candle::Error::Msg("image height does not fit usize".into()))?;
    if input_width == 0 || input_height == 0 {
        candle::bail!("image dimensions must be positive")
    }
    let width = u32::try_from(output_width)
        .map_err(|_| candle::Error::Msg("resized image width exceeds u32".into()))?;
    let height = u32::try_from(output_height)
        .map_err(|_| candle::Error::Msg("resized image height exceeds u32".into()))?;
    if input_width == output_width && input_height == output_height {
        return clone_rgb_image(image);
    }

    let horizontal = resize_axis(input_width, output_width)?;
    let vertical = resize_axis(input_height, output_height)?;

    let horizontal_size = input_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| candle::Error::Msg("image resize output size overflow".into()))?;
    let mut horizontal_output =
        try_filled_vec(horizontal_size, 0.0f32, "image resize horizontal buffer")?;
    let input = image.as_raw();
    let expected_input = input_height
        .checked_mul(input_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| candle::Error::Msg("image input size overflow".into()))?;
    if input.len() != expected_input {
        candle::bail!("RGB image storage has an unexpected length")
    }

    for row in 0..input_height {
        for col in 0..output_width {
            let weights = &horizontal.outputs[col];
            for channel in 0..3 {
                let mut value = 0.0f32;
                for (source_col, weight) in weights.indices.iter().zip(&weights.weights) {
                    let source_index = (row * input_width + source_col) * 3 + channel;
                    value += f32::from(input[source_index]) * weight;
                }
                let output_index = (row * output_width + col) * 3 + channel;
                horizontal_output[output_index] = value;
            }
        }
    }

    let output_size = output_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| candle::Error::Msg("image resize output size overflow".into()))?;
    let mut output = try_filled_vec(output_size, 0u8, "image resize output buffer")?;
    for row in 0..output_height {
        for col in 0..output_width {
            let weights = &vertical.outputs[row];
            for channel in 0..3 {
                let mut value = 0.0f32;
                let mut shadow = 0.0f64;
                for (source_row, weight) in weights.indices.iter().zip(&weights.weights) {
                    let source_index = (*source_row * output_width + col) * 3 + channel;
                    let source = horizontal_output[source_index];
                    shadow += f64::from(source) * f64::from(*weight);
                    // The pinned Torch 2.8 CPU path contracts short vertical
                    // support windows and uses scalar accumulation for longer
                    // dynamic windows. Spell that split out so debug and
                    // release builds produce the same bytes.
                    if weights.indices.len() <= 4 {
                        value = source.mul_add(*weight, value);
                    } else {
                        value += source * weight;
                    }
                }
                let output_index = (row * output_width + col) * 3 + channel;
                output[output_index] = round_ties_even_to_u8(value, shadow)?;
            }
        }
    }

    RgbImage::from_raw(width, height, output)
        .ok_or_else(|| candle::Error::Msg("failed to construct resized RGB image".into()))
}

fn resize_axis(input: usize, output: usize) -> Result<ResizeAxis> {
    let mut outputs = try_vec_with_capacity(output, "image resize weight table")?;
    for index in 0..output {
        let (indices, weights) = floating_resize_weights(input, output, index)?;
        outputs.push(ResizeWeights { indices, weights });
    }
    Ok(ResizeAxis { outputs })
}

fn floating_resize_weights(
    input: usize,
    output: usize,
    index: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if input == 0 || output == 0 || index >= output {
        candle::bail!("invalid image resize dimensions")
    }
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    if !scale.is_finite() || !support.is_finite() {
        candle::bail!("image resize scale is not finite")
    }
    let max_length = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| candle::Error::Msg("image resize support overflow".into()))?;
    let max_length_i64 = i64::try_from(max_length)
        .map_err(|_| candle::Error::Msg("image resize support is too large".into()))?;
    let input_i64 = i64::try_from(input)
        .map_err(|_| candle::Error::Msg("image resize dimension is too large".into()))?;
    let center = scale * (index as f32 + 0.5);
    let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let start = ((center - support + 0.5) as i64).max(0);
    let end = ((center + support + 0.5) as i64).min(input_i64);
    let length_i64 = end
        .checked_sub(start)
        .ok_or_else(|| candle::Error::Msg("image resize range overflow".into()))?
        .clamp(0, max_length_i64);
    let length = usize::try_from(length_i64)
        .map_err(|_| candle::Error::Msg("image resize range is too large".into()))?;
    let mut indices = try_vec_with_capacity(length, "image resize source indices")?;
    let mut weights = try_vec_with_capacity(length, "image resize weights")?;
    let mut total = 0.0f32;
    for offset in 0..length {
        let source = usize::try_from(start)
            .ok()
            .and_then(|value| value.checked_add(offset))
            .ok_or_else(|| candle::Error::Msg("image resize source index overflow".into()))?;
        let argument = (source as f32 - center + 0.5) * inv_scale;
        let weight = (1.0 - argument.abs()).max(0.0);
        indices.push(source);
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        candle::bail!("image resize weights have zero or invalid normalization")
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok((indices, weights))
}

fn try_vec_with_capacity<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to allocate {label} ({capacity} elements): {err}"
        ))
    })?;
    Ok(values)
}

fn try_filled_vec<T: Clone>(length: usize, value: T, label: &str) -> Result<Vec<T>> {
    let mut values = try_vec_with_capacity(length, label)?;
    values.resize(length, value);
    Ok(values)
}

fn clone_rgb_image(image: &RgbImage) -> Result<RgbImage> {
    let input = image.as_raw();
    let mut output = try_vec_with_capacity(input.len(), "image resize identity buffer")?;
    output.extend_from_slice(input);
    RgbImage::from_raw(image.width(), image.height(), output)
        .ok_or_else(|| candle::Error::Msg("failed to clone RGB image".into()))
}

fn round_ties_even_to_u8(value: f32, shadow: f64) -> Result<u8> {
    if !value.is_finite() {
        candle::bail!("image resize produced a non-finite value")
    }
    let mut adjusted = value;
    if value.is_sign_positive() && value.fract() == 0.5 {
        let next_up = f32::from_bits(value.to_bits() + 1);
        let ulp = f64::from(next_up) - f64::from(value);
        let residual = shadow - f64::from(value);
        // TensorIterator can retain one more bit than the scalar F32 sum at a
        // half boundary. Only move a representable half tie when the shadow
        // sum differs by at least one complete output ULP.
        if residual >= ulp {
            adjusted = next_up;
        } else if residual <= -ulp {
            adjusted = f32::from_bits(value.to_bits() - 1);
        }
    }
    Ok(adjusted.round_ties_even().clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn resize_matches_small_uint8_antialias_examples() -> Result<()> {
        let image = RgbImage::from_fn(4, 1, |x, _| Rgb([x as u8 * 10, 0, 0]));
        let down = resize_bilinear_antialias(&image, 2, 1)?;
        assert_eq!(down.as_raw(), &[7, 0, 0, 23, 0, 0]);
        let up = resize_bilinear_antialias(&image, 6, 1)?;
        assert_eq!(
            up.as_raw(),
            &[0, 0, 0, 5, 0, 0, 12, 0, 0, 18, 0, 0, 25, 0, 0, 30, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn resize_rejects_unrepresentable_output_before_allocation() {
        let image = RgbImage::from_pixel(1, 1, Rgb([1, 2, 3]));
        if let Ok(oversized) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(resize_bilinear_antialias(&image, oversized, 1).is_err());
        }
    }

    #[test]
    fn resize_matches_torchvision_half_boundaries() -> Result<()> {
        let image = RgbImage::from_fn(48, 32, |col, row| {
            Rgb([
                (79 + row * 17 + col * 3) as u8,
                (79 + row * 5 + col * 19 + 31) as u8,
                (79 + row * 23 + col * 7 + 67) as u8,
            ])
        });
        let canvas = resize_bilinear_antialias(&image, 16, 16)?;
        assert_eq!(canvas.get_pixel(10, 2)[0], 153);
        assert_eq!(canvas.get_pixel(2, 3)[0], 210);
        assert_eq!(canvas.get_pixel(14, 5)[0], 130);
        assert_eq!(canvas.get_pixel(2, 13)[0], 38);

        let thumbnail = resize_bilinear_antialias(&image, 16, 12)?;
        assert_eq!(thumbnail.get_pixel(3, 7)[0], 185);
        assert_eq!(thumbnail.get_pixel(5, 10)[0], 82);
        assert_eq!(thumbnail.get_pixel(13, 10)[0], 154);
        Ok(())
    }

    #[test]
    fn resize_matches_odd_fixture_in_every_channel() -> Result<()> {
        let image = RgbImage::from_fn(7, 5, |col, row| {
            Rgb([
                (53 + row * 17 + col * 3) as u8,
                (53 + row * 5 + col * 19 + 31) as u8,
                (53 + row * 23 + col * 7 + 67) as u8,
            ])
        });
        let resized = resize_bilinear_antialias(&image, 8, 4)?;
        let expected = [
            57, 85, 126, 60, 101, 131, 62, 117, 138, 65, 134, 144, 68, 151, 150, 70, 167, 156, 73,
            184, 162, 75, 199, 168, 77, 91, 153, 80, 107, 158, 82, 123, 164, 85, 140, 171, 87, 156,
            177, 90, 173, 183, 93, 190, 189, 95, 205, 195, 97, 97, 179, 99, 112, 185, 102, 129,
            191, 105, 146, 197, 107, 162, 203, 110, 179, 210, 112, 195, 216, 115, 211, 221, 117,
            103, 206, 119, 118, 212, 122, 135, 218, 124, 151, 224, 127, 168, 230, 130, 185, 236,
            132, 201, 243, 135, 217, 248,
        ];
        assert_eq!(resized.as_raw(), &expected);
        Ok(())
    }

    #[test]
    fn grayscale_and_alpha_are_converted_to_rgb() {
        let gray = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(1, 1, image::Luma([7])));
        assert_eq!(to_rgb8(&gray).as_raw(), &[7, 7, 7]);
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([10, 20, 30, 128]),
        ));
        assert_eq!(to_rgb8(&rgba).as_raw().len(), 3);
    }
}
