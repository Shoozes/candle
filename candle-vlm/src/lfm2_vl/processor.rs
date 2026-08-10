//! RGB image processing, NaFlex tiling, and packed patch construction.

use super::config::Lfm2VlProcessorConfig;
use super::types::{CropKind, CropMeta, ImageMeta, ProcessedVisionBatch};
use crate::image::{resize_bilinear_antialias, to_rgb8};
use candle::{Device, Result, Tensor};
use image::{imageops, DynamicImage, RgbImage};

const TARGET_RATIO_ORDER: &[(usize, usize)] = &[
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
];

#[derive(Debug)]
struct CropWork {
    image_index: usize,
    kind: CropKind,
    patch_rows: usize,
    patch_cols: usize,
    projected_tokens: usize,
    patches: Vec<f32>,
}

#[derive(Debug)]
struct ImageWork {
    crops: Vec<CropWork>,
    rows: usize,
    cols: usize,
    resized_width: usize,
    resized_height: usize,
}

/// Processor for the raw RGB-to-packed-tensor part of LFM2.5-VL.
#[derive(Clone, Debug)]
pub struct Lfm2VlProcessor {
    config: Lfm2VlProcessorConfig,
}

impl Lfm2VlProcessor {
    pub fn new(config: Lfm2VlProcessorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Self::new(Lfm2VlProcessorConfig::from_json(json)?)
    }

    pub fn from_config(config: &Lfm2VlProcessorConfig) -> Result<Self> {
        Self::new(config.clone())
    }

    pub fn config(&self) -> &Lfm2VlProcessorConfig {
        &self.config
    }

    /// Process images in input order into the packed format consumed by the
    /// Phase 2 SigLIP2 model.  The processor currently owns the batch layout;
    /// model text batching remains a later phase.
    pub fn process(
        &self,
        images: &[DynamicImage],
        device: &Device,
    ) -> Result<ProcessedVisionBatch> {
        if images.is_empty() {
            candle::bail!("LFM2-VL processor requires at least one image")
        }
        if !self.config.do_pad {
            candle::bail!("LFM2-VL processor requires do_pad=true for packed output")
        }
        let mut work = try_vec_with_capacity(images.len(), "LFM2-VL image work list")?;
        for (image_index, image) in images.iter().enumerate() {
            work.push(self.process_image(image, image_index)?);
        }

        let crop_count = work.iter().try_fold(0usize, |count, image| {
            count
                .checked_add(image.crops.len())
                .ok_or_else(|| candle::Error::Msg("LFM2-VL crop count overflow".into()))
        })?;
        if crop_count == 0 {
            candle::bail!("LFM2-VL processor produced no crops")
        }
        let max_patches = self.config.effective_max_num_patches()?;
        let patch_dimension = self
            .config
            .encoder_patch_size
            .checked_mul(self.config.encoder_patch_size)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch dimension overflow".into()))?;

        let values_capacity = crop_count
            .checked_mul(max_patches)
            .and_then(|value| value.checked_mul(patch_dimension))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL packed tensor size overflow".into()))?;
        let mut pixel_values =
            try_vec_with_capacity(values_capacity, "LFM2-VL packed pixel tensor")?;
        let mask_capacity = crop_count
            .checked_mul(max_patches)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL mask size overflow".into()))?;
        let mut pixel_attention_mask =
            try_vec_with_capacity(mask_capacity, "LFM2-VL pixel attention mask")?;
        let shape_capacity = crop_count
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL shape metadata overflow".into()))?;
        let mut spatial_shapes =
            try_vec_with_capacity(shape_capacity, "LFM2-VL spatial shape metadata")?;
        let mut crops = try_vec_with_capacity(crop_count, "LFM2-VL crop metadata")?;
        let mut metadata_images = try_vec_with_capacity(work.len(), "LFM2-VL image metadata")?;
        let mut crop_offset = 0usize;

        for image in work {
            let image_crop_start = crop_offset;
            for (crop_index, crop) in image.crops.into_iter().enumerate() {
                let valid_patches = crop
                    .patch_rows
                    .checked_mul(crop.patch_cols)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL patch count overflow".into()))?;
                if valid_patches > max_patches {
                    candle::bail!(
                        "LFM2-VL crop has {valid_patches} patches, exceeding packed maximum {max_patches}"
                    )
                }
                let expected_values =
                    valid_patches.checked_mul(patch_dimension).ok_or_else(|| {
                        candle::Error::Msg("LFM2-VL crop tensor size overflow".into())
                    })?;
                if crop.patches.len() != expected_values {
                    candle::bail!("LFM2-VL crop patch data has an unexpected length")
                }
                pixel_values.extend_from_slice(&crop.patches);
                let padding = max_patches - valid_patches;
                let padding_values = padding
                    .checked_mul(patch_dimension)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL padding size overflow".into()))?;
                pixel_values.extend(std::iter::repeat_n(0.0f32, padding_values));
                pixel_attention_mask.extend(std::iter::repeat_n(1i32, valid_patches));
                pixel_attention_mask.extend(std::iter::repeat_n(0i32, padding));
                spatial_shapes.push(i64::try_from(crop.patch_rows).map_err(|_| {
                    candle::Error::Msg("LFM2-VL patch row does not fit i64".into())
                })?);
                spatial_shapes.push(i64::try_from(crop.patch_cols).map_err(|_| {
                    candle::Error::Msg("LFM2-VL patch column does not fit i64".into())
                })?);
                crops.push(CropMeta {
                    image_index: crop.image_index,
                    crop_index,
                    kind: crop.kind,
                    patch_rows: crop.patch_rows,
                    patch_cols: crop.patch_cols,
                    projected_tokens: crop.projected_tokens,
                });
                crop_offset = crop_offset
                    .checked_add(1)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL crop range overflow".into()))?;
            }
            metadata_images.push(ImageMeta {
                crop_range: image_crop_start..crop_offset,
                rows: image.rows,
                cols: image.cols,
                resized_width: image.resized_width,
                resized_height: image.resized_height,
            });
        }

        let pixel_values = Tensor::from_vec(
            pixel_values,
            (crop_count, max_patches, patch_dimension),
            device,
        )?;
        let pixel_attention_mask =
            Tensor::from_vec(pixel_attention_mask, (crop_count, max_patches), device)?;
        let spatial_shapes = Tensor::from_vec(spatial_shapes, (crop_count, 2), device)?;
        Ok(ProcessedVisionBatch {
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            crops,
            images: metadata_images,
        })
    }

    /// Return the Python 3.10.12 reference order for the supported 2..10
    /// target-ratio range.  For other ranges the same area ordering is made
    /// deterministic with this captured order as the tie key.
    pub fn target_ratios(min_tiles: usize, max_tiles: usize) -> Result<Vec<(usize, usize)>> {
        if min_tiles == 0 || max_tiles < min_tiles || max_tiles > 10 {
            candle::bail!("LFM2-VL tile range is invalid")
        }
        let ratio_capacity = max_tiles
            .checked_mul(max_tiles)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tile ratio count overflow".into()))?;
        let mut ratios = try_vec_with_capacity(ratio_capacity, "LFM2-VL tile ratio table")?;
        for rows in 1..=max_tiles {
            for cols in 1..=max_tiles {
                let area = rows
                    .checked_mul(cols)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL tile ratio overflow".into()))?;
                if area >= min_tiles && area <= max_tiles {
                    ratios.push((cols, rows));
                }
            }
        }
        ratios.sort_by_key(|&(cols, rows)| {
            let area = cols * rows;
            let rank = TARGET_RATIO_ORDER
                .iter()
                .position(|&ratio| ratio == (cols, rows))
                .unwrap_or(usize::MAX);
            (area, rank, cols, rows)
        });
        Ok(ratios)
    }

    pub fn round_by_factor(number: usize, factor: usize) -> Result<usize> {
        if factor == 0 {
            candle::bail!("LFM2-VL rounding factor must be positive")
        }
        let quotient = number / factor;
        let remainder = number % factor;
        let twice_remainder = remainder
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL rounding overflow".into()))?;
        let rounded_quotient = if twice_remainder < factor {
            quotient
        } else if twice_remainder > factor || quotient % 2 == 1 {
            quotient
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL rounded dimension overflow".into()))?
        } else {
            quotient
        };
        rounded_quotient
            .checked_mul(factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL rounded dimension overflow".into()))
    }

    pub fn smart_resize(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        if width == 0 || height == 0 {
            candle::bail!("LFM2-VL image dimensions must be positive")
        }
        let total_factor = self.config.total_factor()?;
        let factor_squared = total_factor
            .checked_mul(total_factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL smart resize factor overflow".into()))?;
        let min_pixels = self
            .config
            .min_image_tokens
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL minimum pixel limit overflow".into()))?;
        let max_pixels = self
            .config
            .max_image_tokens
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL maximum pixel limit overflow".into()))?;
        let mut height_bar = Self::round_by_factor(height, total_factor)?.max(total_factor);
        let mut width_bar = Self::round_by_factor(width, total_factor)?.max(total_factor);
        let area = height
            .checked_mul(width)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image area overflow".into()))?;
        let rounded_area = height_bar
            .checked_mul(width_bar)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL rounded image area overflow".into()))?;
        if rounded_area > max_pixels {
            let beta = ((area as f64) / (max_pixels as f64)).sqrt();
            height_bar = floor_multiple(height as f64 / beta, total_factor)?;
            width_bar = floor_multiple(width as f64 / beta, total_factor)?;
        } else if rounded_area < min_pixels {
            let beta = ((min_pixels as f64) / (area as f64)).sqrt();
            height_bar = ceil_multiple(height as f64 * beta, total_factor)?;
            width_bar = ceil_multiple(width as f64 * beta, total_factor)?;
        }
        Ok((width_bar.max(total_factor), height_bar.max(total_factor)))
    }

    pub fn is_image_too_large(&self, width: usize, height: usize) -> Result<bool> {
        if width == 0 || height == 0 {
            candle::bail!("LFM2-VL image dimensions must be positive")
        }
        let total_factor = self.config.total_factor()?;
        let h_bar =
            Self::round_by_factor(height, total_factor)?.max(self.config.encoder_patch_size);
        let w_bar = Self::round_by_factor(width, total_factor)?.max(self.config.encoder_patch_size);
        let patch_squared = self
            .config
            .encoder_patch_size
            .checked_mul(self.config.encoder_patch_size)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch area overflow".into()))?;
        let factor_squared = self
            .config
            .downsample_factor
            .checked_mul(self.config.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL factor area overflow".into()))?;
        let max_tokens = self
            .config
            .max_image_tokens
            .checked_mul(patch_squared)
            .and_then(|value| value.checked_mul(factor_squared))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL large-image threshold overflow".into()))?;
        let rounded_area = h_bar
            .checked_mul(w_bar)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL rounded image area overflow".into()))?;
        Ok((rounded_area as f64) > (max_tokens as f64) * self.config.max_pixels_tolerance)
    }

    fn process_image(&self, image: &DynamicImage, image_index: usize) -> Result<ImageWork> {
        let rgb = to_rgb8(image);
        let width = usize::try_from(rgb.width())
            .map_err(|_| candle::Error::Msg("LFM2-VL image width does not fit usize".into()))?;
        let height = usize::try_from(rgb.height())
            .map_err(|_| candle::Error::Msg("LFM2-VL image height does not fit usize".into()))?;
        if width == 0 || height == 0 {
            candle::bail!("LFM2-VL image dimensions must be positive")
        }
        let split = self.config.do_image_splitting && self.is_image_too_large(width, height)?;
        if split {
            self.process_tiled(&rgb, image_index)
        } else {
            let (resized_width, resized_height) = if self.config.do_resize {
                self.smart_resize(width, height)?
            } else {
                (width, height)
            };
            let resized = if resized_width == width && resized_height == height {
                rgb
            } else {
                resize_bilinear_antialias(&rgb, resized_width, resized_height)?
            };
            let crop = self.make_crop(&resized, image_index, CropKind::Whole)?;
            Ok(ImageWork {
                crops: vec![crop],
                rows: 1,
                cols: 1,
                resized_width,
                resized_height,
            })
        }
    }

    fn process_tiled(&self, image: &RgbImage, image_index: usize) -> Result<ImageWork> {
        let width = usize::try_from(image.width())
            .map_err(|_| candle::Error::Msg("LFM2-VL image width does not fit usize".into()))?;
        let height = usize::try_from(image.height())
            .map_err(|_| candle::Error::Msg("LFM2-VL image height does not fit usize".into()))?;
        let (metadata_width, metadata_height) = if self.config.do_resize {
            self.smart_resize(width, height)?
        } else {
            (width, height)
        };
        let (cols, rows) = self.closest_tile_grid(width, height)?;
        let resized_width = self
            .config
            .tile_size
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tiled width overflow".into()))?;
        let resized_height = self
            .config
            .tile_size
            .checked_mul(rows)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tiled height overflow".into()))?;
        let resized = resize_bilinear_antialias(image, resized_width, resized_height)?;
        let tile_width = self.config.tile_size;
        let tile_height = self.config.tile_size;
        let tile_count = rows
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tile count overflow".into()))?;
        let crop_capacity = tile_count
            .checked_add(usize::from(self.config.use_thumbnail && tile_count > 1))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL crop count overflow".into()))?;
        let mut crops = try_vec_with_capacity(crop_capacity, "LFM2-VL tiled crop list")?;
        for row in 0..rows {
            for col in 0..cols {
                let x = col
                    .checked_mul(tile_width)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL tile x overflow".into()))?;
                let y = row
                    .checked_mul(tile_height)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL tile y overflow".into()))?;
                let tile = imageops::crop_imm(
                    &resized,
                    u32::try_from(x)
                        .map_err(|_| candle::Error::Msg("LFM2-VL tile x exceeds u32".into()))?,
                    u32::try_from(y)
                        .map_err(|_| candle::Error::Msg("LFM2-VL tile y exceeds u32".into()))?,
                    u32::try_from(tile_width)
                        .map_err(|_| candle::Error::Msg("LFM2-VL tile width exceeds u32".into()))?,
                    u32::try_from(tile_height).map_err(|_| {
                        candle::Error::Msg("LFM2-VL tile height exceeds u32".into())
                    })?,
                )
                .to_image();
                crops.push(self.make_crop(&tile, image_index, CropKind::Tile { row, col })?);
            }
        }
        if self.config.use_thumbnail && tile_count > 1 {
            let (thumbnail_width, thumbnail_height) = self.smart_resize(width, height)?;
            let thumbnail = if thumbnail_width == width && thumbnail_height == height {
                image.clone()
            } else {
                resize_bilinear_antialias(image, thumbnail_width, thumbnail_height)?
            };
            crops.push(self.make_crop(&thumbnail, image_index, CropKind::Thumbnail)?);
        }
        Ok(ImageWork {
            crops,
            rows,
            cols,
            resized_width: metadata_width,
            resized_height: metadata_height,
        })
    }

    fn closest_tile_grid(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        let aspect = width as f64 / height as f64;
        let source_area = width
            .checked_mul(height)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image area overflow".into()))?;
        let tile_area = self
            .config
            .tile_size
            .checked_mul(self.config.tile_size)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tile area overflow".into()))?;
        let mut best: Option<(f64, usize, usize)> = None;
        for (cols, rows) in Self::target_ratios(self.config.min_tiles, self.config.max_tiles)? {
            let target_aspect = cols as f64 / rows as f64;
            let difference = (aspect - target_aspect).abs();
            let target_area = tile_area
                .checked_mul(cols)
                .and_then(|value| value.checked_mul(rows))
                .ok_or_else(|| candle::Error::Msg("LFM2-VL target tile area overflow".into()))?;
            let should_replace = match best {
                None => true,
                Some((best_difference, _, _)) if difference < best_difference => true,
                Some((best_difference, _, _)) if difference > best_difference => false,
                Some((_, _, _)) => (source_area as f64) > 0.5 * target_area as f64,
            };
            if should_replace {
                best = Some((difference, cols, rows));
            }
        }
        best.map(|(_, cols, rows)| (cols, rows))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL no valid tile grid".into()))
    }

    fn make_crop(&self, image: &RgbImage, image_index: usize, kind: CropKind) -> Result<CropWork> {
        let width = usize::try_from(image.width())
            .map_err(|_| candle::Error::Msg("LFM2-VL crop width does not fit usize".into()))?;
        let height = usize::try_from(image.height())
            .map_err(|_| candle::Error::Msg("LFM2-VL crop height does not fit usize".into()))?;
        let patch = self.config.encoder_patch_size;
        if width == 0 || height == 0 || width % patch != 0 || height % patch != 0 {
            candle::bail!(
                "LFM2-VL crop dimensions [{width}, {height}] must be positive and divisible by patch size {patch}"
            )
        }
        let patch_rows = height / patch;
        let patch_cols = width / patch;
        let projected_tokens = self.config.projected_token_count(patch_rows, patch_cols)?;
        let raw = image.as_raw();
        let expected = height
            .checked_mul(width)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL crop size overflow".into()))?;
        if raw.len() != expected {
            candle::bail!("LFM2-VL RGB crop storage has an unexpected length")
        }
        let pixels = expected;
        let mut chw = try_filled_vec(pixels, 0.0f32, "LFM2-VL normalized crop buffer")?;
        for channel in 0..3 {
            for row in 0..height {
                for col in 0..width {
                    let source = (row * width + col) * 3 + channel;
                    let mut value = f32::from(raw[source]);
                    if self.config.do_rescale {
                        value *= self.config.rescale_factor;
                    }
                    if self.config.do_normalize {
                        value = (value - self.config.image_mean[channel])
                            / self.config.image_std[channel];
                    }
                    chw[channel * height * width + row * width + col] = value;
                }
            }
        }
        let patch_dimension = patch
            .checked_mul(patch)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch dimension overflow".into()))?;
        let patch_count = patch_rows
            .checked_mul(patch_cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch count overflow".into()))?;
        let output_size = patch_count
            .checked_mul(patch_dimension)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch tensor size overflow".into()))?;
        let mut patches = try_vec_with_capacity(output_size, "LFM2-VL patchification buffer")?;
        for patch_row in 0..patch_rows {
            for patch_col in 0..patch_cols {
                for patch_row_offset in 0..patch {
                    for patch_col_offset in 0..patch {
                        for channel in 0..3 {
                            let row = patch_row * patch + patch_row_offset;
                            let col = patch_col * patch + patch_col_offset;
                            patches.push(chw[channel * height * width + row * width + col]);
                        }
                    }
                }
            }
        }
        if patches.len() != output_size {
            candle::bail!("LFM2-VL patchification produced an unexpected length")
        }
        Ok(CropWork {
            image_index,
            kind,
            patch_rows,
            patch_cols,
            projected_tokens,
            patches,
        })
    }
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

fn floor_multiple(value: f64, factor: usize) -> Result<usize> {
    if factor == 0 || !value.is_finite() || value < 0.0 {
        candle::bail!("LFM2-VL resize calculation is not finite")
    }
    let quotient = (value / factor as f64).floor();
    let quotient = usize::try_from(quotient as u128)
        .map_err(|_| candle::Error::Msg("LFM2-VL resize dimension is too large".into()))?;
    quotient
        .checked_mul(factor)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL resize dimension overflow".into()))
}

fn ceil_multiple(value: f64, factor: usize) -> Result<usize> {
    if factor == 0 || !value.is_finite() || value < 0.0 {
        candle::bail!("LFM2-VL resize calculation is not finite")
    }
    let quotient = (value / factor as f64).ceil();
    let quotient = usize::try_from(quotient as u128)
        .map_err(|_| candle::Error::Msg("LFM2-VL resize dimension is too large".into()))?;
    quotient
        .checked_mul(factor)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL resize dimension overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
    use std::collections::HashMap;

    const PROCESSOR_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_processor_tiny/tensors.safetensors");
    const PROCESSOR_METADATA: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_processor_tiny/metadata.json");

    fn tiny_config() -> Lfm2VlProcessorConfig {
        Lfm2VlProcessorConfig {
            do_resize: true,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            do_pad: true,
            downsample_factor: 2,
            encoder_patch_size: 2,
            do_image_splitting: true,
            min_tiles: 2,
            max_tiles: 4,
            use_thumbnail: true,
            tile_size: 8,
            min_image_tokens: 1,
            max_image_tokens: 16,
            max_num_patches: Some(64),
            max_pixels_tolerance: 2.0,
            context_length: Some(256),
        }
    }

    fn image(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x % 251) as u8, (y % 251) as u8, ((x + y) % 251) as u8])
        }))
    }

    #[test]
    fn target_ratio_order_is_captured_for_python_range() -> Result<()> {
        assert_eq!(
            Lfm2VlProcessor::target_ratios(2, 10)?,
            vec![
                (1, 2),
                (2, 1),
                (3, 1),
                (1, 3),
                (2, 2),
                (4, 1),
                (1, 4),
                (5, 1),
                (1, 5),
                (1, 6),
                (6, 1),
                (3, 2),
                (2, 3),
                (7, 1),
                (1, 7),
                (4, 2),
                (2, 4),
                (1, 8),
                (8, 1),
                (1, 9),
                (3, 3),
                (9, 1),
                (2, 5),
                (5, 2),
                (10, 1),
                (1, 10),
            ]
        );
        assert!(Lfm2VlProcessor::target_ratios(1, 11).is_err());
        Ok(())
    }

    #[test]
    fn configured_packed_capacity_overflow_is_controlled() -> Result<()> {
        let mut config = tiny_config();
        config.max_num_patches = Some(usize::MAX);
        let processor = Lfm2VlProcessor::new(config)?;
        assert!(processor.process(&[image(1, 1)], &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    fn round_and_resize_are_factor_aligned() -> Result<()> {
        assert_eq!(Lfm2VlProcessor::round_by_factor(5, 2)?, 4);
        assert_eq!(Lfm2VlProcessor::round_by_factor(7, 2)?, 8);
        assert_eq!(Lfm2VlProcessor::round_by_factor(3, 2)?, 4);
        let processor = Lfm2VlProcessor::new(tiny_config())?;
        let (width, height) = processor.smart_resize(3, 5)?;
        assert_eq!(width % 4, 0);
        assert_eq!(height % 4, 0);
        Ok(())
    }

    #[test]
    fn smart_resize_uses_rounded_area_boundaries() -> Result<()> {
        let mut config = tiny_config();
        config.min_image_tokens = 1;
        config.max_image_tokens = 4;
        config.max_num_patches = None;
        let processor = Lfm2VlProcessor::new(config)?;
        assert_eq!(processor.smart_resize(25, 9)?, (12, 4));
        let mut boundary = tiny_config();
        boundary.max_num_patches = None;
        let boundary_processor = Lfm2VlProcessor::new(boundary)?;
        assert_eq!(boundary_processor.smart_resize(7, 9)?, (8, 8));
        let mut upper_boundary = tiny_config();
        upper_boundary.min_image_tokens = 1;
        upper_boundary.max_image_tokens = 8;
        upper_boundary.max_num_patches = None;
        let upper_processor = Lfm2VlProcessor::new(upper_boundary)?;
        assert_eq!(upper_processor.smart_resize(15, 9)?, (16, 8));
        Ok(())
    }

    #[test]
    fn processor_emits_packed_metadata_and_rgb_conversion() -> Result<()> {
        let processor = Lfm2VlProcessor::new(tiny_config())?;
        let gray = DynamicImage::ImageLuma8(GrayImage::from_pixel(4, 4, Luma([128])));
        let rgba = DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, Rgba([64, 32, 16, 127])));
        let output = processor.process(&[gray, rgba], &Device::Cpu)?;
        assert_eq!(output.pixel_values.dims3()?.0, 2);
        assert_eq!(output.pixel_attention_mask.dims(), [2, 64]);
        assert_eq!(output.spatial_shapes.dims(), [2, 2]);
        assert_eq!(output.crops.len(), 2);
        assert_eq!(output.images[0].crop_range, 0..1);
        assert_eq!(output.images[1].crop_range, 1..2);
        Ok(())
    }

    #[test]
    fn large_input_is_tiled_row_major_with_thumbnail() -> Result<()> {
        let mut config = tiny_config();
        config.max_pixels_tolerance = 0.01;
        let processor = Lfm2VlProcessor::new(config)?;
        let output = processor.process(&[image(32, 16)], &Device::Cpu)?;
        assert!(output.crops.len() > 1);
        assert_eq!(output.images[0].crop_range.start, 0);
        assert_eq!(output.images[0].crop_range.end, output.crops.len());
        assert!(matches!(
            output.crops.last().map(|crop| &crop.kind),
            Some(CropKind::Thumbnail)
        ));
        for (index, crop) in output.crops.iter().enumerate() {
            assert_eq!(crop.crop_index, index);
        }
        Ok(())
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing processor fixture tensor {name}")))
    }

    fn fixture_image(
        tensors: &HashMap<String, Tensor>,
        case_name: &str,
        image_index: usize,
        mode: &str,
    ) -> Result<DynamicImage> {
        let suffix = mode.to_ascii_lowercase();
        let name = if image_index == 0 && case_name != "multiple_images" {
            format!("input.{case_name}.{suffix}_u8")
        } else {
            format!("input.{case_name}.{image_index}.{suffix}_u8")
        };
        let tensor = fixture_tensor(tensors, &name)?;
        let dims = tensor.dims().to_vec();
        let data = tensor.flatten_all()?.to_vec1::<u8>()?;
        let (height, width, channels) = match dims.as_slice() {
            [height, width] => (*height, *width, 1),
            [height, width, channels] => (*height, *width, *channels),
            _ => candle::bail!("processor fixture input has an invalid image shape"),
        };
        let width = u32::try_from(width)
            .map_err(|_| candle::Error::Msg("processor fixture width exceeds u32".into()))?;
        let height = u32::try_from(height)
            .map_err(|_| candle::Error::Msg("processor fixture height exceeds u32".into()))?;
        match channels {
            1 => Ok(DynamicImage::ImageLuma8(
                GrayImage::from_raw(width, height, data)
                    .ok_or_else(|| candle::Error::Msg("invalid grayscale fixture".into()))?,
            )),
            3 => Ok(DynamicImage::ImageRgb8(
                RgbImage::from_raw(width, height, data)
                    .ok_or_else(|| candle::Error::Msg("invalid RGB fixture".into()))?,
            )),
            4 => Ok(DynamicImage::ImageRgba8(
                RgbaImage::from_raw(width, height, data)
                    .ok_or_else(|| candle::Error::Msg("invalid RGBA fixture".into()))?,
            )),
            _ => candle::bail!("processor fixture has unsupported channel count {channels}"),
        }
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let actual_values = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected_values = expected.flatten_all()?.to_vec1::<f32>()?;
        if actual_values.len() != expected_values.len() {
            candle::bail!("{label} length mismatch");
        }
        let mut max_abs = 0.0f32;
        let mut max_index = 0usize;
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for (index, (&actual_value, &expected_value)) in
            actual_values.iter().zip(&expected_values).enumerate()
        {
            let absolute_error = (actual_value - expected_value).abs();
            if absolute_error > max_abs {
                max_abs = absolute_error;
                max_index = index;
            }
            dot += f64::from(actual_value) * f64::from(expected_value);
            actual_norm += f64::from(actual_value) * f64::from(actual_value);
            expected_norm += f64::from(expected_value) * f64::from(expected_value);
        }
        let cosine = if actual_norm == 0.0 || expected_norm == 0.0 {
            1.0
        } else {
            dot / (actual_norm.sqrt() * expected_norm.sqrt())
        };
        eprintln!(
            "{label}: max_abs={max_abs:.9e} index={max_index} actual={:.9e} expected={:.9e} cosine={cosine:.9}",
            actual_values[max_index], expected_values[max_index]
        );
        assert!(
            max_abs <= tolerance,
            "{label} max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.99999, "{label} cosine={cosine} < 0.99999");
        Ok(())
    }

    #[test]
    fn official_processor_fixture_matches_all_image_modes_and_shapes() -> Result<()> {
        let device = Device::Cpu;
        let tensors = candle::safetensors::load_buffer(PROCESSOR_FIXTURE, &device)?;
        let metadata: serde_json::Value = serde_json::from_slice(PROCESSOR_METADATA)
            .map_err(|err| candle::Error::Msg(format!("invalid processor metadata: {err}")))?;
        let processor = Lfm2VlProcessor::new(tiny_config())?;
        let cases = [
            ("square", &["RGB"][..]),
            ("wide", &["RGB"][..]),
            ("tall", &["RGB"][..]),
            ("very_wide", &["RGB"][..]),
            ("very_tall", &["RGB"][..]),
            ("odd", &["RGB"][..]),
            ("grayscale", &["L"][..]),
            ("rgba", &["RGBA"][..]),
            ("small_upscaled", &["RGB"][..]),
            ("large_tiled", &["RGB"][..]),
            ("tiled_thumbnail", &["RGB"][..]),
            ("multiple_images", &["RGB", "RGB"][..]),
        ];
        for (case_name, modes) in cases {
            let images = modes
                .iter()
                .enumerate()
                .map(|(index, mode)| fixture_image(&tensors, case_name, index, mode))
                .collect::<Result<Vec<_>>>()?;
            let actual = processor.process(&images, &device)?;
            let case_metadata = metadata
                .get("cases")
                .and_then(|value| value.get(case_name))
                .ok_or_else(|| {
                    candle::Error::Msg(format!("missing processor metadata case {case_name}"))
                })?;
            let expected_crop_count = case_metadata
                .get("crop_count")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "processor metadata case {case_name} has no crop count"
                    ))
                })?;
            assert_eq!(actual.crops.len(), expected_crop_count);
            let expected_values =
                fixture_tensor(&tensors, &format!("output.{case_name}.pixel_values"))?;
            assert_eq!(actual.pixel_values.dims(), expected_values.dims());
            assert_close(
                &actual.pixel_values,
                expected_values,
                2e-5,
                &format!("processor {case_name} pixel_values"),
            )?;
            let expected_mask = fixture_tensor(
                &tensors,
                &format!("output.{case_name}.pixel_attention_mask"),
            )?;
            assert_eq!(actual.pixel_attention_mask.dims(), expected_mask.dims());
            assert_eq!(
                actual.pixel_attention_mask.to_vec2::<i32>()?,
                expected_mask.to_vec2::<i32>()?
            );
            let expected_shapes =
                fixture_tensor(&tensors, &format!("output.{case_name}.spatial_shapes"))?;
            assert_eq!(
                actual.spatial_shapes.to_vec2::<i64>()?,
                expected_shapes.to_vec2::<i64>()?
            );
            let expected_rows =
                fixture_tensor(&tensors, &format!("output.{case_name}.image_rows"))?
                    .to_vec1::<i64>()?;
            let expected_cols =
                fixture_tensor(&tensors, &format!("output.{case_name}.image_cols"))?
                    .to_vec1::<i64>()?;
            let expected_sizes =
                fixture_tensor(&tensors, &format!("output.{case_name}.image_sizes"))?
                    .to_vec2::<i64>()?;
            assert_eq!(actual.images.len(), expected_rows.len());
            let mut expected_crop_start = 0usize;
            for (index, image) in actual.images.iter().enumerate() {
                let rows = usize::try_from(expected_rows[index])
                    .map_err(|_| candle::Error::Msg("fixture image row is negative".into()))?;
                let cols = usize::try_from(expected_cols[index])
                    .map_err(|_| candle::Error::Msg("fixture image column is negative".into()))?;
                assert_eq!(image.rows, rows);
                assert_eq!(image.cols, cols);
                assert_eq!(
                    image.resized_height,
                    usize::try_from(expected_sizes[index][0]).map_err(|_| {
                        candle::Error::Msg("fixture image height is negative".into())
                    })?
                );
                assert_eq!(
                    image.resized_width,
                    usize::try_from(expected_sizes[index][1]).map_err(|_| {
                        candle::Error::Msg("fixture image width is negative".into())
                    })?
                );
                let tile_count = rows
                    .checked_mul(cols)
                    .ok_or_else(|| candle::Error::Msg("fixture tile count overflow".into()))?;
                let crop_count = if tile_count > 1 && processor.config.use_thumbnail {
                    tile_count
                        .checked_add(1)
                        .ok_or_else(|| candle::Error::Msg("fixture crop count overflow".into()))?
                } else {
                    1
                };
                let expected_crop_end = expected_crop_start
                    .checked_add(crop_count)
                    .ok_or_else(|| candle::Error::Msg("fixture crop range overflow".into()))?;
                assert_eq!(image.crop_range, expected_crop_start..expected_crop_end);
                for (local_index, crop_index) in image.crop_range.clone().enumerate() {
                    let crop = &actual.crops[crop_index];
                    assert_eq!(crop.image_index, index);
                    assert_eq!(crop.crop_index, local_index);
                    match (&crop.kind, tile_count > 1, local_index < tile_count) {
                        (CropKind::Whole, false, _) => {}
                        (CropKind::Tile { row, col }, true, true) => {
                            assert_eq!((*row, *col), (local_index / cols, local_index % cols));
                        }
                        (CropKind::Thumbnail, true, false) => {}
                        _ => candle::bail!(
                            "processor {case_name} crop {local_index} has unexpected kind"
                        ),
                    }
                }
                expected_crop_start = expected_crop_end;
            }
            assert_eq!(expected_crop_start, actual.crops.len());
            let spatial = expected_shapes.to_vec2::<i64>()?;
            assert_eq!(actual.crops.len(), spatial.len());
            for (index, crop) in actual.crops.iter().enumerate() {
                let rows = usize::try_from(spatial[index][0]).unwrap();
                let cols = usize::try_from(spatial[index][1]).unwrap();
                assert_eq!((crop.patch_rows, crop.patch_cols), (rows, cols));
                assert_eq!(
                    crop.projected_tokens,
                    processor.config.projected_token_count(rows, cols)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn official_real_dimension_metadata_covers_spec_and_preserves_orientation() -> Result<()> {
        let metadata: serde_json::Value = serde_json::from_slice(PROCESSOR_METADATA)
            .map_err(|err| candle::Error::Msg(format!("invalid processor metadata: {err}")))?;
        let dimensions = metadata
            .get("real_dimension_oracles")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| candle::Error::Msg("real-dimension metadata is missing".into()))?;
        assert_eq!(dimensions.len(), 10);
        for name in [
            "256x256",
            "277x512",
            "512x277",
            "512x384",
            "384x512",
            "512x512",
            "128x512",
            "512x128",
            "1000x3000",
            "3000x1000",
        ] {
            assert!(
                dimensions.contains_key(name),
                "missing dimension case {name}"
            );
        }
        let processor = Lfm2VlProcessor::new(Lfm2VlProcessorConfig::default())?;
        for (name, oracle) in dimensions {
            let read_usize = |field: &str| -> Result<usize> {
                oracle
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "real-dimension oracle {name} has no valid {field}"
                        ))
                    })
            };
            let width = read_usize("input_width")?;
            let height = read_usize("input_height")?;
            let expected_width = read_usize("smart_width")?;
            let expected_height = read_usize("smart_height")?;
            assert_eq!(
                processor.smart_resize(width, height)?,
                (expected_width, expected_height),
                "smart resize mismatch for {name}"
            );
            let expected_large = oracle
                .get("too_large")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "real-dimension oracle {name} has no too_large flag"
                    ))
                })?;
            assert_eq!(
                processor.is_image_too_large(width, height)?,
                expected_large,
                "large-image decision mismatch for {name}"
            );
            let selected_grid = oracle
                .get("selected_grid")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("real-dimension oracle {name} has no selected grid"))
                })?;
            if selected_grid.len() != 2 {
                candle::bail!("real-dimension oracle {name} has an invalid selected grid")
            }
            let expected_cols = selected_grid[0]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "real-dimension oracle {name} has invalid grid columns"
                    ))
                })?;
            let expected_rows = selected_grid[1]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "real-dimension oracle {name} has invalid grid rows"
                    ))
                })?;
            if expected_large {
                assert_eq!(
                    processor.closest_tile_grid(width, height)?,
                    (expected_cols, expected_rows),
                    "tile-grid mismatch for {name}"
                );
                let expected_canvas = oracle
                    .get("tile_canvas")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "real-dimension oracle {name} has no tile canvas"
                        ))
                    })?;
                assert_eq!(
                    expected_canvas,
                    &vec![
                        serde_json::json!(expected_cols * processor.config.tile_size),
                        serde_json::json!(expected_rows * processor.config.tile_size),
                    ],
                    "tile-canvas mismatch for {name}"
                );
            } else {
                assert_eq!((expected_cols, expected_rows), (1, 1));
                assert!(oracle
                    .get("tile_canvas")
                    .is_some_and(serde_json::Value::is_null));
            }

            let crop_order = oracle
                .get("crop_order")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("real-dimension oracle {name} has no crop order"))
                })?;
            let mut expected_order = Vec::new();
            if expected_large {
                for row in 0..expected_rows {
                    for col in 0..expected_cols {
                        expected_order.push(serde_json::json!({
                            "kind": "tile",
                            "row": row,
                            "col": col,
                        }));
                    }
                }
                if processor.config.use_thumbnail && expected_rows * expected_cols > 1 {
                    expected_order.push(serde_json::json!({"kind": "thumbnail"}));
                }
            } else {
                expected_order.push(serde_json::json!({"kind": "whole"}));
            }
            assert_eq!(
                crop_order, &expected_order,
                "crop-order mismatch for {name}"
            );
        }
        Ok(())
    }
}
