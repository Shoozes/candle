//! RGB image processing, NaFlex tiling, and packed patch construction.

use super::config::Lfm2VlProcessorConfig;
use super::types::{CropKind, CropMeta, ImageMeta, ProcessedVisionBatch};
use crate::image::{resize_bilinear_antialias, to_rgb8};
use candle::{Device, Result, Tensor};
use image::{imageops, DynamicImage, RgbImage};

include!("processor/types.rs");
include!("processor/entry.rs");
include!("processor/budget.rs");
include!("processor/crops.rs");
include!("processor/helpers.rs");

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
            vision_limits: Default::default(),
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
        assert!(Lfm2VlProcessor::new(config).is_err());
        Ok(())
    }

    #[test]
    fn request_limits_fail_at_source_crop_and_token_boundaries() -> Result<()> {
        let mut exact = tiny_config();
        exact.do_image_splitting = false;
        exact.vision_limits.max_source_pixels = 16;
        exact.vision_limits.max_images = 1;
        exact.vision_limits.max_crops_per_image = 1;
        exact.vision_limits.max_total_crops = 1;
        exact.vision_limits.max_patches_per_crop = 64;
        exact.vision_limits.max_total_projected_tokens = 1;
        let processor = Lfm2VlProcessor::new(exact.clone())?;
        let output = processor.process(&[image(4, 4)], &Device::Cpu)?;
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.crops.len(), 1);
        assert_eq!(output.crops[0].projected_tokens, 1);

        let mut source_over = exact.clone();
        source_over.vision_limits.max_source_pixels = 15;
        let error = Lfm2VlProcessor::new(source_over)?
            .process(&[image(4, 4)], &Device::Cpu)
            .expect_err("source pixel limit should reject one-over input");
        assert!(error.to_string().contains("source image has 16 pixels"));

        let error = processor
            .process(&[image(4, 4), image(4, 4)], &Device::Cpu)
            .expect_err("image count limit should reject one-over input");
        assert!(error.to_string().contains("2 images"));

        let mut token_over = exact.clone();
        token_over.vision_limits.max_source_pixels = 32;
        let error = Lfm2VlProcessor::new(token_over)?
            .process(&[image(8, 4)], &Device::Cpu)
            .expect_err("projected token limit should reject one-over input");
        assert!(error.to_string().contains("2 projected tokens"));

        let mut crop_over = tiny_config();
        crop_over.max_pixels_tolerance = 0.01;
        crop_over.vision_limits.max_crops_per_image = 1;
        crop_over.vision_limits.max_total_crops = 1;
        let error = Lfm2VlProcessor::new(crop_over)?
            .process(&[image(32, 16)], &Device::Cpu)
            .expect_err("crop limit should reject a tiled image");
        assert!(error
            .to_string()
            .contains("crops, exceeding per-image limit 1"));

        let mut total_crop_over = exact;
        total_crop_over.vision_limits.max_images = 2;
        let error = Lfm2VlProcessor::new(total_crop_over)?
            .process(&[image(4, 4), image(4, 4)], &Device::Cpu)
            .expect_err("total crop limit should reject the second image");
        assert!(error
            .to_string()
            .contains("2 crops, exceeding total limit 1"));
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
