impl Lfm2VlProcessor {
    fn process_image(&self, image: &DynamicImage, image_index: usize) -> Result<ImageWork> {
        let (source_width, source_height) = dynamic_image_dimensions(image)?;
        self.config
            .vision_limits
            .check_source_image(source_width, source_height)?;
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
