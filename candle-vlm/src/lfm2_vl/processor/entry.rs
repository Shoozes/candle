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
        self.preflight_request(images)?;
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
        let total_projected_tokens = work.iter().try_fold(0usize, |total, image| {
            self.config
                .vision_limits
                .check_crops_per_image(image.crops.len())?;
            image.crops.iter().try_fold(total, |total, crop| {
                self.config.vision_limits.check_crop(
                    crop.patch_rows,
                    crop.patch_cols,
                    crop.projected_tokens,
                )?;
                total.checked_add(crop.projected_tokens).ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL projected token total overflow".into())
                })
            })
        })?;
        self.config.vision_limits.check_request(
            images.len(),
            crop_count,
            total_projected_tokens,
        )?;
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
}
