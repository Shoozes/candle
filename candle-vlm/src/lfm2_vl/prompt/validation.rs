impl Lfm2VlPrompt {
    fn encode_without_truncation(&self, text: &str) -> Result<Encoding> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_truncation(None).map_err(|err| {
            candle::Error::Msg(format!("failed to disable tokenizer truncation: {err}"))
        })?;
        tokenizer
            .encode(text, false)
            .map_err(|err| candle::Error::Msg(format!("LFM2-VL prompt tokenization failed: {err}")))
    }

    fn check_context_length(&self, length: usize) -> Result<()> {
        let limit = self.options.context_length.or(self.config.context_length);
        if let Some(limit) = limit {
            if length > limit {
                candle::bail!(
                    "LFM2-VL expanded prompt length {length} exceeds context length {limit}"
                )
            }
        }
        Ok(())
    }

    fn preflight_expanded_context(
        &self,
        text: &str,
        sentinel_count: usize,
        expected_placeholder_count: usize,
        images: &ProcessedVisionBatch,
    ) -> Result<()> {
        let Some(limit) = self.options.context_length.or(self.config.context_length) else {
            return Ok(());
        };
        let raw = self.encode_without_truncation(text)?;
        let raw_image_tokens = raw
            .get_ids()
            .iter()
            .filter(|&&token_id| token_id == self.special_tokens.image_token_id)
            .count();
        if raw_image_tokens != sentinel_count {
            candle::bail!(
                "LFM2-VL raw prompt encoded {raw_image_tokens} image tokens for {sentinel_count} sentinels"
            )
        }
        let marker_tokens = if self.options.use_image_special_tokens {
            let wrappers =
                images.images.len().checked_mul(2).ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL image wrapper count overflow".into())
                })?;
            let crop_markers = images
                .crops
                .iter()
                .filter(|crop| !matches!(crop.kind, CropKind::Whole))
                .count();
            wrappers
                .checked_add(crop_markers)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL image marker count overflow".into()))?
        } else {
            0
        };
        let predicted = raw
            .get_ids()
            .len()
            .checked_sub(sentinel_count)
            .and_then(|value| value.checked_add(expected_placeholder_count))
            .and_then(|value| value.checked_add(marker_tokens))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL prompt token count overflow".into()))?;
        if predicted > limit {
            candle::bail!(
                "LFM2-VL predicted expanded prompt length {predicted} exceeds context length {limit}"
            )
        }
        Ok(())
    }

    fn expected_crop_lengths(&self, images: &ProcessedVisionBatch) -> Result<Vec<usize>> {
        let mut lengths =
            try_vec_with_capacity(images.crops.len(), "LFM2-VL expected crop lengths")?;
        let mut total_projected_tokens = 0usize;
        for crop in &images.crops {
            let expected = self
                .config
                .projected_token_count(crop.patch_rows, crop.patch_cols)?;
            if crop.projected_tokens != expected {
                candle::bail!(
                    "LFM2-VL crop metadata projects to {}, expected {expected}",
                    crop.projected_tokens
                )
            }
            self.config
                .vision_limits
                .check_crop(crop.patch_rows, crop.patch_cols, expected)?;
            total_projected_tokens =
                total_projected_tokens
                    .checked_add(expected)
                    .ok_or_else(|| {
                        candle::Error::Msg("LFM2-VL projected token total overflow".into())
                    })?;
            lengths.push(expected);
        }
        self.config
            .vision_limits
            .check_total_projected_tokens(total_projected_tokens)?;
        Ok(lengths)
    }

    fn validate_image_metadata(&self, images: &ProcessedVisionBatch) -> Result<()> {
        if images.images.is_empty() {
            candle::bail!("LFM2-VL prompt requires at least one image")
        }
        let crop_count = images.crops.len();
        self.config
            .vision_limits
            .check_image_count(images.images.len())?;
        self.config.vision_limits.check_total_crops(crop_count)?;
        self.validate_tensor_crop_counts(images, crop_count)?;
        let mut next_crop = 0usize;
        for (image_index, image) in images.images.iter().enumerate() {
            if image.crop_range.start != next_crop
                || image.crop_range.start >= image.crop_range.end
                || image.crop_range.end > crop_count
            {
                candle::bail!("LFM2-VL image crop ranges are not ordered and contiguous")
            }
            self.config
                .vision_limits
                .check_crops_per_image(image.crop_range.len())?;
            if image.rows == 0 || image.cols == 0 {
                candle::bail!("LFM2-VL image tile-grid dimensions must be positive")
            }
            self.config.vision_limits.check_image_surface(
                "resized image metadata",
                image.resized_width,
                image.resized_height,
            )?;
            let tile_count = image
                .rows
                .checked_mul(image.cols)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL tile grid overflow".into()))?;
            let tiled = tile_count > 1;
            if tiled
                && (!self.config.do_image_splitting
                    || tile_count < self.config.min_tiles
                    || tile_count > self.config.max_tiles)
            {
                candle::bail!("LFM2-VL image tile grid does not match processor configuration")
            }
            let thumbnail_count = usize::from(tiled && self.config.use_thumbnail);
            let expected_crops = if tiled {
                tile_count
                    .checked_add(thumbnail_count)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL crop count overflow".into()))?
            } else {
                1
            };
            if image.crop_range.len() != expected_crops {
                candle::bail!("LFM2-VL crop kinds do not match image grid metadata")
            }
            for (local_crop_index, crop_index) in image.crop_range.clone().enumerate() {
                let crop = &images.crops[crop_index];
                if crop.image_index != image_index || crop.crop_index != local_crop_index {
                    candle::bail!("LFM2-VL crop metadata does not match image ranges")
                }
                let expected_kind = if !tiled {
                    CropKind::Whole
                } else if local_crop_index < tile_count {
                    CropKind::Tile {
                        row: local_crop_index / image.cols,
                        col: local_crop_index % image.cols,
                    }
                } else {
                    CropKind::Thumbnail
                };
                if !crop_kind_matches(&crop.kind, &expected_kind) {
                    candle::bail!("LFM2-VL crop metadata is not in row-major processor order")
                }
                let (expected_patch_rows, expected_patch_cols) = match expected_kind {
                    CropKind::Tile { .. } => {
                        let tile_patches = self.config.tile_size / self.config.encoder_patch_size;
                        (tile_patches, tile_patches)
                    }
                    CropKind::Whole | CropKind::Thumbnail => {
                        if image.resized_height % self.config.encoder_patch_size != 0
                            || image.resized_width % self.config.encoder_patch_size != 0
                        {
                            candle::bail!(
                                "LFM2-VL resized image metadata is not patch-size aligned"
                            )
                        }
                        (
                            image.resized_height / self.config.encoder_patch_size,
                            image.resized_width / self.config.encoder_patch_size,
                        )
                    }
                };
                if crop.patch_rows != expected_patch_rows || crop.patch_cols != expected_patch_cols
                {
                    candle::bail!("LFM2-VL crop patch shape does not match image metadata")
                }
                let projected_tokens = self
                    .config
                    .projected_token_count(crop.patch_rows, crop.patch_cols)?;
                self.config.vision_limits.check_crop(
                    crop.patch_rows,
                    crop.patch_cols,
                    projected_tokens,
                )?;
            }
            next_crop = image.crop_range.end;
        }
        if next_crop != crop_count {
            candle::bail!("LFM2-VL image ranges do not cover every crop")
        }
        Ok(())
    }

    fn validate_empty_image_metadata(&self, images: &ProcessedVisionBatch) -> Result<()> {
        self.validate_tensor_crop_counts(images, 0)?;
        if !images.crops.is_empty() {
            candle::bail!("LFM2-VL empty image metadata contains crop records")
        }
        Ok(())
    }

    fn validate_tensor_crop_counts(
        &self,
        images: &ProcessedVisionBatch,
        expected: usize,
    ) -> Result<()> {
        let (pixel_crops, _, _) = images.pixel_values.dims3()?;
        let (mask_crops, _) = images.pixel_attention_mask.dims2()?;
        let (shape_crops, shape_width) = images.spatial_shapes.dims2()?;
        if pixel_crops != expected
            || mask_crops != expected
            || shape_crops != expected
            || shape_width != 2
        {
            candle::bail!("LFM2-VL packed tensor crop counts do not match metadata")
        }
        Ok(())
    }
}
