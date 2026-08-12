impl Lfm2VlProcessor {
    fn preflight_request(&self, images: &[DynamicImage]) -> Result<()> {
        self.config.vision_limits.validate()?;
        self.config.vision_limits.check_image_count(images.len())?;
        let mut total_crops = 0usize;
        let mut total_projected_tokens = 0usize;
        for image in images {
            let (width, height) = dynamic_image_dimensions(image)?;
            self.config
                .vision_limits
                .check_source_image(width, height)?;
            let budget = self.image_budget(width, height)?;
            self.config
                .vision_limits
                .check_crops_per_image(budget.crop_count)?;
            total_crops = total_crops
                .checked_add(budget.crop_count)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL crop count overflow".into()))?;
            total_projected_tokens = total_projected_tokens
                .checked_add(budget.projected_tokens)
                .ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL projected token total overflow".into())
                })?;
        }
        self.config
            .vision_limits
            .check_request(images.len(), total_crops, total_projected_tokens)
    }

    fn image_budget(&self, width: usize, height: usize) -> Result<ImageBudget> {
        let split = self.config.do_image_splitting && self.is_image_too_large(width, height)?;
        if !split {
            let (resized_width, resized_height) = if self.config.do_resize {
                self.smart_resize(width, height)?
            } else {
                (width, height)
            };
            let projected_tokens = self.crop_projected_tokens(resized_width, resized_height)?;
            return Ok(ImageBudget {
                crop_count: 1,
                projected_tokens,
            });
        }

        let (cols, rows) = self.closest_tile_grid(width, height)?;
        let tile_count = rows
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tile count overflow".into()))?;
        let tile_projected_tokens =
            self.crop_projected_tokens(self.config.tile_size, self.config.tile_size)?;
        let canvas_width = self
            .config
            .tile_size
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tiled width overflow".into()))?;
        let canvas_height = self
            .config
            .tile_size
            .checked_mul(rows)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL tiled height overflow".into()))?;
        self.config.vision_limits.check_image_surface(
            "tiled resize canvas",
            canvas_width,
            canvas_height,
        )?;
        let mut projected_tokens =
            tile_count
                .checked_mul(tile_projected_tokens)
                .ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL tiled projected token count overflow".into())
                })?;
        let has_thumbnail = self.config.use_thumbnail && tile_count > 1;
        if has_thumbnail {
            let (thumbnail_width, thumbnail_height) = self.smart_resize(width, height)?;
            projected_tokens = projected_tokens
                .checked_add(self.crop_projected_tokens(thumbnail_width, thumbnail_height)?)
                .ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL thumbnail projected token count overflow".into())
                })?;
        }
        let crop_count = tile_count
            .checked_add(usize::from(has_thumbnail))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL crop count overflow".into()))?;
        Ok(ImageBudget {
            crop_count,
            projected_tokens,
        })
    }

    fn crop_projected_tokens(&self, width: usize, height: usize) -> Result<usize> {
        self.config
            .vision_limits
            .check_image_surface("processed crop", width, height)?;
        let patch = self.config.encoder_patch_size;
        if width == 0 || height == 0 || width % patch != 0 || height % patch != 0 {
            candle::bail!(
                "LFM2-VL crop dimensions [{width}, {height}] must be positive and divisible by patch size {patch}"
            )
        }
        let patch_rows = height / patch;
        let patch_cols = width / patch;
        let projected_tokens = self.config.projected_token_count(patch_rows, patch_cols)?;
        self.config
            .vision_limits
            .check_crop(patch_rows, patch_cols, projected_tokens)?;
        Ok(projected_tokens)
    }
}
