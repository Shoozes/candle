impl Lfm2VlPrompt {
    fn image_block(&self, images: &ProcessedVisionBatch, image_index: usize) -> Result<String> {
        let image = images
            .images
            .get(image_index)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image index is out of bounds".into()))?;
        let mut block = String::new();
        if self.options.use_image_special_tokens {
            try_push_str(
                &mut block,
                &self.special_tokens.image_start_token,
                "LFM2-VL image prompt block",
            )?;
        }
        for crop_index in image.crop_range.clone() {
            let crop = images
                .crops
                .get(crop_index)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL crop index is out of bounds".into()))?;
            if self.options.use_image_special_tokens {
                match crop.kind {
                    CropKind::Whole => {}
                    CropKind::Tile { row, col } => {
                        let marker_row = row.checked_add(1).ok_or_else(|| {
                            candle::Error::Msg("LFM2-VL tile row marker overflow".into())
                        })?;
                        let marker_col = col.checked_add(1).ok_or_else(|| {
                            candle::Error::Msg("LFM2-VL tile column marker overflow".into())
                        })?;
                        let name = tile_token_name(marker_row, marker_col);
                        if !self
                            .special_tokens
                            .tile_tokens
                            .contains_key(&(marker_row, marker_col))
                        {
                            let _ = resolve_atomic_token(&self.tokenizer, &name)?;
                        }
                        try_push_str(&mut block, &name, "LFM2-VL image prompt block")?;
                    }
                    CropKind::Thumbnail => {
                        let _ = resolve_atomic_token(
                            &self.tokenizer,
                            &self.special_tokens.image_thumbnail_token,
                        )?;
                        try_push_str(
                            &mut block,
                            &self.special_tokens.image_thumbnail_token,
                            "LFM2-VL image prompt block",
                        )?;
                    }
                }
            }
            append_repeated(
                &mut block,
                &self.special_tokens.image_token,
                crop.projected_tokens,
            )?;
        }
        if self.options.use_image_special_tokens {
            try_push_str(
                &mut block,
                &self.special_tokens.image_end_token,
                "LFM2-VL image prompt block",
            )?;
        }
        Ok(block)
    }
}
