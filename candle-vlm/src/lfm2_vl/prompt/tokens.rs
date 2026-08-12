impl Lfm2VlSpecialTokens {
    pub fn resolve(
        tokenizer: &Tokenizer,
        expected_image_token_id: Option<u32>,
        max_tiles: usize,
        require_markers: bool,
    ) -> Result<Self> {
        let image_token_id = resolve_atomic_token(tokenizer, IMAGE_SENTINEL)?;
        if let Some(expected) = expected_image_token_id {
            if expected != image_token_id {
                candle::bail!(
                    "LFM2-VL tokenizer image token id {image_token_id} does not match model id {expected}"
                )
            }
        }
        let image_start_token = IMAGE_START.to_owned();
        let image_end_token = IMAGE_END.to_owned();
        let image_thumbnail_token = IMAGE_THUMBNAIL.to_owned();
        let mut tile_tokens = HashMap::new();
        if require_markers {
            let _ = resolve_atomic_token(tokenizer, IMAGE_START)?;
            let _ = resolve_atomic_token(tokenizer, IMAGE_END)?;
            let _ = resolve_atomic_token(tokenizer, IMAGE_THUMBNAIL)?;
            for row in 1..=max_tiles {
                for col in 1..=max_tiles {
                    let token = tile_token_name(row, col);
                    if resolve_atomic_token_if_present(tokenizer, &token)?.is_some() {
                        tile_tokens.insert((row, col), token);
                    }
                }
            }
        }
        Ok(Self {
            image_token: IMAGE_SENTINEL.to_owned(),
            image_token_id,
            image_start_token,
            image_end_token,
            image_thumbnail_token,
            tile_tokens,
        })
    }
}
