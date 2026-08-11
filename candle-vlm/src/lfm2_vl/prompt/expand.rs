impl Lfm2VlPrompt {

    pub fn new(
        tokenizer: Tokenizer,
        expected_image_token_id: Option<u32>,
        config: Lfm2VlProcessorConfig,
        options: PromptOptions,
    ) -> Result<Self> {
        config.validate()?;
        if options.context_length == Some(0) {
            candle::bail!("LFM2-VL prompt context length must be positive")
        }
        let special_tokens = Lfm2VlSpecialTokens::resolve(
            &tokenizer,
            expected_image_token_id,
            config.max_tiles,
            options.use_image_special_tokens,
        )?;
        Ok(Self {
            tokenizer,
            special_tokens,
            config,
            options,
        })
    }

    pub fn from_processor_config(
        tokenizer: Tokenizer,
        expected_image_token_id: Option<u32>,
        config: &Lfm2VlProcessorConfig,
        options: PromptOptions,
    ) -> Result<Self> {
        Self::new(tokenizer, expected_image_token_id, config.clone(), options)
    }

    pub fn special_tokens(&self) -> &Lfm2VlSpecialTokens {
        &self.special_tokens
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Expand user-provided sentinels without moving them.
    pub fn expand(&self, text: &str, images: &ProcessedVisionBatch) -> Result<ExpandedPrompt> {
        let sentinel_count = text.match_indices(IMAGE_SENTINEL).count();
        if images.images.is_empty() {
            self.validate_empty_image_metadata(images)?;
            if sentinel_count != 0 {
                candle::bail!(
                    "LFM2-VL prompt contains {sentinel_count} image sentinels but no images"
                )
            }
            let encoding = self.encode_without_truncation(text)?;
            let mut input_ids =
                try_vec_with_capacity(encoding.get_ids().len(), "LFM2-VL prompt token IDs")?;
            input_ids.extend_from_slice(encoding.get_ids());
            self.check_context_length(input_ids.len())?;
            let mut expanded_text =
                try_string_with_capacity(text.len(), "LFM2-VL text-only prompt")?;
            expanded_text.push_str(text);
            return Ok(ExpandedPrompt {
                expanded_text,
                input_ids,
                image_spans: Vec::new(),
                per_image_spans: Vec::new(),
                span_image_indices: Vec::new(),
                span_crop_indices: Vec::new(),
            });
        }
        self.validate_image_metadata(images)?;
        if sentinel_count != images.images.len() {
            candle::bail!(
                "LFM2-VL prompt contains {sentinel_count} image sentinels for {} images",
                images.images.len()
            )
        }
        let expected_lengths = self.expected_crop_lengths(images)?;
        let expected_placeholder_count =
            expected_lengths.iter().try_fold(0usize, |total, &length| {
                total
                    .checked_add(length)
                    .ok_or_else(|| candle::Error::Msg("LFM2-VL placeholder count overflow".into()))
            })?;
        if let Some(limit) = self.options.context_length.or(self.config.context_length) {
            if expected_placeholder_count > limit {
                candle::bail!(
                    "LFM2-VL image placeholders {expected_placeholder_count} exceed context length {limit}"
                )
            }
        }
        self.preflight_expanded_context(text, sentinel_count, expected_placeholder_count, images)?;
        let sentinel_bytes = sentinel_count
            .checked_mul(IMAGE_SENTINEL.len())
            .ok_or_else(|| candle::Error::Msg("LFM2-VL sentinel byte count overflow".into()))?;
        let mut expanded_bytes = text
            .len()
            .checked_sub(sentinel_bytes)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL sentinel byte count is invalid".into()))?;
        let mut image_blocks =
            try_vec_with_capacity(images.images.len(), "LFM2-VL image prompt blocks")?;
        for image_index in 0..images.images.len() {
            let block = self.image_block(images, image_index)?;
            expanded_bytes = expanded_bytes.checked_add(block.len()).ok_or_else(|| {
                candle::Error::Msg("LFM2-VL expanded prompt size overflow".into())
            })?;
            image_blocks.push(block);
        }
        let mut expanded_text =
            try_string_with_capacity(expanded_bytes, "LFM2-VL expanded prompt")?;
        let mut source_end = 0usize;
        let mut image_index = 0usize;
        for (start, _) in text.match_indices(IMAGE_SENTINEL) {
            expanded_text.push_str(&text[source_end..start]);
            let block = image_blocks.get(image_index).ok_or_else(|| {
                candle::Error::Msg("LFM2-VL image prompt block index is out of bounds".into())
            })?;
            expanded_text.push_str(block);
            source_end = start
                .checked_add(IMAGE_SENTINEL.len())
                .ok_or_else(|| candle::Error::Msg("LFM2-VL prompt offset overflow".into()))?;
            image_index = image_index
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL image index overflow".into()))?;
        }
        expanded_text.push_str(&text[source_end..]);
        let encoding = self.encode_without_truncation(&expanded_text)?;
        let mut input_ids =
            try_vec_with_capacity(encoding.get_ids().len(), "LFM2-VL expanded token IDs")?;
        input_ids.extend_from_slice(encoding.get_ids());
        self.check_context_length(input_ids.len())?;
        let image_spans = find_crop_spans(
            &input_ids,
            self.special_tokens.image_token_id,
            &expected_lengths,
        )?;
        let mut per_image_spans =
            try_vec_with_capacity(images.images.len(), "LFM2-VL per-image span table")?;
        let mut span_image_indices =
            try_vec_with_capacity(image_spans.len(), "LFM2-VL span image indices")?;
        let mut span_crop_indices =
            try_vec_with_capacity(image_spans.len(), "LFM2-VL span crop indices")?;
        for (image_index, image) in images.images.iter().enumerate() {
            let start = image.crop_range.start;
            let end = image.crop_range.end;
            let mut spans = try_vec_with_capacity(
                end.checked_sub(start).ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL per-image span range is invalid".into())
                })?,
                "LFM2-VL per-image crop spans",
            )?;
            spans.extend_from_slice(&image_spans[start..end]);
            for (local_crop_index, _) in spans.iter().enumerate() {
                span_image_indices.push(image_index);
                span_crop_indices.push(local_crop_index);
            }
            per_image_spans.push(spans);
        }
        Ok(ExpandedPrompt {
            expanded_text,
            input_ids,
            image_spans,
            per_image_spans,
            span_image_indices,
            span_crop_indices,
        })
    }

    /// Default helper for a chat turn with accompanying text. If explicit
    /// sentinels are present, their positions are preserved; otherwise one
    /// sentinel is placed before the text for each supplied image.
    pub fn image_before_text(
        &self,
        text: &str,
        images: &ProcessedVisionBatch,
    ) -> Result<ExpandedPrompt> {
        if text.match_indices(IMAGE_SENTINEL).count() != 0 {
            return self.expand(text, images);
        }
        let mut prompt = String::new();
        append_repeated(&mut prompt, IMAGE_SENTINEL, images.images.len())?;
        try_push_str(&mut prompt, text, "LFM2-VL image-first prompt")?;
        self.expand(&prompt, images)
    }

    pub fn expand_image_first(
        &self,
        text: &str,
        images: &ProcessedVisionBatch,
    ) -> Result<ExpandedPrompt> {
        self.image_before_text(text, images)
    }
}
