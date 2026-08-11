//! Tokenizer-backed LFM2-VL image-sentinel expansion and span validation.

use super::config::Lfm2VlProcessorConfig;
use super::types::{CropKind, ImageTokenSpan, ProcessedVisionBatch};
use candle::Result;
use std::collections::HashMap;
use tokenizers::{Encoding, Tokenizer};

const IMAGE_SENTINEL: &str = "<image>";
const IMAGE_START: &str = "<|image_start|>";
const IMAGE_END: &str = "<|image_end|>";
const IMAGE_THUMBNAIL: &str = "<|img_thumbnail|>";

#[derive(Clone, Debug)]
pub struct Lfm2VlSpecialTokens {
    pub image_token: String,
    pub image_token_id: u32,
    pub image_start_token: String,
    pub image_end_token: String,
    pub image_thumbnail_token: String,
    pub tile_tokens: HashMap<(usize, usize), String>,
}

#[derive(Clone, Debug)]
pub struct PromptOptions {
    pub use_image_special_tokens: bool,
    pub context_length: Option<usize>,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            // The pinned Lfm2VlProcessorKwargs default is true. Callers that
            // need adjacent placeholder runs must opt out explicitly.
            use_image_special_tokens: true,
            context_length: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExpandedPrompt {
    pub expanded_text: String,
    pub input_ids: Vec<u32>,
    /// One span per crop in `ProcessedVisionBatch` order, including a
    /// thumbnail crop. Adjacent spans are retained when marker tokens are not
    /// enabled.
    pub image_spans: Vec<ImageTokenSpan>,
    pub per_image_spans: Vec<Vec<ImageTokenSpan>>,
    pub span_image_indices: Vec<usize>,
    pub span_crop_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Lfm2VlPrompt {
    tokenizer: Tokenizer,
    special_tokens: Lfm2VlSpecialTokens,
    config: Lfm2VlProcessorConfig,
    options: PromptOptions,
}

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

fn crop_kind_matches(actual: &CropKind, expected: &CropKind) -> bool {
    match (actual, expected) {
        (CropKind::Whole, CropKind::Whole) | (CropKind::Thumbnail, CropKind::Thumbnail) => true,
        (
            CropKind::Tile {
                row: actual_row,
                col: actual_col,
            },
            CropKind::Tile {
                row: expected_row,
                col: expected_col,
            },
        ) => actual_row == expected_row && actual_col == expected_col,
        _ => false,
    }
}

fn tile_token_name(row: usize, col: usize) -> String {
    format!("<|img_row_{row}_col_{col}|>")
}

fn append_repeated(target: &mut String, token: &str, count: usize) -> Result<()> {
    let bytes = token
        .len()
        .checked_mul(count)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL placeholder string size overflow".into()))?;
    target
        .try_reserve(bytes)
        .map_err(|_| candle::Error::Msg("LFM2-VL placeholder string allocation failed".into()))?;
    for _ in 0..count {
        target.push_str(token);
    }
    Ok(())
}

fn try_string_with_capacity(capacity: usize, label: &str) -> Result<String> {
    let mut value = String::new();
    value.try_reserve_exact(capacity).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to allocate {label} ({capacity} bytes): {err}"
        ))
    })?;
    Ok(value)
}

fn try_push_str(target: &mut String, value: &str, label: &str) -> Result<()> {
    target.try_reserve(value.len()).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to grow {label} by {} bytes: {err}",
            value.len()
        ))
    })?;
    target.push_str(value);
    Ok(())
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

fn resolve_atomic_token(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    let id = tokenizer.token_to_id(token).ok_or_else(|| {
        candle::Error::Msg(format!("tokenizer is missing required token {token:?}"))
    })?;
    let encoding = tokenizer
        .encode(token, false)
        .map_err(|err| candle::Error::Msg(format!("failed to encode token {token:?}: {err}")))?;
    if encoding.get_ids() != [id] {
        candle::bail!("tokenizer does not encode {token:?} as one atomic token")
    }
    Ok(id)
}

fn resolve_atomic_token_if_present(tokenizer: &Tokenizer, token: &str) -> Result<Option<u32>> {
    match tokenizer.token_to_id(token) {
        Some(id) => {
            let resolved = resolve_atomic_token(tokenizer, token)?;
            if resolved != id {
                candle::bail!("tokenizer token resolution changed for {token:?}")
            }
            Ok(Some(id))
        }
        None => Ok(None),
    }
}

fn find_crop_spans(
    input_ids: &[u32],
    image_token_id: u32,
    expected_lengths: &[usize],
) -> Result<Vec<ImageTokenSpan>> {
    if expected_lengths.is_empty() {
        candle::bail!("LFM2-VL prompt contains no expected crop spans")
    }
    let mut spans = try_vec_with_capacity(expected_lengths.len(), "LFM2-VL crop span table")?;
    let mut cursor = 0usize;
    for &length in expected_lengths {
        if length == 0 {
            candle::bail!("LFM2-VL crop span length must be positive")
        }
        while cursor < input_ids.len() && input_ids[cursor] != image_token_id {
            cursor += 1;
        }
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image span overflow".into()))?;
        if end > input_ids.len() {
            candle::bail!("LFM2-VL prompt has fewer image placeholders than expected")
        }
        if input_ids[cursor..end]
            .iter()
            .any(|&token_id| token_id != image_token_id)
        {
            candle::bail!("LFM2-VL image placeholders are not contiguous")
        }
        spans.push(ImageTokenSpan::new(0, cursor, end));
        cursor = end;
    }
    if input_ids[cursor..].contains(&image_token_id) {
        candle::bail!("LFM2-VL prompt contains unexpected image placeholders")
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm2_vl::types::{CropMeta, ImageMeta};
    use candle::{DType, Device, Tensor};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::AddedToken;

    const PROCESSOR_METADATA: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_processor_tiny/metadata.json");

    fn tokenizer(with_markers: bool) -> Tokenizer {
        let vocab = [
            (0, "<unk>"),
            (1, "hello"),
            (2, "world"),
            (3, "<image>"),
            (4, "<|image_start|>"),
            (5, "<|image_end|>"),
            (6, "<|img_thumbnail|>"),
            (7, "<|img_row_1_col_1|>"),
            (8, "<|img_row_1_col_2|>"),
            (9, "<|img_row_1_col_3|>"),
            (10, "<|img_row_2_col_1|>"),
            (11, "<|img_row_2_col_2|>"),
            (12, "<|img_row_2_col_3|>"),
            (13, "and"),
            (14, "Describe"),
            (15, "this"),
            (16, "image"),
            (17, "turn"),
            (18, "one"),
            (19, "two"),
        ]
        .into_iter()
        .map(|(id, token)| (token.to_owned(), id))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .expect("fixed test tokenizer");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let mut tokens = vec![AddedToken::from("<image>", true)];
        if with_markers {
            tokens.extend([
                AddedToken::from("<|image_start|>", true),
                AddedToken::from("<|image_end|>", true),
                AddedToken::from("<|img_thumbnail|>", true),
                AddedToken::from("<|img_row_1_col_1|>", true),
                AddedToken::from("<|img_row_1_col_2|>", true),
                AddedToken::from("<|img_row_1_col_3|>", true),
                AddedToken::from("<|img_row_2_col_1|>", true),
                AddedToken::from("<|img_row_2_col_2|>", true),
                AddedToken::from("<|img_row_2_col_3|>", true),
            ]);
        }
        tokenizer.add_special_tokens(&tokens);
        tokenizer
    }

    fn config() -> Lfm2VlProcessorConfig {
        Lfm2VlProcessorConfig {
            downsample_factor: 2,
            encoder_patch_size: 2,
            min_tiles: 2,
            max_tiles: 4,
            tile_size: 8,
            min_image_tokens: 1,
            max_image_tokens: 16,
            max_num_patches: Some(64),
            context_length: Some(128),
            ..Lfm2VlProcessorConfig::default()
        }
    }

    fn batch(crops: Vec<CropMeta>, images: Vec<ImageMeta>) -> ProcessedVisionBatch {
        ProcessedVisionBatch {
            pixel_values: Tensor::zeros((crops.len(), 64, 12), DType::F32, &Device::Cpu)
                .expect("fixed test tensor"),
            pixel_attention_mask: Tensor::zeros((crops.len(), 64), DType::I32, &Device::Cpu)
                .expect("fixed test tensor"),
            spatial_shapes: Tensor::zeros((crops.len(), 2), DType::I64, &Device::Cpu)
                .expect("fixed test tensor"),
            crops,
            images,
        }
    }

    fn whole_batch(image_count: usize) -> ProcessedVisionBatch {
        let crops = (0..image_count)
            .map(|image_index| CropMeta {
                image_index,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            })
            .collect::<Vec<_>>();
        let images = (0..image_count)
            .map(|index| ImageMeta {
                crop_range: index..index + 1,
                rows: 1,
                cols: 1,
                resized_width: 8,
                resized_height: 8,
            })
            .collect();
        batch(crops, images)
    }

    #[test]
    fn preserves_sentinel_position_and_records_one_span_per_crop() -> Result<()> {
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let output = prompt.expand("hello <image> world", &whole_batch(1))?;
        assert_eq!(output.image_spans.len(), 1);
        assert_eq!(output.input_ids[0], 1);
        assert_eq!(*output.input_ids.last().unwrap(), 2);
        Ok(())
    }

    #[test]
    fn tiled_markers_remain_text_and_crop_spans_are_separate() -> Result<()> {
        let crops = vec![
            CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Tile { row: 0, col: 0 },
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            },
            CropMeta {
                image_index: 0,
                crop_index: 1,
                kind: CropKind::Tile { row: 0, col: 1 },
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            },
            CropMeta {
                image_index: 0,
                crop_index: 2,
                kind: CropKind::Thumbnail,
                patch_rows: 4,
                patch_cols: 8,
                projected_tokens: 8,
            },
        ];
        let images = vec![ImageMeta {
            crop_range: 0..3,
            rows: 1,
            cols: 2,
            resized_width: 16,
            resized_height: 8,
        }];
        let prompt = Lfm2VlPrompt::new(
            tokenizer(true),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: true,
                context_length: None,
            },
        )?;
        let output = prompt.expand("<image>", &batch(crops, images))?;
        assert_eq!(
            output.image_spans,
            vec![
                ImageTokenSpan::new(0, 2, 6),
                ImageTokenSpan::new(0, 7, 11),
                ImageTokenSpan::new(0, 12, 20),
            ]
        );
        assert_eq!(output.input_ids[output.image_spans[0].end], 8);
        assert_eq!(output.input_ids[output.image_spans[1].end], 6);
        Ok(())
    }

    #[test]
    fn disabling_special_markers_keeps_adjacent_crop_spans() -> Result<()> {
        let crops = vec![
            CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Tile { row: 0, col: 0 },
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            },
            CropMeta {
                image_index: 0,
                crop_index: 1,
                kind: CropKind::Tile { row: 0, col: 1 },
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            },
        ];
        let images = vec![ImageMeta {
            crop_range: 0..2,
            rows: 1,
            cols: 2,
            resized_width: 16,
            resized_height: 8,
        }];
        let mut processor_config = config();
        processor_config.use_thumbnail = false;
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            processor_config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let output = prompt.expand("<image>", &batch(crops, images))?;
        assert_eq!(output.image_spans[0].end, output.image_spans[1].start);
        Ok(())
    }

    #[test]
    fn image_before_text_places_each_image_in_order() -> Result<()> {
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let output = prompt.image_before_text("hello", &whole_batch(2))?;
        assert_eq!(output.image_spans.len(), 2);
        assert_eq!(output.input_ids.last(), Some(&1));
        Ok(())
    }

    #[test]
    fn prompt_revalidates_request_limits_for_external_batches() -> Result<()> {
        let mut processor_config = config();
        processor_config.vision_limits.max_images = 1;
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            processor_config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let error = prompt
            .image_before_text("hello", &whole_batch(2))
            .expect_err("prompt must enforce the image-count limit");
        assert!(error.to_string().contains("2 images"));

        let mut processor_config = config();
        processor_config.vision_limits.max_total_projected_tokens = 3;
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            processor_config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let error = prompt
            .expand("<image>", &whole_batch(1))
            .expect_err("prompt must enforce the projected-token limit");
        assert!(error.to_string().contains("4 projected tokens"));

        let mut processor_config = config();
        processor_config.vision_limits.max_source_pixels = 63;
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            processor_config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        let error = prompt
            .expand("<image>", &whole_batch(1))
            .expect_err("prompt must enforce the resized-surface limit");
        assert!(error.to_string().contains("resized image metadata"));
        Ok(())
    }

    #[test]
    fn missing_marker_and_count_mismatch_are_controlled_errors() -> Result<()> {
        let missing = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: true,
                context_length: None,
            },
        );
        assert!(missing.is_err());
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;
        assert!(prompt.expand("hello", &whole_batch(1)).is_err());
        Ok(())
    }

    #[test]
    fn empty_and_malformed_crop_metadata_are_controlled_errors() -> Result<()> {
        let mut processor_config = config();
        processor_config.use_thumbnail = false;
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            processor_config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: None,
            },
        )?;

        let inconsistent_empty = batch(
            vec![CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 4,
                patch_cols: 4,
                projected_tokens: 4,
            }],
            Vec::new(),
        );
        assert!(prompt.expand("hello", &inconsistent_empty).is_err());

        let malformed_tiles = batch(
            vec![
                CropMeta {
                    image_index: 0,
                    crop_index: 0,
                    kind: CropKind::Tile { row: 0, col: 0 },
                    patch_rows: 4,
                    patch_cols: 4,
                    projected_tokens: 4,
                },
                CropMeta {
                    image_index: 0,
                    crop_index: 1,
                    kind: CropKind::Tile {
                        row: usize::MAX,
                        col: 1,
                    },
                    patch_rows: 4,
                    patch_cols: 4,
                    projected_tokens: 4,
                },
            ],
            vec![ImageMeta {
                crop_range: 0..2,
                rows: 1,
                cols: 2,
                resized_width: 16,
                resized_height: 8,
            }],
        );
        assert!(prompt.expand("<image>", &malformed_tiles).is_err());
        Ok(())
    }

    #[test]
    fn projected_token_mismatch_and_context_overflow_are_rejected() -> Result<()> {
        let prompt = Lfm2VlPrompt::new(
            tokenizer(false),
            Some(3),
            config(),
            PromptOptions {
                use_image_special_tokens: false,
                context_length: Some(2),
            },
        )?;
        let mut images = whole_batch(1);
        images.crops[0].projected_tokens = 3;
        assert!(prompt.expand("<image>", &images).is_err());
        let images = whole_batch(1);
        assert!(prompt.expand("<image>", &images).is_err());
        Ok(())
    }

    fn fixture_prompt_batch(
        metadata: &serde_json::Value,
        prompt_name: &str,
    ) -> Result<ProcessedVisionBatch> {
        let prompt = metadata
            .get("prompt_cases")
            .and_then(|value| value.get(prompt_name))
            .ok_or_else(|| candle::Error::Msg("missing prompt fixture case".into()))?;
        let image_info = prompt
            .get("images")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| candle::Error::Msg("prompt fixture has no image metadata".into()))?;
        let mut crops = Vec::new();
        let mut images = Vec::new();
        for (image_index, info) in image_info.iter().enumerate() {
            let rows = usize::try_from(
                info.get("rows")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| candle::Error::Msg("prompt fixture row is missing".into()))?,
            )
            .map_err(|_| candle::Error::Msg("prompt fixture row is too large".into()))?;
            let cols = usize::try_from(
                info.get("cols")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| candle::Error::Msg("prompt fixture column is missing".into()))?,
            )
            .map_err(|_| candle::Error::Msg("prompt fixture column is too large".into()))?;
            let size = info
                .get("image_size")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| candle::Error::Msg("prompt fixture image size is missing".into()))?;
            let height =
                usize::try_from(size[0].as_u64().ok_or_else(|| {
                    candle::Error::Msg("prompt fixture height is invalid".into())
                })?)
                .map_err(|_| candle::Error::Msg("prompt fixture height is too large".into()))?;
            let width = usize::try_from(
                size[1]
                    .as_u64()
                    .ok_or_else(|| candle::Error::Msg("prompt fixture width is invalid".into()))?,
            )
            .map_err(|_| candle::Error::Msg("prompt fixture width is too large".into()))?;
            let crop_start = crops.len();
            if rows > 1 || cols > 1 {
                for row in 0..rows {
                    for col in 0..cols {
                        crops.push(CropMeta {
                            image_index,
                            crop_index: row * cols + col,
                            kind: CropKind::Tile { row, col },
                            patch_rows: config().tile_size / config().encoder_patch_size,
                            patch_cols: config().tile_size / config().encoder_patch_size,
                            projected_tokens: config().projected_token_count(
                                config().tile_size / config().encoder_patch_size,
                                config().tile_size / config().encoder_patch_size,
                            )?,
                        });
                    }
                }
                if config().use_thumbnail {
                    let patch_rows = height / config().encoder_patch_size;
                    let patch_cols = width / config().encoder_patch_size;
                    crops.push(CropMeta {
                        image_index,
                        crop_index: rows * cols,
                        kind: CropKind::Thumbnail,
                        patch_rows,
                        patch_cols,
                        projected_tokens: config().projected_token_count(patch_rows, patch_cols)?,
                    });
                }
            } else {
                let patch_rows = height / config().encoder_patch_size;
                let patch_cols = width / config().encoder_patch_size;
                crops.push(CropMeta {
                    image_index,
                    crop_index: 0,
                    kind: CropKind::Whole,
                    patch_rows,
                    patch_cols,
                    projected_tokens: config().projected_token_count(patch_rows, patch_cols)?,
                });
            }
            images.push(ImageMeta {
                crop_range: crop_start..crops.len(),
                rows,
                cols,
                resized_width: width,
                resized_height: height,
            });
        }
        Ok(batch(crops, images))
    }

    #[test]
    fn official_prompt_fixture_matches_exact_strings_ids_and_per_crop_spans() -> Result<()> {
        let metadata: serde_json::Value =
            serde_json::from_slice(PROCESSOR_METADATA).map_err(|err| {
                candle::Error::Msg(format!("invalid processor fixture metadata: {err}"))
            })?;
        let prompt =
            Lfm2VlPrompt::new(tokenizer(true), Some(3), config(), PromptOptions::default())?;
        for prompt_name in [
            "image_before_text",
            "image_between_text",
            "two_images",
            "images_across_turns",
            "tiled_thumbnail_prompt",
        ] {
            let fixture = &metadata["prompt_cases"][prompt_name];
            let text = fixture["text"]
                .as_str()
                .ok_or_else(|| candle::Error::Msg("prompt fixture text is invalid".into()))?;
            let batch = fixture_prompt_batch(&metadata, prompt_name)?;
            let output = prompt.expand(text, &batch)?;
            let expected_text = fixture["expanded_text"].as_str().ok_or_else(|| {
                candle::Error::Msg("prompt fixture expanded text is invalid".into())
            })?;
            assert_eq!(output.expanded_text, expected_text);
            let expected_ids = fixture["input_ids"]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("prompt fixture IDs are invalid".into()))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| candle::Error::Msg("prompt fixture ID is invalid".into()))
                })
                .collect::<Result<Vec<u32>>>()?;
            assert_eq!(output.input_ids, expected_ids);
            let expected_spans = fixture["image_spans"]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("prompt fixture spans are invalid".into()))?
                .iter()
                .map(|span| {
                    let values = span.as_array().ok_or_else(|| {
                        candle::Error::Msg("prompt fixture span is invalid".into())
                    })?;
                    Ok(ImageTokenSpan::new(
                        0,
                        usize::try_from(values[0].as_u64().ok_or_else(|| {
                            candle::Error::Msg("prompt fixture span start is invalid".into())
                        })?)
                        .map_err(|_| {
                            candle::Error::Msg("prompt fixture span start is too large".into())
                        })?,
                        usize::try_from(values[1].as_u64().ok_or_else(|| {
                            candle::Error::Msg("prompt fixture span end is invalid".into())
                        })?)
                        .map_err(|_| {
                            candle::Error::Msg("prompt fixture span end is too large".into())
                        })?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(output.image_spans, expected_spans);
        }
        Ok(())
    }
}
