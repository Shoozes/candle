//! Tokenizer-backed LFM2-VL image-sentinel expansion and span validation.

use super::config::Lfm2VlProcessorConfig;
use super::types::{CropKind, ImageTokenSpan, ProcessedVisionBatch};
use candle::Result;
use std::collections::HashMap;
use tokenizers::{Encoding, Tokenizer};

include!("prompt/types.rs");
include!("prompt/tokens.rs");
include!("prompt/expand.rs");
include!("prompt/validation.rs");
include!("prompt/image_block.rs");
include!("prompt/helpers.rs");

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
