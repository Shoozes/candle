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
