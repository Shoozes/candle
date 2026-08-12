pub struct InferenceRequest<'a> {
    pub backend: &'a str,
    pub model_inputs: &'a [PathBuf],
    pub prompt: &'a str,
    pub image_paths: &'a [PathBuf],
    pub max_new_tokens: usize,
    pub vision_batch_size: usize,
    pub eos_token_id: Option<u32>,
    pub timings: bool,
    pub benchmark_generation: bool,
    pub trace_output: Option<&'a Path>,
}

#[derive(Debug, Serialize)]
pub struct InferenceReport {
    pub contract: &'static str,
    pub backend: String,
    pub model_inputs: Vec<InputPathEvidence>,
    pub prompt: String,
    pub expanded_prompt: String,
    pub input_ids: Vec<u32>,
    pub image_files: Vec<ImageFileEvidence>,
    pub processed_images: Vec<ProcessedImageEvidence>,
    pub processed_crops: Vec<ProcessedCropEvidence>,
    pub image_spans: Vec<ImageSpanEvidence>,
    pub packed_tensors: PackedTensorEvidence,
    pub context_length: usize,
    pub vision_batch_size: usize,
    pub max_new_tokens: usize,
    pub eos: EosEvidence,
    pub generation: GenerationTrace,
    pub cache_reset_exact: bool,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct InputPathEvidence {
    pub path: String,
    pub kind: String,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageFileEvidence {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize)]
pub struct ProcessedImageEvidence {
    pub image_index: usize,
    pub crop_start: usize,
    pub crop_end: usize,
    pub rows: usize,
    pub cols: usize,
    pub resized_width: usize,
    pub resized_height: usize,
}

#[derive(Debug, Serialize)]
pub struct ProcessedCropEvidence {
    pub image_index: usize,
    pub crop_index: usize,
    pub kind: String,
    pub tile_row: Option<usize>,
    pub tile_col: Option<usize>,
    pub patch_rows: usize,
    pub patch_cols: usize,
    pub projected_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ImageSpanEvidence {
    pub batch_index: usize,
    pub image_index: usize,
    pub crop_index: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize)]
pub struct PackedTensorEvidence {
    pub pixel_values: Vec<usize>,
    pub pixel_attention_mask: Vec<usize>,
    pub spatial_shapes: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct EosEvidence {
    pub token_id: Option<u32>,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationTrace {
    pub prefill_logits_sha256: String,
    pub prefill_top_k: Vec<RankedToken>,
    pub steps: Vec<GenerationStep>,
    pub generated_ids: Vec<u32>,
    pub generated_tokens: Vec<Option<String>>,
    pub decoded_skip_special_tokens: String,
    pub decoded_with_special_tokens: String,
    pub stop_reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationStep {
    pub step: usize,
    pub sample_position: usize,
    pub model_input_position: Option<usize>,
    pub model_input_token_id: Option<u32>,
    pub selected_token_id: u32,
    pub selected_token: Option<String>,
    pub logits_sha256: String,
    pub top_k: Vec<RankedToken>,
    pub is_eos: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedToken {
    pub token_id: u32,
    pub token: Option<String>,
    pub logit: f32,
}

struct LogitsEvidence {
    sha256: String,
    top_k: Vec<RankedToken>,
}

struct GenerationInputs<'a> {
    tokenizer: &'a Tokenizer,
    input_ids: &'a Tensor,
    image_spans: &'a [ImageTokenSpan],
    encoded_images: Option<&'a EncodedImages>,
    prompt_len: usize,
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
}
