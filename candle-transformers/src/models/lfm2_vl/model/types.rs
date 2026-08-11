#[derive(Clone, Debug)]
pub enum CropKind {
    Whole,
    Tile { row: usize, col: usize },
    Thumbnail,
}

#[derive(Clone, Debug)]
pub struct CropMeta {
    pub image_index: usize,
    pub crop_index: usize,
    pub kind: CropKind,
    pub patch_rows: usize,
    pub patch_cols: usize,
    pub projected_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct ImageMeta {
    pub crop_range: Range<usize>,
    pub rows: usize,
    pub cols: usize,
    pub resized_width: usize,
    pub resized_height: usize,
}

#[derive(Debug)]
pub struct ProcessedVisionBatch {
    pub pixel_values: Tensor,
    pub pixel_attention_mask: Tensor,
    pub spatial_shapes: Tensor,
    pub crops: Vec<CropMeta>,
    pub images: Vec<ImageMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageTokenSpan {
    /// Spans are ordered one-per-crop, including a thumbnail crop.
    pub batch_index: usize,
    pub start: usize,
    pub end: usize,
}

impl ImageTokenSpan {
    pub fn new(batch_index: usize, start: usize, end: usize) -> Self {
        Self {
            batch_index,
            start,
            end,
        }
    }

    pub fn len(&self) -> Option<usize> {
        self.end.checked_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

#[derive(Debug)]
pub struct EncodedImages {
    /// `[total_projected_tokens, text_hidden]` in crop/image order.
    pub embeddings: Tensor,
    pub per_image_ranges: Vec<Range<usize>>,
    pub per_crop_ranges: Vec<Range<usize>>,
}

/// Vision and projector tensors captured by the bounded native parity trace.
///
/// The trace API is deliberately separate from ordinary inference so callers
/// cannot accidentally retain all intermediate activations during production
/// generation. The example trace lane currently accepts one crop at a time;
/// this keeps its peak memory bounded while matching the first production
/// parity checkpoint's deterministic single-image input contract.
#[derive(Debug)]
pub struct Lfm2VlImageTrace {
    pub vision_patch_embedding: Tensor,
    pub vision_resized_position_embedding: Tensor,
    pub vision_embeddings_with_position: Tensor,
    pub vision_encoder_layers: Vec<Tensor>,
    pub vision_last_hidden_state: Tensor,
    pub projector: Lfm2VlProjectorTrace,
}

#[derive(Debug)]
pub struct Lfm2VlProjectorTrace {
    pub input: Tensor,
    pub pixel_unshuffle: Tensor,
    pub layer_norm: Option<Tensor>,
    pub linear_1: Tensor,
    pub activation: Tensor,
    pub linear_2: Tensor,
    pub output: Tensor,
}

#[derive(Debug)]
pub struct Lfm2VlPrefillTrace {
    pub input_embeddings: Tensor,
    pub merged_embeddings: Tensor,
    pub hidden_states: Tensor,
    pub logits: Tensor,
}

#[derive(Debug)]
pub struct Lfm2VlDecodeTrace {
    pub input_embeddings: Tensor,
    pub hidden_states: Tensor,
    pub logits: Tensor,
}

#[derive(Debug)]
