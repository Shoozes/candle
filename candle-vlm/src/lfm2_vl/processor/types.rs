const TARGET_RATIO_ORDER: &[(usize, usize)] = &[
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
];

#[derive(Debug)]
struct CropWork {
    image_index: usize,
    kind: CropKind,
    patch_rows: usize,
    patch_cols: usize,
    projected_tokens: usize,
    patches: Vec<f32>,
}

#[derive(Debug)]
struct ImageWork {
    crops: Vec<CropWork>,
    rows: usize,
    cols: usize,
    resized_width: usize,
    resized_height: usize,
}

#[derive(Clone, Copy, Debug)]
struct ImageBudget {
    crop_count: usize,
    projected_tokens: usize,
}

/// Processor for the raw RGB-to-packed-tensor part of LFM2.5-VL.
#[derive(Clone, Debug)]
pub struct Lfm2VlProcessor {
    config: Lfm2VlProcessorConfig,
}
