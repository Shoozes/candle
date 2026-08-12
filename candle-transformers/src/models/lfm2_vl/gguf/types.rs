#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMmprojExecution {
    DenseCompatibility,
    Q8_0,
}

#[derive(Debug, Clone)]
pub struct GgufMmprojMetadata {
    pub general_architecture: String,
    pub projector_type: String,
    pub general_name: Option<String>,
    pub vision_layer_count: usize,
    pub image_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub source_byte_count: u64,
    pub dense_byte_count: u64,
    pub estimated_peak_byte_count: u64,
    pub tensor_dtypes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct AllocationReport {
    source_byte_count: u64,
    dense_byte_count: u64,
    estimated_peak_byte_count: u64,
}

#[derive(Debug)]
struct ParsedMetadata {
    general_name: Option<String>,
    image_size: usize,
    vision_hidden_size: usize,
    vision_intermediate_size: usize,
    vision_layer_count: usize,
    vision_head_count: usize,
    layer_norm_eps: f64,
    patch_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    downsample_factor: usize,
    text_hidden_size: usize,
    preproc: Option<(usize, usize, usize)>,
}

#[derive(Debug)]
struct ExpectedTensor {
    native_name: String,
    shape: Vec<usize>,
    patch_layout: bool,
    quantized_linear: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedExecution {
    Dense,
    Q8,
    Auto,
}
