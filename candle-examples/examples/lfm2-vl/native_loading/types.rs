#[derive(Clone, Copy, Debug)]
pub struct NativeLoadOptions<'a> {
    pub vision_dtype: DType,
    pub text_dtype: DType,
    pub vision_device: &'a Device,
    pub text_device: &'a Device,
}

#[derive(Clone, Debug)]
pub struct NativeLoadReport {
    pub loaded_tensors: Vec<String>,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub shape_or_dtype_mismatches: Vec<String>,
    pub resolved_vision_root: String,
    pub resolved_projector_root: String,
    pub resolved_language_root: String,
    pub tied_output_resolution: String,
    pub shard_count: usize,
    pub indexed: bool,
    pub total_file_bytes: u64,
    pub vision_dtype: String,
    pub text_dtype: String,
    pub vision_device: String,
    pub text_device: String,
}

impl NativeLoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_tensors.is_empty()
            && self.unexpected_tensors.is_empty()
            && self.shape_or_dtype_mismatches.is_empty()
    }

    fn require_clean(&self) -> Result<()> {
        if !self.is_clean() {
            bail!(
                "native LFM2-VL tensor validation failed; missing={:?}, unexpected={:?}, mismatches={:?}",
                self.missing_tensors,
                self.unexpected_tensors,
                self.shape_or_dtype_mismatches
            )
        }
        Ok(())
    }
}

pub struct LoadedNative {
    pub model: Lfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
    pub report: NativeLoadReport,
    pub source_files: Vec<PathBuf>,
}
