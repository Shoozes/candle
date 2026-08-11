#[derive(Debug)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: Tensor,
    base_grid_side: usize,
    hidden_size: usize,
    dtype: DType,
    device: Device,
    position_cache: RwLock<HashMap<(usize, usize), Tensor>>,
}

impl VisionEmbeddings {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = linear(
            config.patch_dimension()?,
            config.hidden_size,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = vb
            .pp("position_embedding")
            .get((config.num_patches, config.hidden_size), "weight")?;
        if position_embedding.dims() != [config.num_patches, config.hidden_size] {
            candle::bail!(
                "SigLIP2 position_embedding has shape {:?}, expected [{}, {}]",
                position_embedding.dims(),
                config.num_patches,
                config.hidden_size
            )
        }
        let dtype = patch_embedding.weight().dtype();
        let device = patch_embedding.weight().device().clone();
        Ok(Self {
            patch_embedding,
            position_embedding,
            base_grid_side: config.base_grid_side()?,
            hidden_size: config.hidden_size,
            dtype,
            device,
            position_cache: RwLock::new(HashMap::new()),
        })
    }

    fn forward(
        &self,
        inputs: &PackedVisionInputs<'_>,
        shapes: &[(usize, usize)],
    ) -> Result<EmbeddingStages> {
        let pixel_values = inputs.pixel_values.to_dtype(self.dtype)?;
        let patch_embedding = self.patch_embedding.forward(&pixel_values)?;
        let resized_position_embedding =
            self.position_embeddings(shapes, patch_embedding.dim(1)?)?;
        let embeddings_with_position =
            patch_embedding.broadcast_add(&resized_position_embedding)?;
        Ok(EmbeddingStages {
            patch_embedding,
            resized_position_embedding,
            embeddings_with_position,
        })
    }

    fn position_embeddings(&self, shapes: &[(usize, usize)], max_patches: usize) -> Result<Tensor> {
        let mut per_crop = Vec::with_capacity(shapes.len());
        for &(rows, cols) in shapes {
            let valid_patches = rows
                .checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
            if valid_patches > max_patches {
                candle::bail!(
                    "SigLIP2 spatial shape [{rows}, {cols}] needs {valid_patches} patches, but max_patches is {max_patches}"
                )
            }
            let resized = self.resized_positions(rows, cols)?;
            let padded = if valid_patches == max_patches {
                resized
            } else {
                let first = resized.i(0)?.reshape((1, self.hidden_size))?;
                let padding =
                    first.broadcast_as((max_patches - valid_patches, self.hidden_size))?;
                Tensor::cat(&[&resized, &padding], 0)?
            };
            per_crop.push(padded.unsqueeze(0)?);
        }
        let per_crop: Vec<&Tensor> = per_crop.iter().collect();
        Tensor::cat(&per_crop, 0)
    }

    fn resized_positions(&self, rows: usize, cols: usize) -> Result<Tensor> {
        if let Some(cached) = self
            .position_cache
            .read()
            .map_err(|_| candle::Error::Msg("SigLIP2 position cache read lock poisoned".into()))?
            .get(&(rows, cols))
            .cloned()
        {
            return Ok(cached);
        }
        let base = self
            .position_embedding
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let resized = resize_bilinear_antialias(
            &base,
            self.base_grid_side,
            self.base_grid_side,
            rows,
            cols,
            self.hidden_size,
        )?;
        let resized = Tensor::from_vec(
            resized,
            (
                rows.checked_mul(cols).ok_or_else(|| {
                    candle::Error::Msg("SigLIP2 resized position count overflow".into())
                })?,
                self.hidden_size,
            ),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        self.position_cache
            .write()
            .map_err(|_| candle::Error::Msg("SigLIP2 position cache write lock poisoned".into()))?
            .insert((rows, cols), resized.clone());
        Ok(resized)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct EmbeddingStages {
    pub(crate) patch_embedding: Tensor,
    pub(crate) resized_position_embedding: Tensor,
    pub(crate) embeddings_with_position: Tensor,
}

fn mixed_linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    quantized_weights: &mut HashMap<String, QTensor>,
    weight_name: &str,
) -> Result<LinearOp> {
    match quantized_weights.remove(weight_name) {
        Some(weight) => {
            let bias = vb.get(out_dim, "bias")?;
            Ok(LinearOp::from_qtensor(weight, Some(bias)))
        }
        None => Ok(LinearOp::Dense(linear(in_dim, out_dim, vb)?)),
    }
}
