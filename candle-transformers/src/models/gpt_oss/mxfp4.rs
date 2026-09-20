//! CPU reference support for the packed MXFP4 representation used by GPT-OSS.
//!
//! The official checkpoint stores each MXFP4 tensor as a `*.blocks` U8 tensor
//! and a matching `*.scales` U8 tensor.  A block contains 32 FP4 values in 16
//! bytes; the scale is an E8M0 exponent with bias 127.  This module keeps
//! those bytes packed and decodes individual values while doing the reference
//! expert operation.  It deliberately has no dense-weight escape hatch.

use candle::{bail, DType, Device, Result, Tensor};
use std::path::Path;

pub const VALUES_PER_BLOCK: usize = 32;
pub const BYTES_PER_BLOCK: usize = 16;
pub const SCALE_BIAS: i32 = 127;
pub const SWIGLU_ALPHA: f32 = 1.702;

/// MXFP4's lookup table, in low-nibble index order.
pub const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Packed bytes plus their logical dense shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMxfp4 {
    shape: Vec<usize>,
    blocks: Vec<u8>,
    scales: Vec<u8>,
}

impl PackedMxfp4 {
    /// Build a packed tensor from the exact on-disk blocks/scales payloads.
    pub fn from_parts(shape: &[usize], blocks: Vec<u8>, scales: Vec<u8>) -> Result<Self> {
        let (expected_blocks, expected_block_bytes) = expected_storage(shape)?;
        if blocks.len() != expected_block_bytes {
            bail!(
                "MXFP4 blocks payload has {} bytes, expected {}",
                blocks.len(),
                expected_block_bytes
            );
        }
        if scales.len() != expected_blocks {
            bail!(
                "MXFP4 scales payload has {} bytes, expected {}",
                scales.len(),
                expected_blocks
            );
        }
        for (index, &scale) in scales.iter().enumerate() {
            if scale == u8::MAX {
                bail!("MXFP4 scale at block {index} uses reserved E8M0 exponent 255");
            }
        }

        let mut stored_shape = Vec::new();
        stored_shape
            .try_reserve_exact(shape.len())
            .map_err(|error| {
                candle::Error::Msg(format!("MXFP4 shape allocation failed: {error}"))
            })?;
        stored_shape.extend_from_slice(shape);
        Ok(Self {
            shape: stored_shape,
            blocks,
            scales,
        })
    }

    /// Load matching U8 `blocks` and `scales` tensors from a safetensors file.
    ///
    /// Only the packed U8 payloads are retained.  No F32/BF16 tensor is
    /// materialized by this loader.
    pub fn from_safetensors_file(
        path: impl AsRef<Path>,
        blocks_name: &str,
        scales_name: &str,
        shape: &[usize],
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| {
            candle::Error::Msg(format!(
                "failed to read MXFP4 safetensors {:?}: {error}",
                path
            ))
        })?;
        let tensors = candle::safetensors::SliceSafetensors::new(&bytes)?;
        let blocks = tensors.load(blocks_name, &Device::Cpu)?;
        let scales = tensors.load(scales_name, &Device::Cpu)?;
        Self::from_tensors(&blocks, &scales, shape)
    }

    /// Build from Candle U8 tensors.  Both tensors must be CPU-resident.
    pub fn from_tensors(blocks: &Tensor, scales: &Tensor, shape: &[usize]) -> Result<Self> {
        if !blocks.device().is_cpu() || !scales.device().is_cpu() {
            bail!("MXFP4 packed loading currently supports CPU tensors only");
        }
        if blocks.dtype() != DType::U8 || scales.dtype() != DType::U8 {
            bail!(
                "MXFP4 blocks/scales must be U8 tensors, got {:?}/{:?}",
                blocks.dtype(),
                scales.dtype()
            );
        }
        let (expected_blocks, _) = expected_storage(shape)?;
        let mut expected_blocks_shape = shape[..shape.len() - 1].to_vec();
        expected_blocks_shape.push(shape[shape.len() - 1] / VALUES_PER_BLOCK);
        expected_blocks_shape.push(BYTES_PER_BLOCK);
        if blocks.dims() != expected_blocks_shape.as_slice() {
            bail!(
                "MXFP4 blocks shape {:?} does not match expected {:?}",
                blocks.dims(),
                expected_blocks_shape
            );
        }
        let mut expected_scales_shape = shape[..shape.len() - 1].to_vec();
        expected_scales_shape.push(shape[shape.len() - 1] / VALUES_PER_BLOCK);
        if scales.dims() != expected_scales_shape.as_slice() {
            bail!(
                "MXFP4 scales shape {:?} does not match expected {:?}",
                scales.dims(),
                expected_scales_shape
            );
        }
        let blocks = blocks.flatten_all()?.to_vec1::<u8>()?;
        let scales = scales.flatten_all()?.to_vec1::<u8>()?;
        if scales.len() != expected_blocks {
            bail!(
                "MXFP4 scales tensor has {} values, expected {}",
                scales.len(),
                expected_blocks
            );
        }
        Self::from_parts(shape, blocks, scales)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn blocks(&self) -> &[u8] {
        &self.blocks
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn resident_bytes(&self) -> usize {
        self.blocks.len() + self.scales.len()
    }

    pub fn block_count(&self) -> usize {
        self.scales.len()
    }

    /// Return whether this object still owns only the packed representation.
    pub const fn is_packed(&self) -> bool {
        true
    }

    /// Matrix multiply for a rank-2 `[output, input]` packed tensor.
    pub fn matmul(&self, input: &[f32], input_rows: usize) -> Result<Vec<f32>> {
        if self.shape.len() != 2 {
            bail!("MXFP4 matmul expects rank-2 weights, got {:?}", self.shape);
        }
        let output = self.shape[0];
        let input_width = self.shape[1];
        let expected_input = checked_product(input_rows, input_width, "MXFP4 input")?;
        if input.len() != expected_input {
            bail!(
                "MXFP4 matmul input has {} values, expected {}",
                input.len(),
                expected_input
            );
        }
        let mut result = allocate_f32(checked_product(input_rows, output, "MXFP4 output")?)?;
        for row in 0..input_rows {
            let input_row = input
                .get(row * input_width..(row + 1) * input_width)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 input row is out of bounds".to_string())
                })?;
            let output_row = result
                .get_mut(row * output..(row + 1) * output)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 output row is out of bounds".to_string())
                })?;
            self.matmul_expert_row(0, output, input_width, input_row, output_row)?;
        }
        Ok(result)
    }

    /// Matrix multiply for one expert of a rank-3 `[experts, output, input]`
    /// packed tensor.
    pub fn matmul_expert(
        &self,
        expert: usize,
        input: &[f32],
        input_rows: usize,
    ) -> Result<Vec<f32>> {
        self.check_expert_shape()?;
        let experts = self.shape[0];
        if expert >= experts {
            bail!("MXFP4 expert index {expert} is out of range for {experts} experts");
        }
        let output = self.shape[1];
        let input_width = self.shape[2];
        let expected_input = checked_product(input_rows, input_width, "MXFP4 input")?;
        if input.len() != expected_input {
            bail!(
                "MXFP4 expert input has {} values, expected {}",
                input.len(),
                expected_input
            );
        }
        let mut result = allocate_f32(checked_product(input_rows, output, "MXFP4 output")?)?;
        for row in 0..input_rows {
            let input_row = input
                .get(row * input_width..(row + 1) * input_width)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 input row is out of bounds".to_string())
                })?;
            let output_row = result
                .get_mut(row * output..(row + 1) * output)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 output row is out of bounds".to_string())
                })?;
            self.matmul_expert_row(expert, output, input_width, input_row, output_row)?;
        }
        Ok(result)
    }

    /// Matrix multiply where every input row selects one expert.
    pub fn matmul_selected(
        &self,
        input: &[f32],
        input_rows: usize,
        experts: &[usize],
    ) -> Result<Vec<f32>> {
        self.check_expert_shape()?;
        if experts.len() != input_rows {
            bail!(
                "MXFP4 selected-expert count {} does not match input rows {}",
                experts.len(),
                input_rows
            );
        }
        let output = self.shape[1];
        let input_width = self.shape[2];
        let expected_input = checked_product(input_rows, input_width, "MXFP4 input")?;
        if input.len() != expected_input {
            bail!(
                "MXFP4 selected input has {} values, expected {}",
                input.len(),
                expected_input
            );
        }
        let mut result = allocate_f32(checked_product(input_rows, output, "MXFP4 output")?)?;
        for row in 0..input_rows {
            let input_row = input
                .get(row * input_width..(row + 1) * input_width)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 input row is out of bounds".to_string())
                })?;
            let output_row = result
                .get_mut(row * output..(row + 1) * output)
                .ok_or_else(|| {
                    candle::Error::Msg("MXFP4 output row is out of bounds".to_string())
                })?;
            self.matmul_expert_row(
                *experts.get(row).ok_or_else(|| {
                    candle::Error::Msg("MXFP4 selected expert is out of bounds".to_string())
                })?,
                output,
                input_width,
                input_row,
                output_row,
            )?;
        }
        Ok(result)
    }

    fn check_expert_shape(&self) -> Result<()> {
        if self.shape.len() != 3 {
            bail!(
                "MXFP4 expert operation expects rank-3 weights, got {:?}",
                self.shape
            );
        }
        Ok(())
    }

    fn matmul_expert_row(
        &self,
        expert: usize,
        output: usize,
        input_width: usize,
        input: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        if expert >= self.shape[0] {
            bail!("MXFP4 expert index {expert} is out of range");
        }
        if input.len() != input_width || result.len() != output {
            bail!(
                "MXFP4 row dimensions do not match packed shape {:?}",
                self.shape
            );
        }
        let blocks_per_row = input_width / VALUES_PER_BLOCK;
        let rows_before_expert = checked_product(expert, output, "MXFP4 expert offset")?;
        for output_row in 0..output {
            let mut sum = 0.0f32;
            for input_index in 0..input_width {
                let block = rows_before_expert
                    .checked_add(output_row)
                    .and_then(|row| row.checked_mul(blocks_per_row))
                    .and_then(|row| row.checked_add(input_index / VALUES_PER_BLOCK))
                    .ok_or_else(|| {
                        candle::Error::Msg("MXFP4 block index overflowed".to_string())
                    })?;
                let value = self.value_at(block, input_index % VALUES_PER_BLOCK)?;
                let input_value = *input.get(input_index).ok_or_else(|| {
                    candle::Error::Msg("MXFP4 input value is out of bounds".to_string())
                })?;
                sum += input_value * value;
            }
            *result.get_mut(output_row).ok_or_else(|| {
                candle::Error::Msg("MXFP4 output value is out of bounds".to_string())
            })? = sum;
        }
        Ok(())
    }

    fn value_at(&self, block: usize, offset: usize) -> Result<f32> {
        if offset >= VALUES_PER_BLOCK {
            bail!("MXFP4 value offset {offset} exceeds block size");
        }
        let byte_index = block
            .checked_mul(BYTES_PER_BLOCK)
            .and_then(|index| index.checked_add(offset / 2))
            .ok_or_else(|| candle::Error::Msg("MXFP4 byte index overflowed".to_string()))?;
        let byte = *self
            .blocks
            .get(byte_index)
            .ok_or_else(|| candle::Error::Msg("MXFP4 blocks payload is truncated".to_string()))?;
        let nibble = if offset.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        } as usize;
        let base = *FP4_VALUES
            .get(nibble)
            .ok_or_else(|| candle::Error::Msg("MXFP4 nibble is out of range".to_string()))?;
        let exponent =
            *self.scales.get(block).ok_or_else(|| {
                candle::Error::Msg("MXFP4 scales payload is truncated".to_string())
            })? as i32
                - SCALE_BIAS;
        Ok(base * 2f32.powi(exponent))
    }
}

/// CPU reference implementation of the GPT-OSS selected-expert operation.
///
/// The two projection weights remain packed for the full call.  The only
/// temporary dense buffers are the selected token rows and their intermediate
/// activation, so this type cannot silently replace MXFP4 residency with a
/// dense model copy.
#[derive(Debug, Clone)]
pub struct Mxfp4ExpertOperation {
    mlp1: PackedMxfp4,
    mlp2: PackedMxfp4,
    mlp1_bias: Vec<f32>,
    mlp2_bias: Vec<f32>,
    swiglu_limit: f32,
}

impl Mxfp4ExpertOperation {
    pub fn new(
        mlp1: PackedMxfp4,
        mlp1_bias: Vec<f32>,
        mlp2: PackedMxfp4,
        mlp2_bias: Vec<f32>,
        swiglu_limit: f32,
    ) -> Result<Self> {
        if !swiglu_limit.is_finite() || swiglu_limit <= 0.0 {
            bail!("GPT-OSS SwiGLU limit must be finite and positive");
        }
        let mlp1_shape = mlp1.shape();
        let mlp2_shape = mlp2.shape();
        if mlp1_shape.len() != 3 || mlp2_shape.len() != 3 {
            bail!("GPT-OSS MXFP4 expert weights must both be rank 3");
        }
        let expected_mlp1_width = mlp2_shape[2]
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS MLP1 width overflowed".to_string()))?;
        if mlp1_shape[0] != mlp2_shape[0]
            || mlp1_shape[1] != expected_mlp1_width
            || mlp1_shape[2] != mlp2_shape[1]
        {
            bail!(
                "GPT-OSS MXFP4 expert shapes are incompatible: mlp1={mlp1_shape:?}, mlp2={mlp2_shape:?}"
            );
        }
        let expected_mlp1_bias = checked_product(mlp1_shape[0], mlp1_shape[1], "MLP1 bias")?;
        let expected_mlp2_bias = checked_product(mlp2_shape[0], mlp2_shape[1], "MLP2 bias")?;
        if mlp1_bias.len() != expected_mlp1_bias || mlp2_bias.len() != expected_mlp2_bias {
            bail!(
                "GPT-OSS expert bias lengths are {}/{}; expected {}/{}",
                mlp1_bias.len(),
                mlp2_bias.len(),
                expected_mlp1_bias,
                expected_mlp2_bias
            );
        }
        if mlp1_bias
            .iter()
            .chain(mlp2_bias.iter())
            .any(|value| !value.is_finite())
        {
            bail!("GPT-OSS expert biases must be finite");
        }
        Ok(Self {
            mlp1,
            mlp2,
            mlp1_bias,
            mlp2_bias,
            swiglu_limit,
        })
    }

    pub fn expert_count(&self) -> usize {
        self.mlp1.shape()[0]
    }

    pub fn hidden_size(&self) -> usize {
        self.mlp1.shape()[2]
    }

    pub fn intermediate_size(&self) -> usize {
        self.mlp2.shape()[2]
    }

    pub fn packed_resident_bytes(&self) -> usize {
        self.mlp1.resident_bytes() + self.mlp2.resident_bytes()
    }

    /// Apply the selected experts and add the residual input.
    ///
    /// `expert_indices` and `expert_weights` are flattened row-major tensors
    /// with shape `[tokens, top_k]`.  Routing is intentionally supplied by the
    /// caller in this C1a reference; gate/top-k model integration belongs to
    /// C1b.
    pub fn forward(
        &self,
        inputs: &[f32],
        token_count: usize,
        expert_indices: &[usize],
        expert_weights: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden = self.hidden_size();
        let expected_input = checked_product(token_count, hidden, "GPT-OSS expert input")?;
        if inputs.len() != expected_input {
            bail!(
                "GPT-OSS expert input has {} values, expected {}",
                inputs.len(),
                expected_input
            );
        }
        if token_count == 0 {
            bail!("GPT-OSS expert operation requires at least one token");
        }
        if expert_indices.len() != expert_weights.len()
            || !expert_indices.len().is_multiple_of(token_count)
        {
            bail!("GPT-OSS expert routing must be a rectangular token-by-top-k array");
        }
        let top_k = expert_indices.len() / token_count;
        if top_k == 0 {
            bail!("GPT-OSS expert routing must select at least one expert per token");
        }
        for (index, &weight) in expert_weights.iter().enumerate() {
            if !weight.is_finite() {
                bail!("GPT-OSS expert routing weight {index} is not finite");
            }
        }
        let intermediate = self.intermediate_size();
        let mlp1_width = intermediate
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS MLP1 width overflowed".to_string()))?;
        let mut output = allocate_f32(inputs.len())?;
        output.copy_from_slice(inputs);
        let mut mlp1_output = allocate_f32(mlp1_width)?;
        let mut activated = allocate_f32(intermediate)?;
        let mut mlp2_output = allocate_f32(hidden)?;
        for token in 0..token_count {
            let input = inputs
                .get(token * hidden..(token + 1) * hidden)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS input row is out of bounds".to_string())
                })?;
            let output_row = output
                .get_mut(token * hidden..(token + 1) * hidden)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS output row is out of bounds".to_string())
                })?;
            for slot in 0..top_k {
                let route_index = token * top_k + slot;
                let expert = *expert_indices.get(route_index).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert index is out of bounds".to_string())
                })?;
                if expert >= self.expert_count() {
                    bail!(
                        "GPT-OSS expert index {expert} is out of range for {} experts",
                        self.expert_count()
                    );
                }
                let weight = *expert_weights.get(route_index).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS expert weight is out of bounds".to_string())
                })?;
                self.mlp1.matmul_expert_row(
                    expert,
                    self.mlp1.shape()[1],
                    hidden,
                    input,
                    &mut mlp1_output,
                )?;
                let mlp1_bias_offset = expert.checked_mul(mlp1_width).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS bias offset overflowed".to_string())
                })?;
                for (index, activated_value) in activated.iter_mut().enumerate() {
                    let index_offset = index.checked_mul(2).ok_or_else(|| {
                        candle::Error::Msg("GPT-OSS MLP1 index overflowed".to_string())
                    })?;
                    let glu = mlp1_output[index_offset]
                        + *self
                            .mlp1_bias
                            .get(mlp1_bias_offset + index_offset)
                            .ok_or_else(|| {
                                candle::Error::Msg("GPT-OSS MLP1 bias is out of bounds".to_string())
                            })?;
                    let linear = mlp1_output[index_offset + 1]
                        + *self
                            .mlp1_bias
                            .get(mlp1_bias_offset + index_offset + 1)
                            .ok_or_else(|| {
                                candle::Error::Msg("GPT-OSS MLP1 bias is out of bounds".to_string())
                            })?;
                    let glu = glu.min(self.swiglu_limit);
                    let linear = linear.clamp(-self.swiglu_limit, self.swiglu_limit);
                    let gate = 1.0 / (1.0 + (-SWIGLU_ALPHA * glu).exp());
                    *activated_value = glu * gate * (linear + 1.0);
                }
                self.mlp2.matmul_expert_row(
                    expert,
                    hidden,
                    intermediate,
                    &activated,
                    &mut mlp2_output,
                )?;
                let mlp2_bias_offset = expert.checked_mul(hidden).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS bias offset overflowed".to_string())
                })?;
                for index in 0..hidden {
                    let value = *mlp2_output.get(index).ok_or_else(|| {
                        candle::Error::Msg("GPT-OSS MLP2 output is out of bounds".to_string())
                    })? + *self.mlp2_bias.get(mlp2_bias_offset + index).ok_or_else(
                        || candle::Error::Msg("GPT-OSS MLP2 bias is out of bounds".to_string()),
                    )?;
                    *output_row.get_mut(index).ok_or_else(|| {
                        candle::Error::Msg("GPT-OSS output value is out of bounds".to_string())
                    })? += weight * value;
                }
            }
        }
        Ok(output)
    }
}

fn expected_storage(shape: &[usize]) -> Result<(usize, usize)> {
    let Some(&input_width) = shape.last() else {
        bail!("MXFP4 shape cannot be empty");
    };
    if input_width == 0 || !input_width.is_multiple_of(VALUES_PER_BLOCK) {
        bail!(
            "MXFP4 logical input width {input_width} must be positive and divisible by {VALUES_PER_BLOCK}"
        );
    }
    let mut elements = 1usize;
    for (index, &dimension) in shape.iter().enumerate() {
        if dimension == 0 {
            bail!("MXFP4 shape dimension {index} is zero");
        }
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| candle::Error::Msg("MXFP4 logical shape overflowed".to_string()))?;
    }
    if !elements.is_multiple_of(VALUES_PER_BLOCK) {
        bail!("MXFP4 logical element count is not divisible by block size");
    }
    let blocks = elements / VALUES_PER_BLOCK;
    let block_bytes = blocks
        .checked_mul(BYTES_PER_BLOCK)
        .ok_or_else(|| candle::Error::Msg("MXFP4 packed byte count overflowed".to_string()))?;
    Ok((blocks, block_bytes))
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| candle::Error::Msg(format!("{label} size overflowed")))
}

fn allocate_f32(length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|error| {
        candle::Error::Msg(format!("MXFP4 temporary allocation failed: {error}"))
    })?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn reference_value(blocks: &[u8], scales: &[u8], block: usize, offset: usize) -> f32 {
        let byte = blocks[block * BYTES_PER_BLOCK + offset / 2];
        let nibble = if offset % 2 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        } as usize;
        FP4_VALUES[nibble] * 2f32.powi(scales[block] as i32 - SCALE_BIAS)
    }

    fn synthetic(shape: &[usize]) -> PackedMxfp4 {
        let elements = shape.iter().product::<usize>();
        let blocks = elements / VALUES_PER_BLOCK;
        let packed: Vec<u8> = (0..blocks * BYTES_PER_BLOCK)
            .map(|index| ((index * 13 + 3) as u8).rotate_left(1))
            .collect();
        let scales: Vec<u8> = (0..blocks).map(|index| 127 + (index % 3) as u8).collect();
        PackedMxfp4::from_parts(shape, packed, scales).unwrap()
    }

    #[test]
    fn selected_matmul_matches_independent_dense_reference() -> Result<()> {
        let packed = synthetic(&[2, 4, 32]);
        let input: Vec<f32> = (0..96).map(|index| (index as f32 - 24.0) / 37.0).collect();
        let selected = packed.matmul_selected(&input, 3, &[1, 0, 1])?;
        let mut expected = vec![0.0f32; 12];
        for row in 0..3 {
            for output in 0..4 {
                let mut sum = 0.0;
                let block = ([1usize, 0, 1][row] * 4) + output;
                for input_index in 0..32 {
                    sum += input[row * 32 + input_index]
                        * reference_value(packed.blocks(), packed.scales(), block, input_index);
                }
                expected[row * 4 + output] = sum;
            }
        }
        for (actual, expected) in selected.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn expert_operation_matches_dense_reference_for_two_routes() -> Result<()> {
        let mlp1 = synthetic(&[2, 64, 32]);
        let mlp2 = synthetic(&[2, 32, 32]);
        let mlp1_bias: Vec<f32> = (0..128).map(|index| index as f32 / 100.0).collect();
        let mlp2_bias: Vec<f32> = (0..64).map(|index| index as f32 / 200.0).collect();
        let operation = Mxfp4ExpertOperation::new(
            mlp1.clone(),
            mlp1_bias.clone(),
            mlp2.clone(),
            mlp2_bias.clone(),
            7.0,
        )?;
        let inputs: Vec<f32> = (0..64).map(|index| index as f32 / 11.0).collect();
        let actual = operation.forward(&inputs, 2, &[0, 1, 1, 0], &[0.25, 0.75, 0.6, 0.4])?;

        let mut expected = inputs.clone();
        for token in 0..2 {
            for slot in 0..2 {
                let expert = [0usize, 1, 1, 0][token * 2 + slot];
                let weight = [0.25f32, 0.75, 0.6, 0.4][token * 2 + slot];
                let mut first = vec![0.0; 64];
                for output in 0..64 {
                    let mut sum = 0.0;
                    for input_index in 0..32 {
                        let block = expert * 64 + output;
                        sum += inputs[token * 32 + input_index]
                            * reference_value(mlp1.blocks(), mlp1.scales(), block, input_index);
                    }
                    first[output] = sum + mlp1_bias[expert * 64 + output];
                }
                let activated: Vec<f32> = (0..32)
                    .map(|index| {
                        let glu = first[index * 2].min(7.0);
                        let linear = first[index * 2 + 1].clamp(-7.0, 7.0);
                        let gate = 1.0 / (1.0 + (-SWIGLU_ALPHA * glu).exp());
                        glu * gate * (linear + 1.0)
                    })
                    .collect();
                for output in 0..32 {
                    let mut sum = 0.0;
                    for input_index in 0..32 {
                        let block = expert * 32 + output;
                        sum += activated[input_index]
                            * reference_value(mlp2.blocks(), mlp2.scales(), block, input_index);
                    }
                    expected[token * 32 + output] +=
                        weight * (sum + mlp2_bias[expert * 32 + output]);
                }
            }
        }
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        assert_eq!(
            operation.packed_resident_bytes(),
            mlp1.resident_bytes() + mlp2.resident_bytes()
        );
        Ok(())
    }

    #[test]
    fn loads_packed_safetensors_without_dense_weight() -> Result<()> {
        let shape = [1, 2, 32];
        let packed = synthetic(&shape);
        let blocks = Tensor::from_slice(packed.blocks(), &[1, 2, 1, 16], &Device::Cpu)?;
        let scales = Tensor::from_slice(packed.scales(), &[1, 2, 1], &Device::Cpu)?;
        let path = std::env::temp_dir().join(format!(
            "candle-gpt-oss-mxfp4-{}-{:.0}.safetensors",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| candle::Error::Msg(error.to_string()))?
                .as_nanos() as f64
        ));
        let tensors = HashMap::from([("w.blocks", blocks), ("w.scales", scales)]);
        candle::safetensors::save(&tensors, &path)?;
        let loaded = PackedMxfp4::from_safetensors_file(&path, "w.blocks", "w.scales", &shape)?;
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded, packed);
        assert!(loaded.is_packed());
        Ok(())
    }

    #[test]
    fn rejects_truncated_invalid_and_unsupported_payloads() -> Result<()> {
        let error = PackedMxfp4::from_parts(&[1, 1, 32], vec![0; 15], vec![127])
            .expect_err("truncated blocks must fail");
        assert!(error.to_string().contains("blocks payload"));
        let error = PackedMxfp4::from_parts(&[1, 1, 32], vec![0; 16], vec![255])
            .expect_err("reserved scale must fail");
        assert!(error.to_string().contains("reserved E8M0"));
        let error = PackedMxfp4::from_parts(&[1, 1, 31], vec![0; 16], vec![127])
            .expect_err("non-block-aligned shape must fail");
        assert!(error.to_string().contains("divisible"));
        let blocks = Tensor::zeros((1, 1, 1, 16), DType::F32, &Device::Cpu)?;
        let scales = Tensor::zeros((1, 1, 1), DType::U8, &Device::Cpu)?;
        let error = PackedMxfp4::from_tensors(&blocks, &scales, &[1, 1, 32])
            .expect_err("non-U8 blocks must fail");
        assert!(error.to_string().contains("must be U8"));
        Ok(())
    }
}
