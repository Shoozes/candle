//! Ordered F32 dot products over the retained adjacent-nibble blocks.

use super::{
    checked_product, PackedMxfp4, BYTES_PER_BLOCK, FP4_VALUES, SCALE_BIAS, VALUES_PER_BLOCK,
};
use candle::{bail, Result};

pub(super) fn matmul_expert_row(
    packed: &PackedMxfp4,
    expert: usize,
    output: usize,
    input_width: usize,
    input: &[f32],
    result: &mut [f32],
) -> Result<()> {
    if packed.shape.first().is_none_or(|&count| expert >= count) {
        bail!("MXFP4 expert index {expert} is out of range");
    }
    if input.len() != input_width || result.len() != output {
        bail!(
            "MXFP4 row dimensions do not match packed shape {:?}",
            packed.shape
        );
    }
    if input_width == 0 || !input_width.is_multiple_of(VALUES_PER_BLOCK) {
        bail!("MXFP4 input width must be a positive multiple of {VALUES_PER_BLOCK}");
    }
    let blocks_per_row = input_width / VALUES_PER_BLOCK;
    let rows_before_expert = checked_product(expert, output, "MXFP4 expert offset")?;
    for (output_row, destination) in result.iter_mut().enumerate() {
        let start = rows_before_expert
            .checked_add(output_row)
            .and_then(|row| row.checked_mul(blocks_per_row))
            .ok_or_else(|| candle::Error::Msg("MXFP4 block index overflowed".to_string()))?;
        let end = start
            .checked_add(blocks_per_row)
            .ok_or_else(|| candle::Error::Msg("MXFP4 block span overflowed".to_string()))?;
        let byte_start = checked_product(start, BYTES_PER_BLOCK, "MXFP4 byte offset")?;
        let byte_end = checked_product(end, BYTES_PER_BLOCK, "MXFP4 byte end")?;
        let bytes = packed
            .blocks
            .get(byte_start..byte_end)
            .ok_or_else(|| candle::Error::Msg("MXFP4 blocks payload is truncated".to_string()))?;
        let scales = packed
            .scales
            .get(start..end)
            .ok_or_else(|| candle::Error::Msg("MXFP4 scales payload is truncated".to_string()))?;
        let mut sum = 0.0f32;
        for ((input_block, byte_block), &scale) in input
            .chunks_exact(VALUES_PER_BLOCK)
            .zip(bytes.chunks_exact(BYTES_PER_BLOCK))
            .zip(scales)
        {
            let multiplier = 2f32.powi(i32::from(scale) - SCALE_BIAS);
            let decoded = FP4_VALUES.map(|value| value * multiplier);
            for (pair, &byte) in input_block.chunks_exact(2).zip(byte_block) {
                let low = *decoded.get(usize::from(byte & 0x0f)).ok_or_else(|| {
                    candle::Error::Msg("MXFP4 nibble is out of range".to_string())
                })?;
                let high = *decoded.get(usize::from(byte >> 4)).ok_or_else(|| {
                    candle::Error::Msg("MXFP4 nibble is out of range".to_string())
                })?;
                for (&input_value, value) in pair.iter().zip([low, high]) {
                    sum += input_value * value;
                }
            }
        }
        *destination = sum;
    }
    Ok(())
}
