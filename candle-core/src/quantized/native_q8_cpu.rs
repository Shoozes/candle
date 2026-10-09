//! Q8 weight blocks multiplied directly by F32 activation rows.
use super::{GgmlDType, QuantizedType};
use crate::Result;
use rayon::prelude::*;

pub(super) fn matmul(
    storage: &dyn QuantizedType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    dst: &mut [f32],
) -> Result<()> {
    let blocks = n
        .checked_mul(k / 32)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| crate::Error::Msg("native Q8 storage size overflow".into()))?;
    if storage.dtype() != GgmlDType::Q8_0
        || !k.is_multiple_of(32)
        || k == 0
        || n == 0
        || storage.storage_size_in_bytes() != blocks
        || m.checked_mul(k) != Some(lhs.len())
        || m.checked_mul(n) != Some(dst.len())
    {
        crate::bail!("native Q8/F32 CPU dimensions or storage do not match")
    }
    // The QuantizedType pointer/length contract also backs QStorage::data.
    let weights = unsafe { std::slice::from_raw_parts(storage.as_ptr(), blocks) };
    // A single wide row otherwise occupies one Rayon chunk. Keep small projections
    // and single-worker calls on the existing path; each task owns its output columns.
    const COLUMN_TILE: usize = 256;
    const MIN_PARALLEL_BYTES: usize = 1024 * 1024;
    if m == 1 && n > COLUMN_TILE && blocks >= MIN_PARALLEL_BYTES && rayon::current_num_threads() > 1
    {
        single_row_columns(weights, k, lhs, dst, COLUMN_TILE);
        return Ok(());
    }
    row_tiles(weights, k, n, lhs, dst)
}

// Keep column scheduling separate from the existing four-row weight reuse.
#[inline(never)]
fn single_row_columns(weights: &[u8], k: usize, lhs: &[f32], dst: &mut [f32], columns: usize) {
    let column_bytes = (k / 32) * 34;
    dst.par_chunks_mut(columns)
        .enumerate()
        .for_each(|(tile, output)| {
            let first = tile * columns;
            for (offset, value) in output.iter_mut().enumerate() {
                let column = first + offset;
                let weight = &weights[column * column_bytes..(column + 1) * column_bytes];
                *value = dot(weight, lhs, 1, k)[0];
            }
        });
}

#[inline(never)]
fn row_tiles(weights: &[u8], k: usize, n: usize, lhs: &[f32], dst: &mut [f32]) -> Result<()> {
    let tile = n
        .checked_mul(4)
        .ok_or_else(|| crate::Error::Msg("native Q8 output tile overflow".into()))?;
    dst.par_chunks_mut(tile)
        .enumerate()
        .for_each(|(index, output)| {
            let rows = output.len() / n;
            let input = &lhs[index * 4 * k..index * 4 * k + rows * k];
            for column in 0..n {
                let row = &weights[column * (k / 32) * 34..(column + 1) * (k / 32) * 34];
                let values = dot(row, input, rows, k);
                for r in 0..rows {
                    output[r * n + column] = values[r];
                }
            }
        });
    Ok(())
}

fn dot(weights: &[u8], input: &[f32], rows: usize, k: usize) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Validated rows/storage bounds; runtime detection establishes ISA support.
        return unsafe {
            match rows {
                1 => avx_dot::<1>(weights, input, k),
                2 => avx_dot::<2>(weights, input, k),
                3 => avx_dot::<3>(weights, input, k),
                4 => avx_dot::<4>(weights, input, k),
                _ => unreachable!("native Q8 dot requires one to four validated rows"),
            }
        };
    }
    let mut result = [0f32; 4];
    for (block, values) in weights.chunks_exact(34).enumerate() {
        let scale = half::f16::from_bits(u16::from_le_bytes([values[0], values[1]])).to_f32();
        for i in 0..32 {
            let weight = f32::from(values[i + 2] as i8) * scale;
            for row in 0..rows {
                result[row] += input[row * k + block * 32 + i] * weight;
            }
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx_dot<const ROWS: usize>(weights: &[u8], input: &[f32], k: usize) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    for (block, values) in weights.chunks_exact(34).enumerate() {
        let scale = _mm256_set1_ps(
            half::f16::from_bits(u16::from_le_bytes([values[0], values[1]])).to_f32(),
        );
        for group in 0..4 {
            let packed = _mm_loadl_epi64(values.as_ptr().add(2 + group * 8).cast());
            let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), scale);
            for (row, sum) in sums.iter_mut().take(ROWS).enumerate() {
                let x = _mm256_loadu_ps(input.as_ptr().add(row * k + block * 32 + group * 8));
                *sum = _mm256_fmadd_ps(weight, x, *sum);
            }
        }
    }
    let mut result = [0f32; 4];
    for row in 0..ROWS {
        let mut values = [0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), sums[row]);
        result[row] = values.iter().sum();
    }
    result
}

#[cfg(test)]
mod tests;
