//! CPU schedules carried over from the retained LFM2-MoE diagnostics.
//! Quantized weights stay in their canonical GGUF storage.

// The reference SiLU approximation adapts llama.cpp 582eab71, ggml/src/ggml-cpu/vec.h.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use candle::{
    quantized::{GgmlDType, QMatMul},
    DType, Module, Result, Tensor, D,
};
use rayon::prelude::*;

use half::f16;
const BLOCK: usize = 256;
const Q4_BYTES: usize = 144;
const Q6_BYTES: usize = 210;

struct Q8Block {
    scale: f32,
    values: [i8; BLOCK],
    sums: [i16; BLOCK / 16],
}

impl Q8Block {
    fn from_f32(values: &[f32]) -> Self {
        let mut signed_max = 0.0f32;
        for &value in values {
            if value.abs() > signed_max.abs() {
                signed_max = value;
            }
        }
        let mut block = Self {
            scale: 0.0,
            values: [0; BLOCK],
            sums: [0; BLOCK / 16],
        };
        if signed_max == 0.0 {
            return block;
        }

        // The pinned x86 indexed path quantizes one shared activation row per token.
        let inverse_scale = -127.0f32 / signed_max;
        for (index, (&value, quant)) in values.iter().zip(&mut block.values).enumerate() {
            *quant = (inverse_scale * value).round_ties_even().min(127.0) as i8;
            block.sums[index / 16] += i16::from(*quant);
        }
        block.scale = 1.0f32 / inverse_scale;
        block
    }
}

fn scale_and_min(scales: &[u8], group: usize) -> (i32, i32) {
    let (scale, min) = if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        let low = scales[group + 4];
        (
            (low & 15) | ((scales[group - 4] >> 6) << 4),
            (low >> 4) | ((scales[group] >> 6) << 4),
        )
    };
    (i32::from(scale), i32::from(min))
}

fn dot_q4(weight: &[u8], activation: &[Q8Block], fold_block: bool) -> f32 {
    let mut scale_acc = 0.0f32;
    let mut min_acc = 0.0f32;
    for (q4, q8) in weight.chunks_exact(Q4_BYTES).zip(activation) {
        let weight_scale = f16::from_bits(u16::from_le_bytes([q4[0], q4[1]])).to_f32();
        let weight_min = f16::from_bits(u16::from_le_bytes([q4[2], q4[3]])).to_f32();
        let mut scale_term = 0i32;
        let mut min_term = 0i32;

        // Apply the traced packed schedule to each canonical Q4_K output channel.
        // The octet layout only rearranges lanes. Fold every subgroup in integers.
        for (group, activation_group) in q8.values.chunks_exact(32).enumerate() {
            let (scale, min) = scale_and_min(&q4[4..16], group);
            let q4_start = 16 + (group / 2) * 32;
            let shift = 4 * (group % 2);
            let product: i32 = q4[q4_start..q4_start + 32]
                .iter()
                .zip(activation_group)
                .map(|(&packed, &x)| i32::from((packed >> shift) & 15) * i32::from(x))
                .sum();
            scale_term += scale * product;
            let sum = i32::from(q8.sums[2 * group]) + i32::from(q8.sums[2 * group + 1]);
            min_term += min * sum;
            if !fold_block && group % 2 == 1 {
                scale_acc = (scale_term as f32).mul_add(weight_scale * q8.scale, scale_acc);
                min_acc = (min_term as f32).mul_add(weight_min * q8.scale, min_acc);
                scale_term = 0;
                min_term = 0;
            }
        }
        if fold_block {
            scale_acc = (scale_term as f32).mul_add(weight_scale * q8.scale, scale_acc);
            min_acc = (min_term as f32).mul_add(weight_min * q8.scale, min_acc);
        }
    }
    scale_acc - min_acc
}

struct WeightBlock {
    scale: f32,
    unsigned_scaled: [i16; BLOCK],
    subscales: [i8; BLOCK / 16],
}

fn decode_weight(bytes: &[u8]) -> WeightBlock {
    let scale = f16::from_bits(u16::from_le_bytes([bytes[208], bytes[209]])).to_f32();
    let subscales = std::array::from_fn(|group| bytes[192 + group] as i8);
    let unsigned_scaled = std::array::from_fn(|lane| {
        let half = lane / 128;
        let group = lane % 128 / 32;
        let slot = lane % 32;
        let low = (bytes[half * 64 + (group % 2) * 32 + slot] >> (4 * (group / 2))) & 15;
        let high = (bytes[128 + half * 32 + slot] >> (2 * group)) & 3;
        i16::from(low | (high << 4)) * i16::from(subscales[lane / 16])
    });
    WeightBlock {
        scale,
        unsigned_scaled,
        subscales,
    }
}

fn dot_tiled(weights: &[WeightBlock], activations: &[Q8Block]) -> f32 {
    let mut result = 0.0f32;
    for (weight, q8) in weights.iter().zip(activations) {
        let unsigned_dot: i32 = weight
            .unsigned_scaled
            .iter()
            .zip(&q8.values)
            .map(|(&w, &x)| i32::from(w) * i32::from(x))
            .sum();
        let offset: i32 = weight
            .subscales
            .iter()
            .zip(&q8.sums)
            .map(|(&s, &x)| i32::from(s) * i32::from(x))
            .sum();
        let dot = unsigned_dot - 32 * offset;
        // The bound MSVC tiled build rounds each scale multiplication before the F32 add.
        let weight_scaled = weight.scale * dot as f32;
        let term = weight_scaled * q8.scale;
        result += term;
    }
    result
}

fn dot_plain_avx2(weights: &[WeightBlock], activations: &[Q8Block]) -> f32 {
    let mut accum = [0.0f32; 8];
    for (weight, q8) in weights.iter().zip(activations) {
        let mut lanes = [0i32; 8];
        for (index, (w, x)) in weight
            .unsigned_scaled
            .chunks_exact(4)
            .zip(q8.values.chunks_exact(4))
            .enumerate()
        {
            let terms: i32 = w
                .iter()
                .zip(x)
                .map(|(&w, &x)| i32::from(w) * i32::from(x))
                .sum();
            lanes[index % 8] += terms;
        }
        let factor = weight.scale * q8.scale;
        for (lane, value) in accum.iter_mut().enumerate() {
            // The plain dot assigns the 32-value offset correction to one lane,
            // then applies a lane-wise F32 FMA before the final horizontal sum.
            let offset = i32::from(weight.subscales[2 * lane]) * i32::from(q8.sums[2 * lane])
                + i32::from(weight.subscales[2 * lane + 1]) * i32::from(q8.sums[2 * lane + 1]);
            let term = lanes[lane] - 32 * offset;
            *value = factor.mul_add(term as f32, *value);
        }
    }
    let a = accum[4] + accum[0];
    let b = accum[5] + accum[1];
    let c = accum[6] + accum[2];
    let d = accum[7] + accum[3];
    (a + c) + (b + d)
}

/// Dense attention uses the paired sub-block fold with a one-row GEMV tail.
/// Dense GEMM/GEMV uses GEMV for every row outside a complete group of four.
/// Indexed gate/up use the full integer fold. Indexed down selects its dot
/// from the number of rows routed to that expert.
pub(crate) enum Projection {
    Dense,
    Indexed,
    DenseGemmGemv,
}

impl Projection {
    fn q4_gemv(&self, row: usize, rows_per_sequence: usize) -> bool {
        let row = row % rows_per_sequence;
        match self {
            Self::Dense => rows_per_sequence % 4 == 1 && row == rows_per_sequence - 1,
            Self::Indexed => true,
            // The pinned repack scheduler sends every remaining row to GEMV.
            Self::DenseGemmGemv => row >= rows_per_sequence - rows_per_sequence % 4,
        }
    }
}

pub(crate) fn project_q4_gemm_gemv(
    weight: &QMatMul,
    input: &Tensor,
    fallback: impl FnOnce(&QMatMul, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    if input.device().is_cpu()
        && matches!(weight, QMatMul::QTensor(q) if q.dtype() == GgmlDType::Q4K)
    {
        project(weight, input, Projection::DenseGemmGemv)
    } else {
        fallback(weight, input)
    }
}

pub(crate) fn project(weight: &QMatMul, input: &Tensor, mode: Projection) -> Result<Tensor> {
    let QMatMul::QTensor(weight_tensor) = weight else {
        return weight.forward(input);
    };
    let dtype = weight_tensor.dtype();
    if !input.device().is_cpu() || !matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K) {
        return weight.forward(input);
    }
    let (outputs, width) = weight_tensor.shape().dims2()?;
    if input.dtype() != DType::F32
        || input.dim(D::Minus1)? != width
        || width == 0
        || !width.is_multiple_of(BLOCK)
    {
        candle::bail!("invalid CPU LFM2-MoE projection dimensions or dtype")
    }
    let data = input.flatten_all()?.to_vec1::<f32>()?;
    if data.iter().any(|v| !v.is_finite()) {
        candle::bail!("non-finite CPU LFM2-MoE projection input")
    }
    let rows = data.len() / width;
    let blocks = width / BLOCK;
    let q8: Vec<_> = data.chunks_exact(BLOCK).map(Q8Block::from_f32).collect();
    let width_bytes = blocks * dtype.type_size();
    let raw = weight_tensor.data()?;
    if outputs.checked_mul(width_bytes) != Some(raw.len()) {
        candle::bail!("invalid CPU LFM2-MoE quantized projection storage")
    }
    let count = outputs
        .checked_mul(rows)
        .ok_or_else(|| candle::Error::Msg("CPU LFM2-MoE projection output overflows".into()))?;
    if rows == 0 {
        candle::bail!("CPU LFM2-MoE projection has no rows")
    }
    let mut transposed = vec![0.0_f32; count];
    let rows_per_sequence = if input.rank() == 3 {
        input.dim(1)?
    } else {
        rows
    };
    transposed
        .par_chunks_mut(rows)
        .zip(raw.par_chunks_exact(width_bytes))
        .for_each(|(column, raw)| match dtype {
            GgmlDType::Q4K => {
                for (row, (out, activation)) in
                    column.iter_mut().zip(q8.chunks_exact(blocks)).enumerate()
                {
                    *out = dot_q4(raw, activation, mode.q4_gemv(row, rows_per_sequence));
                }
            }
            GgmlDType::Q6K => {
                let decoded: Vec<_> = raw.chunks_exact(Q6_BYTES).map(decode_weight).collect();
                let dot = if matches!(mode, Projection::Indexed) && rows < 8 {
                    dot_plain_avx2
                } else {
                    dot_tiled
                };
                for (out, activation) in column.iter_mut().zip(q8.chunks_exact(blocks)) {
                    *out = dot(&decoded, activation);
                }
            }
            _ => unreachable!("checked K-quant dtype"),
        });
    let mut shape = input.dims().to_vec();
    shape[input.rank() - 1] = outputs;
    Tensor::from_vec(transposed, (outputs, rows), input.device())?
        .t()?
        .contiguous()?
        .reshape(shape)
}

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    if !input.device().is_cpu() {
        return candle_nn::ops::rms_norm(input, weight, epsilon);
    }
    let width = input.dim(D::Minus1)?;
    if width == 0 || weight.dims() != [width] {
        candle::bail!("invalid CPU LFM2-MoE RMSNorm width")
    }
    let mut values = input.flatten_all()?.to_vec1::<f32>()?;
    let scales = weight.to_vec1::<f32>()?;
    for row in values.chunks_exact_mut(width) {
        let sum = row
            .iter()
            .map(|&value| f64::from(value * value))
            .sum::<f64>();
        let reciprocal = 1.0_f32 / (((sum / width as f64) as f32) + epsilon).sqrt();
        for (value, &scale) in row.iter_mut().zip(&scales) {
            *value = *value * reciprocal * scale;
        }
    }
    Tensor::from_vec(values, input.shape(), input.device())
}

pub(crate) fn router(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    let (experts, weight_width) = weight.dims2()?;
    if !input.device().is_cpu() || !width.is_multiple_of(8) {
        return input.matmul(&weight.t()?);
    }
    if width == 0 || experts == 0 || width != weight_width {
        candle::bail!("invalid CPU LFM2-MoE router dimensions")
    }
    let values = input.flatten_all()?.to_vec1::<f32>()?;
    let weights = weight.flatten_all()?.to_vec1::<f32>()?;
    let mut logits = vec![
        0.0_f32;
        rows.checked_mul(experts).ok_or_else(|| {
            candle::Error::Msg("CPU LFM2-MoE router output overflows".into())
        })?
    ];
    logits
        .par_chunks_mut(experts)
        .zip(values.par_chunks_exact(width))
        .for_each(|(out, row)| {
            for (value, w) in out.iter_mut().zip(weights.chunks_exact(width)) {
                let mut a = [0.0_f32; 8];
                for (x, w) in row.chunks_exact(8).zip(w.chunks_exact(8)) {
                    for j in 0..8 {
                        a[j] = w[j].mul_add(x[j], a[j]);
                    }
                }
                *value = ((a[4] + a[0]) + (a[6] + a[2])) + ((a[5] + a[1]) + (a[7] + a[3]));
            }
        });
    Tensor::from_vec(logits, (rows, experts), input.device())
}

pub(crate) fn rotary(context: usize, width: usize, base: f32) -> Result<(Tensor, Tensor)> {
    let step = base.powf(-2.0 / width as f32);
    let angles: Vec<_> = (0..context)
        .flat_map(|position| {
            let mut angle = position as f32;
            (0..width / 2).map(move |_| {
                let value = angle;
                angle *= step;
                value
            })
        })
        .collect();
    let make = |f: fn(f32) -> f32| {
        Tensor::from_vec(
            angles.iter().copied().map(f).collect::<Vec<_>>(),
            (context, width / 2),
            &candle::Device::Cpu,
        )
    };
    Ok((make(f32::cos)?, make(f32::sin)?))
}

pub(crate) fn silu(gate: &Tensor) -> Result<Tensor> {
    if !gate.device().is_cpu() {
        return candle_nn::ops::silu(gate);
    }
    let mut values = gate.flatten_all()?.to_vec1::<f32>()?;
    for x in &mut values {
        *x = silu_scalar(*x);
    }
    Tensor::from_vec(values, gate.shape(), gate.device())
}

fn silu_scalar(x: f32) -> f32 {
    let f = f32::from_bits;
    let negative = 0.0_f32 - x;
    let shift = f(0x4b400000);
    let reduced = negative.mul_add(f(0x3fb8aa3b), shift);
    let n = reduced - shift;
    if !n.is_finite() || n.abs() > 126.0 {
        return x / (1.0 + negative.exp());
    }
    let high = (-n).mul_add(f(0x3f317200), negative);
    let b = (-n).mul_add(f(0x35bfbe8e), high);
    let u = b * b;
    let lower = f(0x3e2aaf33).mul_add(b, f(0x3efffedb));
    let upper = f(0x3c072010).mul_add(b, f(0x3d2b9f17));
    let j = upper.mul_add(u, lower).mul_add(u, f(0x3f7ffff6) * b);
    let scale = f((reduced.to_bits() << 23).wrapping_add(1.0_f32.to_bits()));
    x / (1.0 + j.mul_add(scale, scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{
        quantized::{QStorage, QTensor},
        Device,
    };
    use sha2::{Digest, Sha256};
    use std::{borrow::Cow, path::Path, sync::Arc};

    #[test]
    fn q8_activation_preserves_ties_and_integer_sums() {
        let mut values = [0.0_f32; BLOCK];
        values[..5].copy_from_slice(&[-127., 0.5, 1.5, -0.5, -1.5]);
        let activation = Q8Block::from_f32(&values);
        assert_eq!(activation.scale.to_bits(), 1f32.to_bits());
        assert_eq!(&activation.values[..5], &[-127, 0, 2, 0, -2]);
        assert_eq!(activation.sums[0], -127);
        assert!(activation.sums[1..].iter().all(|&sum| sum == 0));
    }

    #[test]
    fn dense_q4_uses_gemv_for_every_tail_row() {
        for rows in 1..=8 {
            for row in 0..2 * rows {
                let in_gemm_group = (row % rows) / 4 < rows / 4;
                assert_eq!(Projection::DenseGemmGemv.q4_gemv(row, rows), !in_gemm_group);
            }
        }
    }

    #[test]
    fn indexed_q6_uses_valid_block_scaling_on_both_sides_of_group_cutoff() -> Result<()> {
        // Canonical Q6_K bytes for -32 in every lane before subgroup scaling.
        let mut raw = [0u8; Q6_BYTES];
        raw[192..208].fill(1);
        raw[208..].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        let storage = QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, GgmlDType::Q6K)?;
        let weight = QMatMul::QTensor(Arc::new(QTensor::new(storage, (1, BLOCK))?));
        for rows in [7, 8] {
            let x = Tensor::ones((rows, BLOCK), DType::F32, &Device::Cpu)?;
            let y = project(&weight, &x, Projection::Indexed)?;
            assert_eq!(y.to_vec2::<f32>()?, vec![vec![-8192.0]; rows]);
        }
        Ok(())
    }

    fn bound(root: &Path, relative: &str, expected: &str) -> Result<Vec<u8>> {
        let data = std::fs::read(root.join(relative))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&data)),
            expected,
            "{relative}"
        );
        Ok(data)
    }

    fn tensor(bytes: &[u8], shape: impl Into<candle::Shape>) -> Result<Tensor> {
        assert!(bytes.len().is_multiple_of(4));
        let data = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu)
    }

    fn exact(actual: &Tensor, bytes: &[u8]) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len() * 4, bytes.len());
        for (i, (value, reference)) in actual.iter().zip(bytes.chunks_exact(4)).enumerate() {
            assert_eq!(value.to_le_bytes(), reference, "flat position {i}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires retained 9c8006a5 operator operands, no checkpoint is loaded"]
    fn retained_layer2_cpu_schedules() -> Result<()> {
        let runs = std::env::var_os("CANDLE_MOE_RETAINED_RUNS").ok_or_else(|| {
            candle::Error::Msg("set CANDLE_MOE_RETAINED_RUNS to the evidence root".into())
        })?;
        let runs = Path::new(&runs);
        let input = bound(
            runs,
            "lfm2-moe-layer2-expert0-gate-dot-20261114-q4k/input-row.f32",
            "0ce76124dcb3685f22b77c7239727d1676df6e29039d722817af87cd2de25ba3",
        )?;
        let gate = bound(
            runs,
            "lfm2-moe-layer2-expert0-gate-dot-20261114-q4k/gate-quad.q4k",
            "164fc78c38581ab181a28a7f2233721077b9d68296c9b3d4077a34a253ae1f05",
        )?;
        let storage = QStorage::from_data(Cow::Borrowed(&gate), &Device::Cpu, GgmlDType::Q4K)?;
        let weight = QMatMul::QTensor(Arc::new(QTensor::new(storage, (4, 2048))?));
        let x = tensor(&input, (1, 2048))?;
        let got = project(&weight, &x, Projection::Indexed)?.to_vec2::<f32>()?;
        assert_eq!(got[0][0].to_bits(), 0x3ca52844);
        let dense = project(&weight, &x, Projection::Dense)?.to_vec2::<f32>()?;
        assert_eq!(dense[0][0].to_bits(), 0x3ca52844);

        let norm = bound(runs, "lfm2-moe-layer2-connected-admission-20261115-67b14101/computed/first-difference-ffn_norm.f32",
                         "1917a28aab2989f3b32586e5253b92abb390221c87cbb420c866abaf07e93eaf")?;
        let weights = bound(
            runs,
            "lfm2-moe-layer2-router-20261114-73fb1e0b/weight.f32",
            "2573a97b856fe3ebbb9586eeb642b199ed096ad766a4fd3cd19e0f2fcf848d80",
        )?;
        let logits = bound(runs, "lfm2-moe-layer2-connected-norm-20261115-67b14101/computed/first-difference-router_logits.f32",
                           "c407da71eeb551b48bc350a2d22889baf384858da44a13563c973c42ed92d362")?;
        exact(
            &router(&tensor(&norm, (37, 2048))?, &tensor(&weights, (64, 2048))?)?,
            &logits,
        )?;

        let gate = bound(
            runs,
            "lfm2-moe-layer2-successor-20261113-4df21393/reference/expert_gate.f32",
            "eef06e98ffdb7cd4e55ee9b2a114908c02182943d4d256999678f6185b98dcaf",
        )?;
        let up = bound(
            runs,
            "lfm2-moe-layer2-successor-20261113-4df21393/reference/expert_up.f32",
            "344cfb083f38a8454748ef1642adafb6f6991e13826dee6d13e38b9c7a16c36a",
        )?;
        let swiglu = bound(
            runs,
            "lfm2-moe-layer2-successor-20261113-4df21393/reference/expert_swiglu.f32",
            "05b37900e6c8a3d025c6bd7250598be972701a0342ce05578d6b6d0ab6c09448",
        )?;
        let shape = (37, 4, 1536);
        exact(
            &(silu(&tensor(&gate, shape)?)? * tensor(&up, shape)?)?,
            &swiglu,
        )
    }
}
