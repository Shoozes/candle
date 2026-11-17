//! Reference-order F32 attention for the quantized MoE CPU mixer.

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::forward;

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn forward(
    _: &candle::Tensor,
    _: &candle::Tensor,
    _: &candle::Tensor,
    _: Option<&candle::Tensor>,
) -> candle::Result<Option<candle::Tensor>> {
    Ok(None)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use candle::{DType, Result, Tensor};
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    const DOT_STEP: usize = 4 * LANES;
    const REFERENCE_KEY_BLOCK: usize = 256;

    #[derive(Clone, Copy)]
    enum Stage {
        Scores,
        Probabilities,
    }

    pub(crate) fn forward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        forward_observing(q, k, v, mask, |_, _, _| Ok(()))
    }

    fn forward_observing(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        observe: impl Fn(Stage, usize, &[f32]) -> Result<()> + Sync,
    ) -> Result<Option<Tensor>> {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || [q, k, v]
                .iter()
                .any(|t| !t.device().is_cpu() || t.dtype() != DType::F32)
        {
            return Ok(None);
        }
        let (batch, heads, queries, width) = q.dims4()?;
        let (kv_batch, kv_heads, keys, kv_width) = k.dims4()?;
        // The pinned vector-dot path folds full 32-channel blocks.
        if width == 0 || !width.is_multiple_of(DOT_STEP) {
            return Ok(None);
        }
        if batch == 0
            || heads == 0
            || queries == 0
            || keys == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || batch != kv_batch
            || width != kv_width
            || k.shape() != v.shape()
        {
            candle::bail!("invalid quantized MoE CPU attention dimensions");
        }
        if mask.is_some_and(|mask| {
            mask.dims() != [queries, keys] || mask.dtype() != DType::U8 || !mask.device().is_cpu()
        }) {
            return Ok(None);
        }
        let padded = keys
            .checked_next_multiple_of(LANES)
            .ok_or_else(|| candle::Error::Msg("quantized MoE attention row overflows".into()))?;
        let mask = mask
            .map(|mask| mask.flatten_all()?.to_vec1::<u8>())
            .transpose()?;
        let device = q.device();
        let q = q.flatten_all()?.to_vec1::<f32>()?;
        let k = k.flatten_all()?.to_vec1::<f32>()?;
        let v = v.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0.0; q.len()];
        let scale = (width as f32).sqrt().recip();
        let groups = heads / kv_heads;
        output
            .par_chunks_exact_mut(width)
            .zip(q.par_chunks_exact(width))
            .enumerate()
            .try_for_each_init(
                || vec![0.0f32; padded],
                |scores, (row, (dst, q))| -> Result<()> {
                    let (batch_head, query) = (row / queries, row % queries);
                    let kv_offset = batch_head / groups * keys * width;
                    let k = &k[kv_offset..kv_offset + keys * width];
                    let v = &v[kv_offset..kv_offset + keys * width];
                    let mask = mask.as_ref().map(|m| &m[query * keys..(query + 1) * keys]);

                    // FMA/AVX2 presence was checked above. Each query/key is width
                    // floats, and width is a multiple of the 32-float dot step.
                    for (key, (slot, k)) in scores.iter_mut().zip(k.chunks_exact(width)).enumerate()
                    {
                        *slot = if mask.is_some_and(|m| m[key] != 0) {
                            f32::NEG_INFINITY
                        } else {
                            unsafe { score_dot(q, k) }
                        };
                    }
                    observe(Stage::Scores, row, &scores[..keys])?;
                    let (visible, padding) = scores.split_at_mut(keys);
                    for score in visible {
                        *score = *score * scale + 0.0;
                    }
                    padding.fill(f32::NEG_INFINITY);
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if !max.is_finite() {
                        candle::bail!("quantized MoE attention row {row} has no finite score");
                    }
                    // Remaining full GGML padding groups contribute positive zero
                    // to the exponent sum and need no allocation.
                    let sum = unsafe { exponents(scores, max) };
                    if !sum.is_finite() || sum <= 0.0 {
                        candle::bail!("invalid quantized MoE attention denominator at row {row}");
                    }
                    let inverse = (1.0f64 / sum) as f32;
                    for score in scores.iter_mut() {
                        *score *= inverse;
                    }
                    observe(Stage::Probabilities, row, scores)?;
                    unsafe { reduce_values(&scores[..keys], v, dst) };
                    Ok(())
                },
            )?;
        Ok(Some(Tensor::from_vec(
            output,
            (batch, heads, queries, width),
            device,
        )?))
    }

    // ggml_vec_dot_f32 uses four vector accumulators followed by the
    // GGML_F32x8_REDUCE fold and two horizontal pair adds.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn score_dot(q: &[f32], k: &[f32]) -> f32 {
        let mut accum = [_mm256_setzero_ps(); 4];
        for (q, k) in q.chunks_exact(DOT_STEP).zip(k.chunks_exact(DOT_STEP)) {
            for (i, acc) in accum.iter_mut().enumerate() {
                let q = _mm256_loadu_ps(q.as_ptr().add(i * LANES));
                let k = _mm256_loadu_ps(k.as_ptr().add(i * LANES));
                *acc = _mm256_fmadd_ps(q, k, *acc);
            }
        }
        let sum = _mm256_add_ps(
            _mm256_add_ps(accum[0], accum[2]),
            _mm256_add_ps(accum[1], accum[3]),
        );
        let pair = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
        let pair = _mm_hadd_ps(pair, pair);
        _mm_cvtss_f32(_mm_hadd_ps(pair, pair))
    }

    // ggml_vec_soft_max_f32 reduces each vector in F32 and sums subgroups in F64.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exponents(row: &mut [f32], max: f32) -> f64 {
        let mut sum = 0.0f64;
        for block in row.chunks_exact_mut(LANES) {
            let centered = _mm256_sub_ps(_mm256_loadu_ps(block.as_ptr()), _mm256_set1_ps(max));
            let values = reference_exp(centered);
            _mm256_storeu_ps(block.as_mut_ptr(), values);
            let pair = _mm_add_ps(
                _mm256_extractf128_ps::<1>(values),
                _mm256_castps256_ps128(values),
            );
            let half = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
            sum += f64::from(_mm_cvtss_f32(_mm_add_ss(half, _mm_movehdup_ps(half))));
        }
        sum
    }

    // Vectorize over output channels. Each element of each vector follows the
    // pinned llamafile key-modulo-eight FMA and final horizontal reduction order.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn reduce_values(probabilities: &[f32], values: &[f32], output: &mut [f32]) {
        let width = output.len();
        let keys = probabilities.len();
        let zero = _mm256_setzero_ps();
        let virtual_zeros =
            ((REFERENCE_KEY_BLOCK - keys % REFERENCE_KEY_BLOCK) % REFERENCE_KEY_BLOCK).min(LANES);
        for (block, out) in output.chunks_exact_mut(LANES).enumerate() {
            let mut lanes = [zero; LANES];
            for (key, (&prob, value)) in probabilities
                .iter()
                .zip(values.chunks_exact(width))
                .enumerate()
            {
                let lane = key % LANES;
                let value = _mm256_loadu_ps(value.as_ptr().add(block * LANES));
                lanes[lane] = _mm256_fmadd_ps(_mm256_set1_ps(prob), value, lanes[lane]);
            }
            // One positive zero per padded lane has the same signed-zero effect
            // as the full sequence of padded zero-valued FMA terms.
            for key in keys..keys + virtual_zeros {
                lanes[key % LANES] = _mm256_add_ps(lanes[key % LANES], zero);
            }
            let a = _mm256_add_ps(lanes[4], lanes[0]);
            let b = _mm256_add_ps(lanes[5], lanes[1]);
            let c = _mm256_add_ps(lanes[6], lanes[2]);
            let d = _mm256_add_ps(lanes[7], lanes[3]);
            _mm256_storeu_ps(
                out.as_mut_ptr(),
                _mm256_add_ps(_mm256_add_ps(a, c), _mm256_add_ps(b, d)),
            );
        }
    }

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
    // Adapted from ggml/src/ggml-cpu/vec.h and vec.cpp at
    // 582eab718608b99792286729890aac1675547c29, MIT, copyright The ggml authors.
    // The ggml exponent is adapted from an Arm Limited optimized routine.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn reference_exp(x: __m256) -> __m256 {
        let r = _mm256_set1_ps(f32::from_bits(0x4b400000));
        let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8aa3b)), r);
        let n = _mm256_sub_ps(z, r);
        let b = _mm256_fnmadd_ps(
            n,
            _mm256_set1_ps(f32::from_bits(0x35bfbe8e)),
            _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f317200)), x),
        );
        let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
        let k = _mm256_castsi256_ps(_mm256_add_epi32(
            e,
            _mm256_set1_epi32(1.0f32.to_bits() as i32),
        ));
        let abs_n = _mm256_andnot_ps(_mm256_set1_ps(-0.0), n);
        let exceptional = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(126.0));
        let u = _mm256_mul_ps(b, b);
        let p_hi = _mm256_fmadd_ps(
            _mm256_set1_ps(f32::from_bits(0x3c072010)),
            b,
            _mm256_set1_ps(f32::from_bits(0x3d2b9f17)),
        );
        let p_lo = _mm256_fmadd_ps(
            _mm256_set1_ps(f32::from_bits(0x3e2aaf33)),
            b,
            _mm256_set1_ps(f32::from_bits(0x3efffedb)),
        );
        let j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(p_hi, u, p_lo),
            u,
            _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7ffff6)), b),
        );
        if _mm256_movemask_ps(exceptional) == 0 {
            return _mm256_fmadd_ps(j, k, k);
        }

        let negative = _mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps());
        let g = _mm256_and_si256(
            _mm256_castps_si256(negative),
            _mm256_set1_epi32(0x8200_0000u32 as i32),
        );
        let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
        let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        let saturated = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(192.0));
        let finite = _mm256_or_ps(
            _mm256_and_ps(exceptional, _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
            _mm256_andnot_ps(exceptional, _mm256_fmadd_ps(k, j, k)),
        );
        _mm256_or_ps(
            _mm256_and_ps(saturated, _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(saturated, finite),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle::Device;
        use sha2::{Digest, Sha256};
        use std::{fs, path::Path};

        fn bits(x: &Tensor) -> Result<Vec<u32>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        }

        #[test]
        fn rectangular_prefill_and_cached_decode_keep_causal_grouped_attention() -> Result<()> {
            let device = Device::Cpu;
            let shape = |heads| (2, heads, 5, 64);
            let tensor = |heads, seed| {
                let data = (0..2 * heads * 5 * 64)
                    .map(|i| (((i * 71 + seed) % 127) as f32 - 63.0) / 128.0)
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, shape(heads), &device)
            };
            let q = tensor(4, 3)?;
            let k = tensor(2, 7)?;
            let v = tensor(2, 11)?;
            let full_mask = crate::utils::build_causal_mask(5, 0, &device)?;
            let Some(full) = forward(&q, &k, &v, Some(&full_mask))? else {
                return Ok(());
            };
            let chunk_mask = crate::utils::build_causal_mask(2, 3, &device)?;
            let chunk = forward(&q.narrow(2, 3, 2)?, &k, &v, Some(&chunk_mask))?.unwrap();
            assert_eq!(bits(&chunk)?, bits(&full.narrow(2, 3, 2)?)?);
            let decode = forward(&q.narrow(2, 4, 1)?, &k, &v, None)?.unwrap();
            assert_eq!(bits(&decode)?, bits(&full.narrow(2, 4, 1)?)?);

            let masked = Tensor::from_vec(vec![1u8; 25], (5, 5), &device)?;
            assert!(forward(&q, &k, &v, Some(&masked)).is_err());
            Ok(())
        }

        #[test]
        #[ignore = "requires the bound local Unicode attention packet"]
        fn unicode_attention_frozen() -> Result<()> {
            let manifest = std::env::var_os("CANDLE_MOE_ATTENTION_PROOF")
                .expect("set CANDLE_MOE_ATTENTION_PROOF to the connected replay inputs.json");
            let manifest = fs::read(manifest)?;
            assert_eq!(
                format!("{:x}", Sha256::digest(&manifest)),
                "78f20cafd58ac5688bea5e01c563816e98500b4ed4ea25634d4dcf8b757138da"
            );
            let doc: serde_json::Value =
                serde_json::from_slice(&manifest).map_err(candle::Error::wrap)?;
            let read = |key: &str, count| -> Result<Vec<u32>> {
                let entry = &doc["inputs"][key];
                let data = fs::read(Path::new(entry["path"].as_str().unwrap()))?;
                assert_eq!(data.len(), count * 4);
                assert_eq!(
                    format!("{:x}", Sha256::digest(&data)),
                    entry["sha256"].as_str().unwrap()
                );
                Ok(data
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect())
            };
            let device = Device::Cpu;
            let tensor = |key, heads| -> Result<Tensor> {
                let data: Vec<_> = read(key, 32 * heads * 64)?
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                Tensor::from_vec(data, (1, 32, heads, 64), &device)?.transpose(1, 2)
            };
            let q = tensor("q_rope", 32)?;
            let k = tensor("k_rope", 8)?;
            let v = tensor("v", 8)?;
            assert_eq!(read("positions", 32)?, (0..32).collect::<Vec<_>>());
            let mask = crate::utils::build_causal_mask(32, 0, &device)?;
            let scores = read("qk", 16896)?;
            let probabilities = read("softmax", 16896)?;
            let out = forward_observing(&q, &k, &v, Some(&mask), |stage, row, actual| {
                let head = row / 32;
                let token = row % 32;
                let start = head * 528 + token * (token + 1) / 2;
                let expected = match stage {
                    Stage::Scores => &scores,
                    Stage::Probabilities => &probabilities,
                };
                for (key, actual) in actual[..=token].iter().enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected[start + key],
                        "head {head}, token {token}, key {key}"
                    );
                }
                if matches!(stage, Stage::Probabilities) {
                    assert!(actual[token + 1..].iter().all(|p| p.to_bits() == 0));
                }
                Ok(())
            })?
            .expect("the frozen regression requires AVX2/FMA");
            let actual = bits(&out.transpose(1, 2)?.contiguous()?)?;
            assert_eq!(actual, read("values", 65536)?);
            assert_eq!(actual[2048 + 8], 0xbac911ed);
            Ok(())
        }
    }
}
