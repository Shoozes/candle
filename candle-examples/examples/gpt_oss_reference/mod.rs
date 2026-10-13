//! Shared decoder and unchanged numerical rules for the pinned short oracle.
use candle::{bail, Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
pub struct Fixture {
    pub fixture_id: String,
    pub prompt: String,
    pub tokenizer_add_bos: bool,
    pub tokenizer_add_eos: bool,
    pub token_ids: Vec<u32>,
    pub reference_prefix_token_ids: Vec<u32>,
    pub reference_context: usize,
    pub reference_vocab_size: usize,
    pub reference_rows: usize,
    pub reference_file_bytes: usize,
    pub reference_file_sha256: String,
    pub reference_executable: String,
    pub reference_command: String,
    pub reference_row_top1: Vec<ReferenceTop1>,
    pub comparison: Comparison,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ReferenceTop1 {
    pub token_id: u32,
    pub log_prob: f32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Comparison {
    pub top_k: usize,
    pub top_k_log_probability_abs_tolerance: f32,
    pub mean_abs_log_probability_tolerance: f32,
    pub total_variation_probability_tolerance: f32,
    pub tolerance_rationale: String,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct StageReport {
    pub stage: String,
    pub max_abs_log_probability_error: f32,
    pub mean_abs_log_probability_error: f32,
    pub top_k_max_abs_log_probability_error: f32,
    pub total_variation_probability_error: f32,
    pub top_reference: Vec<TopLogProbability>,
    pub top_candle: Vec<TopLogProbability>,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct TopLogProbability {
    pub token_id: usize,
    pub log_probability: f32,
}

pub fn pinned_fixture() -> Result<Fixture> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/gpt_oss_task3_short_parity/reference.json"
    ))
    .map_err(Error::wrap)?;
    if fixture.reference_rows != 3
        || fixture.reference_context != 8
        || fixture.reference_prefix_token_ids.len() != fixture.reference_context
        || !fixture
            .token_ids
            .starts_with(&fixture.reference_prefix_token_ids)
        || fixture.reference_row_top1.len() != fixture.reference_rows
        || fixture.reference_vocab_size == 0
    {
        bail!("pinned short-reference workload is invalid");
    }
    Ok(fixture)
}

pub fn load_reference(path: &Path, fixture: &Fixture) -> Result<Vec<Vec<f32>>> {
    if fs::metadata(path)?.len() != fixture.reference_file_bytes as u64 {
        bail!("reference identity size mismatch");
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(fixture.reference_file_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() != fixture.reference_file_bytes
        || format!("{:x}", Sha256::digest(&bytes)) != fixture.reference_file_sha256
    {
        bail!("reference identity SHA-256 mismatch");
    }
    let rows = decode_reference_bytes(
        &bytes,
        &fixture.reference_prefix_token_ids,
        fixture.reference_context,
        fixture.reference_vocab_size,
        fixture.reference_rows,
    )?;
    for (row, expected) in rows.iter().zip(&fixture.reference_row_top1) {
        if top_log_probabilities(row, 1).first().map(|x| x.token_id)
            != Some(expected.token_id as usize)
        {
            bail!("reference top-1 identity mismatch");
        }
    }
    Ok(rows)
}

pub fn fixed_prefixes(fixture: &Fixture) -> Result<Vec<Vec<u32>>> {
    let start = fixture
        .reference_context
        .checked_sub(fixture.reference_rows)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Msg("reference prefix is invalid".into()))?;
    (0..fixture.reference_rows)
        .map(|index| {
            fixture
                .reference_prefix_token_ids
                .get(..start + index)
                .map(<[u32]>::to_vec)
                .ok_or_else(|| Error::Msg("reference token prefix is missing".into()))
        })
        .collect()
}

pub fn prefix_sha256(prefix: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in prefix {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CapturedRow {
    pub file: String,
    pub segment: String,
    pub row_index: usize,
    pub input_prefix_token_ids: Vec<u32>,
    pub input_prefix_sha256: String,
    pub processed_tokens: usize,
    pub vocabulary_size: usize,
    pub dtype: String,
    pub source_commit: String,
    pub model_sha256: String,
    pub size_bytes: usize,
    pub sha256: String,
}

impl CapturedRow {
    pub fn write(
        directory: &Path,
        values: &[f32],
        prefix: &[u32],
        segment: &str,
        row_index: usize,
        source: &str,
        model: &str,
    ) -> Result<Self> {
        if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
            bail!("captured row is missing or nonfinite");
        }
        if !matches!(segment, "numerical" | "replay") || row_index >= 3 {
            bail!("captured row position is invalid");
        }
        let size_bytes = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Msg("row byte count overflowed".into()))?;
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let file = format!("{segment}-{row_index}.f32");
        use std::io::Write;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(&file))?;
        output.write_all(&bytes)?;
        output.flush()?;
        Ok(Self {
            file,
            segment: segment.into(),
            row_index,
            input_prefix_token_ids: prefix.into(),
            input_prefix_sha256: prefix_sha256(prefix),
            processed_tokens: prefix.len(),
            vocabulary_size: values.len(),
            dtype: "f32-le".into(),
            source_commit: source.into(),
            model_sha256: model.into(),
            size_bytes,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    pub fn read_bound(
        &self,
        directory: &Path,
        prefix: &[u32],
        segment: &str,
        row_index: usize,
        vocabulary: usize,
        identity: (&str, &str),
    ) -> Result<Vec<f32>> {
        let (source, model) = identity;
        let size = vocabulary
            .checked_mul(4)
            .ok_or_else(|| Error::Msg("row byte count overflowed".into()))?;
        if self.file != format!("{segment}-{row_index}.f32")
            || self.segment != segment
            || self.row_index != row_index
            || self.processed_tokens != prefix.len()
            || self.input_prefix_token_ids != prefix
            || self.input_prefix_sha256 != prefix_sha256(prefix)
            || self.vocabulary_size != vocabulary
            || self.dtype != "f32-le"
            || self.source_commit != source
            || self.model_sha256 != model
            || self.size_bytes != size
        {
            bail!("captured row prefix/position/source/model/dtype binding mismatch");
        }
        let path = directory.join(&self.file);
        if fs::metadata(&path)?.len() != size as u64 {
            bail!("captured row is truncated or oversized");
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(size as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() != size || format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            bail!("captured row digest mismatch");
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                chunk
                    .try_into()
                    .map(f32::from_le_bytes)
                    .map_err(Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|x| !x.is_finite()) {
            bail!("captured row is nonfinite");
        }
        Ok(values)
    }
}

fn reference_size(tokens: &[u32], context: usize, vocab: usize, rows: usize) -> Result<usize> {
    if context == 0 || vocab == 0 || rows == 0 || tokens.len() != context || rows >= context {
        bail!("reference dimensions are invalid");
    }
    let row_bytes = vocab
        .div_ceil(2)
        .checked_mul(4)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| Error::Msg("reference row byte count overflowed".into()))?;
    context
        .checked_mul(4)
        .and_then(|n| n.checked_add(20))
        .and_then(|n| rows.checked_mul(row_bytes).and_then(|r| n.checked_add(r)))
        .ok_or_else(|| Error::Msg("reference byte count overflowed".into()))
}

pub fn read_reference_logits(
    path: &Path,
    expected_tokens: &[u32],
    expected_context: usize,
    expected_vocab: usize,
    expected_rows: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected_bytes = reference_size(
        expected_tokens,
        expected_context,
        expected_vocab,
        expected_rows,
    )?;
    if fs::metadata(path)?.len() != expected_bytes as u64 {
        bail!("reference logits file size mismatch");
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(expected_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    decode_reference_bytes(
        &bytes,
        expected_tokens,
        expected_context,
        expected_vocab,
        expected_rows,
    )
}

pub fn decode_reference_bytes(
    bytes: &[u8],
    expected_tokens: &[u32],
    expected_context: usize,
    expected_vocab: usize,
    expected_rows: usize,
) -> Result<Vec<Vec<f32>>> {
    if bytes.len()
        != reference_size(
            expected_tokens,
            expected_context,
            expected_vocab,
            expected_rows,
        )?
    {
        bail!("reference logits file is truncated or has trailing bytes");
    }
    let mut offset = 0usize;
    let magic = take(bytes, &mut offset, 8)?;
    if magic != b"_logits_" {
        return Err(Error::Msg(
            "reference logits magic does not match _logits_".into(),
        ));
    }
    let context = read_i32(bytes, &mut offset)?;
    let vocab = read_i32(bytes, &mut offset)?;
    let chunks = read_i32(bytes, &mut offset)?;
    if usize::try_from(context).ok() != Some(expected_context)
        || usize::try_from(vocab).ok() != Some(expected_vocab)
        || chunks != 1
    {
        return Err(Error::Msg(format!(
            "reference logits header mismatch: context={context} vocab={vocab} chunks={chunks}"
        )));
    }
    let token_bytes = expected_tokens
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::Msg("reference token byte count overflowed".into()))?;
    let token_data = take(bytes, &mut offset, token_bytes)?;
    let actual_tokens = token_data
        .chunks_exact(4)
        .map(|chunk| {
            chunk
                .try_into()
                .map(u32::from_le_bytes)
                .map_err(Error::wrap)
        })
        .collect::<Result<Vec<_>>>()?;
    if actual_tokens != expected_tokens {
        return Err(Error::Msg(format!(
            "reference token IDs differ: actual={actual_tokens:?}, expected={expected_tokens:?}"
        )));
    }
    let vocab = expected_vocab;
    let nv = vocab
        .div_ceil(2)
        .checked_mul(2)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| Error::Msg("reference row width overflowed".into()))?;
    let row_bytes = nv
        .checked_mul(2)
        .ok_or_else(|| Error::Msg("reference row byte count overflowed".into()))?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(expected_rows)
        .map_err(|error| Error::Msg(format!("reference row allocation failed: {error}")))?;
    for _ in 0..expected_rows {
        let row = take(bytes, &mut offset, row_bytes)?;
        let scale = f32::from_le_bytes(row[..4].try_into().map_err(Error::wrap)?);
        let minimum = f32::from_le_bytes(row[4..8].try_into().map_err(Error::wrap)?);
        if !scale.is_finite() || scale <= 0. || !minimum.is_finite() {
            bail!("reference quantization parameters are invalid");
        }
        let values = row[8..]
            .chunks_exact(2)
            .take(vocab)
            .map(|chunk| {
                chunk
                    .try_into()
                    .map(u16::from_le_bytes)
                    .map(|value| minimum + scale * value as f32)
                    .map_err(Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|value| !value.is_finite()) {
            bail!("reference row is nonfinite");
        }
        rows.push(values);
    }
    if offset != bytes.len() {
        return Err(Error::Msg(format!(
            "reference logits has {} trailing bytes",
            bytes.len() - offset
        )));
    }
    Ok(rows)
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::Msg("reference byte range overflowed".into()))?;
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| Error::Msg("reference logits file is truncated".into()))?;
    *offset = end;
    Ok(slice)
}

fn read_i32(bytes: &[u8], offset: &mut usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        take(bytes, offset, 4)?.try_into().map_err(Error::wrap)?,
    ))
}

pub fn compare_stage(
    stage: &str,
    actual: &[f32],
    expected: &[f32],
    comparison: &Comparison,
) -> Result<StageReport> {
    if comparison.top_k == 0
        || comparison.top_k > actual.len()
        || [
            comparison.top_k_log_probability_abs_tolerance,
            comparison.mean_abs_log_probability_tolerance,
            comparison.total_variation_probability_tolerance,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.)
    {
        bail!("parity comparison bounds are invalid");
    }
    if actual.len() != expected.len() {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: output length {} != {} earliest_boundary=output_logits",
            actual.len(),
            expected.len()
        )));
    }
    for value in actual.iter().chain(expected) {
        if !value.is_finite() {
            return Err(Error::Msg(format!(
                "parity mismatch at {stage}: non-finite output earliest_boundary=output_logits"
            )));
        }
    }
    let actual_log_probabilities = log_probabilities(actual)?;
    let reference_floor = expected
        .iter()
        .copied()
        .reduce(f32::min)
        .ok_or_else(|| Error::Msg("reference log-probability row is empty".into()))?;
    if !reference_floor.is_finite() {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: reference clamp floor is non-finite earliest_boundary=output_logits"
        )));
    }
    let projected_log_probabilities = actual_log_probabilities
        .iter()
        .map(|value| value.max(reference_floor))
        .collect::<Vec<_>>();
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    let mut max_token_id = 0usize;
    for (index, (actual, expected)) in projected_log_probabilities.iter().zip(expected).enumerate()
    {
        let error = (actual - expected).abs();
        sum += error;
        if error > max {
            max = error;
            max_token_id = index;
        }
    }
    let top_reference = top_log_probabilities(expected, comparison.top_k);
    let top_candle = top_log_probabilities(&projected_log_probabilities, comparison.top_k);
    let mut top_union = BTreeSet::new();
    top_union.extend(top_reference.iter().map(|entry| entry.token_id));
    top_union.extend(top_candle.iter().map(|entry| entry.token_id));
    let top_k_max = top_union
        .iter()
        .map(|&token_id| (projected_log_probabilities[token_id] - expected[token_id]).abs())
        .fold(0.0f32, f32::max);
    let normalized_reference = log_probabilities(expected)?;
    let normalized_candle = log_probabilities(&projected_log_probabilities)?;
    let total_variation = normalized_reference
        .iter()
        .zip(&normalized_candle)
        .map(|(reference, candle)| (reference.exp() - candle.exp()).abs())
        .sum::<f32>()
        * 0.5;
    let top1_matches = top_reference.first().map(|entry| entry.token_id)
        == top_candle.first().map(|entry| entry.token_id);
    let mean = sum / actual.len() as f32;
    if !top1_matches
        || top_k_max > comparison.top_k_log_probability_abs_tolerance
        || mean > comparison.mean_abs_log_probability_tolerance
        || total_variation > comparison.total_variation_probability_tolerance
    {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: max_abs_log_probability_error={max}, mean_abs_log_probability_error={mean}, top_k_max_abs_log_probability_error={top_k_max}, total_variation_probability_error={total_variation}, top_k_tolerance={}, mean_tolerance={}, total_variation_tolerance={}, max_error_token_id={max_token_id}, candle_log_probability={}, reference_log_probability={}, reference_floor={reference_floor}, reference_top_k={top_reference:?}, candle_top_k={top_candle:?}, earliest_boundary=output_logits",
            comparison.top_k_log_probability_abs_tolerance,
            comparison.mean_abs_log_probability_tolerance,
            comparison.total_variation_probability_tolerance,
            projected_log_probabilities[max_token_id],
            expected[max_token_id],
        )));
    }
    Ok(StageReport {
        stage: stage.to_string(),
        max_abs_log_probability_error: max,
        mean_abs_log_probability_error: mean,
        top_k_max_abs_log_probability_error: top_k_max,
        total_variation_probability_error: total_variation,
        top_reference,
        top_candle,
    })
}

pub fn log_probabilities(logits: &[f32]) -> Result<Vec<f32>> {
    let maximum = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| Error::Msg("cannot normalize an empty logits vector".into()))?;
    if !maximum.is_finite() {
        return Err(Error::Msg("logits contain no finite maximum".into()));
    }
    let sum_exp = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f32>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Err(Error::Msg("logits could not be normalized".into()));
    }
    let log_sum_exp = sum_exp.ln();
    logits
        .iter()
        .map(|value| {
            let normalized = value - maximum - log_sum_exp;
            if normalized.is_finite() {
                Ok(normalized)
            } else {
                Err(Error::Msg(
                    "normalized log probability is non-finite".into(),
                ))
            }
        })
        .collect()
}

pub fn top_log_probabilities(values: &[f32], count: usize) -> Vec<TopLogProbability> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|token_id| TopLogProbability {
            token_id,
            log_probability: values[token_id],
        })
        .collect()
}

pub fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

pub fn compare_replay(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.is_empty()
        || left.len() != right.len()
        || left.iter().chain(right).any(|x| !x.is_finite())
    {
        bail!("reset/replay rows are missing, malformed or nonfinite");
    }
    let error = max_abs_error(left, right);
    if error > 1e-5 {
        bail!("reset/replay mismatch: max_abs_error={error}");
    }
    Ok(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bounds() -> Comparison {
        Comparison {
            top_k: 2,
            top_k_log_probability_abs_tolerance: 0.25,
            mean_abs_log_probability_tolerance: 0.01,
            total_variation_probability_tolerance: 0.01,
            tolerance_rationale: "fixture".into(),
        }
    }
    fn tiny_reference() -> Vec<u8> {
        let mut data = b"_logits_".to_vec();
        for n in [3_i32, 2, 1] {
            data.extend(n.to_le_bytes());
        }
        for n in [1_u32, 2, 3] {
            data.extend(n.to_le_bytes());
        }
        data.extend(0.25_f32.to_le_bytes());
        data.extend((-1_f32).to_le_bytes());
        data.extend(0_u16.to_le_bytes());
        data.extend(1_u16.to_le_bytes());
        data
    }
    #[test]
    fn pinned_schedule_and_tolerances() -> Result<()> {
        let f = pinned_fixture()?;
        let p = fixed_prefixes(&f)?;
        assert_eq!(p.iter().map(Vec::len).collect::<Vec<_>>(), [5, 6, 7]);
        assert_eq!(f.reference_vocab_size, 201088);
        assert_eq!(f.comparison.top_k, 8);
        assert_eq!(f.comparison.top_k_log_probability_abs_tolerance, 0.25);
        assert_eq!(f.comparison.mean_abs_log_probability_tolerance, 0.01);
        assert_eq!(f.comparison.total_variation_probability_tolerance, 0.01);
        Ok(())
    }
    #[test]
    fn decoder_rejects_missing_trailing_wrong_prefix_and_nonfinite() -> Result<()> {
        let good = tiny_reference();
        assert_eq!(
            decode_reference_bytes(&good, &[1, 2, 3], 3, 2, 1)?,
            [vec![-1., -0.75]]
        );
        for n in 0..good.len() {
            assert!(decode_reference_bytes(&good[..n], &[1, 2, 3], 3, 2, 1).is_err());
        }
        let mut bad = good.clone();
        bad.push(0);
        assert!(decode_reference_bytes(&bad, &[1, 2, 3], 3, 2, 1).is_err());
        assert!(decode_reference_bytes(&good, &[2, 1, 3], 3, 2, 1).is_err());
        let mut bad = good;
        bad[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_reference_bytes(&bad, &[1, 2, 3], 3, 2, 1).is_err());
        assert!(reference_size(&[], usize::MAX, usize::MAX, 1).is_err());
        Ok(())
    }
    #[test]
    fn numerical_failure_and_replay_rules() -> Result<()> {
        let expected = log_probabilities(&[0., -1., -2.])?;
        assert!(compare_stage("row", &[0., -1., -2.], &expected, &bounds()).is_ok());
        assert!(compare_stage("row", &[-1., 0., -2.], &expected, &bounds()).is_err());
        assert!(compare_stage("row", &[0., -1.3, -2.], &expected, &bounds()).is_err());
        assert!(compare_stage("row", &[0., f32::NAN, -2.], &expected, &bounds()).is_err());
        assert!(compare_stage("row", &[0., -1.], &expected, &bounds()).is_err());
        assert_eq!(compare_replay(&[1., 2.], &[1., 2.])?, 0.);
        assert!(compare_replay(&[1., 2.], &[1.]).is_err());
        assert!(compare_replay(&[1., 2.], &[1., 2.001]).is_err());
        Ok(())
    }
}
