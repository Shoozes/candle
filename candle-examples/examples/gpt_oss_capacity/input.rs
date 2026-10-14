use candle::{bail, Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub(super) const TOTAL: usize = 8192;
pub(super) const PREFILL: usize = 6144;
pub(super) const CHUNK: usize = 128;
pub(super) const DECODES: usize = 2048;
pub(super) const FORWARDS: usize = 2096;
pub(super) const MODEL_BYTES: u64 = 13_792_638_656;
pub(super) const MODEL_SHA: &str =
    "aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778";

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Backend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Identity {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Input {
    pub schema: String,
    pub source_commit: String,
    pub backend: Backend,
    pub device_index: usize,
    pub model: Identity,
    pub token_file: Identity,
    pub prefill_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub fixed_decode_tokens: usize,
    pub sampled_outputs: usize,
}

impl Input {
    pub fn validate(&self) -> Result<()> {
        if !self.model.path.is_absolute() || !self.token_file.path.is_absolute() {
            bail!("capacity requires explicit absolute model and token paths");
        }
        if self.schema != "candle.gpt-oss.capacity-input.v1"
            || self.source_commit.len() != 40
            || !self.source_commit.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("capacity input version/source mismatch");
        }
        if self.prefill_tokens != PREFILL
            || self.prefill_chunk_tokens != CHUNK
            || self.fixed_decode_tokens != DECODES
            || self.sampled_outputs != 0
        {
            bail!("capacity input differs from frozen 6144/128/2048 fixed-token work");
        }
        if self.device_index != 0
            || self.model.size_bytes != MODEL_BYTES
            || self.model.sha256 != MODEL_SHA
        {
            bail!("capacity model/device identity differs from retained qualification");
        }
        if self.token_file.size_bytes != (TOTAL * 4) as u64
            || self.token_file.sha256.len() != 64
            || !self
                .token_file
                .sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            bail!("capacity token identity must bind exactly 8192 little-endian u32 IDs");
        }
        Ok(())
    }
}

pub(super) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let cap = limit
        .checked_add(1)
        .ok_or_else(|| Error::Msg("capacity file bound overflow".into()))?;
    fs::File::open(path)?
        .take(cap as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("capacity file exceeds {limit}-byte bound");
    }
    Ok(bytes)
}

pub(crate) fn read_input(path: &Path) -> Result<(Input, String, Vec<u32>)> {
    let bytes = read_bounded(path, 16384)?;
    let input: Input = serde_json::from_slice(&bytes).map_err(Error::wrap)?;
    input.validate()?;
    let token_bytes = read_bounded(&input.token_file.path, TOTAL * 4)?;
    if token_bytes.len() != TOTAL * 4
        || format!("{:x}", Sha256::digest(&token_bytes)) != input.token_file.sha256
    {
        bail!("capacity token file size/digest mismatch");
    }
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/gpt_oss_task3_short_parity/reference.json"
    ))
    .map_err(Error::wrap)?;
    let prefix: Vec<u32> = serde_json::from_value(fixture["reference_prefix_token_ids"].clone())
        .map_err(Error::wrap)?;
    if prefix.len() != 8 {
        bail!("capacity pinned prefix must have eight IDs");
    }
    let mut tokens = Vec::with_capacity(TOTAL);
    for (index, word) in token_bytes.chunks_exact(4).enumerate() {
        let word: [u8; 4] = word.try_into().map_err(Error::wrap)?;
        let token = u32::from_le_bytes(word);
        if prefix.get(index % prefix.len()).copied() != Some(token) {
            bail!("capacity token {index} differs from frozen repeated prefix");
        }
        tokens.push(token);
    }
    Ok((input, format!("{:x}", Sha256::digest(&bytes)), tokens))
}
