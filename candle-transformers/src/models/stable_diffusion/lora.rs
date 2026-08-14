//! Validated SDXL LoRA tensor parsing and merge planning.
//!
//! This module owns tensor-format behavior only. Applications remain
//! responsible for adapter discovery, paths, licensing, and model-specific
//! name conversion through [`SdxlLoraTargetResolver`].

use std::collections::BTreeMap;

use candle::{DType, Result, Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};

const LORA_SUFFIXES: &[(&str, LoraTensorKind)] = &[
    (".lora_down.weight", LoraTensorKind::Down),
    (".lora_up.weight", LoraTensorKind::Up),
    ("_lora_down.weight", LoraTensorKind::Down),
    ("_lora_up.weight", LoraTensorKind::Up),
    (".down.weight", LoraTensorKind::Down),
    (".up.weight", LoraTensorKind::Up),
    ("_down.weight", LoraTensorKind::Down),
    ("_up.weight", LoraTensorKind::Up),
    (".lora_A.weight", LoraTensorKind::Down),
    (".lora_B.weight", LoraTensorKind::Up),
    (".alpha", LoraTensorKind::Alpha),
    ("_alpha", LoraTensorKind::Alpha),
];

/// Mutable SDXL model component addressed by an adapter tensor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SdxlLoraComponent {
    Unet,
    TextEncoder1,
    TextEncoder2,
}

impl SdxlLoraComponent {
    /// Stable evidence key for this component.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unet => "unet",
            Self::TextEncoder1 => "text_encoder_1",
            Self::TextEncoder2 => "text_encoder_2",
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Unet => 0,
            Self::TextEncoder1 => 1,
            Self::TextEncoder2 => 2,
        }
    }

    pub(crate) const ALL: [Self; 3] = [Self::Unet, Self::TextEncoder1, Self::TextEncoder2];
}

/// Tensor role within one LoRA pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraTensorKind {
    Down,
    Up,
    Alpha,
}

/// One validated up/down LoRA pair and its optional alpha value.
#[derive(Clone, Debug)]
pub struct LoraPair {
    component: SdxlLoraComponent,
    stem: String,
    down: Tensor,
    up: Tensor,
    alpha: Option<f64>,
    rank: usize,
}

impl LoraPair {
    pub fn component(&self) -> SdxlLoraComponent {
        self.component
    }

    pub fn stem(&self) -> &str {
        &self.stem
    }

    pub fn down(&self) -> &Tensor {
        &self.down
    }

    pub fn up(&self) -> &Tensor {
        &self.up
    }

    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Application-owned mapping from a parsed adapter stem to a model weight.
///
/// Returning `None` rejects that stem. This keeps model-family naming policy
/// outside Candle while Candle still validates the returned target exactly.
pub trait SdxlLoraTargetResolver {
    fn resolve_target(&self, component: SdxlLoraComponent, stem: &str) -> Option<String>;
}

impl<F> SdxlLoraTargetResolver for F
where
    F: Fn(SdxlLoraComponent, &str) -> Option<String>,
{
    fn resolve_target(&self, component: SdxlLoraComponent, stem: &str) -> Option<String> {
        self(component, stem)
    }
}

/// Deterministic evidence for one effective target delta.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LoraTargetEvidence {
    pub target: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub base_sha256: String,
    pub delta_sha256: String,
    pub merged_sha256: String,
}

/// Per-component results from one planned adapter replacement.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LoraComponentApplyStats {
    pub component: SdxlLoraComponent,
    pub pair_count: usize,
    pub applied_count: usize,
    pub zero_delta_count: usize,
    pub targets: Vec<LoraTargetEvidence>,
    pub restored_target_keys: Vec<String>,
}

impl LoraComponentApplyStats {
    pub(crate) fn empty(component: SdxlLoraComponent) -> Self {
        Self {
            component,
            pair_count: 0,
            applied_count: 0,
            zero_delta_count: 0,
            targets: Vec::new(),
            restored_target_keys: Vec::new(),
        }
    }
}

/// Stable three-component statistics for an adapter replacement or clear.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LoraApplyStats {
    pub components: [LoraComponentApplyStats; 3],
}

impl Default for LoraApplyStats {
    fn default() -> Self {
        Self {
            components: SdxlLoraComponent::ALL.map(LoraComponentApplyStats::empty),
        }
    }
}

impl LoraApplyStats {
    pub fn component(&self, component: SdxlLoraComponent) -> &LoraComponentApplyStats {
        &self.components[component.index()]
    }

    pub fn applied_count(&self) -> usize {
        self.components
            .iter()
            .map(|component| component.applied_count)
            .sum()
    }

    pub fn pair_count(&self) -> usize {
        self.components
            .iter()
            .map(|component| component.pair_count)
            .sum()
    }
}

#[derive(Default)]
struct PendingPair {
    down: Option<Tensor>,
    up: Option<Tensor>,
    alpha: Option<f64>,
}

/// Parse and validate all tensors in one SDXL LoRA adapter.
///
/// Unknown tensor names, duplicate pair members, missing members, non-floating
/// tensors, zero ranks, malformed alpha tensors, and cross-device pairs fail
/// before any model mutation can be planned.
pub fn parse_sdxl_lora_pairs<'a>(
    lora_tensors: impl IntoIterator<Item = (&'a String, &'a Tensor)>,
) -> Result<Vec<LoraPair>> {
    let mut tensors = lora_tensors.into_iter().collect::<Vec<_>>();
    if tensors.is_empty() {
        candle::bail!("SDXL LoRA adapter contains no tensors")
    }
    tensors.sort_by_key(|(left, _)| *left);

    let mut pending = BTreeMap::<(SdxlLoraComponent, String), PendingPair>::new();
    for (name, tensor) in tensors {
        let (raw_stem, kind) = parse_lora_key(name).ok_or_else(|| {
            candle::Error::Msg(format!(
                "unsupported SDXL LoRA tensor name {name}; expected paired up/down weights or alpha"
            ))
            .bt()
        })?;
        let (component, stem) = split_lora_component(raw_stem)?;
        if stem.is_empty() {
            candle::bail!("SDXL LoRA tensor {name} has an empty target stem")
        }
        let entry = pending.entry((component, stem.to_owned())).or_default();
        let duplicate = match kind {
            LoraTensorKind::Down => entry.down.replace(tensor.clone()).is_some(),
            LoraTensorKind::Up => entry.up.replace(tensor.clone()).is_some(),
            LoraTensorKind::Alpha => entry.alpha.replace(read_alpha(name, tensor)?).is_some(),
        };
        if duplicate {
            candle::bail!(
                "duplicate {:?} tensor for SDXL LoRA target {}:{}",
                kind,
                component.as_str(),
                stem
            )
        }
    }

    let mut pairs = Vec::new();
    pairs
        .try_reserve_exact(pending.len())
        .map_err(|_| candle::Error::Msg("allocating SDXL LoRA pair inventory".to_owned()).bt())?;
    for ((component, stem), pair) in pending {
        let down = pair.down.ok_or_else(|| {
            candle::Error::Msg(format!(
                "SDXL LoRA target {}:{stem} is missing its down tensor",
                component.as_str()
            ))
            .bt()
        })?;
        let up = pair.up.ok_or_else(|| {
            candle::Error::Msg(format!(
                "SDXL LoRA target {}:{stem} is missing its up tensor",
                component.as_str()
            ))
            .bt()
        })?;
        ensure_supported_float(&down, component, &stem, "down")?;
        ensure_supported_float(&up, component, &stem, "up")?;
        if !down.device().same_device(up.device()) {
            candle::bail!(
                "SDXL LoRA target {}:{stem} has up/down tensors on different devices",
                component.as_str()
            )
        }
        let (rank, _) = down.dims2().map_err(|_| {
            candle::Error::Msg(format!(
                "SDXL LoRA target {}:{stem} down tensor must be rank 2, found {:?}",
                component.as_str(),
                down.dims()
            ))
            .bt()
        })?;
        let (_, up_rank) = up.dims2().map_err(|_| {
            candle::Error::Msg(format!(
                "SDXL LoRA target {}:{stem} up tensor must be rank 2, found {:?}",
                component.as_str(),
                up.dims()
            ))
            .bt()
        })?;
        if rank == 0 {
            candle::bail!(
                "SDXL LoRA target {}:{stem} has rank zero",
                component.as_str()
            )
        }
        if up_rank != rank {
            candle::bail!(
                "SDXL LoRA target {}:{stem} rank mismatch: down rank {rank}, up input {up_rank}",
                component.as_str()
            )
        }
        if pair
            .alpha
            .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0)
        {
            candle::bail!(
                "SDXL LoRA target {}:{stem} alpha must be finite and greater than zero",
                component.as_str()
            )
        }
        pairs.push(LoraPair {
            component,
            stem,
            down,
            up,
            alpha: pair.alpha,
            rank,
        });
    }
    Ok(pairs)
}

fn parse_lora_key(key: &str) -> Option<(&str, LoraTensorKind)> {
    LORA_SUFFIXES
        .iter()
        .find_map(|(suffix, kind)| key.strip_suffix(suffix).map(|stem| (stem, *kind)))
}

fn split_lora_component(stem: &str) -> Result<(SdxlLoraComponent, &str)> {
    for prefix in [
        "lora_te2_",
        "lora.te2.",
        "te2.",
        "te2_",
        "text_encoder_2.",
        "text_encoder_2_",
    ] {
        if let Some(stem) = stem.strip_prefix(prefix) {
            return Ok((SdxlLoraComponent::TextEncoder2, stem));
        }
    }
    for prefix in [
        "lora_te1_",
        "lora.te1.",
        "te1.",
        "te1_",
        "text_encoder_1.",
        "text_encoder_1_",
        "text_encoder.",
        "text_encoder_",
    ] {
        if let Some(stem) = stem.strip_prefix(prefix) {
            return Ok((SdxlLoraComponent::TextEncoder1, stem));
        }
    }
    for prefix in ["lora_unet_", "lora.unet.", "unet.", "unet_"] {
        if let Some(stem) = stem.strip_prefix(prefix) {
            return Ok((SdxlLoraComponent::Unet, stem));
        }
    }
    if stem.starts_with("lora_te")
        || stem.starts_with("lora.te")
        || stem.starts_with("text_encoder")
    {
        candle::bail!("unsupported SDXL LoRA component prefix in target {stem}")
    }
    Ok((SdxlLoraComponent::Unet, stem))
}

fn read_alpha(name: &str, tensor: &Tensor) -> Result<f64> {
    ensure_supported_float(tensor, SdxlLoraComponent::Unet, name, "alpha")?;
    if tensor.elem_count() != 1 {
        candle::bail!(
            "SDXL LoRA alpha tensor {name} must contain exactly one value, found {}",
            tensor.elem_count()
        )
    }
    let alpha = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .next()
        .ok_or_else(|| candle::Error::Msg(format!("SDXL LoRA alpha tensor {name} is empty")).bt())?
        as f64;
    if !alpha.is_finite() || alpha <= 0.0 {
        candle::bail!("SDXL LoRA alpha tensor {name} must be finite and greater than zero")
    }
    Ok(alpha)
}

fn ensure_supported_float(
    tensor: &Tensor,
    component: SdxlLoraComponent,
    stem: &str,
    role: &str,
) -> Result<()> {
    if !matches!(
        tensor.dtype(),
        DType::BF16 | DType::F16 | DType::F32 | DType::F64
    ) {
        candle::bail!(
            "SDXL LoRA {}:{stem} {role} tensor uses unsupported dtype {:?}",
            component.as_str(),
            tensor.dtype()
        )
    }
    Ok(())
}

pub(crate) struct PreparedLoraTarget {
    pub merged: Tensor,
    pub evidence: LoraTargetEvidence,
}

pub(crate) fn prepare_lora_target(
    pair: &LoraPair,
    target: &str,
    base: &Tensor,
    strength: f64,
) -> Result<Option<PreparedLoraTarget>> {
    if !strength.is_finite() {
        candle::bail!("SDXL LoRA strength must be finite, found {strength}")
    }
    ensure_supported_float(base, pair.component, target, "base")?;
    if !base.device().same_device(pair.down.device()) {
        candle::bail!(
            "SDXL LoRA target {}:{target} model and adapter tensors are on different devices",
            pair.component.as_str()
        )
    }
    ensure_finite_tensor(pair.down(), pair.component, pair.stem(), "down")?;
    ensure_finite_tensor(pair.up(), pair.component, pair.stem(), "up")?;

    let down = pair.down.to_dtype(DType::F32)?;
    let up = pair.up.to_dtype(DType::F32)?;
    let delta = up.matmul(&down)?;
    let delta = match base.rank() {
        2 => delta,
        4 => {
            let (_, _, kernel_h, kernel_w) = base.dims4()?;
            if kernel_h != 1 || kernel_w != 1 {
                candle::bail!(
                    "SDXL LoRA target {}:{target} only supports 1x1 rank-4 weights, found {:?}",
                    pair.component.as_str(),
                    base.dims()
                )
            }
            delta.unsqueeze(2)?.unsqueeze(3)?
        }
        rank => candle::bail!(
            "SDXL LoRA target {}:{target} base tensor rank {rank} is unsupported; expected 2 or 4",
            pair.component.as_str()
        ),
    };
    if delta.dims() != base.dims() {
        candle::bail!(
            "SDXL LoRA target {}:{target} delta shape {:?} does not match base shape {:?}",
            pair.component.as_str(),
            delta.dims(),
            base.dims()
        )
    }

    let rank = pair.rank as f64;
    let alpha = pair.alpha.unwrap_or(rank);
    let scale = alpha / rank * strength;
    if !scale.is_finite() {
        candle::bail!(
            "SDXL LoRA target {}:{target} produced non-finite scale",
            pair.component.as_str()
        )
    }
    let scaled_delta = (&delta * scale)?;
    let scaled_values = finite_f32_values(
        &scaled_delta,
        &format!(
            "SDXL LoRA target {}:{target} scaled delta",
            pair.component.as_str()
        ),
    )?;
    if scaled_values.iter().all(|value| *value == 0.0) {
        return Ok(None);
    }

    let delta_sha256 = canonical_hash_from_values(scaled_delta.dims(), &scaled_values)?;
    let base_sha256 = canonical_lora_tensor_sha256(base)?;
    let scaled_delta = scaled_delta.to_dtype(base.dtype())?;
    let merged = (base + &scaled_delta)?;
    let merged_sha256 = canonical_lora_tensor_sha256(&merged)?;
    Ok(Some(PreparedLoraTarget {
        merged,
        evidence: LoraTargetEvidence {
            target: target.to_owned(),
            shape: base.dims().to_vec(),
            dtype: base.dtype().as_str().to_owned(),
            base_sha256,
            delta_sha256,
            merged_sha256,
        },
    }))
}

fn ensure_finite_tensor(
    tensor: &Tensor,
    component: SdxlLoraComponent,
    stem: &str,
    role: &str,
) -> Result<()> {
    finite_f32_values(
        tensor,
        &format!("SDXL LoRA {}:{stem} {role} tensor", component.as_str()),
    )?;
    Ok(())
}

fn finite_f32_values(tensor: &Tensor, label: &str) -> Result<Vec<f32>> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        candle::bail!("{label} contains a non-finite value at flattened index {index}")
    }
    Ok(values)
}

/// Hash tensor shape and canonical F32 values using a fixed little-endian
/// contract suitable for cross-consumer LoRA evidence.
pub fn canonical_lora_tensor_sha256(tensor: &Tensor) -> Result<String> {
    let values = finite_f32_values(tensor, "SDXL LoRA evidence tensor")?;
    canonical_hash_from_values(tensor.dims(), &values)
}

fn canonical_hash_from_values(shape: &[usize], values: &[f32]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"candle-sdxl-lora-tensor-f32-v1\0");
    let rank = u64::try_from(shape.len())
        .map_err(|_| candle::Error::Msg("SDXL LoRA tensor rank exceeds u64".to_owned()).bt())?;
    hasher.update(rank.to_le_bytes());
    for &dim in shape {
        let dim = u64::try_from(dim).map_err(|_| {
            candle::Error::Msg("SDXL LoRA tensor dimension exceeds u64".to_owned()).bt()
        })?;
        hasher.update(dim.to_le_bytes());
    }
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn tensor(values: &[f32], shape: impl Into<candle::Shape>) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn parser_splits_all_three_components_and_is_deterministic() {
        let tensors = BTreeMap::from([
            (
                "lora_te2_text.k_proj.lora_up.weight".to_owned(),
                tensor(&[1., 1.], (2, 1)),
            ),
            (
                "lora_unet_down.to_q.lora_down.weight".to_owned(),
                tensor(&[1., 2.], (1, 2)),
            ),
            (
                "lora_te1_text.q_proj.lora_down.weight".to_owned(),
                tensor(&[1., 2.], (1, 2)),
            ),
            (
                "lora_te2_text.k_proj.lora_down.weight".to_owned(),
                tensor(&[1., 2.], (1, 2)),
            ),
            (
                "lora_unet_down.to_q.lora_up.weight".to_owned(),
                tensor(&[1., 1.], (2, 1)),
            ),
            (
                "lora_te1_text.q_proj.lora_up.weight".to_owned(),
                tensor(&[1., 1.], (2, 1)),
            ),
        ]);
        let pairs = parse_sdxl_lora_pairs(&tensors).unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].component(), SdxlLoraComponent::Unet);
        assert_eq!(pairs[1].component(), SdxlLoraComponent::TextEncoder1);
        assert_eq!(pairs[2].component(), SdxlLoraComponent::TextEncoder2);
    }

    #[test]
    fn parser_rejects_missing_members_bad_rank_alpha_and_unknown_names() {
        let down = tensor(&[1., 2.], (1, 2));
        let mut tensors = BTreeMap::from([("layer.lora_down.weight".to_owned(), down.clone())]);
        assert!(parse_sdxl_lora_pairs(&tensors)
            .unwrap_err()
            .to_string()
            .contains("missing its up tensor"));

        tensors.insert("layer.lora_up.weight".to_owned(), tensor(&[1., 1.], (2, 1)));
        tensors.insert("layer.alpha".to_owned(), tensor(&[1., 2.], (2,)));
        assert!(parse_sdxl_lora_pairs(&tensors)
            .unwrap_err()
            .to_string()
            .contains("exactly one value"));

        let zero_rank = BTreeMap::from([
            (
                "layer.lora_down.weight".to_owned(),
                Tensor::zeros((0, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "layer.lora_up.weight".to_owned(),
                Tensor::zeros((2, 0), DType::F32, &Device::Cpu).unwrap(),
            ),
        ]);
        assert!(parse_sdxl_lora_pairs(&zero_rank)
            .unwrap_err()
            .to_string()
            .contains("rank zero"));

        let invalid_alpha = BTreeMap::from([
            (
                "layer.lora_down.weight".to_owned(),
                tensor(&[1., 2.], (1, 2)),
            ),
            ("layer.lora_up.weight".to_owned(), tensor(&[1., 1.], (2, 1))),
            ("layer.alpha".to_owned(), tensor(&[0.], (1,))),
        ]);
        assert!(parse_sdxl_lora_pairs(&invalid_alpha)
            .unwrap_err()
            .to_string()
            .contains("greater than zero"));

        let unknown = BTreeMap::from([("layer.dora_scale".to_owned(), down)]);
        assert!(parse_sdxl_lora_pairs(&unknown)
            .unwrap_err()
            .to_string()
            .contains("unsupported SDXL LoRA tensor name"));

        let unknown_component = BTreeMap::from([
            (
                "lora_te3_layer.lora_down.weight".to_owned(),
                tensor(&[1., 2.], (1, 2)),
            ),
            (
                "lora_te3_layer.lora_up.weight".to_owned(),
                tensor(&[1., 1.], (2, 1)),
            ),
        ]);
        assert!(parse_sdxl_lora_pairs(&unknown_component)
            .unwrap_err()
            .to_string()
            .contains("unsupported SDXL LoRA component prefix"));
    }

    #[test]
    fn canonical_hash_includes_shape_and_normalizes_float_dtype() {
        let a = tensor(&[1., 2., 3., 4.], (2, 2));
        let b = a.to_dtype(DType::BF16).unwrap();
        let reshaped = a.reshape((4, 1)).unwrap();
        assert_eq!(
            canonical_lora_tensor_sha256(&a).unwrap(),
            canonical_lora_tensor_sha256(&b).unwrap()
        );
        assert_ne!(
            canonical_lora_tensor_sha256(&a).unwrap(),
            canonical_lora_tensor_sha256(&reshaped).unwrap()
        );
    }
}
