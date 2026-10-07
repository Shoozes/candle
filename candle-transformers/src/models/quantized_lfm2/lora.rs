//! Bounded low-rank deltas beside retained quantized LFM2 linear weights.

use candle::quantized::QMatMul;
use candle::{bail, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::{HashMap, HashSet};

const MAX_ADAPTER_BYTES: usize = 256 * 1024 * 1024;
const MAX_RANK: usize = 256;

/// One F32 B(Ax) delta for a canonical quantized-LFM2 linear target.
#[derive(Clone, Debug)]
pub struct Lfm2LoraPair {
    pub target: String,
    pub down: Tensor,
    pub up: Tensor,
    pub alpha: f64,
}

/// Tensor inputs whose file identity and training provenance the caller admits.
#[derive(Clone, Debug)]
pub struct Lfm2LoraAdapter {
    pub base_sha256: String,
    pub adapter_sha256: String,
    pub declared_targets: Vec<String>,
    pub pairs: Vec<Lfm2LoraPair>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lfm2LoraTarget {
    pub name: String,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedPair {
    down: Tensor,
    up: Tensor,
    scale: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct AdaptedLinear {
    base: QMatMul,
    name: String,
    input_size: usize,
    output_size: usize,
    active: Option<PreparedPair>,
}

impl AdaptedLinear {
    pub(crate) fn new(name: String, base: QMatMul) -> Result<Self> {
        let dims = match &base {
            QMatMul::QTensor(tensor) => tensor.shape().dims(),
            QMatMul::Tensor(tensor) | QMatMul::TensorF16(tensor) => tensor.dims(),
        };
        let [output_size, input_size] = dims else {
            bail!("quantized LFM2 LoRA target {name} must have a rank-2 base weight")
        };
        let (output_size, input_size) = (*output_size, *input_size);
        if input_size == 0 || output_size == 0 {
            bail!("quantized LFM2 LoRA target {name} has an empty base dimension")
        }
        Ok(Self {
            base,
            name,
            input_size,
            output_size,
            active: None,
        })
    }

    pub(crate) fn target(&self) -> Lfm2LoraTarget {
        Lfm2LoraTarget {
            name: self.name.clone(),
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn set(&mut self, prepared: Option<PreparedPair>) {
        self.active = prepared;
    }
}

impl Module for AdaptedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let base = self.base.forward(xs)?;
        let Some(pair) = &self.active else {
            return Ok(base);
        };
        let down = Linear::new(pair.down.clone(), None).forward(xs)?;
        let up = Linear::new(pair.up.clone(), None).forward(&down)?;
        base + (up * pair.scale)?
    }
}

pub(crate) fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn prepare(
    adapter: Lfm2LoraAdapter,
    base_sha256: &str,
    targets: &[Lfm2LoraTarget],
    device: &Device,
    max_bytes: usize,
) -> Result<(String, HashMap<String, PreparedPair>)> {
    if max_bytes == 0 || max_bytes > MAX_ADAPTER_BYTES {
        bail!("quantized LFM2 LoRA byte ceiling must be 1..={MAX_ADAPTER_BYTES}")
    }
    if !valid_sha256(&adapter.base_sha256) || !adapter.base_sha256.eq_ignore_ascii_case(base_sha256)
    {
        bail!("quantized LFM2 LoRA base SHA-256 does not match the admitted model")
    }
    if !valid_sha256(&adapter.adapter_sha256) {
        bail!("quantized LFM2 LoRA adapter SHA-256 identity is invalid")
    }
    if adapter.pairs.is_empty() || adapter.pairs.len() > targets.len() {
        bail!("quantized LFM2 LoRA target count is empty or exceeds the model inventory")
    }
    let declared = adapter.declared_targets.iter().collect::<HashSet<_>>();
    if declared.len() != adapter.pairs.len()
        || adapter.declared_targets.len() != adapter.pairs.len()
    {
        bail!("quantized LFM2 LoRA declared target set is incomplete or duplicated")
    }
    let inventory = targets
        .iter()
        .map(|target| (target.name.as_str(), target))
        .collect::<HashMap<_, _>>();
    let mut prepared = HashMap::with_capacity(adapter.pairs.len());
    let mut total_bytes = 0usize;
    for pair in adapter.pairs {
        if !declared.contains(&pair.target) || prepared.contains_key(&pair.target) {
            bail!(
                "quantized LFM2 LoRA target {} is undeclared or duplicated",
                pair.target
            )
        }
        let target = inventory.get(pair.target.as_str()).ok_or_else(|| {
            candle::Error::Msg(format!(
                "quantized LFM2 LoRA target {} is unsupported",
                pair.target
            ))
        })?;
        if pair.down.dtype() != DType::F32 || pair.up.dtype() != DType::F32 {
            bail!(
                "quantized LFM2 LoRA target {} requires F32 tensors",
                pair.target
            )
        }
        if !device.same_device(pair.down.device()) || !device.same_device(pair.up.device()) {
            bail!(
                "quantized LFM2 LoRA target {} has a different device",
                pair.target
            )
        }
        let [rank, input_size] = pair.down.dims() else {
            bail!(
                "quantized LFM2 LoRA target {} down tensor must be rank 2",
                pair.target
            )
        };
        let [output_size, up_rank] = pair.up.dims() else {
            bail!(
                "quantized LFM2 LoRA target {} up tensor must be rank 2",
                pair.target
            )
        };
        if *rank == 0
            || *rank > MAX_RANK
            || *rank != *up_rank
            || *input_size != target.input_size
            || *output_size != target.output_size
        {
            bail!(
                "quantized LFM2 LoRA target {} has incompatible rank or shape",
                pair.target
            )
        }
        if !pair.alpha.is_finite() || pair.alpha <= 0.0 {
            bail!(
                "quantized LFM2 LoRA target {} requires positive finite alpha",
                pair.target
            )
        }
        let scale = pair.alpha / *rank as f64;
        let execution_scale = scale as f32;
        if !execution_scale.is_finite() || execution_scale == 0.0 {
            bail!(
                "quantized LFM2 LoRA target {} has invalid F32 scale",
                pair.target
            )
        }
        let bytes = pair
            .down
            .elem_count()
            .checked_add(pair.up.elem_count())
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| candle::Error::Msg("quantized LFM2 LoRA byte count overflow".into()))?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| candle::Error::Msg("quantized LFM2 LoRA total byte overflow".into()))?;
        if total_bytes > max_bytes {
            bail!("quantized LFM2 LoRA tensor bytes exceed caller ceiling")
        }
        for (role, tensor) in [("down", &pair.down), ("up", &pair.up)] {
            let values = tensor.flatten_all()?.to_vec1::<f32>()?;
            if values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "quantized LFM2 LoRA target {} {role} contains non-finite values",
                    pair.target
                )
            }
            if values.iter().all(|value| *value == 0.0) {
                bail!(
                    "quantized LFM2 LoRA target {} {role} has zero effect",
                    pair.target
                )
            }
        }
        prepared.insert(
            pair.target,
            PreparedPair {
                down: pair.down,
                up: pair.up,
                scale,
            },
        );
    }
    Ok((adapter.adapter_sha256.to_ascii_lowercase(), prepared))
}
