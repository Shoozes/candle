//! Bounded, token-sequential CPU execution for admitted Qwen3.5 GGUF text models.

use super::admission::{Artifact, Config};
use candle::quantized::{gguf_file::TensorInfo, QMatMul, QTensor};
use candle::{bail, Device, Result, Tensor};
use std::sync::Arc;

type ModelObserver<'a> = &'a mut dyn FnMut(usize, &str, &[f32]);
type ComponentObserver<'a> = &'a mut dyn FnMut(&str, &[f32]);

struct Recurrent {
    qkv: QMatMul,
    z: QMatMul,
    alpha: QMatMul,
    beta: QMatMul,
    out: QMatMul,
    conv: Vec<f32>,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
}

struct Attention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    out: QMatMul,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

enum Mixer {
    Recurrent(Recurrent),
    Attention(Attention),
}

struct Layer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    mixer: Mixer,
}

enum Cache {
    Recurrent {
        fifo: Vec<f32>,
        state: Vec<f32>,
    },
    Attention {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    },
}

pub struct CpuModel {
    config: Config,
    max_positions: usize,
    position: usize,
    embedding: Arc<QTensor>,
    output_norm: Vec<f32>,
    output: QMatMul,
    layers: Vec<Layer>,
    cache: Vec<Cache>,
}

/// CUDA-resident projections with host-side hybrid mixer state.
#[cfg(feature = "cuda")]
pub struct HybridGpuModel {
    inner: CpuModel,
}

#[cfg(feature = "cuda")]
impl HybridGpuModel {
    pub fn load(
        artifact: &mut Artifact,
        max_positions: usize,
        max_state_bytes: usize,
        device: &Device,
    ) -> Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            bail!("qwen35: hybrid GPU model requires a CUDA device")
        }
        Ok(Self {
            inner: CpuModel::load_with_projection_device(
                artifact,
                max_positions,
                max_state_bytes,
                device,
            )?,
        })
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.inner.prefill(tokens)
    }

    /// Cancels between tokens and clears all partially updated host state.
    pub fn prefill_with_cancel(
        &mut self,
        tokens: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>> {
        if self.inner.position != 0 || tokens.is_empty() || tokens.len() > self.inner.max_positions
        {
            bail!("qwen35: prefill requires an empty cache and bounded nonempty tokens")
        }
        let mut logits = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            if cancelled() {
                self.inner.reset();
                bail!("qwen35: hybrid CUDA prefill cancelled at token boundary")
            }
            let next = if index + 1 == tokens.len() {
                self.inner.decode(token)
            } else {
                self.inner.decode_without_logits(token)
            };
            match next {
                Ok(next) => logits = next,
                Err(error) => {
                    self.inner.reset();
                    return Err(error);
                }
            }
        }
        Ok(logits)
    }

    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.inner.decode(token)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn position(&self) -> usize {
        self.inner.position()
    }
}

fn qtensor(artifact: &mut Artifact, name: &str, device: &Device) -> Result<QTensor> {
    let tensor = artifact
        .tensors
        .get(name)
        .ok_or_else(|| candle::Error::Msg(format!("qwen35: admitted tensor {name} missing")))?;
    TensorInfo {
        ggml_dtype: tensor.dtype,
        shape: tensor.shape.clone().into(),
        offset: tensor.byte_start,
    }
    .read(artifact.file(), 0, device)
}

fn matrix(artifact: &mut Artifact, name: &str, device: &Device) -> Result<QMatMul> {
    QMatMul::from_qtensor(qtensor(artifact, name, device)?)
}

fn dense(artifact: &mut Artifact, name: &str) -> Result<Vec<f32>> {
    qtensor(artifact, name, &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()
}

#[cfg(test)]
mod bench_profile {
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::time::Duration;

    type Key = (&'static str, usize, usize);
    thread_local! {
        static STATS: RefCell<Option<BTreeMap<Key, (u64, Duration)>>> = const { RefCell::new(None) };
    }

    pub fn start() {
        STATS.with(|stats| *stats.borrow_mut() = Some(BTreeMap::new()));
    }

    pub fn record(weight: &QMatMul, input: usize, output: usize, elapsed: Duration) {
        STATS.with(|stats| {
            if let Some(stats) = stats.borrow_mut().as_mut() {
                let kind = match weight {
                    QMatMul::QTensor(tensor)
                        if tensor.dtype() == candle::quantized::GgmlDType::Q8_0 =>
                    {
                        "q8_0"
                    }
                    _ => "dense",
                };
                let entry = stats.entry((kind, input, output)).or_default();
                entry.0 += 1;
                entry.1 += elapsed;
            }
        });
    }

    pub fn record_stage(kind: &'static str, input: usize, output: usize, elapsed: Duration) {
        STATS.with(|stats| {
            if let Some(stats) = stats.borrow_mut().as_mut() {
                let entry = stats.entry((kind, input, output)).or_default();
                entry.0 += 1;
                entry.1 += elapsed;
            }
        });
    }

    pub fn take() -> Vec<serde_json::Value> {
        let stats = STATS.with(|stats| stats.borrow_mut().take().unwrap_or_default());
        let mut rows = stats
            .into_iter()
            .map(|((kind, input, output), (calls, elapsed))| {
                serde_json::json!({
                    "kind": kind,
                    "input": input,
                    "output": output,
                    "calls": calls,
                    "total_ms": elapsed.as_secs_f64() * 1_000.0
                })
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|a, b| {
            b["total_ms"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a["total_ms"].as_f64().unwrap_or(0.0))
        });
        rows
    }
}

fn projection(weight: &QMatMul, input: &[f32]) -> Result<Vec<f32>> {
    #[cfg(test)]
    let started = std::time::Instant::now();
    let device = match weight {
        QMatMul::QTensor(tensor) => tensor.device(),
        QMatMul::Tensor(tensor) | QMatMul::TensorF16(tensor) => tensor.device().clone(),
    };
    let x = Tensor::from_slice(input, (1, input.len()), &Device::Cpu)?.to_device(&device)?;
    #[cfg(test)]
    let h2d_done = std::time::Instant::now();
    let result = weight.forward_ggml_q8_0(&x)?;
    #[cfg(test)]
    let kernel_done = std::time::Instant::now();
    let output = result.flatten_all()?.to_vec1()?;
    #[cfg(test)]
    {
        let done = std::time::Instant::now();
        bench_profile::record_stage(
            "projection.h2d",
            input.len(),
            output.len(),
            h2d_done - started,
        );
        bench_profile::record_stage(
            "projection.kernel",
            input.len(),
            output.len(),
            kernel_done - h2d_done,
        );
        bench_profile::record_stage(
            "projection.d2h",
            input.len(),
            output.len(),
            done - kernel_done,
        );
        bench_profile::record(weight, input.len(), output.len(), done - started);
    }
    Ok(output)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Scalar translation of ggml's Zen4 F32 vector exponential, MIT; see SOURCES.md.
fn ggml_exp_f32(x: f32) -> f32 {
    const R: f32 = f32::from_bits(0x4b40_0000);
    const LOG2_E: f32 = f32::from_bits(0x3fb8_aa3b);
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const P0: f32 = f32::from_bits(0x3c07_2010);
    const P1: f32 = f32::from_bits(0x3d2b_9f17);
    const P2: f32 = f32::from_bits(0x3e2a_af33);
    const P3: f32 = f32::from_bits(0x3eff_fedb);
    const P4: f32 = f32::from_bits(0x3f7f_fff6);
    if !x.is_finite() {
        return x.exp();
    }
    let z = x.mul_add(LOG2_E, R);
    let n = z - R;
    if n.abs() > 192.0 {
        return if n > 0.0 { f32::INFINITY } else { 0.0 };
    }
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let u = b * b;
    let j = P0
        .mul_add(b, P1)
        .mul_add(u, P2.mul_add(b, P3))
        .mul_add(u, P4.mul_add(b, 1.0));
    let exponent = n as i32;
    let scale = if (-126..=127).contains(&exponent) {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        2.0f32.powi(exponent)
    };
    j * scale
}

fn silu(x: f32) -> f32 {
    x / (1.0 + ggml_exp_f32(-x))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn rms_norm(input: &mut [f32], weights: &[f32], eps: f32) {
    let variance = (input.iter().map(|x| (x * x) as f64).sum::<f64>() / input.len() as f64) as f32;
    let inv = (variance + eps).sqrt().recip();
    for (value, weight) in input.iter_mut().zip(weights) {
        *value = (*value * inv) * weight;
    }
}

fn l2_norm(input: &mut [f32], eps: f32) {
    let n = input.len() as f32;
    let mean = (input.iter().map(|x| (x * x) as f64).sum::<f64>() / input.len() as f64) as f32;
    let scale = (mean + eps / n).sqrt().recip();
    let factor = n.sqrt().recip();
    for value in input {
        *value = (*value * scale) * factor;
    }
}

fn ggml_f32_dot(pairs: impl Iterator<Item = (f32, f32)>) -> f32 {
    let mut groups = [[0.0f32; 16]; 4];
    for (key, (left, right)) in pairs.enumerate() {
        let group = (key / 16) % 4;
        let lane = key % 16;
        groups[group][lane] = left.mul_add(right, groups[group][lane]);
    }
    let mut lanes = [0.0f32; 16];
    for lane in 0..16 {
        lanes[lane] = (groups[0][lane] + groups[2][lane]) + (groups[1][lane] + groups[3][lane]);
    }
    let mut halves = [0.0f32; 8];
    for lane in 0..8 {
        halves[lane] = lanes[lane] + lanes[lane + 8];
    }
    let mut quarters = [0.0f32; 4];
    for lane in 0..4 {
        quarters[lane] = halves[lane] + halves[lane + 4];
    }
    (quarters[0] + quarters[2]) + (quarters[1] + quarters[3])
}

#[cfg(test)]
fn recurrent_dot(state: &[f32], stride: usize, vector: &[f32], offset: usize) -> f32 {
    ggml_f32_dot(
        vector
            .iter()
            .enumerate()
            .map(|(key, &value)| (state[key * stride + offset], value)),
    )
}

fn reduce_dot_groups(groups: &[f32], dim: usize, output: &mut [f32]) {
    for (index, out) in output.iter_mut().enumerate() {
        let lane = |group: usize, lane: usize| groups[(group * 16 + lane) * dim + index];
        let mut quarters = [0.0f32; 4];
        for (quarter, result) in quarters.iter_mut().enumerate() {
            let low = quarter;
            let high = quarter + 4;
            let sum_lane = |lane_index: usize| {
                (lane(0, lane_index) + lane(2, lane_index))
                    + (lane(1, lane_index) + lane(3, lane_index))
            };
            *result = (sum_lane(low) + sum_lane(low + 8)) + (sum_lane(high) + sum_lane(high + 8));
        }
        *out = (quarters[0] + quarters[2]) + (quarters[1] + quarters[3]);
    }
}

// Read contiguous key rows while retaining ggml_f32_dot's accumulation order.
fn recurrent_matvec(
    state: &[f32],
    vector: &[f32],
    dim: usize,
    groups: &mut [f32],
    output: &mut [f32],
) {
    groups.fill(0.0);
    for (key, &weight) in vector.iter().enumerate() {
        let slot = key % 64;
        let accum = &mut groups[slot * dim..(slot + 1) * dim];
        let row = &state[key * dim..(key + 1) * dim];
        for (acc, &value) in accum.iter_mut().zip(row) {
            *acc = value.mul_add(weight, *acc);
        }
    }
    reduce_dot_groups(groups, dim, output);
}

// Accumulate one value head token-major while preserving ggml_f32_dot's 64 lanes.
fn attention_weighted_values(
    values: &[Vec<f32>],
    scores: &[f32],
    kv_head: usize,
    dim: usize,
    groups: &mut [f32],
    output: &mut [f32],
) {
    groups.fill(0.0);
    for (token, (&score, value)) in scores.iter().zip(values).enumerate() {
        let slot = token % 64;
        let accum = &mut groups[slot * dim..(slot + 1) * dim];
        let row = &value[kv_head * dim..(kv_head + 1) * dim];
        for (acc, &value) in accum.iter_mut().zip(row) {
            *acc = value.mul_add(score, *acc);
        }
    }
    reduce_dot_groups(groups, dim, output);
}

fn attention_softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for chunk in scores.chunks_mut(16) {
        let mut lanes = [0.0f32; 16];
        for (lane, score) in chunk.iter_mut().enumerate() {
            *score = ggml_exp_f32(*score - max);
            lanes[lane] = *score;
        }
        for offset in [8, 4, 2, 1] {
            for lane in 0..offset {
                lanes[lane] += lanes[lane + offset];
            }
        }
        sum += lanes[0] as f64;
    }
    let inv = (1.0 / sum) as f32;
    for score in scores {
        *score *= inv;
    }
}

impl CpuModel {
    /// Loads only a previously hash-checked and directory-admitted artifact.
    pub fn load(
        artifact: &mut Artifact,
        max_positions: usize,
        max_state_bytes: usize,
    ) -> Result<Self> {
        Self::load_with_projection_device(artifact, max_positions, max_state_bytes, &Device::Cpu)
    }

    fn load_with_projection_device(
        artifact: &mut Artifact,
        max_positions: usize,
        max_state_bytes: usize,
        projection_device: &Device,
    ) -> Result<Self> {
        let config = artifact.config.clone();
        if max_positions == 0 || max_positions > config.context_length {
            bail!("qwen35: invalid caller position ceiling")
        }
        let key_width = config
            .state_size
            .checked_mul(config.key_groups)
            .ok_or_else(|| candle::Error::Msg("qwen35: key width overflow".into()))?;
        let channels = key_width
            .checked_mul(2)
            .and_then(|v| v.checked_add(config.inner_size))
            .ok_or_else(|| candle::Error::Msg("qwen35: conv width overflow".into()))?;
        let mut state_bytes = 0usize;
        for index in 0..config.block_count {
            let floats = if config.is_recurrent(index) {
                config
                    .value_heads
                    .checked_mul(config.state_size)
                    .and_then(|v| v.checked_mul(config.state_size))
                    .and_then(|v| {
                        channels
                            .checked_mul(config.conv_kernel - 1)
                            .and_then(|c| v.checked_add(c))
                    })
            } else {
                max_positions
                    .checked_mul(config.num_kv_heads)
                    .and_then(|v| v.checked_mul(config.attention_head_dim))
                    .and_then(|v| v.checked_mul(2))
            }
            .ok_or_else(|| candle::Error::Msg("qwen35: state size overflow".into()))?;
            state_bytes = floats
                .checked_mul(4)
                .and_then(|v| state_bytes.checked_add(v))
                .ok_or_else(|| candle::Error::Msg("qwen35: state byte overflow".into()))?;
        }
        if state_bytes > max_state_bytes {
            bail!("qwen35: state budget {state_bytes} exceeds caller limit {max_state_bytes}")
        }
        let embedding = Arc::new(qtensor(artifact, "token_embd.weight", projection_device)?);
        let output_norm = dense(artifact, "output_norm.weight")?;
        let output = if artifact.tensors.contains_key("output.weight") {
            matrix(artifact, "output.weight", projection_device)?
        } else {
            QMatMul::from_arc(embedding.clone())?
        };
        let mut layers = Vec::with_capacity(config.block_count);
        let mut cache = Vec::with_capacity(config.block_count);
        for index in 0..config.block_count {
            let prefix = format!("blk.{index}");
            let attn_norm = dense(artifact, &format!("{prefix}.attn_norm.weight"))?;
            let post_norm = dense(artifact, &format!("{prefix}.post_attention_norm.weight"))?;
            let gate = matrix(
                artifact,
                &format!("{prefix}.ffn_gate.weight"),
                projection_device,
            )?;
            let up = matrix(
                artifact,
                &format!("{prefix}.ffn_up.weight"),
                projection_device,
            )?;
            let down = matrix(
                artifact,
                &format!("{prefix}.ffn_down.weight"),
                projection_device,
            )?;
            let (mixer, layer_cache) = if config.is_recurrent(index) {
                let recurrence = Recurrent {
                    qkv: matrix(
                        artifact,
                        &format!("{prefix}.attn_qkv.weight"),
                        projection_device,
                    )?,
                    z: matrix(
                        artifact,
                        &format!("{prefix}.attn_gate.weight"),
                        projection_device,
                    )?,
                    alpha: matrix(
                        artifact,
                        &format!("{prefix}.ssm_alpha.weight"),
                        projection_device,
                    )?,
                    beta: matrix(
                        artifact,
                        &format!("{prefix}.ssm_beta.weight"),
                        projection_device,
                    )?,
                    out: matrix(
                        artifact,
                        &format!("{prefix}.ssm_out.weight"),
                        projection_device,
                    )?,
                    conv: dense(artifact, &format!("{prefix}.ssm_conv1d.weight"))?,
                    a: dense(artifact, &format!("{prefix}.ssm_a"))?,
                    dt: dense(artifact, &format!("{prefix}.ssm_dt.bias"))?,
                    norm: dense(artifact, &format!("{prefix}.ssm_norm.weight"))?,
                };
                let fifo = vec![0.0; channels * (config.conv_kernel - 1)];
                let state = vec![0.0; config.value_heads * config.state_size * config.state_size];
                (
                    Mixer::Recurrent(recurrence),
                    Cache::Recurrent { fifo, state },
                )
            } else {
                let attention = Attention {
                    q: matrix(
                        artifact,
                        &format!("{prefix}.attn_q.weight"),
                        projection_device,
                    )?,
                    k: matrix(
                        artifact,
                        &format!("{prefix}.attn_k.weight"),
                        projection_device,
                    )?,
                    v: matrix(
                        artifact,
                        &format!("{prefix}.attn_v.weight"),
                        projection_device,
                    )?,
                    out: matrix(
                        artifact,
                        &format!("{prefix}.attn_output.weight"),
                        projection_device,
                    )?,
                    q_norm: dense(artifact, &format!("{prefix}.attn_q_norm.weight"))?,
                    k_norm: dense(artifact, &format!("{prefix}.attn_k_norm.weight"))?,
                };
                (
                    Mixer::Attention(attention),
                    Cache::Attention {
                        keys: Vec::new(),
                        values: Vec::new(),
                    },
                )
            };
            layers.push(Layer {
                attn_norm,
                post_norm,
                gate,
                up,
                down,
                mixer,
            });
            cache.push(layer_cache);
        }
        Ok(Self {
            config,
            max_positions,
            position: 0,
            embedding,
            output_norm,
            output,
            layers,
            cache,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for cache in &mut self.cache {
            match cache {
                Cache::Recurrent { fifo, state } => {
                    fifo.fill(0.0);
                    state.fill(0.0);
                }
                Cache::Attention { keys, values } => {
                    keys.clear();
                    values.clear();
                }
            }
        }
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.position != 0 || tokens.is_empty() || tokens.len() > self.max_positions {
            bail!("qwen35: prefill requires an empty cache and bounded nonempty tokens")
        }
        let (last, prefix) = tokens
            .split_last()
            .ok_or_else(|| candle::Error::Msg("qwen35: empty prefill".into()))?;
        for &token in prefix {
            self.decode_without_logits(token)?;
        }
        self.decode(*last)
    }

    /// Appends one token and returns its raw vocabulary logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decode_observed(token, None)
    }

    fn decode_without_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decode_observed_inner(token, None, false)
    }

    fn decode_observed(
        &mut self,
        token: u32,
        observe: Option<ModelObserver<'_>>,
    ) -> Result<Vec<f32>> {
        self.decode_observed_inner(token, observe, true)
    }

    fn decode_observed_inner(
        &mut self,
        token: u32,
        mut observe: Option<ModelObserver<'_>>,
        emit_logits: bool,
    ) -> Result<Vec<f32>> {
        if self.position >= self.max_positions || token as usize >= self.config.vocab_size {
            bail!("qwen35: token or position exceeds admitted ceiling")
        }
        let ids = Tensor::from_slice(&[token], 1, &Device::Cpu)?;
        let mut x = self
            .embedding
            .embedding(&ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (layer, cache)) in self.layers.iter().zip(self.cache.iter_mut()).enumerate() {
            if let Some(observer) = observe.as_mut() {
                let mut layer_observer = |name: &str, values: &[f32]| observer(index, name, values);
                x = layer.forward(
                    &x,
                    cache,
                    &self.config,
                    self.position,
                    Some(&mut layer_observer),
                )?;
                observer(index, "l_out", &x);
            } else {
                x = layer.forward(&x, cache, &self.config, self.position, None)?;
            }
        }
        let logits = if emit_logits {
            rms_norm(&mut x, &self.output_norm, self.config.rms_norm_eps);
            projection(&self.output, &x)?
        } else {
            Vec::new()
        };
        self.position += 1;
        Ok(logits)
    }
}

impl Layer {
    fn forward(
        &self,
        input: &[f32],
        cache: &mut Cache,
        config: &Config,
        position: usize,
        mut observe: Option<ComponentObserver<'_>>,
    ) -> Result<Vec<f32>> {
        let mut normalized = input.to_vec();
        rms_norm(&mut normalized, &self.attn_norm, config.rms_norm_eps);
        if let Some(observer) = observe.as_mut() {
            observer("attn_norm", &normalized);
        }
        #[cfg(test)]
        let mixer_started = std::time::Instant::now();
        let mixed = match (&self.mixer, cache) {
            (Mixer::Recurrent(weights), Cache::Recurrent { fifo, state }) => {
                if let Some(observer) = &mut observe {
                    weights.forward(&normalized, fifo, state, config, Some(&mut **observer))?
                } else {
                    weights.forward(&normalized, fifo, state, config, None)?
                }
            }
            (Mixer::Attention(weights), Cache::Attention { keys, values }) => {
                if let Some(observer) = &mut observe {
                    weights.forward(
                        &normalized,
                        keys,
                        values,
                        config,
                        position,
                        Some(&mut **observer),
                    )?
                } else {
                    weights.forward(&normalized, keys, values, config, position, None)?
                }
            }
            _ => bail!("qwen35: layer cache type mismatch"),
        };
        #[cfg(test)]
        bench_profile::record_stage(
            "mixer.total",
            input.len(),
            mixed.len(),
            mixer_started.elapsed(),
        );
        let mut residual = input
            .iter()
            .zip(&mixed)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        if let Some(observer) = observe.as_mut() {
            observer("attn_residual", &residual);
        }
        let mut normalized = residual.clone();
        rms_norm(&mut normalized, &self.post_norm, config.rms_norm_eps);
        if let Some(observer) = observe.as_mut() {
            observer("attn_post_norm", &normalized);
        }
        let gate = projection(&self.gate, &normalized)?;
        if let Some(observer) = observe.as_mut() {
            observer("ffn_gate", &gate);
        }
        let up = projection(&self.up, &normalized)?;
        if let Some(observer) = observe.as_mut() {
            observer("ffn_up", &up);
        }
        let product = gate
            .iter()
            .zip(&up)
            .map(|(&a, &b)| silu(a) * b)
            .collect::<Vec<_>>();
        if let Some(observer) = observe.as_mut() {
            observer("ffn_swiglu", &product);
        }
        let ffn = projection(&self.down, &product)?;
        if let Some(observer) = observe.as_mut() {
            observer("ffn_out", &ffn);
        }
        for (value, update) in residual.iter_mut().zip(ffn) {
            *value += update;
        }
        if let Some(observer) = observe.as_mut() {
            observer("post_ffn", &residual);
        }
        Ok(residual)
    }
}

impl Recurrent {
    fn forward(
        &self,
        input: &[f32],
        fifo: &mut [f32],
        state: &mut [f32],
        config: &Config,
        mut observe: Option<ComponentObserver<'_>>,
    ) -> Result<Vec<f32>> {
        let dim = config.state_size;
        #[cfg(test)]
        let recurrent_started = std::time::Instant::now();
        let heads = config.value_heads;
        let key_heads = config.key_groups;
        let key_width = dim * key_heads;
        let channels = key_width * 2 + config.inner_size;
        let kernel = config.conv_kernel;
        let qkv = projection(&self.qkv, input)?;
        if let Some(observer) = observe.as_mut() {
            observer("linear_attn_qkv_mixed", &qkv);
        }
        let z = projection(&self.z, input)?;
        let alpha = projection(&self.alpha, input)?;
        let beta = projection(&self.beta, input)?;
        #[cfg(test)]
        let conv_started = std::time::Instant::now();
        let mut convolved = vec![0.0; channels];
        let mut observed_raw = observe.as_ref().map(|_| vec![0.0; channels]);
        for channel in 0..channels {
            let old = &mut fifo[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
            let weights = &self.conv[channel * kernel..(channel + 1) * kernel];
            let mut sum = 0.0;
            for (&previous, &weight) in old.iter().zip(weights) {
                sum = previous.mul_add(weight, sum);
            }
            sum = qkv[channel].mul_add(weights[kernel - 1], sum);
            if let Some(raw) = observed_raw.as_mut() {
                raw[channel] = sum;
            }
            if kernel > 1 {
                old.rotate_left(1);
                old[kernel - 2] = qkv[channel];
            }
            convolved[channel] = silu(sum);
        }
        #[cfg(test)]
        bench_profile::record_stage("recurrent.conv", channels, channels, conv_started.elapsed());
        if let Some(observer) = observe.as_mut() {
            if let Some(raw) = observed_raw.as_ref() {
                observer("conv_output_raw", raw);
            }
            observer("conv_output_silu", &convolved);
        }
        let mut q = convolved[..key_width].to_vec();
        let mut k = convolved[key_width..2 * key_width].to_vec();
        for h in 0..key_heads {
            l2_norm(&mut q[h * dim..(h + 1) * dim], config.rms_norm_eps);
            l2_norm(&mut k[h * dim..(h + 1) * dim], config.rms_norm_eps);
        }
        let v = &convolved[2 * key_width..];
        let mut observed_parameters = observe.as_ref().map(|_| {
            (
                Vec::with_capacity(heads),
                Vec::with_capacity(heads),
                Vec::with_capacity(heads),
            )
        });
        let mut output = vec![0.0; config.inner_size];
        let q_scale = (dim as f32).sqrt().recip();
        let mut observed_input = observe.as_ref().map(|_| {
            (
                Vec::with_capacity(config.inner_size),
                Vec::with_capacity(config.inner_size),
            )
        });
        let mut observed_raw = observe.as_ref().map(|_| vec![0.0; config.inner_size]);
        let mut dot_groups = vec![0.0; 64 * dim];
        let mut dot_output = vec![0.0; dim];
        #[cfg(test)]
        let state_started = std::time::Instant::now();
        for head in 0..heads {
            let key_head = head % key_heads;
            let qh = &q[key_head * dim..(key_head + 1) * dim];
            let kh = &k[key_head * dim..(key_head + 1) * dim];
            if let Some((observed_q, observed_k)) = observed_input.as_mut() {
                observed_q.extend_from_slice(qh);
                observed_k.extend_from_slice(kh);
            }
            let vh = &v[head * dim..(head + 1) * dim];
            let sh = &mut state[head * dim * dim..(head + 1) * dim * dim];
            let alpha_softplus = softplus(alpha[head] + self.dt[head]);
            let gate = alpha_softplus * self.a[head];
            let decay = gate.exp();
            let beta = sigmoid(beta[head]);
            if let Some((observed_softplus, observed_gate, observed_beta)) =
                observed_parameters.as_mut()
            {
                observed_softplus.push(alpha_softplus);
                observed_gate.push(gate);
                observed_beta.push(beta);
            }
            for value in sh.iter_mut() {
                *value *= decay;
            }
            let mut delta = vec![0.0; dim];
            recurrent_matvec(sh, kh, dim, &mut dot_groups, &mut dot_output);
            for value in 0..dim {
                delta[value] = (vh[value] - dot_output[value]) * beta;
            }
            for key in 0..dim {
                for value in 0..dim {
                    sh[key * dim + value] = kh[key].mul_add(delta[value], sh[key * dim + value]);
                }
            }
            let oh = &mut output[head * dim..(head + 1) * dim];
            recurrent_matvec(sh, qh, dim, &mut dot_groups, oh);
            for value in oh.iter_mut() {
                *value *= q_scale;
            }
            if let Some(raw) = observed_raw.as_mut() {
                raw[head * dim..(head + 1) * dim].copy_from_slice(oh);
            }
            rms_norm(oh, &self.norm, config.rms_norm_eps);
            for (value, gate) in oh.iter_mut().zip(&z[head * dim..(head + 1) * dim]) {
                *value *= silu(*gate);
            }
        }
        #[cfg(test)]
        bench_profile::record_stage(
            "recurrent.state",
            heads * dim * dim,
            output.len(),
            state_started.elapsed(),
        );
        if let Some(observer) = observe.as_mut() {
            if let Some((observed_q, observed_k)) = observed_input.as_ref() {
                observer("q_conv_predelta", observed_q);
                observer("k_conv_predelta", observed_k);
                observer("v_conv_predelta", v);
                observer(
                    "q_in",
                    &observed_q
                        .iter()
                        .map(|value| value * q_scale)
                        .collect::<Vec<_>>(),
                );
                observer("k_in", observed_k);
            }
            if let Some((observed_softplus, observed_gate, observed_beta)) =
                observed_parameters.as_ref()
            {
                observer("a_softplus", observed_softplus);
                observer("gate", observed_gate);
                observer("beta_sigmoid", observed_beta);
            }
            if let Some(raw) = observed_raw.as_ref() {
                observer("attn_output", raw);
            }
            observer("final_output", &output);
        }
        let result = projection(&self.out, &output)?;
        if let Some(observer) = observe.as_mut() {
            observer("linear_attn_out", &result);
        }
        #[cfg(test)]
        bench_profile::record_stage(
            "recurrent.total",
            input.len(),
            result.len(),
            recurrent_started.elapsed(),
        );
        Ok(result)
    }
}

fn rope(head: &mut [f32], position: usize, rotary: usize, theta: f32) {
    let half = rotary / 2;
    let theta_scale = theta.powf(-2.0 / rotary as f32);
    let mut angle = position as f32;
    for index in 0..half {
        let sin = angle.sin();
        let cos = angle.cos();
        let a = head[index];
        let b = head[index + half];
        head[index] = a.mul_add(cos, -(b * sin));
        head[index + half] = a.mul_add(sin, b * cos);
        angle *= theta_scale;
    }
}

impl Attention {
    fn forward(
        &self,
        input: &[f32],
        keys: &mut Vec<Vec<f32>>,
        values: &mut Vec<Vec<f32>>,
        config: &Config,
        position: usize,
        mut observe: Option<ComponentObserver<'_>>,
    ) -> Result<Vec<f32>> {
        #[cfg(test)]
        let attention_started = std::time::Instant::now();
        let dim = config.attention_head_dim;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_kv_heads;
        let repeats = heads / kv_heads;
        let qg = projection(&self.q, input)?;
        let mut k = projection(&self.k, input)?;
        let v = projection(&self.v, input)?;
        if let Some(observer) = observe.as_mut() {
            observer("Qcur_full", &qg);
            observer("Vcur", &v);
        }
        let mut queries = vec![0.0; heads * dim];
        let mut gates = vec![0.0; heads * dim];
        for head in 0..heads {
            queries[head * dim..(head + 1) * dim]
                .copy_from_slice(&qg[head * 2 * dim..head * 2 * dim + dim]);
            gates[head * dim..(head + 1) * dim]
                .copy_from_slice(&qg[head * 2 * dim + dim..(head + 1) * 2 * dim]);
            let qh = &mut queries[head * dim..(head + 1) * dim];
            rms_norm(qh, &self.q_norm, config.rms_norm_eps);
        }
        if let Some(observer) = observe.as_mut() {
            observer("Qcur_normed", &queries);
            observer("gate_reshaped", &gates);
        }
        for head in 0..heads {
            rope(
                &mut queries[head * dim..(head + 1) * dim],
                position,
                config.rope_dimension_count,
                config.rope_theta,
            );
        }
        for head in 0..kv_heads {
            let kh = &mut k[head * dim..(head + 1) * dim];
            rms_norm(kh, &self.k_norm, config.rms_norm_eps);
        }
        if let Some(observer) = observe.as_mut() {
            observer("Kcur_normed", &k);
        }
        for head in 0..kv_heads {
            let kh = &mut k[head * dim..(head + 1) * dim];
            rope(kh, position, config.rope_dimension_count, config.rope_theta);
        }
        if let Some(observer) = observe.as_mut() {
            observer("Qcur", &queries);
            observer("Kcur", &k);
        }
        keys.push(k);
        values.push(v);
        let mut combined = vec![0.0; heads * dim];
        let mut value_groups = vec![0.0; 64 * dim];
        let scale = (dim as f32).sqrt().recip();
        let mut observed_kq = observe
            .as_ref()
            .map(|_| Vec::with_capacity(heads * keys.len()));
        let mut observed_softmax = observe
            .as_ref()
            .map(|_| Vec::with_capacity(heads * keys.len()));
        #[cfg(test)]
        let scores_started = std::time::Instant::now();
        for head in 0..heads {
            let kv_head = head / repeats;
            let qh = &queries[head * dim..(head + 1) * dim];
            let mut scores = Vec::with_capacity(keys.len());
            for token in keys.iter() {
                let kh = &token[kv_head * dim..(kv_head + 1) * dim];
                let dot = ggml_f32_dot(kh.iter().copied().zip(qh.iter().copied()));
                if let Some(values) = observed_kq.as_mut() {
                    values.push(dot);
                }
                scores.push(dot * scale);
            }
            attention_softmax(&mut scores);
            if let Some(values) = observed_softmax.as_mut() {
                values.extend_from_slice(&scores);
            }
            let oh = &mut combined[head * dim..(head + 1) * dim];
            attention_weighted_values(values, &scores, kv_head, dim, &mut value_groups, oh);
        }
        #[cfg(test)]
        bench_profile::record_stage(
            "attention.scores_values",
            keys.len(),
            heads * dim,
            scores_started.elapsed(),
        );
        if let Some(observer) = observe.as_mut() {
            if let Some(values) = observed_kq.as_ref() {
                observer("kq", values);
            }
            if let Some(values) = observed_softmax.as_ref() {
                observer("kq_soft_max", values);
            }
            observer("attn_pregate", &combined);
        }
        let gates = gates.into_iter().map(sigmoid).collect::<Vec<_>>();
        if let Some(observer) = observe.as_mut() {
            observer("gate_sigmoid", &gates);
        }
        for (value, gate) in combined.iter_mut().zip(gates) {
            *value *= gate;
        }
        if let Some(observer) = observe.as_mut() {
            observer("attn_gated", &combined);
        }
        let result = projection(&self.out, &combined)?;
        if let Some(observer) = observe.as_mut() {
            observer("attn_output", &result);
        }
        #[cfg(test)]
        bench_profile::record_stage(
            "attention.total",
            input.len(),
            result.len(),
            attention_started.elapsed(),
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::admission::AdmissionLimits;
    use std::path::Path;

    #[test]
    fn row_major_recurrent_matvec_keeps_ggml_dot_order() {
        for dim in [1, 15, 16, 17, 63, 64, 65, 128] {
            let state = (0..dim * dim)
                .map(|index| ((index * 37 + 19) % 239) as f32 / 131.0 - 0.8)
                .collect::<Vec<_>>();
            let vector = (0..dim)
                .map(|index| ((index * 23 + 11) % 97) as f32 / 97.0)
                .collect::<Vec<_>>();
            let mut groups = vec![0.0; 64 * dim];
            let mut actual = vec![0.0; dim];
            recurrent_matvec(&state, &vector, dim, &mut groups, &mut actual);
            for (index, &value) in actual.iter().enumerate() {
                let expected = recurrent_dot(&state, dim, &vector, index);
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "dim {dim}, index {index}"
                );
            }
        }
    }

    #[test]
    fn token_major_attention_values_keep_ggml_dot_order() {
        for len in [1, 15, 16, 17, 63, 64, 65, 1024] {
            for dim in [1, 4, 128] {
                let values = (0..len)
                    .map(|token| {
                        (0..dim * 2)
                            .map(|index| ((token * 97 + index * 13) % 211) as f32 / 117.0 - 0.8)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let scores = (0..len)
                    .map(|token| ((token * 23 + 17) % 101) as f32 / 101.0)
                    .collect::<Vec<_>>();
                let mut groups = vec![0.0; dim * 64];
                let mut actual = vec![0.0; dim];
                attention_weighted_values(&values, &scores, 1, dim, &mut groups, &mut actual);
                for (index, &value) in actual.iter().enumerate() {
                    let expected = ggml_f32_dot(
                        values
                            .iter()
                            .map(|token| token[dim + index])
                            .zip(scores.iter().copied()),
                    );
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "len {len}, dim {dim}, index {index}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires sealed 1024-token Edge input, exact GGUF, CUDA, and bounded runner"]
    fn hybrid_cuda_sealed_context_profile() -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::time::Instant;

        let input_path = std::env::var("QWEN35_CONTEXT_TOKENS_FILE").map_err(|error| {
            candle::Error::Msg(format!("set QWEN35_CONTEXT_TOKENS_FILE: {error}"))
        })?;
        let input_bytes = std::fs::read(&input_path)?;
        let input_hash = format!("{:x}", Sha256::digest(&input_bytes));
        if input_hash != "fceba3897357a86a2e9063b38e528549ac5df93661487f2f7fda0789e014fbdb" {
            bail!("qwen35: sealed context token file hash changed")
        }
        let tokens: Vec<u32> = serde_json::from_slice(&input_bytes)
            .map_err(|error| candle::Error::Msg(format!("context token IDs: {error}")))?;
        if tokens.len() != 1024 {
            bail!("qwen35: sealed context requires exactly 1024 tokens")
        }
        let gate_path = std::env::var("QWEN35_CONTEXT_GATE_FILE").map_err(|error| {
            candle::Error::Msg(format!("set QWEN35_CONTEXT_GATE_FILE: {error}"))
        })?;
        let gate_bytes = std::fs::read(&gate_path)?;
        if format!("{:x}", Sha256::digest(&gate_bytes))
            != "8b1996d6347c936b4410f393b87732ad21e19ef56d2a1054d1df284d622ada4e"
        {
            bail!("qwen35: sealed Edge gate hash changed")
        }
        let gate: serde_json::Value = serde_json::from_slice(&gate_bytes)
            .map_err(|error| candle::Error::Msg(format!("Edge gate JSON: {error}")))?;
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let started = Instant::now();
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        let admission_ms = started.elapsed().as_secs_f64() * 1000.0;
        let device = Device::new_cuda(0)?;
        let started = Instant::now();
        let mut model = HybridGpuModel::load(&mut artifact, 1088, 134_217_728, &device)?;
        device.synchronize()?;
        let materialize_ms = started.elapsed().as_secs_f64() * 1000.0;
        bench_profile::start();
        let started = Instant::now();
        let mut logits = model.prefill(&tokens)?;
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
        let first_logits = logits.clone();
        let mut trace = Vec::new();
        for (index, selected) in [19u32, 248046].into_iter().enumerate() {
            let actual = top_ten(&logits);
            let expected = gate["context_baseline"]["output"]["top_logprobs"][index]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("Edge top-logprob trace missing".into()))?;
            if actual.len() != expected.len() || actual[0].0 != selected as usize {
                bail!("qwen35: sealed context selected token differs at {index}")
            }
            let mut actual_ids = actual.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let mut expected_ids = expected
                .iter()
                .map(|row| {
                    row[0]
                        .as_u64()
                        .and_then(|id| usize::try_from(id).ok())
                        .ok_or_else(|| candle::Error::Msg("invalid Edge top-ten token ID".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            actual_ids.sort_unstable();
            expected_ids.sort_unstable();
            if actual_ids != expected_ids {
                bail!("qwen35: sealed context top-ten token set differs at {index}")
            }
            let max_error = actual
                .iter()
                .map(|(id, logprob)| {
                    let row = expected
                        .iter()
                        .find(|row| row[0].as_u64() == Some(*id as u64));
                    row.and_then(|row| row[1].as_f64())
                        .map(|reference| (f64::from(*logprob) - reference).abs())
                        .unwrap_or(f64::INFINITY)
                })
                .fold(0.0f64, f64::max);
            if max_error > 0.01 {
                bail!("qwen35: sealed context logprob error {max_error} exceeds 0.01 at {index}")
            }
            trace.push(
                serde_json::json!({"selected": selected, "max_abs_raw_logprob_error": max_error}),
            );
            if index == 0 {
                let started = Instant::now();
                logits = model.decode(selected)?;
                trace.push(serde_json::json!({"cached_decode_ms": started.elapsed().as_secs_f64() * 1000.0}));
            }
        }
        let stages = bench_profile::take();
        let repeat = if std::env::var("QWEN35_CONTEXT_REPEAT").as_deref() == Ok("1") {
            let Device::Cuda(cuda) = &device else {
                bail!("qwen35: context repeat requires CUDA")
            };
            let gpu_free = || {
                cuda.cuda_stream()
                    .context()
                    .mem_get_info()
                    .map(|(free, _)| free)
                    .map_err(|error| candle::Error::Msg(format!("CUDA memory query: {error}")))
            };
            let second_logits = logits;
            device.synchronize()?;
            let free_after_first = gpu_free()?;
            model.reset();
            let started = Instant::now();
            if model.position() != 0 || model.prefill(&tokens)? != first_logits {
                bail!("qwen35: sealed context reset replay differed")
            }
            let replay_ms = started.elapsed().as_secs_f64() * 1000.0;
            if model.decode(19)? != second_logits {
                bail!("qwen35: sealed context cached decode replay differed")
            }
            device.synchronize()?;
            let free_after_replay = gpu_free()?;
            model.reset();
            let mut checked_tokens = 0usize;
            let cancelled = model.prefill_with_cancel(&tokens, &mut || {
                checked_tokens += 1;
                checked_tokens == 512
            });
            if cancelled
                .as_ref()
                .err()
                .is_none_or(|error| !error.to_string().contains("cancelled at token boundary"))
                || model.position() != 0
            {
                bail!("qwen35: sealed context cancellation did not clear state")
            }
            if model.prefill(&tokens)? != first_logits {
                bail!("qwen35: sealed context post-cancellation replay differed")
            }
            device.synchronize()?;
            let free_after_cancel_replay = gpu_free()?;
            if free_after_first.saturating_sub(free_after_cancel_replay) > 256 * 1024 * 1024 {
                bail!("qwen35: CUDA free memory fell by over 256 MiB across context sessions")
            }
            Some(serde_json::json!({
                "replay_ms": replay_ms,
                "reset_exact": true,
                "cached_decode_exact": true,
                "cancelled_after_checks": checked_tokens,
                "post_cancel_replay_exact": true,
                "gpu_free_after_first": free_after_first,
                "gpu_free_after_replay": free_after_replay,
                "gpu_free_after_cancel_replay": free_after_cancel_replay,
            }))
        } else {
            None
        };
        eprintln!(
            "QWEN35_CONTEXT_PROFILE_JSON={}",
            serde_json::json!({
                "schema": "candle.qwen35.sealed-context-profile.v1",
                "prompt_sha256": input_hash,
                "model_sha256": "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
                "admission_ms": admission_ms,
                "materialize_ms": materialize_ms,
                "prefill_ms": prefill_ms,
                "prefill_tokens_per_second": 1_024_000.0 / prefill_ms,
                "trace": trace,
                "stages": stages,
                "position": model.position(),
                "repeat": repeat,
            })
        );
        Ok(())
    }

    fn top_ten(logits: &[f32]) -> Vec<(usize, f32)> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = max + logits.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        let mut ranked = logits
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, x - log_sum))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(10);
        ranked
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires the hash-pinned external GGUF and an available CUDA device"]
    fn cuda_q8_projection_proof() -> Result<()> {
        use candle::quantized::{cuda::set_force_dmmv, ggml_file::qtensor_from_ggml};
        use std::time::Instant;

        struct ResetDmmv;
        impl Drop for ResetDmmv {
            fn drop(&mut self) {
                set_force_dmmv(false);
            }
        }
        let _reset_dmmv = ResetDmmv;
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let token = fixture["prompt_token_ids"][0]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| candle::Error::Msg("first fixture token missing".into()))?;
        let mut model = CpuModel::load(&mut artifact, 64, 128 * 1024 * 1024)?;
        let mut activation = None;
        model.decode_observed(
            token,
            Some(&mut |layer, name, values: &[f32]| {
                if layer == 0 && name == "attn_post_norm" {
                    activation = Some(values.to_vec());
                }
            }),
        )?;
        let activation =
            activation.ok_or_else(|| candle::Error::Msg("layer zero activation missing".into()))?;
        let cpu_output = projection(&model.layers[0].gate, &activation)?;
        let cpu_weight = match &model.layers[0].gate {
            QMatMul::QTensor(weight) => weight,
            _ => bail!("qwen35: admitted layer zero gate is not quantized"),
        };
        if cpu_weight.dtype() != candle::quantized::GgmlDType::Q8_0 {
            bail!("qwen35: layer zero gate is not Q8_0")
        }
        let device = Device::new_cuda(0)?;
        let weight_bytes = cpu_weight.data()?;
        let started = Instant::now();
        let gpu_weight = qtensor_from_ggml(
            cpu_weight.dtype(),
            &weight_bytes,
            cpu_weight.shape().dims().to_vec(),
            &device,
        )?;
        device.synchronize()?;
        let weight_h2d_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let gpu_projection = QMatMul::from_qtensor(gpu_weight)?;
        let started = Instant::now();
        let gpu_input = Tensor::from_slice(&activation, (1, activation.len()), &Device::Cpu)?
            .to_device(&device)?;
        device.synchronize()?;
        let activation_h2d_ms = started.elapsed().as_secs_f64() * 1_000.0;
        for force_dmmv in [false, true] {
            set_force_dmmv(force_dmmv);
            // Warm the selected kernel before the separately timed calls.
            let forward = |input: &Tensor| {
                if force_dmmv {
                    candle::Module::forward(&gpu_projection, input)
                } else {
                    gpu_projection.forward_ggml_q8_0(input)
                }
            };
            let warm = forward(&gpu_input)?;
            device.synchronize()?;
            drop(warm);
            let mut kernel_ms = Vec::new();
            let mut gpu_output = None;
            for _ in 0..3 {
                let started = Instant::now();
                let output = forward(&gpu_input)?;
                device.synchronize()?;
                kernel_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                gpu_output = Some(output);
            }
            let started = Instant::now();
            let output = gpu_output
                .ok_or_else(|| candle::Error::Msg("CUDA output missing".into()))?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let output_d2h_ms = started.elapsed().as_secs_f64() * 1_000.0;
            if output.len() != cpu_output.len() {
                bail!("qwen35: CUDA projection output length mismatch")
            }
            let mut max_abs = 0f32;
            let mut squared = 0f64;
            let mut max_index = 0usize;
            for (index, (&actual, &expected)) in output.iter().zip(&cpu_output).enumerate() {
                if !actual.is_finite() {
                    bail!("qwen35: non-finite CUDA projection output at {index}")
                }
                let error = (actual - expected).abs();
                squared += f64::from(error).powi(2);
                if error > max_abs {
                    max_abs = error;
                    max_index = index;
                }
            }
            eprintln!(
                "QWEN35_CUDA_PROJECTION_JSON={}",
                serde_json::json!({
                    "force_dmmv": force_dmmv,
                    "output_len": output.len(),
                    "max_abs": max_abs,
                    "max_index": max_index,
                    "rms": (squared / output.len() as f64).sqrt(),
                    "weight_h2d_ms": weight_h2d_ms,
                    "activation_h2d_ms": activation_h2d_ms,
                    "kernel_ms": kernel_ms,
                    "output_d2h_ms": output_d2h_ms,
                })
            );
            if !force_dmmv && max_abs > 0.00001 {
                bail!("qwen35: default CUDA Q8_0 projection exceeded 1e-5 CPU component error")
            }
        }
        set_force_dmmv(true);
        let forced_scope = gpu_projection
            .forward_ggml_q8_0(&gpu_input)
            .err()
            .ok_or_else(|| {
                candle::Error::Msg("scoped CUDA forced dequantization was accepted".into())
            })?;
        if !forced_scope
            .to_string()
            .contains("rejects forced dequantization")
        {
            bail!("qwen35: forced scoped CUDA call failed for unexpected reason: {forced_scope}")
        }
        set_force_dmmv(false);
        let doubled = Tensor::from_slice(
            &[activation.as_slice(), activation.as_slice()].concat(),
            (2, activation.len()),
            &Device::Cpu,
        )?
        .to_device(&device)?;
        let unsupported = gpu_projection
            .forward_ggml_q8_0(&doubled)
            .err()
            .ok_or_else(|| candle::Error::Msg("scoped CUDA batch two was accepted".into()))?;
        if !unsupported
            .to_string()
            .contains("requires the F32 MMVQ path")
        {
            bail!("qwen35: scoped CUDA batch two failed for unexpected reason: {unsupported}")
        }
        Ok(())
    }

    #[test]
    fn rope_rotates_only_declared_partial_dimensions() {
        let mut head = [1.0, 0.0, 0.0, 1.0, 7.0, 8.0];
        rope(&mut head, 1, 4, 10_000.0);
        assert!((head[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((head[2] - 1.0f32.sin()).abs() < 1e-6);
        assert!((head[1] - (-0.01f32).sin()).abs() < 1e-6);
        assert!((head[3] - 0.01f32.cos()).abs() < 1e-6);
        assert_eq!(&head[4..], &[7.0, 8.0]);
    }

    #[test]
    #[ignore = "requires hash-pinned external GGUF and bounded CPU job"]
    fn exact_layer_components() -> Result<()> {
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let output = std::env::var("QWEN35_COMPONENT_DIR")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_COMPONENT_DIR: {error}")))?;
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let prompt = fixture["prompt_token_ids"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("prompt IDs missing".into()))?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid prompt token".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if prompt.len() != 49 {
            bail!("qwen35: expected 49 fixture prompt tokens")
        }
        let capture_index = std::env::var("QWEN35_CAPTURE_TOKEN_INDEX")
            .unwrap_or_else(|_| "48".into())
            .parse::<usize>()
            .map_err(|error| candle::Error::Msg(format!("capture token index: {error}")))?;
        if capture_index >= prompt.len() {
            bail!("qwen35: capture index exceeds fixture prompt")
        }
        let capture_layer = std::env::var("QWEN35_CAPTURE_LAYER_INDEX")
            .unwrap_or_else(|_| "2".into())
            .parse::<usize>()
            .map_err(|error| candle::Error::Msg(format!("capture layer index: {error}")))?;
        let mut model = CpuModel::load(&mut artifact, 64, 128 * 1024 * 1024)?;
        if capture_layer >= model.config.block_count {
            bail!("qwen35: capture layer exceeds model")
        }
        let mut vectors = Vec::new();
        let timeline = std::env::var("QWEN35_CAPTURE_TIMELINE").as_deref() == Ok("1");
        for (index, token) in prompt.iter().copied().take(capture_index + 1).enumerate() {
            if index == capture_index || timeline {
                let mut capture = |layer, name: &str, values: &[f32]| {
                    if timeline && (name == "l_out" || (layer == 19 && name == "Kcur")) {
                        vectors.push((format!("timeline-{name}-{layer}-{index}"), values.to_vec()));
                    }
                    if index == capture_index
                        && (layer <= 2 || layer == capture_layer || name == "l_out")
                    {
                        vectors.push((format!("{name}-{layer}"), values.to_vec()));
                    }
                };
                model.decode_observed(token, Some(&mut capture))?;
            } else {
                model.decode(token)?;
            }
        }
        if vectors.len() < model.config.block_count {
            bail!("qwen35: incomplete layer capture")
        }
        std::fs::create_dir_all(&output)?;
        for (name, values) in vectors {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(Path::new(&output).join(format!("{name}.bin")), bytes)?;
            eprintln!("qwen35 {name} first={:?}", &values[..4]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires hash-pinned external GGUF and bounded CPU job"]
    fn exact_reference_trace_and_reset() -> Result<()> {
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let prompt = fixture["prompt_token_ids"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("prompt IDs missing".into()))?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid prompt token".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(prompt.len(), 49);
        let mut model = CpuModel::load(&mut artifact, 64, 128 * 1024 * 1024)?;
        let first = model.prefill(&prompt)?;
        let mut logits = first.clone();
        let mut failures = Vec::new();
        for (index, expected) in fixture["expected"]["probability_trace"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference trace missing".into()))?
            .iter()
            .enumerate()
        {
            let actual = top_ten(&logits);
            let expected_top = expected["top_logprobs"]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("reference top ten missing".into()))?;
            eprintln!("qwen35 trace {index}: {actual:?}");
            let mut actual_ids = actual.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let mut expected_ids = expected_top
                .iter()
                .map(|entry| {
                    entry["token_id"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| candle::Error::Msg("invalid reference token ID".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            actual_ids.sort_unstable();
            expected_ids.sort_unstable();
            if actual_ids != expected_ids {
                failures.push(format!("trace {index} top-ten token set differs"));
            }
            for entry in expected_top {
                let id = entry["token_id"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid reference token ID".into()))?;
                let expected_logprob = entry["logprob"]
                    .as_f64()
                    .ok_or_else(|| candle::Error::Msg("invalid reference logprob".into()))?;
                let actual_logprob = actual
                    .iter()
                    .find(|(token, _)| *token == id)
                    .map(|(_, logprob)| *logprob as f64)
                    .unwrap_or(f64::NAN);
                if (actual_logprob - expected_logprob).abs() > 0.01 {
                    failures.push(format!("trace {index} token {id}: actual {actual_logprob}, expected {expected_logprob}"));
                }
            }
            let selected = expected["selected_token_id"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| candle::Error::Msg("invalid selected token ID".into()))?;
            if actual[0].0 != selected as usize {
                failures.push(format!(
                    "trace {index} greedy token: actual {}, expected {selected}",
                    actual[0].0
                ));
            }
            if index == 0 {
                logits = model.decode(selected)?;
            }
        }
        model.reset();
        for token in [10u32, 20, 30] {
            model.decode(token)?;
        }
        model.reset();
        assert_eq!(model.position(), 0);
        if model.prefill(&prompt)? != first {
            failures.push("reset did not exactly replay Candle initial raw logits".into());
        }
        if !failures.is_empty() {
            bail!("qwen35 reference gate: {}", failures.join("; "))
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires hash-pinned external GGUF and bounded CUDA job"]
    fn hybrid_cuda_reference_trace_and_reset() -> Result<()> {
        fn gpu_free(device: &Device) -> Result<usize> {
            let Device::Cuda(cuda) = device else {
                bail!("qwen35: CUDA memory query requires CUDA device")
            };
            cuda.cuda_stream()
                .context()
                .mem_get_info()
                .map(|(free, _)| free)
                .map_err(|error| candle::Error::Msg(format!("CUDA memory query: {error}")))
        }
        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        if HybridGpuModel::load(&mut artifact, 64, 128 * 1024 * 1024, &Device::Cpu).is_ok() {
            bail!("qwen35: hybrid CUDA accepted a CPU device")
        }
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let prompt = fixture["prompt_token_ids"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("prompt IDs missing".into()))?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid prompt token".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if prompt.len() != 49 {
            bail!("qwen35: reference prompt length changed")
        }
        let device = Device::new_cuda(0)?;
        let mut model = HybridGpuModel::load(&mut artifact, 64, 128 * 1024 * 1024, &device)?;
        device.synchronize()?;
        let free_after_load = gpu_free(&device)?;
        let first = model.prefill(&prompt)?;
        let mut logits = first.clone();
        let mut failures = Vec::new();
        let mut max_logprob_error = 0f64;
        for (index, expected) in fixture["expected"]["probability_trace"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference trace missing".into()))?
            .iter()
            .enumerate()
        {
            let actual = top_ten(&logits);
            let expected_top = expected["top_logprobs"]
                .as_array()
                .ok_or_else(|| candle::Error::Msg("reference top ten missing".into()))?;
            eprintln!("qwen35 CUDA trace {index}: {actual:?}");
            let mut actual_ids = actual.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let mut expected_ids = expected_top
                .iter()
                .map(|entry| {
                    entry["token_id"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| candle::Error::Msg("invalid reference token ID".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            actual_ids.sort_unstable();
            expected_ids.sort_unstable();
            if actual_ids != expected_ids {
                failures.push(format!("trace {index} top-ten token set differs"));
            }
            for entry in expected_top {
                let id = entry["token_id"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid reference token ID".into()))?;
                let expected_logprob = entry["logprob"]
                    .as_f64()
                    .ok_or_else(|| candle::Error::Msg("invalid reference logprob".into()))?;
                let actual_logprob = actual
                    .iter()
                    .find(|(token, _)| *token == id)
                    .map(|(_, logprob)| *logprob as f64)
                    .unwrap_or(f64::NAN);
                let error = (actual_logprob - expected_logprob).abs();
                if error > max_logprob_error {
                    max_logprob_error = error;
                }
                if error > 0.01 {
                    failures.push(format!("trace {index} token {id}: actual {actual_logprob}, expected {expected_logprob}"));
                }
            }
            let selected = expected["selected_token_id"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| candle::Error::Msg("invalid selected token ID".into()))?;
            if actual[0].0 != selected as usize {
                failures.push(format!(
                    "trace {index} greedy token: actual {}, expected {selected}",
                    actual[0].0
                ));
            }
            if index == 0 {
                logits = model.decode(selected)?;
            }
        }
        model.reset();
        for token in [10u32, 20, 30] {
            model.decode(token)?;
        }
        model.reset();
        device.synchronize()?;
        let free_before_replay = gpu_free(&device)?;
        if model.position() != 0 || model.prefill(&prompt)? != first {
            failures.push("reset did not exactly replay CUDA initial raw logits".into());
        }
        device.synchronize()?;
        let free_after_replay = gpu_free(&device)?;
        model.reset();
        let mut checked_tokens = 0usize;
        let cancelled = model.prefill_with_cancel(&prompt, &mut || {
            checked_tokens += 1;
            checked_tokens == 8
        });
        if cancelled
            .as_ref()
            .err()
            .is_none_or(|error| !error.to_string().contains("cancelled at token boundary"))
            || model.position() != 0
        {
            failures.push("token-boundary cancellation did not clear partial state".into());
        }
        if model.prefill(&prompt)? != first {
            failures.push("post-cancellation CUDA replay differed".into());
        }
        device.synchronize()?;
        let free_after_cancel_replay = gpu_free(&device)?;
        if free_after_replay.saturating_sub(free_after_cancel_replay) > 256 * 1024 * 1024 {
            failures.push("CUDA free memory fell by over 256 MiB across short sessions".into());
        }
        eprintln!(
            "QWEN35_CUDA_TRACE_JSON={}",
            serde_json::json!({
                "max_logprob_error": max_logprob_error,
                "reset_exact": !failures.iter().any(|failure| failure.contains("reset")),
                "gpu_free_after_load": free_after_load,
                "gpu_free_before_replay": free_before_replay,
                "gpu_free_after_replay": free_after_replay,
                "gpu_free_after_cancel_replay": free_after_cancel_replay,
                "cancelled_after_checks": checked_tokens,
                "failures": failures,
            })
        );
        if !failures.is_empty() {
            bail!("qwen35 CUDA reference gate: {}", failures.join("; "))
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires hash-pinned external GGUF and bounded CUDA job"]
    fn cuda_first_token_components() -> Result<()> {
        use std::collections::BTreeMap;

        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            AdmissionLimits {
                max_file_bytes: 5_000_000_000,
                max_header_bytes: 16 * 1024 * 1024,
                max_tensors: 500,
                max_metadata: 64,
                max_string_bytes: 1_000_000,
                max_array_elements: 250_000,
                max_layers: 32,
                max_tensor_bytes: 700_000_000,
                max_total_tensor_bytes: 4_500_000_000,
            },
        )?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let token = fixture["prompt_token_ids"][0]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| candle::Error::Msg("first fixture token missing".into()))?;
        let mut cpu = CpuModel::load(&mut artifact, 64, 128 * 1024 * 1024)?;
        let device = Device::new_cuda(0)?;
        let mut gpu = HybridGpuModel::load(&mut artifact, 64, 128 * 1024 * 1024, &device)?;
        let mut cpu_values = Vec::new();
        cpu.decode_observed(
            token,
            Some(&mut |layer, name, values: &[f32]| {
                if layer == 0 || name == "l_out" {
                    cpu_values.push((layer, name.to_owned(), values.to_vec()));
                }
            }),
        )?;
        let mut gpu_values = BTreeMap::new();
        gpu.inner.decode_observed(
            token,
            Some(&mut |layer, name, values: &[f32]| {
                if layer == 0 || name == "l_out" {
                    gpu_values.insert((layer, name.to_owned()), values.to_vec());
                }
            }),
        )?;
        let normalized = cpu_values
            .iter()
            .find(|(layer, name, _)| *layer == 0 && name == "attn_norm")
            .map(|(_, _, values)| values.as_slice())
            .ok_or_else(|| candle::Error::Msg("CPU layer zero normalization missing".into()))?;
        let mut activation_quant_differences = 0usize;
        for block in normalized.chunks_exact(32) {
            let amax = block.iter().map(|value| value.abs()).fold(0f32, f32::max);
            if amax == 0.0 {
                continue;
            }
            let d = amax / 127.0;
            for &value in block {
                let cpu_q = (value * (127.0 / amax)).round_ties_even() as i8;
                let cuda_q = (value / d).round() as i8;
                if cpu_q != cuda_q {
                    activation_quant_differences += 1;
                }
            }
        }
        eprintln!("QWEN35_CUDA_QKV_ACTIVATION_QUANT_DIFFS={activation_quant_differences}");
        let (cpu_qkv, gpu_qkv) = match (&cpu.layers[0].mixer, &gpu.inner.layers[0].mixer) {
            (Mixer::Recurrent(cpu), Mixer::Recurrent(gpu)) => (&cpu.qkv, &gpu.qkv),
            _ => bail!("qwen35: layer zero recurrent mixer missing"),
        };
        let dtype = match cpu_qkv {
            QMatMul::QTensor(tensor) => format!("{:?}", tensor.dtype()),
            QMatMul::Tensor(_) => "F32 tensor".into(),
            QMatMul::TensorF16(_) => "F16 tensor".into(),
        };
        eprintln!("QWEN35_CUDA_QKV_DTYPE={dtype}");
        let cpu_qkv_output = projection(cpu_qkv, normalized)?;
        for forced in [false, true] {
            candle::quantized::cuda::set_force_dmmv(forced);
            let gpu_qkv_output = projection(gpu_qkv, normalized)?;
            let max_abs = cpu_qkv_output
                .iter()
                .zip(&gpu_qkv_output)
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "QWEN35_CUDA_QKV_JSON={}",
                serde_json::json!({
                    "force_dmmv": forced,
                    "max_abs": max_abs,
                })
            );
        }
        candle::quantized::cuda::set_force_dmmv(false);
        for (layer, name, cpu_vector) in cpu_values {
            let gpu_vector = gpu_values
                .get(&(layer, name.clone()))
                .ok_or_else(|| candle::Error::Msg(format!("CUDA missing {name}-{layer}")))?;
            if gpu_vector.len() != cpu_vector.len() {
                bail!("qwen35: CUDA {name}-{layer} length mismatch")
            }
            let mut max_abs = 0f32;
            let mut max_index = 0usize;
            for (index, (&cpu_value, &gpu_value)) in cpu_vector.iter().zip(gpu_vector).enumerate() {
                if !gpu_value.is_finite() {
                    bail!("qwen35: CUDA {name}-{layer} non-finite at {index}")
                }
                let error = (cpu_value - gpu_value).abs();
                if error > max_abs {
                    max_abs = error;
                    max_index = index;
                }
            }
            eprintln!(
                "QWEN35_CUDA_COMPONENT_JSON={}",
                serde_json::json!({
                    "layer": layer,
                    "name": name,
                    "length": cpu_vector.len(),
                    "max_abs": max_abs,
                    "max_index": max_index,
                })
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires hash-pinned external GGUF and bounded release benchmark job"]
    fn release_cpu_benchmark() -> Result<()> {
        use std::hint::black_box;
        use std::time::Instant;

        let path = std::env::var("QWEN35_GGUF_FILE")
            .map_err(|error| candle::Error::Msg(format!("set QWEN35_GGUF_FILE: {error}")))?;
        let limits = AdmissionLimits {
            max_file_bytes: 5_000_000_000,
            max_header_bytes: 16 * 1024 * 1024,
            max_tensors: 500,
            max_metadata: 64,
            max_string_bytes: 1_000_000,
            max_array_elements: 250_000,
            max_layers: 32,
            max_tensor_bytes: 700_000_000,
            max_total_tensor_bytes: 4_500_000_000,
        };
        let started = Instant::now();
        let mut artifact = Artifact::open(
            Path::new(&path),
            "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
            limits,
        )?;
        let admission_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        let mut model = CpuModel::load(&mut artifact, 64, 128 * 1024 * 1024)?;
        let materialize_ms = started.elapsed().as_secs_f64() * 1_000.0;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen35_codename_b/reference.json"
        ))
        .map_err(|error| candle::Error::Msg(format!("reference JSON: {error}")))?;
        let prompt = fixture["prompt_token_ids"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("prompt IDs missing".into()))?
            .iter()
            .take(16)
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| candle::Error::Msg("invalid prompt token".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if prompt.len() != 16 {
            bail!("qwen35 benchmark: missing prompt tokens")
        }
        let decode_tokens = (10u32..26).collect::<Vec<_>>();

        let started = Instant::now();
        black_box(model.prefill(&prompt[..4])?);
        for token in [10u32, 11] {
            black_box(model.decode(token)?);
        }
        let warmup_ms = started.elapsed().as_secs_f64() * 1_000.0;
        model.reset();

        let profile = std::env::var("QWEN35_BENCH_PROFILE").as_deref() == Ok("1");
        if profile {
            bench_profile::start();
        }
        let mut samples = Vec::with_capacity(3);
        let mut first_result: Option<Vec<f32>> = None;
        for repeat in 0..3 {
            let started = Instant::now();
            let mut logits = model.prefill(&prompt)?;
            let prefill_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            for &token in &decode_tokens {
                logits = model.decode(token)?;
            }
            let decode_ms = started.elapsed().as_secs_f64() * 1_000.0;
            black_box(&logits);
            if let Some(first) = first_result.as_ref() {
                if &logits != first {
                    bail!("qwen35 benchmark: reset did not replay final logits")
                }
            } else {
                first_result = Some(logits);
            }
            samples.push(serde_json::json!({
                "repeat": repeat,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "prefill_tokens_per_second": 16_000.0 / prefill_ms,
                "decode_tokens_per_second": 16_000.0 / decode_ms,
                "end_position": model.position(),
            }));
            model.reset();
        }
        let projections = if profile {
            bench_profile::take()
        } else {
            Vec::new()
        };
        eprintln!(
            "QWEN35_BENCH_JSON={}",
            serde_json::json!({
                "schema": "candle.qwen35.cpu-release-benchmark.v1",
                "artifact_bytes": 4_482_403_200u64,
                "artifact_sha256": "fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572",
                "admission_ms": admission_ms,
                "materialize_ms": materialize_ms,
                "warmup_ms": warmup_ms,
                "prefill_token_count": prompt.len(),
                "decode_token_count": decode_tokens.len(),
                "decode_mode": "teacher_forced_fixed_ids_10_to_25_no_eos",
                "profile_enabled": profile,
                "samples": samples,
                "projections": projections,
                "reset_replayed_final_logits": true,
            })
        );
        Ok(())
    }
}
