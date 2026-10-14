//! Opt-in CUDA execution for the packed GPT-OSS CPU proof model.
//!
//! Activations and dense transformer arithmetic remain F32. Losslessly
//! representable embedding/head tables use BF16 storage. MXFP4
//! expert weights remain U8 blocks plus U8 E8M0 scales on the device and are
//! consumed by the dedicated packed kernel; no dense expert fallback exists.

use super::cache::PagedKvCache;
use super::memory::{cuda_cache_capacity_bytes_for_tokens, cuda_peak_device_bytes_for_forward};
use super::model::{rope_parameters, DenseLinear, GptOssLayerWeights, GptOssWeights};
use super::mxfp4::{Mxfp4ExpertOperation, PackedMxfp4, BYTES_PER_BLOCK, VALUES_PER_BLOCK};
use super::runtime::{
    GptOssCancellationToken, GptOssLoadedHandle, GptOssResourceLimits, GptOssResourceUsage,
};
use super::GptOssConfig;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{DType, Device, Shape, Storage, Tensor, D};
use candle_nn::{ops, rotary_emb};
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

const F32_BYTES: usize = std::mem::size_of::<f32>();
const SWIGLU_ALPHA: f64 = 1.702;

/// Typed failures for the opt-in packed CUDA executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GptOssCudaError {
    UnsupportedDtype {
        requested: String,
    },
    DeviceUnavailable {
        device_index: usize,
        message: String,
    },
    ResourceLimit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
    Overflow {
        operation: &'static str,
    },
    Cancelled,
    Invalid {
        message: String,
    },
    Backend {
        message: String,
    },
    Kernel {
        status: i32,
    },
}

impl Display for GptOssCudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDtype { requested } => {
                write!(
                    f,
                    "GPT-OSS CUDA does not support activation dtype {requested}"
                )
            }
            Self::DeviceUnavailable {
                device_index,
                message,
            } => write!(
                f,
                "GPT-OSS CUDA device {device_index} is unavailable: {message}"
            ),
            Self::ResourceLimit {
                resource,
                requested,
                limit,
            } => write!(
                f,
                "GPT-OSS CUDA {resource} request {requested} exceeds limit {limit}"
            ),
            Self::Overflow { operation } => write!(f, "GPT-OSS CUDA {operation} overflowed"),
            Self::Cancelled => write!(f, "GPT-OSS CUDA operation cancelled"),
            Self::Invalid { message } => write!(f, "invalid GPT-OSS CUDA input: {message}"),
            Self::Backend { message } => write!(f, "GPT-OSS CUDA backend error: {message}"),
            Self::Kernel { status } => {
                write!(f, "GPT-OSS MXFP4 CUDA kernel failed with status {status}")
            }
        }
    }
}

impl std::error::Error for GptOssCudaError {}

impl From<candle::Error> for GptOssCudaError {
    fn from(error: candle::Error) -> Self {
        Self::Backend {
            message: error.to_string(),
        }
    }
}

pub type GptOssCudaResult<T> = std::result::Result<T, GptOssCudaError>;

/// Explicit device, activation, and static-weight budget selection.
#[derive(Debug, Clone, Copy)]
pub struct GptOssCudaConfig {
    pub device_index: usize,
    pub activation_dtype: DType,
    pub max_weight_bytes: usize,
    pub limits: GptOssResourceLimits,
}

impl GptOssCudaConfig {
    pub fn new(
        device_index: usize,
        activation_dtype: DType,
        max_weight_bytes: usize,
        limits: GptOssResourceLimits,
    ) -> GptOssCudaResult<Self> {
        if activation_dtype != DType::F32 {
            return Err(GptOssCudaError::UnsupportedDtype {
                requested: format!("{activation_dtype:?}"),
            });
        }
        if max_weight_bytes == 0 {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "static weight bytes",
                requested: 1,
                limit: 0,
            });
        }
        Ok(Self {
            device_index,
            activation_dtype,
            max_weight_bytes,
            limits,
        })
    }

    fn validate(&self, config: &GptOssConfig) -> GptOssCudaResult<()> {
        if self.activation_dtype != DType::F32 {
            return Err(GptOssCudaError::UnsupportedDtype {
                requested: format!("{:?}", self.activation_dtype),
            });
        }
        if self.max_weight_bytes == 0 {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "static weight bytes",
                requested: 1,
                limit: 0,
            });
        }
        self.limits
            .validate(config)
            .map_err(|error| GptOssCudaError::Invalid {
                message: error.to_string(),
            })
    }
}

#[derive(Debug, Clone)]
struct CudaLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl CudaLinear {
    fn from_cpu(linear: &DenseLinear, device: &Device) -> GptOssCudaResult<Self> {
        let weight = Tensor::from_slice(linear.weights(), (linear.rows(), linear.cols()), device)?;
        let bias = linear
            .bias()
            .map(|bias| Tensor::from_slice(bias, (bias.len(),), device))
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn from_cpu_table(linear: &DenseLinear, device: &Device) -> GptOssCudaResult<Self> {
        let weight = super::cuda_storage::upload_table(
            linear.weights(),
            (linear.rows(), linear.cols()),
            device,
        )?;
        let bias = linear
            .bias()
            .map(|bias| Tensor::from_slice(bias, (bias.len(),), device))
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor) -> GptOssCudaResult<Tensor> {
        let mut output = if self.weight.dtype() == DType::BF16 {
            super::cuda_storage::matmul(input, &self.weight)?
        } else {
            input.matmul(&self.weight.t()?)?
        };
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct CudaPackedMxfp4 {
    shape: Vec<usize>,
    blocks: Tensor,
    scales: Tensor,
    resident_bytes: usize,
}

impl CudaPackedMxfp4 {
    fn from_cpu(packed: &PackedMxfp4, device: &Device) -> GptOssCudaResult<Self> {
        if packed.shape().len() != 3 {
            return Err(GptOssCudaError::Invalid {
                message: format!("CUDA expert shape must be rank 3, got {:?}", packed.shape()),
            });
        }
        let shape = packed.shape().to_vec();
        let blocks = Tensor::from_slice(packed.blocks(), (packed.blocks().len(),), device)?;
        let scales = Tensor::from_slice(packed.scales(), (packed.scales().len(),), device)?;
        Ok(Self {
            shape,
            blocks,
            scales,
            resident_bytes: packed.resident_bytes(),
        })
    }
}

#[derive(Debug, Clone)]
struct CudaExperts {
    mlp1: CudaPackedMxfp4,
    mlp2: CudaPackedMxfp4,
    mlp1_bias: Tensor,
    mlp2_bias: Tensor,
    swiglu_limit: f64,
}

impl CudaExperts {
    fn from_cpu(experts: &Mxfp4ExpertOperation, device: &Device) -> GptOssCudaResult<Self> {
        let mlp1 = CudaPackedMxfp4::from_cpu(experts.mlp1(), device)?;
        let mlp2 = CudaPackedMxfp4::from_cpu(experts.mlp2(), device)?;
        let mlp1_bias = Tensor::from_slice(
            experts.mlp1_bias(),
            (experts.expert_count(), experts.mlp1().shape()[1]),
            device,
        )?;
        let mlp2_bias = Tensor::from_slice(
            experts.mlp2_bias(),
            (experts.expert_count(), experts.mlp2().shape()[1]),
            device,
        )?;
        Ok(Self {
            mlp1,
            mlp2,
            mlp1_bias,
            mlp2_bias,
            swiglu_limit: f64::from(experts.swiglu_limit()),
        })
    }

    fn packed_resident_bytes(&self) -> usize {
        self.mlp1.resident_bytes + self.mlp2.resident_bytes
    }
}

#[derive(Debug, Clone)]
struct CudaLayer {
    attention_norm: Tensor,
    qkv: CudaLinear,
    attention_out: CudaLinear,
    sinks: Tensor,
    moe_norm: Tensor,
    gate: CudaLinear,
    experts: CudaExperts,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CudaLayerTrace {
    post_attention_hidden: Tensor,
    expert_contribution: Tensor,
    post_moe_hidden: Tensor,
    attention: Tensor,
    router: Tensor,
    expert_indices: Tensor,
    expert_weights: Tensor,
}

impl CudaLayer {
    fn from_cpu(layer: &GptOssLayerWeights, device: &Device) -> GptOssCudaResult<Self> {
        Ok(Self {
            attention_norm: Tensor::from_slice(
                layer.attention_norm(),
                (layer.attention_norm().len(),),
                device,
            )?,
            qkv: CudaLinear::from_cpu(layer.qkv(), device)?,
            attention_out: CudaLinear::from_cpu(layer.attention_out(), device)?,
            sinks: Tensor::from_slice(layer.sinks(), (layer.sinks().len(),), device)?,
            moe_norm: Tensor::from_slice(layer.moe_norm(), (layer.moe_norm().len(),), device)?,
            gate: CudaLinear::from_cpu(layer.gate(), device)?,
            experts: CudaExperts::from_cpu(layer.experts(), device)?,
        })
    }
}

#[derive(Debug, Clone)]
struct CudaWeights {
    token_embedding: Tensor,
    layers: Vec<CudaLayer>,
    final_norm: Tensor,
    lm_head: CudaLinear,
    packed_resident_bytes: usize,
    static_resident_bytes: usize,
}

impl CudaWeights {
    fn from_cpu(
        config: &GptOssConfig,
        weights: GptOssWeights,
        device: &Device,
        static_resident_bytes: usize,
    ) -> GptOssCudaResult<Self> {
        let GptOssWeights {
            token_embedding: source_embedding,
            layers: source_layers,
            final_norm: source_norm,
            lm_head: source_head,
            ..
        } = weights;
        let token_embedding = if super::cuda_storage::lossless_bf16(&source_embedding) {
            let tensor = super::cuda_storage::upload_table(
                &source_embedding,
                (config.vocab_size, config.hidden_size),
                device,
            )?;
            device.synchronize()?;
            drop(source_embedding);
            tensor
        } else {
            Tensor::from_vec(
                source_embedding,
                (config.vocab_size, config.hidden_size),
                device,
            )?
        };
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source_layers.len())
            .map_err(|error| GptOssCudaError::Backend {
                message: format!("CUDA layer allocation failed: {error}"),
            })?;
        let mut packed_resident_bytes = 0usize;
        for source_layer in source_layers {
            let upload = CudaLayer::from_cpu(&source_layer, device);
            // Release each CPU layer after its uploads have completed.
            let synchronized = device.synchronize();
            let layer = upload?;
            synchronized?;
            drop(source_layer);
            packed_resident_bytes = packed_resident_bytes
                .checked_add(layer.experts.packed_resident_bytes())
                .ok_or(GptOssCudaError::Overflow {
                    operation: "packed expert byte count",
                })?;
            layers.push(layer);
        }
        let norm_len = source_norm.len();
        let final_norm = Tensor::from_vec(source_norm, (norm_len,), device)?;
        let upload = CudaLinear::from_cpu_table(&source_head, device);
        let synchronized = device.synchronize();
        let lm_head = upload?;
        synchronized?;
        drop(source_head);
        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            lm_head,
            packed_resident_bytes,
            static_resident_bytes,
        })
    }
}

type CudaCacheLayer = PagedKvCache;

#[derive(Debug, Clone)]
struct CudaCache {
    layers: Vec<CudaCacheLayer>,
    tokens: usize,
}

impl CudaCache {
    fn new(layer_count: usize) -> GptOssCudaResult<Self> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(layer_count)
            .map_err(|error| GptOssCudaError::Backend {
                message: format!("CUDA cache metadata allocation failed: {error}"),
            })?;
        layers.resize_with(layer_count, CudaCacheLayer::default);
        Ok(Self { layers, tokens: 0 })
    }
}

/// Packed MXFP4 CUDA executor with explicit resource and cancellation seams.
#[derive(Debug, Clone)]
pub struct GptOssCudaModel {
    config: GptOssConfig,
    weights: CudaWeights,
    cache: CudaCache,
    device: Device,
    limits: GptOssResourceLimits,
    activation_dtype: DType,
    #[allow(dead_code)]
    load_handle: Option<GptOssLoadedHandle>,
}

impl GptOssCudaModel {
    pub fn new(
        config: GptOssConfig,
        weights: GptOssWeights,
        cuda_config: GptOssCudaConfig,
    ) -> GptOssCudaResult<Self> {
        config.validate().map_err(GptOssCudaError::from)?;
        cuda_config.validate(&config)?;
        if weights.layers().len() != config.num_hidden_layers {
            return Err(GptOssCudaError::Invalid {
                message: "weight layer count does not match configuration".to_string(),
            });
        }
        let static_resident_bytes = cuda_weight_resident_bytes(&weights)?;
        if static_resident_bytes > cuda_config.max_weight_bytes {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "static weight bytes",
                requested: static_resident_bytes,
                limit: cuda_config.max_weight_bytes,
            });
        }
        let initial_bytes =
            cuda_peak_device_bytes_for_forward(&config, static_resident_bytes, 0, 1)?;
        if initial_bytes > cuda_config.limits.max_total_device_bytes {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "total device bytes",
                requested: initial_bytes,
                limit: cuda_config.limits.max_total_device_bytes,
            });
        }
        let device = Device::new_cuda(cuda_config.device_index).map_err(|error| {
            GptOssCudaError::DeviceUnavailable {
                device_index: cuda_config.device_index,
                message: error.to_string(),
            }
        })?;
        let load_handle = weights.load_handle();
        let weights = CudaWeights::from_cpu(&config, weights, &device, static_resident_bytes)?;
        let cache = CudaCache::new(config.num_hidden_layers)?;
        Ok(Self {
            config,
            weights,
            cache,
            device,
            limits: cuda_config.limits,
            activation_dtype: cuda_config.activation_dtype,
            load_handle,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn activation_dtype(&self) -> DType {
        self.activation_dtype
    }

    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    pub fn cache_len(&self) -> usize {
        self.cache.tokens
    }

    pub fn resource_limits(&self) -> GptOssResourceLimits {
        self.limits
    }

    pub fn resource_usage(&self) -> GptOssCudaResult<GptOssResourceUsage> {
        let row_bytes = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim)
            .and_then(|n| n.checked_mul(8))
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA cache row bytes",
            })?;
        let (cache_bytes, capacity_bytes) =
            self.cache
                .layers
                .iter()
                .try_fold((0usize, 0usize), |(logical, capacity), layer| {
                    let logical = layer
                        .rows()
                        .checked_mul(row_bytes)
                        .and_then(|n| logical.checked_add(n))
                        .ok_or(GptOssCudaError::Overflow {
                            operation: "CUDA logical cache bytes",
                        })?;
                    let capacity = capacity.checked_add(layer.storage_bytes()?).ok_or(
                        GptOssCudaError::Overflow {
                            operation: "CUDA cache capacity bytes",
                        },
                    )?;
                    Ok::<_, GptOssCudaError>((logical, capacity))
                })?;
        Ok(GptOssResourceUsage {
            sequence_tokens: self.cache.tokens,
            cache_bytes,
            cache_capacity_bytes: capacity_bytes,
        })
    }

    pub fn packed_resident_bytes(&self) -> usize {
        self.weights.packed_resident_bytes
    }

    pub fn static_resident_bytes(&self) -> usize {
        self.weights.static_resident_bytes
    }

    pub fn reset_cache(&mut self) -> GptOssCudaResult<()> {
        self.cache = CudaCache::new(self.weights.layers.len())?;
        Ok(())
    }

    pub fn evict_cache(&mut self) -> GptOssCudaResult<()> {
        self.reset_cache()
    }

    pub fn synchronize(&self) -> GptOssCudaResult<()> {
        self.device.synchronize()?;
        Ok(())
    }

    pub fn prefill(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> GptOssCudaResult<Tensor> {
        self.forward_transaction(input_ids, cancellation, None)
    }

    pub fn decode(
        &mut self,
        input_id: u32,
        cancellation: &GptOssCancellationToken,
    ) -> GptOssCudaResult<Tensor> {
        self.forward_transaction(&[input_id], cancellation, None)
    }

    pub fn forward_with_cancellation(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> GptOssCudaResult<Tensor> {
        self.forward_transaction(input_ids, cancellation, None)
    }

    pub fn forward_uncached(
        &self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> GptOssCudaResult<Tensor> {
        validate_input_ids(&self.config, input_ids)?;
        if cancellation.is_cancelled() {
            return Err(GptOssCudaError::Cancelled);
        }
        let fresh_peak = cuda_peak_device_bytes_for_forward(
            &self.config,
            self.static_resident_bytes(),
            0,
            input_ids.len(),
        )?;
        // The original root stays alive while the temporary model evaluates.
        let peak = fresh_peak
            .checked_add(self.resource_usage()?.cache_capacity_bytes)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA uncached owner peak",
            })?;
        if peak > self.limits.max_total_device_bytes {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "total device bytes",
                requested: peak,
                limit: self.limits.max_total_device_bytes,
            });
        }
        let mut model = self.clone();
        model.reset_cache()?;
        model.forward_transaction(input_ids, cancellation, None)
    }

    pub(crate) fn forward_with_validation_timing(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
    ) -> GptOssCudaResult<(Tensor, Duration)> {
        let mut validation = Duration::ZERO;
        let logits = self.forward_transaction(input_ids, cancellation, Some(&mut validation))?;
        Ok((logits, validation))
    }

    fn forward_transaction(
        &mut self,
        input_ids: &[u32],
        cancellation: &GptOssCancellationToken,
        validation_timing: Option<&mut Duration>,
    ) -> GptOssCudaResult<Tensor> {
        validate_input_ids(&self.config, input_ids)?;
        cancellation_checkpoint(cancellation)?;
        self.admit(input_ids.len())?;
        let mut working = self.cache.clone();
        let mut logits = None;
        for (token_index, &input_id) in input_ids.iter().enumerate() {
            cancellation_checkpoint(cancellation)?;
            let token = Tensor::from_slice(&[input_id], (1,), &self.device)?;
            let mut hidden = self
                .weights
                .token_embedding
                .index_select(&token, 0)?
                .to_dtype(DType::F32)?;
            for (layer_index, layer) in self.weights.layers.iter().enumerate() {
                hidden = self.forward_layer(
                    layer,
                    hidden,
                    working.layers.get_mut(layer_index).ok_or_else(|| {
                        GptOssCudaError::Invalid {
                            message: "CUDA cache layer is out of bounds".to_string(),
                        }
                    })?,
                    working.tokens,
                    layer_index,
                )?;
                cancellation_checkpoint(cancellation)?;
            }
            // Intermediate output projections cannot affect the retained KV state.
            if token_index + 1 == input_ids.len() {
                let hidden =
                    ops::rms_norm(&hidden, &self.weights.final_norm, self.config.rms_norm_eps)?;
                logits = Some(self.weights.lm_head.forward(&hidden)?);
            }
            working.tokens = working
                .tokens
                .checked_add(1)
                .ok_or(GptOssCudaError::Overflow {
                    operation: "CUDA cache token count",
                })?;
            cancellation_checkpoint(cancellation)?;
        }
        let logits = logits.ok_or_else(|| GptOssCudaError::Invalid {
            message: "CUDA forward produced no logits".to_string(),
        })?;
        let validation_start = validation_timing.as_ref().map(|_| Instant::now());
        self.device.synchronize()?;
        let output = logits.flatten_all()?.to_vec1::<f32>()?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA forward produced non-finite logits".to_string(),
            });
        }
        if let (Some(timing), Some(start)) = (validation_timing, validation_start) {
            *timing = start.elapsed();
        }
        cancellation_checkpoint(cancellation)?;
        self.cache = working;
        Ok(logits)
    }

    pub(crate) fn admit(&self, incoming_tokens: usize) -> GptOssCudaResult<()> {
        let requested_tokens =
            self.cache
                .tokens
                .checked_add(incoming_tokens)
                .ok_or(GptOssCudaError::Overflow {
                    operation: "CUDA sequence length",
                })?;
        if requested_tokens > self.limits.max_sequence_tokens {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "sequence tokens",
                requested: requested_tokens,
                limit: self.limits.max_sequence_tokens,
            });
        }
        let requested_bytes = cuda_cache_capacity_bytes_for_tokens(&self.config, requested_tokens)?;
        if requested_bytes > self.limits.max_cache_bytes {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "KV-cache bytes",
                requested: requested_bytes,
                limit: self.limits.max_cache_bytes,
            });
        }
        let total_bytes = cuda_peak_device_bytes_for_forward(
            &self.config,
            self.weights.static_resident_bytes,
            self.cache.tokens,
            incoming_tokens,
        )?;
        if total_bytes > self.limits.max_total_device_bytes {
            return Err(GptOssCudaError::ResourceLimit {
                resource: "total device bytes",
                requested: total_bytes,
                limit: self.limits.max_total_device_bytes,
            });
        }
        Ok(())
    }

    fn forward_layer(
        &self,
        layer: &CudaLayer,
        hidden: Tensor,
        cache: &mut CudaCacheLayer,
        position: usize,
        layer_index: usize,
    ) -> GptOssCudaResult<Tensor> {
        Ok(self
            .forward_layer_trace(layer, hidden, cache, position, layer_index)?
            .post_moe_hidden)
    }

    fn forward_layer_trace(
        &self,
        layer: &CudaLayer,
        hidden: Tensor,
        cache: &mut CudaCacheLayer,
        position: usize,
        layer_index: usize,
    ) -> GptOssCudaResult<CudaLayerTrace> {
        let normalized = ops::rms_norm(&hidden, &layer.attention_norm, self.config.rms_norm_eps)?;
        let qkv = layer.qkv.forward(&normalized)?;
        let q_width = self
            .config
            .num_attention_heads
            .checked_mul(self.config.head_dim)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA query width",
            })?;
        let kv_width = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA KV width",
            })?;
        let qkv_width = q_width
            .checked_add(kv_width.checked_mul(2).ok_or(GptOssCudaError::Overflow {
                operation: "CUDA QKV width",
            })?)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA QKV width",
            })?;
        if qkv.dims2()?.1 != qkv_width {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA QKV output width does not match configuration".to_string(),
            });
        }
        let (cos, sin) = rope_tensors(&self.config, position, &self.device)?;
        let query = qkv.narrow(1, 0, q_width)?.reshape((
            1,
            self.config.num_attention_heads,
            1,
            self.config.head_dim,
        ))?;
        let query = rotary_emb::rope(&query, &cos, &sin)?;
        let key = qkv.narrow(1, q_width, kv_width)?.reshape((
            1,
            self.config.num_key_value_heads,
            1,
            self.config.head_dim,
        ))?;
        let key = rotary_emb::rope(&key, &cos, &sin)?;
        let value = qkv.narrow(1, q_width + kv_width, kv_width)?.reshape((
            1,
            self.config.num_key_value_heads,
            1,
            self.config.head_dim,
        ))?;
        let key = key.reshape((1, kv_width))?;
        let value = value.reshape((1, kv_width))?;
        let sliding = self.config.attention_type(layer_index)?
            == super::GptOssAttentionType::SlidingAttention;
        cache.append(&key, &value, sliding.then_some(self.config.sliding_window))?;
        let (next_keys, next_values) = cache.materialize()?;
        let attended = cuda_attention(
            &query,
            &next_keys,
            &next_values,
            &layer.sinks,
            &self.config,
            sliding,
        )?;
        let attention_update = layer.attention_out.forward(&attended)?;
        let post_attention_hidden = hidden.broadcast_add(&attention_update)?;

        let normalized = ops::rms_norm(
            &post_attention_hidden,
            &layer.moe_norm,
            self.config.rms_norm_eps,
        )?;
        let gate = layer.gate.forward(&normalized)?;
        let top_k = self.config.experts_per_token;
        let (sorted_values, sorted_indices) = gate.sort_last_dim(false)?;
        let top_values = sorted_values.narrow(D::Minus1, 0, top_k)?;
        let top_indices = sorted_indices.narrow(D::Minus1, 0, top_k)?;
        let max = top_values.max_keepdim(D::Minus1)?;
        let top_weights = top_values.broadcast_sub(&max)?.exp()?.broadcast_div(
            &top_values
                .broadcast_sub(&max)?
                .exp()?
                .sum_keepdim(D::Minus1)?,
        )?;
        let route_inputs = normalized
            .unsqueeze(1)?
            .broadcast_as(Shape::from_dims(&[1, top_k, self.config.hidden_size]))?
            .reshape((top_k, self.config.hidden_size))?
            .contiguous()?;
        let route_indices = top_indices.reshape((top_k,))?.contiguous()?;
        let route_weights = top_weights.reshape((top_k, 1))?.contiguous()?;
        let first = cuda_mxfp4_matmul(&route_inputs, &layer.experts.mlp1, &route_indices)?;
        let first_bias = layer.experts.mlp1_bias.index_select(&route_indices, 0)?;
        let first = first.broadcast_add(&first_bias)?;
        let intermediate = self.config.intermediate_size;
        let first = first.reshape((top_k, intermediate, 2))?;
        let glu = first
            .narrow(D::Minus1, 0, 1)?
            .squeeze(D::Minus1)?
            .minimum(layer.experts.swiglu_limit)?;
        let linear = first
            .narrow(D::Minus1, 1, 1)?
            .squeeze(D::Minus1)?
            .clamp(-layer.experts.swiglu_limit, layer.experts.swiglu_limit)?;
        let activated = (&glu * ops::sigmoid(&(&glu * SWIGLU_ALPHA)?)?)?
            .broadcast_mul(&(&linear + 1.0)?)?
            .contiguous()?;
        let second = cuda_mxfp4_matmul(&activated, &layer.experts.mlp2, &route_indices)?;
        let second_bias = layer.experts.mlp2_bias.index_select(&route_indices, 0)?;
        let second = second.broadcast_add(&second_bias)?;
        let expert_contribution = second.broadcast_mul(&route_weights)?.sum(0)?.unsqueeze(0)?;
        let post_moe_hidden = post_attention_hidden.broadcast_add(&expert_contribution)?;
        Ok(CudaLayerTrace {
            post_attention_hidden,
            expert_contribution,
            post_moe_hidden,
            attention: attended,
            router: gate,
            expert_indices: top_indices,
            expert_weights: top_weights,
        })
    }
}

fn validate_input_ids(config: &GptOssConfig, input_ids: &[u32]) -> GptOssCudaResult<()> {
    if input_ids.is_empty() {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA forward requires at least one input token".to_string(),
        });
    }
    for (index, &token) in input_ids.iter().enumerate() {
        if token as usize >= config.vocab_size {
            return Err(GptOssCudaError::Invalid {
                message: format!(
                    "CUDA input token {token} at position {index} exceeds vocab size {}",
                    config.vocab_size
                ),
            });
        }
    }
    Ok(())
}

fn cancellation_checkpoint(token: &GptOssCancellationToken) -> GptOssCudaResult<()> {
    token
        .checkpoint()
        .map_err(|error| match error.to_string().contains("cancelled") {
            true => GptOssCudaError::Cancelled,
            false => GptOssCudaError::Backend {
                message: error.to_string(),
            },
        })
}

fn rope_tensors(
    config: &GptOssConfig,
    position: usize,
    device: &Device,
) -> GptOssCudaResult<(Tensor, Tensor)> {
    let half = config.head_dim / 2;
    if half == 0 {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA rotary head dimension must be at least two".to_string(),
        });
    }
    let (concentration, inv_freq) = rope_parameters(config, half)?;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for frequency in inv_freq {
        let angle = position as f32 * frequency;
        cos.push(angle.cos() * concentration);
        sin.push(angle.sin() * concentration);
    }
    Ok((
        Tensor::from_slice(&cos, (1, half), device)?,
        Tensor::from_slice(&sin, (1, half), device)?,
    ))
}

fn cuda_attention(
    query: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    sinks: &Tensor,
    config: &GptOssConfig,
    sliding: bool,
) -> GptOssCudaResult<Tensor> {
    let (key_tokens, kv_width) = keys.dims2()?;
    if key_tokens == 0 {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA attention requires at least one cached token".to_string(),
        });
    }
    let expected_kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or(GptOssCudaError::Overflow {
            operation: "CUDA attention KV width",
        })?;
    if kv_width != expected_kv_width || values.dims2()? != (key_tokens, kv_width) {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA attention cache shape does not match configuration".to_string(),
        });
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
    {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA attention heads are not divisible by KV heads".to_string(),
        });
    }
    let window_start = if sliding {
        key_tokens.saturating_sub(config.sliding_window)
    } else {
        0
    };
    let (keys, values, key_tokens) = if window_start > 0 {
        let len = key_tokens - window_start;
        (
            keys.narrow(0, window_start, len)?,
            values.narrow(0, window_start, len)?,
            len,
        )
    } else {
        (keys.clone(), values.clone(), key_tokens)
    };
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let q_per_kv = heads / kv_heads;
    // Batched GQA never expands K/V to one copy per query head.
    let key = keys
        .reshape((key_tokens, kv_heads, head_dim))?
        .permute((1, 2, 0))?
        .contiguous()?;
    let value = values
        .reshape((key_tokens, kv_heads, head_dim))?
        .permute((1, 0, 2))?
        .contiguous()?;
    let query = query.reshape((kv_heads, q_per_kv, head_dim))?;
    let scores = (query.matmul(&key)? * (head_dim as f64).sqrt().recip())?;
    let sink = sinks.reshape((kv_heads, q_per_kv, 1))?;
    let scores = Tensor::cat(&[&scores, &sink], D::Minus1)?;
    let weights = ops::softmax(&scores, D::Minus1)?.narrow(D::Minus1, 0, key_tokens)?;
    let attended = weights.contiguous()?.matmul(&value)?;
    let width = heads
        .checked_mul(head_dim)
        .ok_or(GptOssCudaError::Overflow {
            operation: "CUDA attention output width",
        })?;
    Ok(attended.reshape((1, width))?)
}

fn cuda_mxfp4_matmul(
    input: &Tensor,
    weights: &CudaPackedMxfp4,
    experts: &Tensor,
) -> GptOssCudaResult<Tensor> {
    cuda_mxfp4_matmul_with_views(
        input,
        &weights.blocks,
        &weights.scales,
        &weights.shape,
        experts,
    )
}

fn cuda_mxfp4_matmul_with_views(
    input: &Tensor,
    blocks: &Tensor,
    scales: &Tensor,
    shape: &[usize],
    experts: &Tensor,
) -> GptOssCudaResult<Tensor> {
    if !input.device().is_cuda()
        || !blocks.device().is_cuda()
        || !scales.device().is_cuda()
        || !experts.device().is_cuda()
    {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA MXFP4 inputs, packed storage, and route indices must be CUDA tensors"
                .to_string(),
        });
    }
    for (name, tensor) in [("blocks", blocks), ("scales", scales), ("experts", experts)] {
        if !input.device().same_device(tensor.device()) {
            return Err(GptOssCudaError::Invalid {
                message: format!("CUDA MXFP4 {name} tensor is on a different device"),
            });
        }
    }
    if input.dtype() != DType::F32
        || blocks.dtype() != DType::U8
        || scales.dtype() != DType::U8
        || experts.dtype() != DType::U32
    {
        return Err(GptOssCudaError::UnsupportedDtype {
            requested: format!(
                "input={:?}, blocks={:?}, scales={:?}, experts={:?}",
                input.dtype(),
                blocks.dtype(),
                scales.dtype(),
                experts.dtype()
            ),
        });
    }
    let (route_count, input_width) = input.dims2()?;
    let (expert_count, output_width, weight_input_width) = match shape {
        [expert_count, output_width, input_width] => (*expert_count, *output_width, *input_width),
        shape => {
            return Err(GptOssCudaError::Invalid {
                message: format!("CUDA MXFP4 shape is not rank 3: {shape:?}"),
            })
        }
    };
    if input_width != weight_input_width || experts.dims1()? != route_count {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA MXFP4 route dimensions do not match".to_string(),
        });
    }
    if route_count == 0 || output_width == 0 || !input_width.is_multiple_of(VALUES_PER_BLOCK) {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA MXFP4 dimensions must be non-zero and block-aligned".to_string(),
        });
    }
    let blocks_per_row = input_width / VALUES_PER_BLOCK;
    let block_count = expert_count
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(blocks_per_row))
        .ok_or(GptOssCudaError::Overflow {
            operation: "CUDA MXFP4 block count",
        })?;
    let block_bytes =
        block_count
            .checked_mul(BYTES_PER_BLOCK)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA MXFP4 block bytes",
            })?;
    if blocks.dims1()? != block_bytes || scales.dims1()? != block_count {
        return Err(GptOssCudaError::Invalid {
            message: "CUDA MXFP4 packed storage does not match its logical shape".to_string(),
        });
    }
    let route_count_i32 = checked_i32(route_count, "CUDA MXFP4 route count")?;
    let output_width_i32 = checked_i32(output_width, "CUDA MXFP4 output width")?;
    let input_width_i32 = checked_i32(input_width, "CUDA MXFP4 input width")?;
    let expert_count_i32 = checked_i32(expert_count, "CUDA MXFP4 expert count")?;
    // Raw CUDA pointers must describe the logical view, not an arbitrary
    // parent allocation. Materializing a non-zero-offset or strided view
    // makes narrowed offsets equivalent to the already-contiguous path before
    // pointer extraction, while preserving the zero-offset fast path.
    let input = materialize_cuda_view(input)?;
    let blocks = materialize_cuda_view(blocks)?;
    let scales = materialize_cuda_view(scales)?;
    let experts = materialize_cuda_view(experts)?;
    let device = input.device().as_cuda_device()?.clone();
    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA MXFP4 input is not on CUDA".to_string(),
            })
        }
    };
    let (block_storage, _) = blocks.storage_and_layout();
    let block_slice = match &*block_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u8>()?,
        _ => {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA MXFP4 blocks are not on CUDA".to_string(),
            })
        }
    };
    let (scale_storage, _) = scales.storage_and_layout();
    let scale_slice = match &*scale_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u8>()?,
        _ => {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA MXFP4 scales are not on CUDA".to_string(),
            })
        }
    };
    let (expert_storage, _) = experts.storage_and_layout();
    let expert_slice = match &*expert_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u32>()?,
        _ => {
            return Err(GptOssCudaError::Invalid {
                message: "CUDA MXFP4 route indices are not on CUDA".to_string(),
            })
        }
    };
    let output_elements =
        route_count
            .checked_mul(output_width)
            .ok_or(GptOssCudaError::Overflow {
                operation: "CUDA MXFP4 output elements",
            })?;
    let mut output = unsafe { device.alloc::<f32>(output_elements) }?;
    let status = {
        let stream = device.cuda_stream().clone();
        let (input_ptr, _input_guard) = input_slice.device_ptr(&stream);
        let (block_ptr, _block_guard) = block_slice.device_ptr(&stream);
        let (scale_ptr, _scale_guard) = scale_slice.device_ptr(&stream);
        let (expert_ptr, _expert_guard) = expert_slice.device_ptr(&stream);
        let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
        unsafe {
            candle::cuda_backend::kernels::ffi::launch_gpt_oss_mxfp4_matmul(
                input_ptr as *const f32,
                block_ptr as *const u8,
                scale_ptr as *const u8,
                expert_ptr as *const u32,
                output_ptr as *mut f32,
                route_count_i32,
                output_width_i32,
                input_width_i32,
                expert_count_i32,
                stream.cu_stream() as i64,
            )
        }
    };
    if status != 0 {
        return Err(GptOssCudaError::Kernel { status });
    }
    let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        (route_count, output_width),
        candle::op::BackpropOp::none(),
        false,
    ))
}

fn materialize_cuda_view(tensor: &Tensor) -> candle::Result<Tensor> {
    match tensor.layout().contiguous_offsets() {
        Some((0, end)) if end == tensor.elem_count() => Ok(tensor.clone()),
        _ => tensor.force_contiguous(),
    }
}

fn checked_i32(value: usize, operation: &'static str) -> GptOssCudaResult<i32> {
    i32::try_from(value).map_err(|_| GptOssCudaError::Overflow { operation })
}

fn cuda_weight_resident_bytes(weights: &GptOssWeights) -> GptOssCudaResult<usize> {
    let mut total = 0usize;
    add_elements(
        &mut total,
        weights.token_embedding().len(),
        table_element_bytes(weights.token_embedding()),
        "token embedding",
    )?;
    for layer in weights.layers() {
        add_elements(
            &mut total,
            layer.attention_norm().len(),
            F32_BYTES,
            "attention norm",
        )?;
        add_linear_bytes(&mut total, layer.qkv(), "QKV")?;
        add_linear_bytes(&mut total, layer.attention_out(), "attention output")?;
        add_elements(
            &mut total,
            layer.sinks().len(),
            F32_BYTES,
            "attention sinks",
        )?;
        add_elements(&mut total, layer.moe_norm().len(), F32_BYTES, "MoE norm")?;
        add_linear_bytes(&mut total, layer.gate(), "router")?;
        add_elements(
            &mut total,
            layer.experts().mlp1_bias().len(),
            F32_BYTES,
            "MLP1 bias",
        )?;
        add_elements(
            &mut total,
            layer.experts().mlp2_bias().len(),
            F32_BYTES,
            "MLP2 bias",
        )?;
        add_elements(
            &mut total,
            layer.experts().mlp1().resident_bytes(),
            1,
            "MLP1 packed bytes",
        )?;
        add_elements(
            &mut total,
            layer.experts().mlp2().resident_bytes(),
            1,
            "MLP2 packed bytes",
        )?;
    }
    add_elements(
        &mut total,
        weights.final_norm().len(),
        F32_BYTES,
        "final norm",
    )?;
    add_elements(
        &mut total,
        weights.lm_head().weights().len(),
        table_element_bytes(weights.lm_head().weights()),
        "LM head",
    )?;
    if let Some(bias) = weights.lm_head().bias() {
        add_elements(&mut total, bias.len(), F32_BYTES, "LM head bias")?;
    }
    Ok(total)
}

fn table_element_bytes(values: &[f32]) -> usize {
    if super::cuda_storage::lossless_bf16(values) {
        2
    } else {
        F32_BYTES
    }
}

fn add_linear_bytes(
    total: &mut usize,
    linear: &DenseLinear,
    operation: &'static str,
) -> GptOssCudaResult<()> {
    add_elements(total, linear.weights().len(), F32_BYTES, operation)?;
    if let Some(bias) = linear.bias() {
        add_elements(total, bias.len(), F32_BYTES, operation)?;
    }
    Ok(())
}

fn add_elements(
    total: &mut usize,
    elements: usize,
    bytes_per_element: usize,
    operation: &'static str,
) -> GptOssCudaResult<()> {
    let bytes = elements
        .checked_mul(bytes_per_element)
        .ok_or(GptOssCudaError::Overflow { operation })?;
    *total = total
        .checked_add(bytes)
        .ok_or(GptOssCudaError::Overflow { operation })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{cache_bytes_for_tokens, GptOssModel};
    use super::*;
    use crate::models::gpt_oss::mxfp4::{
        Mxfp4ExpertOperation, BYTES_PER_BLOCK, SCALE_BIAS, VALUES_PER_BLOCK,
    };
    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    const ORACLE_BYTES: &[u8] =
        include_bytes!("../../../../tests/fixtures/gpt_oss_task2/oracle.json");
    const ORACLE_SHA256: &str = "db79441a688bce500217635051372270fce02b7e6f8f03d769855ac290d4ab04";
    const TWO_LAYER_ORACLE_BYTES: &[u8] =
        include_bytes!("../../../../tests/fixtures/gpt_oss_task3_two_layer/oracle.json");
    const TWO_LAYER_ORACLE_SHA256: &str =
        "0530081353cc7268d796ae070c4168ec4e6bb036f62d54670b975a43556bc173";

    #[test]
    #[ignore = "explicit authored full-vocabulary projection diagnostic, no model loading"]
    fn profile_full_vocabulary_prefill_projection() -> TestResult {
        const VOCAB: usize = 201_088;
        const WIDTH: usize = 32;
        const TOKENS: usize = 128;
        let elements = VOCAB
            .checked_mul(WIDTH)
            .expect("fixed diagnostic dimensions");
        assert!(elements * F32_BYTES <= 32 * 1024 * 1024);
        let source = DenseLinear::new(
            VOCAB,
            WIDTH,
            (0..elements)
                .map(|i| ((i % 17) as f32 - 8.) * 0.015625)
                .collect(),
            None,
        )?;
        let device = Device::new_cuda(0)?;
        let head = CudaLinear::from_cpu_table(&source, &device)?;
        assert_eq!(head.weight.dtype(), DType::BF16);
        drop(source);
        let norm = Tensor::ones(WIDTH, DType::F32, &device)?;
        let inputs = (0..TOKENS)
            .map(|token| {
                Tensor::from_vec(
                    (0..WIDTH)
                        .map(|i| (((i + token) % 13) as f32 - 6.) * 0.03125)
                        .collect::<Vec<_>>(),
                    (1, WIDTH),
                    &device,
                )
            })
            .collect::<candle::Result<Vec<_>>>()?;
        device.synchronize()?;
        let measure = |all_tokens: bool| -> GptOssCudaResult<(Vec<u32>, [f64; 5], usize)> {
            let total = Instant::now();
            let mut logits = None;
            let mut head_calls = 0;
            for (index, hidden) in inputs.iter().enumerate() {
                if all_tokens || index + 1 == inputs.len() {
                    let normalized = ops::rms_norm(hidden, &norm, 1e-5)?;
                    logits = Some(head.forward(&normalized)?);
                    head_calls += 1;
                }
            }
            let before_sync = total.elapsed();
            let stage = Instant::now();
            device.synchronize()?;
            let synchronize = stage.elapsed();
            let stage = Instant::now();
            let values = logits
                .expect("fixed nonempty diagnostic span")
                .flatten_all()?
                .to_vec1::<f32>()?;
            let readback = stage.elapsed();
            let stage = Instant::now();
            let nonfinite = values.iter().any(|value| !value.is_finite());
            let finite_scan = stage.elapsed();
            let total = total.elapsed();
            assert!(!nonfinite);
            assert_eq!(values.len(), VOCAB);
            let bits = values.iter().map(|value| value.to_bits()).collect();
            Ok((
                bits,
                [before_sync, synchronize, readback, finite_scan, total]
                    .map(|duration| duration.as_secs_f64() * 1000.),
                head_calls,
            ))
        };
        let mut expected = None;
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for iteration in 0..10 {
            for all_tokens in if iteration % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let (bits, stages, calls) = measure(all_tokens)?;
                assert_eq!(calls, if all_tokens { TOKENS } else { 1 });
                if let Some(expected) = &expected {
                    assert_eq!(&bits, expected);
                } else {
                    expected = Some(bits);
                }
                if iteration >= 2 {
                    if all_tokens {
                        &mut baseline
                    } else {
                        &mut candidate
                    }
                    .push(stages);
                }
            }
        }
        println!(
            "synthetic-projection {}",
            serde_json::json!({"vocab": VOCAB, "hidden": WIDTH, "tokens": TOKENS,
                "warmup_pairs": 2, "measured_pairs": 8,
                "stages": ["before_final_sync", "synchronize", "full_readback", "finite_scan", "total"],
                "baseline_head_calls": TOKENS, "candidate_head_calls": 1,
                "baseline_ms": baseline, "candidate_ms": candidate,
                "final_row_sha256": format!("{:x}", Sha256::digest(expected.expect("measured row")
                    .iter().flat_map(|bits| bits.to_le_bytes()).collect::<Vec<_>>()))})
        );
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct Oracle {
        fixture_id: String,
        forward: OracleForward,
        packed: OraclePacked,
        provenance: OracleProvenance,
        transitions: OracleTransitions,
    }

    #[derive(Debug, Deserialize)]
    struct OracleForward {
        prompt: Vec<u32>,
        prefill: Vec<u32>,
        decode: Vec<u32>,
        uncached_logits: Vec<f32>,
        decode_logits: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct OraclePacked {
        input: Vec<f32>,
        output: Vec<f32>,
        seed: u8,
        selected_expert: u8,
        shape: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleProvenance {
        tolerance_abs: f32,
    }

    #[derive(Debug, Deserialize)]
    struct OracleTransitions {
        token_1: OracleTransition,
        token_3_decode: OracleTransition,
    }

    #[derive(Debug, Deserialize)]
    struct OracleTransition {
        attention: Vec<f32>,
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
        router: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracle {
        fixture_id: String,
        forward: TwoLayerOracleForward,
        provenance: OracleProvenance,
        trace: TwoLayerOracleTraceBundle,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleForward {
        prompt: Vec<u32>,
        prefill: Vec<u32>,
        decode: Vec<u32>,
        uncached_logits: Vec<f32>,
        decode_logits: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleTraceBundle {
        token: u32,
        position: usize,
        layers: Vec<TwoLayerOracleLayer>,
    }

    #[derive(Debug, Deserialize)]
    struct TwoLayerOracleLayer {
        sliding: bool,
        attention: Vec<f32>,
        post_attention_hidden: Vec<f32>,
        expert_contribution: Vec<f32>,
        post_moe_hidden: Vec<f32>,
        router: Vec<f32>,
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
    }

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn oracle() -> std::result::Result<Oracle, Box<dyn std::error::Error>> {
        let actual = format!("{:x}", Sha256::digest(ORACLE_BYTES));
        assert_eq!(actual, ORACLE_SHA256, "Task 2 oracle fixture changed");
        Ok(serde_json::from_slice(ORACLE_BYTES)?)
    }

    fn two_layer_oracle() -> std::result::Result<TwoLayerOracle, Box<dyn std::error::Error>> {
        let actual = format!("{:x}", Sha256::digest(TWO_LAYER_ORACLE_BYTES));
        assert_eq!(
            actual, TWO_LAYER_ORACLE_SHA256,
            "Task 3 two-layer oracle fixture changed"
        );
        Ok(serde_json::from_slice(TWO_LAYER_ORACLE_BYTES)?)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "CUDA oracle mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn config() -> GptOssConfig {
        GptOssConfig {
            model_type: "gpt_oss".to_string(),
            num_hidden_layers: 1,
            num_experts: 2,
            experts_per_token: 2,
            vocab_size: 8,
            hidden_size: 32,
            intermediate_size: 32,
            swiglu_limit: 7.0,
            head_dim: 8,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            sliding_window: 3,
            initial_context_length: 8,
            rope_theta: 10_000.0,
            rope_scaling_factor: 1.0,
            rope_ntk_alpha: 1.0,
            rope_ntk_beta: 32.0,
            architectures: vec!["GptOssForCausalLM".to_string()],
            layer_types: Vec::new(),
            rms_norm_eps: 1e-5,
            max_position_embeddings: None,
        }
    }

    fn config_with_layers(layer_count: usize) -> GptOssConfig {
        let mut config = config();
        config.num_hidden_layers = layer_count;
        config
    }

    fn packed(shape: &[usize], seed: u8) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        packed_with_scale(shape, seed, SCALE_BIAS as u8)
    }

    fn packed_with_scale(
        shape: &[usize],
        seed: u8,
        scale: u8,
    ) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        let elements = shape.iter().product::<usize>();
        let blocks = elements / VALUES_PER_BLOCK;
        let bytes: Vec<u8> = (0..blocks * BYTES_PER_BLOCK)
            .map(|index| seed.wrapping_add(index as u8).rotate_left(1))
            .collect();
        let scales = vec![scale; blocks];
        crate::models::gpt_oss::mxfp4::PackedMxfp4::from_parts(shape, bytes, scales).unwrap()
    }

    fn packed_or_zero(
        shape: &[usize],
        seed: u8,
        zero: bool,
        scale: u8,
    ) -> crate::models::gpt_oss::mxfp4::PackedMxfp4 {
        if zero {
            let elements = shape.iter().product::<usize>();
            let blocks = elements / VALUES_PER_BLOCK;
            crate::models::gpt_oss::mxfp4::PackedMxfp4::from_parts(
                shape,
                vec![0; blocks * BYTES_PER_BLOCK],
                vec![scale; blocks],
            )
            .unwrap()
        } else {
            packed_with_scale(shape, seed, scale)
        }
    }

    fn linear(rows: usize, cols: usize, seed: f32, bias: bool) -> DenseLinear {
        let weights = (0..rows * cols)
            .map(|index| ((index % 11) as f32 - 5.0) * seed)
            .collect();
        let bias = bias.then(|| (0..rows).map(|index| index as f32 * seed).collect());
        DenseLinear::new(rows, cols, weights, bias).unwrap()
    }

    fn model_parts_with_options(
        layer_count: usize,
        zero_experts: bool,
        nonzero_sinks: bool,
    ) -> (GptOssConfig, GptOssWeights) {
        let config = config_with_layers(layer_count);
        let qkv_rows =
            config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads);
        let expert_scale = if nonzero_sinks {
            (SCALE_BIAS - 4) as u8
        } else {
            SCALE_BIAS as u8
        };
        let layers = (0..layer_count)
            .map(|layer_index| {
                let layer_offset = layer_index as u8;
                let layer_seed = layer_index as f32;
                let experts = Mxfp4ExpertOperation::new(
                    packed_or_zero(
                        &[
                            config.num_experts,
                            config.intermediate_size * 2,
                            config.hidden_size,
                        ],
                        1u8.wrapping_add(layer_offset.wrapping_mul(17)),
                        zero_experts,
                        expert_scale,
                    ),
                    vec![0.0; config.num_experts * config.intermediate_size * 2],
                    packed_or_zero(
                        &[
                            config.num_experts,
                            config.hidden_size,
                            config.intermediate_size,
                        ],
                        7u8.wrapping_add(layer_offset.wrapping_mul(19)),
                        zero_experts,
                        expert_scale,
                    ),
                    vec![0.0; config.num_experts * config.hidden_size],
                    config.swiglu_limit,
                )
                .unwrap();
                let sinks = if nonzero_sinks {
                    (0..config.num_attention_heads)
                        .map(|head| (head as f32 - 1.5) * 0.125 + layer_seed * 0.25)
                        .collect()
                } else {
                    vec![0.0; config.num_attention_heads]
                };
                GptOssLayerWeights::new(
                    &config,
                    vec![1.0; config.hidden_size],
                    linear(
                        qkv_rows,
                        config.hidden_size,
                        0.0003 + layer_seed * 0.00007,
                        true,
                    ),
                    linear(
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim,
                        0.0002 + layer_seed * 0.00005,
                        true,
                    ),
                    sinks,
                    vec![1.0; config.hidden_size],
                    linear(
                        config.num_experts,
                        config.hidden_size,
                        0.0004 + layer_seed * 0.0003,
                        true,
                    ),
                    experts,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let weights = GptOssWeights::new(
            &config,
            (0..config.vocab_size * config.hidden_size)
                .map(|index| (index % 17) as f32 * 0.001)
                .collect(),
            layers,
            vec![1.0; config.hidden_size],
            linear(config.vocab_size, config.hidden_size, 0.0001, false),
        )
        .unwrap();
        (config, weights)
    }

    fn model_parts() -> (GptOssConfig, GptOssWeights) {
        model_parts_with_options(1, false, false)
    }

    fn cuda_model() -> std::result::Result<GptOssCudaModel, Box<dyn std::error::Error>> {
        let (config, weights) = model_parts();
        cuda_model_from_parts(config, weights, 4)
    }

    fn cuda_model_from_parts(
        config: GptOssConfig,
        weights: GptOssWeights,
        max_sequence_tokens: usize,
    ) -> std::result::Result<GptOssCudaModel, Box<dyn std::error::Error>> {
        let limits = GptOssResourceLimits {
            max_sequence_tokens,
            max_cache_bytes: cache_bytes_for_tokens(&config, max_sequence_tokens)?,
            max_total_device_bytes: usize::MAX,
        };
        let max_weight_bytes = cuda_weight_resident_bytes(&weights)?;
        let cuda_config = GptOssCudaConfig::new(0, DType::F32, max_weight_bytes, limits)?;
        Ok(GptOssCudaModel::new(config, weights, cuda_config)?)
    }

    fn trace_token(
        model: &mut GptOssCudaModel,
        input_id: u32,
    ) -> std::result::Result<CudaLayerTrace, Box<dyn std::error::Error>> {
        let traces = trace_token_layers(model, input_id)?;
        traces
            .into_iter()
            .next()
            .ok_or_else(|| "missing synthetic CUDA layer".into())
    }

    fn trace_token_layers(
        model: &mut GptOssCudaModel,
        input_id: u32,
    ) -> std::result::Result<Vec<CudaLayerTrace>, Box<dyn std::error::Error>> {
        let token = Tensor::from_slice(&[input_id], (1,), &model.device)?;
        let mut hidden = model.weights.token_embedding.index_select(&token, 0)?;
        let mut working = model.cache.clone();
        let position = working.tokens;
        let mut traces = Vec::new();
        for (layer_index, layer) in model.weights.layers.iter().enumerate() {
            let cache = working
                .layers
                .get_mut(layer_index)
                .ok_or("missing CUDA cache layer")?;
            let trace = model.forward_layer_trace(layer, hidden, cache, position, layer_index)?;
            hidden = trace.post_moe_hidden.clone();
            traces.push(trace);
        }
        working.tokens += 1;
        model.cache = working;
        Ok(traces)
    }

    fn flatten_tensor(
        tensor: &Tensor,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn packed_kernel_and_cuda_forward_match_independent_oracle() -> TestResult {
        let oracle = oracle()?;
        assert_eq!(oracle.fixture_id, "synthetic-gpt-oss-task2-oracle-v1");
        let tolerance = oracle.provenance.tolerance_abs;
        let (config, weights) = model_parts();
        let mut cpu = crate::models::gpt_oss::GptOssModel::new(config.clone(), weights.clone())?;
        let cpu_uncached = cpu.forward_uncached(&oracle.forward.prompt)?;
        assert_close(&cpu_uncached, &oracle.forward.uncached_logits, tolerance);
        let mut cuda = cuda_model()?;
        let packed_weights = packed(&oracle.packed.shape, oracle.packed.seed);
        let packed_cuda = CudaPackedMxfp4::from_cpu(&packed_weights, cuda.device())?;
        let packed_input = Tensor::from_slice(
            &oracle.packed.input,
            (1, oracle.packed.input.len()),
            cuda.device(),
        )?;
        let packed_expert = Tensor::from_slice(
            &[u32::from(oracle.packed.selected_expert)],
            (1,),
            cuda.device(),
        )?;
        let packed_output = cuda_mxfp4_matmul(&packed_input, &packed_cuda, &packed_expert)?;
        let packed_output = packed_output
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&packed_output, &oracle.packed.output, tolerance);

        let uncached =
            cuda.forward_uncached(&oracle.forward.prompt, &GptOssCancellationToken::new())?;
        let uncached = uncached
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&uncached, &oracle.forward.uncached_logits, tolerance);
        assert_eq!(cuda.static_resident_bytes(), 19_416);
        assert_eq!(
            cuda.packed_resident_bytes(),
            packed(&[2, 64, 32], 1).resident_bytes() + packed(&[2, 32, 32], 7).resident_bytes()
        );
        assert_eq!(
            cuda.weights.layers[0].experts.mlp1.blocks.dtype(),
            DType::U8
        );
        assert_eq!(
            cuda.weights.layers[0].experts.mlp1.scales.dtype(),
            DType::U8
        );

        let mut traced = cuda_model()?;
        let token_1 = trace_token(&mut traced, 1)?;
        let _token_2 = trace_token(&mut traced, 2)?;
        let token_3 = trace_token(&mut traced, 3)?;
        for (actual, expected) in [
            (&token_1, &oracle.transitions.token_1),
            (&token_3, &oracle.transitions.token_3_decode),
        ] {
            assert_close(
                &flatten_tensor(&actual.attention)?,
                &expected.attention,
                tolerance,
            );
            assert_close(
                &flatten_tensor(&actual.router)?,
                &expected.router,
                tolerance,
            );
            let actual_indices = actual
                .expert_indices
                .flatten_all()?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>();
            assert_eq!(actual_indices, expected.expert_indices);
            assert_close(
                &flatten_tensor(&actual.expert_weights)?,
                &expected.expert_weights,
                tolerance,
            );
        }

        cuda.prefill(&oracle.forward.prefill, &GptOssCancellationToken::new())?;
        let decode_token = *oracle
            .forward
            .decode
            .first()
            .ok_or("missing oracle decode token")?;
        let decode = cuda.decode(decode_token, &GptOssCancellationToken::new())?;
        let decode = decode
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&decode, &oracle.forward.decode_logits, tolerance);
        assert_eq!(cuda.cache_len(), 3);
        let usage = cuda.resource_usage()?;
        assert_eq!(usage.sequence_tokens, 3);
        assert_eq!(usage.cache_bytes, 384);
        assert_eq!(usage.cache_capacity_bytes, 384);

        let cpu_cancellation = GptOssCancellationToken::new();
        cpu.prefill(&oracle.forward.prefill, &cpu_cancellation)?;
        let cpu_decode = cpu.decode(decode_token, &cpu_cancellation)?;
        assert_close(&decode, &cpu_decode, tolerance);
        cuda.synchronize()?;
        Ok(())
    }

    #[test]
    fn packed_cuda_narrowed_views_match_materialized_copy() -> TestResult {
        let (config, weights) = model_parts();
        let cuda = cuda_model_from_parts(config, weights, 1)?;
        let packed_weights = packed(&[2, 64, 32], 1);
        let packed_cuda = CudaPackedMxfp4::from_cpu(&packed_weights, cuda.device())?;
        let input_values: Vec<f32> = (0..32).map(|index| index as f32 * 0.03125).collect();
        let mut padded_input = vec![99.0f32];
        padded_input.extend_from_slice(&input_values);
        padded_input.push(-99.0);
        let input_view =
            Tensor::from_slice(&padded_input, (1, 34), cuda.device())?.narrow(1, 1, 32)?;

        let mut padded_blocks = vec![0u8];
        padded_blocks.extend_from_slice(packed_weights.blocks());
        padded_blocks.push(0);
        let block_view = Tensor::from_slice(&padded_blocks, (padded_blocks.len(),), cuda.device())?
            .narrow(0, 1, packed_weights.blocks().len())?;
        let mut padded_scales = vec![0u8];
        padded_scales.extend_from_slice(packed_weights.scales());
        padded_scales.push(0);
        let scale_view = Tensor::from_slice(&padded_scales, (padded_scales.len(),), cuda.device())?
            .narrow(0, 1, packed_weights.scales().len())?;
        let padded_expert =
            Tensor::from_slice(&[77u32, 0], (2,), cuda.device())?.narrow(0, 1, 1)?;
        let view_output = cuda_mxfp4_matmul_with_views(
            &input_view,
            &block_view,
            &scale_view,
            packed_weights.shape(),
            &padded_expert,
        )?;
        let full_input = Tensor::from_slice(&input_values, (1, 32), cuda.device())?;
        let full_expert = Tensor::from_slice(&[0u32], (1,), cuda.device())?;
        let materialized_output = cuda_mxfp4_matmul(&full_input, &packed_cuda, &full_expert)?;
        let view_output = view_output.flatten_all()?.to_vec1::<f32>()?;
        let materialized_output = materialized_output.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&view_output, &materialized_output, 1e-6);
        cuda.synchronize()?;
        Ok(())
    }

    #[test]
    fn zero_expert_output_preserves_nonzero_post_attention_hidden() -> TestResult {
        let (config, weights) = model_parts_with_options(1, true, false);
        let mut cuda = cuda_model_from_parts(config.clone(), weights, 1)?;
        let traces = trace_token_layers(&mut cuda, 1)?;
        let trace = traces.first().ok_or("missing zero-expert CUDA trace")?;
        let post_attention = flatten_tensor(&trace.post_attention_hidden)?;
        let expert_contribution = flatten_tensor(&trace.expert_contribution)?;
        let post_moe = flatten_tensor(&trace.post_moe_hidden)?;
        assert!(
            post_attention.iter().any(|value| value.abs() > 1e-6),
            "regression requires a nonzero post-attention residual"
        );
        assert_close(&expert_contribution, &vec![0.0; config.hidden_size], 1e-6);
        assert_close(&post_moe, &post_attention, 1e-6);
        cuda.synchronize()?;
        Ok(())
    }

    #[test]
    fn task3_two_layer_cpu_reference_cuda_and_hidden_traces_match_oracle() -> TestResult {
        let oracle = two_layer_oracle()?;
        assert_eq!(
            oracle.fixture_id,
            "synthetic-gpt-oss-task3-two-layer-oracle-v1"
        );
        assert_eq!(oracle.trace.token, 4);
        assert_eq!(oracle.trace.position, 3);
        let tolerance = oracle.provenance.tolerance_abs;
        let (config, weights) = model_parts_with_options(2, false, true);
        assert_eq!(config.num_hidden_layers, 2);
        assert!(weights
            .layers()
            .iter()
            .all(|layer| { layer.sinks().iter().any(|value| value.abs() > 0.0) }));

        let mut cpu = crate::models::gpt_oss::GptOssModel::new(config.clone(), weights.clone())?;
        let cpu_uncached = cpu.forward_uncached(&oracle.forward.prompt)?;
        assert_close(&cpu_uncached, &oracle.forward.uncached_logits, tolerance);

        let mut cuda = cuda_model_from_parts(config.clone(), weights.clone(), 4)?;
        let cuda_uncached = cuda
            .forward_uncached(&oracle.forward.prompt, &GptOssCancellationToken::new())?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&cuda_uncached, &oracle.forward.uncached_logits, tolerance);
        assert_close(&cuda_uncached, &cpu_uncached, tolerance);

        cpu.prefill(&oracle.forward.prefill, &GptOssCancellationToken::new())?;
        let decode_token = *oracle
            .forward
            .decode
            .first()
            .ok_or("missing two-layer decode token")?;
        let cpu_decode = cpu.decode(decode_token, &GptOssCancellationToken::new())?;
        assert_close(&cpu_decode, &oracle.forward.decode_logits, tolerance);

        cuda.prefill(&oracle.forward.prefill, &GptOssCancellationToken::new())?;
        let cuda_decode = cuda
            .decode(decode_token, &GptOssCancellationToken::new())?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&cuda_decode, &oracle.forward.decode_logits, tolerance);
        assert_close(&cuda_decode, &cpu_decode, tolerance);

        let mut traced = cuda_model_from_parts(config, weights, 4)?;
        for token in [1u32, 2, 3] {
            trace_token_layers(&mut traced, token)?;
        }
        let actual_layers = trace_token_layers(&mut traced, oracle.trace.token)?;
        assert_eq!(actual_layers.len(), oracle.trace.layers.len());
        assert_eq!(traced.cache_len(), 4);
        for (layer_index, (actual, expected)) in
            actual_layers.iter().zip(&oracle.trace.layers).enumerate()
        {
            assert_eq!(expected.sliding, layer_index.is_multiple_of(2));
            assert_close(
                &flatten_tensor(&actual.attention)?,
                &expected.attention,
                tolerance,
            );
            assert_close(
                &flatten_tensor(&actual.post_attention_hidden)?,
                &expected.post_attention_hidden,
                tolerance,
            );
            assert_close(
                &flatten_tensor(&actual.expert_contribution)?,
                &expected.expert_contribution,
                tolerance,
            );
            assert_close(
                &flatten_tensor(&actual.post_moe_hidden)?,
                &expected.post_moe_hidden,
                tolerance,
            );
            assert_close(
                &flatten_tensor(&actual.router)?,
                &expected.router,
                tolerance,
            );
            let actual_indices = actual
                .expert_indices
                .flatten_all()?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>();
            assert_eq!(actual_indices, expected.expert_indices);
            assert_close(
                &flatten_tensor(&actual.expert_weights)?,
                &expected.expert_weights,
                tolerance,
            );
            assert!(
                expected
                    .expert_weights
                    .windows(2)
                    .any(|weights| { (weights[0] - weights[1]).abs() > tolerance }),
                "layer {layer_index} must exercise unequal routing weights"
            );
        }
        traced.synchronize()?;
        Ok(())
    }

    #[test]
    fn lossless_tables_preserve_full_model_cpu_parity_and_selected_bytes() -> TestResult {
        let (config, mut weights) = model_parts_with_options(2, false, true);
        // Author a synthetic BF16-valued source; runtime never rounds F32 weights.
        for value in &mut weights.token_embedding {
            *value = f32::from_bits(value.to_bits() & 0xffff0000);
        }
        let table: Vec<f32> = weights
            .lm_head
            .weights()
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xffff0000))
            .collect();
        weights.lm_head = DenseLinear::new(
            weights.lm_head.rows(),
            weights.lm_head.cols(),
            table,
            weights.lm_head.bias().map(|b| b.to_vec()),
        )?;
        let table_elements = weights.token_embedding.len() + weights.lm_head.weights().len();
        let static_bytes = cuda_weight_resident_bytes(&weights)?;
        let mut cpu = GptOssModel::new(config.clone(), weights.clone())?;
        let mut cuda = cuda_model_from_parts(config, weights, 8)?;
        assert_eq!(cuda.weights.token_embedding.dtype(), DType::BF16);
        assert_eq!(cuda.weights.lm_head.weight.dtype(), DType::BF16);
        assert_eq!(cuda.static_resident_bytes(), static_bytes);
        assert_eq!(
            cuda.weights.token_embedding.elem_count()
                * cuda.weights.token_embedding.dtype().size_in_bytes()
                + cuda.weights.lm_head.weight.elem_count()
                    * cuda.weights.lm_head.weight.dtype().size_in_bytes(),
            table_elements * 2
        );
        let cancel = GptOssCancellationToken::new();
        let expected = cpu.prefill(&[1, 2, 3, 4], &cancel)?;
        let actual = cuda
            .prefill(&[1, 2, 3, 4], &cancel)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&actual, &expected, 1e-4);
        cuda.reset_cache()?;
        assert_eq!(
            actual,
            cuda.prefill(&[1, 2, 3, 4], &cancel)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn bf16_table_storage_keeps_f32_inputs_and_component_parity() -> TestResult {
        let device = Device::new_cuda(0)?;
        for width in [1, 31, 32, 33, 2880] {
            let rows = 9;
            let weights: Vec<f32> = (0..rows * width)
                .map(|i| ((i % 17) as f32 - 8.) * 0.03125)
                .collect();
            let table = super::super::cuda_storage::upload_table(&weights, (rows, width), &device)?;
            assert_eq!(table.dtype(), DType::BF16);
            assert_eq!(
                table
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                weights
            );
            let inputs: Vec<f32> = (0..2 * width)
                .map(|i| f32::from_bits(0x3f80_0001) * ((i % 11) as f32 - 5.) * 0.015625)
                .collect();
            let input = Tensor::from_slice(&inputs, (2, width), &device)?;
            let actual = super::super::cuda_storage::matmul(&input, &table)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let dense = Tensor::from_slice(&weights, (rows, width), &device)?;
            let expected = input.matmul(&dense.t()?)?.flatten_all()?.to_vec1::<f32>()?;
            assert_close(&actual, &expected, 1e-4);
            // Exercise both nonzero offsets and non-contiguous input/weight views.
            let padded_input = Tensor::cat(&[&input, &input], 0)?.narrow(0, 2, 2)?;
            let padded_weight = Tensor::cat(&[&table, &table], 0)?.narrow(0, rows, rows)?;
            let viewed = super::super::cuda_storage::matmul(&padded_input, &padded_weight)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, viewed);
            let permuted_input = input.t()?.force_contiguous()?.t()?;
            let permuted_weight = table.t()?.force_contiguous()?.t()?;
            assert_eq!(
                actual,
                super::super::cuda_storage::matmul(&permuted_input, &permuted_weight)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
        let input = Tensor::from_slice(&[f32::from_bits(0x3f80_0001)], (1, 1), &device)?;
        let one = super::super::cuda_storage::upload_table(&[1.], (1, 1), &device)?;
        assert_eq!(
            super::super::cuda_storage::matmul(&input, &one)?.to_vec2::<f32>()?[0][0].to_bits(),
            0x3f80_0001
        );
        let f32_table = super::super::cuda_storage::upload_table(
            &[f32::from_bits(0x3f80_0001)],
            (1, 1),
            &device,
        )?;
        assert_eq!(f32_table.dtype(), DType::F32);
        assert!(super::super::cuda_storage::matmul(&input, &f32_table).is_err());
        let wrong = Tensor::zeros((1, 2), DType::BF16, &device)?;
        assert!(super::super::cuda_storage::matmul(&input, &wrong).is_err());
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn paged_cuda_window_crossing_rollback_and_cpu_replay() -> TestResult {
        let (config, weights) = model_parts_with_options(2, true, true);
        let mut cpu = GptOssModel::new_with_limits(
            config.clone(),
            weights.clone(),
            GptOssResourceLimits {
                max_sequence_tokens: 140,
                max_cache_bytes: cache_bytes_for_tokens(&config, 140)?,
                max_total_device_bytes: usize::MAX,
            },
        )?;
        let mut cuda = cuda_model_from_parts(config.clone(), weights, 140)?;
        let cancel = GptOssCancellationToken::new();
        let prefix: Vec<u32> = (0..63).map(|i| (i % 8) as u32).collect();
        let expected = cpu.prefill(&prefix, &cancel)?;
        let actual = cuda
            .prefill(&prefix, &cancel)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&actual, &expected, 1e-4);
        let usage = cuda.resource_usage()?;
        let before = cuda
            .cache
            .layers
            .iter()
            .map(|l| l.materialize()?.0.to_vec2::<f32>())
            .collect::<candle::Result<Vec<_>>>()?;
        assert!(matches!(
            cuda.prefill(&[1, 2, 3], &GptOssCancellationToken::cancel_after_checks(8)),
            Err(GptOssCudaError::Cancelled)
        ));
        assert_eq!(cuda.resource_usage()?, usage);
        let saved_head = cuda.weights.lm_head.weight.clone();
        cuda.weights.lm_head.weight =
            Tensor::from_slice(&vec![f32::NAN; 8 * 32], (8, 32), cuda.device())?;
        assert!(cuda.prefill(&[1, 2, 3], &cancel).is_err());
        cuda.weights.lm_head.weight = saved_head;
        assert_eq!(cuda.resource_usage()?, usage);
        for (layer, old) in cuda.cache.layers.iter().zip(before) {
            assert_eq!(layer.materialize()?.0.to_vec2::<f32>()?, old);
        }
        for i in 63..131 {
            let token = (i % 8) as u32;
            let expected = cpu.decode(token, &cancel)?;
            let actual = cuda
                .decode(token, &cancel)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_close(&actual, &expected, 1e-4);
            assert_eq!(cuda.cache.layers[0].rows(), config.sliding_window);
            assert_eq!(cuda.cache.layers[1].rows(), i + 1);
            assert_eq!(
                cuda.resource_usage()?.cache_bytes,
                super::super::cuda_cache_bytes_for_tokens(&config, i + 1)?
            );
            assert!(
                cuda.resource_usage()?.cache_capacity_bytes
                    <= cuda_cache_capacity_bytes_for_tokens(&config, i + 1)?
            );
        }
        let total =
            cuda_peak_device_bytes_for_forward(&config, cuda.static_resident_bytes(), 131, 1)?;
        cuda.limits.max_total_device_bytes = total - 1;
        let before = cuda.resource_usage()?;
        assert!(cuda.decode(0, &cancel).is_err());
        assert_eq!(cuda.resource_usage()?, before);
        cuda.limits.max_total_device_bytes = total;
        cuda.decode(0, &cancel)?;
        let committed = cuda.resource_usage()?;
        let fresh_peak =
            cuda_peak_device_bytes_for_forward(&config, cuda.static_resident_bytes(), 0, 2)?
                + committed.cache_capacity_bytes;
        cuda.limits.max_total_device_bytes = fresh_peak - 1;
        assert!(matches!(
            cuda.forward_uncached(&[1, 2], &cancel),
            Err(GptOssCudaError::ResourceLimit {
                resource: "total device bytes",
                ..
            })
        ));
        assert_eq!(cuda.resource_usage()?, committed);
        cuda.limits.max_total_device_bytes = fresh_peak;
        cuda.forward_uncached(&[1, 2], &cancel)?;
        assert_eq!(cuda.resource_usage()?, committed);
        cuda.reset_cache()?;
        assert_eq!(cuda.resource_usage()?.cache_capacity_bytes, 0);
        Ok(())
    }

    #[test]
    fn cuda_admission_cancellation_and_eviction_are_failure_atomic() -> TestResult {
        let mut cuda = cuda_model()?;
        let cancelled = GptOssCancellationToken::new();
        cancelled.cancel();
        assert!(matches!(
            cuda.prefill(&[1], &cancelled),
            Err(GptOssCudaError::Cancelled)
        ));
        assert_eq!(cuda.cache_len(), 0);

        let during_work = GptOssCancellationToken::cancel_after_checks(4);
        assert!(matches!(
            cuda.prefill(&[1, 2], &during_work),
            Err(GptOssCudaError::Cancelled)
        ));
        assert_eq!(cuda.cache_len(), 0);
        assert_eq!(cuda.resource_usage()?.cache_bytes, 0);

        cuda.prefill(&[1, 2, 3], &GptOssCancellationToken::new())?;
        assert_eq!(cuda.cache_len(), 3);
        cuda.decode(4, &GptOssCancellationToken::new())?;
        assert_eq!(cuda.cache_len(), 4);
        assert!(matches!(
            cuda.decode(0, &GptOssCancellationToken::new()),
            Err(GptOssCudaError::ResourceLimit {
                resource: "sequence tokens",
                requested: 5,
                limit: 4,
            })
        ));
        assert_eq!(cuda.cache_len(), 4);

        cuda.evict_cache()?;
        assert_eq!(cuda.cache_len(), 0);
        assert_eq!(cuda.resource_usage()?.cache_bytes, 0);
        cuda.decode(3, &GptOssCancellationToken::new())?;
        assert_eq!(cuda.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn cuda_delayed_cancellation_and_nonfinite_output_are_failure_atomic_and_recoverable(
    ) -> TestResult {
        let mut cuda = cuda_model()?;
        let delayed = GptOssCancellationToken::cancel_after_checks(4);
        assert!(matches!(
            cuda.decode(1, &delayed),
            Err(GptOssCudaError::Cancelled)
        ));
        assert_eq!(cuda.cache_len(), 0);

        let valid_lm_head = cuda.weights.lm_head.weight.clone();
        cuda.weights.lm_head.weight =
            Tensor::from_slice(&vec![f32::NAN; 8 * 32], (8, 32), cuda.device())?;
        let error = cuda
            .decode(1, &GptOssCancellationToken::new())
            .expect_err("non-finite output must not commit cache state");
        assert!(matches!(
            error,
            GptOssCudaError::Invalid { ref message } if message.contains("non-finite")
        ));
        assert_eq!(cuda.cache_len(), 0);

        cuda.weights.lm_head.weight = valid_lm_head;
        let output = cuda.decode(1, &GptOssCancellationToken::new())?;
        assert!(output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(cuda.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn cuda_rejects_unsupported_dtype_and_static_budget_before_device_open() -> TestResult {
        assert!(matches!(
            GptOssCudaConfig::new(
                0,
                DType::F16,
                usize::MAX,
                GptOssResourceLimits {
                    max_sequence_tokens: 1,
                    max_cache_bytes: 128,
                    max_total_device_bytes: usize::MAX,
                },
            ),
            Err(GptOssCudaError::UnsupportedDtype { .. })
        ));
        let (config, weights) = model_parts();
        let static_bytes = cuda_weight_resident_bytes(&weights)?;
        let limits = GptOssResourceLimits {
            max_sequence_tokens: 1,
            max_cache_bytes: cache_bytes_for_tokens(&config, 1)?,
            max_total_device_bytes: usize::MAX,
        };
        let cuda_config = GptOssCudaConfig::new(usize::MAX, DType::F32, static_bytes - 1, limits)?;
        assert!(matches!(
            GptOssCudaModel::new(config, weights, cuda_config),
            Err(GptOssCudaError::ResourceLimit {
                resource: "static weight bytes",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn cuda_model_revalidates_all_public_config_fields_before_device_open() -> TestResult {
        let (config, weights) = model_parts();
        let limits = GptOssResourceLimits::for_config(&config)?;
        let invalid_dtype = GptOssCudaConfig {
            device_index: 0,
            activation_dtype: DType::F16,
            max_weight_bytes: usize::MAX,
            limits,
        };
        assert!(matches!(
            GptOssCudaModel::new(config.clone(), weights.clone(), invalid_dtype),
            Err(GptOssCudaError::UnsupportedDtype { .. })
        ));

        let invalid_budget = GptOssCudaConfig {
            device_index: 0,
            activation_dtype: DType::F32,
            max_weight_bytes: 0,
            limits,
        };
        assert!(matches!(
            GptOssCudaModel::new(config.clone(), weights.clone(), invalid_budget),
            Err(GptOssCudaError::ResourceLimit {
                resource: "static weight bytes",
                ..
            })
        ));

        let invalid_limits = GptOssCudaConfig {
            device_index: 0,
            activation_dtype: DType::F32,
            max_weight_bytes: usize::MAX,
            limits: GptOssResourceLimits {
                max_sequence_tokens: 0,
                max_cache_bytes: 0,
                max_total_device_bytes: 0,
            },
        };
        assert!(matches!(
            GptOssCudaModel::new(config, weights, invalid_limits),
            Err(GptOssCudaError::Invalid { ref message }) if message.contains("max_sequence_tokens")
        ));
        Ok(())
    }
}
