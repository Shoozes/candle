//! Backend-neutral local GPT-OSS loading and failure-atomic token execution.

use super::{
    GptOssCancellationToken, GptOssConfig, GptOssFileIdentity, GptOssGgufArtifact,
    GptOssLoadRegistry, GptOssModel, GptOssResourceLimits, GptOssResourceUsage,
    GptOssSafetensorsArtifact, GptOssSafetensorsSource,
};
use candle::{bail, DType, Result};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssDevice {
    Cpu,
    Cuda { device_index: usize },
}

#[derive(Debug, Clone)]
pub enum GptOssSource {
    Gguf(GptOssFileIdentity),
    PackedSafetensors(GptOssSafetensorsSource),
}

#[derive(Debug, Clone)]
pub struct GptOssLoadOptions {
    pub source: GptOssSource,
    pub device: GptOssDevice,
    pub limits: GptOssResourceLimits,
    pub max_weight_bytes: usize,
}

enum Artifact {
    Gguf(GptOssGgufArtifact),
    Safetensors(GptOssSafetensorsArtifact),
}

enum Backend {
    Cpu(Box<GptOssModel>),
    #[cfg(feature = "cuda")]
    Cuda(Box<super::GptOssCudaModel>),
}

/// Device-native logits. GPU readback is a separate fallible operation after
/// inference, so a readback error cannot masquerade as a rolled-back forward.
pub enum GptOssLogits {
    Cpu(Vec<f32>),
    #[cfg(feature = "cuda")]
    Cuda(candle::Tensor),
}

/// Successful whole-forward wall time, excluding initial session admission and
/// caller readback. CUDA final validation is nested within `forward_ms`, never
/// additive: it includes outstanding device work, transfer and finite checking,
/// not isolated DMA or kernel time. Neither field includes failed calls.
#[derive(Debug, Clone, Copy)]
pub struct GptOssForwardTimings {
    pub forward_ms: f64,
    pub cuda_final_validation_ms: Option<f64>,
}

impl GptOssLogits {
    pub fn into_cpu(self) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(values) => Ok(values),
            #[cfg(feature = "cuda")]
            Self::Cuda(values) => values.flatten_all()?.to_vec1::<f32>(),
        }
    }
}

/// Raw token/logit session. Tokenization, Harmony, sampling, process policy and
/// provider qualification belong to callers. Both backends use F32 activations
/// with packed MXFP4 experts; this API does not advertise a faster precision mode.
pub struct GptOssSession {
    backend: Backend,
    artifact: Artifact,
    device: GptOssDevice,
    limits: GptOssResourceLimits,
    static_weight_bytes: usize,
}

impl GptOssSession {
    pub fn load(
        options: &GptOssLoadOptions,
        registry: &GptOssLoadRegistry,
        cancel: &GptOssCancellationToken,
    ) -> Result<Self> {
        cancel.checkpoint()?;
        #[cfg(not(feature = "cuda"))]
        if matches!(options.device, GptOssDevice::Cuda { .. }) {
            bail!("GPT-OSS CUDA loading requires candle-transformers' cuda feature");
        }
        if options.max_weight_bytes == 0 || options.limits.max_total_device_bytes == 0 {
            bail!("GPT-OSS loading requires positive weight and execution budgets");
        }
        let budget = match options.device {
            GptOssDevice::Cpu => options
                .max_weight_bytes
                .min(options.limits.max_total_device_bytes),
            GptOssDevice::Cuda { .. } => options.max_weight_bytes,
        };
        let (artifact, config, weights, static_weight_bytes) = match &options.source {
            GptOssSource::Gguf(identity) => {
                let (artifact, handle) = GptOssGgufArtifact::open_identified_with_registry(
                    &identity.path,
                    &identity.sha256,
                    registry,
                    Some(cancel),
                )?;
                if artifact.file_size() != identity.size_bytes {
                    bail!("GPT-OSS GGUF retained file size mismatch");
                }
                options.limits.validate(artifact.config())?;
                if options.limits.max_sequence_tokens > artifact.context_length() {
                    bail!("GPT-OSS sequence limit exceeds the GGUF checkpoint context");
                }
                let static_bytes = artifact.weight_resident_bytes()?;
                Self::admit_one_token(
                    artifact.config(),
                    static_bytes,
                    options.limits,
                    options.device,
                )?;
                let mut weights = artifact.load_weights_with_cancellation(cancel, budget)?;
                weights.attach_load_handle(handle);
                let config = artifact.config().clone();
                (Artifact::Gguf(artifact), config, weights, static_bytes)
            }
            GptOssSource::PackedSafetensors(source) => {
                let (mut artifact, handle) =
                    GptOssSafetensorsArtifact::open_with_registry(source, registry, cancel)?;
                options.limits.validate(artifact.config())?;
                if options.limits.max_sequence_tokens
                    > artifact.config().maximum_sequence_tokens()?
                {
                    bail!("GPT-OSS sequence limit exceeds the packed checkpoint context");
                }
                let config = artifact.config().clone();
                let static_bytes = artifact.weight_resident_bytes()?;
                Self::admit_one_token(&config, static_bytes, options.limits, options.device)?;
                let mut weights = artifact.load_weights_with_cancellation(cancel, budget)?;
                weights.attach_load_handle(handle);
                (
                    Artifact::Safetensors(artifact),
                    config,
                    weights,
                    static_bytes,
                )
            }
        };
        cancel.checkpoint()?;
        let backend = match options.device {
            GptOssDevice::Cpu => Backend::Cpu(Box::new(GptOssModel::new_with_limits(
                config,
                weights,
                options.limits,
            )?)),
            #[cfg(feature = "cuda")]
            GptOssDevice::Cuda { device_index } => {
                let cuda = super::GptOssCudaModel::new(
                    config,
                    weights,
                    super::GptOssCudaConfig {
                        device_index,
                        activation_dtype: DType::F32,
                        max_weight_bytes: budget,
                        limits: options.limits,
                    },
                )
                .map_err(candle::Error::wrap)?;
                cancel.checkpoint()?;
                Backend::Cuda(Box::new(cuda))
            }
            #[cfg(not(feature = "cuda"))]
            GptOssDevice::Cuda { .. } => bail!("GPT-OSS CUDA loading requires the cuda feature"),
        };
        cancel.checkpoint()?;
        #[cfg(feature = "cuda")]
        let static_weight_bytes = match &backend {
            Backend::Cpu(_) => static_weight_bytes,
            Backend::Cuda(model) => model.static_resident_bytes(),
        };
        Ok(Self {
            backend,
            artifact,
            device: options.device,
            limits: options.limits,
            static_weight_bytes,
        })
    }

    pub fn config(&self) -> &GptOssConfig {
        match &self.backend {
            Backend::Cpu(model) => model.config(),
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => model.config(),
        }
    }

    pub fn device(&self) -> GptOssDevice {
        self.device
    }

    pub fn activation_dtype(&self) -> DType {
        DType::F32
    }

    pub fn static_weight_bytes(&self) -> usize {
        self.static_weight_bytes
    }

    fn admit_one_token(
        config: &GptOssConfig,
        static_bytes: usize,
        limits: GptOssResourceLimits,
        device: GptOssDevice,
    ) -> Result<()> {
        // CUDA selects its exact lossless table representation after bounded
        // source assembly, and checks device admission before opening CUDA.
        if matches!(device, GptOssDevice::Cuda { .. }) {
            return Ok(());
        }
        let total = super::total_device_bytes_for_tokens(config, static_bytes, 1)?;
        if total > limits.max_total_device_bytes {
            bail!("GPT-OSS static weights plus one-token cache/workspace require {total} bytes, limit is {}",limits.max_total_device_bytes);
        }
        Ok(())
    }

    fn admit_forward(&self, incoming: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Backend::Cuda(model) = &self.backend {
            return model.admit(incoming).map_err(candle::Error::wrap);
        }
        let current = self.resource_usage()?.sequence_tokens;
        self.limits.admit(self.config(), current, incoming)?;
        let tokens = current
            .checked_add(incoming)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS session token count overflowed".into()))?;
        let total =
            super::total_device_bytes_for_tokens(self.config(), self.static_weight_bytes, tokens)?;
        if total > self.limits.max_total_device_bytes {
            bail!(
                "GPT-OSS static/cache/workspace admission requests {total} bytes, limit is {}",
                self.limits.max_total_device_bytes
            );
        }
        Ok(())
    }

    pub fn consumed_files(&self) -> Vec<GptOssFileIdentity> {
        match &self.artifact {
            Artifact::Gguf(artifact) => vec![GptOssFileIdentity {
                path: artifact.path().to_path_buf(),
                size_bytes: artifact.file_size(),
                sha256: artifact.sha256().to_owned(),
            }],
            Artifact::Safetensors(artifact) => artifact.inventory(),
        }
    }

    pub fn resource_usage(&self) -> Result<GptOssResourceUsage> {
        match &self.backend {
            Backend::Cpu(model) => model.resource_usage(),
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => model.resource_usage().map_err(candle::Error::wrap),
        }
    }

    pub fn prefill(
        &mut self,
        tokens: &[u32],
        cancel: &GptOssCancellationToken,
    ) -> Result<GptOssLogits> {
        self.admit_forward(tokens.len())?;
        match &mut self.backend {
            Backend::Cpu(model) => model.prefill(tokens, cancel).map(GptOssLogits::Cpu),
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => model
                .prefill(tokens, cancel)
                .map(GptOssLogits::Cuda)
                .map_err(candle::Error::wrap),
        }
    }

    pub fn decode(&mut self, token: u32, cancel: &GptOssCancellationToken) -> Result<GptOssLogits> {
        self.admit_forward(1)?;
        match &mut self.backend {
            Backend::Cpu(model) => model.decode(token, cancel).map(GptOssLogits::Cpu),
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => model
                .decode(token, cancel)
                .map(GptOssLogits::Cuda)
                .map_err(candle::Error::wrap),
        }
    }

    pub fn prefill_profiled(
        &mut self,
        tokens: &[u32],
        cancel: &GptOssCancellationToken,
    ) -> Result<(GptOssLogits, GptOssForwardTimings)> {
        self.admit_forward(tokens.len())?;
        let start = Instant::now();
        let (logits, validation_ms) = match &mut self.backend {
            Backend::Cpu(model) => (GptOssLogits::Cpu(model.prefill(tokens, cancel)?), None),
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => {
                let (logits, validation) = model
                    .forward_with_validation_timing(tokens, cancel)
                    .map_err(candle::Error::wrap)?;
                (
                    GptOssLogits::Cuda(logits),
                    Some(validation.as_secs_f64() * 1000.),
                )
            }
        };
        Ok((
            logits,
            GptOssForwardTimings {
                forward_ms: start.elapsed().as_secs_f64() * 1000.,
                cuda_final_validation_ms: validation_ms,
            },
        ))
    }

    pub fn decode_profiled(
        &mut self,
        token: u32,
        cancel: &GptOssCancellationToken,
    ) -> Result<(GptOssLogits, GptOssForwardTimings)> {
        self.prefill_profiled(&[token], cancel)
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        match &mut self.backend {
            Backend::Cpu(model) => {
                model.reset_cache();
                Ok(())
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda(model) => model.reset_cache().map_err(candle::Error::wrap),
        }
    }
}
