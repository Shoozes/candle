//! LiquidCodec speech-code synthesis, using native Candle operations and F32 inverse STFT.
//!
//! Graph reference: LiquidAI/LFM3-Audio-2B, revision 9d49701c372d6011d900f6abb40c6b296fb706dd.

use super::lfm3::{Backbone, Config as LfmConfig, LayerType, RopeParameters};
use candle::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

pub(crate) mod signal;
mod stream;
pub use stream::StreamingCache;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub encoder_dim: usize,
    pub quantizer_dim: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    pub stft_frame_stack: usize,
    pub fsq_levels: Vec<u32>,
    pub n_q: usize,
    pub decoder_window: usize,
    pub sampling_rate: u32,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if [
            self.encoder_dim,
            self.quantizer_dim,
            self.stft_n_fft,
            self.stft_hop_length,
            self.stft_win_length,
            self.stft_frame_stack,
            self.n_q,
            self.decoder_window,
        ]
        .contains(&0)
            || self.fsq_levels.is_empty()
            || self.fsq_levels.iter().any(|v| *v < 2)
            || self.sampling_rate == 0
        {
            candle::bail!("LiquidCodec dimensions, FSQ levels and sample rate must be positive")
        }
        if self.stft_n_fft != self.stft_win_length
            || !self.stft_n_fft.is_multiple_of(2)
            || self.stft_hop_length >= self.stft_win_length
            || !(self.stft_win_length - self.stft_hop_length).is_multiple_of(2)
        {
            candle::bail!("LiquidCodec requires an even full-length FFT window and symmetric overlap trimming")
        }
        self.codebook_size()?;
        let spectrum_width = self
            .stft_n_fft
            .checked_add(2)
            .ok_or_else(|| candle::Error::Msg("LiquidCodec spectrum width overflow".into()))?;
        for (a, b) in [
            (self.quantizer_dim, self.encoder_dim),
            (self.encoder_dim, spectrum_width),
            (self.stft_n_fft, 2),
        ] {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| {
                    candle::Error::Msg("LiquidCodec tensor byte size overflow".into())
                })?;
        }
        self.n_q
            .checked_mul(self.fsq_levels.len())
            .and_then(|n| n.checked_mul(self.quantizer_dim))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| candle::Error::Msg("LiquidCodec projection overflow".into()))?;
        self.stft_hop_length
            .checked_mul(self.stft_frame_stack)
            .ok_or_else(|| candle::Error::Msg("LiquidCodec frame size overflow".into()))?;
        Ok(())
    }

    pub fn codebook_size(&self) -> Result<u32> {
        self.fsq_levels
            .iter()
            .try_fold(1u32, |n, level| n.checked_mul(*level))
            .ok_or_else(|| candle::Error::Msg("LiquidCodec FSQ codebook size overflow".into()))
    }

    pub fn frame_samples(&self) -> Result<usize> {
        self.stft_frame_stack
            .checked_mul(self.stft_hop_length)
            .ok_or_else(|| candle::Error::Msg("LiquidCodec samples per frame overflow".into()))
    }

    fn decoder_config(&self) -> LfmConfig {
        // These are the architecture constants of the pinned LiquidCodecConfig.lfm_config,
        // whose released file exposes encoder_dim and decoder_window as configurable widths.
        LfmConfig {
            vocab_size: 1,
            hidden_size: self.encoder_dim,
            intermediate_size: 2304,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 64,
            max_position_embeddings: 131_072,
            norm_eps: 1e-5,
            tie_word_embeddings: false,
            conv_bias: false,
            conv_kernel_size: 7,
            layer_types: (0..8)
                .map(|i| {
                    if [2, 4, 6].contains(&i) {
                        LayerType::SlidingAttention
                    } else {
                        LayerType::Conv
                    }
                })
                .collect(),
            sliding_window: self.decoder_window,
            use_sandwich_norm: true,
            is_causal: true,
            rope_parameters: HashMap::from([(
                "sliding_attention".into(),
                RopeParameters {
                    rope_type: "default".into(),
                    rope_theta: 1_000_000.,
                    partial_rotary_factor: 1.,
                },
            )]),
        }
    }
}

/// Mixed-radix inverse FSQ, in frame, codebook, scalar-dimension order.
pub fn decode_fsq(config: &Config, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
    config.validate()?;
    let size = config.codebook_size()?;
    let capacity = codes
        .len()
        .checked_mul(config.n_q)
        .and_then(|n| n.checked_mul(config.fsq_levels.len()))
        .ok_or_else(|| candle::Error::Msg("LiquidCodec FSQ allocation overflow".into()))?;
    let mut values = Vec::with_capacity(capacity);
    for frame in codes {
        if frame.len() != config.n_q {
            candle::bail!("LiquidCodec frame has the wrong number of codebooks")
        }
        for &code in frame {
            if code >= size {
                candle::bail!("LiquidCodec code {code} is outside 0..{size}; remove EOS frames before decoding")
            }
            let mut index = code;
            for &level in &config.fsq_levels {
                let half = (level / 2) as f32;
                values.push(((index % level) as f32 - half) / half);
                index /= level;
            }
        }
    }
    Ok(values)
}

#[derive(Debug, Clone)]
pub struct Decoder {
    config: Config,
    project_out: Option<Linear>,
    from_quantizer: Linear,
    backbone: Backbone,
    output: Linear,
    window: Vec<f32>,
}

impl Decoder {
    /// Load the original released audio_tokenizer/model.safetensors namespace.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        if vb.dtype() != DType::F32 {
            candle::bail!("LiquidCodec currently requires F32 execution")
        }
        let fsq_dim = config.n_q * config.fsq_levels.len();
        let project_out = if fsq_dim == config.quantizer_dim {
            None
        } else {
            Some(candle_nn::linear(
                fsq_dim,
                config.quantizer_dim,
                vb.pp("quantizer.project_out"),
            )?)
        };
        let backbone = Backbone::new(&config.decoder_config(), vb.pp("decoder.lfm"))?;
        let bins = config.stft_n_fft / 2 + 1;
        let window = vb
            .get(config.stft_win_length, "decoder.istft.window")?
            .to_vec1::<f32>()?;
        if window.iter().any(|v| !v.is_finite()) {
            candle::bail!("LiquidCodec synthesis window is nonfinite")
        }
        Ok(Self {
            config: config.clone(),
            project_out,
            from_quantizer: candle_nn::linear(
                config.quantizer_dim,
                config.encoder_dim,
                vb.pp("from_quantizer"),
            )?,
            backbone,
            output: candle_nn::linear(config.encoder_dim, 2 * bins, vb.pp("decoder.lin"))?,
            window,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Decode one finite segment. EOS codes are rejected rather than coerced into codec codes.
    pub fn decode(&self, codes: &[Vec<u32>], max_frames: usize) -> Result<Vec<f32>> {
        if codes.is_empty() || max_frames == 0 || codes.len() > max_frames {
            candle::bail!("LiquidCodec requires a nonempty segment within the caller's frame limit")
        }
        let repeated = self.upsample_codes(codes)?;
        let features = self
            .output
            .forward(&self.backbone.forward_uncached(&repeated)?)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        signal::synthesize(&features, &self.window, self.config.stft_hop_length)
    }

    fn upsample_codes(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        let c = &self.config;
        let frames = codes
            .len()
            .checked_mul(c.stft_frame_stack)
            .ok_or_else(|| candle::Error::Msg("LiquidCodec upsample length overflow".into()))?;
        if frames > self.backbone.config().max_position_embeddings {
            candle::bail!("LiquidCodec segment exceeds its backbone context")
        }
        let latent = Tensor::from_vec(
            decode_fsq(c, codes)?,
            (1, codes.len(), c.n_q * c.fsq_levels.len()),
            self.backbone.device(),
        )?;
        let latent = match &self.project_out {
            Some(p) => p.forward(&latent)?,
            None => latent,
        };
        let embeddings = self.from_quantizer.forward(&latent)?;
        // Integer nearest-exact upsampling repeats each codec frame frame_stack times.
        embeddings
            .unsqueeze(2)?
            .expand((1, codes.len(), c.stft_frame_stack, c.encoder_dim))?
            .contiguous()?
            .reshape((1, frames, c.encoder_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_pcm_matches_full_and_finalization_and_failure_are_transactional() -> Result<()> {
        let device = candle::Device::Cpu;
        let lfm = super::super::lfm3::tests::tiny_config();
        let body = Backbone::new(
            &lfm,
            VarBuilder::from_tensors(
                super::super::lfm3::tests::weights(&lfm)?,
                DType::F32,
                &device,
            ),
        )?;
        let config = Config {
            encoder_dim: 8,
            quantizer_dim: 4,
            stft_n_fft: 8,
            stft_hop_length: 2,
            stft_win_length: 8,
            stft_frame_stack: 2,
            fsq_levels: vec![2, 3],
            n_q: 2,
            decoder_window: 3,
            sampling_rate: 24000,
        };
        let linear = |output, input| -> Result<Linear> {
            let data = (0..output * input)
                .map(|i| ((i as f32 + 1.) * 0.71).sin() * 0.05)
                .collect::<Vec<_>>();
            Ok(Linear::new(
                Tensor::from_vec(data, (output, input), &device)?,
                Some(Tensor::zeros(output, DType::F32, &device)?),
            ))
        };
        let decoder = Decoder {
            config,
            project_out: None,
            from_quantizer: linear(8, 4)?,
            backbone: body,
            output: linear(10, 8)?,
            window: (0..8)
                .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / 8.).cos())
                .collect(),
        };
        let codes = vec![vec![1, 2], vec![3, 4], vec![5, 0]];
        let full = decoder.decode(&codes, 3)?;
        let mut cache = decoder.streaming_cache(3)?;
        assert!(decoder.decode_chunk(&[], &mut cache, true).is_err());
        let mut streamed = decoder.decode_chunk(&codes[..1], &mut cache, false)?;
        assert_eq!(cache.frames(), 1);
        assert!(decoder
            .decode_chunk(&[vec![6, 0]], &mut cache, false)
            .is_err());
        assert_eq!(cache.frames(), 1);
        streamed.extend(decoder.decode_chunk(&codes[1..], &mut cache, false)?);
        assert!(decoder
            .decode_chunk(&codes[..1], &mut cache, false)
            .is_err());
        streamed.extend(decoder.decode_chunk(&[], &mut cache, true)?);
        assert!(cache.is_finished());
        assert_eq!(cache.retained_overlap_samples(), 0);
        assert!(decoder.decode_chunk(&[], &mut cache, true).is_err());
        assert_eq!(full.len(), 12);
        assert_eq!(streamed.len(), full.len());
        for (a, b) in full.iter().zip(streamed) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }
    #[test]
    fn fsq_mixed_radix_and_eos_rejection() -> Result<()> {
        let c: Config = serde_json::from_str(r#"{"encoder_dim":512,"quantizer_dim":256,"stft_n_fft":1280,"stft_hop_length":320,"stft_win_length":1280,"stft_frame_stack":6,"fsq_levels":[8,7,6,6],"n_q":12,"decoder_window":30,"sampling_rate":24000}"#).map_err(candle::Error::wrap)?;
        assert_eq!(c.codebook_size()?, 2016);
        assert_eq!(c.frame_samples()?, 1920);
        let values = decode_fsq(&c, &[vec![0; 12], vec![2015; 12]])?;
        assert_eq!(&values[..4], &[-1.; 4]);
        let high = &values[48..52];
        for (a, b) in high.iter().zip([0.75, 1., 2. / 3., 2. / 3.]) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(decode_fsq(&c, &[vec![2016; 12]]).is_err());
        assert!(decode_fsq(&c, &[vec![0; 11]]).is_err());
        let mut invalid = c.clone();
        invalid.stft_n_fft = usize::MAX - 1;
        invalid.stft_win_length = invalid.stft_n_fft;
        assert!(invalid.validate().is_err());
        let mut invalid = c.clone();
        invalid.n_q = 1;
        invalid.fsq_levels = vec![2];
        invalid.quantizer_dim = usize::MAX / 8;
        assert!(invalid.validate().is_err());
        Ok(())
    }
}
