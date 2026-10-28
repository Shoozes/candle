//! LFM3-Audio's native text backbone, speech embeddings and MaskGIT depth decoder.
//!
//! Applications own task templates, request scheduling and audio playback. These operations
//! consume released safetensors directly; they do not invoke another inference runtime.

use super::lfm3::{self, Backbone, Cache};
use candle::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};
use rand::{seq::SliceRandom, Rng};
use serde::Deserialize;

pub mod encoder;
pub mod features;

/// Replace only audio placeholder embeddings, preserving every other text position.
pub fn merge_audio_embeddings(
    ids: &[u32],
    embeddings: &Tensor,
    audio_token: u32,
    encoded: &Tensor,
) -> Result<Tensor> {
    let (batch, length, width) = embeddings.dims3()?;
    let (audio_batch, frames, audio_width) = encoded.dims3()?;
    let count = ids.iter().filter(|&&id| id == audio_token).count();
    if batch != 1
        || audio_batch != 1
        || length != ids.len()
        || length == 0
        || frames == 0
        || frames != count
        || width != audio_width
        || embeddings.dtype() != encoded.dtype()
        || !embeddings.device().same_device(encoded.device())
    {
        candle::bail!("audio placeholders and encoded frame embeddings must match exactly")
    }
    let mut pieces = Vec::with_capacity(length);
    let mut cursor = 0;
    for (i, &id) in ids.iter().enumerate() {
        pieces.push(if id == audio_token {
            let frame = encoded.narrow(1, cursor, 1)?;
            cursor += 1;
            frame
        } else {
            embeddings.narrow(1, i, 1)?
        });
    }
    Tensor::cat(&pieces, 1)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(alias = "lfm")]
    pub text_config: lfm3::Config,
    #[serde(alias = "depthformer")]
    pub depth_config: lfm3::Config,
    #[serde(alias = "codebooks")]
    pub num_codebooks: usize,
    #[serde(alias = "audio_vocab_size")]
    pub codebook_size: usize,
    pub maskgit_steps: usize,
    #[serde(default)]
    pub maskgit_schedule: Option<Vec<usize>>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        self.text_config.validate()?;
        self.depth_config.validate()?;
        if !self.text_config.is_causal
            || self.depth_config.is_causal
            || self.num_codebooks == 0
            || self.codebook_size < 2
            || self.num_codebooks > self.depth_config.max_position_embeddings
        {
            candle::bail!("invalid LFM3-Audio text/depth causality or codebook dimensions")
        }
        self.num_codebooks
            .checked_mul(self.depth_config.hidden_size)
            .and_then(|n| n.checked_mul(self.text_config.hidden_size))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                candle::Error::Msg("LFM3-Audio depth projection size overflow".into())
            })?;
        let rows = self
            .num_codebooks
            .checked_mul(self.codebook_size)
            .ok_or_else(|| candle::Error::Msg("LFM3-Audio codebook table overflow".into()))?;
        if rows > u32::MAX as usize {
            candle::bail!("LFM3-Audio codebook table exceeds u32 indices")
        }
        rows.checked_mul(self.text_config.hidden_size)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| candle::Error::Msg("LFM3-Audio codebook byte size overflow".into()))?;
        self.schedule()?;
        Ok(())
    }

    pub fn schedule(&self) -> Result<Vec<usize>> {
        if let Some(schedule) = &self.maskgit_schedule {
            if schedule.is_empty()
                || schedule.contains(&0)
                || schedule.iter().try_fold(0usize, |a, n| a.checked_add(*n))
                    != Some(self.num_codebooks)
            {
                candle::bail!("LFM3-Audio MaskGIT schedule must reveal every codebook exactly once")
            }
            return Ok(schedule.clone());
        }
        if self.maskgit_steps == 0 || self.maskgit_steps > self.num_codebooks {
            candle::bail!("LFM3-Audio MaskGIT steps must be in 1..=num_codebooks")
        }
        let mut schedule = vec![self.num_codebooks / self.maskgit_steps; self.maskgit_steps];
        if let Some(last) = schedule.last_mut() {
            *last += self.num_codebooks % self.maskgit_steps;
        }
        Ok(schedule)
    }

    pub fn audio_eos_code(&self) -> Result<u32> {
        self.codebook_size
            .checked_sub(1)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| candle::Error::Msg("invalid audio EOS codebook size".into()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// Number of preceding audio frames considered per codebook; zero disables history.
    pub repetition_window: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.,
            top_k: 128,
            repetition_penalty: 1.1,
            repetition_window: 4096,
        }
    }
}

impl Sampling {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.
        {
            candle::bail!("sampling temperature must be finite/nonnegative and repetition penalty finite/positive")
        }
        Ok(())
    }

    pub fn sample(&self, logits: &[f32], history: &[u32], rng: &mut impl Rng) -> Result<u32> {
        self.validate()?;
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            candle::bail!("cannot sample empty or nonfinite logits")
        }
        let mut scores = logits.to_vec();
        let history = &history[history.len().saturating_sub(self.repetition_window)..];
        let mut seen = std::collections::HashSet::new();
        for &id in history {
            if seen.insert(id) {
                let s = scores.get_mut(id as usize).ok_or_else(|| {
                    candle::Error::Msg("sampling history token out of vocabulary".into())
                })?;
                *s = if *s < 0. {
                    *s * self.repetition_penalty
                } else {
                    *s / self.repetition_penalty
                };
            }
        }
        if scores.iter().any(|v| !v.is_finite()) {
            candle::bail!("sampling repetition penalty overflow")
        }
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        if self.temperature == 0. {
            return Ok(order[0] as u32);
        }
        let count = if self.top_k == 0 {
            order.len()
        } else {
            self.top_k.min(order.len())
        };
        let cutoff = scores[order[count - 1]];
        order.retain(|&i| scores[i] >= cutoff);
        let max = scores[order[0]];
        let weights = order
            .iter()
            .map(|&i| ((scores[i] - max) / self.temperature).exp())
            .collect::<Vec<_>>();
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0. {
            candle::bail!("invalid sampling probability mass")
        }
        let mut draw = rng.random::<f32>() * total;
        for (&index, weight) in order.iter().zip(weights) {
            draw -= weight;
            if draw <= 0. {
                return Ok(index as u32);
            }
        }
        order
            .last()
            .map(|i| *i as u32)
            .ok_or_else(|| candle::Error::Msg("empty sampling candidate set".into()))
    }
}

#[derive(Debug, Clone)]
struct Codebook {
    embedding: Embedding,
    norm: RmsNorm,
}

#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    pub language_model: lfm3::Model,
    depth: Backbone,
    depth_linear: Linear,
    mask: Tensor,
    codebooks: Vec<Codebook>,
    audio_embedding: Embedding,
}

impl Model {
    /// Original released names, including `lfm.model` and `depthformer.model`.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let language_model = lfm3::Model::new(
            &config.text_config,
            vb.pp("lfm.model"),
            Some(vb.pp("lfm.lm_head")),
        )?;
        let d = &config.depth_config;
        let mut codebooks = Vec::with_capacity(config.num_codebooks);
        for i in 0..config.num_codebooks {
            let b = vb.pp("depth_embeddings").pp(i);
            codebooks.push(Codebook {
                embedding: candle_nn::embedding(
                    config.codebook_size,
                    d.hidden_size,
                    b.pp("embedding"),
                )?,
                norm: candle_nn::rms_norm(d.hidden_size, d.norm_eps, b.pp("embedding_norm"))?,
            });
        }
        Ok(Self {
            config: config.clone(),
            language_model,
            depth: Backbone::new(d, vb.pp("depthformer.model"))?,
            depth_linear: candle_nn::linear(
                config.text_config.hidden_size,
                config.num_codebooks * d.hidden_size,
                vb.pp("depth_linear"),
            )?,
            mask: vb.get(d.hidden_size, "mask_embedding")?,
            codebooks,
            audio_embedding: candle_nn::embedding(
                config.num_codebooks * config.codebook_size,
                config.text_config.hidden_size,
                vb.pp("codebook_embedding"),
            )?,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn prefill(&self, ids: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let x = self.language_model.embed_tokens(ids)?;
        let hidden = self.language_model.backbone.forward_embeds(&x, cache)?;
        let length = hidden.dim(1)?;
        hidden.i((0, length - 1, ..))
    }

    pub fn audio_frame_embedding(&self, codes: &[u32]) -> Result<Tensor> {
        self.validate_codes(codes)?;
        let offset = codes
            .iter()
            .enumerate()
            .map(|(i, &code)| code + (i * self.config.codebook_size) as u32)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(offset, self.config.num_codebooks, self.depth.device())?;
        self.audio_embedding.forward(&ids)?.sum(0)?.reshape((
            1,
            1,
            self.config.text_config.hidden_size,
        ))
    }

    fn validate_codes(&self, codes: &[u32]) -> Result<()> {
        if codes.len() != self.config.num_codebooks
            || codes
                .iter()
                .any(|&id| id as usize >= self.config.codebook_size)
        {
            candle::bail!("audio frame must contain one in-range code per codebook")
        }
        Ok(())
    }

    pub fn depth_logits(&self, hidden: &Tensor, codes: &[u32], masked: &[bool]) -> Result<Tensor> {
        self.validate_codes(codes)?;
        if hidden.dims() != [self.config.text_config.hidden_size] || masked.len() != codes.len() {
            candle::bail!("invalid LFM3-Audio depth hidden state or mask shape")
        }
        let mut embedded = Vec::with_capacity(codes.len());
        for ((&code, &masked), b) in codes.iter().zip(masked).zip(&self.codebooks) {
            embedded.push(if masked {
                self.mask
                    .reshape((1, 1, self.config.depth_config.hidden_size))?
            } else {
                b.embedding
                    .forward(&Tensor::new(&[code], hidden.device())?)?
                    .unsqueeze(0)?
            });
        }
        let projected = self.depth_linear.forward(&hidden.unsqueeze(0)?)?.reshape((
            1,
            self.config.num_codebooks,
            self.config.depth_config.hidden_size,
        ))?;
        let depth = self
            .depth
            .forward_uncached(&(projected + Tensor::cat(&embedded, 1)?)?)?;
        let heads = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let h = b.norm.forward(&depth.i((0, i, ..))?)?;
                h.unsqueeze(0)?
                    .matmul(&b.embedding.embeddings().t()?)?
                    .squeeze(0)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&heads, 0)
    }

    /// One complete frame, with one bidirectional pass per configured schedule slice.
    /// RNG/seed identity is Rust-specific; it is not PyTorch RNG compatibility.
    pub fn sample_audio_frame(
        &self,
        hidden: &Tensor,
        history: &[Vec<u32>],
        sampling: Sampling,
        rng: &mut impl Rng,
    ) -> Result<Vec<u32>> {
        sampling.validate()?;
        let history = &history[history.len().saturating_sub(sampling.repetition_window)..];
        for frame in history {
            self.validate_codes(frame)?;
        }
        let mut order = (0..self.config.num_codebooks).collect::<Vec<_>>();
        order.shuffle(rng);
        let mut codes = vec![0u32; order.len()];
        let mut masked = vec![true; order.len()];
        let mut start = 0;
        for count in self.config.schedule()? {
            let logits = self
                .depth_logits(hidden, &codes, &masked)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let selected = order
                .get(start..start + count)
                .ok_or_else(|| candle::Error::Msg("invalid MaskGIT schedule range".into()))?;
            for &i in selected {
                let previous = history.iter().map(|f| f[i]).collect::<Vec<_>>();
                codes[i] = sampling.sample(&logits[i], &previous, rng)?;
            }
            for &i in selected {
                masked[i] = false;
            }
            start += count;
        }
        Ok(codes)
    }

    pub fn advance_audio_frame(&self, codes: &[u32], cache: &mut Cache) -> Result<Tensor> {
        let output = self
            .language_model
            .backbone
            .forward_embeds(&self.audio_frame_embedding(codes)?, cache)?;
        output.i((0, 0, ..))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn depth_masks_hide_codes_and_audio_embedding_offsets_cannot_cross_tables() -> Result<()> {
        use candle::Device;
        let text = lfm3::tests::tiny_config();
        let mut depth = text.clone();
        depth.num_hidden_layers = 1;
        depth.layer_types = vec![lfm3::LayerType::FullAttention];
        depth.use_sandwich_norm = false;
        depth.is_causal = false;
        depth.max_position_embeddings = 4;
        let cfg = Config {
            text_config: text,
            depth_config: depth,
            num_codebooks: 4,
            codebook_size: 7,
            maskgit_steps: 3,
            maskgit_schedule: None,
        };
        assert_eq!(cfg.schedule()?, vec![1, 1, 2]);
        let device = Device::Cpu;
        let mut w = std::collections::HashMap::new();
        for (root, config) in [
            ("lfm.model", &cfg.text_config),
            ("depthformer.model", &cfg.depth_config),
        ] {
            for (name, tensor) in lfm3::tests::weights(config)? {
                w.insert(format!("{root}.{name}"), tensor);
            }
        }
        let mut add = |name: String, shape: Vec<usize>, norm: bool| -> Result<()> {
            let data = (0..shape.iter().product())
                .map(|i| if norm { 1. } else { (i % 31) as f32 * 0.03 })
                .collect::<Vec<_>>();
            w.insert(name, Tensor::from_vec(data, shape, &device)?);
            Ok(())
        };
        add("depth_linear.weight".into(), vec![32, 8], false)?;
        add("depth_linear.bias".into(), vec![32], false)?;
        add("mask_embedding".into(), vec![8], false)?;
        add("codebook_embedding.weight".into(), vec![28, 8], false)?;
        for i in 0..4 {
            add(
                format!("depth_embeddings.{i}.embedding.weight"),
                vec![7, 8],
                false,
            )?;
            add(
                format!("depth_embeddings.{i}.embedding_norm.weight"),
                vec![8],
                true,
            )?;
        }
        let model = Model::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &device))?;
        let hidden = Tensor::from_vec(vec![0.1f32; 8], 8, &device)?;
        let a = model.depth_logits(&hidden, &[1; 4], &[true; 4])?;
        let b = model.depth_logits(&hidden, &[2; 4], &[true; 4])?;
        assert_eq!((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?, 0.);
        let frame = [1, 2, 3, 4];
        let embeds = model
            .audio_frame_embedding(&frame)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (d, actual) in embeds.iter().enumerate() {
            let expected: f32 = frame
                .iter()
                .enumerate()
                .map(|(book, code)| (((book * 7 + *code as usize) * 8 + d) % 31) as f32 * 0.03)
                .sum();
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(model.audio_frame_embedding(&[7; 4]).is_err());
        let mut first = rand::rngs::StdRng::seed_from_u64(99);
        let mut second = rand::rngs::StdRng::seed_from_u64(99);
        assert_eq!(
            model.sample_audio_frame(&hidden, &[], Sampling::default(), &mut first)?,
            model.sample_audio_frame(&hidden, &[], Sampling::default(), &mut second)?
        );
        Ok(())
    }

    #[test]
    fn audio_embedding_replacement_preserves_text_and_rejects_count_mismatch() -> Result<()> {
        let e = Tensor::from_vec(
            (0..10).map(|v| v as f32).collect(),
            (1, 5, 2),
            &candle::Device::Cpu,
        )?;
        let a = Tensor::new(&[[[20f32, 21.], [30., 31.]]], &candle::Device::Cpu)?;
        let out = merge_audio_embeddings(&[1, 9, 2, 9, 3], &e, 9, &a)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(out, vec![0., 1., 20., 21., 4., 5., 30., 31., 8., 9.]);
        assert!(merge_audio_embeddings(&[1, 9, 2, 3, 4], &e, 9, &a).is_err());
        Ok(())
    }

    #[test]
    fn greedy_sampling_applies_unique_history_penalty_before_selection() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let s = Sampling {
            temperature: 0.,
            top_k: 1,
            repetition_penalty: 2.,
            repetition_window: 4096,
        };
        assert_eq!(s.sample(&[4., 3., -1.], &[0, 0], &mut rng)?, 1);
        assert_eq!(s.sample(&[-2., -3.], &[0], &mut rng)?, 1);
        assert!(s.sample(&[0.], &[1], &mut rng).is_err());
        assert!(s.sample(&[f32::NAN], &[], &mut rng).is_err());
        Ok(())
    }

    #[test]
    fn top_one_sampling_and_fixed_seed_replay() -> Result<()> {
        let mut a = rand::rngs::StdRng::seed_from_u64(7);
        let mut b = rand::rngs::StdRng::seed_from_u64(7);
        let top_one = Sampling {
            top_k: 1,
            ..Sampling::default()
        };
        for _ in 0..10 {
            assert_eq!(top_one.sample(&[0., 1., 0.5], &[], &mut a)?, 1);
        }
        let mut a = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..20 {
            assert_eq!(
                Sampling::default().sample(&[0., 1., 0.5], &[], &mut a)?,
                Sampling::default().sample(&[0., 1., 0.5], &[], &mut b)?
            );
        }
        Ok(())
    }

    #[test]
    fn top_k_sampling_keeps_cutoff_ties() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let s = Sampling {
            top_k: 1,
            ..Sampling::default()
        };
        let mut seen = [false; 2];
        for _ in 0..64 {
            let token = s.sample(&[3., 3., 2.], &[], &mut rng)? as usize;
            assert!(token < 2, "a score below the cutoff must be excluded");
            seen[token] = true;
        }
        assert_eq!(
            seen,
            [true, true],
            "both cutoff ties must remain candidates"
        );
        assert_eq!(
            Sampling {
                temperature: 0.,
                ..s
            }
            .sample(&[3., 3., 2.], &[], &mut rng)?,
            0
        );
        Ok(())
    }

    #[test]
    fn direct_sampling_respects_the_repetition_window() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut s = Sampling {
            temperature: 0.,
            repetition_penalty: 2.,
            repetition_window: 0,
            ..Sampling::default()
        };
        assert_eq!(s.sample(&[4., 3.], &[0], &mut rng)?, 0);
        s.repetition_window = 1;
        assert_eq!(s.sample(&[4., 3.], &[0, 1], &mut rng)?, 0);
        assert_eq!(s.sample(&[4., 3.], &[99, 1], &mut rng)?, 0);
        assert!(s.sample(&[4., 3.], &[0, 99], &mut rng).is_err());
        s.repetition_window = 2;
        assert_eq!(s.sample(&[4., 3.], &[0, 1], &mut rng)?, 0);
        assert_eq!(s.sample(&[4., 3.], &[0], &mut rng)?, 1);
        Ok(())
    }
}
