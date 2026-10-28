use super::{signal, Config, Decoder};
use crate::models::lfm3::Cache;
use candle::{Module, Result};

/// Request-local synthesis state. It retains bounded attention/convolution state
/// and unfinished audio *and* squared-window envelope, never prior PCM output.
#[derive(Debug, Clone)]
pub struct StreamingCache {
    config: Config,
    backbone: Cache,
    max_frames: usize,
    frames: usize,
    tail: Option<(Vec<f32>, Vec<f32>)>,
    started: bool,
    finished: bool,
}

impl StreamingCache {
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn retained_overlap_samples(&self) -> usize {
        self.tail.as_ref().map_or(0, |t| t.0.len())
    }
    pub fn attention_lengths(&self) -> Result<Vec<usize>> {
        self.backbone.attention_lengths()
    }
}

impl Decoder {
    pub fn streaming_cache(&self, max_frames: usize) -> Result<StreamingCache> {
        let positions = max_frames
            .checked_mul(self.config.stft_frame_stack)
            .ok_or_else(|| candle::Error::Msg("LiquidCodec streaming allowance overflow".into()))?;
        if max_frames == 0
            || self.config.frame_samples()?
                < (self.config.stft_win_length - self.config.stft_hop_length) / 2
        {
            candle::bail!("invalid LiquidCodec stream frame allowance or overlap geometry")
        }
        Ok(StreamingCache {
            config: self.config.clone(),
            backbone: Cache::new(self.backbone.config(), positions)?,
            max_frames,
            frames: 0,
            tail: None,
            started: false,
            finished: false,
        })
    }

    /// A final empty chunk flushes the saved overlap. Successful finalization is
    /// terminal. Any rejected chunk leaves the caller's entire state unchanged.
    pub fn decode_chunk(
        &self,
        codes: &[Vec<u32>],
        cache: &mut StreamingCache,
        finalize: bool,
    ) -> Result<Vec<f32>> {
        if cache.config != self.config
            || cache.finished
            || (codes.is_empty() && (!finalize || !cache.started))
        {
            candle::bail!(
                "LiquidCodec stream is mismatched, finished or missing a nonempty first chunk"
            )
        }
        let frames = cache
            .frames
            .checked_add(codes.len())
            .ok_or_else(|| candle::Error::Msg("LiquidCodec stream frame count overflow".into()))?;
        if frames > cache.max_frames {
            candle::bail!("LiquidCodec stream exceeds its frame allowance")
        }
        let mut next = cache.clone();
        let overlap = self.config.stft_win_length - self.config.stft_hop_length;
        let trim = overlap / 2;
        let (mut audio, mut envelope) = if codes.is_empty() {
            next.tail
                .take()
                .ok_or_else(|| candle::Error::Msg("LiquidCodec missing final overlap".into()))?
        } else {
            let embeddings = self.upsample_codes(codes)?;
            let hidden = self
                .backbone
                .forward_embeds(&embeddings, &mut next.backbone)?;
            let features = self.output.forward(&hidden)?.squeeze(0)?.to_vec2::<f32>()?;
            let (mut audio, mut envelope) =
                signal::overlap(&features, &self.window, self.config.stft_hop_length)?;
            if let Some((previous_audio, previous_envelope)) = next.tail.take() {
                if previous_audio.len() != overlap
                    || previous_envelope.len() != overlap
                    || audio.len() < overlap
                {
                    candle::bail!("LiquidCodec streaming overlap shape mismatch")
                }
                for i in 0..overlap {
                    audio[i] += previous_audio[i];
                    envelope[i] += previous_envelope[i];
                }
            }
            (audio, envelope)
        };
        if !next.started {
            audio.drain(..trim);
            envelope.drain(..trim);
            next.started = true;
        }
        if finalize {
            let keep = audio
                .len()
                .checked_sub(trim)
                .ok_or_else(|| candle::Error::Msg("short LiquidCodec final overlap".into()))?;
            audio.truncate(keep);
            envelope.truncate(keep);
            next.finished = true;
        } else {
            let keep = audio
                .len()
                .checked_sub(overlap)
                .ok_or_else(|| candle::Error::Msg("short LiquidCodec pending overlap".into()))?;
            next.tail = Some((audio.split_off(keep), envelope.split_off(keep)));
        }
        let pcm = signal::normalize(&audio, &envelope)?;
        next.frames = frames;
        *cache = next;
        Ok(pcm)
    }
}
