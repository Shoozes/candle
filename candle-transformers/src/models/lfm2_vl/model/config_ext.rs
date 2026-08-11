impl siglip2::Siglip2VisionConfig {
    pub(super) fn patch_dimension_for_vl(&self) -> Result<usize> {
        self.num_channels
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch dimension overflow".into()))
    }
}
