use candle::{DType, Result, Tensor, D};

/// The expert bias affects selection only, not the gathered mixture weights.
pub(crate) fn route(
    logits: &Tensor,
    bias: Option<&Tensor>,
    topk: usize,
    normalize: bool,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let (_, experts) = logits.dims2()?;
    if topk == 0 || topk > experts || !scale.is_finite() || scale <= 0. {
        candle::bail!("invalid LFM2-MoE routing top-K or scale")
    }
    let weights = candle_nn::ops::sigmoid(logits)?;
    let mut scores = weights.to_dtype(DType::F32)?;
    if let Some(bias) = bias {
        if bias.dims() != [experts] {
            candle::bail!("LFM2-MoE expert bias shape mismatch")
        }
        scores = scores.broadcast_add(&bias.to_dtype(DType::F32)?)?;
    }
    if scores
        .to_vec2::<f32>()?
        .iter()
        .flatten()
        .any(|v| !v.is_finite())
    {
        candle::bail!("LFM2-MoE routing scores must be finite")
    }
    let ids = scores
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, topk)?
        .contiguous()?;
    let mut selected = weights.gather(&ids, D::Minus1)?;
    if normalize {
        selected = selected.broadcast_div(&(&selected.sum_keepdim(D::Minus1)? + 1e-6)?)?;
    }
    Ok((ids, (selected * scale)?))
}

/// Group selected token rows without moving hidden states or expert math off device.
pub(crate) fn groups(ids: &Tensor, experts: usize) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let (_, k) = ids.dims2()?;
    let mut groups = vec![(Vec::new(), Vec::new()); experts];
    for (token, row) in ids.to_vec2::<u32>()?.into_iter().enumerate() {
        let token_id = u32::try_from(token).map_err(candle::Error::wrap)?;
        for (slot, expert) in row.into_iter().enumerate() {
            let group = groups.get_mut(expert as usize).ok_or_else(|| {
                candle::Error::Msg("LFM2-MoE route contains an invalid expert".into())
            })?;
            let weight_index = token
                .checked_mul(k)
                .and_then(|n| n.checked_add(slot))
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE route index overflows".into()))?;
            group.0.push(token_id);
            group.1.push(weight_index);
        }
    }
    Ok(groups)
}
