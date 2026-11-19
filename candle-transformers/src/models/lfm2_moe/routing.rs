use candle::{DType, Result, Tensor, D};

/// The expert bias affects selection only, not the gathered mixture weights.
pub(crate) fn route(
    logits: &Tensor,
    bias: Option<&Tensor>,
    topk: usize,
    normalize: bool,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    if !scale.is_finite() || scale <= 0. {
        candle::bail!("invalid LFM2-MoE routing scale")
    }
    let (ids, mut selected) = select(logits, bias, topk, candle_nn::ops::sigmoid)?;
    if normalize {
        selected = selected.broadcast_div(&denominator(selected.sum_keepdim(D::Minus1)?)?)?;
    }
    Ok((ids, (selected * scale)?))
}

pub(crate) fn route_gguf(logits: &Tensor, bias: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (ids, selected) = select(logits, Some(bias), topk, gguf_sigmoid)?;
    let weights = selected.broadcast_div(&denominator(gguf_raw_sum(&selected)?)?)?;
    Ok((ids, weights))
}

fn gguf_sigmoid(logits: &Tensor) -> Result<Tensor> {
    if logits.device().is_cuda() && logits.dtype() == DType::F32 {
        // Round the exponent to F32 before the add, as in the CPU sigmoid.
        let exp = logits
            .to_dtype(DType::F64)?
            .neg()?
            .exp()?
            .to_dtype(DType::F32)?;
        return (exp + 1.0)?.recip();
    }
    candle_nn::ops::sigmoid(logits)
}

fn gguf_raw_sum(selected: &Tensor) -> Result<Tensor> {
    if selected.device().is_cpu() || (selected.device().is_cuda() && selected.dtype() == DType::F32)
    {
        // Accumulate in F64 and round once before applying the GGUF denominator.
        return selected
            .to_dtype(DType::F64)?
            .sum_keepdim(D::Minus1)?
            .to_dtype(selected.dtype());
    }
    selected.sum_keepdim(D::Minus1)
}

fn denominator(sum: Tensor) -> Result<Tensor> {
    sum + 1e-6
}

fn select(
    logits: &Tensor,
    bias: Option<&Tensor>,
    topk: usize,
    sigmoid: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (_, experts) = logits.dims2()?;
    if topk == 0 || topk > experts {
        candle::bail!("invalid LFM2-MoE routing top-K")
    }
    let weights = sigmoid(logits)?;
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
    let selected = weights.gather(&ids, D::Minus1)?;
    Ok((ids, selected))
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

#[cfg(test)]
#[test]
fn gguf_routing_rounds_the_raw_sum_before_adding_epsilon() -> Result<()> {
    let weights = [0x3f171661, 0x3ea27484, 0x3e8ba988, 0x3e8049ce].map(f32::from_bits);
    let selected = Tensor::from_slice(&weights, (1, 4), &candle::Device::Cpu)?;
    let sum = gguf_raw_sum(&selected)?;
    assert_eq!(sum.to_vec2::<f32>()?[0][0].to_bits(), 0x3fb72527);
    let denominator = denominator(sum)?;
    assert_eq!(denominator.to_vec2::<f32>()?[0][0].to_bits(), 0x3fb7252f);
    #[cfg(feature = "cuda")]
    {
        let cuda = candle::Device::new_cuda(0)?;
        let sum = gguf_raw_sum(&selected.to_device(&cuda)?)?;
        assert_eq!(sum.to_vec2::<f32>()?[0][0].to_bits(), 0x3fb72527);
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[test]
fn gguf_cuda_sigmoid_rounds_before_adding_one() -> Result<()> {
    use candle::Device;

    let cpu = Tensor::from_vec(
        (0..259)
            .map(|i| (i as f32 - 129.) / 17.)
            .chain([f32::NEG_INFINITY, -0., 0., f32::INFINITY])
            .collect::<Vec<_>>(),
        (1, 263),
        &Device::Cpu,
    )?;
    let cuda = Device::new_cuda(0)?;
    let reference = gguf_sigmoid(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let actual = gguf_sigmoid(&cpu.to_device(&cuda)?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(
        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        reference.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
    );
    Ok(())
}
