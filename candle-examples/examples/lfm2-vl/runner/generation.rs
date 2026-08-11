fn generate_once(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
) -> Result<GenerationTrace> {
    runtime.reset()?;
    let prefill_logits =
        runtime.prefill(inputs.input_ids, inputs.image_spans, inputs.encoded_images)?;
    let prefill = analyze_logits(&last_logits(&prefill_logits)?, inputs.tokenizer)?;
    validate_eos_id(inputs.eos_token_id, &prefill_logits)?;
    let mut generated_ids = Vec::new();
    generated_ids
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating generated token buffer"))?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating generation trace"))?;
    let mut next = LogitsEvidence {
        sha256: prefill.sha256.clone(),
        top_k: prefill.top_k.clone(),
    };
    let mut model_input_position = None;
    let mut model_input_token_id = None;
    let mut stop_reason = "max_new_tokens".to_owned();

    for step in 0..inputs.max_new_tokens {
        let selected = next
            .top_k
            .first()
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL logits did not contain any tokens"))?;
        let selected_token_id = selected.token_id;
        let is_eos = inputs.eos_token_id == Some(selected_token_id);
        generated_ids.push(selected_token_id);
        steps.push(GenerationStep {
            step,
            sample_position: inputs
                .prompt_len
                .checked_add(step)
                .ok_or_else(|| anyhow::anyhow!("LFM2-VL sample position overflow"))?,
            model_input_position,
            model_input_token_id,
            selected_token_id,
            selected_token: inputs.tokenizer.id_to_token(selected_token_id),
            logits_sha256: next.sha256,
            top_k: next.top_k,
            is_eos,
        });
        if is_eos {
            stop_reason = "eos".to_owned();
            break;
        }
        if step + 1 == inputs.max_new_tokens {
            break;
        }
        let input_position = inputs
            .prompt_len
            .checked_add(step)
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL decode position overflow"))?;
        let decode_ids = Tensor::new(&[selected_token_id], runtime.text_device())?.unsqueeze(0)?;
        let decode_logits = runtime.decode(&decode_ids, input_position)?;
        next = analyze_logits(&last_logits(&decode_logits)?, inputs.tokenizer)?;
        model_input_position = Some(input_position);
        model_input_token_id = Some(selected_token_id);
    }

    let generated_tokens = generated_ids
        .iter()
        .map(|&token_id| inputs.tokenizer.id_to_token(token_id))
        .collect();
    let decoded_skip_special_tokens = inputs
        .tokenizer
        .decode(&generated_ids, true)
        .map_err(anyhow::Error::msg)?;
    let decoded_with_special_tokens = inputs
        .tokenizer
        .decode(&generated_ids, false)
        .map_err(anyhow::Error::msg)?;
    Ok(GenerationTrace {
        prefill_logits_sha256: prefill.sha256,
        prefill_top_k: prefill.top_k,
        steps,
        generated_ids,
        generated_tokens,
        decoded_skip_special_tokens,
        decoded_with_special_tokens,
        stop_reason,
    })
}

fn capture_trace(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
    capture: &mut NativeTraceCapture,
) -> Result<()> {
    runtime.reset()?;
    let prefill = runtime
        .prefill_with_trace(inputs.input_ids, inputs.image_spans, inputs.encoded_images)?
        .ok_or_else(|| anyhow::anyhow!("selected runtime did not return prefill trace"))?;
    let mut next = last_logits(&prefill.logits)?
        .argmax(0)?
        .to_scalar::<u32>()?;
    capture.prefill = Some(prefill);
    capture
        .decode_input_ids
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating native trace decode input IDs"))?;
    capture
        .decode
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating native trace decode tensors"))?;
    for step in 0..inputs.max_new_tokens {
        let input_position = inputs
            .prompt_len
            .checked_add(step)
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL trace decode position overflow"))?;
        let decode_ids = Tensor::new(&[next], runtime.text_device())?.unsqueeze(0)?;
        let decode = runtime
            .decode_with_trace(&decode_ids, input_position)?
            .ok_or_else(|| anyhow::anyhow!("selected runtime did not return decode trace"))?;
        next = last_logits(&decode.logits)?.argmax(0)?.to_scalar::<u32>()?;
        capture.decode_input_ids.push(decode_ids);
        capture.decode.push(decode);
    }
    Ok(())
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        3 => {
            let (_, sequence, _) = logits.dims3()?;
            if sequence == 0 {
                bail!("LFM2-VL prefill returned an empty logits sequence")
            }
            logits.i((0, sequence - 1)).map_err(anyhow::Error::from)
        }
        2 => {
            let (batch, _) = logits.dims2()?;
            if batch != 1 {
                bail!("LFM2-VL logits batch size {batch} is unsupported; expected 1")
            }
            logits.i(0).map_err(anyhow::Error::from)
        }
        rank => bail!("LFM2-VL logits rank {rank} is unsupported; expected 2 or 3"),
    }
}

#[cfg(test)]
fn top_k(logits: &Tensor, tokenizer: &Tokenizer) -> Result<Vec<RankedToken>> {
    Ok(analyze_logits(logits, tokenizer)?.top_k)
}

fn analyze_logits(logits: &Tensor, tokenizer: &Tokenizer) -> Result<LogitsEvidence> {
    let values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        bail!("LFM2-VL logits vocabulary is empty")
    }
    if values.len() > MAX_REPORTED_VOCAB {
        bail!(
            "LFM2-VL logits vocabulary {} exceeds evidence limit {MAX_REPORTED_VOCAB}",
            values.len()
        )
    }
    if let Some((token_id, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("LFM2-VL logit for token {token_id} is not finite")
    }
    let mut hasher = Sha256::new();
    for value in &values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    let count = TOP_K.min(values.len());
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(count)
        .map_err(|_| anyhow::anyhow!("allocating top-k evidence"))?;
    for _ in 0..count {
        let mut best: Option<(usize, f32)> = None;
        for (token_id, &logit) in values.iter().enumerate() {
            if selected
                .iter()
                .any(|entry: &RankedToken| entry.token_id as usize == token_id)
            {
                continue;
            }
            if best.is_none_or(|(best_id, best_logit)| {
                logit > best_logit || (logit == best_logit && token_id < best_id)
            }) {
                best = Some((token_id, logit));
            }
        }
        let (token_id, logit) =
            best.ok_or_else(|| anyhow::anyhow!("selecting LFM2-VL top-k logits"))?;
        let token_id =
            u32::try_from(token_id).map_err(|_| anyhow::anyhow!("LFM2-VL token ID exceeds u32"))?;
        selected.push(RankedToken {
            token_id,
            token: tokenizer.id_to_token(token_id),
            logit,
        });
    }
    Ok(LogitsEvidence {
        sha256: format!("{:x}", hasher.finalize()),
        top_k: selected,
    })
}

fn validate_eos_id(eos_token_id: Option<u32>, logits: &Tensor) -> Result<()> {
    if let Some(eos_token_id) = eos_token_id {
        let vocab = *logits
            .dims()
            .last()
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL logits have no vocabulary dimension"))?;
        let eos_index = usize::try_from(eos_token_id)
            .map_err(|_| anyhow::anyhow!("EOS token ID cannot be represented as usize"))?;
        if eos_index >= vocab {
            bail!("EOS token ID {eos_token_id} is outside model vocabulary size {vocab}")
        }
    }
    Ok(())
}

fn resolve_eos(
    explicit: Option<u32>,
    model_default: Option<u32>,
    model_source: &'static str,
    tokenizer: &Tokenizer,
) -> EosEvidence {
    if let Some(token_id) = explicit {
        return EosEvidence {
            token_id: Some(token_id),
            source: "cli".to_owned(),
        };
    }
    if let Some(token_id) = model_default {
        return EosEvidence {
            token_id: Some(token_id),
            source: model_source.to_owned(),
        };
    }
    let vocab = tokenizer.get_vocab(true);
    if let Some(token_id) = EOS_CANDIDATES
        .iter()
        .find_map(|candidate| vocab.get(*candidate).copied())
    {
        return EosEvidence {
            token_id: Some(token_id),
            source: "tokenizer_guess".to_owned(),
        };
    }
    EosEvidence {
        token_id: None,
        source: "none".to_owned(),
    }
}
