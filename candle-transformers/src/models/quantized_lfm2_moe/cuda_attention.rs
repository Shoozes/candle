//! Reference-order F32 attention for the quantized MoE CUDA mixer.

use candle::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{LaunchConfig, PushKernelArg},
        kernels, WrapErr,
    },
    CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor,
};

#[derive(Clone, Copy)]
pub(super) enum Stage {
    #[cfg(test)]
    Scores,
    #[cfg(test)]
    Probabilities,
    Values,
}

struct OrderedAttention<'a> {
    mask: Option<&'a Tensor>,
    _stage: Stage,
}

impl CustomOp3 for OrderedAttention<'_> {
    fn name(&self) -> &'static str {
        "lfm2-moe-ordered-attention"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("ordered MoE attention requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        q_layout: &Layout,
        k: &CudaStorage,
        k_layout: &Layout,
        v: &CudaStorage,
        v_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, heads, queries, width) = q_layout.shape().dims4()?;
        let (_, kv_heads, keys, _) = k_layout.shape().dims4()?;
        let offsets = |layout: &Layout| {
            layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())
        };
        let (q_start, q_end) = offsets(q_layout)?;
        let (k_start, k_end) = offsets(k_layout)?;
        let (v_start, v_end) = offsets(v_layout)?;
        let number = |n| u32::try_from(n).map_err(candle::Error::wrap);
        let count = q_end - q_start;
        let rows = count / width;
        let cells = rows
            .checked_mul(keys)
            .ok_or_else(|| candle::Error::Msg("MoE CUDA attention scores overflow".into()))?;
        let cells_arg = number(cells)?;
        let rows_arg = number(rows)?;
        let elements_arg = number(count)?;
        let queries_arg = number(queries)?;
        let keys_arg = number(keys)?;
        let width_arg = number(width)?;
        let heads_per_kv = number(heads / kv_heads)?;
        if rows_arg > i32::MAX as u32 {
            candle::bail!("MoE CUDA attention row grid exceeds its limit")
        }
        let device = q.device();
        let q = q.as_cuda_slice::<f32>()?.slice(q_start..q_end);
        let k = k.as_cuda_slice::<f32>()?.slice(k_start..k_end);
        let v = v.as_cuda_slice::<f32>()?.slice(v_start..v_end);
        let mut scores = unsafe { device.alloc::<f32>(cells)? };

        let mask_storage = self.mask.map(Tensor::storage_and_layout);
        let mask = match &mask_storage {
            Some((storage, layout)) => {
                let Storage::Cuda(storage) = &**storage else {
                    candle::bail!("MoE CUDA attention mask requires CUDA storage")
                };
                let (start, end) = offsets(layout)?;
                Some(storage.as_cuda_slice::<u8>()?.slice(start..end))
            }
            None => None,
        };
        let score_kernel =
            device.get_or_load_func("lfm2_moe_attention_scores_f32", &kernels::REDUCE)?;
        let mut launch = score_kernel.builder();
        launch.arg(&q).arg(&k);
        if let Some(mask) = &mask {
            launch.arg(mask);
        } else {
            // The kernel leaves this argument unread when the mask flag is zero.
            launch.arg(&q);
        }
        launch.arg(&mut scores);
        let masked = u32::from(mask.is_some());
        candle::builder_arg!(
            launch,
            cells_arg,
            queries_arg,
            keys_arg,
            width_arg,
            heads_per_kv,
            masked
        );
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (cells_arg.div_ceil(16), 1, 1),
                block_dim: (8, 16, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        drop(launch);
        #[cfg(test)]
        if matches!(self._stage, Stage::Scores) {
            return Ok((
                CudaStorage::wrap_cuda_slice(scores, device.clone()),
                (batch, heads, queries, keys).into(),
            ));
        }

        let softmax =
            device.get_or_load_func("lfm2_moe_attention_softmax_f32", &kernels::REDUCE)?;
        let mut launch = softmax.builder();
        launch.arg(&mut scores);
        let scale = (width as f32).sqrt().recip();
        candle::builder_arg!(launch, keys_arg, scale);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows_arg, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        drop(launch);
        #[cfg(test)]
        if matches!(self._stage, Stage::Probabilities) {
            return Ok((
                CudaStorage::wrap_cuda_slice(scores, device.clone()),
                (batch, heads, queries, keys).into(),
            ));
        }

        let mut output = unsafe { device.alloc::<f32>(count)? };
        let (kernel, config) = if queries == 1 {
            (
                "lfm2_moe_attention_decode_values_f32",
                LaunchConfig {
                    grid_dim: (elements_arg.div_ceil(16), 1, 1),
                    block_dim: (8, 16, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                "lfm2_moe_attention_values_f32",
                LaunchConfig::for_num_elems(elements_arg),
            )
        };
        let values = device.get_or_load_func(kernel, &kernels::REDUCE)?;
        let mut launch = values.builder();
        launch.arg(&scores).arg(&v).arg(&mut output);
        candle::builder_arg!(
            launch,
            elements_arg,
            queries_arg,
            keys_arg,
            width_arg,
            heads_per_kv
        );
        unsafe { launch.launch(config) }.w()?;
        drop(launch);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (batch, heads, queries, width).into(),
        ))
    }
}

pub(crate) fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    forward_stage(q, k, v, mask, Stage::Values)
}

pub(super) fn forward_stage(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    stage: Stage,
) -> Result<Option<Tensor>> {
    if [q, k, v]
        .iter()
        .any(|t| !t.device().is_cuda() || t.dtype() != DType::F32)
    {
        return Ok(None);
    }
    let (batch, heads, queries, width) = q.dims4()?;
    let (kv_batch, kv_heads, keys, kv_width) = k.dims4()?;
    if width == 0 || !width.is_multiple_of(32) {
        return Ok(None);
    }
    if batch == 0
        || heads == 0
        || queries == 0
        || keys == 0
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || batch != kv_batch
        || width != kv_width
        || k.shape() != v.shape()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
    {
        candle::bail!("invalid quantized MoE CUDA attention dimensions or devices")
    }
    if mask.is_some_and(|m| {
        m.dims() != [queries, keys] || m.dtype() != DType::U8 || !m.device().same_device(q.device())
    }) {
        return Ok(None);
    }
    let mask = mask.map(Tensor::contiguous).transpose()?;
    let op = OrderedAttention {
        mask: mask.as_ref(),
        _stage: stage,
    };
    Ok(Some(q.contiguous()?.apply_op3_no_bwd(
        &k.contiguous()?,
        &v.contiguous()?,
        &op,
    )?))
}
