use candle::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{LaunchConfig, PushKernelArg},
        kernels, WrapErr,
    },
    quantized::{GgmlDType, QMatMul},
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Module, Result, Shape,
    Tensor, D,
};

struct OrderedRope;

impl CustomOp3 for OrderedRope {
    fn name(&self) -> &'static str {
        "lfm2-moe-ordered-rope"
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
        candle::bail!("ordered MoE RoPE requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
        cos: &CudaStorage,
        cos_layout: &Layout,
        sin: &CudaStorage,
        sin_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (_, _, tokens, width) = layout.shape().dims4()?;
        let (rows, half) = cos_layout.shape().dims2()?;
        if rows != tokens
            || width % 2 != 0
            || half != width / 2
            || sin_layout.shape() != cos_layout.shape()
        {
            candle::bail!("invalid ordered MoE RoPE shapes")
        }
        let offsets = |layout: &Layout| {
            layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())
        };
        let (start, end) = offsets(layout)?;
        let (cos_start, cos_end) = offsets(cos_layout)?;
        let (sin_start, sin_end) = offsets(sin_layout)?;
        let elements = u32::try_from(end - start).map_err(candle::Error::wrap)?;
        let pairs = elements / 2;
        let half = u32::try_from(half).map_err(candle::Error::wrap)?;
        let tokens = u32::try_from(tokens).map_err(candle::Error::wrap)?;
        let input_slice = input.as_cuda_slice::<f32>()?.slice(start..end);
        let cos_slice = cos.as_cuda_slice::<f32>()?.slice(cos_start..cos_end);
        let sin_slice = sin.as_cuda_slice::<f32>()?.slice(sin_start..sin_end);
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(elements as usize)? };
        let function = device.get_or_load_func("lfm2_moe_rope_f32", &kernels::REDUCE)?;
        let mut launch = function.builder();
        launch
            .arg(&input_slice)
            .arg(&cos_slice)
            .arg(&sin_slice)
            .arg(&mut output);
        candle::builder_arg!(launch, pairs, tokens, half);
        unsafe { launch.launch(LaunchConfig::for_num_elems(pairs)) }.w()?;
        drop(launch);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    input.apply_op3_no_bwd(cos, sin, &OrderedRope)
}

struct ReferenceSilu;

impl CustomOp1 for ReferenceSilu {
    fn name(&self) -> &'static str {
        "lfm2-moe-reference-silu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("MoE reference CUDA SiLU requires CUDA storage")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())?;
        let count = u32::try_from(end - start).map_err(candle::Error::wrap)?;
        let input_slice = input.as_cuda_slice::<f32>()?.slice(start..end);
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(count as usize)? };
        let function = device.get_or_load_func("lfm2_moe_silu_f32", &kernels::REDUCE)?;
        let mut launch = function.builder();
        launch.arg(&input_slice).arg(&mut output);
        candle::builder_arg!(launch, count);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        drop(launch);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

pub(super) fn silu(gate: &Tensor) -> Result<Tensor> {
    if gate.elem_count() == 0 {
        return Ok(gate.clone());
    }
    gate.contiguous()?.apply_op1_no_bwd(&ReferenceSilu)
}

struct OrderedRouter;

impl CustomOp2 for OrderedRouter {
    fn name(&self) -> &'static str {
        "lfm2-moe-ordered-router"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("ordered MoE router requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = layout.shape().dims2()?;
        let (experts, weight_width) = weight_layout.shape().dims2()?;
        if rows == 0
            || experts == 0
            || width == 0
            || width != weight_width
            || !width.is_multiple_of(8)
        {
            candle::bail!("invalid ordered MoE router dimensions")
        }
        let offsets = |layout: &Layout| {
            layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())
        };
        let (start, end) = offsets(layout)?;
        let (weight_start, weight_end) = offsets(weight_layout)?;
        let cells = rows
            .checked_mul(experts)
            .ok_or_else(|| candle::Error::Msg("ordered MoE router output overflows".into()))?;
        let cells_arg = u32::try_from(cells).map_err(candle::Error::wrap)?;
        let experts_arg = u32::try_from(experts).map_err(candle::Error::wrap)?;
        let width_arg = u32::try_from(width).map_err(candle::Error::wrap)?;
        let input_slice = input.as_cuda_slice::<f32>()?.slice(start..end);
        let weight_slice = weight
            .as_cuda_slice::<f32>()?
            .slice(weight_start..weight_end);
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(cells)? };
        let kernel = if rows == 1 && width.is_multiple_of(32) {
            "lfm2_moe_router_decode_f32"
        } else {
            "lfm2_moe_router_f32"
        };
        let function = device.get_or_load_func(kernel, &kernels::REDUCE)?;
        let mut launch = function.builder();
        launch.arg(&input_slice).arg(&weight_slice).arg(&mut output);
        candle::builder_arg!(launch, cells_arg, experts_arg, width_arg);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (cells_arg.div_ceil(16), 1, 1),
                block_dim: (8, 16, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        drop(launch);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (rows, experts).into(),
        ))
    }
}

pub(super) fn router(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    input
        .contiguous()?
        .apply_op2_no_bwd(&weight.contiguous()?, &OrderedRouter)
}

struct OrderedRmsNorm {
    eps: f32,
}

impl CustomOp2 for OrderedRmsNorm {
    fn name(&self) -> &'static str {
        "lfm2-moe-ordered-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("ordered MoE CUDA RMSNorm requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let width = weight_layout.shape().dims1()?;
        let elements = layout.shape().elem_count();
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())?;
        let (weight_start, weight_end) = weight_layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::RequiresContiguous { op: self.name() }.bt())?;
        let input_slice = input.as_cuda_slice::<f32>()?.slice(start..end);
        let weight_slice = weight
            .as_cuda_slice::<f32>()?
            .slice(weight_start..weight_end);
        let device = input.device();
        let rows = u32::try_from(elements / width).map_err(candle::Error::wrap)?;
        if rows > i32::MAX as u32 {
            candle::bail!("ordered MoE CUDA RMSNorm row grid exceeds its limit")
        }
        let width = u32::try_from(width).map_err(candle::Error::wrap)?;
        let function = device.get_or_load_func("rmsnorm_f32_ggml_cpu_order", &kernels::REDUCE)?;
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let mut launch = function.builder();
        launch.arg(&input_slice).arg(&mut output).arg(&weight_slice);
        candle::builder_arg!(launch, width, self.eps);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        drop(launch);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn project_qkv(weight: &QMatMul, input: &Tensor) -> Result<Tensor> {
    match weight {
        QMatMul::QTensor(q) if q.dtype() == GgmlDType::Q4K => {
            weight.forward_cuda_q4k_q8k(&input.contiguous()?)
        }
        QMatMul::QTensor(q) if q.dtype() == GgmlDType::Q6K => {
            weight.forward_cuda_q6k_q8k_dense(&input.contiguous()?)
        }
        _ => weight.forward(input),
    }
}

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if width == 0 || input.elem_count() == 0 || weight.dims() != [width] {
        candle::bail!("invalid ordered MoE CUDA RMSNorm dimensions")
    }
    input
        .contiguous()?
        .apply_op2_no_bwd(&weight.contiguous()?, &OrderedRmsNorm { eps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn lfm2_moe_cuda_router_matches_cpu_order() -> Result<()> {
        let host = Device::Cpu;
        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(
            (0..3 * 128)
                .map(|i| ((i * 11 % 173) as f32 - 86.) / 29.)
                .collect::<Vec<_>>(),
            (3, 128),
            &host,
        )?;
        let weight = Tensor::from_vec(
            (0..19 * 128)
                .map(|i| ((i * 17 % 211) as f32 - 105.) / 37.)
                .collect::<Vec<_>>(),
            (19, 128),
            &host,
        )?;
        let bits = |t: Tensor| -> Result<Vec<u32>> {
            Ok(t.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        let cuda_weight = weight.to_device(&device)?;
        for rows in [1, 3] {
            let input = input.narrow(0, 0, rows)?;
            let expected = super::super::cpu::router(&input, &weight)?;
            let actual = super::super::cpu::router(&input.to_device(&device)?, &cuda_weight)?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(bits(actual)?, bits(expected)?);
        }
        Ok(())
    }

    #[test]
    fn lfm2_moe_cuda_rope_matches_cpu_order() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(
            (0..2 * 3 * 5 * 64)
                .map(|i| (i as f32 % 139. - 69.) / 37.)
                .collect::<Vec<_>>(),
            (2, 3, 5, 64),
            &Device::Cpu,
        )?;
        let (cos, sin) = super::super::cpu::rotary(8, 64, 10_000_000.)?;
        let cos = cos.narrow(0, 3, 5)?;
        let sin = sin.narrow(0, 3, 5)?;
        let expected = candle_nn::rotary_emb::rope(&input, &cos, &sin)?;
        let actual = rope(
            &input.to_device(&device)?,
            &cos.to_device(&device)?,
            &sin.to_device(&device)?,
        )?;
        assert_eq!(actual.dims(), expected.dims());
        let bits = |x: Tensor| -> Result<Vec<u32>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        assert_eq!(bits(actual)?, bits(expected)?);
        Ok(())
    }

    #[test]
    fn lfm2_moe_cuda_silu_matches_cpu_order() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let gate = Tensor::from_vec(
            (0..1027)
                .map(|i| (i as f32 - 513.) / 31.)
                .collect::<Vec<_>>(),
            1027,
            &Device::Cpu,
        )?;
        let expected = super::super::cpu::silu(&gate)?.to_vec1::<f32>()?;
        let actual = silu(&gate.to_device(&device)?)?.to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        Ok(())
    }

    #[test]
    fn lfm2_moe_cuda_rmsnorm_matches_cpu_on_shared_rows() -> Result<()> {
        let host = Device::Cpu;
        let device = Device::new_cuda(0)?;
        let width = 2048;
        let input = Tensor::from_vec(
            (0..3 * width)
                .map(|i| {
                    if i < width {
                        0.
                    } else {
                        ((i * 37 % 293) as f32 - 146.) / 89.
                    }
                })
                .collect::<Vec<_>>(),
            (1, 3, width),
            &host,
        )?;
        let weight = Tensor::from_vec(
            (0..width)
                .map(|i| (i % 37) as f32 / 23. - 0.5)
                .collect::<Vec<_>>(),
            width,
            &host,
        )?;
        let eps = 1e-5;
        let expected = super::super::cpu::rms_norm(&input, &weight, eps)?;
        let actual = rms_norm(&input.to_device(&device)?, &weight.to_device(&device)?, eps)?;
        let bits = |t: Tensor| -> Result<Vec<u32>> {
            Ok(t.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        assert_eq!(bits(actual)?, bits(expected)?);
        Ok(())
    }
}
