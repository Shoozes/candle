//! Lossless BF16 table residency with F32 inputs, arithmetic, and outputs.

pub(crate) fn lossless_bf16(values: &[f32]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && value.to_bits() & 0xffff == 0)
}

#[cfg(feature = "cuda")]
mod device {
    use super::lossless_bf16;
    use crate::models::gpt_oss::cuda::{GptOssCudaError, GptOssCudaResult};
    use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle::{DType, Device, Shape, Storage, Tensor};

    pub(crate) fn upload_table(
        values: &[f32],
        shape: impl Into<Shape>,
        device: &Device,
    ) -> GptOssCudaResult<Tensor> {
        if !lossless_bf16(values) {
            return Ok(Tensor::from_slice(values, shape, device)?);
        }
        let source = Tensor::from_slice(values, shape, &Device::Cpu)?;
        let packed = source.to_dtype(DType::BF16)?;
        drop(source);
        Ok(packed.to_device(device)?)
    }

    pub(crate) fn matmul(input: &Tensor, weight: &Tensor) -> GptOssCudaResult<Tensor> {
        let (batch, width) = input.dims2()?;
        let (rows, cols) = weight.dims2()?;
        if input.dtype() != DType::F32
            || weight.dtype() != DType::BF16
            || !input.device().same_device(weight.device())
            || cols != width
            || batch == 0
            || rows == 0
            || width == 0
            || batch > 65535
        {
            return Err(GptOssCudaError::Invalid { message: "BF16-storage matmul requires matching nonempty CUDA tables and F32 input; batch limit is 65535".into() });
        }
        let dims = [batch, rows, width].map(|n| {
            i32::try_from(n).map_err(|_| GptOssCudaError::Overflow {
                operation: "BF16-storage matmul dimension",
            })
        });
        let [batch_i32, rows_i32, width_i32] = dims;
        let (batch_i32, rows_i32, width_i32) = (batch_i32?, rows_i32?, width_i32?);
        let elements = batch.checked_mul(rows).ok_or(GptOssCudaError::Overflow {
            operation: "BF16-storage output elements",
        })?;
        elements.checked_mul(4).ok_or(GptOssCudaError::Overflow {
            operation: "BF16-storage output bytes",
        })?;
        let contiguous = |tensor: &Tensor| match tensor.layout().contiguous_offsets() {
            Some((0, end)) if end == tensor.elem_count() => Ok(tensor.clone()),
            _ => tensor.force_contiguous(),
        };
        let input = contiguous(input)?;
        let weight = contiguous(weight)?;
        let device = input.device().as_cuda_device()?.clone();
        let (input_storage, _) = input.storage_and_layout();
        let (weight_storage, _) = weight.storage_and_layout();
        let (Storage::Cuda(inputs), Storage::Cuda(weights)) = (&*input_storage, &*weight_storage)
        else {
            return Err(GptOssCudaError::Invalid {
                message: "BF16-storage matmul requires CUDA storage".into(),
            });
        };
        let inputs = inputs.as_cuda_slice::<f32>()?;
        let candle::cuda_backend::CudaStorageSlice::BF16(weights) = &weights.slice else {
            return Err(GptOssCudaError::Invalid {
                message: "BF16 table has incompatible storage".into(),
            });
        };
        let input_elements = batch.checked_mul(width).ok_or(GptOssCudaError::Overflow {
            operation: "BF16-storage input elements",
        })?;
        let weight_elements = rows.checked_mul(width).ok_or(GptOssCudaError::Overflow {
            operation: "BF16-storage weight elements",
        })?;
        if inputs.len() < input_elements || weights.len() < weight_elements {
            return Err(GptOssCudaError::Invalid {
                message: "BF16-storage matmul buffers are shorter than their logical dimensions"
                    .into(),
            });
        }
        // All dimensions and output capacity are checked before raw pointer use.
        let mut output = unsafe { device.alloc::<f32>(elements) }?;
        let status = {
            let stream = device.cuda_stream().clone();
            let (input_ptr, _input_guard) = inputs.device_ptr(&stream);
            let (weight_ptr, _weight_guard) = weights.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
            unsafe {
                candle::cuda_backend::kernels::ffi::launch_gpt_oss_bf16_f32_matmul(
                    input_ptr as *const f32,
                    weight_ptr as *const u16,
                    output_ptr as *mut f32,
                    batch_i32,
                    rows_i32,
                    width_i32,
                    stream.cu_stream() as i64,
                )
            }
        };
        if status != 0 {
            return Err(GptOssCudaError::Kernel { status });
        }
        Ok(Tensor::from_storage(
            Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, device)),
            (batch, rows),
            candle::op::BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use device::{matmul, upload_table};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_selection_never_rounds_f32_values() {
        assert!(lossless_bf16(&[
            0.,
            -0.,
            1.,
            -2.,
            f32::from_bits(0x0001_0000)
        ]));
        assert!(!lossless_bf16(&[1., f32::from_bits(0x3f80_0001)]));
        assert!(!lossless_bf16(&[f32::from_bits(1)]));
        assert!(!lossless_bf16(&[f32::INFINITY]));
    }
}
