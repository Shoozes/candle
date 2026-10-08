use candle::quantized::{QMatMul, QTensor};
use candle::{Module, Result, Tensor};
use std::sync::Arc;

/// A vision/projector linear that preserves dense weights or executes Q8 weights in-place.
#[derive(Clone, Debug)]
pub enum LinearOp {
    Dense(candle_nn::Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl LinearOp {
    pub(crate) fn from_qtensor(weight: QTensor, bias: Option<Tensor>) -> Self {
        // Construct the variant directly. QMatMul::from_qtensor honors
        // CANDLE_DEQUANTIZE_ALL, which would defeat this explicit Q8 path.
        let weight = QMatMul::QTensor(Arc::new(weight));
        Self::Quantized { weight, bias }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(&xs.contiguous()?),
            Self::Quantized { weight, bias } => {
                let dims = xs.dims();
                let Some(&input_width) = dims.last() else {
                    candle::bail!("quantized linear requires a non-scalar input")
                };
                if input_width == 0 {
                    candle::bail!("quantized linear requires a positive input width")
                }
                let rows = dims[..dims.len() - 1]
                    .iter()
                    .try_fold(1usize, |rows, &dim| {
                        rows.checked_mul(dim).ok_or_else(|| {
                            candle::Error::Msg("quantized linear row count overflow".into())
                        })
                    })?;
                if rows == 0 {
                    candle::bail!("quantized linear cannot forward an empty input")
                }
                let flat = xs.contiguous()?.reshape((rows, input_width))?;
                let output = weight.forward(&flat)?;
                let (_, output_width) = output.dims2()?;
                let mut shape = dims.to_vec();
                if let Some(last) = shape.last_mut() {
                    *last = output_width;
                }
                let output = output.reshape(shape)?;
                match bias {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                }
            }
        }
    }
}

impl Module for LinearOp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::GgmlDType;
    use candle::{DType, Device};

    #[test]
    fn q8_linear_retains_qtensor_storage() -> Result<()> {
        let device = Device::Cpu;
        let values = (0..(32 * 32))
            .map(|index| (index as f32 - 511.5) / 1024.0)
            .collect::<Vec<_>>();
        let dense_weight = Tensor::from_vec(values, (32, 32), &device)?;
        let quantized = QTensor::quantize(&dense_weight, GgmlDType::Q8_0)?;
        let linear =
            LinearOp::from_qtensor(quantized, Some(Tensor::zeros(32, DType::F32, &device)?));
        match &linear {
            LinearOp::Quantized {
                weight: QMatMul::QTensor(weight),
                ..
            } => assert_eq!(weight.dtype(), GgmlDType::Q8_0),
            other => panic!("expected retained Q8_0 storage, got {other:?}"),
        }
        let output = linear.forward(&Tensor::ones((2, 32), DType::F32, &device)?)?;
        assert_eq!(output.dims(), [2, 32]);
        Ok(())
    }

    fn rank_four_case(device: &Device) -> Result<()> {
        let values = Tensor::arange(0f32, 2048f32, device)?.reshape((64, 32))?;
        let weight = QTensor::quantize(&values, GgmlDType::Q8_0)?;
        let linear = LinearOp::from_qtensor(weight, None);
        let input = Tensor::arange(0f32, 640f32, device)?
            .reshape((5, 4, 32))?
            .narrow(0, 1, 3)?
            .reshape((1, 3, 4, 32))?
            .permute((0, 2, 1, 3))?;
        let (result, report) = candle::quantized::with_native_q8_0(|| linear.forward(&input));
        let output = result?;
        assert_eq!(output.dims(), [1, 4, 3, 64]);
        let (flat, _) = candle::quantized::with_native_q8_0(|| {
            linear.forward(&input.contiguous()?.reshape((12, 32))?)
        });
        let flat = flat?.reshape((1, 4, 3, 64))?;
        let error = (&output - flat)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert_eq!(error, 0.0);
        assert!(report.cpu_matmuls + report.cuda_f32_q8 > 0);
        Ok(())
    }

    #[test]
    fn q8_rank_four_permuted_offset_cpu_preserves_shape() -> Result<()> {
        rank_four_case(&Device::Cpu)
    }

    #[test]
    fn q8_rank_four_contiguous_permuted_offset_match_q8_operands_cpu() -> Result<()> {
        let device = Device::Cpu;
        let dense = Tensor::arange(0f32, 2048f32, &device)?
            .affine(1. / 2048., 0.)?
            .reshape((64, 32))?;
        let weight = QTensor::quantize(&dense, GgmlDType::Q8_0)?;
        let reference = weight.dequantize(&device)?.t()?;
        let linear = LinearOp::from_qtensor(weight, None);
        let base = Tensor::arange(0f32, 640f32, &device)?
            .affine(1. / 640., 0.)?
            .reshape((1, 5, 4, 32))?;
        let offset = base.narrow(1, 1, 3)?;
        for input in [base, offset.clone(), offset.permute((0, 2, 1, 3))?] {
            let actual = linear.forward(&input)?;
            let rows = input.elem_count() / 32;
            let mut shape = input.dims().to_vec();
            if let Some(width) = shape.last_mut() {
                *width = 64;
            }
            // Native CPU Q8 matmul quantizes its F32 activation rows internally.
            let flat = input.contiguous()?.reshape((rows, 32))?;
            // Give the oracle owned rows: QTensor quantization expects zero-offset storage.
            let rows_data = flat
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let owned = Tensor::from_vec(rows_data, (rows, 32), &device)?;
            let activation = QTensor::quantize(&owned, GgmlDType::Q8_0)?.dequantize(&device)?;
            let expected = activation.matmul(&reference)?.reshape(shape)?;
            let error = (actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                error <= 1e-4,
                "native Q8 versus dequantized Q8 operands: {error}"
            );
            let dense = flat.matmul(&reference)?;
            let (native, _) = candle::quantized::with_native_q8_0(|| linear.forward(&input));
            let error = (native?.flatten_to(2)?.reshape((rows, 64))? - dense)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                error <= 1e-4,
                "scoped Q8/F32 versus dense operands: {error}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn q8_rank_four_permuted_offset_cuda_preserves_shape() -> Result<()> {
        rank_four_case(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn q8_f32_cuda_contiguous_permuted_offset_match_dense_operands() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let weight = Tensor::arange(0f32, 4096f32, &device)?
            .affine(1. / 2048., -1.)?
            .reshape((64, 64))?;
        let weight = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
        let reference = weight.dequantize(&device)?.t()?;
        let linear = LinearOp::from_qtensor(weight, None);
        let base = Tensor::arange(0f32, 1280f32, &device)?
            .affine(1. / 640., -1.)?
            .reshape((1, 5, 4, 64))?;
        let offset = base.narrow(1, 1, 3)?;
        for input in [base, offset.clone(), offset.permute((0, 2, 1, 3))?] {
            let rows = input.elem_count() / 64;
            let expected = input
                .contiguous()?
                .reshape((rows, 64))?
                .matmul(&reference)?;
            let (actual, dispatch) = candle::quantized::with_native_q8_0(|| linear.forward(&input));
            let actual = actual?.reshape((rows, 64))?;
            let error = (actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error <= 1e-4, "CUDA Q8/F32 versus dense operands: {error}");
            assert_eq!(dispatch.cuda_f32_q8, 1);
            assert_eq!(dispatch.cpu_matmuls, 0);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dense_linear_materializes_non_contiguous_cuda_input() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dense = candle_nn::Linear::new(Tensor::ones((4, 3), DType::F32, &device)?, None);
        let linear = LinearOp::Dense(dense);
        let input = Tensor::arange(0f32, 12f32, &device)?
            .reshape((1, 2, 2, 3))?
            .permute((0, 2, 1, 3))?;
        let output = linear.forward(&input)?;
        assert_eq!(output.dims(), [1, 2, 2, 4]);
        Ok(())
    }
}
