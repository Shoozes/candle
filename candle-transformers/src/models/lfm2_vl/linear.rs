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
                let output = weight.forward(&xs.contiguous()?)?;
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
