use super::*;
use std::time::Instant;

// Frozen authored operands: no retained model bytes or tokens are read.
const INPUT: usize = 2048;
const WARMUP: usize = 10;
const SAMPLES: usize = 100;

fn operands(widths: &[usize], device: &Device) -> Result<(Vec<QMatMul>, Vec<f32>)> {
    let weights = widths
        .iter()
        .enumerate()
        .map(|(seed, &width)| recovery_tests::matrix(width, INPUT, seed + 23, device))
        .collect::<Result<Vec<_>>>()?;
    let input = (0..INPUT)
        .map(|i| ((i * 23 + 7) % 127) as f32 / 127.0 - 0.5)
        .collect();
    Ok((weights, input))
}

fn measured_sequential(weights: &[QMatMul], input: &[f32], device: &Device) -> Result<[f64; 3]> {
    let mut stages = [0.0; 3];
    for weight in weights {
        let start = Instant::now();
        let x = Tensor::from_slice(input, (1, input.len()), &Device::Cpu)?.to_device(device)?;
        device.synchronize()?;
        stages[0] += start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let y = weight.forward_ggml_q8_0(&x)?;
        device.synchronize()?;
        stages[1] += start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let _values = y.flatten_all()?.to_vec1::<f32>()?;
        device.synchronize()?;
        stages[2] += start.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(stages)
}

#[test]
#[ignore = "authored CUDA profiling; run through a bounded native Job"]
fn projection_transfer_baseline() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (name, widths) in [
        ("recurrent", vec![6144, 2048, 16, 16]),
        ("ffn", vec![5632, 5632]),
    ] {
        let (weights, input) = operands(&widths, &device)?;
        for _ in 0..WARMUP {
            measured_sequential(&weights, &input, &device)?;
        }
        let samples = (0..SAMPLES)
            .map(|_| measured_sequential(&weights, &input, &device))
            .collect::<Result<Vec<_>>>()?;
        println!(
            "{}",
            serde_json::json!({"workload":name, "input":INPUT, "widths":widths,
            "warmup":WARMUP,"samples":samples,"stage_order":["h2d","kernel_sync","d2h"],
            "units":"ms", "weights":"authored q8_0 seed 23+index"})
        );
    }
    let mut model = recovery_tests::tiny(&device)?;
    model.max_positions = 64;
    bench_profile::start();
    let started = Instant::now();
    model.prefill(&(0..32).collect::<Vec<_>>())?;
    println!(
        "{}",
        serde_json::json!({"authored_host_profile": bench_profile::take(),
        "total_ms":started.elapsed().as_secs_f64()*1000.0,"tokens":32,
        "note":"host stages synchronous; legacy projection kernel stage measures launch, not GPU execution"})
    );
    device.synchronize()
}

#[test]
fn grouped_components_and_observer_order_are_bit_exact() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut sequential = recovery_tests::tiny(&device)?;
    let mut grouped = recovery_tests::tiny(&device)?;
    grouped.projection_mode = HybridProjectionMode::Grouped;
    for token in [1, 2, 3, 4] {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let left = sequential.decode_observed(
            token,
            Some(&mut |layer, name, values| {
                a.push((
                    layer,
                    name.to_owned(),
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ));
            }),
        )?;
        let right = grouped.decode_observed(
            token,
            Some(&mut |layer, name, values| {
                b.push((
                    layer,
                    name.to_owned(),
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ));
            }),
        )?;
        assert_eq!(a, b);
        assert_eq!(
            left.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            right.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    // Admitted F32 matrices retain the same arithmetic in a mixed group.
    let dense = QMatMul::Tensor(Tensor::ones((16, 2048), candle::DType::F32, &device)?);
    let (weights, input) = operands(&[16], &device)?;
    let a = projections(
        [&weights[0], &dense],
        &input,
        HybridProjectionMode::Sequential,
    )?;
    let b = projections([&weights[0], &dense], &input, HybridProjectionMode::Grouped)?;
    for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let cpu = recovery_tests::matrix(16, 2048, 23, &Device::Cpu)?;
    assert!(projections([&cpu, &weights[0]], &input, HybridProjectionMode::Grouped).is_err());
    assert!(projections([], &input, HybridProjectionMode::Grouped).is_err());
    device.synchronize()
}

fn timed(
    weights: &[QMatMul],
    input: &[f32],
    device: &Device,
    mode: HybridProjectionMode,
) -> Result<(f64, Vec<Vec<f32>>)> {
    device.synchronize()?;
    let start = Instant::now();
    let values = match weights {
        [a, b] => projections([a, b], input, mode)?.to_vec(),
        [a, b, c, d] => projections([a, b, c, d], input, mode)?.to_vec(),
        _ => bail!("authored workload group differs"),
    };
    device.synchronize()?;
    Ok((start.elapsed().as_secs_f64() * 1000.0, values))
}

#[test]
#[ignore = "authored paired CUDA timing; run through a bounded native Job"]
fn projection_grouped_paired_measurements() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (name, widths) in [
        ("recurrent", vec![6144, 2048, 16, 16]),
        ("ffn", vec![5632, 5632]),
    ] {
        let (weights, input) = operands(&widths, &device)?;
        for _ in 0..WARMUP {
            timed(&weights, &input, &device, HybridProjectionMode::Sequential)?;
            timed(&weights, &input, &device, HybridProjectionMode::Grouped)?;
        }
        let mut samples = Vec::new();
        for index in 0..SAMPLES {
            let modes = if index % 2 == 0 {
                [
                    HybridProjectionMode::Sequential,
                    HybridProjectionMode::Grouped,
                ]
            } else {
                [
                    HybridProjectionMode::Grouped,
                    HybridProjectionMode::Sequential,
                ]
            };
            let (first, a) = timed(&weights, &input, &device, modes[0])?;
            let (second, b) = timed(&weights, &input, &device, modes[1])?;
            for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            samples.push(if index % 2 == 0 {
                [first, second]
            } else {
                [second, first]
            });
        }
        println!(
            "{}",
            serde_json::json!({"workload":name,"input":INPUT,"widths":widths,
            "warmup":WARMUP,"samples":samples,"column_order":["sequential","grouped"],
            "units":"ms","bit_exact":true})
        );
    }
    device.synchronize()
}
