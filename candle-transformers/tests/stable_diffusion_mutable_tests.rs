#![cfg(feature = "test-utils")]

use std::collections::BTreeMap;

use candle::{DType, Device, Tensor};
use candle_nn::{Init, VarMap};
use candle_transformers::models::stable_diffusion::{
    lora::SdxlLoraComponent, mutable::VarMapSwapTransaction,
};

const UNET_TARGET: &str = "down_blocks.0.attentions.0.to_q.weight";
const TEXT_1_TARGET: &str = "text_model.encoder.layers.0.self_attn.q_proj.weight";
const TEXT_2_TARGET: &str = "text_model.encoder.layers.0.self_attn.k_proj.weight";

fn component_varmap(target: &str, values: [f32; 4]) -> (VarMap, Tensor) {
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let observed = varmap
        .get((2, 2), target, Init::Const(0.0), DType::F32, &device)
        .unwrap();
    let base = Tensor::from_vec(values.to_vec(), (2, 2), &device).unwrap();
    varmap.set_one(target, &base).unwrap();
    (varmap, observed)
}

fn test_transaction() -> (VarMapSwapTransaction, [Tensor; 3]) {
    let (unet, observed_unet) = component_varmap(UNET_TARGET, [10.0, 20.0, 30.0, 40.0]);
    let (text_1, observed_text_1) = component_varmap(TEXT_1_TARGET, [50.0, 60.0, 70.0, 80.0]);
    let (text_2, observed_text_2) = component_varmap(TEXT_2_TARGET, [90.0, 100.0, 110.0, 120.0]);
    (
        VarMapSwapTransaction::new(unet, text_1, text_2),
        [observed_unet, observed_text_1, observed_text_2],
    )
}

fn resolver(component: SdxlLoraComponent, _stem: &str) -> Option<String> {
    Some(
        match component {
            SdxlLoraComponent::Unet => UNET_TARGET,
            SdxlLoraComponent::TextEncoder1 => TEXT_1_TARGET,
            SdxlLoraComponent::TextEncoder2 => TEXT_2_TARGET,
        }
        .to_owned(),
    )
}

fn adapter(scale: f32, components: &[SdxlLoraComponent]) -> BTreeMap<String, Tensor> {
    let device = Device::Cpu;
    let mut tensors = BTreeMap::new();
    for component in components {
        let stem = match component {
            SdxlLoraComponent::Unet => "lora_unet_down_blocks_0_attentions_0_to_q",
            SdxlLoraComponent::TextEncoder1 => {
                "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
            }
            SdxlLoraComponent::TextEncoder2 => {
                "lora_te2_text_model_encoder_layers_0_self_attn_k_proj"
            }
        };
        tensors.insert(
            format!("{stem}.lora_down.weight"),
            Tensor::from_vec(vec![scale, scale], (1, 2), &device).unwrap(),
        );
        tensors.insert(
            format!("{stem}.lora_up.weight"),
            Tensor::from_vec(vec![1.0, 1.0], (2, 1), &device).unwrap(),
        );
    }
    tensors
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn public_component_failure_hook_restores_every_prior_adapter_write() {
    let all = [
        SdxlLoraComponent::Unet,
        SdxlLoraComponent::TextEncoder1,
        SdxlLoraComponent::TextEncoder2,
    ];
    for failed_component in [
        SdxlLoraComponent::TextEncoder1,
        SdxlLoraComponent::TextEncoder2,
    ] {
        let (mut transaction, observed) = test_transaction();
        transaction
            .swap_adapter(&adapter(1.0, &all), 1.0, &resolver)
            .unwrap();
        let before = observed.each_ref().map(values);
        let plan = transaction
            .prepare_adapter(&adapter(2.0, &all), 1.0, &resolver)
            .unwrap();

        let error = transaction
            .apply_plan_with_injected_component_failure(plan, failed_component)
            .unwrap_err();

        assert!(error.to_string().contains("was rolled back"));
        assert_eq!(transaction.revision(), 1);
        for (index, tensor) in observed.iter().enumerate() {
            assert_eq!(values(tensor), before[index]);
        }
        for component in all {
            assert_eq!(
                transaction.component(component).active_target_keys().len(),
                1
            );
        }
    }
}

#[test]
fn public_component_failure_hook_rejects_an_unplanned_component_before_mutation() {
    let (mut transaction, observed) = test_transaction();
    let before = observed.each_ref().map(values);
    let plan = transaction
        .prepare_adapter(&adapter(1.0, &[SdxlLoraComponent::Unet]), 1.0, &resolver)
        .unwrap();

    let error = transaction
        .apply_plan_with_injected_component_failure(plan, SdxlLoraComponent::TextEncoder1)
        .unwrap_err();

    assert!(error.to_string().contains("no planned writes"));
    assert_eq!(transaction.revision(), 0);
    for (index, tensor) in observed.iter().enumerate() {
        assert_eq!(values(tensor), before[index]);
    }
}
