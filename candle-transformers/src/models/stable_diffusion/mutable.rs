//! Atomic replacement of SDXL LoRA targets held in three mutable model maps.

use std::collections::{BTreeMap, BTreeSet};

use candle::{Result, Tensor};
use candle_nn::VarMap;

use super::lora::{
    parse_sdxl_lora_pairs, prepare_lora_target, LoraApplyStats, LoraComponentApplyStats, LoraPair,
    SdxlLoraComponent, SdxlLoraTargetResolver,
};

/// One mutable SDXL component with lazily retained independent base tensors.
pub struct MutableModelComponent {
    component: SdxlLoraComponent,
    varmap: VarMap,
    base_tensors: BTreeMap<String, Tensor>,
    active_target_keys: Vec<String>,
}

impl MutableModelComponent {
    pub fn new(component: SdxlLoraComponent, varmap: VarMap) -> Self {
        Self {
            component,
            varmap,
            base_tensors: BTreeMap::new(),
            active_target_keys: Vec::new(),
        }
    }

    pub fn component(&self) -> SdxlLoraComponent {
        self.component
    }

    pub fn active_target_keys(&self) -> &[String] {
        &self.active_target_keys
    }

    fn prepare_adapter(
        &mut self,
        pairs: &[&LoraPair],
        strength: f64,
        resolver: &impl SdxlLoraTargetResolver,
    ) -> Result<ComponentSwapPlan> {
        let current = self.current_tensors()?;
        let mut merged = BTreeMap::new();
        let mut resolved_targets = BTreeSet::new();
        let mut stats = LoraComponentApplyStats::empty(self.component);
        stats.pair_count = pairs.len();

        for pair in pairs {
            let target = resolver
                .resolve_target(self.component, pair.stem())
                .ok_or_else(|| {
                    candle::Error::msg(format!(
                        "unmatched SDXL LoRA target {}:{}",
                        self.component.as_str(),
                        pair.stem()
                    ))
                })?;
            if target.is_empty() || target.trim() != target {
                candle::bail!(
                    "SDXL LoRA resolver returned an invalid target for {}:{}",
                    self.component.as_str(),
                    pair.stem()
                )
            }
            if !resolved_targets.insert(target.clone()) {
                candle::bail!(
                    "multiple SDXL LoRA pairs resolve to {} target {target}",
                    self.component.as_str()
                )
            }
            let current_tensor = current.get(&target).ok_or_else(|| {
                candle::Error::msg(format!(
                    "SDXL LoRA {} target is absent from the mutable model: {target}",
                    self.component.as_str()
                ))
            })?;
            if !self.base_tensors.contains_key(&target) {
                self.base_tensors.insert(
                    target.clone(),
                    current_tensor.copy().map_err(|error| {
                        error.context(format!(
                            "copying immutable SDXL LoRA base {}:{target}",
                            self.component.as_str()
                        ))
                    })?,
                );
            }
            let base = self.base_tensors.get(&target).ok_or_else(|| {
                candle::Error::msg(format!(
                    "immutable SDXL LoRA base is unavailable for {}:{target}",
                    self.component.as_str()
                ))
            })?;
            match prepare_lora_target(pair, &target, base, strength)? {
                Some(prepared) => {
                    stats.targets.push(prepared.evidence);
                    merged.insert(target, prepared.merged);
                }
                None => stats.zero_delta_count += 1,
            }
        }

        stats
            .targets
            .sort_by(|left, right| left.target.cmp(&right.target));
        stats.applied_count = stats.targets.len();
        let next_target_keys = merged.keys().cloned().collect::<Vec<_>>();
        self.plan_transition(next_target_keys, merged, stats)
    }

    fn prepare_clear(&self) -> Result<ComponentSwapPlan> {
        self.plan_transition(
            Vec::new(),
            BTreeMap::new(),
            LoraComponentApplyStats::empty(self.component),
        )
    }

    fn plan_transition(
        &self,
        next_target_keys: Vec<String>,
        merged: BTreeMap<String, Tensor>,
        mut stats: LoraComponentApplyStats,
    ) -> Result<ComponentSwapPlan> {
        let affected = self
            .active_target_keys
            .iter()
            .chain(next_target_keys.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        let next = next_target_keys.iter().cloned().collect::<BTreeSet<_>>();
        stats.restored_target_keys = self
            .active_target_keys
            .iter()
            .filter(|target| !next.contains(*target))
            .cloned()
            .collect();

        let current = self.current_tensors()?;
        let mut updates = Vec::new();
        updates
            .try_reserve_exact(affected.len())
            .map_err(|_| candle::Error::msg("allocating SDXL LoRA component update plan"))?;
        for target in affected {
            let previous = current
                .get(&target)
                .ok_or_else(|| {
                    candle::Error::msg(format!(
                        "SDXL LoRA {} target disappeared while planning: {target}",
                        self.component.as_str()
                    ))
                })?
                .copy()?;
            let next = if next.contains(&target) {
                merged.get(&target).ok_or_else(|| {
                    candle::Error::msg(format!(
                        "merged SDXL LoRA {} target is unavailable: {target}",
                        self.component.as_str()
                    ))
                })?
            } else {
                self.base_tensors.get(&target).ok_or_else(|| {
                    candle::Error::msg(format!(
                        "immutable SDXL LoRA {} base is unavailable: {target}",
                        self.component.as_str()
                    ))
                })?
            };
            updates.push(TargetSwap {
                target,
                previous,
                next: next.clone(),
            });
        }
        Ok(ComponentSwapPlan {
            component: self.component,
            next_target_keys,
            stats,
            updates,
        })
    }

    fn current_tensors(&self) -> Result<BTreeMap<String, Tensor>> {
        let variables = self.varmap.data().lock().map_err(|_| {
            candle::Error::msg(format!(
                "SDXL LoRA {} VarMap lock is poisoned",
                self.component.as_str()
            ))
        })?;
        Ok(variables
            .iter()
            .map(|(name, value)| (name.clone(), value.as_tensor().clone()))
            .collect())
    }

    fn verify_snapshots(&self, plan: &ComponentSwapPlan) -> Result<()> {
        let current = self.current_tensors()?;
        for update in &plan.updates {
            let tensor = current.get(&update.target).ok_or_else(|| {
                candle::Error::msg(format!(
                    "SDXL LoRA {} target disappeared before apply: {}",
                    self.component.as_str(),
                    update.target
                ))
            })?;
            if !tensor_exactly_matches(tensor, &update.previous)? {
                candle::bail!(
                    "SDXL LoRA {} plan is stale because target changed: {}",
                    self.component.as_str(),
                    update.target
                )
            }
        }
        Ok(())
    }
}

/// Opaque, revision-bound three-component adapter replacement.
#[derive(Debug)]
pub struct LoraApplyPlan {
    revision: u64,
    components: [ComponentSwapPlan; 3],
    stats: LoraApplyStats,
}

impl LoraApplyPlan {
    pub fn stats(&self) -> &LoraApplyStats {
        &self.stats
    }
}

/// Atomic SDXL LoRA replacement over UNet and both text encoders.
///
/// Plans are computed from independent base copies. Applying adapter B after A
/// therefore computes `base + B`, not `base + A + B`. All targets are
/// snapshotted and revalidated before the first write; later failures roll
/// every component back in reverse order. The caller must hold its model's
/// exclusive execution/mutation lease while preparing and applying a plan;
/// this transaction cannot serialize inference performed outside its VarMaps.
pub struct VarMapSwapTransaction {
    components: [MutableModelComponent; 3],
    revision: u64,
}

impl VarMapSwapTransaction {
    pub fn new(unet: VarMap, text_encoder_1: VarMap, text_encoder_2: VarMap) -> Self {
        Self {
            components: [
                MutableModelComponent::new(SdxlLoraComponent::Unet, unet),
                MutableModelComponent::new(SdxlLoraComponent::TextEncoder1, text_encoder_1),
                MutableModelComponent::new(SdxlLoraComponent::TextEncoder2, text_encoder_2),
            ],
            revision: 0,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn component(&self, component: SdxlLoraComponent) -> &MutableModelComponent {
        &self.components[component.index()]
    }

    pub fn prepare_adapter<'a>(
        &mut self,
        lora_tensors: impl IntoIterator<Item = (&'a String, &'a Tensor)>,
        strength: f64,
        resolver: &impl SdxlLoraTargetResolver,
    ) -> Result<LoraApplyPlan> {
        if !strength.is_finite() {
            candle::bail!("SDXL LoRA strength must be finite, found {strength}")
        }
        let pairs = parse_sdxl_lora_pairs(lora_tensors)?;
        let by_component = |component| {
            pairs
                .iter()
                .filter(|pair| pair.component() == component)
                .collect::<Vec<_>>()
        };
        let unet_pairs = by_component(SdxlLoraComponent::Unet);
        let text_1_pairs = by_component(SdxlLoraComponent::TextEncoder1);
        let text_2_pairs = by_component(SdxlLoraComponent::TextEncoder2);
        let unet = self.components[0].prepare_adapter(&unet_pairs, strength, resolver)?;
        let text_encoder_1 =
            self.components[1].prepare_adapter(&text_1_pairs, strength, resolver)?;
        let text_encoder_2 =
            self.components[2].prepare_adapter(&text_2_pairs, strength, resolver)?;
        let stats = LoraApplyStats {
            components: [
                unet.stats.clone(),
                text_encoder_1.stats.clone(),
                text_encoder_2.stats.clone(),
            ],
        };
        if stats.applied_count() == 0 {
            candle::bail!("SDXL LoRA adapter has no nonzero effective target deltas")
        }
        Ok(LoraApplyPlan {
            revision: self.revision,
            components: [unet, text_encoder_1, text_encoder_2],
            stats,
        })
    }

    pub fn apply_plan(&mut self, plan: LoraApplyPlan) -> Result<LoraApplyStats> {
        self.apply_plan_with_failure(plan, None)
    }

    pub fn swap_adapter<'a>(
        &mut self,
        lora_tensors: impl IntoIterator<Item = (&'a String, &'a Tensor)>,
        strength: f64,
        resolver: &impl SdxlLoraTargetResolver,
    ) -> Result<LoraApplyStats> {
        let plan = self.prepare_adapter(lora_tensors, strength, resolver)?;
        self.apply_plan(plan)
    }

    pub fn clear(&mut self) -> Result<LoraApplyStats> {
        if self
            .components
            .iter()
            .all(|component| component.active_target_keys.is_empty())
        {
            return Ok(LoraApplyStats::default());
        }
        let unet = self.components[0].prepare_clear()?;
        let text_encoder_1 = self.components[1].prepare_clear()?;
        let text_encoder_2 = self.components[2].prepare_clear()?;
        let stats = LoraApplyStats {
            components: [
                unet.stats.clone(),
                text_encoder_1.stats.clone(),
                text_encoder_2.stats.clone(),
            ],
        };
        self.apply_plan(LoraApplyPlan {
            revision: self.revision,
            components: [unet, text_encoder_1, text_encoder_2],
            stats,
        })
    }

    fn apply_plan_with_failure(
        &mut self,
        plan: LoraApplyPlan,
        fail_after_writes: Option<usize>,
    ) -> Result<LoraApplyStats> {
        if plan.revision != self.revision {
            candle::bail!(
                "stale SDXL LoRA plan revision {}; current revision is {}",
                plan.revision,
                self.revision
            )
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| candle::Error::msg("SDXL LoRA transaction revision overflow"))?;
        for component_plan in &plan.components {
            self.components[component_plan.component.index()].verify_snapshots(component_plan)?;
        }

        let mut write_index = 0usize;
        let mut apply_error = None;
        'apply: for component_plan in &plan.components {
            let component = &mut self.components[component_plan.component.index()];
            for update in &component_plan.updates {
                let result = if fail_after_writes.is_some_and(|limit| write_index >= limit) {
                    component
                        .varmap
                        .set_one("__candle_injected_missing_lora_target__", &update.next)
                } else {
                    component.varmap.set_one(&update.target, &update.next)
                };
                if let Err(error) = result {
                    apply_error = Some(error);
                    break 'apply;
                }
                write_index = write_index.checked_add(1).ok_or_else(|| {
                    candle::Error::msg("SDXL LoRA applied-write counter overflow")
                })?;
            }
        }

        if let Some(error) = apply_error {
            let apply_message = error.to_string();
            for component_plan in plan.components.iter().rev() {
                let component = &mut self.components[component_plan.component.index()];
                for update in component_plan.updates.iter().rev() {
                    if let Err(rollback_error) =
                        component.varmap.set_one(&update.target, &update.previous)
                    {
                        candle::bail!(
                            "SDXL LoRA apply failed: {apply_message}; rollback also failed at {}:{}: {rollback_error}",
                            component_plan.component.as_str(),
                            update.target
                        )
                    }
                }
            }
            candle::bail!("SDXL LoRA apply failed and was rolled back: {apply_message}")
        }

        for component_plan in &plan.components {
            self.components[component_plan.component.index()].active_target_keys =
                component_plan.next_target_keys.clone();
        }
        self.revision = next_revision;
        Ok(plan.stats)
    }
}

#[derive(Debug)]
struct ComponentSwapPlan {
    component: SdxlLoraComponent,
    next_target_keys: Vec<String>,
    stats: LoraComponentApplyStats,
    updates: Vec<TargetSwap>,
}

#[derive(Debug)]
struct TargetSwap {
    target: String,
    previous: Tensor,
    next: Tensor,
}

fn tensor_exactly_matches(left: &Tensor, right: &Tensor) -> Result<bool> {
    if left.dims() != right.dims()
        || left.dtype() != right.dtype()
        || !left.device().same_device(right.device())
    {
        return Ok(false);
    }
    Ok(left
        .eq(right)?
        .flatten_all()?
        .to_vec1::<u8>()?
        .into_iter()
        .all(|value| value == 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::Init;

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
        let (unet, observed_unet) = component_varmap(UNET_TARGET, [10., 20., 30., 40.]);
        let (text_1, observed_text_1) = component_varmap(TEXT_1_TARGET, [50., 60., 70., 80.]);
        let (text_2, observed_text_2) = component_varmap(TEXT_2_TARGET, [90., 100., 110., 120.]);
        (
            VarMapSwapTransaction::new(unet, text_1, text_2),
            [observed_unet, observed_text_1, observed_text_2],
        )
    }

    fn pair(stem: &str, value: f32) -> [(String, Tensor); 2] {
        let device = Device::Cpu;
        [
            (
                format!("{stem}.lora_down.weight"),
                Tensor::from_vec(vec![value, value], (1, 2), &device).unwrap(),
            ),
            (
                format!("{stem}.lora_up.weight"),
                Tensor::from_vec(vec![1., 1.], (2, 1), &device).unwrap(),
            ),
        ]
    }

    fn adapter(value: f32, components: &[SdxlLoraComponent]) -> BTreeMap<String, Tensor> {
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
            tensors.extend(pair(stem, value));
        }
        tensors
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

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn unet_only_and_text_only_adapters_touch_only_their_components() {
        for component in [
            SdxlLoraComponent::Unet,
            SdxlLoraComponent::TextEncoder1,
            SdxlLoraComponent::TextEncoder2,
        ] {
            let (mut transaction, observed) = test_transaction();
            let before = observed.each_ref().map(values);
            let tensors = adapter(1., &[component]);
            let stats = transaction.swap_adapter(&tensors, 1.0, &resolver).unwrap();
            assert_eq!(stats.applied_count(), 1);
            for (index, tensor) in observed.iter().enumerate() {
                assert_eq!(values(tensor) != before[index], index == component.index());
            }
        }
    }

    #[test]
    fn mixed_adapter_replaces_all_components_and_clear_restores_exact_base() {
        let (mut transaction, observed) = test_transaction();
        let base = observed.each_ref().map(values);
        let tensors = adapter(2., &SdxlLoraComponent::ALL);
        let stats = transaction.swap_adapter(&tensors, 0.5, &resolver).unwrap();
        assert_eq!(stats.applied_count(), 3);
        assert!(stats
            .components
            .iter()
            .all(|component| component.targets[0].delta_sha256.len() == 64));
        assert_eq!(transaction.revision(), 1);
        for (index, tensor) in observed.iter().enumerate() {
            assert_ne!(values(tensor), base[index]);
        }

        let clear = transaction.clear().unwrap();
        assert_eq!(clear.applied_count(), 0);
        assert!(clear
            .components
            .iter()
            .all(|component| component.restored_target_keys.len() == 1));
        for (index, tensor) in observed.iter().enumerate() {
            assert_eq!(values(tensor), base[index]);
        }
    }

    #[test]
    fn adapter_b_is_computed_from_base_not_adapter_a() {
        let (mut transaction, observed) = test_transaction();
        let base = values(&observed[0]);
        let a = adapter(1., &[SdxlLoraComponent::Unet]);
        let b = adapter(3., &[SdxlLoraComponent::Unet]);
        transaction.swap_adapter(&a, 1.0, &resolver).unwrap();
        let adapter_a = values(&observed[0]);
        transaction.swap_adapter(&b, 1.0, &resolver).unwrap();
        let adapter_b = values(&observed[0]);
        assert_ne!(adapter_a, adapter_b);
        assert_eq!(
            adapter_b,
            base.into_iter()
                .map(|value| value + 3.0)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn failures_in_component_two_or_three_roll_back_every_write() {
        for fail_after in [1, 2] {
            let (mut transaction, observed) = test_transaction();
            let a = adapter(1., &SdxlLoraComponent::ALL);
            transaction.swap_adapter(&a, 1.0, &resolver).unwrap();
            let before = observed.each_ref().map(values);
            let b = adapter(2., &SdxlLoraComponent::ALL);
            let plan = transaction.prepare_adapter(&b, 1.0, &resolver).unwrap();
            let error = transaction
                .apply_plan_with_failure(plan, Some(fail_after))
                .unwrap_err();
            assert!(error.to_string().contains("was rolled back"));
            for (index, tensor) in observed.iter().enumerate() {
                assert_eq!(values(tensor), before[index]);
            }
            assert_eq!(transaction.revision(), 1);
        }
    }

    #[test]
    fn invalid_later_component_never_mutates_an_earlier_component() {
        let (mut transaction, observed) = test_transaction();
        let before = observed.each_ref().map(values);
        let mut tensors = adapter(1., &SdxlLoraComponent::ALL);
        tensors.insert(
            "lora_te2_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight".to_owned(),
            Tensor::zeros((3, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(transaction
            .prepare_adapter(&tensors, 1.0, &resolver)
            .unwrap_err()
            .to_string()
            .contains("delta shape"));
        for (index, tensor) in observed.iter().enumerate() {
            assert_eq!(values(tensor), before[index]);
        }
    }

    #[test]
    fn zero_effect_nonfinite_strength_and_duplicate_targets_fail_closed() {
        let (mut transaction, observed) = test_transaction();
        let before = values(&observed[0]);
        let mut zero = adapter(1., &[SdxlLoraComponent::Unet]);
        zero.insert(
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight".to_owned(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(transaction
            .prepare_adapter(&zero, 1.0, &resolver)
            .unwrap_err()
            .to_string()
            .contains("no nonzero effective"));
        let valid = adapter(1., &[SdxlLoraComponent::Unet]);
        assert!(transaction
            .prepare_adapter(&valid, f64::NAN, &resolver)
            .unwrap_err()
            .to_string()
            .contains("strength must be finite"));

        let mut duplicate = valid;
        duplicate.extend(pair("lora_unet_second_name", 1.));
        assert!(transaction
            .prepare_adapter(&duplicate, 1.0, &resolver)
            .unwrap_err()
            .to_string()
            .contains("multiple SDXL LoRA pairs"));
        assert_eq!(values(&observed[0]), before);
    }

    #[test]
    fn rank_four_bf16_one_by_one_target_is_supported() {
        let device = Device::Cpu;
        let mut unet = VarMap::new();
        let observed = unet
            .get(
                (2, 2, 1, 1),
                UNET_TARGET,
                Init::Const(0.0),
                DType::BF16,
                &device,
            )
            .unwrap();
        let base = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (2, 2, 1, 1), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        unet.set_one(UNET_TARGET, &base).unwrap();
        let (text_1, _) = component_varmap(TEXT_1_TARGET, [1., 2., 3., 4.]);
        let (text_2, _) = component_varmap(TEXT_2_TARGET, [1., 2., 3., 4.]);
        let mut transaction = VarMapSwapTransaction::new(unet, text_1, text_2);
        let tensors = adapter(1., &[SdxlLoraComponent::Unet])
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_dtype(DType::BF16).unwrap()))
            .collect::<BTreeMap<_, _>>();
        transaction.swap_adapter(&tensors, 1.0, &resolver).unwrap();
        assert_ne!(values(&observed), values(&base));
    }

    #[test]
    fn revision_and_live_snapshot_changes_reject_stale_plans() {
        let (mut transaction, _observed) = test_transaction();
        let a = adapter(1., &[SdxlLoraComponent::Unet]);
        let b = adapter(2., &[SdxlLoraComponent::Unet]);
        let stale = transaction.prepare_adapter(&b, 1.0, &resolver).unwrap();
        transaction.swap_adapter(&a, 1.0, &resolver).unwrap();
        assert!(transaction
            .apply_plan(stale)
            .unwrap_err()
            .to_string()
            .contains("stale SDXL LoRA plan revision"));

        let plan = transaction.prepare_adapter(&b, 1.0, &resolver).unwrap();
        let changed = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        transaction.components[0]
            .varmap
            .set_one(UNET_TARGET, &changed)
            .unwrap();
        assert!(transaction
            .apply_plan(plan)
            .unwrap_err()
            .to_string()
            .contains("plan is stale because target changed"));
    }

    #[test]
    fn revision_overflow_fails_before_any_tensor_mutation() {
        let (mut transaction, observed) = test_transaction();
        let before = observed.each_ref().map(values);
        let adapter = adapter(1., &SdxlLoraComponent::ALL);
        transaction.revision = u64::MAX;
        let plan = transaction
            .prepare_adapter(&adapter, 1.0, &resolver)
            .unwrap();

        let error = transaction.apply_plan(plan).unwrap_err();
        assert!(error.to_string().contains("revision overflow"));
        for (index, tensor) in observed.iter().enumerate() {
            assert_eq!(values(tensor), before[index]);
        }
        assert_eq!(transaction.revision(), u64::MAX);
    }
}
