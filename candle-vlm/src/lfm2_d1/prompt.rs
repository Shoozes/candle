use super::{Answer, D1Limits, OrderedMap, Question, State};
use candle::{Result, Tensor};
use candle_transformers::models::lfm2_d1::option_probabilities;
use std::collections::HashSet;
use tokenizers::Tokenizer;

/// Identity-calibration, json_only, no-system, no-thinking reference policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct D1PolicyV1;

#[derive(Clone, Debug)]
pub struct PreparedQuestion {
    question_text: String,
    token_groups: Vec<Vec<u32>>,
    question: Question,
}

impl D1PolicyV1 {
    pub const ID: &'static str = "lfm-d1-systemone-v1";
    pub const REFERENCE_REVISION: &'static str = "051bcc464b01b9f92942b364d9586b0ef5912432";

    pub fn prepare(
        self,
        tokenizer: &Tokenizer,
        question: &Question,
        limits: &D1Limits,
    ) -> Result<PreparedQuestion> {
        validate_question_payload(question, limits)?;
        let (instructions, text, groups) = match question {
            Question::Noul {
                instructions,
                criteria,
            } => {
                if limits.max_options < 2 {
                    candle::bail!("d1 noul requires two allowed options")
                }
                let yes = single_ids(tokenizer, &["yes", "Yes", "YES"])?;
                let no = single_ids(tokenizer, &["no", "No", "NO"])?;
                if yes.is_empty() || no.is_empty() {
                    candle::bail!("d1 tokenizer has no single-token yes/no forms")
                }
                let extra = criteria
                    .as_ref()
                    .filter(|value| !value.0.is_empty())
                    .map(|value| {
                        let description = |key: &str| {
                            value
                                .0
                                .iter()
                                .find(|(name, _)| name == key)
                                .and_then(|(_, value)| value.as_deref())
                                .unwrap_or("None")
                        };
                        format!(
                            "\nYes: {}\nNo: {}",
                            description("true"),
                            description("false")
                        )
                    })
                    .unwrap_or_default();
                (
                    instructions,
                    format!("{instructions}{extra}\n\nReply with yes or no only."),
                    vec![yes, no],
                )
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                if criteria.0.is_empty() || criteria.0.len() > limits.max_options {
                    candle::bail!("d1 choice count exceeds its configured range")
                }
                let mut names = HashSet::new();
                for (name, _) in &criteria.0 {
                    if name.trim().is_empty() || !names.insert(name) {
                        candle::bail!("d1 choice labels must be nonempty and unique")
                    }
                }
                let labels: Vec<_> = criteria
                    .0
                    .iter()
                    .map(|(name, _)| {
                        name.trim_matches(|ch: char| {
                            ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
                        })
                    })
                    .collect();
                // Python isalpha selects Unicode Letter, not Other_Alphabetic marks.
                let letter = tokenizers::utils::SysRegex::new(r"^\p{L}$")
                    .map_err(|error| candle::Error::Msg(error.to_string()))?;
                let letters = labels
                    .iter()
                    .all(|label| letter.find_iter(label).next().is_some());
                let codes: Vec<String> = if letters {
                    labels.iter().map(|label| (*label).into()).collect()
                } else if labels.len() <= 26 {
                    (0..labels.len())
                        .map(|i| char::from(b'A' + i as u8).to_string())
                        .collect()
                } else {
                    (0..labels.len()).map(|i| format!("{i:02}")).collect()
                };
                let pool = fallback_pool();
                let mut used = HashSet::new();
                let mut lines = Vec::new();
                let mut groups = Vec::new();
                for ((name, description), code) in criteria.0.iter().zip(codes) {
                    let mut selected = take(tokenizer, &code, &mut used)?;
                    if selected.is_none() {
                        for fallback in &pool {
                            selected = take(tokenizer, fallback, &mut used)?;
                            if selected.is_some() {
                                break;
                            }
                        }
                    }
                    let (code, id) = selected.ok_or_else(|| {
                        candle::Error::Msg(
                            "d1 tokenizer exhausted distinct single-token option aliases".into(),
                        )
                    })?;
                    let mut group = vec![id];
                    for extra in single_ids(tokenizer, &[&format!(" {code}")])? {
                        if extra != id {
                            group.push(extra);
                        }
                    }
                    let description = description
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .unwrap_or_else(|| name.replace('_', " "));
                    lines.push(format!("{code} {description}"));
                    groups.push(group);
                }
                (
                    instructions,
                    format!(
                        "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                        lines.join("\n")
                    ),
                    groups,
                )
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                if !(2..=10).contains(&criteria.len()) || criteria.len() > limits.max_options {
                    candle::bail!("d1 scores require 2..=10 ordered levels")
                }
                let mut groups = Vec::new();
                let mut legend = Vec::new();
                for (index, label) in criteria.iter().enumerate() {
                    let ids = single_ids(tokenizer, &[&index.to_string()])?;
                    if ids.is_empty() {
                        candle::bail!("d1 score level {index} has no single-token digit")
                    }
                    groups.push(ids);
                    legend.push(format!("{index} {label}"));
                }
                (
                    instructions,
                    format!(
                        "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                        legend.join("\n"),
                        criteria.len() - 1
                    ),
                    groups,
                )
            }
        };
        if instructions.trim().is_empty() || text.len() > limits.max_prompt_bytes {
            candle::bail!("d1 question is empty or exceeds its prompt byte limit")
        }
        Ok(PreparedQuestion {
            question_text: text,
            token_groups: groups,
            question: question.clone(),
        })
    }

    pub fn render(
        self,
        tokenizer: &Tokenizer,
        state: &State,
        question: &PreparedQuestion,
        images: usize,
        max_bytes: usize,
    ) -> Result<String> {
        validate_state(state, max_bytes)?;
        let bos = if tokenizer.token_to_id("<|startoftext|>").is_some() {
            "<|startoftext|>"
        } else {
            ""
        };
        let mut text = super::render_state::BoundedText::new(max_bytes);
        text.push(bos)?;
        text.push("<|im_start|>user\n")?;
        for _ in 0..images {
            text.push("<image>")?;
        }
        match state {
            State::Null => (),
            State::Text(value) => {
                text.push(value)?;
                text.push("\n\n\nQUESTION:\n")?;
            }
            value => {
                super::render_state::render(value, &mut text, 0)?;
                text.push("\n\n\nQUESTION:\n")?;
            }
        }
        text.push(&question.question_text)?;
        text.push("<|im_end|>\n<|im_start|>assistant\n")?;
        text.finish()
    }
}

fn validate_question_payload(question: &Question, limits: &D1Limits) -> Result<()> {
    let instructions = match question {
        Question::Noul { instructions, .. }
        | Question::Choice { instructions, .. }
        | Question::Score { instructions, .. } => instructions,
    };
    if instructions.trim().is_empty() {
        candle::bail!("d1 question instructions must be nonempty")
    }
    let mut bytes = instructions.len();
    if bytes > limits.max_prompt_bytes {
        candle::bail!("d1 question exceeds its prompt byte limit")
    }
    let mut add = |text: &str| -> Result<()> {
        bytes = bytes
            .checked_add(text.len())
            .ok_or_else(|| candle::Error::Msg("d1 question payload size overflow".into()))?;
        if bytes > limits.max_prompt_bytes {
            candle::bail!("d1 question exceeds its prompt byte limit")
        }
        Ok(())
    };
    match question {
        Question::Noul { criteria, .. } => {
            let mut names = HashSet::new();
            if let Some(criteria) = criteria {
                for (name, description) in &criteria.0 {
                    if name != "true" && name != "false" {
                        candle::bail!("d1 noul criteria only admit true/false descriptions")
                    }
                    if !names.insert(name) {
                        candle::bail!("d1 noul criteria must be unique")
                    }
                    if let Some(description) = description {
                        add(description)?;
                    }
                }
            }
        }
        Question::Choice { criteria, .. } => {
            for (name, description) in &criteria.0 {
                add(description
                    .as_deref()
                    .filter(|value| !value.is_empty())
                    .unwrap_or(name))?;
            }
        }
        Question::Score { criteria, .. } => {
            for label in criteria {
                add(label)?;
            }
        }
    }
    Ok(())
}

impl PreparedQuestion {
    pub fn question_text(&self) -> &str {
        &self.question_text
    }
    pub fn token_groups(&self) -> &[Vec<u32>] {
        &self.token_groups
    }

    pub fn readout(&self, logits: &Tensor) -> Result<Answer> {
        let probabilities = option_probabilities(logits, &self.token_groups)?;
        let mut best = 0usize;
        let mut confidence = *probabilities
            .first()
            .ok_or_else(|| candle::Error::Msg("d1 readout produced no options".into()))?;
        for (index, &value) in probabilities.iter().enumerate().skip(1) {
            if value > confidence {
                best = index;
                confidence = value;
            }
        }
        Ok(match &self.question {
            Question::Noul { .. } => Answer::Noul {
                noul: confidence_for_yes(&probabilities)?,
            },
            Question::Choice { criteria, .. } => {
                let choice = criteria
                    .0
                    .get(best)
                    .ok_or_else(|| {
                        candle::Error::Msg("d1 choice readout cardinality mismatch".into())
                    })?
                    .0
                    .clone();
                Answer::Choice {
                    choice,
                    confidence,
                    probabilities: OrderedMap(
                        criteria
                            .0
                            .iter()
                            .map(|(name, _)| name.clone())
                            .zip(probabilities)
                            .collect(),
                    ),
                }
            }
            Question::Score { criteria, .. } => Answer::Score {
                score: probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum(),
                confidence,
                probabilities: OrderedMap(
                    probabilities
                        .into_iter()
                        .enumerate()
                        .map(|(i, p)| (i.to_string(), p))
                        .collect(),
                ),
                legend: OrderedMap(
                    criteria
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(i, name)| (i.to_string(), name))
                        .collect(),
                ),
            },
        })
    }
}

fn confidence_for_yes(values: &[f64]) -> Result<f64> {
    values
        .first()
        .copied()
        .ok_or_else(|| candle::Error::Msg("d1 noul readout is empty".into()))
}

fn single_ids(tokenizer: &Tokenizer, forms: &[&str]) -> Result<Vec<u32>> {
    let mut result = Vec::new();
    for form in forms {
        let encoded = tokenizer
            .encode(*form, false)
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        if let [id] = encoded.get_ids() {
            if !result.contains(id) {
                result.push(*id);
            }
        }
    }
    Ok(result)
}

fn take(
    tokenizer: &Tokenizer,
    text: &str,
    used: &mut HashSet<u32>,
) -> Result<Option<(String, u32)>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    if let [id] = encoded.get_ids() {
        if used.insert(*id) {
            return Ok(Some((text.into(), *id)));
        }
    }
    Ok(None)
}

fn fallback_pool() -> Vec<String> {
    let mut pool: Vec<_> = ('A'..='Z').map(|c| c.to_string()).collect();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend(('a'..='z').map(|c| c.to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    for a in 'A'..='Z' {
        for b in 'A'..='Z' {
            pool.push(format!("{a}{b}"));
        }
    }
    pool
}

fn validate_state(state: &State, max_bytes: usize) -> Result<()> {
    let mut pending = vec![(state, 0usize)];
    let mut total = 0usize;
    while let Some((value, depth)) = pending.pop() {
        if depth > 64 {
            candle::bail!("d1 state exceeds depth 64")
        }
        let bytes = match value {
            State::Text(s) => s.len(),
            State::Array(values) => {
                pending.extend(values.iter().map(|v| (v, depth + 1)));
                values.len()
            }
            State::Object(values) => {
                for (name, v) in &values.0 {
                    total = total
                        .checked_add(name.len())
                        .ok_or_else(|| candle::Error::Msg("d1 state size overflow".into()))?;
                    pending.push((v, depth + 1));
                }
                values.0.len()
            }
            _ => 8,
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| candle::Error::Msg("d1 state size overflow".into()))?;
        if total > max_bytes {
            candle::bail!("d1 state exceeds its byte limit")
        }
    }
    Ok(())
}
