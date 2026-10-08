use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::HashSet, fmt, marker::PhantomData};

/// An insertion-ordered JSON object. Duplicate keys are rejected on input.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedMap<T>(pub Vec<(String, T)>);

impl<T> Default for OrderedMap<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: Serialize> Serialize for OrderedMap<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for OrderedMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MapVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for MapVisitor<T> {
            type Value = OrderedMap<T>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an ordered object with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut result = Vec::new();
                let mut names = HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if result.len() >= 4096 || !names.insert(key.clone()) {
                        return Err(serde::de::Error::custom(
                            "d1 object exceeds its limit or repeats a key",
                        ));
                    }
                    result.push((key, map.next_value()?));
                }
                Ok(OrderedMap(result))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

/// Text, image-only null, or structured JSON whose object order is retained.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    Text(String),
    Array(Vec<State>),
    Object(OrderedMap<State>),
}

impl Serialize for State {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::Text(value) => serializer.serialize_str(value),
            Self::Object(value) => value.serialize(serializer),
            Self::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(value)?;
                }
                seq.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for State {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StateVisitor;
        impl<'de> Visitor<'de> for StateVisitor {
            type Value = State;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON state")
            }
            fn visit_unit<E>(self) -> Result<State, E> {
                Ok(State::Null)
            }
            fn visit_bool<E>(self, value: bool) -> Result<State, E> {
                Ok(State::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<State, E> {
                Ok(State::Number(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<State, E> {
                Ok(State::Number(value.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<State, E> {
                serde_json::Number::from_f64(value)
                    .map(State::Number)
                    .ok_or_else(|| E::custom("nonfinite d1 state number"))
            }
            fn visit_str<E>(self, value: &str) -> Result<State, E> {
                Ok(State::Text(value.into()))
            }
            fn visit_string<E>(self, value: String) -> Result<State, E> {
                Ok(State::Text(value))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<State, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    if values.len() >= 4096 {
                        return Err(serde::de::Error::custom("d1 state array is too large"));
                    }
                    values.push(value);
                }
                Ok(State::Array(values))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<State, A::Error> {
                let mut values = Vec::new();
                let mut names = HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.len() >= 4096 || !names.insert(key.clone()) {
                        return Err(serde::de::Error::custom(
                            "d1 state object is too large or repeats a key",
                        ));
                    }
                    values.push((key, map.next_value()?));
                }
                Ok(State::Object(OrderedMap(values)))
            }
        }
        deserializer.deserialize_any(StateVisitor)
    }
}

pub type YesNoCriteria = OrderedMap<Option<String>>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<YesNoCriteria>,
    },
    Choice {
        instructions: String,
        criteria: OrderedMap<Option<String>>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub state: State,
    pub questions: OrderedMap<Question>,
}

impl DecisionRequest {
    pub fn from_json(bytes: &[u8], max_bytes: usize) -> candle::Result<Self> {
        if bytes.is_empty() || bytes.len() > max_bytes {
            candle::bail!("d1 request JSON exceeds its byte limit")
        }
        serde_json::from_slice(bytes)
            .map_err(|error| candle::Error::Msg(format!("invalid d1 request JSON: {error}")))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: OrderedMap<f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        probabilities: OrderedMap<f64>,
        legend: OrderedMap<String>,
    },
}

#[derive(Clone, Debug)]
pub struct D1Limits {
    pub max_questions: usize,
    pub max_options: usize,
    pub max_prompt_bytes: usize,
    pub max_context_tokens: Option<usize>,
    pub max_images: usize,
}

impl Default for D1Limits {
    fn default() -> Self {
        Self {
            max_questions: 64,
            max_options: 256,
            max_prompt_bytes: 256 * 1024,
            max_context_tokens: None,
            max_images: 8,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DecisionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub language_forwards: u64,
    pub vision_forwards: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecutionReport {
    pub policy: String,
    pub text_device: String,
    pub vision_device: String,
    pub mmproj_quantized_tensors: usize,
    pub text_quantized_linears: usize,
    pub source_float_mmproj_linears: Vec<String>,
    pub cpu_q8_matmuls: u64,
    pub cuda_mmvq: u64,
    pub cuda_mmq: u64,
    pub cuda_other_q8: u64,
    pub cuda_f32_q8: u64,
}

impl ExecutionReport {
    pub(crate) fn accumulate(
        &mut self,
        value: candle::quantized::NativeQ8Execution,
    ) -> candle::Result<()> {
        for (target, count) in [
            (&mut self.cpu_q8_matmuls, value.cpu_matmuls),
            (&mut self.cuda_mmvq, value.cuda_mmvq),
            (&mut self.cuda_mmq, value.cuda_mmq),
            (&mut self.cuda_other_q8, value.cuda_other_q8),
            (&mut self.cuda_f32_q8, value.cuda_f32_q8),
        ] {
            *target = target
                .checked_add(count)
                .ok_or_else(|| candle::Error::Msg("d1 dispatch counter overflow".into()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DecisionResponse {
    pub answers: OrderedMap<Answer>,
    pub usage: DecisionUsage,
    pub execution: ExecutionReport,
    pub planned_questions: usize,
    pub unattempted_questions: usize,
}

#[derive(Debug)]
pub struct DecisionFailure {
    pub message: String,
    pub question_index: Option<usize>,
    pub partial: Box<DecisionResponse>,
}

impl fmt::Display for DecisionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DecisionFailure {}
