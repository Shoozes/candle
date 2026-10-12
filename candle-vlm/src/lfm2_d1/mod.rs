//! Local, non-generative LFM d1 decisions using independent question prefills.

mod backend;
mod image_cap;
mod loading;
mod profiling;
mod prompt;
mod render_state;
mod session;
mod types;

pub use crate::lfm2_vl::{Lfm2EmbeddingLoadMode, Lfm2EmbeddingStorage};
pub use loading::{
    load_d1_q8, load_d1_q8_with_embedding_mode, load_d1_text_q8,
    load_d1_text_q8_with_embedding_mode, D1LoadOptions, D1TextLoadOptions,
};
pub use profiling::{D1QuestionTiming, D1Timings};
pub use prompt::{D1PolicyV1, PreparedQuestion};
pub use session::D1Session;
pub use session::D1TraceEvent;
pub use types::{
    Answer, D1Limits, DecisionFailure, DecisionRequest, DecisionResponse, DecisionUsage,
    ExecutionReport, OrderedMap, Question, State, YesNoCriteria,
};

#[cfg(test)]
mod tests;
