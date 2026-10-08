//! Local, non-generative LFM d1 decisions using independent question prefills.

mod image_cap;
mod prompt;
mod render_state;
mod session;
mod types;

pub use prompt::{D1PolicyV1, PreparedQuestion};
pub use session::D1TraceEvent;
pub use session::{load_d1_q8, D1LoadOptions, D1Session};
pub use types::{
    Answer, D1Limits, DecisionFailure, DecisionRequest, DecisionResponse, DecisionUsage,
    ExecutionReport, OrderedMap, Question, State, YesNoCriteria,
};

#[cfg(test)]
mod tests;
