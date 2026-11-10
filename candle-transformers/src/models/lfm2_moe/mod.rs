//! LFM2 mixture-of-experts inference, including LiquidAI LFM2.5-24B-A2B.
//!
//! Numerical authority: Transformers 5.12.1, ddb849abe009d1089e6c691bfc897f27211c663c.

mod config;
mod model;
pub(crate) mod routing;

pub use config::{Config, RopeParameters};
pub use model::{Cache, Model};

#[cfg(test)]
mod tests;
