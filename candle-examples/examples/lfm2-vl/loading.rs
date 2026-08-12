//! Thin CLI-facing names for the public `candle-vlm` hybrid loader.

pub use candle_vlm::lfm2_vl::{
    load_lfm2_vl_hybrid as load_hybrid, Lfm2VlHybridLoadOptions as HybridLoadOptions,
    Lfm2VlMmprojSource as MmprojInput,
};
