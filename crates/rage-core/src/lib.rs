pub mod config;
pub mod error;
pub mod tags;
pub mod types;

pub use config::ExtractionConfig;
pub use error::RageError;
pub use tags::MoodTag;
pub use types::{AudioBuffer, EmotionResult, MoodPrediction, ValenceArousal};
