use serde::{Deserialize, Serialize};

use crate::tags::MoodTag;

/// Raw decoded audio samples (mono, f32, normalized to [-1, 1]).
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// PCM samples (mono, f32).
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Valence-arousal prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceArousal {
    /// Valence score in [-1, 1]. Negative = sad/negative, positive = happy/positive.
    pub valence: f32,
    /// Arousal score in [-1, 1]. Negative = calm/low energy, positive = energetic/excited.
    pub arousal: f32,
}

/// A single mood tag prediction with its probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodPrediction {
    pub tag: MoodTag,
    pub probability: f32,
}

/// Complete emotion analysis result for an audio input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionResult {
    /// Source identifier (filename, URL, etc.).
    pub source: String,
    /// Valence-arousal regression result (if V-A model was run).
    pub valence_arousal: Option<ValenceArousal>,
    /// Mood tag predictions sorted by probability descending.
    pub mood_tags: Vec<MoodPrediction>,
}
