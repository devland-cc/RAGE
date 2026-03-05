use std::fmt;

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

// --- Deep analysis types ---

/// Major or minor mode for a musical key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyMode {
    Major,
    Minor,
}

/// A musical key (pitch class + mode).
/// Pitch class: 0=C, 1=C#, 2=D, ..., 11=B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MusicalKey {
    pub pitch_class: u8,
    pub mode: KeyMode,
}

impl MusicalKey {
    pub fn new(pitch_class: u8, mode: KeyMode) -> Self {
        Self {
            pitch_class: pitch_class % 12,
            mode,
        }
    }

    fn pitch_name(&self) -> &'static str {
        match self.pitch_class {
            0 => "C",
            1 => "C#",
            2 => "D",
            3 => "D#",
            4 => "E",
            5 => "F",
            6 => "F#",
            7 => "G",
            8 => "G#",
            9 => "A",
            10 => "A#",
            11 => "B",
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for MusicalKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode_str = match self.mode {
            KeyMode::Major => "major",
            KeyMode::Minor => "minor",
        };
        write!(f, "{} {}", self.pitch_name(), mode_str)
    }
}

/// A single beat with its timestamp, local BPM, and detected key.
#[derive(Debug, Clone)]
pub struct BeatEntry {
    pub time_secs: f32,
    pub bpm: f32,
    pub key: MusicalKey,
}

/// Emotion analysis for a time segment.
#[derive(Debug, Clone)]
pub struct EmotionSegment {
    pub time_secs: f32,
    pub valence: f32,
    pub arousal: f32,
    pub mood_tags: Vec<MoodPrediction>,
}

/// Summary statistics for a deep analysis.
#[derive(Debug, Clone)]
pub struct DeepSummary {
    pub dominant_bpm: f32,
    pub dominant_key: MusicalKey,
    pub avg_valence: f32,
    pub avg_arousal: f32,
    pub top_moods: Vec<MoodPrediction>,
}

/// Complete deep analysis result.
#[derive(Debug, Clone)]
pub struct DeepAnalysis {
    pub source: String,
    pub duration_secs: f32,
    pub summary: DeepSummary,
    pub beats: Vec<BeatEntry>,
    pub segments: Vec<EmotionSegment>,
}
