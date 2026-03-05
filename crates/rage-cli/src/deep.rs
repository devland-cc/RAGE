use std::collections::HashMap;

use anyhow::Result;
use rage_classifier::Classifier;
use rage_core::types::MoodPrediction;
use rage_core::{
    AudioBuffer, BeatEntry, DeepAnalysis, DeepSummary, EmotionSegment, ExtractionConfig,
    MusicalKey,
};
use rage_extractor::{chroma, key, onset, stft, tempo};

/// Run deep analysis on a loaded audio buffer.
///
/// Returns a `DeepAnalysis` containing per-beat BPM/key data,
/// segmented emotion analysis, and summary statistics.
pub fn run_deep_analysis(
    audio: &AudioBuffer,
    classifier: &mut Classifier,
    config: &ExtractionConfig,
    source: &str,
    segment_secs: f32,
    top_k: usize,
) -> Result<DeepAnalysis> {
    let sr = config.sample_rate;
    let hop = config.hop_length;
    let fps = sr as f32 / hop as f32;
    let duration_secs = audio.duration_secs();

    // --- Full-length STFT and chroma for BPM/key detection ---
    let stft_mag = stft::stft_magnitude(&audio.samples, config.n_fft, hop);
    let chroma_full = chroma::chroma(&stft_mag, sr, config.n_fft);

    // --- Onset detection and beat tracking ---
    let onset_env = onset::onset_strength(&stft_mag);
    let estimated_bpm = tempo::estimate_tempo(&onset_env, fps);
    let beat_frames = tempo::track_beats(&onset_env, estimated_bpm, fps);
    let local_bpms = tempo::local_bpm(&beat_frames, fps);

    // --- Per-beat key detection ---
    let beats: Vec<BeatEntry> = beat_frames
        .iter()
        .enumerate()
        .map(|(i, &frame)| {
            let time_secs = frame as f32 / fps;
            let bpm = local_bpms[i];

            // Window = half a beat period in frames
            let half_window = if bpm > 0.0 {
                ((fps * 60.0 / bpm) / 2.0).round() as usize
            } else {
                (fps * 0.5).round() as usize // fallback: 0.5s
            };

            let mean_chroma = key::mean_chroma_window(&chroma_full, frame, half_window);
            let detected_key = key::detect_key(&mean_chroma);

            BeatEntry {
                time_secs,
                bpm,
                key: detected_key,
            }
        })
        .collect();

    // --- Segmented emotion analysis ---
    let segment_samples = (segment_secs * sr as f32) as usize;
    let min_segment_samples = (5.0 * sr as f32) as usize; // 5 seconds minimum
    let total_samples = audio.samples.len();

    let mut segments: Vec<EmotionSegment> = Vec::new();
    let mut offset = 0usize;

    while offset < total_samples {
        let mut end = (offset + segment_samples).min(total_samples);

        // If the remaining chunk would be too short, merge with current
        if end < total_samples && (total_samples - end) < min_segment_samples {
            end = total_samples;
        }

        let window = &audio.samples[offset..end];
        let time_secs = offset as f32 / sr as f32;

        let features = rage_extractor::extract_features_window(window, config)?;
        let emotion = classifier.classify(&features, source)?;

        let (valence, arousal) = emotion
            .valence_arousal
            .map(|va| (va.valence, va.arousal))
            .unwrap_or((0.0, 0.0));

        let mood_tags: Vec<MoodPrediction> = emotion
            .mood_tags
            .into_iter()
            .take(top_k)
            .collect();

        segments.push(EmotionSegment {
            time_secs,
            valence,
            arousal,
            mood_tags,
        });

        offset = end;
    }

    // --- Compute summary statistics ---
    let summary = compute_summary(&beats, &segments, top_k);

    Ok(DeepAnalysis {
        source: source.to_string(),
        duration_secs,
        summary,
        beats,
        segments,
    })
}

/// Compute summary statistics from beats and emotion segments.
fn compute_summary(
    beats: &[BeatEntry],
    segments: &[EmotionSegment],
    top_k: usize,
) -> DeepSummary {
    // Dominant BPM: mode of rounded BPM values
    let dominant_bpm = if beats.is_empty() {
        0.0
    } else {
        let mut bpm_counts: HashMap<i32, usize> = HashMap::new();
        for beat in beats {
            if beat.bpm > 0.0 {
                let rounded = beat.bpm.round() as i32;
                *bpm_counts.entry(rounded).or_insert(0) += 1;
            }
        }
        bpm_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(bpm, _)| bpm as f32)
            .unwrap_or(0.0)
    };

    // Dominant key: most frequent key across beats
    let dominant_key = if beats.is_empty() {
        MusicalKey::new(0, rage_core::KeyMode::Major)
    } else {
        let mut key_counts: HashMap<(u8, rage_core::KeyMode), usize> = HashMap::new();
        for beat in beats {
            *key_counts
                .entry((beat.key.pitch_class, beat.key.mode))
                .or_insert(0) += 1;
        }
        let ((pc, mode), _) = key_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap();
        MusicalKey::new(pc, mode)
    };

    // Average valence/arousal
    let (avg_valence, avg_arousal) = if segments.is_empty() {
        (0.0, 0.0)
    } else {
        let n = segments.len() as f32;
        let v: f32 = segments.iter().map(|s| s.valence).sum::<f32>() / n;
        let a: f32 = segments.iter().map(|s| s.arousal).sum::<f32>() / n;
        (v, a)
    };

    // Top moods: count appearances across segments (prob > 0.05),
    // average probabilities, rank by count then avg probability
    let mut mood_stats: HashMap<rage_core::MoodTag, (usize, f32)> = HashMap::new();
    for seg in segments {
        for mood in &seg.mood_tags {
            if mood.probability > 0.05 {
                let entry = mood_stats.entry(mood.tag).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += mood.probability;
            }
        }
    }

    let mut top_moods: Vec<MoodPrediction> = mood_stats
        .into_iter()
        .map(|(tag, (count, total_prob))| {
            let avg_prob = total_prob / count as f32;
            MoodPrediction {
                tag,
                probability: avg_prob,
            }
        })
        .collect();

    top_moods.sort_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_moods.truncate(top_k);

    DeepSummary {
        dominant_bpm,
        dominant_key,
        avg_valence,
        avg_arousal,
        top_moods,
    }
}
