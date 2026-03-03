use ndarray::{Array1, Array2};
use rage_core::{AudioBuffer, ExtractionConfig, RageError};

use crate::{chroma, mel, mfcc, spectral, stft, temporal};

/// Extracted audio features.
#[derive(Debug)]
pub struct FeatureSet {
    /// Log-mel spectrogram in dB, shape [n_mels, n_frames]. CNN input.
    pub log_mel_spectrogram: Array2<f32>,
    /// MFCCs, shape [n_mfcc, n_frames].
    pub mfccs: Array2<f32>,
    /// Chroma features, shape [12, n_frames].
    pub chroma: Array2<f32>,
    /// Spectral centroid per frame, shape [n_frames].
    pub spectral_centroid: Array1<f32>,
    /// Spectral rolloff per frame, shape [n_frames].
    pub spectral_rolloff: Array1<f32>,
    /// Spectral contrast, shape [7, n_frames].
    pub spectral_contrast: Array2<f32>,
    /// RMS energy per frame, shape [n_frames].
    pub rms_energy: Array1<f32>,
    /// Zero crossing rate per frame, shape [n_frames].
    pub zero_crossing_rate: Array1<f32>,
    /// Aggregated summary vector (statistics over all per-frame features).
    /// 42 features x 7 statistics = 294 dimensions.
    pub summary_vector: Vec<f32>,
}

/// Extract all features from an audio buffer.
pub fn extract_features(
    audio: &AudioBuffer,
    config: &ExtractionConfig,
) -> Result<FeatureSet, RageError> {
    if audio.samples.is_empty() {
        return Err(RageError::Extraction("empty audio buffer".into()));
    }

    // Truncate or use the configured window
    let window_samples = (config.window_seconds * config.sample_rate as f32) as usize;
    let samples = if audio.samples.len() > window_samples {
        let start = (audio.samples.len() - window_samples) / 2;
        &audio.samples[start..start + window_samples]
    } else {
        &audio.samples
    };

    // STFT
    let magnitude = stft::stft_magnitude(samples, config.n_fft, config.hop_length);
    let power = stft::power_spectrum(&magnitude);

    // Mel spectrogram → log-mel
    let mel_spec = mel::mel_spectrogram(&power, config);
    let log_mel = mel::power_to_db(&mel_spec);

    // MFCCs
    let mfccs = mfcc::mfcc(&log_mel, config.n_mfcc);

    // Chroma
    let chroma_feat = chroma::chroma(&magnitude, config.sample_rate, config.n_fft);

    // Spectral features
    let centroid = spectral::spectral_centroid(&magnitude, config.sample_rate, config.n_fft);
    let rolloff = spectral::spectral_rolloff(&magnitude, config.sample_rate, config.n_fft, 0.85);
    let contrast = spectral::spectral_contrast(&power, config.sample_rate, config.n_fft);

    // Temporal features
    let rms = temporal::rms_energy(samples, config.n_fft, config.hop_length);
    let zcr = temporal::zero_crossing_rate(samples, config.n_fft, config.hop_length);

    // Build summary vector: 42 features x 7 statistics = 294 dimensions
    let summary_vector = build_summary_vector(
        &mfccs, &chroma_feat, &centroid, &rolloff, &contrast, &rms, &zcr,
    );

    Ok(FeatureSet {
        log_mel_spectrogram: log_mel,
        mfccs,
        chroma: chroma_feat,
        spectral_centroid: centroid,
        spectral_rolloff: rolloff,
        spectral_contrast: contrast,
        rms_energy: rms,
        zero_crossing_rate: zcr,
        summary_vector,
    })
}

/// Compute 7 summary statistics for a 1-D time series.
/// Returns: [mean, std, min, max, median, skewness, kurtosis]
fn compute_stats(values: &[f32]) -> [f32; 7] {
    if values.is_empty() {
        return [0.0; 7];
    }

    let n = values.len() as f32;

    let mean = values.iter().sum::<f32>() / n;

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let skewness = if std > 1e-10 {
        values.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f32>() / n
    } else {
        0.0
    };

    let kurtosis = if std > 1e-10 {
        values.iter().map(|v| ((v - mean) / std).powi(4)).sum::<f32>() / n - 3.0
    } else {
        0.0
    };

    [mean, std, min, max, median, skewness, kurtosis]
}

/// Build the 294-dimensional summary vector from per-frame features.
///
/// Features (42 total):
///  - 20 MFCCs
///  - 12 chroma bins
///  -  1 spectral centroid
///  -  1 spectral rolloff
///  -  7 spectral contrast bands
///  -  1 RMS energy
///  -  1 zero crossing rate  (originally planned as separate, but this is the standard set)
///  = 43... wait, let me recount:
///
/// Actually: 20 + 12 + 1 + 1 + 7 + 1 + 1 = 43 features
/// But the plan specified 42 features × 7 stats = 294.
/// We use the first 42: drop the spectral contrast mean (last row)
/// which is redundant (it's the average of the 6 contrast bands).
///
/// 42 features × 7 statistics = 294 dimensions.
fn build_summary_vector(
    mfccs: &Array2<f32>,        // [20, T]
    chroma: &Array2<f32>,       // [12, T]
    centroid: &Array1<f32>,     // [T]
    rolloff: &Array1<f32>,      // [T]
    contrast: &Array2<f32>,     // [7, T] (6 bands + 1 mean)
    rms: &Array1<f32>,          // [T]
    zcr: &Array1<f32>,          // [T]
) -> Vec<f32> {
    let n_frames = mfccs.shape()[1];
    let n_stats = 7;

    // Collect all per-frame feature time series
    // 20 MFCCs + 12 chroma + 1 centroid + 1 rolloff + 6 contrast bands + 1 RMS + 1 ZCR = 42
    let mut all_features: Vec<Vec<f32>> = Vec::with_capacity(42);

    // MFCCs (20)
    for k in 0..mfccs.shape()[0] {
        all_features.push((0..n_frames).map(|t| mfccs[[k, t]]).collect());
    }

    // Chroma (12)
    for c in 0..chroma.shape()[0] {
        all_features.push((0..n_frames).map(|t| chroma[[c, t]]).collect());
    }

    // Spectral centroid (1)
    all_features.push(centroid.to_vec());

    // Spectral rolloff (1)
    all_features.push(rolloff.to_vec());

    // Spectral contrast — 6 sub-bands only, skip the mean row (index 6)
    for b in 0..6 {
        all_features.push((0..n_frames).map(|t| contrast[[b, t]]).collect());
    }

    // RMS energy (1)
    all_features.push(rms.to_vec());

    // Zero crossing rate (1)
    all_features.push(zcr.to_vec());

    assert_eq!(all_features.len(), 42, "expected 42 features, got {}", all_features.len());

    // Compute 7 statistics for each feature → 42 × 7 = 294
    let mut summary = Vec::with_capacity(42 * n_stats);
    for feature in &all_features {
        let stats = compute_stats(feature);
        summary.extend_from_slice(&stats);
    }

    assert_eq!(summary.len(), 294);
    summary
}
