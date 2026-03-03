use ndarray::Array2;
use rage_core::{AudioBuffer, ExtractionConfig, RageError};

use crate::mel;
use crate::stft;

/// Extracted audio features.
#[derive(Debug)]
pub struct FeatureSet {
    /// Log-mel spectrogram in dB, shape [n_mels, n_frames].
    pub log_mel_spectrogram: Array2<f32>,
}

/// Extract features from an audio buffer.
///
/// Currently extracts the log-mel spectrogram (CNN input).
/// Additional features (MFCCs, chroma, spectral, temporal) will be added
/// in Phase 2.
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
        // Use the middle section of the audio
        let start = (audio.samples.len() - window_samples) / 2;
        &audio.samples[start..start + window_samples]
    } else {
        &audio.samples
    };

    // STFT
    let magnitude = stft::stft_magnitude(samples, config.n_fft, config.hop_length);
    let power = stft::power_spectrum(&magnitude);

    // Mel spectrogram
    let mel_spec = mel::mel_spectrogram(&power, config);
    let log_mel = mel::power_to_db(&mel_spec);

    Ok(FeatureSet {
        log_mel_spectrogram: log_mel,
    })
}
