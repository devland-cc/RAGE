use ndarray::Array2;
use rage_core::ExtractionConfig;

/// Convert frequency in Hz to mel scale (HTK formula).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale back to Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Construct a mel filterbank matrix of shape [n_mels, n_freq_bins].
///
/// Each row is a triangular filter centered at a mel-spaced frequency.
pub fn mel_filterbank(config: &ExtractionConfig) -> Array2<f32> {
    let n_freq = config.n_freq_bins();
    let n_mels = config.n_mels;

    let mel_min = hz_to_mel(config.fmin);
    let mel_max = hz_to_mel(config.fmax);

    // n_mels + 2 equally spaced points in mel space
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel points back to Hz, then to FFT bin indices
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            hz * config.n_fft as f32 / config.sample_rate as f32
        })
        .collect();

    let mut filterbank = Array2::<f32>::zeros((n_mels, n_freq));

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..n_freq {
            let k_f = k as f32;

            if k_f >= f_left && k_f <= f_center && f_center > f_left {
                filterbank[[m, k]] = (k_f - f_left) / (f_center - f_left);
            } else if k_f > f_center && k_f <= f_right && f_right > f_center {
                filterbank[[m, k]] = (f_right - k_f) / (f_right - f_center);
            }
        }
    }

    // Slaney-style normalization: normalize each filter by its bandwidth
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m]));
        for k in 0..n_freq {
            filterbank[[m, k]] *= enorm;
        }
    }

    filterbank
}

/// Compute the mel spectrogram from a power spectrum.
///
/// Input: power_spectrum of shape [n_freq_bins, n_frames]
/// Output: mel spectrogram of shape [n_mels, n_frames]
pub fn mel_spectrogram(
    power_spectrum: &Array2<f32>,
    config: &ExtractionConfig,
) -> Array2<f32> {
    let fb = mel_filterbank(config);
    // fb: [n_mels, n_freq], power: [n_freq, n_frames]
    // result: [n_mels, n_frames]
    fb.dot(power_spectrum)
}

/// Convert a mel spectrogram to log scale (dB).
///
/// Output values are in dB: 10 * log10(max(S, amin))
pub fn power_to_db(spectrogram: &Array2<f32>) -> Array2<f32> {
    let amin = 1e-10f32;
    let top_db = 80.0f32;

    let log_spec = spectrogram.mapv(|s| 10.0 * s.max(amin).log10());

    // Clip to top_db below the maximum value
    let max_val = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    log_spec.mapv(|s| s.max(max_val - top_db))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_hz_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 0.01, "got {back}");
    }

    #[test]
    fn filterbank_shape() {
        let config = ExtractionConfig::default();
        let fb = mel_filterbank(&config);
        assert_eq!(fb.shape(), &[128, 1025]);
    }

    #[test]
    fn filterbank_non_negative() {
        let config = ExtractionConfig::default();
        let fb = mel_filterbank(&config);
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn mel_spectrogram_shape() {
        let config = ExtractionConfig::default();
        let power = Array2::<f32>::ones((1025, 100));
        let mel = mel_spectrogram(&power, &config);
        assert_eq!(mel.shape(), &[128, 100]);
    }
}
