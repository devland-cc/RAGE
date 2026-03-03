use ndarray::{Array1, Array2};

/// Compute spectral centroid for each frame.
///
/// The spectral centroid is the weighted mean of frequencies,
/// indicating the "center of mass" of the spectrum.
///
/// Input: stft_magnitude of shape [n_freq, n_frames]
/// Output: Array1 of shape [n_frames]
pub fn spectral_centroid(
    stft_magnitude: &Array2<f32>,
    sample_rate: u32,
    n_fft: usize,
) -> Array1<f32> {
    let n_freq = stft_magnitude.shape()[0];
    let n_frames = stft_magnitude.shape()[1];
    let freq_per_bin = sample_rate as f32 / n_fft as f32;

    let mut result = Array1::<f32>::zeros(n_frames);

    for t in 0..n_frames {
        let mut weighted_sum = 0.0f32;
        let mut total_mag = 0.0f32;

        for f in 0..n_freq {
            let mag = stft_magnitude[[f, t]];
            let freq = f as f32 * freq_per_bin;
            weighted_sum += mag * freq;
            total_mag += mag;
        }

        result[t] = if total_mag > 1e-10 {
            weighted_sum / total_mag
        } else {
            0.0
        };
    }

    result
}

/// Compute spectral rolloff for each frame.
///
/// The rolloff frequency is the frequency below which a given percentage
/// (default 85%) of the total spectral energy is contained.
///
/// Input: stft_magnitude of shape [n_freq, n_frames]
/// Output: Array1 of shape [n_frames]
pub fn spectral_rolloff(
    stft_magnitude: &Array2<f32>,
    sample_rate: u32,
    n_fft: usize,
    roll_percent: f32,
) -> Array1<f32> {
    let n_freq = stft_magnitude.shape()[0];
    let n_frames = stft_magnitude.shape()[1];
    let freq_per_bin = sample_rate as f32 / n_fft as f32;

    let mut result = Array1::<f32>::zeros(n_frames);

    for t in 0..n_frames {
        let total_energy: f32 = (0..n_freq).map(|f| stft_magnitude[[f, t]]).sum();
        let threshold = total_energy * roll_percent;

        let mut cumulative = 0.0f32;
        for f in 0..n_freq {
            cumulative += stft_magnitude[[f, t]];
            if cumulative >= threshold {
                result[t] = f as f32 * freq_per_bin;
                break;
            }
        }
    }

    result
}

/// Compute spectral contrast for each frame.
///
/// Measures the difference between spectral peaks and valleys in
/// sub-bands. Uses 6 octave sub-bands plus one valley band.
///
/// Input: power_spectrum of shape [n_freq, n_frames]
/// Output: Array2 of shape [7, n_frames] (6 contrasts + 1 valley mean)
pub fn spectral_contrast(
    power_spectrum: &Array2<f32>,
    sample_rate: u32,
    n_fft: usize,
) -> Array2<f32> {
    let n_freq = power_spectrum.shape()[0];
    let n_frames = power_spectrum.shape()[1];
    let freq_per_bin = sample_rate as f32 / n_fft as f32;
    let n_bands = 6;

    // Define octave band edges (Hz)
    let band_edges: Vec<f32> = {
        let fmin = 200.0f32;
        let mut edges = Vec::with_capacity(n_bands + 2);
        edges.push(0.0);
        for i in 0..=n_bands {
            edges.push(fmin * 2.0f32.powi(i as i32));
        }
        edges
    };

    let mut result = Array2::<f32>::zeros((n_bands + 1, n_frames));

    // Fraction of bins to consider as peaks/valleys (top/bottom 20%)
    let alpha = 0.2f32;

    for t in 0..n_frames {
        for band in 0..n_bands {
            let f_lo = (band_edges[band] / freq_per_bin).round() as usize;
            let f_hi = ((band_edges[band + 1] / freq_per_bin).round() as usize).min(n_freq);

            if f_hi <= f_lo {
                continue;
            }

            // Collect power values in this band
            let mut band_power: Vec<f32> = (f_lo..f_hi)
                .map(|f| power_spectrum[[f, t]])
                .collect();

            band_power.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_bins = band_power.len();
            let n_edge = (n_bins as f32 * alpha).ceil() as usize;
            let n_edge = n_edge.max(1).min(n_bins);

            // Valley: mean of bottom alpha fraction
            let valley: f32 =
                band_power[..n_edge].iter().sum::<f32>() / n_edge as f32;

            // Peak: mean of top alpha fraction
            let peak: f32 =
                band_power[n_bins - n_edge..].iter().sum::<f32>() / n_edge as f32;

            // Contrast = log(peak) - log(valley) in dB-like scale
            let peak_db = 10.0 * (peak + 1e-10).log10();
            let valley_db = 10.0 * (valley + 1e-10).log10();
            result[[band, t]] = peak_db - valley_db;
        }

        // Last row: mean valley across all bands (as a summary)
        let overall_valley: f32 = (0..n_bands).map(|b| result[[b, t]]).sum::<f32>()
            / n_bands as f32;
        result[[n_bands, t]] = overall_valley;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_shape() {
        let mag = Array2::<f32>::ones((1025, 50));
        let result = spectral_centroid(&mag, 22050, 2048);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn centroid_pure_tone() {
        // Energy at a single bin (440 Hz → bin 41)
        let mut mag = Array2::<f32>::zeros((1025, 1));
        mag[[41, 0]] = 1.0;

        let result = spectral_centroid(&mag, 22050, 2048);
        let expected = 41.0 * 22050.0 / 2048.0;
        assert!(
            (result[0] - expected).abs() < 1.0,
            "got {}, expected ~{expected}",
            result[0]
        );
    }

    #[test]
    fn rolloff_shape() {
        let mag = Array2::<f32>::ones((1025, 50));
        let result = spectral_rolloff(&mag, 22050, 2048, 0.85);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn contrast_shape() {
        let power = Array2::<f32>::ones((1025, 50));
        let result = spectral_contrast(&power, 22050, 2048);
        assert_eq!(result.shape(), &[7, 50]);
    }
}
