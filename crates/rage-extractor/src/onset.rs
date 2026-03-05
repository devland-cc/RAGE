use ndarray::{Array1, Array2};

/// Compute onset strength from STFT magnitude using spectral flux.
///
/// For each frame, computes half-wave rectified spectral flux:
///   flux[t] = sum_f max(0, mag[f,t] - mag[f,t-1])
///
/// Output is normalized to [0, 1].
///
/// Input: stft_magnitude of shape [n_freq, n_frames]
/// Output: onset strength of shape [n_frames]
pub fn onset_strength(stft_mag: &Array2<f32>) -> Array1<f32> {
    let n_frames = stft_mag.shape()[1];
    let n_freq = stft_mag.shape()[0];

    if n_frames == 0 {
        return Array1::zeros(0);
    }

    let mut onset = Array1::<f32>::zeros(n_frames);

    // First frame has no previous frame to compare against
    for t in 1..n_frames {
        let mut flux = 0.0f32;
        for f in 0..n_freq {
            let diff = stft_mag[[f, t]] - stft_mag[[f, t - 1]];
            flux += diff.max(0.0); // half-wave rectification
        }
        onset[t] = flux;
    }

    // Normalize to [0, 1]
    let max_val = onset.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 1e-10 {
        onset.mapv_inplace(|v| v / max_val);
    }

    onset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onset_shape() {
        let stft_mag = Array2::<f32>::zeros((1025, 50));
        let onset = onset_strength(&stft_mag);
        assert_eq!(onset.len(), 50);
    }

    #[test]
    fn onset_silence() {
        // Silent signal should produce zero onset strength
        let stft_mag = Array2::<f32>::zeros((1025, 50));
        let onset = onset_strength(&stft_mag);
        for &v in onset.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn onset_impulse() {
        // Create a spectrum with a sudden onset at frame 10
        let mut stft_mag = Array2::<f32>::zeros((1025, 50));
        for f in 0..1025 {
            for t in 10..50 {
                stft_mag[[f, t]] = 1.0;
            }
        }

        let onset = onset_strength(&stft_mag);

        // Frame 10 should have the highest onset strength
        let peak_frame = onset
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        assert_eq!(peak_frame, 10);
        assert!((onset[10] - 1.0).abs() < 1e-5, "peak should be 1.0 after normalization");
    }

    #[test]
    fn onset_periodic_impulses() {
        // Create periodic onsets every 10 frames
        let mut stft_mag = Array2::<f32>::zeros((100, 50));
        for f in 0..100 {
            for &t in &[10, 20, 30, 40] {
                stft_mag[[f, t]] = 1.0;
            }
        }

        let onset = onset_strength(&stft_mag);

        // Onset frames should have high values, others near zero
        for &t in &[10, 20, 30, 40] {
            assert!(onset[t] > 0.5, "frame {t} should have high onset: {}", onset[t]);
        }
        for &t in &[0, 5, 15, 25, 35, 45] {
            assert!(onset[t] < 0.01, "frame {t} should have low onset: {}", onset[t]);
        }
    }

    #[test]
    fn onset_empty() {
        let stft_mag = Array2::<f32>::zeros((1025, 0));
        let onset = onset_strength(&stft_mag);
        assert_eq!(onset.len(), 0);
    }
}
