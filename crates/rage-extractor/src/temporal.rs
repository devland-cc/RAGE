use ndarray::Array1;

/// Compute RMS (Root Mean Square) energy per frame.
///
/// Input: raw audio samples, frame parameters
/// Output: Array1 of shape [n_frames]
pub fn rms_energy(samples: &[f32], n_fft: usize, hop_length: usize) -> Array1<f32> {
    let n_frames = if samples.is_empty() {
        0
    } else {
        1 + (samples.len().saturating_sub(1)) / hop_length
    };

    let pad = n_fft / 2;
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    let mut result = Array1::<f32>::zeros(n_frames);

    for t in 0..n_frames {
        let start = t * hop_length;
        let mut sum_sq = 0.0f32;
        let mut count = 0;

        for i in 0..n_fft {
            let idx = start + i;
            if idx < padded.len() {
                sum_sq += padded[idx] * padded[idx];
                count += 1;
            }
        }

        result[t] = if count > 0 {
            (sum_sq / count as f32).sqrt()
        } else {
            0.0
        };
    }

    result
}

/// Compute Zero Crossing Rate per frame.
///
/// Counts the number of times the signal changes sign within each frame,
/// normalized by the frame length.
///
/// Input: raw audio samples, frame parameters
/// Output: Array1 of shape [n_frames]
pub fn zero_crossing_rate(samples: &[f32], n_fft: usize, hop_length: usize) -> Array1<f32> {
    let n_frames = if samples.is_empty() {
        0
    } else {
        1 + (samples.len().saturating_sub(1)) / hop_length
    };

    let pad = n_fft / 2;
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    let mut result = Array1::<f32>::zeros(n_frames);

    for t in 0..n_frames {
        let start = t * hop_length;
        let mut crossings = 0u32;

        for i in 1..n_fft {
            let idx = start + i;
            let prev_idx = start + i - 1;
            if idx < padded.len() && prev_idx < padded.len() {
                if (padded[idx] >= 0.0) != (padded[prev_idx] >= 0.0) {
                    crossings += 1;
                }
            }
        }

        result[t] = crossings as f32 / (n_fft - 1) as f32;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_silence() {
        let samples = vec![0.0f32; 22050];
        let rms = rms_energy(&samples, 2048, 512);
        assert!(rms.iter().all(|&v| v < 1e-10));
    }

    #[test]
    fn rms_sine() {
        // RMS of a sine wave should be 1/sqrt(2) ≈ 0.707
        let sr = 22050;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let rms = rms_energy(&samples, 2048, 512);

        // Check a frame in the middle (away from padding effects)
        let mid = rms.len() / 2;
        let expected = 1.0 / 2.0f32.sqrt();
        assert!(
            (rms[mid] - expected).abs() < 0.05,
            "got {}, expected ~{expected}",
            rms[mid]
        );
    }

    #[test]
    fn zcr_shape() {
        let samples = vec![0.0f32; 22050];
        let zcr = zero_crossing_rate(&samples, 2048, 512);
        assert!(zcr.len() > 0);
    }

    #[test]
    fn zcr_sine() {
        // ZCR of a 440 Hz sine at 22050 Hz sample rate
        // Expected: ~2 * 440 / 22050 ≈ 0.0399 crossings per sample
        let sr = 22050;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let zcr = zero_crossing_rate(&samples, 2048, 512);
        let mid = zcr.len() / 2;

        // 440 Hz → 880 zero crossings per second → 880/22050 ≈ 0.04 per sample
        // Per frame of 2048 samples: ~82 crossings → 82/2047 ≈ 0.04
        assert!(
            zcr[mid] > 0.03 && zcr[mid] < 0.06,
            "got {}, expected ~0.04",
            zcr[mid]
        );
    }
}
