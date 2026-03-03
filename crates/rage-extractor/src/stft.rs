use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::window;

/// Compute the Short-Time Fourier Transform (magnitude spectrum).
///
/// Returns an Array2<f32> of shape [n_freq_bins, n_frames] where
/// n_freq_bins = n_fft / 2 + 1.
///
/// The input is zero-padded at the end if necessary.
pub fn stft_magnitude(
    samples: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Array2<f32> {
    let win = window::hann(n_fft);
    let n_freq = n_fft / 2 + 1;

    // Calculate number of frames (center-pad like librosa)
    let n_frames = if samples.is_empty() {
        0
    } else {
        1 + (samples.len().saturating_sub(1)) / hop_length
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut magnitude = Array2::<f32>::zeros((n_freq, n_frames));

    // Pad samples for center framing
    let pad = n_fft / 2;
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0); n_fft];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Fill FFT buffer with windowed samples
        for i in 0..n_fft {
            let sample_idx = start + i;
            let sample = if sample_idx < padded.len() {
                padded[sample_idx]
            } else {
                0.0
            };
            fft_buffer[i] = Complex::new(sample * win[i], 0.0);
        }

        fft.process(&mut fft_buffer);

        // Extract magnitude of positive frequencies
        for (freq_idx, val) in fft_buffer[..n_freq].iter().enumerate() {
            magnitude[[freq_idx, frame_idx]] = val.norm();
        }
    }

    magnitude
}

/// Compute the power spectrum from the magnitude spectrum.
pub fn power_spectrum(magnitude: &Array2<f32>) -> Array2<f32> {
    magnitude.mapv(|m| m * m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_shape() {
        // 1 second of silence at 22050 Hz
        let samples = vec![0.0f32; 22050];
        let mag = stft_magnitude(&samples, 2048, 512);
        assert_eq!(mag.shape()[0], 1025); // n_fft/2 + 1
        // With center padding: 1 + (22050 - 1) / 512 = 44 frames
        assert!(mag.shape()[1] > 0);
    }

    #[test]
    fn stft_pure_tone() {
        // Generate a 440 Hz sine wave at 22050 Hz sample rate
        let sr = 22050.0f32;
        let freq = 440.0f32;
        let duration = 1.0f32;
        let n_samples = (sr * duration) as usize;

        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();

        let mag = stft_magnitude(&samples, 2048, 512);

        // The peak should be near bin index for 440 Hz
        // bin = freq * n_fft / sr = 440 * 2048 / 22050 ≈ 40.9
        let mid_frame = mag.shape()[1] / 2;
        let frame = mag.column(mid_frame);

        let (peak_bin, _peak_val) = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Allow ±2 bins of tolerance
        let expected_bin = (freq * 2048.0 / sr).round() as usize;
        assert!(
            (peak_bin as i32 - expected_bin as i32).unsigned_abs() <= 2,
            "expected peak near bin {expected_bin}, got {peak_bin}"
        );
    }
}
