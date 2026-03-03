use ndarray::Array2;

/// Compute chroma features (pitch class profiles) from the STFT magnitude.
///
/// Maps FFT bins to 12 pitch classes (C, C#, D, ..., B) by folding
/// the spectrum into a single octave.
///
/// Input: stft_magnitude of shape [n_freq, n_frames]
/// Output: chroma of shape [12, n_frames]
pub fn chroma(stft_magnitude: &Array2<f32>, sample_rate: u32, n_fft: usize) -> Array2<f32> {
    let n_freq = stft_magnitude.shape()[0];
    let n_frames = stft_magnitude.shape()[1];
    let n_chroma = 12;

    // Build the chroma filterbank: maps FFT bins to pitch classes
    let chromafb = chroma_filterbank(sample_rate, n_fft, n_freq, n_chroma);

    // chromafb: [12, n_freq], stft_magnitude: [n_freq, n_frames]
    // result: [12, n_frames]
    let mut result = Array2::<f32>::zeros((n_chroma, n_frames));

    for t in 0..n_frames {
        for c in 0..n_chroma {
            let mut sum = 0.0f32;
            for f in 0..n_freq {
                sum += chromafb[[c, f]] * stft_magnitude[[f, t]];
            }
            result[[c, t]] = sum;
        }
    }

    // Normalize each frame to sum to 1 (L1 normalization)
    for t in 0..n_frames {
        let sum: f32 = (0..n_chroma).map(|c| result[[c, t]]).sum();
        if sum > 1e-10 {
            for c in 0..n_chroma {
                result[[c, t]] /= sum;
            }
        }
    }

    result
}

/// Build a chroma filterbank matrix mapping FFT bins to pitch classes.
///
/// Returns shape [n_chroma, n_freq].
fn chroma_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_freq: usize,
    n_chroma: usize,
) -> Array2<f32> {
    let mut fb = Array2::<f32>::zeros((n_chroma, n_freq));

    let freq_per_bin = sample_rate as f32 / n_fft as f32;

    for f in 1..n_freq {
        let freq = f as f32 * freq_per_bin;

        // Convert frequency to pitch class using MIDI note mapping
        // MIDI note = 69 + 12 * log2(freq / 440)
        // Pitch class = MIDI note mod 12
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let chroma_bin = ((midi % 12.0 + 12.0) % 12.0) as usize % n_chroma;

        fb[[chroma_bin, f]] += 1.0;
    }

    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_shape() {
        let stft_mag = Array2::<f32>::ones((1025, 50));
        let result = chroma(&stft_mag, 22050, 2048);
        assert_eq!(result.shape(), &[12, 50]);
    }

    #[test]
    fn chroma_normalized() {
        let stft_mag = Array2::<f32>::ones((1025, 10));
        let result = chroma(&stft_mag, 22050, 2048);

        // Each frame should sum to ~1 after L1 normalization
        for t in 0..10 {
            let sum: f32 = (0..12).map(|c| result[[c, t]]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "frame {t} sums to {sum}"
            );
        }
    }

    #[test]
    fn chroma_a440() {
        // Create a spectrum with energy only at 440 Hz (A4)
        // Bin for 440 Hz: 440 * 2048 / 22050 ≈ 40.9 → bin 41
        let mut stft_mag = Array2::<f32>::zeros((1025, 1));
        stft_mag[[41, 0]] = 1.0;

        let result = chroma(&stft_mag, 22050, 2048);

        // A4 should map to pitch class A (index 9 in C, C#, D, ..., B)
        let peak = (0..12)
            .max_by(|&a, &b| result[[a, 0]].partial_cmp(&result[[b, 0]]).unwrap())
            .unwrap();

        assert_eq!(peak, 9, "expected A (index 9), got index {peak}");
    }
}
