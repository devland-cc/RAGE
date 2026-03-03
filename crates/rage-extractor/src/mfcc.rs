use ndarray::Array2;

/// Compute MFCCs from a log-mel spectrogram using Type-II DCT.
///
/// Input: log_mel_spectrogram of shape [n_mels, n_frames]
/// Output: MFCCs of shape [n_mfcc, n_frames]
///
/// Applies the Type-II DCT along the mel axis for each frame,
/// keeping the first `n_mfcc` coefficients.
pub fn mfcc(log_mel: &Array2<f32>, n_mfcc: usize) -> Array2<f32> {
    let n_mels = log_mel.shape()[0];
    let n_frames = log_mel.shape()[1];

    let mut result = Array2::<f32>::zeros((n_mfcc, n_frames));

    // Precompute DCT-II basis (ortho-normalized)
    let dct_basis = dct2_basis(n_mfcc, n_mels);

    for t in 0..n_frames {
        for k in 0..n_mfcc {
            let mut sum = 0.0f32;
            for n in 0..n_mels {
                sum += dct_basis[[k, n]] * log_mel[[n, t]];
            }
            result[[k, t]] = sum;
        }
    }

    result
}

/// Compute the Type-II DCT basis matrix with orthogonal normalization.
///
/// Returns shape [n_mfcc, n_mels], matching scipy.fftpack.dct(type=2, norm='ortho').
fn dct2_basis(n_mfcc: usize, n_mels: usize) -> Array2<f32> {
    let mut basis = Array2::<f32>::zeros((n_mfcc, n_mels));

    for k in 0..n_mfcc {
        for n in 0..n_mels {
            basis[[k, n]] = (std::f32::consts::PI * k as f32 * (n as f32 + 0.5)
                / n_mels as f32)
                .cos();
        }
    }

    // Orthogonal normalization
    let scale_0 = (1.0 / n_mels as f32).sqrt();
    let scale_k = (2.0 / n_mels as f32).sqrt();

    for n in 0..n_mels {
        basis[[0, n]] *= scale_0;
    }
    for k in 1..n_mfcc {
        for n in 0..n_mels {
            basis[[k, n]] *= scale_k;
        }
    }

    basis
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn mfcc_shape() {
        let log_mel = Array2::<f32>::zeros((128, 100));
        let result = mfcc(&log_mel, 20);
        assert_eq!(result.shape(), &[20, 100]);
    }

    #[test]
    fn mfcc_constant_input() {
        // Constant input should produce energy only in the 0th coefficient
        let log_mel = Array2::from_elem((128, 10), 1.0f32);
        let result = mfcc(&log_mel, 20);

        // DC coefficient should be non-zero
        assert!(result[[0, 0]].abs() > 0.1);

        // Higher coefficients should be near zero for constant input
        for k in 1..20 {
            assert!(
                result[[k, 0]].abs() < 1e-5,
                "coeff {k} = {} (expected ~0)",
                result[[k, 0]]
            );
        }
    }
}
