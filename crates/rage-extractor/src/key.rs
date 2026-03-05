use ndarray::Array2;
use rage_core::{KeyMode, MusicalKey};

/// Krumhansl-Kessler major key profile (starting from C).
const KK_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Krumhansl-Kessler minor key profile (starting from C).
const KK_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Detect the musical key from a mean chroma vector using Krumhansl-Kessler profiles.
///
/// Computes Pearson correlation between the chroma vector and all 24 key profiles
/// (12 major + 12 minor, each a rotation of the base profile).
/// Returns the key with the highest correlation.
pub fn detect_key(mean_chroma: &[f32; 12]) -> MusicalKey {
    let mut best_corr = f32::NEG_INFINITY;
    let mut best_key = MusicalKey::new(0, KeyMode::Major);

    for pitch_class in 0..12u8 {
        // Rotate the profile so it starts at this pitch class
        let major_profile = rotate_profile(&KK_MAJOR, pitch_class as usize);
        let minor_profile = rotate_profile(&KK_MINOR, pitch_class as usize);

        let major_corr = pearson_correlation(mean_chroma, &major_profile);
        let minor_corr = pearson_correlation(mean_chroma, &minor_profile);

        if major_corr > best_corr {
            best_corr = major_corr;
            best_key = MusicalKey::new(pitch_class, KeyMode::Major);
        }
        if minor_corr > best_corr {
            best_corr = minor_corr;
            best_key = MusicalKey::new(pitch_class, KeyMode::Minor);
        }
    }

    best_key
}

/// Compute the mean chroma vector over a window centered at `center`.
///
/// `chroma` has shape [12, n_frames].
/// Returns the mean chroma over frames [center - half_window, center + half_window],
/// clamped to array bounds.
pub fn mean_chroma_window(chroma: &Array2<f32>, center: usize, half_window: usize) -> [f32; 12] {
    let n_frames = chroma.shape()[1];
    let start = center.saturating_sub(half_window);
    let end = (center + half_window + 1).min(n_frames);

    let mut mean = [0.0f32; 12];
    let count = (end - start) as f32;

    if count < 1.0 {
        return mean;
    }

    for t in start..end {
        for c in 0..12 {
            mean[c] += chroma[[c, t]];
        }
    }

    for c in 0..12 {
        mean[c] /= count;
    }

    mean
}

/// Compute the mean chroma vector over the full chroma matrix.
///
/// `chroma` has shape [12, n_frames].
pub fn mean_chroma_full(chroma: &Array2<f32>) -> [f32; 12] {
    let n_frames = chroma.shape()[1];
    if n_frames == 0 {
        return [0.0; 12];
    }

    let mut mean = [0.0f32; 12];
    for t in 0..n_frames {
        for c in 0..12 {
            mean[c] += chroma[[c, t]];
        }
    }
    for c in 0..12 {
        mean[c] /= n_frames as f32;
    }
    mean
}

/// Rotate a 12-element profile by `shift` positions.
/// rotate_profile([a,b,c,...], 2) → [c,d,...,a,b] (shift the reference pitch)
fn rotate_profile(profile: &[f32; 12], shift: usize) -> [f32; 12] {
    let mut rotated = [0.0f32; 12];
    for i in 0..12 {
        rotated[i] = profile[(i + 12 - shift) % 12];
    }
    rotated
}

/// Pearson correlation coefficient between two 12-element vectors.
fn pearson_correlation(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let n = 12.0f32;

    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;

    for i in 0..12 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_c_major() {
        // Feed the C major profile itself as the chroma vector
        let chroma = KK_MAJOR;
        let key = detect_key(&chroma);
        assert_eq!(key.pitch_class, 0, "expected C, got {}", key);
        assert_eq!(key.mode, KeyMode::Major, "expected major, got {}", key);
    }

    #[test]
    fn detect_a_minor() {
        // A minor: rotate the minor profile so A is at position 0
        // A = pitch class 9, so we need the profile starting at A
        let mut chroma = [0.0f32; 12];
        for i in 0..12 {
            chroma[(i + 9) % 12] = KK_MINOR[i];
        }
        let key = detect_key(&chroma);
        assert_eq!(key.pitch_class, 9, "expected A, got {}", key);
        assert_eq!(key.mode, KeyMode::Minor, "expected minor, got {}", key);
    }

    #[test]
    fn detect_g_major() {
        // G major: rotate so G (pitch class 7) is the tonic
        let mut chroma = [0.0f32; 12];
        for i in 0..12 {
            chroma[(i + 7) % 12] = KK_MAJOR[i];
        }
        let key = detect_key(&chroma);
        assert_eq!(key.pitch_class, 7, "expected G, got {}", key);
        assert_eq!(key.mode, KeyMode::Major, "expected major, got {}", key);
    }

    #[test]
    fn display_keys() {
        assert_eq!(
            format!("{}", MusicalKey::new(0, KeyMode::Major)),
            "C major"
        );
        assert_eq!(
            format!("{}", MusicalKey::new(6, KeyMode::Minor)),
            "F# minor"
        );
        assert_eq!(
            format!("{}", MusicalKey::new(9, KeyMode::Major)),
            "A major"
        );
    }

    #[test]
    fn mean_chroma_window_basic() {
        // 12 x 5 chroma with known values
        let mut chroma = Array2::<f32>::zeros((12, 5));
        for t in 0..5 {
            chroma[[0, t]] = 1.0; // C is always 1.0
        }

        let mean = mean_chroma_window(&chroma, 2, 1);
        assert!((mean[0] - 1.0).abs() < 1e-5);
        for c in 1..12 {
            assert!((mean[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn mean_chroma_full_basic() {
        let mut chroma = Array2::<f32>::zeros((12, 10));
        chroma[[3, 5]] = 1.0;
        let mean = mean_chroma_full(&chroma);
        assert!((mean[3] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn rotate_profile_identity() {
        let rotated = rotate_profile(&KK_MAJOR, 0);
        assert_eq!(rotated, KK_MAJOR);
    }

    #[test]
    fn pearson_self_correlation() {
        let corr = pearson_correlation(&KK_MAJOR, &KK_MAJOR);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "self-correlation should be 1.0, got {corr}"
        );
    }
}
