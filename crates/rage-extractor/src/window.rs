use std::f32::consts::PI;

/// Generate a Hann window of the given length.
pub fn hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| {
            0.5 * (1.0 - (2.0 * PI * n as f32 / length as f32).cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_properties() {
        let w = hann(1024);
        assert_eq!(w.len(), 1024);
        // Hann window starts and ends near zero
        assert!(w[0].abs() < 1e-6);
        assert!(w[1023].abs() < 0.01);
        // Peak at center
        assert!((w[512] - 1.0).abs() < 0.01);
    }
}
