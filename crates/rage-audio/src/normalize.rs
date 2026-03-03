/// Convert interleaved multi-channel audio to mono by averaging channels.
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Peak normalize audio samples to [-1, 1].
///
/// If the audio is silent (all zeros), returns the samples unchanged.
pub fn peak_normalize(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if peak < 1e-10 {
        return samples.to_vec();
    }

    let scale = 1.0 / peak;
    samples.iter().map(|s| s * scale).collect()
}
