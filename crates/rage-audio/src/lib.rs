pub mod decoder;
pub mod normalize;
pub mod resampler;

use rage_core::{AudioBuffer, ExtractionConfig, RageError};
use std::path::Path;

/// Load an audio file, decode it, resample to the target sample rate,
/// and normalize to mono f32 in [-1, 1].
pub fn load_audio(
    path: &Path,
    config: &ExtractionConfig,
) -> Result<AudioBuffer, RageError> {
    let (samples, sample_rate, channels) = decoder::decode_file(path)?;
    let mono = normalize::to_mono(&samples, channels);
    let resampled = if sample_rate != config.sample_rate {
        resampler::resample(&mono, sample_rate, config.sample_rate)?
    } else {
        mono
    };
    let normalized = normalize::peak_normalize(&resampled);
    Ok(AudioBuffer::new(normalized, config.sample_rate))
}
