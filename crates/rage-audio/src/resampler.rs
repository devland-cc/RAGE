use rage_core::RageError;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resample mono audio from `from_rate` to `to_rate`.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, RageError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f64>::new(
        ratio,
        2.0,
        params,
        chunk_size,
        1, // mono
    )
    .map_err(|e| RageError::Resample(format!("failed to create resampler: {e}")))?;

    let mut output = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk_size);

    // Process in chunks
    let mut pos = 0;
    while pos + chunk_size <= samples.len() {
        let chunk: Vec<f64> = samples[pos..pos + chunk_size]
            .iter()
            .map(|&s| s as f64)
            .collect();

        let result = resampler
            .process(&[chunk], None)
            .map_err(|e| RageError::Resample(format!("resample error: {e}")))?;

        output.extend(result[0].iter().map(|&s| s as f32));
        pos += chunk_size;
    }

    // Process remaining samples (pad with zeros to fill the last chunk)
    if pos < samples.len() {
        let remaining = samples.len() - pos;
        let mut chunk = vec![0.0f64; chunk_size];
        for (i, &s) in samples[pos..].iter().enumerate() {
            chunk[i] = s as f64;
        }

        let result = resampler
            .process(&[chunk], None)
            .map_err(|e| RageError::Resample(format!("resample error: {e}")))?;

        // Only take the proportional amount of output
        let expected = (remaining as f64 * ratio).ceil() as usize;
        let take = expected.min(result[0].len());
        output.extend(result[0][..take].iter().map(|&s| s as f32));
    }

    Ok(output)
}
