use serde::{Deserialize, Serialize};

/// Configuration for the audio feature extraction pipeline.
///
/// Defaults are aligned with librosa to ensure parity between
/// Python training and Rust inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Target sample rate in Hz.
    pub sample_rate: u32,
    /// FFT window size.
    pub n_fft: usize,
    /// Hop length between successive frames.
    pub hop_length: usize,
    /// Number of mel filterbank bins.
    pub n_mels: usize,
    /// Number of MFCC coefficients to extract.
    pub n_mfcc: usize,
    /// Minimum frequency for mel filterbank (Hz).
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (Hz).
    pub fmax: f32,
    /// Analysis window duration in seconds.
    pub window_seconds: f32,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
            n_mfcc: 20,
            fmin: 20.0,
            fmax: 11025.0,
            window_seconds: 30.0,
        }
    }
}

impl ExtractionConfig {
    /// Number of frequency bins in the STFT output (n_fft / 2 + 1).
    pub fn n_freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Expected number of time frames for the configured window duration.
    pub fn n_frames(&self) -> usize {
        let n_samples = (self.window_seconds * self.sample_rate as f32) as usize;
        1 + (n_samples.saturating_sub(self.n_fft)) / self.hop_length
    }
}
