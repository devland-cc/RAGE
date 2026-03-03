use thiserror::Error;

#[derive(Error, Debug)]
pub enum RageError {
    #[error("audio decoding error: {0}")]
    Decode(String),

    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("resampling error: {0}")]
    Resample(String),

    #[error("feature extraction error: {0}")]
    Extraction(String),

    #[error("model inference error: {0}")]
    Inference(String),

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("audio file is empty or too short (need at least {min_samples} samples, got {actual_samples})")]
    AudioTooShort {
        min_samples: usize,
        actual_samples: usize,
    },
}
