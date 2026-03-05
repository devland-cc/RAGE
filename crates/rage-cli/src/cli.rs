use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rage_classifier::Classifier;
use rage_core::ExtractionConfig;

use crate::{deep, output, rage_file};

#[derive(Parser)]
#[command(
    name = "rage",
    about = "RAGE - Rust Aura-Gathering Engine\nAnalyze the mood and emotion of music",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Analyze mood and emotion of audio files
    Analyze(AnalyzeCmd),
    /// Deep analysis: per-beat BPM/key + segmented emotion → .rage file
    Deep(DeepCmd),
    /// Extract audio features from a file (debug/development)
    Extract(ExtractCmd),
    /// Show version and model info
    Info,
}

#[derive(Parser)]
pub struct AnalyzeCmd {
    /// Audio file path(s)
    #[arg(required = true)]
    pub files: Vec<PathBuf>,

    /// Path to ONNX models directory (uses embedded models if omitted)
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Number of top mood tags to show
    #[arg(long, default_value = "10")]
    pub top_k: usize,

    /// Output format
    #[arg(short, long, default_value = "table")]
    pub output: OutputFormat,
}

#[derive(Parser)]
pub struct ExtractCmd {
    /// Audio file path(s)
    #[arg(required = true)]
    pub files: Vec<PathBuf>,

    /// Output format
    #[arg(short, long, default_value = "table")]
    pub output: OutputFormat,
}

#[derive(Parser)]
pub struct DeepCmd {
    /// Audio file path(s)
    #[arg(required = true)]
    pub files: Vec<PathBuf>,

    /// Path to ONNX models directory (uses embedded models if omitted)
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Output directory for .rage files (default: same as input file)
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Segment length in seconds for emotion analysis
    #[arg(long, default_value = "20")]
    pub segment_secs: f32,

    /// Number of top mood tags per segment
    #[arg(long, default_value = "5")]
    pub top_k: usize,

    /// Also print summary to stdout
    #[arg(long)]
    pub print: bool,
}

#[derive(Clone, clap::ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
}

/// Create a classifier from either a custom model directory or embedded models.
fn load_classifier(model_dir: &Option<PathBuf>) -> Result<Classifier> {
    match model_dir {
        Some(dir) => Ok(Classifier::from_dir(dir)?),
        None => Ok(Classifier::embedded()?),
    }
}

impl AnalyzeCmd {
    pub fn run(&self) -> Result<()> {
        let config = ExtractionConfig::default();
        let mut classifier = load_classifier(&self.model_dir)?;

        for path in &self.files {
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());

            let audio = rage_audio::load_audio(path, &config)?;
            let features = rage_extractor::extract_features(&audio, &config)?;
            let result = classifier.classify(&features, &filename)?;

            match self.output {
                OutputFormat::Table => output::print_emotion_table(&result, self.top_k),
                OutputFormat::Json => output::print_emotion_json(&result)?,
            }
        }

        Ok(())
    }
}

impl DeepCmd {
    pub fn run(&self) -> Result<()> {
        let config = ExtractionConfig::default();
        let mut classifier = load_classifier(&self.model_dir)?;

        for path in &self.files {
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());

            println!("Deep analysis: {filename}");

            let audio = rage_audio::load_audio(path, &config)?;

            let analysis = deep::run_deep_analysis(
                &audio,
                &mut classifier,
                &config,
                &filename,
                self.segment_secs,
                self.top_k,
            )?;

            // Determine output path
            let output_path = if let Some(ref dir) = self.output_dir {
                std::fs::create_dir_all(dir)?;
                dir.join(path.file_stem().unwrap_or_default())
                    .with_extension("rage")
            } else {
                path.with_extension("rage")
            };

            rage_file::write_rage_file(&analysis, &output_path)?;
            println!("  Written: {}", output_path.display());

            if self.print {
                println!();
                rage_file::print_deep_summary(&analysis);
            }
        }

        Ok(())
    }
}

impl ExtractCmd {
    pub fn run(&self) -> Result<()> {
        let config = ExtractionConfig::default();

        for path in &self.files {
            println!("Processing: {}", path.display());
            println!();

            let audio = rage_audio::load_audio(path, &config)?;

            println!(
                "  Audio: {:.2}s, {} Hz, {} samples",
                audio.duration_secs(),
                audio.sample_rate,
                audio.samples.len()
            );

            let features = rage_extractor::extract_features(&audio, &config)?;
            let shape = features.log_mel_spectrogram.shape();

            let n_frames = shape[1];

            match self.output {
                OutputFormat::Table => {
                    println!("  Mel spectrogram:   [{} mels, {} frames]", shape[0], n_frames);
                    println!("  MFCCs:             [{}, {}]", features.mfccs.shape()[0], n_frames);
                    println!("  Chroma:            [{}, {}]", features.chroma.shape()[0], n_frames);
                    println!("  Spectral contrast: [{}, {}]", features.spectral_contrast.shape()[0], n_frames);
                    println!("  Summary vector:    {} dimensions", features.summary_vector.len());
                    println!();
                }
                OutputFormat::Json => {
                    let result = serde_json::json!({
                        "file": path.display().to_string(),
                        "duration_secs": audio.duration_secs(),
                        "sample_rate": audio.sample_rate,
                        "n_samples": audio.samples.len(),
                        "n_frames": n_frames,
                        "features": {
                            "log_mel_spectrogram": [shape[0], n_frames],
                            "mfccs": [features.mfccs.shape()[0], n_frames],
                            "chroma": [features.chroma.shape()[0], n_frames],
                            "spectral_centroid": n_frames,
                            "spectral_rolloff": n_frames,
                            "spectral_contrast": [features.spectral_contrast.shape()[0], n_frames],
                            "rms_energy": n_frames,
                            "zero_crossing_rate": n_frames,
                            "summary_vector": features.summary_vector.len(),
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
        }

        Ok(())
    }
}
