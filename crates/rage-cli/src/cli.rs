use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rage_core::ExtractionConfig;

#[derive(Parser)]
#[command(
    name = "rage",
    about = "RAGE - Rust Aura Grabbing Engine\nAnalyze the mood and emotion of music",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Extract audio features from a file
    Extract(ExtractCmd),
    /// Show version and model info
    Info,
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

#[derive(Clone, clap::ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
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

            match self.output {
                OutputFormat::Table => {
                    println!(
                        "  Mel spectrogram: [{} mels, {} frames]",
                        shape[0], shape[1]
                    );

                    // Summary statistics
                    let mel = &features.log_mel_spectrogram;
                    let min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = mel.iter().sum::<f32>() / mel.len() as f32;

                    println!("  Mel stats: min={min:.2} dB, max={max:.2} dB, mean={mean:.2} dB");
                    println!();
                }
                OutputFormat::Json => {
                    let result = serde_json::json!({
                        "file": path.display().to_string(),
                        "duration_secs": audio.duration_secs(),
                        "sample_rate": audio.sample_rate,
                        "n_samples": audio.samples.len(),
                        "mel_spectrogram": {
                            "shape": [shape[0], shape[1]],
                            "min_db": mel_stat_min(&features.log_mel_spectrogram),
                            "max_db": mel_stat_max(&features.log_mel_spectrogram),
                            "mean_db": mel_stat_mean(&features.log_mel_spectrogram),
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
        }

        Ok(())
    }
}

fn mel_stat_min(arr: &ndarray::Array2<f32>) -> f32 {
    arr.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn mel_stat_max(arr: &ndarray::Array2<f32>) -> f32 {
    arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn mel_stat_mean(arr: &ndarray::Array2<f32>) -> f32 {
    arr.iter().sum::<f32>() / arr.len() as f32
}
