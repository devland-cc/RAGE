mod cli;
mod output;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = cli::Cli::parse();

    match args.command {
        cli::Command::Analyze(cmd) => cmd.run()?,
        cli::Command::Extract(cmd) => cmd.run()?,
        cli::Command::Info => {
            println!("RAGE - Rust Aura Grabbing Engine v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Supported formats: WAV, MP3, FLAC");
            println!("Mood tags: {} tags (MTG-Jamendo)", rage_core::MoodTag::ALL.len());
            println!("Output: mood tags + valence/arousal scores");
        }
    }

    Ok(())
}
