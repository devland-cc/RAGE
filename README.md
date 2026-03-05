# RAGE - Rust Aura-Gathering Engine

A lightweight, pure-Rust system for analyzing the mood and emotion of music. RAGE extracts audio features and classifies mood/theme tags and valence-arousal values from audio files.

> **Status**: Fully functional — download, run, and analyze music out of the box. Pre-trained ONNX models are embedded in the binary.

> *Analyze the mood, tempo, and key of any song with a single command — no Python, no setup, no API keys.*

## Installation

### Download a pre-built binary

Grab the latest release for your platform from the [Releases page](https://github.com/devland-cc/RAGE/releases):

| Platform | File |
|----------|------|
| macOS (Apple Silicon) | `rage-*-aarch64-apple-darwin.tar.gz` |
| Linux (x86_64) | `rage-*-x86_64-unknown-linux-gnu.tar.gz` |
| Linux (aarch64) | `rage-*-aarch64-unknown-linux-gnu.tar.gz` |

> **macOS (Intel)**: Pre-built binaries are not available for Intel Macs. Please [build from source](#build-from-source) instead.

```bash
# Example: download and install on macOS Apple Silicon
tar xzf rage-1.1.1-aarch64-apple-darwin.tar.gz
sudo mv rage /usr/local/bin/
```

> **macOS note**: If Gatekeeper blocks the binary, run: `xattr -d com.apple.quarantine rage`

### Build from source

```bash
git clone https://github.com/devland-cc/RAGE.git
cd RAGE
cargo build --release
# Binary is at target/release/rage
```

## Architecture

RAGE is organized as a Cargo workspace with five crates:

```
crates/
  rage-core/        Shared types, configuration, error types, mood tag definitions
  rage-audio/       Audio decoding (Symphonia), resampling (Rubato), normalization
  rage-extractor/   Feature extraction: STFT, mel spectrogram, MFCCs, chroma,
                    spectral features, temporal features, 294-dim summary vector
  rage-classifier/  ML inference via ONNX Runtime (ort crate)
  rage-cli/         CLI binary
```

### Feature Extraction

All extraction parameters are aligned with librosa for training/inference parity:

- Sample rate: 22050 Hz
- FFT size: 2048, hop length: 512
- 128 mel bins (HTK scale, Slaney normalization)
- 20 MFCCs, 12 chroma bins
- Spectral centroid, rolloff, contrast
- RMS energy, zero-crossing rate
- 294-dimensional summary vector (42 features x 7 statistics)

### Pre-trained Models (included)

RAGE ships with two ONNX models in `models/` — no extra downloads or setup needed:

| Model | File | Description |
|-------|------|-------------|
| **MoodTagger** | `mood_tagger.onnx` (8.7 MB) | 5-layer CNN predicting 56 mood/theme tags (trained on MTG-Jamendo) |
| **ValenceArousalModel** | `valence_arousal.onnx` (10 MB) | Dual-branch CNN+MLP predicting continuous valence and arousal (trained on DEAM) |

## Quick Start

### Analyze music

```bash
# Analyze a song (models are embedded — no extra setup)
rage analyze song.mp3

# JSON output
rage analyze --output json song.mp3

# Multiple files
rage analyze song1.mp3 song2.flac song3.wav

# Override with custom models
rage analyze --model-dir /path/to/models song.mp3
```

Example output:
```
  song.mp3

  Valence: +0.148  (neutral)
  Arousal: +0.035  (moderate)

  Top Mood Tags:
     1. love             0.190
     2. ballad           0.127
     3. energetic        0.085
     4. dark             0.031
     5. melodic          0.029
```

### Deep analysis

Deep analysis processes the entire song and generates a `.rage` file containing per-beat BPM/key tracking and segmented emotion analysis.

```bash
# Deep analysis → generates song.rage
rage deep song.mp3

# Print summary to stdout as well
rage deep song.mp3 --print

# Custom segment length and output directory
rage deep song.mp3 --segment-secs 30 --output-dir results/
```

Example summary output:
```
  song.mp3
  Duration: 240.1s

  Dominant BPM: 104
  Dominant Key: E major

  Avg Valence: +0.503
  Avg Arousal: +0.534

  Top Moods:
     1. energetic        0.164
     2. epic             0.102
     3. dark             0.089
     4. action           0.084
     5. melodic          0.067

  Beats: 395
  Segments: 12
```

See [RAGE_FORMAT.md](RAGE_FORMAT.md) for the `.rage` file format specification.

### Feature extraction (CLI)

```bash
# Table output
rage extract song.mp3

# JSON output
rage extract --output json song.mp3
```

### Retrain models (optional, Mac with Apple Silicon)

```bash
# Set up Python environment
conda create -n rage-train python=3.11 -y
conda activate rage-train
pip install -r scripts/training/requirements.txt

# Verify MPS is available
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Download and preprocess datasets
python scripts/training/datasets/download_jamendo.py --output-dir data/jamendo
python scripts/training/datasets/download_deam.py --output-dir data/deam

# Train mood tagger
python scripts/training/train_mood.py --data-dir data/jamendo --epochs 50

# Train valence-arousal model
python scripts/training/train_va.py --data-dir data/deam --epochs 80

# Export to ONNX
python scripts/training/export_onnx.py \
    --mood-checkpoint models/checkpoints/mood_tagger_best.pt \
    --va-checkpoint models/checkpoints/va_best.pt \
    --output-dir models/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run the test suite:
   ```bash
   # Local
   cargo test --workspace --lib

   # Or via Docker
   docker build -t rage . && docker run rage

   # Parity tests (requires Docker — compares against librosa)
   docker build -f Dockerfile.parity -t rage-parity . && docker run rage-parity
   ```
5. Submit a pull request

### Code style

- Rust: `cargo fmt` and `cargo clippy -D warnings`
- Python: follow existing conventions in `scripts/`

## License

The RAGE source code is licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

### Dependency licenses

- **Symphonia** (audio decoding): MPL-2.0 (file-level copyleft, does not affect RAGE's license)
- All other Rust dependencies: MIT and/or Apache-2.0

### Model weight licenses

- **MTG-Jamendo mood tagger weights**: Subject to MTG-Jamendo dataset terms (CC BY-NC-SA 4.0 for audio)
- **DEAM valence-arousal weights**: Subject to DEAM dataset terms (CC BY-NC 4.0)

Pre-trained model weights are included in the repository under `models/`.

---

[devland.cc/rage](https://devland.cc/rage)
