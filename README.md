# RAGE - Rust Aura-Gathering Engine

A lightweight, pure-Rust system for analyzing the mood and emotion of music. RAGE extracts audio features and classifies mood/theme tags and valence-arousal values from audio files.

> **Status**: Fully functional — clone, build, and analyze music out of the box. Pre-trained ONNX models are included in the repo.

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

### Models (Training)

Two PyTorch models trained on Apple Silicon (MPS), exported to ONNX:

- **MoodTagger**: 5-layer CNN for 56 mood/theme tags (MTG-Jamendo dataset)
- **ValenceArousalModel**: Dual-branch CNN+MLP for continuous valence-arousal prediction (DEAM dataset)

## Prerequisites

- **Rust toolchain** (1.75+ recommended)
- **Python 3.11+** with conda (only needed for model retraining)

## Quick Start

```bash
# Clone and build
git clone https://github.com/devland-cc/RAGE.git
cd RAGE
cargo build --release
```

### Analyze music

```bash
# Analyze a song (table output)
cargo run --release -- analyze song.mp3

# JSON output
cargo run --release -- analyze --output json song.mp3

# Multiple files
cargo run --release -- analyze song1.mp3 song2.flac song3.wav

# Custom model directory and top-k tags
cargo run --release -- analyze song.mp3 --model-dir models/ --top-k 15
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

### Feature extraction (CLI)

```bash
# Table output
cargo run --release -- extract song.mp3

# JSON output
cargo run --release -- extract --output json song.mp3
```

### Train models (Mac with Apple Silicon)

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
4. Run the test suite: `docker build -t rage . && docker run rage`
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
