"""
Preprocess DEAM: extract features matching RAGE pipeline.

For each track:
1. Load audio at sr=22050
2. Take center 30 seconds
3. Compute log-mel spectrogram [128, 1292]
4. Compute 294-dim summary vector (matching rage-extractor pipeline.rs)
5. Load V-A annotations, normalize to [-1, 1]
6. Save features + labels

The summary vector computation must match build_summary_vector() in
crates/rage-extractor/src/pipeline.rs EXACTLY.

Usage:
    python scripts/training/datasets/preprocess_deam.py \
        --audio-dir data/deam/raw \
        --output-dir data/deam/features \
        --splits-dir data/deam/splits
"""

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.fftpack import dct
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)


def compute_features(audio_path: str) -> dict | None:
    """Compute log-mel spectrogram and 294-dim summary vector.

    Returns dict with 'log_mel' [128, 1292] and 'summary' [294], or None.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  Failed to load {audio_path}: {e}", file=sys.stderr)
        return None

    # Center crop to 30 seconds
    target_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]
    elif len(y) < target_samples:
        pad_total = target_samples - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    # STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True)
    magnitude = np.abs(S)
    power = magnitude**2

    # Mel -> log-mel
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,
        norm="slaney",
    )
    mel_spec = mel_fb @ power
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

    # Pad/truncate time axis
    if log_mel.shape[1] > N_FRAMES:
        log_mel = log_mel[:, :N_FRAMES]
    elif log_mel.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=log_mel.min(),
        )

    # Summary vector (must match pipeline.rs:build_summary_vector)
    summary = compute_summary_vector(y, magnitude, power)

    return {
        "log_mel": log_mel.astype(np.float32),
        "summary": summary.astype(np.float32),
    }


def compute_summary_vector(
    y: np.ndarray, magnitude: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """Compute the 294-dim summary vector matching rage-extractor exactly.

    42 features x 7 statistics = 294 dimensions.
    Features: 20 MFCCs + 12 chroma + 1 centroid + 1 rolloff +
              6 spectral contrast bands (skip mean row) + 1 RMS + 1 ZCR
    Statistics: mean, std, min, max, median, skewness, kurtosis
    """
    # Mel -> log-mel for MFCC computation
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,
        norm="slaney",
    )
    mel_spec = mel_fb @ power
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

    # MFCCs from log-mel (DCT type 2, ortho norm) — matches mfcc.rs
    mfccs = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC, :]  # [20, T]

    # Chroma
    chroma = librosa.feature.chroma_stft(
        S=magnitude, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )  # [12, T]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]  # [T]

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85
    )[0]  # [T]

    # Spectral contrast — 6 bands + valley; keep only first 6 (skip mean row)
    contrast = librosa.feature.spectral_contrast(
        S=power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
    )  # [7, T]
    contrast = contrast[:6, :]  # [6, T]

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True)[
        0
    ]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True
    )[0]

    # Collect all 42 feature time series
    all_features = []
    for k in range(N_MFCC):
        all_features.append(mfccs[k, :])
    for c in range(12):
        all_features.append(chroma[c, :])
    all_features.append(centroid)
    all_features.append(rolloff)
    for b in range(6):
        all_features.append(contrast[b, :])
    all_features.append(rms)
    all_features.append(zcr)

    assert len(all_features) == 42, f"Expected 42 features, got {len(all_features)}"

    # Compute 7 statistics for each feature
    summary = []
    for feat in all_features:
        feat = feat.astype(np.float64)
        mean = np.mean(feat)
        std = np.std(feat, ddof=0)  # population std, matching pipeline.rs
        minimum = np.min(feat)
        maximum = np.max(feat)
        median = np.median(feat)

        if std > 1e-10:
            skewness = np.mean(((feat - mean) / std) ** 3)
            kurtosis = np.mean(((feat - mean) / std) ** 4) - 3.0  # excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        summary.extend([mean, std, minimum, maximum, median, skewness, kurtosis])

    assert len(summary) == 294, f"Expected 294 dims, got {len(summary)}"
    return np.array(summary, dtype=np.float32)


def load_deam_annotations(annotations_dir: Path) -> dict:
    """Load static V-A annotations from DEAM.

    Returns: {song_id: (valence, arousal)} normalized to [-1, 1].
    """
    annotations = {}

    # DEAM static annotations are in CSV: song_id, valence_mean, arousal_mean
    # The values are on a 1-9 scale, we normalize to [-1, 1]
    # Find all static annotation CSVs (there are two: songs 1-2000 and 2000-2058)
    static_files = sorted(annotations_dir.rglob("*static*averaged*songs*.csv"))

    if not static_files:
        print(f"Warning: static annotations not found in {annotations_dir}")
        return annotations

    for static_file in static_files:
        print(f"  Loading annotations from {static_file.name}")
        with open(static_file) as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                song_id = row.get("song_id", "").strip()
                valence = float(row.get("valence_mean", 5.0))
                arousal = float(row.get("arousal_mean", 5.0))

                # Normalize from [1, 9] to [-1, 1]
                valence_norm = (valence - 5.0) / 4.0
                arousal_norm = (arousal - 5.0) / 4.0

                # Clamp to [-1, 1]
                valence_norm = max(-1.0, min(1.0, valence_norm))
                arousal_norm = max(-1.0, min(1.0, arousal_norm))

                annotations[song_id] = (valence_norm, arousal_norm)

    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DEAM: extract features"
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annot_dir = audio_dir / "DEAM_Annotations" / "annotations"
    if not annot_dir.exists():
        # Try finding annotations recursively
        for candidate in audio_dir.rglob("annotations"):
            if candidate.is_dir():
                annot_dir = candidate
                break

    annotations = load_deam_annotations(annot_dir)
    print(f"Loaded {len(annotations)} V-A annotations")

    # Find audio files
    audio_files = sorted(
        list(audio_dir.rglob("*.mp3"))
        + list(audio_dir.rglob("*.wav"))
    )
    print(f"Found {len(audio_files)} audio files")

    # Process each track
    processed = []
    for audio_path in tqdm(audio_files, desc="Extracting features"):
        song_id = audio_path.stem
        if song_id not in annotations:
            continue

        npy_name = f"{song_id}.npy"
        npy_path = output_dir / npy_name

        if npy_path.exists():
            valence, arousal = annotations[song_id]
            processed.append((song_id, valence, arousal, npy_name))
            continue

        features = compute_features(str(audio_path))
        if features is None:
            continue

        np.save(npy_path, features, allow_pickle=True)
        valence, arousal = annotations[song_id]
        processed.append((song_id, valence, arousal, npy_name))

    print(f"\nProcessed {len(processed)} tracks with annotations")

    # Create train/val/test splits
    np.random.seed(42)
    indices = np.random.permutation(len(processed))
    n_test = int(len(processed) * args.test_ratio)
    n_val = int(len(processed) * args.val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    for split_name, split_idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        split_path = splits_dir / f"{split_name}.tsv"
        with open(split_path, "w") as f:
            for i in split_idx:
                song_id, valence, arousal, npy_name = processed[i]
                f.write(f"{song_id}\t{valence:.6f}\t{arousal:.6f}\t{npy_name}\n")
        print(f"  {split_name}: {len(split_idx)} samples -> {split_path}")


if __name__ == "__main__":
    main()
