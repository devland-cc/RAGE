"""
Preprocess MTG-Jamendo: compute log-mel spectrograms matching RAGE parameters.

For each track:
1. Load MP3 with librosa at sr=22050
2. Take center 30 seconds (or zero-pad if shorter)
3. Compute STFT -> mel -> log-mel with exact RAGE parameters
4. Save as .npy file [128, 1292]
5. Create split files with binary label vectors

CRITICAL: Must use htk=True, norm='slaney' for mel filterbank.
These match rage-extractor/src/mel.rs exactly.

Usage:
    python scripts/training/datasets/preprocess_jamendo.py \
        --audio-dir data/jamendo/raw \
        --output-dir data/jamendo/mel_spectrograms \
        --splits-dir data/jamendo/splits \
        --num-workers 8
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    MOOD_TAGS,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)


def compute_log_mel(audio_path: str) -> np.ndarray | None:
    """Compute log-mel spectrogram matching rage-extractor exactly.

    Returns: np.ndarray of shape [128, 1292], or None on failure.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  Failed to load {audio_path}: {e}", file=sys.stderr)
        return None

    # Center crop to 30 seconds (or zero-pad if shorter)
    target_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]
    elif len(y) < target_samples:
        pad_total = target_samples - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    # STFT (center=True matches stft.rs)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True)
    power = np.abs(S) ** 2

    # Mel filterbank (HTK + Slaney matches mel.rs)
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

    # Log scale (matches mel::power_to_db in mel.rs)
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

    # Truncate or pad time axis to exactly N_FRAMES
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

    return log_mel.astype(np.float32)


def process_track(args):
    """Process a single track (for multiprocessing)."""
    audio_path, output_path = args
    if output_path.exists():
        return True

    log_mel = compute_log_mel(str(audio_path))
    if log_mel is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, log_mel)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MTG-Jamendo: compute log-mel spectrograms"
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = sorted(
        list(audio_dir.glob("**/*.mp3"))
        + list(audio_dir.glob("**/*.wav"))
        + list(audio_dir.glob("**/*.flac"))
    )
    print(f"Found {len(audio_files)} audio files")

    # Build processing tasks
    tasks = []
    for audio_path in audio_files:
        track_id = audio_path.stem
        output_path = output_dir / f"{track_id}.npy"
        tasks.append((audio_path, output_path))

    # Process with multiprocessing
    success = 0
    with Pool(args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_track, tasks),
            total=len(tasks),
            desc="Computing mel-spectrograms",
        ):
            if result:
                success += 1

    print(f"\nProcessed {success}/{len(tasks)} tracks successfully")

    # Rewrite split files with npy filenames
    for split_file in splits_dir.glob("*.tsv"):
        rewrite_split_with_npy(split_file, output_dir)


def rewrite_split_with_npy(split_file: Path, mel_dir: Path):
    """Rewrite a split file to include npy filenames and normalized tags."""
    output_lines = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            track_id = parts[0]
            tags = [
                t.split("---")[1] if "---" in t else t for t in parts[1:]
            ]
            # Filter to only mood/theme tags
            valid_tags = [t for t in tags if t in MOOD_TAGS]
            if not valid_tags:
                continue

            npy_name = f"{track_id}.npy"
            npy_path = mel_dir / npy_name
            if not npy_path.exists():
                continue

            output_lines.append(
                f"{track_id}\t{npy_name}\t{','.join(valid_tags)}"
            )

    # Write processed split
    processed_path = split_file.parent / f"{split_file.stem}_processed.tsv"
    with open(processed_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"  Wrote {processed_path}: {len(output_lines)} samples")


if __name__ == "__main__":
    main()
