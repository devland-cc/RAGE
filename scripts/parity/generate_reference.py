"""
Generate reference audio features using librosa for parity testing.

This script creates test audio signals and computes features with librosa
using the exact same parameters as rage-extractor. The outputs are saved
as raw f32 little-endian binary files that Rust integration tests load
for comparison.

Parameters (must match rage-core ExtractionConfig defaults):
  sample_rate = 22050
  n_fft       = 2048
  hop_length  = 512
  n_mels      = 128
  fmin        = 20.0
  fmax        = 11025.0
"""

import json
import struct
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Match rage-core ExtractionConfig defaults exactly
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20.0
FMAX = 11025.0

OUTPUT_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "parity"


def save_f32_bin(arr: np.ndarray, path: Path, name: str):
    """Save a numpy array as raw f32 little-endian binary + shape metadata."""
    data = arr.astype(np.float32)
    bin_path = path / f"{name}.bin"
    meta_path = path / f"{name}.json"

    with open(bin_path, "wb") as f:
        f.write(data.tobytes())

    with open(meta_path, "w") as f:
        json.dump({"shape": list(data.shape), "dtype": "f32"}, f)

    print(f"  Saved {name}: shape={list(data.shape)}, {bin_path.stat().st_size} bytes")


def generate_sine_wave(freq: float, duration: float, sr: int) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def generate_chirp(f0: float, f1: float, duration: float, sr: int) -> np.ndarray:
    """Generate a linear chirp from f0 to f1 Hz."""
    t = np.arange(int(sr * duration)) / sr
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    return np.sin(phase).astype(np.float32)


def compute_and_save(signal: np.ndarray, name: str, output_dir: Path):
    """Compute all features with librosa and save reference data."""
    print(f"\n--- {name} ---")
    print(f"  Signal: {len(signal)} samples, {len(signal)/SR:.2f}s")

    # Save the raw audio signal
    save_f32_bin(signal, output_dir, f"{name}_audio")

    # Also save as WAV for the Rust decoder test
    wav_path = output_dir / f"{name}.wav"
    sf.write(str(wav_path), signal, SR, subtype="FLOAT")
    print(f"  Saved {name}.wav: {wav_path.stat().st_size} bytes")

    # 1. STFT magnitude (center=True is librosa's default, matching our Rust impl)
    stft = librosa.stft(
        signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True
    )
    stft_mag = np.abs(stft)
    save_f32_bin(stft_mag, output_dir, f"{name}_stft_magnitude")

    # 2. Power spectrum
    power = stft_mag**2
    save_f32_bin(power, output_dir, f"{name}_power_spectrum")

    # 3. Mel filterbank
    mel_fb = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True, norm="slaney"
    )
    save_f32_bin(mel_fb, output_dir, f"{name}_mel_filterbank")

    # 4. Mel spectrogram (from power spectrum)
    mel_spec = mel_fb @ power
    save_f32_bin(mel_spec, output_dir, f"{name}_mel_spectrogram")

    # 5. Log-mel spectrogram (power_to_db)
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)
    save_f32_bin(log_mel, output_dir, f"{name}_log_mel_spectrogram")

    # Print some stats for manual verification
    print(f"  STFT mag: shape={stft_mag.shape}, min={stft_mag.min():.6f}, max={stft_mag.max():.6f}")
    print(f"  Mel spec: shape={mel_spec.shape}, min={mel_spec.min():.6f}, max={mel_spec.max():.6f}")
    print(f"  Log mel:  shape={log_mel.shape}, min={log_mel.min():.2f}, max={log_mel.max():.2f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parameters: sr={SR}, n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}, fmin={FMIN}, fmax={FMAX}")

    # Test 1: Pure 440 Hz sine wave (1 second)
    sine_440 = generate_sine_wave(440.0, 1.0, SR)
    compute_and_save(sine_440, "sine_440", OUTPUT_DIR)

    # Test 2: Linear chirp from 100 Hz to 4000 Hz (2 seconds)
    # This exercises a wide frequency range across many mel bins
    chirp = generate_chirp(100.0, 4000.0, 2.0, SR)
    compute_and_save(chirp, "chirp", OUTPUT_DIR)

    # Test 3: Silence (edge case)
    silence = np.zeros(SR, dtype=np.float32)
    compute_and_save(silence, "silence", OUTPUT_DIR)

    print("\n=== Reference generation complete ===")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
