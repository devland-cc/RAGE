"""
Download DEAM dataset (MediaEval 2018 Emotion in Music).

Source: https://cvml.unige.ch/databases/DEAM/

Contains:
  - 1802 45-second excerpts (MP3)
  - Continuous V-A annotations at 2Hz
  - Static (average) V-A annotations per song

Usage:
    python scripts/training/datasets/download_deam.py --output-dir data/deam
"""

import argparse
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# DEAM download URLs (Zenodo mirror)
DEAM_AUDIO_URL = (
    "https://zenodo.org/records/2616028/files/"
    "DEAM_audio.zip?download=1"
)
DEAM_ANNOTATIONS_URL = (
    "https://zenodo.org/records/2616028/files/"
    "DEAM_Annotations.zip?download=1"
)


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def main():
    parser = argparse.ArgumentParser(description="Download DEAM dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download audio
    audio_zip = raw_dir / "DEAM_audio.zip"
    print("Downloading DEAM audio (~1.4GB)...")
    download_file(DEAM_AUDIO_URL, audio_zip, desc="DEAM_audio.zip")

    # Download annotations
    annot_zip = raw_dir / "DEAM_Annotations.zip"
    print("Downloading DEAM annotations...")
    download_file(DEAM_ANNOTATIONS_URL, annot_zip, desc="DEAM_Annotations.zip")

    # Extract
    if not (raw_dir / "MEMD_audio").exists():
        extract_zip(audio_zip, raw_dir)
    if not (raw_dir / "DEAM_Annotations").exists():
        extract_zip(annot_zip, raw_dir)

    print(f"\nDEAM dataset downloaded to {raw_dir}")
    print(
        "\nNext step: preprocess features:\n"
        f"  python scripts/training/datasets/preprocess_deam.py "
        f"--audio-dir {raw_dir} "
        f"--output-dir {output_dir}/features "
        f"--splits-dir {output_dir}/splits"
    )


if __name__ == "__main__":
    main()
