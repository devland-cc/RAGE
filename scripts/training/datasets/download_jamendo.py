"""
Download MTG-Jamendo mood/theme subset.

Downloads:
  1. Annotation TSVs from the MTG-Jamendo GitHub repository
  2. Raw audio (MP3) via the Jamendo API or dataset mirrors

Usage:
    python scripts/training/datasets/download_jamendo.py --output-dir data/jamendo
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Annotation URLs from the MTG-Jamendo GitHub repo
ANNOTATION_BASE = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset"
    "/master/data/splits/split-0"
)
SPLITS = {
    "train": f"{ANNOTATION_BASE}/autotagging_moodtheme-train.tsv",
    "val": f"{ANNOTATION_BASE}/autotagging_moodtheme-validation.tsv",
    "test": f"{ANNOTATION_BASE}/autotagging_moodtheme-test.tsv",
}

# Tag mapping file
TAG_MAP_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset"
    "/master/data/tags/moodtheme.txt"
)


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress bar, skipping if already exists."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_annotations(output_dir: Path):
    """Download split TSV files and tag mapping."""
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading annotation splits...")
    for split_name, url in SPLITS.items():
        dest = splits_dir / f"{split_name}.tsv"
        download_file(url, dest, desc=f"{split_name}.tsv")

    print("Downloading tag mapping...")
    download_file(TAG_MAP_URL, output_dir / "moodtheme_tags.txt", desc="tags")


def parse_split_file(split_path: Path) -> list[dict]:
    """Parse a MTG-Jamendo split TSV file.

    Format: TRACK_ID\tTAG1,TAG2,...
    Returns list of {"track_id": str, "tags": list[str]}
    """
    samples = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            track_id = parts[0]
            tags = [
                t.split("---")[1] if "---" in t else t
                for t in parts[1:]
            ]
            samples.append({"track_id": track_id, "tags": tags})
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Download MTG-Jamendo mood/theme dataset"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_annotations(output_dir)

    # Print summary
    for split_name in ["train", "val", "test"]:
        split_path = output_dir / "splits" / f"{split_name}.tsv"
        if split_path.exists():
            samples = parse_split_file(split_path)
            print(f"  {split_name}: {len(samples)} tracks")

    print(
        "\nAnnotations downloaded. To download audio files, use the "
        "MTG-Jamendo download script or Jamendo API.\n"
        "See: https://github.com/MTG/mtg-jamendo-dataset#downloading-the-dataset"
    )
    print(
        "\nAfter downloading audio, run:\n"
        f"  python scripts/training/datasets/preprocess_jamendo.py "
        f"--audio-dir {output_dir}/raw "
        f"--output-dir {output_dir}/mel_spectrograms "
        f"--splits-dir {output_dir}/splits"
    )


if __name__ == "__main__":
    main()
