"""PyTorch Dataset classes for RAGE training."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MOOD_TAGS


class JamendoMoodDataset(Dataset):
    """MTG-Jamendo mood/theme dataset.

    Each sample:
      - mel: [1, 128, 1292] log-mel spectrogram
      - labels: [56] binary multi-label vector

    Split file format (TSV): track_id \\t npy_filename \\t tag1,tag2,...
    """

    def __init__(self, split_file: str, mel_dir: str, transform=None):
        self.mel_dir = Path(mel_dir)
        self.transform = transform
        self.samples = []

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                _track_id, npy_name, tag_str = parts[0], parts[1], parts[2]
                npy_path = self.mel_dir / npy_name

                tags = set(tag_str.split(","))
                label = np.zeros(len(MOOD_TAGS), dtype=np.float32)
                for i, tag in enumerate(MOOD_TAGS):
                    if tag in tags:
                        label[i] = 1.0

                self.samples.append((npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]

        mel = np.load(npy_path)  # [128, 1292]
        mel = torch.from_numpy(mel).unsqueeze(0)  # [1, 128, 1292]

        if self.transform:
            mel = self.transform(mel)

        label = torch.from_numpy(label)
        return mel, label


class DEAMDataset(Dataset):
    """DEAM valence-arousal dataset.

    Each sample:
      - mel: [1, 128, 1292] log-mel spectrogram
      - summary: [294] summary feature vector
      - va: [2] (valence, arousal) in [-1, 1]

    Split file format (TSV): track_id \\t valence \\t arousal \\t npy_filename
    """

    def __init__(self, split_file: str, features_dir: str, transform=None):
        self.features_dir = Path(features_dir)
        self.transform = transform
        self.samples = []

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                track_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[2])
                npy_name = parts[3]
                self.samples.append((track_id, npy_name, valence, arousal))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _track_id, npy_name, valence, arousal = self.samples[idx]

        data = np.load(
            self.features_dir / npy_name, allow_pickle=True
        ).item()
        mel = torch.from_numpy(data["log_mel"]).unsqueeze(0)  # [1, 128, 1292]
        summary = torch.from_numpy(data["summary"])  # [294]

        if self.transform:
            mel = self.transform(mel)

        va = torch.tensor([valence, arousal], dtype=torch.float32)
        return mel, summary, va
