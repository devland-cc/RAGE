"""
MoodTagger: 5-layer CNN for multi-label mood/theme classification.

Input:  [batch, 1, 128, 1292]  (log-mel spectrogram)
Output: [batch, 56]            (logits for 56 mood/theme tags)

Architecture: ~2.3M parameters
"""

import torch
import torch.nn as nn


class MoodTagger(nn.Module):
    def __init__(self, num_tags: int = 56):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: [1, 128, 1292] -> [64, 64, 646]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: [64, 64, 646] -> [128, 32, 323]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: [128, 32, 323] -> [256, 16, 161]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: [256, 16, 161] -> [256, 8, 40]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            # Block 5: [256, 8, 40] -> [512, 1, 1] via adaptive pool
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, 128, 1292] log-mel spectrogram
        Returns:
            logits: [batch, 56] raw logits (apply sigmoid for probabilities)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
