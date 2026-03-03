"""
ValenceArousalModel: Dual-branch architecture for continuous V-A prediction.

Branch 1 (CNN): log-mel spectrogram [1, 128, 1292] -> 256-dim embedding
Branch 2 (MLP): summary vector [294] -> 128-dim embedding

Fusion: concat(256, 128) = 384 -> FC layers -> 2 outputs (valence, arousal)
Output range: [-1, 1] via tanh activation.
~2.6M parameters total.
"""

import torch
import torch.nn as nn


class SpectrogramBranch(nn.Module):
    """CNN branch processing log-mel spectrograms."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: [1, 128, 1292] -> [64, 64, 646]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: [64, 64, 646] -> [128, 32, 323]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: [128, 32, 323] -> [256, 16, 161]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: [256, 16, 161] -> [256, 8, 40]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 4), (2, 4)),
            # Block 5: [256, 8, 40] -> [512, 1, 1]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embed(x)
        return x


class FeatureBranch(nn.Module):
    """MLP branch processing the 294-dim summary vector."""

    def __init__(self, input_dim: int = 294, embed_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ValenceArousalModel(nn.Module):
    """Dual-branch model: CNN on spectrogram + MLP on summary features."""

    def __init__(
        self,
        cnn_embed_dim: int = 256,
        mlp_embed_dim: int = 128,
        summary_dim: int = 294,
    ):
        super().__init__()

        self.cnn_branch = SpectrogramBranch(embed_dim=cnn_embed_dim)
        self.mlp_branch = FeatureBranch(input_dim=summary_dim, embed_dim=mlp_embed_dim)

        fusion_dim = cnn_embed_dim + mlp_embed_dim  # 384

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # valence, arousal
            nn.Tanh(),  # constrain to [-1, 1]
        )

    def forward(
        self,
        mel: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel:     [batch, 1, 128, 1292] log-mel spectrogram
            summary: [batch, 294] summary feature vector
        Returns:
            va: [batch, 2] -> (valence, arousal) in [-1, 1]
        """
        cnn_emb = self.cnn_branch(mel)
        mlp_emb = self.mlp_branch(summary)
        fused = torch.cat([cnn_emb, mlp_emb], dim=1)
        return self.head(fused)
