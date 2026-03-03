"""Custom loss functions for RAGE training."""

import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient loss.

    CCC measures agreement between predicted and ground truth values.
    Loss = 1 - CCC (minimized during training).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)

        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        ccc = (2.0 * covariance) / (
            pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
        )

        return 1.0 - ccc


class CombinedVALoss(nn.Module):
    """Combined CCC loss for valence and arousal."""

    def __init__(self, valence_weight: float = 0.5, arousal_weight: float = 0.5):
        super().__init__()
        self.ccc_loss = CCCLoss()
        self.v_weight = valence_weight
        self.a_weight = arousal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [batch, 2] -> (valence, arousal)
            target: [batch, 2] -> (valence, arousal)
        """
        v_loss = self.ccc_loss(pred[:, 0], target[:, 0])
        a_loss = self.ccc_loss(pred[:, 1], target[:, 1])
        return self.v_weight * v_loss + self.a_weight * a_loss
