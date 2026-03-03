"""SpecAugment: frequency and time masking for spectrograms."""

import random

import torch


class SpecAugment:
    """Apply SpecAugment to log-mel spectrograms during training.

    Reference: Park et al., "SpecAugment", 2019.
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 80,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [1, n_mels, n_frames] or [n_mels, n_frames]
        Returns:
            Augmented mel spectrogram (same shape).
        """
        mel = mel.clone()

        if mel.dim() == 3:
            _, n_mels, n_frames = mel.shape
        else:
            n_mels, n_frames = mel.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            if mel.dim() == 3:
                mel[:, f0 : f0 + f, :] = self.mask_value
            else:
                mel[f0 : f0 + f, :] = self.mask_value

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, n_frames - t))
            if mel.dim() == 3:
                mel[:, :, t0 : t0 + t] = self.mask_value
            else:
                mel[:, t0 : t0 + t] = self.mask_value

        return mel
