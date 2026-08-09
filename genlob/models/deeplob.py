"""DeepLOB-style benchmark classifier for FI-2010.

This module is intentionally separate from the GenLOB diffusion models. FI-2010
is a normalized prediction benchmark, so this classifier is used only for
DeepLOB-style public benchmark comparability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOBStyleClassifier(nn.Module):
    """Compact DeepLOB-style CNN + temporal LSTM classifier.

    Args:
        input_dim: Number of FI-2010 feature columns per event. The common
            DeepLOB setting uses 40 LOB features.
        num_classes: Number of official FI-2010 classes.
        conv_channels: Channels used by the feature compression stem.
        inception_channels: Channels per temporal inception branch.
        lstm_hidden: LSTM hidden size.
        dropout: Dropout before the final classifier.
    """

    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 3,
        conv_channels: int = 32,
        inception_channels: int = 32,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        self.feature_stem = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(conv_channels),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(conv_channels),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(conv_channels),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(inception_channels, inception_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv2d(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(inception_channels, inception_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        lstm_input = inception_channels * 3
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for ``x`` shaped ``[batch, window, features]``."""
        if x.ndim != 3:
            raise ValueError(f"expected [batch, window, features], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")

        h = x.unsqueeze(1)
        h = self.feature_stem(h)
        h = torch.cat([self.branch_3(h), self.branch_5(h), self.branch_pool(h)], dim=1)
        h = h.squeeze(-1).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.classifier(h[:, -1])

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
