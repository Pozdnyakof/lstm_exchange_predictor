"""Dropout-слой, остающийся активным во время инференса.

Базируется на работах Гала и Гарамани (Dropout as a Bayesian Approximation,
2016): прореживание сохраняется при прямом проходе и аппроксимирует
байесовскую неопределённость через стохастические сэмплы.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MonteCarloDropout(nn.Dropout):
    """nn.Dropout с принудительным включением через флаг ``mc_mode``."""

    def __init__(self, p: float = 0.3) -> None:
        super().__init__(p=p)
        self.mc_mode: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.mc_mode
        return F.dropout(x, p=self.p, training=active, inplace=False)


def set_mc_dropout(module: nn.Module, enabled: bool) -> None:
    """Включить или выключить MC-режим у всех ``MonteCarloDropout`` в модели."""
    for child in module.modules():
        if isinstance(child, MonteCarloDropout):
            child.mc_mode = enabled
