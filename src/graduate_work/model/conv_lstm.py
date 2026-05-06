"""Гибридная архитектура 1D-CNN (causal) → LSTM → линейная голова.

Соответствует обоснованию §2.1 ВКР: причинная свёртка обеспечивает
строгую хронологическую каузальность, рекуррентный блок моделирует
долгосрочные зависимости, а голова выдаёт мульти-горизонтный
регрессионный прогноз.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout


class ConvLstmRegressor(nn.Module):
    """Сеть из §2.2 ВКР.

    Каузальная свёртка реализована левосторонним паддингом ``kernel - 1``.
    Слои Dropout - наследники ``MonteCarloDropout``, их состояние
    управляется функцией :func:`set_mc_dropout` без модификации сети.
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.causal_pad = max(cfg.conv_kernel - 1, 0)

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cfg.conv_channels,
            kernel_size=cfg.conv_kernel,
            padding=0,
        )
        self.conv_act = nn.GELU()
        self.conv_drop = MonteCarloDropout(cfg.dropout)

        self.lstm = nn.LSTM(
            input_size=cfg.conv_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.lstm_drop = MonteCarloDropout(cfg.dropout)

        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.fc_hidden),
            nn.GELU(),
            MonteCarloDropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, num_horizons)."""
        x = x.transpose(1, 2)            # (B, F, T)
        if self.causal_pad > 0:
            x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)
        x = self.conv_act(x)
        x = self.conv_drop(x)
        x = x.transpose(1, 2)            # (B, T, C)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        last = self.lstm_drop(last)
        return self.head(last)
