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
from .revin import RevIN


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

        # RevIN - адаптивная per-instance нормализация поверх StandardScaler.
        # Снимает distribution shift между обучением и инференсом.
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cfg.conv_channels,
            kernel_size=cfg.conv_kernel,
            padding=0,
        )
        self.conv_act = nn.GELU()
        self.conv_drop = MonteCarloDropout(cfg.dropout)

        # Стек 1-слойных LSTM с явным MC Dropout между ними.
        # Это нужно потому, что встроенный nn.LSTM(dropout=p) активен
        # ТОЛЬКО при self.training=True - после model.eval() он
        # отключается, и MC Dropout инференс теряет основной источник
        # стохастичности. Разбивая на отдельные блоки, мы делаем
        # межслойный дропаут управляемым через MonteCarloDropout.
        self.lstm_layers = nn.ModuleList()
        self.lstm_dropouts = nn.ModuleList()
        in_dim = cfg.conv_channels
        for _ in range(cfg.lstm_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=cfg.lstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,  # отключаем встроенный, ставим явный MC Dropout
                ),
            )
            self.lstm_dropouts.append(MonteCarloDropout(cfg.dropout))
            in_dim = cfg.lstm_hidden
        # Финальный dropout перед головой - перенесён сюда (раньше был
        # отдельный self.lstm_drop).
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.fc_hidden),
            nn.GELU(),
            MonteCarloDropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, num_horizons)."""
        if self.revin is not None:
            x = self.revin(x)             # per-instance normalization (B, T, F)
        x = x.transpose(1, 2)            # (B, F, T)
        if self.causal_pad > 0:
            x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)
        x = self.conv_act(x)
        x = self.conv_drop(x)
        x = x.transpose(1, 2)            # (B, T, C)

        # Прогон через стек LSTM с MC-дропаутами между слоями.
        for lstm, drop in zip(self.lstm_layers, self.lstm_dropouts):
            x, _ = lstm(x)
            x = drop(x)
        last = x[:, -1, :]
        return self.head(last)
