"""DLinear и NLinear baseline-модели (Zeng et al., AAAI 2023).

Обе модели идут с paper «Are Transformers Effective for Time Series
Forecasting?» (arXiv:2205.13504). Линейная сеть на разложенных по
trend/seasonal-каналам ряд + per-instance нормализация — простые но
сильные baseline'ы, которые регулярно обыгрывают трансформеры на
коротких многомерных временных рядах.

Интерфейс одинаков с TimeXer и ConvLstmRegressor:
    forward: (B, T, F) -> (B, num_horizons)
classification: возвращает логиты; regression: значение прогноза.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout
from .revin import RevIN


class _MovingAvg(nn.Module):
    """Скользящее среднее для извлечения тренда."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        pad = (self.kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> avg по T -> (B, T, F)
        return self.avg(x.transpose(1, 2)).transpose(1, 2)


class _SeriesDecomp(nn.Module):
    """Разложение ряд: (residual, trend)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


class _LinearHead(nn.Module):
    """Голова: Linear -> GELU -> MCDropout -> Linear(num_horizons)."""

    def __init__(self, in_features: int, num_horizons: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            MonteCarloDropout(dropout),
            nn.Linear(hidden, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DLinear(nn.Module):
    """Decomposition-Linear baseline.

    Раскладывает входной ряд на seasonal+trend (через скользящее
    среднее), пропускает каждый компонент через свою линейку
    ``Linear(seq_len -> 1)`` поканально, складывает и пускает в
    голову, выдающую ``num_horizons`` логитов.
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        k = int(cfg.linear_kernel_size)
        seq = int(cfg.linear_seq_len)
        if k % 2 == 0:
            msg = f"linear_kernel_size must be odd, got {k}"
            raise ValueError(msg)
        if seq <= 0:
            msg = f"linear_seq_len must be positive, got {seq}"
            raise ValueError(msg)

        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )
        self.decomp = _SeriesDecomp(k)
        # Per-channel: Linear(seq, 1). Применяется после транспонирования
        # — k features рассматриваются параллельно по batch.
        self.linear_seasonal = nn.Linear(seq, 1)
        self.linear_trend = nn.Linear(seq, 1)
        self.head = _LinearHead(
            in_features=input_dim,
            num_horizons=num_horizons,
            hidden=int(cfg.fc_hidden),
            dropout=float(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x)
        seasonal, trend = self.decomp(x)
        # (B, T, F) -> transpose -> (B, F, T) -> Linear(T->1) -> (B, F, 1) -> squeeze.
        s_out = self.linear_seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        t_out = self.linear_trend(trend.transpose(1, 2)).squeeze(-1)
        combined = s_out + t_out  # (B, F)
        return self.head(combined)


class NLinear(nn.Module):
    """Normalisation-Linear baseline.

    Простейший трюк: вычесть последнее значение каждого канала, прогнать
    линейку, прибавить обратно. Эквивалент per-instance normalization
    (без масштабирования) + единственная линейная проекция. Несмотря на
    простоту, на многих TSF-бенчмарках NLinear превосходит ChiselTime/
    DLinear когда тест-распределение смещено относительно train.
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        seq = int(cfg.linear_seq_len)
        if seq <= 0:
            msg = f"linear_seq_len must be positive, got {seq}"
            raise ValueError(msg)
        # Опциональный RevIN — поверх трюка с last-value не лишний.
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )
        self.proj = nn.Linear(seq, 1)
        self.head = _LinearHead(
            in_features=input_dim,
            num_horizons=num_horizons,
            hidden=int(cfg.fc_hidden),
            dropout=float(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x)
        # NLinear-трюк: subtract last value per (batch, feature).
        last = x[:, -1:, :]            # (B, 1, F)
        x_centered = x - last          # (B, T, F)
        # Линейная проекция T->1 поканально.
        z = self.proj(x_centered.transpose(1, 2)).squeeze(-1)  # (B, F)
        # Прибавляем обратно last value (broadcast по F).
        z = z + last.squeeze(1)
        return self.head(z)
