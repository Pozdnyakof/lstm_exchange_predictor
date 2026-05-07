"""Линейные baseline-модели для time-series forecasting.

Реализует две архитектуры:

* **VLinear** — vanilla Linear baseline из Zeng et al. (AAAI 2023,
  arXiv:2205.13504). Простейшая модель класса LTSF-Linear: одна
  ``nn.Linear(seq_len -> 1)`` поканально + голова, без какой-либо
  декомпозиции, нормализации (поверх RevIN) или mixing'а.

* **XLinear** — MLP-baseline с поддержкой экзогенных входов
  (arXiv:2601.09237, AAAI 2026). Глобальный токен формируется из
  эндогенных каналов (temporal MLP + variate-wise sigmoid gating),
  затем кросс-MLP-блок mixит его с pooled-репрезентацией экзогенных
  каналов (если они есть). Когда ``timexer_n_exo == 0``, вырождается
  в чистую endo-only вариацию без cross-mixing.

Обе модели возвращают единый тензор ``(B, num_horizons)``: для
classification — логиты, для regression — прогноз.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout
from .revin import RevIN


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


class VLinear(nn.Module):
    """Vanilla Linear baseline — простейший вариант LTSF-Linear.

    Per-channel ``Linear(seq_len -> 1)`` поверх транспонированного входа,
    дальше голова на ``num_horizons``. Никакой декомпозиции и тождеств
    last-value. Опционально RevIN на входе.
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
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )
        # Per-channel temporal projection (одно ядро для всех каналов).
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
        # (B, T, F) -> (B, F, T) -> Linear(T->1) -> (B, F, 1) -> squeeze.
        z = self.proj(x.transpose(1, 2)).squeeze(-1)  # (B, F)
        return self.head(z)


class XLinear(nn.Module):
    """XLinear (arXiv:2601.09237) — MLP-forecaster с exo-входами.

    Архитектурный скетч (по описанию paper'а — он бьёт TimeXer/TimeMixer
    на 89% MSE-конфигураций):

    1. **Endo-ветвь**: per-channel temporal MLP ``Linear(T -> d) -> GELU
       -> Dropout`` собирает per-channel embedding.
    2. **Variate gating**: sigmoid-gating вычисляется из mean'а
       endo-каналов по времени, поэлементно умножается на embedding.
       Это и есть «sigmoid-activated MLP, extracting variate-wise
       dependencies».
    3. **Global token**: усреднение по каналам -> ``(B, d)``.
    4. **Exo-ветвь** (если ``n_exo > 0``): аналогичный temporal MLP +
       pooling, далее cross-MLP смешивает (global, exo_pooled) -> d.
    5. **Голова**: Linear -> GELU -> MCDropout -> Linear(num_horizons).

    Когда ``timexer_n_exo == 0`` (наш дефолт), exo-ветвь отключается
    и модель работает только с эндогенным входом.
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        n_exo = int(cfg.timexer_n_exo)
        n_endo = input_dim - n_exo
        if n_endo <= 0:
            msg = (
                f"timexer_n_exo={n_exo} >= input_dim={input_dim};"
                " не осталось эндогенных каналов"
            )
            raise ValueError(msg)
        seq = int(cfg.linear_seq_len)
        if seq <= 0:
            msg = f"linear_seq_len must be positive, got {seq}"
            raise ValueError(msg)
        d = int(cfg.fc_hidden)
        dropout = float(cfg.dropout)

        self.n_endo = n_endo
        self.n_exo = n_exo
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )

        # Endo: per-channel temporal MLP (T -> d).
        self.endo_temporal = nn.Sequential(
            nn.Linear(seq, d),
            nn.GELU(),
            MonteCarloDropout(dropout),
        )
        # Variate gating: sigmoid(MLP(mean_по_времени)) -> вес на канал.
        self.endo_gate = nn.Sequential(
            nn.Linear(n_endo, n_endo),
            nn.Sigmoid(),
        )

        if n_exo > 0:
            self.exo_temporal: nn.Module | None = nn.Sequential(
                nn.Linear(seq, d),
                nn.GELU(),
                MonteCarloDropout(dropout),
            )
            # Cross-mixing: (global || exo_pool) -> d.
            self.cross_mlp: nn.Module | None = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.GELU(),
                MonteCarloDropout(dropout),
            )
        else:
            self.exo_temporal = None
            self.cross_mlp = None

        self.head = _LinearHead(
            in_features=d,
            num_horizons=num_horizons,
            hidden=int(cfg.fc_hidden),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x)

        # ----- Endo ветвь -----
        x_endo = x[:, :, : self.n_endo]                           # (B, T, n_endo)
        # Per-channel temporal: (B, n_endo, T) -> (B, n_endo, d).
        endo_emb = self.endo_temporal(x_endo.transpose(1, 2))
        # Variate gating: sigmoid из mean-по-времени.
        gate = self.endo_gate(x_endo.mean(dim=1))                  # (B, n_endo)
        endo_emb = endo_emb * gate.unsqueeze(-1)                  # (B, n_endo, d)
        # Global endo token = mean по каналам.
        global_tok = endo_emb.mean(dim=1)                          # (B, d)

        # ----- Exo ветвь (опционально) -----
        if (
            self.exo_temporal is not None
            and self.cross_mlp is not None
            and self.n_exo > 0
        ):
            x_exo = x[:, :, self.n_endo:]                          # (B, T, n_exo)
            exo_emb = self.exo_temporal(x_exo.transpose(1, 2))     # (B, n_exo, d)
            exo_pool = exo_emb.mean(dim=1)                         # (B, d)
            combined = self.cross_mlp(
                torch.cat([global_tok, exo_pool], dim=-1),
            )
        else:
            combined = global_tok

        return self.head(combined)
