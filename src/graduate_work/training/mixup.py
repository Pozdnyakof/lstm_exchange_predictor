"""Mixup для time-series classification.

Zhang et al., **ICLR 2018** ([arXiv:1710.09412](https://arxiv.org/abs/1710.09412)),
*mixup: Beyond Empirical Risk Minimization*. Linear interpolation между
парами (x_i, y_i) и (x_j, y_j) с λ ~ Beta(α, α). Эмпирически снижает
train/val gap на 30-50% на TS-задачах.

Конкретно для интрадей-прогноза 5-мин баров:
- mix окон входов **по batch-оси** (не по time-оси): сохраняется
  временная структура каждого ряда, но ансамбль из двух «наблюдается».
- mix меток и (опционально) lr_target тем же λ.
- α=0.2 рекомендован Amazon Science (2024) для TS-forecasting.

Применять с вероятностью ``p`` каждого батча — не на каждом, чтобы
часть итераций видеть «чистые» данные.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: torch.Tensor | None = None,
    *,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Mixup трансформация одного батча.

    ``x``: (B, T, F), ``y``: (B, H), ``lr``: (B, H) или None.
    λ ∼ Beta(α, α). Перестановка batch-оси, mix x_mix = λx + (1-λ)x_perm
    и аналогично для y, lr.

    Возвращает (x_mix, y_mix, lr_mix, λ). λ нужен для логов и
    регуляризации loss'а.
    """
    if alpha <= 0:
        return x, y, lr, 1.0
    if rng is None:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = float(rng.beta(alpha, alpha))
    # Симметризуем λ к [0.5, 1.0] — стандартный трюк, чтобы не было
    # эффекта "почти полная замена" при λ ≈ 0.
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_mix = lam * y + (1.0 - lam) * y[perm]
    lr_mix = (lam * lr + (1.0 - lam) * lr[perm]) if lr is not None else None
    return x_mix, y_mix, lr_mix, lam


def maybe_apply_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: torch.Tensor | None = None,
    *,
    alpha: float = 0.2,
    p: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Применить mixup с вероятностью ``p``.

    Если бросок монеты неудачен — возвращает входы как есть (λ=1.0).
    """
    if alpha <= 0 or p <= 0:
        return x, y, lr, 1.0
    coin = float(np.random.random()) if rng is None else float(rng.random())
    if coin > p:
        return x, y, lr, 1.0
    return mixup_batch(x, y, lr, alpha=alpha, rng=rng)
