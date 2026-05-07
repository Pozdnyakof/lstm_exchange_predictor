"""Целевые переменные для регрессии и классификации.

Регрессия: нормализованная лог-доходность по горизонту (старый путь).

Классификация: бинарная метка «прибыль ≥ 0 после round-trip-костов».
Метка построена согласно `experiment_03/labeling/multi_horizon.py`:
вход по open[t+1], выход по close[t+1+h], издержки включены прямо в
формулу через корректировку цен. Это даёт лейблу содержательное
толкование: «выгодно ли было бы открыть лонг на закрытии бара t».
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# REGRESSION (legacy)
# ---------------------------------------------------------------------------

def normalized_log_returns(
    close: pd.Series,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Нормализованная лог-доходность для регрессии (legacy).

    target[t, h] = (log_close[t+h] - log_close[t]) / h
    """
    out = pd.DataFrame(index=close.index)
    log_close = np.log(close.astype(float))
    for h in horizons:
        future = log_close.shift(-h) - log_close
        out[f"target_h{h}"] = future / float(h)
    return out


def target_columns(horizons: tuple[int, ...]) -> list[str]:
    """Имена целевых колонок (универсально для regression/classification)."""
    return [f"target_h{h}" for h in horizons]


def lr_columns(horizons: tuple[int, ...]) -> list[str]:
    """Имена сырых лог-доходностей с костами - для калибровки Bayes-порога.

    Эти колонки сохраняются ПАРАЛЛЕЛЬНО с target_h{h} в режиме
    классификации (target_h - сглаженные метки {eps, 1-eps}, lr_h - сырая
    величина выгоды без сглаживания).
    """
    return [f"lr_h{h}" for h in horizons]


# ---------------------------------------------------------------------------
# CLASSIFICATION (cost-aware binary labels)
# ---------------------------------------------------------------------------

def _log_return_with_costs(
    next_open: np.ndarray,
    future_close: np.ndarray,
    direction: str,
    entry_cost: float,
    exit_cost: float,
) -> np.ndarray:
    """Лог-доходность ОДНОЙ сделки с учётом round-trip-костов.

    Long: купили по open[t+1] × (1 + entry_cost), продали по
          close[t+h] × (1 - exit_cost).
    Short: симметрично.
    """
    eps = 1e-8
    if direction == "long":
        entry = next_open * (1.0 + entry_cost)
        exit_net = future_close * (1.0 - exit_cost)
        ratio = exit_net / np.maximum(entry, eps)
    else:
        entry = next_open * (1.0 - entry_cost)
        exit_net = future_close * (1.0 + exit_cost)
        ratio = np.maximum(entry, eps) / np.maximum(exit_net, eps)
    return np.log(np.clip(ratio, eps, None)).astype(np.float32)


def cost_aware_classification_labels(
    open_price: pd.Series,
    close_price: pd.Series,
    horizons: tuple[int, ...],
    *,
    entry_cost: float,
    exit_cost: float,
    label_smoothing: float = 0.0,
    direction: str = "long",
) -> pd.DataFrame:
    """Бинарные метки с учётом транзакционных издержек.

    Для каждого бара t и горизонта h:
        next_open  = open[t+1]
        future     = close[t+h]
        lr         = log_return_with_costs(next_open, future, direction, costs)
        label      = (lr > 0).astype(float)
        smoothed   = label * (1 - eps) + (1 - label) * eps   # if smoothing > 0

    Возвращает DataFrame с колонками:
        target_h{h}  - сглаженные метки {eps, 1-eps} ∈ [0, 1]
        lr_h{h}      - сырая лог-доходность с костами (для Bayes-порога)

    Last `h` rows для каждого горизонта - NaN, отбрасываются позже.
    """
    if not 0.0 <= label_smoothing < 0.5:
        msg = f"label_smoothing must be in [0, 0.5), got {label_smoothing}"
        raise ValueError(msg)

    out = pd.DataFrame(index=close_price.index)
    open_arr = open_price.astype(float).to_numpy()
    close_arr = close_price.astype(float).to_numpy()
    next_open = pd.Series(open_arr, index=close_price.index).shift(-1).to_numpy()

    for h in horizons:
        future_close = pd.Series(close_arr, index=close_price.index).shift(-h).to_numpy()
        lr = _log_return_with_costs(
            next_open, future_close,
            direction=direction,
            entry_cost=entry_cost,
            exit_cost=exit_cost,
        )
        # NaN на хвосте, где shift даёт NaN.
        valid = ~(np.isnan(next_open) | np.isnan(future_close))
        lr_full = np.where(valid, lr, np.nan)

        hard = (lr_full > 0).astype(np.float32)
        if label_smoothing > 0.0:
            eps = label_smoothing
            smoothed = hard * (1.0 - eps) + (1.0 - hard) * eps
        else:
            smoothed = hard
        # Восстанавливаем NaN на хвосте: dropna в downstream pipeline их уберёт.
        smoothed = np.where(valid, smoothed, np.nan)

        out[f"target_h{h}"] = smoothed.astype(np.float32)
        out[f"lr_h{h}"] = lr_full.astype(np.float32)

    return out
