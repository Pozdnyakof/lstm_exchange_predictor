"""Consensus-фильтр: long-модель × short-модель.

Идея — обучаем две независимые модели (на одних фичах, разные таргеты:
long-direction и short-direction). На инференсе для каждого бара
получаем ``P_long`` и ``P_short``. Сделка валидируется обеими:

* **OPEN long** ⇔ ``P_long > T_long`` И ``P_short < T_short``
* **CLOSE long** ⇔ ``P_short > T_short`` И ``P_long < T_long``

Это эквивалентно «лонг открываем, когда long-модель уверена в лонге
и short-модель не уверена в шорте; закрываем — наоборот».

Sweep по ``(T_long, T_short)`` дешёвый: MC-инференс уже сделан, при
смене порогов нужно только пересчитать булевы флаги — никакого
повторного forward-pass через сеть.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConsensusThresholds:
    """Пороги бинарных решений по обеим моделям.

    Семантика идентичная: «вероятность того, что модель считает свою
    direction достаточно вероятной для торговли».

        long_signal  ⇔ P_long  > t_long
        short_signal ⇔ P_short > t_short
    """

    t_long: float
    t_short: float

    def __post_init__(self) -> None:
        for name, value in (("t_long", self.t_long), ("t_short", self.t_short)):
            if not 0.0 < value < 1.0:
                msg = f"{name} must be in (0, 1), got {value}"
                raise ValueError(msg)


def build_consensus_frame(
    long_predictions: pd.DataFrame,
    short_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Объединить long/short MC-предсказания в один per-bar DataFrame.

    Стратегия выбора горизонта: argmax по ``P_long`` для каждой пары
    ``(timestamp, ticker)``. На том же горизонте подтягивается
    ``P_short`` (если запись отсутствует — строка отбрасывается).

    Returns:
        DataFrame со столбцами: timestamp, ticker, horizon,
        p_long, p_long_std, p_short, p_short_std.
    """
    if long_predictions.empty or short_predictions.empty:
        return pd.DataFrame(columns=[
            "timestamp", "ticker", "horizon",
            "p_long", "p_long_std", "p_short", "p_short_std",
        ])

    best_long_idx = (
        long_predictions.groupby(["timestamp", "ticker"])["mean"]
        .idxmax().dropna().astype(int)
    )
    best_long = long_predictions.loc[best_long_idx, [
        "timestamp", "ticker", "horizon", "mean", "std",
    ]].rename(columns={"mean": "p_long", "std": "p_long_std"})

    short_lookup = short_predictions[[
        "timestamp", "ticker", "horizon", "mean", "std",
    ]].rename(columns={"mean": "p_short", "std": "p_short_std"})

    merged = best_long.merge(
        short_lookup,
        on=["timestamp", "ticker", "horizon"],
        how="inner",
    )
    return merged.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def apply_consensus_thresholds(
    consensus_frame: pd.DataFrame,
    thresholds: ConsensusThresholds,
) -> pd.DataFrame:
    """Добавить булевы решения по заданным порогам.

    Не пересчитывает MC-предсказания — это и есть точка
    оптимизации thresholds-sweep'а.
    """
    if consensus_frame.empty:
        out = consensus_frame.copy()
        for col in ("long_signal", "short_signal", "open_long", "close_long"):
            out[col] = pd.Series(dtype=bool)
        return out
    out = consensus_frame.copy()
    long_signal = out["p_long"] > thresholds.t_long
    short_signal = out["p_short"] > thresholds.t_short
    out["long_signal"] = long_signal
    out["short_signal"] = short_signal
    out["open_long"] = long_signal & ~short_signal
    out["close_long"] = short_signal & ~long_signal
    return out


def consensus_summary(decisions: pd.DataFrame) -> dict[str, float | int]:
    """Сводка для диагностики: сколько баров под каждым типом сигнала."""
    if decisions.empty:
        return {
            "n_bars": 0, "n_open_long": 0, "n_close_long": 0,
            "frac_open_long": 0.0, "frac_close_long": 0.0,
            "p_long_max": float("nan"), "p_short_max": float("nan"),
        }
    n = len(decisions)
    n_open = int(decisions["open_long"].sum())
    n_close = int(decisions["close_long"].sum())
    return {
        "n_bars": n,
        "n_open_long": n_open,
        "n_close_long": n_close,
        "frac_open_long": n_open / n,
        "frac_close_long": n_close / n,
        "p_long_max": float(decisions["p_long"].max()),
        "p_short_max": float(decisions["p_short"].max()),
        "p_long_mean": float(decisions["p_long"].mean()),
        "p_short_mean": float(decisions["p_short"].mean()),
    }
