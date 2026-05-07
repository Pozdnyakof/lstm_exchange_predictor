"""Двухступенчатый фильтр торговых сигналов.

Реализует логику §2.2 ВКР:
    1. Ранжируем активы по среднему MC-прогнозу нормализованной
       лог-доходности; выбираем горизонт с максимальным mean.
    2. Блокируем сделку, если std MC-распределения превышает заданный
       порог уверенности.
    3. Если для всех активов лучший mean отрицателен - выходим в кэш.

Формат возвращаемого фрейма сигналов:
    timestamp, ticker, horizon, expected_return, uncertainty, action.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TradingConfig


def build_predictions_frame(
    timestamps: np.ndarray,
    tickers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Сформировать длинный фрейм прогнозов: одна строка - один (date, ticker, horizon)."""
    n, h = mean.shape
    if h != len(horizons):
        msg = f"mean has {h} horizons but cfg has {len(horizons)}"
        raise ValueError(msg)
    rows: list[dict] = []
    for i in range(n):
        for j, hz in enumerate(horizons):
            rows.append(
                {
                    "timestamp": timestamps[i],
                    "ticker": str(tickers[i]),
                    "horizon": int(hz),
                    "mean": float(mean[i, j]),
                    "std": float(std[i, j]),
                },
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


class SignalGenerator:
    """Из множества прогнозов на дату генерирует торговое решение."""

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Базовый шаг: выбираем для каждого (date, ticker) лучший горизонт
    # ------------------------------------------------------------------
    def best_per_ticker(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return predictions.assign(action="HOLD")
        idx = (
            predictions.groupby(["timestamp", "ticker"])["mean"]
            .idxmax()
            .dropna()
            .astype(int)
        )
        best = predictions.loc[idx].reset_index(drop=True)
        return best

    # ------------------------------------------------------------------
    # Главная процедура: преобразование прогнозов в сигналы
    # ------------------------------------------------------------------
    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame(
                columns=["timestamp", "ticker", "horizon", "mean", "std", "action"],
            )
        best = self.best_per_ticker(predictions)
        # T2.3: Bonferroni-style коррекция за argmax-выбор горизонта.
        # При выборе MAX из N горизонтов нулевая гипотеза даёт смещение,
        # компенсируемое умножением порога на factor.
        n_horizons = predictions["horizon"].nunique() if not predictions.empty else 1
        correction = (
            self.cfg.horizon_argmax_correction if n_horizons > 1 else 1.0
        )
        effective_threshold = self.cfg.min_expected_return * correction
        sessions: list[pd.DataFrame] = []
        for ts, day in best.groupby("timestamp", sort=True):
            day = day.copy()
            # Шаг 1: глобальный фильтр - все mean < 0 → выход в кэш.
            if (day["mean"] <= 0.0).all():
                day["action"] = "HOLD"
                sessions.append(day)
                continue
            # Шаг 2: ранжируем по mean, режем top-K.
            day = day.sort_values("mean", ascending=False)
            day["action"] = "HOLD"
            top = day.head(self.cfg.max_positions).copy()
            qualifying = (
                (top["mean"] >= effective_threshold)
                & (top["std"] <= self.cfg.max_uncertainty)
            )
            day.loc[top.index[qualifying], "action"] = "BUY"
            sessions.append(day)
        if not sessions:
            return pd.DataFrame(
                columns=["timestamp", "ticker", "horizon", "mean", "std", "action"],
            )
        return pd.concat(sessions, ignore_index=True)
