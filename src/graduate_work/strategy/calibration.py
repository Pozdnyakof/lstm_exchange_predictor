"""Калибровка порога стратегии на валидационной выборке.

§2.2 ВКР описывает двухступенчатый фильтр сигналов с порогом
``min_expected_return``. По умолчанию это число хардкодится в конфиге,
но при работе с разными тикерами / горизонтами оптимальный порог
варьируется. Эта утилита подбирает порог так, чтобы на val-данных
эмпирическая средняя реализованная доходность по сигналу с
``mean ≥ T`` была статистически выше cost-floor'а.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibratedThreshold:
    """Результат калибровки."""

    min_expected_return: float
    n_val_signals: int
    val_avg_return: float
    val_win_rate: float


def calibrate_min_expected_return(
    val_predictions: pd.DataFrame,
    val_actual_targets: pd.DataFrame,
    *,
    cost_per_trade: float,
    candidate_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    floor: float = 1e-5,
) -> CalibratedThreshold:
    """Подобрать ``min_expected_return``, максимизирующий val-edge.

    Аргументы:
      val_predictions:  фрейм с колонками `mean`, `horizon`, `ticker`,
                        `timestamp` (выход `build_predictions_frame`).
      val_actual_targets: фрейм с тем же `(ticker, timestamp, horizon)` и
                          колонкой `actual` - реализованная нормализованная
                          лог-доходность.
      cost_per_trade:   round-trip-cost в долях (commission+slippage)*2.
      candidate_quantiles: квантили `mean`-распределения, которые пробуем.
      floor: минимальный возможный порог.

    Возвращает порог, при котором эмпирический mean(actual | mean ≥ T)
    превышает 2× cost. Если ни один не превышает, возвращает 95-й
    перцентиль (конссервативный консервативный фильтр).
    """
    if val_predictions.empty or val_actual_targets.empty:
        return CalibratedThreshold(floor, 0, 0.0, 0.0)

    merged = val_predictions.merge(
        val_actual_targets,
        on=["timestamp", "ticker", "horizon"],
        how="inner",
    )
    if merged.empty or "actual" not in merged.columns:
        return CalibratedThreshold(floor, 0, 0.0, 0.0)

    target_edge = 2.0 * cost_per_trade
    best: CalibratedThreshold | None = None
    for q in candidate_quantiles:
        threshold = float(merged["mean"].quantile(q))
        if threshold < floor:
            threshold = floor
        sub = merged[merged["mean"] >= threshold]
        if sub.empty:
            continue
        avg = float(sub["actual"].mean())
        wr = float((sub["actual"] > 0).mean())
        candidate = CalibratedThreshold(
            min_expected_return=threshold,
            n_val_signals=len(sub),
            val_avg_return=avg,
            val_win_rate=wr,
        )
        if avg >= target_edge:
            return candidate
        # Сохраняем наименьший порог, прошедший хоть какой-то edge.
        if best is None or candidate.val_avg_return > best.val_avg_return:
            best = candidate

    if best is not None:
        logger.warning(
            "Calibration: ни один порог не дал edge ≥ 2×cost (%.5f); "
            "берём лучший: T=%.5f, avg=%.5f, wr=%.3f",
            target_edge, best.min_expected_return,
            best.val_avg_return, best.val_win_rate,
        )
        return best

    fallback = float(merged["mean"].quantile(0.95))
    return CalibratedThreshold(max(fallback, floor), 0, 0.0, 0.0)


def attach_actual_targets(
    val_dataset: dict,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Из тестового/валидационного `dict` (как в PreparedDataset.val)
    собрать длинный фрейм `(timestamp, ticker, horizon, actual)`.
    """
    n, h = val_dataset["y"].shape
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "ticker", "horizon", "actual"])
    rows: list[dict] = []
    timestamps = val_dataset["timestamp"]
    tickers = val_dataset["ticker"]
    for i in range(n):
        for j, hz in enumerate(horizons):
            rows.append({
                "timestamp": timestamps[i],
                "ticker": str(tickers[i]),
                "horizon": int(hz),
                "actual": float(val_dataset["y"][i, j]),
            })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
