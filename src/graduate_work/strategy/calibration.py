"""Калибровка порога стратегии на валидационной выборке.

Используется для CLASSIFICATION-режима: подбор Bayes-оптимального порога
по формуле `T = c_FP / (c_FP + c_FN)`, где
    c_FP = round-trip cost (commission + slippage за обе стороны),
    c_FN = средний реализованный gain по правильным предсказаниям на val.

См. §1.3 ВКР: «корректное вычисление математического ожидания потенциальной
сделки и адекватные алгоритмы управления капиталом» - именно это даёт
Bayes-порог при асимметричных costs.

Для REGRESSION-режима остаётся старая эмпирическая калибровка
(`calibrate_min_expected_return`) — она ищет квантиль mean-распределения,
выше которого среднее реализованное >= 2×cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BAYES_FLOOR = 0.51   # ≥ random guess для классификации
_BAYES_CEIL = 0.95    # верхняя планка - не задрать порог в стену
_DEFAULT_GAIN = 0.005   # fallback gain если на val нет позитивов
_MIN_GAIN = 1e-4


@dataclass
class CalibratedThreshold:
    """Результат калибровки."""

    min_expected_return: float
    n_val_signals: int
    val_avg_return: float
    val_win_rate: float


# ---------------------------------------------------------------------------
# Regression: эмпирический quantile fit
# ---------------------------------------------------------------------------

def calibrate_min_expected_return(
    val_predictions: pd.DataFrame,
    val_actual_targets: pd.DataFrame,
    *,
    cost_per_trade: float,
    candidate_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    floor: float = 1e-5,
) -> CalibratedThreshold:
    """Подобрать порог mean-прогноза на регрессии (legacy)."""
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
        threshold = max(float(merged["mean"].quantile(q)), floor)
        sub = merged[merged["mean"] >= threshold]
        if sub.empty:
            continue
        avg = float(sub["actual"].mean())
        wr = float((sub["actual"] > 0).mean())
        candidate = CalibratedThreshold(threshold, len(sub), avg, wr)
        if avg >= target_edge:
            return candidate
        if best is None or candidate.val_avg_return > best.val_avg_return:
            best = candidate

    if best is not None:
        logger.warning(
            "Calibration: no threshold cleared 2x cost (%.5f); using best "
            "candidate T=%.5f avg=%.5f wr=%.3f",
            target_edge, best.min_expected_return,
            best.val_avg_return, best.val_win_rate,
        )
        return best
    fallback = float(merged["mean"].quantile(0.95))
    return CalibratedThreshold(max(fallback, floor), 0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Classification: Bayes-порог
# ---------------------------------------------------------------------------

def estimate_gain_from_lr(lr_values: np.ndarray) -> float:
    """Средний positive lr - оценка ожидаемой выгоды одной правильной сделки."""
    if lr_values.size == 0:
        return _DEFAULT_GAIN
    pos = lr_values[lr_values > 0]
    if pos.size < 5:
        return _DEFAULT_GAIN
    return max(float(pos.mean()), _MIN_GAIN)


def bayes_threshold(cost_per_trade: float, expected_gain: float) -> float:
    """Bayes-оптимальный порог при асимметричных costs.

    Под H1 (предсказание-up верное) выигрываем `expected_gain`,
    под H0 (false positive) теряем `cost_per_trade`.
    Безусловно-оптимальный порог при равных приорах:
        T = c_FP / (c_FP + c_FN)
    """
    c_fp = max(cost_per_trade, _MIN_GAIN)
    c_fn = max(expected_gain, _MIN_GAIN)
    raw = c_fp / (c_fp + c_fn)
    return float(np.clip(raw, _BAYES_FLOOR, _BAYES_CEIL))


def calibrate_bayes_threshold(
    val_predictions: pd.DataFrame,
    val_lr_targets: pd.DataFrame,
    *,
    cost_per_trade: float,
) -> CalibratedThreshold:
    """Bayes-калибровка порога вероятности на classification-режиме.

    val_predictions:    `(timestamp, ticker, horizon, mean, std)`
                        где `mean` ∈ [0, 1] - вероятность.
    val_lr_targets:     `(timestamp, ticker, horizon, actual)`
                        где `actual` - сырая лог-доходность с костами
                        (НЕ сглаженная метка!).

    Шаги:
      1) собираем positive lr на val → expected_gain;
      2) T_bayes = c_FP / (c_FP + expected_gain);
      3) для отчёта - проверяем эмпирически: на val сигналы с prob ≥ T_bayes
         должны иметь средний actual ≥ cost.
    """
    if val_predictions.empty or val_lr_targets.empty:
        return CalibratedThreshold(_BAYES_FLOOR, 0, 0.0, 0.0)

    merged = val_predictions.merge(
        val_lr_targets,
        on=["timestamp", "ticker", "horizon"],
        how="inner",
    )
    if merged.empty or "actual" not in merged.columns:
        return CalibratedThreshold(_BAYES_FLOOR, 0, 0.0, 0.0)

    expected_gain = estimate_gain_from_lr(merged["actual"].to_numpy(dtype=np.float64))
    threshold = bayes_threshold(cost_per_trade, expected_gain)

    sub = merged[merged["mean"] >= threshold]
    avg = float(sub["actual"].mean()) if not sub.empty else 0.0
    wr = float((sub["actual"] > 0).mean()) if not sub.empty else 0.0
    logger.info(
        "Bayes threshold: T=%.4f (cost=%.5g, gain=%.5g, val signals=%d, "
        "avg_actual=%.5g, wr=%.3f)",
        threshold, cost_per_trade, expected_gain, len(sub), avg, wr,
    )
    return CalibratedThreshold(threshold, len(sub), avg, wr)


# ---------------------------------------------------------------------------
# Сборка фрейма actual-меток
# ---------------------------------------------------------------------------

def attach_actual_targets(
    val_dataset: dict,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Из ``PreparedDataset.val`` собрать длинный фрейм `(timestamp, ticker, horizon, actual)`.

    `actual` - то, что хранилось в `y` тензоре датасета. Для regression
    это нормализованная лог-доходность; для classification это сглаженная
    бинарная метка - её **нельзя** использовать для Bayes-калибровки!
    Для Bayes используйте отдельную ``attach_lr_targets``.
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


def _normalize_split_df(val_split_df: pd.DataFrame) -> pd.DataFrame:
    """Привести val-split к виду с колонкой `timestamp` (UTC) и `ticker`."""
    df = val_split_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis("timestamp").reset_index()
    elif "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def attach_lr_targets(
    val_split_df: pd.DataFrame,
    val_dataset: dict,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Извлечь сырые `lr_h{h}` из `val_split_df` для (timestamp, ticker)
    каждого окна датасета.

    Используется для Bayes-калибровки порога: lr - чистая лог-доходность
    с костами, без сглаживания меток.
    """
    n = val_dataset["y"].shape[0]
    if n == 0 or "ticker" not in val_split_df.columns:
        return pd.DataFrame(columns=["timestamp", "ticker", "horizon", "actual"])

    df = _normalize_split_df(val_split_df)
    by_key: dict[tuple, int] = {
        (int(t.value), str(tk)): i
        for i, (t, tk) in enumerate(zip(df["timestamp"], df["ticker"]))
    }

    tickers = val_dataset["ticker"]
    timestamps = pd.to_datetime(val_dataset["timestamp"], utc=True)
    lr_cols = [f"lr_h{hz}" for hz in horizons]
    available = [c for c in lr_cols if c in df.columns]
    if not available:
        return pd.DataFrame(columns=["timestamp", "ticker", "horizon", "actual"])

    out_rows: list[dict] = []
    for i in range(n):
        ts = pd.Timestamp(timestamps[i])
        ticker = str(tickers[i])
        idx = by_key.get((ts.value, ticker))
        if idx is None:
            continue
        row = df.iloc[idx]
        for hz, col in zip(horizons, lr_cols):
            if col not in available:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            out_rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "horizon": int(hz),
                "actual": float(value),
            })
    return pd.DataFrame(out_rows)
