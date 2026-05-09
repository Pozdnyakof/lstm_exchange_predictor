"""Стратегии преобразования предсказаний в торговые сигналы.

После Sprint 2 модель перестала overfitить, но Sharpe всё ещё < 0 при
всех calibrate-методах (ACI/DtACI/Bayes). Bottleneck сместился с
"модель не учится" на "стратегия выбора threshold'а неоптимальна".

Этот модуль — для **diagnostic threshold strategy comparison**:
4 стратегии × N горизонтов × {iTransformer, LightGBM} даёт матрицу
Sharpe-метрик. Если хотя бы одна комбинация даёт положительный Sharpe
на test — задача обучаема, проблема была в стратегии. Если все < 0 —
cost-aware labels слишком строгие при текущих фичах/горизонтах.

## Стратегии

1. **Probability cutoff** — текущий baseline, threshold T → BUY если ``mean > T``.
2. **Top-k% selection** — T = (100-k)-й перцентиль предсказаний; форсирует
   trading только на top-k% самых уверенных. Quant-стандарт для слабого
   сигнала (Asness-Moskowitz-Pedersen 2013).
3. **Isotonic calibration** — Niculescu-Mizil & Caruana (ICML 2005). Fit
   IsotonicRegression на val-парах (pred, target), затем обычный 0.5
   threshold. Калибрует биазнутые prob (mean(pred) > P(UP) указывает на
   miscalibration).
4. **Max-PnL threshold** — sweep по T ∈ {0.4, 0.45, ...}, выбираем тот,
   что максимизирует ``mean(lr_actual) - cost`` на val. Прямая
   PnL-оптимизация, не coverage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold computation primitives
# ---------------------------------------------------------------------------

def top_k_threshold(scores: np.ndarray, k_pct: float) -> float:
    """Threshold T такой, что примерно k_pct% scores выше T.

    Для ``k_pct=5.0`` → возвращает 95-й перцентиль ``scores``. На пустом
    массиве возвращает 0.0 (нет данных, ничего не торгуем).
    """
    if not 0.0 < k_pct < 100.0:
        msg = f"k_pct must be in (0, 100), got {k_pct}"
        raise ValueError(msg)
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, 1.0 - k_pct / 100.0))


def max_pnl_threshold(
    val_predictions: np.ndarray,
    val_lrs: np.ndarray,
    *,
    sweep: tuple[float, ...] = (
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    ),
    cost_per_trade: float = 0.001,
    min_trades: int = 100,
) -> tuple[float, list[dict]]:
    """Найти T максимизирующий ``mean(lr_actual - cost)`` на val при ``prob > T``.

    Возвращает (best_T, sweep_table). ``sweep_table`` — список dict'ов
    с 'T', 'n_trades', 'mean_pnl' для inspection.

    Если ни один T не даёт ≥ ``min_trades`` сделок — возвращает первый
    из sweep как fallback.
    """
    if val_predictions.shape != val_lrs.shape:
        msg = (
            f"shapes mismatch: predictions={val_predictions.shape}, "
            f"lrs={val_lrs.shape}"
        )
        raise ValueError(msg)
    table: list[dict] = []
    best_T = float(sweep[0])
    best_pnl = -np.inf
    for T in sweep:
        mask = val_predictions > T
        n = int(mask.sum())
        if n < min_trades:
            table.append({"T": float(T), "n_trades": n, "mean_pnl": float("nan")})
            continue
        mean_pnl = float(val_lrs[mask].mean() - cost_per_trade)
        table.append({"T": float(T), "n_trades": n, "mean_pnl": mean_pnl})
        if mean_pnl > best_pnl:
            best_pnl = mean_pnl
            best_T = float(T)
    if best_pnl == -np.inf:
        logger.warning(
            "max_pnl_threshold: ни один T не дал >= %d сделок; fallback к T=%.2f",
            min_trades, best_T,
        )
    return best_T, table


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

def fit_isotonic_per_horizon(
    val_predictions: pd.DataFrame,
    val_targets: pd.DataFrame,
    *,
    min_samples: int = 100,
) -> dict[int, IsotonicRegression]:
    """Per-horizon isotonic-калибраторы из (pred, target) пар.

    ``val_predictions`` и ``val_targets`` мерджатся по
    (timestamp, ticker, horizon). Для каждого горизонта обучается
    отдельный :class:`IsotonicRegression` (Niculescu-Mizil ICML 2005),
    отображающий ``model.mean → P(target=1)``.
    """
    if val_predictions.empty or val_targets.empty:
        return {}
    merged = val_predictions.merge(
        val_targets, on=["timestamp", "ticker", "horizon"], how="inner",
    )
    if merged.empty or "actual" not in merged.columns:
        return {}
    calibrators: dict[int, IsotonicRegression] = {}
    for h, sub in merged.groupby("horizon"):
        x = sub["mean"].to_numpy(dtype=np.float64)
        y = sub["actual"].to_numpy(dtype=np.float64)
        if x.size < min_samples or np.unique(y).size < 2:
            logger.warning(
                "Horizon %s: insufficient data for isotonic (n=%d, classes=%d)",
                h, x.size, np.unique(y).size,
            )
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        calibrators[int(h)] = iso
    return calibrators


def apply_isotonic_calibration(
    predictions: pd.DataFrame,
    calibrators: dict[int, IsotonicRegression],
) -> pd.DataFrame:
    """Заменить ``mean`` калиброванными значениями по горизонту.

    Возвращает копию predictions с обновлённым ``mean``. Горизонты, для
    которых калибратор отсутствует (insufficient data), остаются неизменными.
    """
    if predictions.empty or not calibrators:
        return predictions.copy()
    out = predictions.copy()
    for h, iso in calibrators.items():
        mask = out["horizon"] == h
        if not mask.any():
            continue
        out.loc[mask, "mean"] = iso.predict(out.loc[mask, "mean"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Signal generation (без std-фильтрации и Šidák — для чистого
# strategy comparison)
# ---------------------------------------------------------------------------

_SIGNALS_COLS = [
    "timestamp", "ticker", "horizon", "mean", "std", "action", "signal",
]


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=_SIGNALS_COLS)


def signals_argmax_threshold(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Argmax-горизонт + threshold cutoff. Чистый baseline.

    1. Per (timestamp, ticker): выбираем горизонт с максимальным ``mean``.
    2. Сортируем по ``mean`` desc, берём top-``max_positions`` за timestamp.
    3. BUY если ``mean > threshold``, иначе HOLD.

    Без std-фильтрации и без Šidák-коррекции — чтобы стратегии
    сравнивались на equal terms (отличаются только threshold'ом).
    """
    if predictions.empty:
        return _empty_signals()
    idx = (
        predictions.groupby(["timestamp", "ticker"])["mean"]
        .idxmax().dropna().astype(int)
    )
    best = predictions.loc[idx].reset_index(drop=True)
    sessions: list[pd.DataFrame] = []
    for _ts, day in best.groupby("timestamp", sort=True):
        day = day.copy().sort_values("mean", ascending=False)
        day["action"] = "HOLD"
        day["signal"] = 0
        top = day.head(int(max_positions)).copy()
        qualifying = top["mean"] > threshold
        buy_idx = top.index[qualifying]
        day.loc[buy_idx, "action"] = "BUY"
        day.loc[buy_idx, "signal"] = 1
        sessions.append(day[_SIGNALS_COLS])
    if not sessions:
        return _empty_signals()
    return pd.concat(sessions, ignore_index=True)


def signals_per_horizon_threshold(
    predictions: pd.DataFrame,
    horizon: int,
    threshold: float,
    *,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Только заданный горизонт + threshold cutoff.

    Удобно для per-horizon диагностики: фиксируем h, варьируем threshold.
    """
    if predictions.empty:
        return _empty_signals()
    h_preds = predictions[predictions["horizon"] == int(horizon)]
    if h_preds.empty:
        return _empty_signals()
    return signals_argmax_threshold(
        h_preds, threshold, max_positions=max_positions,
    )


__all__ = [
    "apply_isotonic_calibration",
    "fit_isotonic_per_horizon",
    "max_pnl_threshold",
    "signals_argmax_threshold",
    "signals_per_horizon_threshold",
    "top_k_threshold",
]
