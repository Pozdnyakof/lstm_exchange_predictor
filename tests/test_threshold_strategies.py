"""Тесты threshold_strategies: top-k, isotonic, max-PnL, signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.strategy import (
    apply_isotonic_calibration,
    fit_isotonic_per_horizon,
    max_pnl_threshold,
    signals_argmax_threshold,
    signals_per_horizon_threshold,
    top_k_threshold,
)


# ---------------------------------------------------------------------------
# top_k_threshold
# ---------------------------------------------------------------------------

def test_top_k_threshold_returns_quantile() -> None:
    """k=5 → 95-й перцентиль."""
    scores = np.linspace(0.0, 1.0, 1001)
    T = top_k_threshold(scores, k_pct=5.0)
    assert abs(T - 0.95) < 1e-6


def test_top_k_threshold_validates_input() -> None:
    """k_pct ∉ (0, 100) → ValueError."""
    for bad in (-1.0, 0.0, 100.0, 150.0):
        try:
            top_k_threshold(np.array([0.5]), k_pct=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on k_pct={bad}")


def test_top_k_threshold_empty_returns_zero() -> None:
    """Пустой массив → 0.0 (нет данных, ничего не торгуем)."""
    assert top_k_threshold(np.array([]), k_pct=5.0) == 0.0


# ---------------------------------------------------------------------------
# max_pnl_threshold
# ---------------------------------------------------------------------------

def test_max_pnl_picks_high_threshold_when_high_preds_winning() -> None:
    """Если только high-conf predictions выигрывают → max_pnl выбирает высокий T."""
    rng = np.random.default_rng(42)
    n = 1000
    preds = rng.uniform(0.3, 0.9, size=n)
    # Победители только при preds > 0.7.
    lrs = np.where(preds > 0.7, 0.005, -0.002)
    best_T, table = max_pnl_threshold(
        preds, lrs,
        sweep=(0.4, 0.5, 0.6, 0.7, 0.8),
        cost_per_trade=0.0, min_trades=10,
    )
    assert best_T >= 0.7


def test_max_pnl_validates_shapes() -> None:
    """Несовпадающие shapes → ValueError."""
    try:
        max_pnl_threshold(np.zeros(10), np.zeros(20))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_max_pnl_returns_sweep_table() -> None:
    """sweep_table содержит запись на каждый T в sweep."""
    preds = np.full(500, 0.5)
    lrs = np.zeros(500)
    sweep = (0.3, 0.5, 0.7)
    _, table = max_pnl_threshold(preds, lrs, sweep=sweep, min_trades=10)
    assert len(table) == 3
    for row, T in zip(table, sweep):
        assert row["T"] == T
        assert "n_trades" in row
        assert "mean_pnl" in row


def test_max_pnl_fallback_when_too_few_trades() -> None:
    """Если ни один T не даёт min_trades — возвращает первый T."""
    preds = np.array([0.1] * 50)  # все ниже порога
    lrs = np.zeros(50)
    best_T, _ = max_pnl_threshold(
        preds, lrs, sweep=(0.5, 0.7), min_trades=100,
    )
    assert best_T == 0.5


# ---------------------------------------------------------------------------
# fit_isotonic_per_horizon + apply_isotonic_calibration
# ---------------------------------------------------------------------------

def _make_pred_target_pair(
    n: int = 500, h_list=(6, 12), shift: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Биазнутые predictions: target=1 при p_real>0.5, но pred смещены на shift."""
    rng = np.random.default_rng(0)
    rows_p, rows_t = [], []
    for h in h_list:
        for i in range(n):
            true_p = float(rng.uniform(0.0, 1.0))
            biased_p = float(np.clip(true_p + shift, 0.0, 1.0))
            target = float(rng.binomial(1, true_p))
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i)
            rows_p.append({
                "timestamp": ts, "ticker": "TST", "horizon": h,
                "mean": biased_p, "std": 0.05,
            })
            rows_t.append({
                "timestamp": ts, "ticker": "TST", "horizon": h, "actual": target,
            })
    return pd.DataFrame(rows_p), pd.DataFrame(rows_t)


def test_fit_isotonic_returns_calibrator_per_horizon() -> None:
    """Создаёт IsotonicRegression на каждый горизонт."""
    preds, targets = _make_pred_target_pair(n=500, h_list=(6, 12))
    calibrators = fit_isotonic_per_horizon(preds, targets)
    assert set(calibrators.keys()) == {6, 12}


def test_isotonic_corrects_systematic_bias() -> None:
    """После калибровки mean(calibrated) ближе к mean(target), чем mean(raw)."""
    preds, targets = _make_pred_target_pair(n=2000, h_list=(6,), shift=0.15)
    calibrators = fit_isotonic_per_horizon(preds, targets)
    calibrated = apply_isotonic_calibration(preds, calibrators)
    raw_mean = preds["mean"].mean()
    calibrated_mean = calibrated["mean"].mean()
    target_mean = targets["actual"].mean()
    raw_bias = abs(raw_mean - target_mean)
    cal_bias = abs(calibrated_mean - target_mean)
    assert cal_bias < raw_bias


def test_apply_isotonic_preserves_dataframe_structure() -> None:
    """Калибровка возвращает DataFrame того же shape с теми же колонками."""
    preds, targets = _make_pred_target_pair(n=300, h_list=(6,))
    cal = fit_isotonic_per_horizon(preds, targets)
    out = apply_isotonic_calibration(preds, cal)
    assert out.shape == preds.shape
    assert list(out.columns) == list(preds.columns)


def test_isotonic_empty_inputs_safe() -> None:
    """Пустые predictions/targets → пустой словарь без ошибок."""
    empty = pd.DataFrame(columns=["timestamp", "ticker", "horizon", "mean", "std"])
    cal = fit_isotonic_per_horizon(empty, empty)
    assert cal == {}
    out = apply_isotonic_calibration(empty, cal)
    assert out.empty


# ---------------------------------------------------------------------------
# signals_argmax_threshold + signals_per_horizon_threshold
# ---------------------------------------------------------------------------

def _make_predictions(n_ts: int = 5, tickers=("A", "B"), horizons=(6, 12)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_ts):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i)
        for tk in tickers:
            for h in horizons:
                rows.append({
                    "timestamp": ts, "ticker": tk, "horizon": h,
                    "mean": float(rng.uniform(0.3, 0.7)), "std": 0.05,
                })
    return pd.DataFrame(rows)


def test_signals_argmax_threshold_basic_columns() -> None:
    """Возвращает DataFrame с обязательными колонками."""
    preds = _make_predictions()
    sig = signals_argmax_threshold(preds, threshold=0.5)
    for col in ("timestamp", "ticker", "horizon", "mean", "action", "signal"):
        assert col in sig.columns


def test_signals_argmax_picks_best_horizon_per_pair() -> None:
    """Для каждой (timestamp, ticker) — ровно одна строка с argmax."""
    preds = _make_predictions(n_ts=3, tickers=("A",), horizons=(6, 12, 24))
    sig = signals_argmax_threshold(preds, threshold=0.0, max_positions=10)
    # 3 timestamps × 1 ticker → 3 строки.
    assert len(sig) == 3


def test_signals_argmax_threshold_zero_buys_all() -> None:
    """threshold=0 → все аргmax'ы становятся BUY."""
    preds = _make_predictions(n_ts=5)
    sig = signals_argmax_threshold(preds, threshold=0.0, max_positions=10)
    assert (sig["action"] == "BUY").all()


def test_signals_argmax_threshold_high_blocks_all() -> None:
    """Очень высокий threshold → ноль BUY."""
    preds = _make_predictions(n_ts=5)
    sig = signals_argmax_threshold(preds, threshold=0.99)
    assert (sig["action"] == "HOLD").all()


def test_signals_argmax_max_positions_limits_buys() -> None:
    """max_positions ограничивает число BUY-сигналов на timestamp."""
    # 10 тикеров, threshold=0 → BUY всех argmax'ов
    rng = np.random.default_rng(0)
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    for tk in [f"T{i}" for i in range(10)]:
        for h in (6, 12):
            rows.append({
                "timestamp": ts, "ticker": tk, "horizon": h,
                "mean": float(rng.uniform(0.5, 0.9)), "std": 0.05,
            })
    preds = pd.DataFrame(rows)
    sig = signals_argmax_threshold(preds, threshold=0.0, max_positions=3)
    n_buy_per_ts = sig[sig["action"] == "BUY"].groupby("timestamp").size()
    assert (n_buy_per_ts <= 3).all()


def test_signals_per_horizon_filters_only_target_horizon() -> None:
    """Per-horizon: только заданный горизонт участвует."""
    preds = _make_predictions(horizons=(6, 12, 24))
    sig = signals_per_horizon_threshold(preds, horizon=12, threshold=0.0)
    # У всех BUY-сигналов horizon должен быть 12.
    assert (sig["horizon"] == 12).all()


def test_signals_empty_predictions_returns_empty() -> None:
    """Пустой preds → пустой результат с правильными колонками."""
    empty = pd.DataFrame(columns=["timestamp", "ticker", "horizon", "mean", "std"])
    sig = signals_argmax_threshold(empty, threshold=0.5)
    assert sig.empty
    assert "action" in sig.columns
