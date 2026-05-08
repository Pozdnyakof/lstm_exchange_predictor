"""Тесты consensus-фильтра (long × short) и signal-driven backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graduate_work.backtest import run_consensus_backtest
from graduate_work.config import TradingConfig
from graduate_work.strategy import (
    ConsensusThresholds,
    apply_consensus_thresholds,
    build_consensus_frame,
    build_predictions_frame,
    consensus_summary,
)


# ---------------------------------------------------------------------------
# ConsensusThresholds
# ---------------------------------------------------------------------------

def test_thresholds_validate_range() -> None:
    with pytest.raises(ValueError):
        ConsensusThresholds(t_long=0.0, t_short=0.5)
    with pytest.raises(ValueError):
        ConsensusThresholds(t_long=0.5, t_short=1.0)
    # Допустимые значения проходят.
    ConsensusThresholds(t_long=0.6, t_short=0.4)


# ---------------------------------------------------------------------------
# build_consensus_frame
# ---------------------------------------------------------------------------

def _make_pred(probs: np.ndarray, *, ticker: str = "A") -> pd.DataFrame:
    n_samples, n_h = probs.shape
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="5min", tz="UTC")
    std = np.full_like(probs, 0.05, dtype=np.float32)
    return build_predictions_frame(
        timestamps=np.array(timestamps),
        tickers=np.array([ticker] * n_samples),
        mean=probs.astype(np.float32),
        std=std.astype(np.float32),
        horizons=tuple(range(1, n_h + 1)),
    )


def test_build_consensus_frame_picks_argmax_long() -> None:
    """argmax по horizon в long, на этом же горизонте подтягивается short."""
    long_probs = np.array([
        [0.3, 0.6, 0.4],   # h=2 (0.6) — argmax
        [0.5, 0.4, 0.7],   # h=3 (0.7)
    ])
    short_probs = np.array([
        [0.2, 0.3, 0.4],
        [0.1, 0.2, 0.5],
    ])
    long_pred = _make_pred(long_probs)
    short_pred = _make_pred(short_probs)
    cons = build_consensus_frame(long_pred, short_pred)
    assert len(cons) == 2
    # Первая строка: argmax-h=2, p_long=0.6, p_short на h=2 = 0.3.
    row0 = cons.iloc[0]
    assert row0["horizon"] == 2
    assert row0["p_long"] == pytest.approx(0.6, abs=1e-5)
    assert row0["p_short"] == pytest.approx(0.3, abs=1e-5)
    # Вторая: argmax-h=3, p_long=0.7, p_short на h=3 = 0.5.
    row1 = cons.iloc[1]
    assert row1["horizon"] == 3
    assert row1["p_short"] == pytest.approx(0.5, abs=1e-5)


def test_build_consensus_frame_empty() -> None:
    out = build_consensus_frame(pd.DataFrame(), pd.DataFrame())
    assert out.empty
    assert "p_long" in out.columns
    assert "p_short" in out.columns


# ---------------------------------------------------------------------------
# apply_consensus_thresholds
# ---------------------------------------------------------------------------

def test_apply_thresholds_open_close_logic() -> None:
    cons = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        "ticker": ["A", "B", "C", "D"],
        "horizon": [1, 1, 1, 1],
        # 4 случая:
        # A: long high, short low  → open
        # B: long low,  short high → close
        # C: long high, short high → hold (оба сигналят)
        # D: long low,  short low  → hold (никто не сигналит)
        "p_long":     [0.7, 0.3, 0.8, 0.2],
        "p_long_std": [0.05] * 4,
        "p_short":    [0.2, 0.7, 0.7, 0.3],
        "p_short_std":[0.05] * 4,
    })
    out = apply_consensus_thresholds(cons, ConsensusThresholds(t_long=0.5, t_short=0.5))
    # Открываем только A.
    assert out.loc[0, "open_long"]
    assert not out.loc[1, "open_long"]
    assert not out.loc[2, "open_long"]
    assert not out.loc[3, "open_long"]
    # Закрываем только B.
    assert out.loc[1, "close_long"]
    assert not out.loc[0, "close_long"]
    assert not out.loc[2, "close_long"]
    assert not out.loc[3, "close_long"]


def test_apply_thresholds_empty_frame_safe() -> None:
    cons = build_consensus_frame(pd.DataFrame(), pd.DataFrame())
    out = apply_consensus_thresholds(cons, ConsensusThresholds(t_long=0.5, t_short=0.5))
    assert out.empty
    assert "open_long" in out.columns and out["open_long"].dtype == bool


# ---------------------------------------------------------------------------
# consensus_summary
# ---------------------------------------------------------------------------

def test_consensus_summary_empty_returns_zeros() -> None:
    s = consensus_summary(pd.DataFrame())
    assert s["n_bars"] == 0
    assert s["frac_open_long"] == 0.0


def test_consensus_summary_counts() -> None:
    decisions = apply_consensus_thresholds(
        pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"] * 4, utc=True),
            "ticker": ["A", "B", "C", "D"],
            "horizon": [1, 1, 1, 1],
            "p_long":     [0.7, 0.3, 0.8, 0.2],
            "p_long_std": [0.05] * 4,
            "p_short":    [0.2, 0.7, 0.7, 0.3],
            "p_short_std":[0.05] * 4,
        }),
        ConsensusThresholds(t_long=0.5, t_short=0.5),
    )
    s = consensus_summary(decisions)
    assert s["n_open_long"] == 1   # только A
    assert s["n_close_long"] == 1  # только B
    assert s["n_bars"] == 4


# ---------------------------------------------------------------------------
# run_consensus_backtest
# ---------------------------------------------------------------------------

def _make_prices(n: int, ticker: str = "TST", *, ascending: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    closes = np.linspace(100.0, 110.0, n) if ascending else np.linspace(110.0, 100.0, n)
    opens = closes - 0.05
    return pd.DataFrame({
        "open": opens, "close": closes, "ticker": ticker,
    }, index=idx)


def _make_decisions(n: int, *, ticker: str = "TST",
                    open_at: list[int], close_at: list[int]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    rows = []
    for i, ts in enumerate(idx):
        rows.append({
            "timestamp": ts,
            "ticker": ticker,
            "horizon": 1,
            "p_long": 0.8 if i in open_at else 0.3,
            "p_long_std": 0.05,
            "p_short": 0.8 if i in close_at else 0.3,
            "p_short_std": 0.05,
            "long_signal": i in open_at,
            "short_signal": i in close_at,
            "open_long": i in open_at,
            "close_long": i in close_at,
        })
    return pd.DataFrame(rows)


def test_consensus_backtest_open_then_close_signal() -> None:
    """Открытие на open_long-баре, exit при первом close_long после."""
    prices = _make_prices(20)
    decisions = _make_decisions(20, open_at=[0], close_at=[5])
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=1,
        commission_rate=0.0, slippage_rate=0.0,
    )
    res = run_consensus_backtest(decisions, prices, cfg, max_hold_bars=20)
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    # Сигнал в bar 0 → entry в bar 1's open. Close-сигнал в bar 5 → exit в bar 6's open.
    assert trade["open_date"] == prices.index[1]
    assert trade["close_date"] == prices.index[6]


def test_consensus_backtest_max_hold_fallback() -> None:
    """Если close-сигнал не сработал — выход через max_hold_bars."""
    prices = _make_prices(30)
    # Открыли в bar 0, ни одного close_signal.
    decisions = _make_decisions(30, open_at=[0], close_at=[])
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=1,
        commission_rate=0.0, slippage_rate=0.0,
    )
    res = run_consensus_backtest(decisions, prices, cfg, max_hold_bars=5)
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    # entry bar 1 (open_long на bar 0), max_hold=5, exit на open[1+5]=open[6].
    assert trade["open_date"] == prices.index[1]
    # bars_held = exit_idx - entry_idx = 6 - 1 = 5.
    assert trade["horizon"] == 5


def test_consensus_backtest_no_double_open_same_ticker() -> None:
    """Повторный open_long при уже открытой позиции игнорируется."""
    prices = _make_prices(30)
    decisions = _make_decisions(30, open_at=[0, 5, 10], close_at=[15])
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=3,
        commission_rate=0.0, slippage_rate=0.0,
    )
    res = run_consensus_backtest(decisions, prices, cfg, max_hold_bars=20)
    # Должна быть ровно одна сделка (первая) — barcounts 5/10 пропускаются.
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["open_date"] == prices.index[1]


def test_consensus_backtest_empty_decisions_returns_initial() -> None:
    cfg = TradingConfig(initial_capital=100_000.0, max_positions=1)
    res = run_consensus_backtest(pd.DataFrame(), pd.DataFrame(), cfg, max_hold_bars=10)
    assert res.trades.empty
    assert res.equity.iloc[0] == cfg.initial_capital


def test_consensus_backtest_rejects_zero_max_hold() -> None:
    cfg = TradingConfig(initial_capital=100_000.0)
    prices = _make_prices(5)
    with pytest.raises(ValueError):
        run_consensus_backtest(pd.DataFrame({"timestamp": [prices.index[0]]}),
                               prices, cfg, max_hold_bars=0)
