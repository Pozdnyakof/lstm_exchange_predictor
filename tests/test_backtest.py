"""Тесты бэктест-движка и метода случайных портфелей."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.backtest import (
    compute_metrics,
    run_backtest,
    run_per_ticker_backtest,
    run_random_portfolios,
)
from graduate_work.config import TradingConfig


def _trending_prices(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    base = pd.DataFrame(
        {"close": np.linspace(100, 130, n), "ticker": "SBER"},
        index=idx,
    )
    other = pd.DataFrame(
        {"close": np.linspace(50, 60, n), "ticker": "GAZP"},
        index=idx,
    )
    return pd.concat([base, other])


def test_buy_signal_produces_trade() -> None:
    prices = _trending_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        commission_rate=0.0001,
        slippage_rate=0.0001,
        max_positions=1,
    )
    signals = pd.DataFrame(
        [
            {"timestamp": prices.index[0], "ticker": "SBER",
             "horizon": 5, "mean": 0.01, "std": 0.001, "action": "BUY"},
        ],
    )
    bt = run_backtest(signals, prices, cfg)
    assert not bt.trades.empty
    metrics = compute_metrics(bt.equity, bt.trades)
    assert metrics["n_trades"] == 1
    # Цена выросла, значит сделка должна быть в плюсе.
    assert bt.trades["pnl"].iloc[0] > 0


def test_no_signals_produce_no_trades() -> None:
    prices = _trending_prices()
    cfg = TradingConfig(initial_capital=100_000)
    signals = pd.DataFrame(columns=["timestamp", "ticker", "horizon", "mean", "std", "action"])
    bt = run_backtest(signals, prices, cfg)
    assert bt.trades.empty


def test_random_portfolios_return_distribution() -> None:
    prices = _trending_prices()
    cfg = TradingConfig(initial_capital=100_000, n_random_portfolios=50)
    report = run_random_portfolios(
        prices, cfg, avg_horizon=5, strategy_final=120_000.0,
    )
    assert report.final_equities.shape == (50,)
    assert report.std >= 0


def test_per_ticker_backtest_returns_one_row_per_ticker() -> None:
    prices = _trending_prices()
    cfg = TradingConfig(initial_capital=100_000)
    signals = pd.DataFrame(
        [
            {"timestamp": prices.index[0], "ticker": "SBER",
             "horizon": 5, "mean": 0.01, "std": 0.001, "action": "BUY"},
            {"timestamp": prices.index[5], "ticker": "GAZP",
             "horizon": 5, "mean": 0.005, "std": 0.001, "action": "BUY"},
        ],
    )
    per_ticker = run_per_ticker_backtest(signals, prices, cfg)
    assert set(per_ticker["ticker"]) == {"SBER", "GAZP"}
    assert (per_ticker["n_trades"] >= 1).all()
    # Капитал поделился поровну -> половина начального на каждого тикера.
    assert (per_ticker["final_equity"] > 0).all()
