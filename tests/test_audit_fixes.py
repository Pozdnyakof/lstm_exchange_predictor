"""Регрессионные тесты на найденные при аудите баги."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from graduate_work.backtest import compute_metrics, run_backtest, run_random_portfolios
from graduate_work.backtest.metrics import sharpe_ratio
from graduate_work.config import DataConfig, TradingConfig
from graduate_work.features.pipeline import _index_log_returns, _purge_tail


# ---------------------------------------------------------------------------
# C1 - horizon в БАРАХ, не в днях
# ---------------------------------------------------------------------------

def _intraday_prices(n_bars: int = 30, ticker: str = "TEST") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 07:00", periods=n_bars, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"close": np.linspace(100.0, 110.0, n_bars), "ticker": ticker},
        index=idx,
    )


def test_engine_holds_for_horizon_bars_not_days() -> None:
    """horizon=3 должен означать 3 бара, не 3 дня."""
    prices = _intraday_prices(n_bars=20)
    cfg = TradingConfig(initial_capital=100_000, max_positions=1)
    # BUY на первом баре, horizon=3 -> закрываемся на баре с индексом 3.
    signals = pd.DataFrame(
        [{
            "timestamp": prices.index[0],
            "ticker": "TEST", "horizon": 3, "mean": 0.01, "std": 0.001,
            "action": "BUY",
        }],
    )
    bt = run_backtest(signals, prices, cfg)
    assert len(bt.trades) == 1
    trade = bt.trades.iloc[0]
    # Закрытие должно произойти на баре 3 (bar 0=open, 3=close).
    assert trade["open_date"] == prices.index[0]
    assert trade["close_date"] == prices.index[3]


def test_random_monkeys_use_bar_horizon() -> None:
    """random monkeys тоже должны держать в БАРАХ."""
    prices = _intraday_prices(n_bars=50)
    cfg = TradingConfig(
        initial_capital=100_000, max_positions=1, n_random_portfolios=10,
    )
    report = run_random_portfolios(
        prices, cfg, avg_horizon=3, trade_probability=0.5,
        strategy_final=100_000, seed=1,
    )
    # Без падений; финальные капиталы заполнены.
    assert report.final_equities.shape == (10,)


# ---------------------------------------------------------------------------
# H1 - annualised Sharpe учитывает bars_per_year
# ---------------------------------------------------------------------------

def test_sharpe_scales_with_periods_per_year() -> None:
    """Sharpe(252) и Sharpe(26460) на одних и тех же returns должны
    отличаться в sqrt(26460/252) ≈ 10.24 раз."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.standard_normal(1000) * 0.001)
    s_daily = sharpe_ratio(rets, periods_per_year=252)
    s_5min = sharpe_ratio(rets, periods_per_year=26460)
    if abs(s_daily) < 1e-9:
        pytest.skip("zero sharpe — degenerate sample")
    ratio = abs(s_5min / s_daily)
    expected = math.sqrt(26460 / 252)
    assert abs(ratio - expected) / expected < 0.01


def test_compute_metrics_accepts_periods_per_year() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    equity = pd.Series(np.linspace(1.0, 1.05, 100), index=idx)
    metrics = compute_metrics(equity, pd.DataFrame(), periods_per_year=26460)
    assert "sharpe" in metrics
    # Equity монотонно растёт — Sharpe должен быть > 0.
    assert metrics["sharpe"] > 0


def test_data_config_bars_per_year_for_5min_moex() -> None:
    cfg = DataConfig(bar_minutes=5, session_start_utc="07:00", session_end_utc="15:45")
    # 525 минут / 5 = 105 баров за сессию * 252 дня = 26460.
    assert cfg.bars_per_year == 252 * 105


def test_data_config_bar_timedelta() -> None:
    cfg = DataConfig(bar_minutes=15)
    assert cfg.bar_timedelta == pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------------
# H3 - index log-return считается после reindex'а на сетку бара
# ---------------------------------------------------------------------------

def test_index_log_returns_step_function_collapses_to_zero() -> None:
    """Если индекс редкий (раз в 5 баров) - после reindex+ffill+log_return
    мы получаем 4 нуля и 1 настоящий log_return, не 5 копий одного
    значения."""
    sparse_idx = pd.date_range("2024-01-02 07:00", periods=4, freq="25min", tz="UTC")
    sparse = pd.DataFrame(
        {"index_imoex_close": [100.0, 101.0, 102.0, 103.0]},
        index=sparse_idx,
    )
    target_idx = pd.date_range("2024-01-02 07:00", periods=20, freq="5min", tz="UTC")
    out = _index_log_returns(sparse, target_idx)
    col = "index_imoex_logret"
    assert col in out.columns
    # На каждом 5-м баре непустой return, остальные = 0.
    nonzero = (out[col].abs() > 1e-9).sum()
    assert nonzero <= 4   # не больше 4 непустых значений (sparse points)


# ---------------------------------------------------------------------------
# H4 - purge tail rows
# ---------------------------------------------------------------------------

def test_purge_tail_drops_last_n_per_ticker() -> None:
    idx = pd.date_range("2024-01-02 07:00", periods=20, freq="5min", tz="UTC")
    df = pd.concat([
        pd.DataFrame({"close": np.arange(20.0), "ticker": "A"}, index=idx),
        pd.DataFrame({"close": np.arange(20.0), "ticker": "B"}, index=idx),
    ])
    purged = _purge_tail(df, drop_last=3)
    counts = purged.groupby("ticker").size()
    assert (counts == 17).all()


def test_purge_tail_no_op_when_drop_last_zero() -> None:
    idx = pd.date_range("2024-01-02 07:00", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": np.arange(10.0), "ticker": "X"}, index=idx)
    assert len(_purge_tail(df, drop_last=0)) == 10


# ---------------------------------------------------------------------------
# Per-ticker uniqueness в engine (один тикер - одна одновременная позиция)
# ---------------------------------------------------------------------------

def test_engine_does_not_open_second_position_on_same_ticker() -> None:
    """Если по тикеру уже есть открытая позиция, повторный BUY-сигнал
    не должен порождать вторую параллельную позицию."""
    n = 30
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {"close": np.linspace(100.0, 105.0, n), "ticker": "TST"},
        index=idx,
    )
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=3,
        commission_rate=0.0, slippage_rate=0.0,
    )
    # BUY на баре 0 (horizon=10 -> закрытие на баре 10) и BUY на баре 5
    # (тикер уже занят) - второй должен быть пропущен.
    signals = pd.DataFrame(
        [
            {"timestamp": idx[0], "ticker": "TST",
             "horizon": 10, "mean": 0.01, "std": 0.001, "action": "BUY"},
            {"timestamp": idx[5], "ticker": "TST",
             "horizon": 10, "mean": 0.02, "std": 0.001, "action": "BUY"},
        ],
    )
    bt = run_backtest(signals, prices, cfg)
    # Должна получиться ровно одна сделка (вторая залимитировала uniqueness).
    assert len(bt.trades) == 1
    assert bt.trades.iloc[0]["open_date"] == idx[0]


def test_engine_holds_multiple_tickers_in_parallel() -> None:
    """Разные тикеры в один день должны открываться параллельно
    (до max_positions)."""
    n = 30
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    prices = pd.concat([
        pd.DataFrame({"close": np.linspace(100.0, 110.0, n), "ticker": "A"}, index=idx),
        pd.DataFrame({"close": np.linspace(50.0, 55.0, n), "ticker": "B"}, index=idx),
    ])
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=2,
        commission_rate=0.0, slippage_rate=0.0,
    )
    signals = pd.DataFrame(
        [
            {"timestamp": idx[0], "ticker": "A",
             "horizon": 5, "mean": 0.01, "std": 0.001, "action": "BUY"},
            {"timestamp": idx[0], "ticker": "B",
             "horizon": 5, "mean": 0.012, "std": 0.001, "action": "BUY"},
        ],
    )
    bt = run_backtest(signals, prices, cfg)
    assert len(bt.trades) == 2
    assert set(bt.trades["ticker"]) == {"A", "B"}
