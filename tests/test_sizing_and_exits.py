"""Тесты position-sizing modes и intra-bar SL/PT в backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.backtest.engine import run_backtest
from graduate_work.config import TradingConfig
from graduate_work.model.kelly_sizing import signal_kelly_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlc_prices(n: int = 30) -> pd.DataFrame:
    """Линейный uptrend для одного тикера с OHLC (high=close+1, low=open-1)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = np.linspace(100.0, 130.0, n)
    opens = np.concatenate([[closes[0]], closes[:-1]])  # open[t] ≈ close[t-1]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "ticker": "SBER",
        },
        index=idx,
    )


def _single_buy_signal(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": prices.index[0],
        "ticker": "SBER",
        "horizon": horizon,
        "mean": 0.7,
        "std": 0.05,
        "action": "BUY",
    }])


# ---------------------------------------------------------------------------
# Sizing modes
# ---------------------------------------------------------------------------

def test_equal_split_uses_full_cash_when_one_slot() -> None:
    """sizing_mode=equal_split + max_positions=1 → весь cash в одну позицию."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="equal_split",
        commission_rate=0.0001,
        slippage_rate=0.0001,
    )
    bt = run_backtest(_single_buy_signal(prices), prices, cfg)
    assert len(bt.trades) == 1
    invested = bt.trades.iloc[0]["entry_price"] * bt.trades.iloc[0]["quantity"]
    # Около 100k (минус fees_in)
    assert invested > 99_000
    assert invested < 100_001


def test_fixed_frac_invests_only_configured_share() -> None:
    """sizing_mode=fixed_frac → invested ≈ initial_capital * fraction."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=5,
        sizing_mode="fixed_frac",
        position_size_fraction=0.10,
        commission_rate=0.0001,
        slippage_rate=0.0001,
    )
    bt = run_backtest(_single_buy_signal(prices), prices, cfg)
    assert len(bt.trades) == 1
    invested = bt.trades.iloc[0]["entry_price"] * bt.trades.iloc[0]["quantity"]
    # ~10k, минус fees
    assert 9_500 < invested < 10_001


def test_max_position_size_caps_fixed_frac() -> None:
    """max_position_size_fraction обрезает даже большую position_size_fraction."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="fixed_frac",
        position_size_fraction=0.50,
        max_position_size_fraction=0.10,
        commission_rate=0.0001,
        slippage_rate=0.0001,
    )
    bt = run_backtest(_single_buy_signal(prices), prices, cfg)
    invested = bt.trades.iloc[0]["entry_price"] * bt.trades.iloc[0]["quantity"]
    # Клип по max_position_size_fraction=10%
    assert invested < 10_001


def test_signal_kelly_uses_size_fraction_column() -> None:
    """sizing_mode=signal_kelly читает size_fraction из строки сигнала."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=5,
        sizing_mode="signal_kelly",
        max_position_size_fraction=0.50,
        commission_rate=0.0001,
        slippage_rate=0.0001,
    )
    sig = _single_buy_signal(prices)
    sig["size_fraction"] = 0.07
    bt = run_backtest(sig, prices, cfg)
    invested = bt.trades.iloc[0]["entry_price"] * bt.trades.iloc[0]["quantity"]
    # ~7k
    assert 6_500 < invested < 7_100


def test_signal_kelly_falls_back_to_fixed_frac_when_no_column() -> None:
    """signal_kelly без size_fraction → берёт position_size_fraction."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=5,
        sizing_mode="signal_kelly",
        position_size_fraction=0.05,
        commission_rate=0.0001,
        slippage_rate=0.0001,
    )
    bt = run_backtest(_single_buy_signal(prices), prices, cfg)
    invested = bt.trades.iloc[0]["entry_price"] * bt.trades.iloc[0]["quantity"]
    # ~5k
    assert 4_500 < invested < 5_100


def test_unknown_sizing_mode_raises() -> None:
    prices = _ohlc_prices()
    cfg = TradingConfig(sizing_mode="bogus", max_positions=1)
    try:
        run_backtest(_single_buy_signal(prices), prices, cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Stop-loss / Profit-target
# ---------------------------------------------------------------------------

def test_stop_loss_triggers_intra_bar() -> None:
    """SL: low падает ниже entry*(1-sl_pct) → раннее закрытие."""
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    # На баре 3 цена обвалилась до 90 (low=89), entry будет на close[0]=100
    # → SL=98 (2%) сработает на баре 3.
    lows[3] = 89.0
    closes[3] = 90.0
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "ticker": "SBER"},
        index=idx,
    )
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="fixed_frac",
        position_size_fraction=0.10,
        stop_loss_pct=0.02,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    sig = pd.DataFrame([{
        "timestamp": idx[0], "ticker": "SBER", "horizon": 10,
        "mean": 0.7, "std": 0.0, "action": "BUY",
    }])
    bt = run_backtest(sig, df, cfg)
    assert len(bt.trades) == 1
    trade = bt.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    # Закрылись на цене ~98 (entry*0.98)
    assert abs(trade["exit_price"] - 98.0) < 1e-6


def test_profit_target_triggers_intra_bar() -> None:
    """PT: high превысил entry*(1+pt_pct) → раннее закрытие в плюс."""
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    # На баре 3 high=105 → PT=103 (3%) сработает.
    highs[3] = 105.0
    closes[3] = 104.0
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "ticker": "SBER"},
        index=idx,
    )
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="fixed_frac",
        position_size_fraction=0.10,
        profit_target_pct=0.03,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    sig = pd.DataFrame([{
        "timestamp": idx[0], "ticker": "SBER", "horizon": 10,
        "mean": 0.7, "std": 0.0, "action": "BUY",
    }])
    bt = run_backtest(sig, df, cfg)
    assert len(bt.trades) == 1
    trade = bt.trades.iloc[0]
    assert trade["exit_reason"] == "profit_target"
    assert abs(trade["exit_price"] - 103.0) < 1e-6


def test_no_sl_pt_means_horizon_exit() -> None:
    """Без SL/PT → exit_reason='horizon' (старое поведение)."""
    prices = _ohlc_prices()
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="equal_split",
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    bt = run_backtest(_single_buy_signal(prices, horizon=5), prices, cfg)
    assert bt.trades.iloc[0]["exit_reason"] == "horizon"


def test_sl_pt_ignored_when_no_high_low_columns() -> None:
    """Если в prices нет high/low — SL/PT просто не срабатывают (no crash)."""
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"close": np.linspace(100.0, 110.0, n), "ticker": "SBER"},
        index=idx,
    )
    cfg = TradingConfig(
        initial_capital=100_000,
        max_positions=1,
        sizing_mode="fixed_frac",
        position_size_fraction=0.10,
        stop_loss_pct=0.02,
        profit_target_pct=0.05,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    sig = pd.DataFrame([{
        "timestamp": idx[0], "ticker": "SBER", "horizon": 5,
        "mean": 0.7, "std": 0.0, "action": "BUY",
    }])
    bt = run_backtest(sig, df, cfg)
    assert len(bt.trades) == 1
    # Без high/low → SL/PT не сработали, exit по horizon.
    assert bt.trades.iloc[0]["exit_reason"] == "horizon"


# ---------------------------------------------------------------------------
# Kelly sizing helper
# ---------------------------------------------------------------------------

def test_kelly_zero_when_below_floor() -> None:
    """primary < floor → size_fraction = 0."""
    sig = pd.DataFrame([
        {"timestamp": pd.Timestamp("2024-01-01"), "ticker": "X",
         "horizon": 5, "mean": 0.45, "meta": 0.60, "action": "BUY"},
    ])
    out = signal_kelly_size(
        sig, kelly_primary_floor=0.50, kelly_meta_floor=0.50,
    )
    assert out["size_fraction"].iloc[0] == 0.0


def test_kelly_max_at_full_certainty() -> None:
    """primary=meta=1.0 → size_fraction достигает кэпа."""
    sig = pd.DataFrame([
        {"timestamp": pd.Timestamp("2024-01-01"), "ticker": "X",
         "horizon": 5, "mean": 1.0, "meta": 1.0, "action": "BUY"},
    ])
    out = signal_kelly_size(
        sig, kelly_scale=0.5, max_position_size_fraction=0.20,
    )
    # raw = 1.0 * 1.0 = 1.0, * scale=0.5 = 0.5, clip к 0.20.
    assert abs(out["size_fraction"].iloc[0] - 0.20) < 1e-6


def test_kelly_works_without_meta_column() -> None:
    """Если meta нет — используется только primary edge."""
    sig = pd.DataFrame([
        {"timestamp": pd.Timestamp("2024-01-01"), "ticker": "X",
         "horizon": 5, "mean": 0.75, "action": "BUY"},
    ])
    out = signal_kelly_size(
        sig, meta_col="meta",  # колонки нет — graceful
        kelly_scale=0.5, kelly_primary_floor=0.50,
        max_position_size_fraction=0.50,
    )
    # edge = (0.75-0.5)/(1-0.5) = 0.5; * scale=0.5 = 0.25
    assert abs(out["size_fraction"].iloc[0] - 0.25) < 1e-3


def test_kelly_monotonic_in_primary() -> None:
    """Больше Primary → больше size_fraction (при том же Meta)."""
    sig = pd.DataFrame([
        {"mean": 0.55, "meta": 0.7},
        {"mean": 0.65, "meta": 0.7},
        {"mean": 0.80, "meta": 0.7},
    ])
    out = signal_kelly_size(sig)
    sizes = out["size_fraction"].to_numpy()
    assert sizes[0] < sizes[1] < sizes[2]
