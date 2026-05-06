"""Тесты двухступенчатого фильтра сигналов."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.config import TradingConfig
from graduate_work.strategy import SignalGenerator, build_predictions_frame


def test_all_negative_means_yields_hold() -> None:
    timestamps = np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[ns]")
    tickers = np.array(["SBER", "GAZP"])
    mean = np.array([[-0.001, -0.002], [-0.0005, -0.001]], dtype=np.float32)
    std = np.array([[0.001, 0.001], [0.001, 0.001]], dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1, 5))
    cfg = TradingConfig(max_positions=2)
    sig = SignalGenerator(cfg).generate(df)
    assert (sig["action"] == "HOLD").all()


def test_high_uncertainty_blocks_buy() -> None:
    timestamps = np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[ns]")
    tickers = np.array(["SBER", "GAZP"])
    mean = np.array([[0.005, 0.001], [0.004, 0.002]], dtype=np.float32)
    std = np.array([[0.5, 0.5], [0.0001, 0.0001]], dtype=np.float32)  # SBER чрезмерно неуверен
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1, 5))
    cfg = TradingConfig(
        max_positions=2,
        min_expected_return=0.001,
        max_uncertainty=0.01,
    )
    sig = SignalGenerator(cfg).generate(df)
    actions = dict(zip(sig["ticker"], sig["action"]))
    assert actions["SBER"] == "HOLD"
    assert actions["GAZP"] == "BUY"


def test_top_k_capped_by_max_positions() -> None:
    ts = np.array(["2024-01-01"] * 5, dtype="datetime64[ns]")
    tickers = np.array(["A", "B", "C", "D", "E"])
    mean = np.array([[0.01], [0.009], [0.008], [0.007], [0.006]], dtype=np.float32)
    std = np.full((5, 1), 0.0001, dtype=np.float32)
    df = build_predictions_frame(ts, tickers, mean, std, horizons=(1,))
    cfg = TradingConfig(
        max_positions=2,
        min_expected_return=0.0,
        max_uncertainty=1.0,
    )
    sig = SignalGenerator(cfg).generate(df)
    buys = sig[sig["action"] == "BUY"]
    assert len(buys) == 2
    assert set(buys["ticker"]) == {"A", "B"}
