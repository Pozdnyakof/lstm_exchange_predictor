"""Тесты расширенных индикаторов и темпоральных признаков."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.features import add_advanced_indicators, add_technical_indicators


def _ohlcv(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    close = 100 + rng.standard_normal(n).cumsum() * 0.05
    high = close + rng.uniform(0.05, 0.30, n)
    low = close - rng.uniform(0.05, 0.30, n)
    open_ = close + rng.standard_normal(n) * 0.02
    vol = rng.integers(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_advanced_adds_macd_bb_stoch_williams() -> None:
    df = add_technical_indicators(_ohlcv())
    out = add_advanced_indicators(df, market="moex")
    expected = {
        "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth",
        "stoch_k", "stoch_d", "williams_r", "cci_20",
        "roc_5", "roc_10", "roc_20",
        "obv_zscore", "mfi_14",
    }
    missing = expected - set(out.columns)
    assert not missing, f"missing: {missing}"


def test_advanced_temporal_features_present() -> None:
    df = add_technical_indicators(_ohlcv())
    out = add_advanced_indicators(df, market="moex", use_temporal=True)
    for col in ["time_sin_daily", "time_cos_daily", "dow_sin", "dow_cos", "dom_sin", "dom_cos"]:
        assert col in out.columns
        # sin/cos должны лежать в [-1, 1].
        assert out[col].abs().max() <= 1.0 + 1e-9


def test_hurst_in_valid_range() -> None:
    df = add_technical_indicators(_ohlcv(800))
    out = add_advanced_indicators(df, hurst_window=60, use_hurst=True)
    hurst = out["hurst_60"].dropna()
    assert len(hurst) > 0
    # Theoretically Hurst in (0, 1); allow small numerical slack.
    assert hurst.between(0.0, 1.5).all()


def test_macd_hist_equals_diff_of_macd_and_signal() -> None:
    df = add_technical_indicators(_ohlcv(400))
    out = add_advanced_indicators(df)
    diff = (out["macd"] - out["macd_signal"] - out["macd_hist"]).dropna()
    assert (diff.abs() < 1e-9).all()


def test_bollinger_pct_b_typical_range() -> None:
    df = add_technical_indicators(_ohlcv(400))
    out = add_advanced_indicators(df)
    pct_b = out["bb_pct_b"].dropna()
    # Большинство значений в [0, 1] (внутри полос); хвосты допустимы.
    in_band = pct_b.between(0.0, 1.0).mean()
    assert in_band > 0.7
