"""Тесты модуля 2: индикаторы, таргеты, скейлер, окна."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.features import (
    StandardScaler,
    add_technical_indicators,
    make_sliding_windows,
    normalized_log_returns,
)


def _sample_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.standard_normal(n) * 0.1
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_indicators_add_columns() -> None:
    df = _sample_ohlcv()
    out = add_technical_indicators(df)
    assert "log_return" in out.columns
    assert any(c.startswith("sma_") for c in out.columns)
    assert any(c.startswith("vol_") for c in out.columns)
    assert "rsi_14" in out.columns
    # RSI ограничен [0, 1] после деления на 100.
    assert out["rsi_14"].between(0, 1).all()


def test_normalized_log_returns_aggregates_to_simple_return() -> None:
    close = pd.Series([100.0, 101.0, 102.5, 103.0, 105.0])
    targets = normalized_log_returns(close, horizons=(1, 2))
    # Проверка, что target_h2 равен среднему лог-доходностей за 2 шага.
    expected_h2 = (np.log(close.iloc[2] / close.iloc[0])) / 2
    assert np.isclose(targets["target_h2"].iloc[0], expected_h2)


def test_scaler_fits_only_on_train() -> None:
    df = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) * 2})
    scaler = StandardScaler()
    train = df.iloc[:80]
    test = df.iloc[80:]
    scaler.fit(train, ["a", "b"])
    transformed = scaler.transform(test)
    # mean/std были посчитаны на train, поэтому test не центрируется в 0.
    assert abs(transformed["a"].mean()) > 0.5


def test_sliding_windows_shape_and_no_nans() -> None:
    df = pd.DataFrame(
        {
            "f1": np.arange(50, dtype=float),
            "f2": np.arange(50, dtype=float) ** 0.5,
            "y": np.linspace(-0.01, 0.01, 50),
        },
        index=pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
    )
    x, y, ts = make_sliding_windows(df, ["f1", "f2"], ["y"], window=10)
    assert x.shape == (50 - 10 + 1, 10, 2)
    assert y.shape[0] == x.shape[0]
    assert ts.shape[0] == x.shape[0]
    assert not np.isnan(x).any()
