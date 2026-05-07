"""Тесты для split-conformal фильтра сделок."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graduate_work.config import TradingConfig
from graduate_work.strategy import (
    ConformalSignalGenerator,
    build_predictions_frame,
)


def _make_predictions(probs: np.ndarray, *, ticker: str = "A") -> pd.DataFrame:
    n = probs.shape[0]
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    std = np.full_like(probs, 0.05, dtype=np.float32)
    return build_predictions_frame(
        timestamps=np.array(timestamps),
        tickers=np.array([ticker] * n),
        mean=probs.astype(np.float32),
        std=std.astype(np.float32),
        horizons=tuple(range(1, probs.shape[1] + 1)),
    )


def test_conformal_alpha_validation() -> None:
    cfg = TradingConfig()
    with pytest.raises(ValueError):
        ConformalSignalGenerator(cfg, alpha=0.0)
    with pytest.raises(ValueError):
        ConformalSignalGenerator(cfg, alpha=1.0)


def test_conformal_calibrate_empty_falls_back_to_half() -> None:
    cfg = TradingConfig()
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    res = gen.calibrate(pd.DataFrame(), pd.DataFrame())
    assert res.quantile == 0.5
    assert res.threshold == 0.5
    assert res.n_val_scores == 0
    assert gen.quantile == 0.5


def test_conformal_calibrate_computes_quantile() -> None:
    """Calibrate на известных скорах должен дать (1-α)·(n+1)/n квантиль."""
    rng = np.random.default_rng(0)
    n = 200
    # Модель предсказывает ~0.5, target — bernoulli ~0.5; |prob-actual| ≈ 0.5.
    probs = rng.uniform(0.45, 0.55, (n, 1))
    val_pred = _make_predictions(probs)
    actual_vals = rng.integers(0, 2, n).astype(float)
    val_targets = val_pred[["timestamp", "ticker", "horizon"]].copy()
    val_targets["actual"] = actual_vals

    cfg = TradingConfig()
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    res = gen.calibrate(val_pred, val_targets)
    # Скоры |prob - 0/1| ∈ [0.45, 0.55]; квантиль около 0.55.
    assert 0.4 <= res.quantile <= 0.6
    assert res.n_val_scores == n
    # threshold = max(q, 1-q) ≥ 0.5.
    assert res.threshold >= 0.5


def test_conformal_generate_filters_below_threshold() -> None:
    """generate() пропускает только prob > max(q, 1-q)."""
    cfg = TradingConfig(max_positions=4)
    gen = ConformalSignalGenerator(cfg, alpha=0.1)

    # Вручную устанавливаем квантиль = 0.55 → threshold=0.55.
    gen._quantile = 0.55

    timestamps = np.array(["2024-01-01"] * 4, dtype="datetime64[ns]")
    tickers = np.array(["A", "B", "C", "D"])
    # 0.50, 0.54 < 0.55; 0.60, 0.70 > 0.55.
    mean = np.array([[0.50], [0.54], [0.60], [0.70]], dtype=np.float32)
    std = np.full((4, 1), 0.05, dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1,))

    sigs = gen.generate(df)
    buys = sigs[sigs["action"] == "BUY"]
    assert set(buys["ticker"]) == {"C", "D"}


def test_conformal_generate_respects_max_positions() -> None:
    """Per-session порядок: top-N по prob, остальные HOLD."""
    cfg = TradingConfig(max_positions=2)
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    gen._quantile = 0.55

    timestamps = np.array(["2024-01-01"] * 4, dtype="datetime64[ns]")
    tickers = np.array(["A", "B", "C", "D"])
    # Все над порогом, но max_positions=2.
    mean = np.array([[0.60], [0.65], [0.70], [0.75]], dtype=np.float32)
    std = np.full((4, 1), 0.05, dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1,))

    sigs = gen.generate(df)
    buys = sigs[sigs["action"] == "BUY"]
    # Только топ-2 (D, C) по mean.
    assert set(buys["ticker"]) == {"C", "D"}
    assert len(buys) == 2


def test_conformal_generate_empty_predictions() -> None:
    cfg = TradingConfig()
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    gen._quantile = 0.5
    out = gen.generate(pd.DataFrame())
    assert out.empty
    assert "action" in out.columns
    assert "signal" in out.columns


def test_conformal_uncalibrated_uses_fallback() -> None:
    """Если generate вызван без calibrate — используется fallback 0.5."""
    cfg = TradingConfig(max_positions=2)
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    # Не вызываем calibrate.
    timestamps = np.array(["2024-01-01"] * 2, dtype="datetime64[ns]")
    tickers = np.array(["A", "B"])
    mean = np.array([[0.4], [0.6]], dtype=np.float32)
    std = np.full((2, 1), 0.05, dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1,))
    sigs = gen.generate(df)
    # threshold = max(0.5, 0.5) = 0.5; B (0.6) > 0.5, A (0.4) < 0.5.
    buys = sigs[sigs["action"] == "BUY"]
    assert set(buys["ticker"]) == {"B"}


def test_conformal_picks_best_horizon_per_ticker() -> None:
    """Если несколько горизонтов — берём argmax по prob."""
    cfg = TradingConfig(max_positions=1)
    gen = ConformalSignalGenerator(cfg, alpha=0.1)
    gen._quantile = 0.55

    timestamps = np.array(["2024-01-01"], dtype="datetime64[ns]")
    tickers = np.array(["A"])
    # h=1: 0.5, h=3: 0.7 → должен выбрать h=3.
    mean = np.array([[0.5, 0.7]], dtype=np.float32)
    std = np.array([[0.05, 0.05]], dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1, 3))
    sigs = gen.generate(df)
    buys = sigs[sigs["action"] == "BUY"]
    assert len(buys) == 1
    assert buys.iloc[0]["horizon"] == 3
    assert buys.iloc[0]["mean"] == pytest.approx(0.7, abs=1e-6)
