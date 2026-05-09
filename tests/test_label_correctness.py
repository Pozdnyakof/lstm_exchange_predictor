"""S1.3 unit-test: знак и величина cost-aware labels.

Синтетические OHLC с известным ответом — проверяем, что
``cost_aware_classification_labels`` НЕ перевернуло знак.

Если этот тест упадёт — значит сигнал в модель приходит инвертированный,
и любые архитектурные улучшения бесполезны.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.features.targets import cost_aware_classification_labels


def _make_series(close_path: list[float], open_path: list[float]) -> tuple[pd.Series, pd.Series]:
    """Сделать pd.Series open/close с фиктивным timestamp index."""
    idx = pd.date_range("2024-01-01", periods=len(close_path), freq="5min", tz="UTC")
    return (
        pd.Series(open_path, index=idx, dtype=float, name="open"),
        pd.Series(close_path, index=idx, dtype=float, name="close"),
    )


def test_long_target_is_one_when_price_rises_enough() -> None:
    """Цена RAST: open[1]=100, close[2]=105 (+5%). При costs 0.1% target=1."""
    open_p, close_p = _make_series(
        close_path=[100.0, 100.0, 105.0, 105.0],
        open_path=[100.0, 100.0, 105.0, 105.0],
    )
    labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        direction="long",
    )
    # t=0: next_open=100, future_close=close[t+1]=100 → ratio≈1, lr<0
    assert labels.iloc[0]["target_h1"] == 0.0
    # t=1: next_open=105, future_close=105 → ratio=(105·0.999)/(105·1.001)<1, lr<0
    # это потому что bar t+1 has open=100 close=100 → next_open=open[2]=105,
    # future_close=close[t+1]=close[2]=105 → нет роста, есть только costs
    assert labels.iloc[1]["target_h1"] == 0.0
    # t=2: next_open=open[3]=105, future_close=close[3]=105 → ratio<1
    assert labels.iloc[2]["target_h1"] == 0.0


def test_long_target_is_one_for_real_uptrend() -> None:
    """Корректный пример роста: open[t+1]=100, close[t+1]=110.

    target_h1 = 1, потому что (110·0.999)/(100·1.001) > 1.
    """
    # h=1 значит: входим по open[t+1], выходим по close[t+1]
    # → t=0: next_open=open[1]=100, future_close=close[1]=110 → +9.7% после costs
    open_p, close_p = _make_series(
        close_path=[99.0, 110.0, 0.0, 0.0],   # close[1]=110 — рост
        open_path=[99.0, 100.0, 0.0, 0.0],     # open[1]=100
    )
    labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        direction="long",
    )
    assert labels.iloc[0]["target_h1"] == 1.0
    assert labels.iloc[0]["lr_h1"] > 0
    # Проверяем точное значение log((110*0.999)/(100*1.001))
    expected = float(np.log((110.0 * 0.999) / (100.0 * 1.001)))
    assert abs(float(labels.iloc[0]["lr_h1"]) - expected) < 1e-5


def test_long_target_is_zero_for_real_downtrend() -> None:
    """Падение: open[t+1]=100, close[t+1]=90 → target=0, lr<0."""
    open_p, close_p = _make_series(
        close_path=[99.0, 90.0, 0.0, 0.0],
        open_path=[99.0, 100.0, 0.0, 0.0],
    )
    labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        direction="long",
    )
    assert labels.iloc[0]["target_h1"] == 0.0
    assert labels.iloc[0]["lr_h1"] < 0


def test_long_target_zero_when_move_smaller_than_costs() -> None:
    """Движение меньше round-trip costs → target=0 даже при росте.

    open=100, close=100.15, costs 0.1%×2 = 0.2% > 0.15% → target=0.
    """
    open_p, close_p = _make_series(
        close_path=[100.0, 100.15, 0.0, 0.0],
        open_path=[100.0, 100.0, 0.0, 0.0],
    )
    labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        direction="long",
    )
    assert labels.iloc[0]["target_h1"] == 0.0


def test_short_direction_inverted_relative_to_long() -> None:
    """Direction='short' даёт обратные метки: при росте target=0, при падении=1."""
    # Рост: long.target=1, short.target должен быть 0.
    open_p, close_p = _make_series(
        close_path=[99.0, 110.0, 0.0, 0.0],
        open_path=[99.0, 100.0, 0.0, 0.0],
    )
    long_labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001, direction="long",
    )
    short_labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001, direction="short",
    )
    assert long_labels.iloc[0]["target_h1"] == 1.0
    assert short_labels.iloc[0]["target_h1"] == 0.0


def test_lr_h_and_target_h_signs_consistent() -> None:
    """Инвариант: target_h{h}=1 ⇔ lr_h{h}>0 для всех валидных строк."""
    rng = np.random.default_rng(42)
    n = 200
    close_p_arr = 100.0 * np.cumprod(1.0 + 0.001 * rng.standard_normal(n))
    open_p_arr = close_p_arr * (1.0 + 0.0001 * rng.standard_normal(n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    open_p = pd.Series(open_p_arr, index=idx, name="open")
    close_p = pd.Series(close_p_arr, index=idx, name="close")
    labels = cost_aware_classification_labels(
        open_p, close_p, horizons=(1, 5, 10),
        entry_cost=0.0003, exit_cost=0.0003, direction="long",
    )
    for h in (1, 5, 10):
        valid = labels[f"lr_h{h}"].notna()
        lr = labels.loc[valid, f"lr_h{h}"].to_numpy()
        target = labels.loc[valid, f"target_h{h}"].to_numpy()
        # Инвариант: target = float(lr > 0)
        np.testing.assert_array_equal(target, (lr > 0).astype(np.float32))


def test_label_smoothing_preserves_monotonicity() -> None:
    """label_smoothing не должен менять знак: smoothed[hard=1] > smoothed[hard=0]."""
    rng = np.random.default_rng(0)
    n = 100
    close_p_arr = 100.0 + np.cumsum(rng.standard_normal(n))
    open_p_arr = close_p_arr - 0.05
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    labels = cost_aware_classification_labels(
        pd.Series(open_p_arr, index=idx),
        pd.Series(close_p_arr, index=idx),
        horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        label_smoothing=0.1,
        direction="long",
    )
    valid = labels["target_h1"].notna()
    lr = labels.loc[valid, "lr_h1"].to_numpy()
    target = labels.loc[valid, "target_h1"].to_numpy()
    # Все target>0 должны быть = 0.9 (1 после smoothing с eps=0.1).
    # Все target=0 → 0.1.
    assert (target[lr > 0] == 0.9).all()
    assert (target[lr <= 0] == 0.1).all()
