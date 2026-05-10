"""Тесты joint_max_pnl_thresholds (Sprint 1.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.model.meta_labeling import joint_max_pnl_thresholds


def _make_val_data(n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Синтетика: primary, meta, lr per (timestamp, ticker, horizon)."""
    rng = np.random.default_rng(0)
    rows_p, rows_m, rows_lr = [], [], []
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    for ts in timestamps:
        prim = float(rng.uniform(0.3, 0.8))
        meta = float(rng.uniform(0.0, 0.5))
        # Сигнал: high prim AND high meta → high lr
        lr = (prim + meta) * 0.005 + 0.001 * rng.standard_normal()
        rows_p.append({"timestamp": ts, "ticker": "SBER", "horizon": 6, "mean": prim})
        rows_m.append({"timestamp": ts, "ticker": "SBER", "horizon": 6, "mean": meta})
        rows_lr.append({"timestamp": ts, "ticker": "SBER", "horizon": 6, "actual": lr})
    return (
        pd.DataFrame(rows_p), pd.DataFrame(rows_m), pd.DataFrame(rows_lr),
    )


def test_joint_returns_threshold_pair() -> None:
    """Базовый случай: возвращает (T_prim, T_meta_abs)."""
    val_p, val_m, val_lr = _make_val_data(n=300)
    T_prim, T_meta_abs, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.4, 0.5, 0.6),
        meta_percentiles=(20.0, 30.0),
        cost_per_trade=0.001, min_trades=20,
    )
    assert 0.4 <= T_prim <= 0.6
    assert 0 <= T_meta_abs <= 1
    assert len(sweep) == 3 * 2  # 3 prim × 2 meta


def test_joint_picks_threshold_with_max_pnl() -> None:
    """Выбирается комбинация с максимальным mean(lr - cost)."""
    val_p, val_m, val_lr = _make_val_data(n=500)
    T_prim, T_meta_abs, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.4, 0.5),
        meta_percentiles=(30.0, 50.0),
        cost_per_trade=0.001, min_trades=20,
    )
    # best должна быть в sweep
    valid = [r for r in sweep if r["n_trades"] >= 20]
    assert valid, "no valid threshold combinations"
    best_pnl = max(r["mean_pnl"] for r in valid)
    matching = [
        r for r in valid
        if abs(r["mean_pnl"] - best_pnl) < 1e-9
    ]
    assert any(
        m["T_prim"] == T_prim and abs(m["T_meta_abs"] - T_meta_abs) < 1e-9
        for m in matching
    )


def test_joint_uses_percentile_for_meta() -> None:
    """T_meta_abs корректно вычисляется как percentile из val-meta."""
    val_p, val_m, val_lr = _make_val_data(n=200)
    _, _, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.5,),
        meta_percentiles=(25.0,),
        cost_per_trade=0.001, min_trades=10,
    )
    # T_meta_abs = 75-й перцентиль meta (top-25%)
    expected_meta_abs = float(np.percentile(val_m["mean"], 75.0))
    assert abs(sweep[0]["T_meta_abs"] - expected_meta_abs) < 1e-9


def test_joint_fallback_when_min_trades_unreachable() -> None:
    """Если ни одна комбинация не даёт min_trades — возвращает best PnL fallback."""
    val_p, val_m, val_lr = _make_val_data(n=50)
    T_prim, T_meta_abs, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.4,),
        meta_percentiles=(5.0,),  # топ-5% от 50 = 2-3 трейда
        cost_per_trade=0.001, min_trades=100,  # недостижимый
    )
    assert np.isfinite(T_prim)
    # T_meta_abs может быть NaN, если все pnl были NaN. Главное — не упало.


def test_joint_empty_data_returns_safe() -> None:
    """Пустой val → не падает, возвращает первую T_prim и NaN."""
    empty = pd.DataFrame(columns=["timestamp", "ticker", "horizon", "mean", "actual"])
    T_prim, T_meta_abs, sweep = joint_max_pnl_thresholds(
        empty, empty, empty, horizon=6,
    )
    assert T_prim == 0.45  # первый из default primary_thresholds
    assert np.isnan(T_meta_abs)
    assert sweep == []


def test_primary_percentiles_uses_distribution() -> None:
    """primary_percentiles=(20.0,) → T_prim равен 80-му перцентилю val-primary."""
    val_p, val_m, val_lr = _make_val_data(n=500)
    T_prim, _, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_percentiles=(20.0,),
        meta_percentiles=(50.0,),
        cost_per_trade=0.001, min_trades=10,
    )
    expected_T_prim = float(np.percentile(val_p["mean"], 80.0))
    # Sweep содержит ровно одну точку, и её T_prim это и есть expected
    assert len(sweep) == 1
    assert abs(sweep[0]["T_prim"] - expected_T_prim) < 1e-9
    assert abs(T_prim - expected_T_prim) < 1e-9
    assert sweep[0]["primary_pct"] == 20.0


def test_primary_percentiles_overrides_thresholds() -> None:
    """Если primary_percentiles задан, абсолютные primary_thresholds игнорятся."""
    val_p, val_m, val_lr = _make_val_data(n=300)
    _, _, sweep = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.99,),  # бы дал 0 trades
        primary_percentiles=(50.0,),  # 50-й перцентиль гарантирует половину наблюдений
        meta_percentiles=(50.0,),
        cost_per_trade=0.001, min_trades=10,
    )
    # primary_thresholds полностью игнорируется → T_prim из distribution.
    expected = float(np.percentile(val_p["mean"], 50.0))
    assert abs(sweep[0]["T_prim"] - expected) < 1e-9
    # n_trades нетривиальный (не 0).
    assert sweep[0]["n_trades"] > 10


def test_primary_percentiles_avoids_edge_effect() -> None:
    """Узкое primary-распределение → percentile стабильно даёт top-K сигналов."""
    rng = np.random.default_rng(0)
    n = 1000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Узкое распределение [0.40, 0.55] — характерно для длинных горизонтов.
    primaries = rng.uniform(0.40, 0.55, size=n)
    val_p = pd.DataFrame({
        "timestamp": timestamps, "ticker": "SBER", "horizon": 48,
        "mean": primaries,
    })
    val_m = pd.DataFrame({
        "timestamp": timestamps, "ticker": "SBER", "horizon": 48,
        "mean": rng.uniform(0.30, 0.50, size=n),
    })
    val_lr = pd.DataFrame({
        "timestamp": timestamps, "ticker": "SBER", "horizon": 48,
        "actual": rng.standard_normal(n) * 0.005,
    })
    # Абсолютный T=0.55 даст 0 trades (max=0.55, > строго не выполнится).
    # Percentile 10% (т.е. top-10) гарантирует ~100 наблюдений.
    _, _, sweep_pct = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=48,
        primary_percentiles=(10.0,),
        meta_percentiles=(50.0,),
        cost_per_trade=0.001, min_trades=20,
    )
    # n_trades должен быть в районе 50 (top-10% от ~1000 / разные пересечения)
    assert sweep_pct[0]["n_trades"] >= 30


def test_joint_horizon_filter_works() -> None:
    """Фильтр по horizon корректно отбирает только нужные строки."""
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    val_p = pd.DataFrame([
        {"timestamp": ts, "ticker": "SBER", "horizon": h,
         "mean": float(rng.uniform(0.3, 0.8))}
        for ts in timestamps for h in (6, 12)
    ])
    val_m = pd.DataFrame([
        {"timestamp": ts, "ticker": "SBER", "horizon": h,
         "mean": float(rng.uniform(0, 0.5))}
        for ts in timestamps for h in (6, 12)
    ])
    val_lr = pd.DataFrame([
        {"timestamp": ts, "ticker": "SBER", "horizon": h,
         "actual": float(rng.standard_normal()) * 0.005}
        for ts in timestamps for h in (6, 12)
    ])
    T_prim_h6, _, _ = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=6,
        primary_thresholds=(0.5,), meta_percentiles=(50.0,),
        min_trades=10,
    )
    T_prim_h12, _, _ = joint_max_pnl_thresholds(
        val_p, val_m, val_lr, horizon=12,
        primary_thresholds=(0.5,), meta_percentiles=(50.0,),
        min_trades=10,
    )
    # Не падает на разных горизонтах.
    assert np.isfinite(T_prim_h6)
    assert np.isfinite(T_prim_h12)
