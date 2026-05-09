"""Тесты cross-sectional features: rank, z-score, relative."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.features.cross_sectional import (
    add_cross_sectional_features,
    cross_sectional_rank,
    cross_sectional_relative,
    cross_sectional_zscore,
    stack_panel,
)


def _wide_panel(rows: list[list[float]], tickers: list[str]) -> pd.DataFrame:
    """Сделать (timestamp × ticker) wide-panel."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=tickers)


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def test_rank_basic() -> None:
    """Rank in [0, 1], minimum=0, maximum=1 в строке с 3 различными значениями."""
    panel = _wide_panel([[0.1, 0.5, 0.3]], ["A", "B", "C"])
    ranks = cross_sectional_rank(panel)
    row = ranks.iloc[0]
    assert row["A"] == 0.0
    assert row["B"] == 1.0
    assert row["C"] == 0.5


def test_rank_neutral_for_single_value_row() -> None:
    """Если в строке только один не-NaN — rank=0.5 (нейтрал)."""
    panel = pd.DataFrame(
        [[0.5, np.nan]], columns=["A", "B"],
        index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"),
    )
    ranks = cross_sectional_rank(panel)
    assert ranks.iloc[0]["A"] == 0.5  # не одно значение → нейтрал


def test_rank_empty_panel() -> None:
    assert cross_sectional_rank(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------

def test_zscore_zero_for_constant_row() -> None:
    """Все одинаковые → std=0 → z=0."""
    panel = _wide_panel([[0.5, 0.5, 0.5]], ["A", "B", "C"])
    zs = cross_sectional_zscore(panel)
    assert (zs.iloc[0] == 0.0).all()


def test_zscore_signs_match_relative_position() -> None:
    """Самое маленькое значение → отрицательный z, самое большое → положительный."""
    panel = _wide_panel([[0.0, 1.0, 2.0]], ["A", "B", "C"])
    zs = cross_sectional_zscore(panel)
    row = zs.iloc[0]
    assert row["A"] < 0
    assert row["C"] > 0
    assert abs(row["B"]) < 1e-6  # центральное → ≈0


# ---------------------------------------------------------------------------
# relative
# ---------------------------------------------------------------------------

def test_relative_ratio_mode() -> None:
    """ratio: x/mean - 1. mean=1.0 → положительные ratio для x>1, отриц. для <1."""
    panel = _wide_panel([[0.5, 1.0, 1.5]], ["A", "B", "C"])
    rel = cross_sectional_relative(panel, mode="ratio")
    row = rel.iloc[0]
    # mean = 1.0
    assert abs(row["A"] - (-0.5)) < 1e-6
    assert abs(row["B"] - 0.0) < 1e-6
    assert abs(row["C"] - 0.5) < 1e-6


def test_relative_diff_mode() -> None:
    """diff: x - mean. Для центрированных фич типа imbalance ∈ [-1, 1]."""
    panel = _wide_panel([[-0.5, 0.0, 0.5]], ["A", "B", "C"])
    rel = cross_sectional_relative(panel, mode="diff")
    row = rel.iloc[0]
    # mean = 0
    assert abs(row["A"] - (-0.5)) < 1e-6
    assert abs(row["B"] - 0.0) < 1e-6
    assert abs(row["C"] - 0.5) < 1e-6


def test_relative_validates_mode() -> None:
    panel = _wide_panel([[1.0, 2.0]], ["A", "B"])
    try:
        cross_sectional_relative(panel, mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad mode")


# ---------------------------------------------------------------------------
# stack_panel + add_cross_sectional_features
# ---------------------------------------------------------------------------

def test_stack_panel_combines_per_ticker() -> None:
    """Из {ticker: DataFrame} → wide (timestamp × ticker) для одной фичи."""
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    per_ticker = {
        "SBER": pd.DataFrame({"foo": [0.1, 0.2, 0.3]}, index=idx),
        "VTBR": pd.DataFrame({"foo": [0.5, 0.4, 0.3]}, index=idx),
    }
    panel = stack_panel(per_ticker, "foo")
    assert list(panel.columns) == ["SBER", "VTBR"]
    assert panel.shape == (3, 2)


def test_add_cross_sectional_features_adds_xrank_xzscore_xrel() -> None:
    """В каждый per-ticker DF добавляются {feat}_xrank, {feat}_xzscore, {feat}_xrel."""
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    per_ticker = {
        "SBER": pd.DataFrame({"vol_imb": [0.1, 0.2, 0.3]}, index=idx),
        "VTBR": pd.DataFrame({"vol_imb": [0.5, 0.4, 0.3]}, index=idx),
        "GAZP": pd.DataFrame({"vol_imb": [-0.1, 0.0, 0.1]}, index=idx),
    }
    enriched = add_cross_sectional_features(
        per_ticker, base_features=["vol_imb"],
        rank=True, zscore=True, relative_mode="diff",
    )
    for ticker in ("SBER", "VTBR", "GAZP"):
        df = enriched[ticker]
        assert "vol_imb_xrank" in df.columns
        assert "vol_imb_xzscore" in df.columns
        assert "vol_imb_xrel" in df.columns
        assert (df["vol_imb_xrank"] >= 0).all() and (df["vol_imb_xrank"] <= 1).all()


def test_add_cross_sectional_skips_missing_feature() -> None:
    """Если фича отсутствует во всех тикерах — функция не падает."""
    idx = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    per_ticker = {
        "SBER": pd.DataFrame({"a": [1.0, 2.0]}, index=idx),
    }
    out = add_cross_sectional_features(
        per_ticker, base_features=["nonexistent"],
    )
    # Без падения — просто без новых колонок
    assert "a" in out["SBER"].columns
    assert "nonexistent_xrank" not in out["SBER"].columns
