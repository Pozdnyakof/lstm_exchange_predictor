"""Тесты FUTOI features под реальную long-form схему ALGOPACK.

Схема (R-0053 inspection):
  ``sess_id, seqnum, tradedate, tradetime, ticker, clgroup,
   pos, pos_long, pos_short, pos_long_num, pos_short_num,
   systime, trade_session_date``

  - ``clgroup``: ``YUR`` (юрлица) или ``FIZ`` (физлица)
  - На один (date, time) — ДВЕ строки (по группе)
"""

from __future__ import annotations

import pandas as pd

from graduate_work.features.futoi_features import (
    FUTURES_TO_SPOT,
    align_to_bar_grid,
    build_futoi_features,
)


def _make_futoi_long_form(rows: list[dict]) -> pd.DataFrame:
    """Long-form FUTOI snapshot.

    Не ставим index — функция читает ``tradedate`` как колонку.
    Setting index с тем же именем создаёт ambiguous-конфликт в groupby.
    """
    return pd.DataFrame(rows)


def test_futures_to_spot_mapping() -> None:
    assert FUTURES_TO_SPOT["SBERF"] == "SBER"
    assert FUTURES_TO_SPOT["GAZPF"] == "GAZP"


def test_build_futoi_returns_empty_on_empty_input() -> None:
    assert build_futoi_features(None).empty
    assert build_futoi_features(pd.DataFrame()).empty


def test_build_futoi_extracts_yur_fiz_features() -> None:
    """С обеими группами строятся pos / imbalance / smart-divergence."""
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 100, "pos_long": 200, "pos_short": -100,
         "pos_long_num": 50, "pos_short_num": 25},
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "FIZ",
         "pos": -100, "pos_long": 50, "pos_short": -150,
         "pos_long_num": 100, "pos_short_num": 200},
        {"tradedate": "2024-10-02", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 120, "pos_long": 220, "pos_short": -100,
         "pos_long_num": 55, "pos_short_num": 25},
        {"tradedate": "2024-10-02", "tradetime": "23:50:00", "clgroup": "FIZ",
         "pos": -120, "pos_long": 50, "pos_short": -170,
         "pos_long_num": 110, "pos_short_num": 220},
    ])
    out = build_futoi_features(df)
    expected = {
        "futoi_yur_pos", "futoi_fiz_pos",
        "futoi_yur_imbalance", "futoi_fiz_imbalance",
        "futoi_yur_pos_d1", "futoi_fiz_pos_d1",
        "futoi_yur_pos_d5", "futoi_fiz_pos_d5",
        "futoi_smart_divergence", "futoi_smart_imbalance_diff",
    }
    assert expected.issubset(set(out.columns))
    # YUR_pos[0] = 100; FIZ_pos[0] = -100 → smart_divergence = 200
    assert out.iloc[0]["futoi_yur_pos"] == 100.0
    assert out.iloc[0]["futoi_fiz_pos"] == -100.0
    assert out.iloc[0]["futoi_smart_divergence"] == 200.0


def test_build_futoi_only_yur_when_fiz_missing() -> None:
    """Только YUR строки → собираются YUR-фичи, smart-divergence нет."""
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 100, "pos_long": 150, "pos_short": -50,
         "pos_long_num": 30, "pos_short_num": 10},
    ])
    out = build_futoi_features(df)
    assert "futoi_yur_pos" in out.columns
    assert "futoi_fiz_pos" not in out.columns
    assert "futoi_smart_divergence" not in out.columns


def test_build_futoi_imbalance_formula() -> None:
    """imbalance = (long - |short|) / (long + |short|)."""
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 100, "pos_long": 150, "pos_short": -50,
         "pos_long_num": 30, "pos_short_num": 10},
    ])
    out = build_futoi_features(df)
    # (150 - 50) / (150 + 50) = 0.5
    assert abs(out.iloc[0]["futoi_yur_imbalance"] - 0.5) < 1e-6


def test_build_futoi_takes_last_row_per_day() -> None:
    """Если у дня несколько 5-min snapshot'ов — берём последний (closing)."""
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "10:00:00", "clgroup": "YUR",
         "pos": 50, "pos_long": 100, "pos_short": -50,
         "pos_long_num": 20, "pos_short_num": 10},
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 200, "pos_long": 250, "pos_short": -50,  # close-of-session
         "pos_long_num": 40, "pos_short_num": 10},
    ])
    out = build_futoi_features(df)
    # Должна остаться close-of-session запись
    assert out.iloc[0]["futoi_yur_pos"] == 200.0


def test_build_futoi_smart_divergence_sign() -> None:
    """smart_divergence = yur_pos − fiz_pos. Правильный знак."""
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 50, "pos_long": 100, "pos_short": -50,
         "pos_long_num": 20, "pos_short_num": 10},
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "FIZ",
         "pos": -50, "pos_long": 50, "pos_short": -100,
         "pos_long_num": 50, "pos_short_num": 20},
    ])
    out = build_futoi_features(df)
    # YUR=+50 (long), FIZ=-50 (short) → smart_divergence = +100 (smart bullish vs crowd bearish)
    assert out.iloc[0]["futoi_smart_divergence"] == 100.0


def test_align_to_bar_grid_ffill() -> None:
    """Daily FUTOI → 5-min bar grid через ffill."""
    daily = pd.DataFrame(
        {"futoi_yur_pos": [50.0, 55.0]},
        index=pd.to_datetime(["2024-10-01", "2024-10-02"], utc=True),
    )
    bar_grid = pd.date_range(
        "2024-10-01 09:00", "2024-10-02 12:00", freq="5min", tz="UTC",
    )
    aligned = align_to_bar_grid(daily, bar_grid)
    assert len(aligned) == len(bar_grid)
    assert (aligned[aligned.index < pd.Timestamp("2024-10-02", tz="UTC")]
            ["futoi_yur_pos"] == 50.0).all()
    assert (aligned[aligned.index >= pd.Timestamp("2024-10-02", tz="UTC")]
            ["futoi_yur_pos"] == 55.0).all()


def test_align_to_bar_grid_empty_returns_empty_columns() -> None:
    """Empty FUTOI → DataFrame с target_index, без колонок."""
    bar_grid = pd.date_range("2024-10-01", periods=10, freq="5min", tz="UTC")
    aligned = align_to_bar_grid(pd.DataFrame(), bar_grid)
    assert aligned.empty
    assert len(aligned.columns) == 0
    assert len(aligned.index) == 10


def test_diff_at_first_row_is_zero_not_nan() -> None:
    df = _make_futoi_long_form([
        {"tradedate": "2024-10-01", "tradetime": "23:50:00", "clgroup": "YUR",
         "pos": 100, "pos_long": 200, "pos_short": -100,
         "pos_long_num": 50, "pos_short_num": 25},
    ])
    out = build_futoi_features(df)
    assert out.iloc[0]["futoi_yur_pos_d1"] == 0.0
    assert out.iloc[0]["futoi_yur_pos_d5"] == 0.0
