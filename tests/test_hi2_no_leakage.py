"""Регрессия R-0054: HI2 имел look-ahead leakage из-за ffill от полночи дня D.

HI2 для дня D физически вычисляется в 18:40 UTC (конец сессии). Старая
реализация конвертировала ``tradedate`` ("2024-01-03") в полночь UTC
2024-01-03 00:00, и `reindex(..., method='ffill')` назначал бары 07:00
утра ТОГО ЖЕ ДНЯ значением hhi_volume, которое будет посчитано лишь
в 18:40 — модель в 07:00 знала о будущих сделках 07:00–18:40.

После fix HI2 для дня D становится доступен только с дня D+1 (через
``availability_delay`` параметр).
"""

from __future__ import annotations

import pandas as pd

from graduate_work.features.algopack_features import hi2_features


def _build_hi2(rows: list[dict]) -> pd.DataFrame:
    """Long-form HI2 как у ALGOPACK: tradedate, tradetime, secid, metric, value."""
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(
        df["tradedate"].astype(str) + " " + df["tradetime"].astype(str),
        utc=True,
    )
    df.index = ts
    df.index.name = "begin"
    return df


def test_no_leakage_morning_bar_uses_yesterday_hi2() -> None:
    """Утренний бар дня D НЕ должен видеть HI2-значение, посчитанное в 18:40 ТОГО ЖЕ ДНЯ.

    Это проверка регрессии: старая реализация возвращала value(2024-01-03)
    для бара 2024-01-03 07:00 (UTC) → look-ahead 11 часов.
    """
    hi2 = _build_hi2([
        {"tradedate": "2024-01-02", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 50},
        {"tradedate": "2024-01-03", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 90},
        {"tradedate": "2024-01-04", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 75},
    ])
    target_grid = pd.date_range(
        "2024-01-03 07:00", "2024-01-03 18:45", freq="5min", tz="UTC",
    )
    result = hi2_features(hi2, target_grid)
    morning_value = result.loc[
        pd.Timestamp("2024-01-03 07:00", tz="UTC"), "aps_hi2_hhi_volume",
    ]
    # Бар 07:00 утра 2024-01-03 должен видеть HI2 от 2024-01-02 (=50),
    # НЕ от 2024-01-03 (=90).
    assert morning_value == 50.0, (
        f"LEAKAGE: bar at 07:00 уже знает HI2 текущего дня "
        f"(value={morning_value}, expected 50 из 2024-01-02)"
    )


def test_next_day_open_sees_yesterday_hi2() -> None:
    """К открытию следующего дня HI2 предыдущего дня уже доступен.

    После fix: HI2(D) = (raw_ts).normalize() + 1day → доступен с 00:00 D+1.
    Бар утром 2024-01-04 видит HI2(2024-01-03) — корректное no-leakage поведение.
    """
    hi2 = _build_hi2([
        {"tradedate": "2024-01-02", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 50},
        {"tradedate": "2024-01-03", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 90},
    ])
    target_grid = pd.date_range(
        "2024-01-04 07:00", "2024-01-04 19:00", freq="5min", tz="UTC",
    )
    result = hi2_features(hi2, target_grid)
    morning_jan04 = result.loc[
        pd.Timestamp("2024-01-04 07:00", tz="UTC"), "aps_hi2_hhi_volume",
    ]
    # Бары 2024-01-04 уже могут использовать HI2(2024-01-03)=90.
    # Это НЕ leakage — данные физически доступны на момент открытия рынка.
    assert morning_jan04 == 90.0


def test_multi_metric_pivot_correct_after_fix() -> None:
    """11 метрик на один день должны оба попасть в результат (R-0052+R-0054)."""
    hi2 = _build_hi2([
        {"tradedate": "2024-01-02", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 100},
        {"tradedate": "2024-01-02", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_aggressive_buy", "value": 50},
        {"tradedate": "2024-01-02", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_passive_sell", "value": 200},
    ])
    target = pd.date_range("2024-01-04", periods=10, freq="5min", tz="UTC")
    result = hi2_features(hi2, target)
    assert "aps_hi2_hhi_volume" in result.columns
    assert "aps_hi2_hhi_aggressive_buy" in result.columns
    assert "aps_hi2_hhi_passive_sell" in result.columns
    # Все три уже доступны (delay прошёл).
    assert result.iloc[0]["aps_hi2_hhi_volume"] == 100.0


def test_zero_delay_for_backward_compat() -> None:
    """``availability_delay=0`` восстанавливает старое поведение
    (для тех ноутбуков/скриптов, которые специально хотели мгновенную доступность)."""
    hi2 = _build_hi2([
        {"tradedate": "2024-01-03", "tradetime": "18:40:00",
         "secid": "SBER", "metric": "hhi_volume", "value": 90},
    ])
    target = pd.date_range(
        "2024-01-03 18:40", periods=3, freq="5min", tz="UTC",
    )
    result = hi2_features(hi2, target, availability_delay=pd.Timedelta(0))
    # При нулевом delay бар 18:40 точно совпадает с публикацией → виден сразу.
    assert result.iloc[0]["aps_hi2_hhi_volume"] == 90.0


def test_hi2_empty_input_safe() -> None:
    target = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    result = hi2_features(pd.DataFrame(), target)
    assert len(result) == len(target)
    assert len(result.columns) == 0


def test_hi2_no_metric_column_safe() -> None:
    df = pd.DataFrame({"tradedate": ["2024-01-01"], "value": [50]})
    target = pd.date_range("2024-01-02", periods=5, freq="5min", tz="UTC")
    result = hi2_features(df, target)
    # Нет колонки 'metric' → пустой результат
    assert len(result.columns) == 0
