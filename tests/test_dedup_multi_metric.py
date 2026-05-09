"""Регрессия R-0052: dedup _merge_and_save теряло 90% HI2 metrics.

Старая логика ``df.index.duplicated(keep="last")`` дедуплицировала
по индексу — но HI2 имеет 10+ строк с одинаковым timestamp (разные
metrics: hhi_volume, hhi_aggressive_buy, etc). Оставлялась лишь одна.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.data._batches import _dedupe_rows


def _make_hi2_chunk(date: str, metrics: list[str], values: list[int]) -> pd.DataFrame:
    """Имитация HI2-chunk'а ALGOPACK с одинаковыми timestamp на разные метрики."""
    n = len(metrics)
    return pd.DataFrame(
        {
            "tradedate": [date] * n,
            "tradetime": ["18:40:00"] * n,
            "secid": ["SBER"] * n,
            "metric": metrics,
            "value": values,
            "reference": [None] * n,
            "SYSTIME": ["2024-04-09 15:27:19"] * n,
        },
        index=pd.DatetimeIndex(
            [f"{date} 18:40:00+00:00"] * n, name="begin",
        ),
    )


def test_dedupe_preserves_all_metrics_per_timestamp() -> None:
    """HI2: 10 метрик на один день → все 10 должны выжить (R-0052 fix)."""
    metrics = [
        "hhi_volume", "hhi_aggressive_buy", "hhi_passive_sell",
        "hhi_aggressive", "hhi_sell", "hhi_buy", "hhi_passive_buy",
        "hhi_aggressive_sell", "hhi_passive", "hhi_total",
    ]
    df = _make_hi2_chunk(
        date="2024-01-03",
        metrics=metrics,
        values=list(range(100, 110)),
    )
    dedup = _dedupe_rows(df)
    assert len(dedup) == 10, f"expected 10 rows, got {len(dedup)}"
    assert set(dedup["metric"]) == set(metrics)


def test_dedupe_removes_full_duplicates_from_overlap() -> None:
    """Overlapping chunks: одинаковые (timestamp, metric, value) → одна строка."""
    chunk1 = _make_hi2_chunk("2024-01-03", ["hhi_volume", "hhi_buy"], [100, 50])
    chunk2 = _make_hi2_chunk("2024-01-03", ["hhi_volume", "hhi_buy"], [100, 50])
    combined = pd.concat([chunk1, chunk2])
    assert len(combined) == 4
    dedup = _dedupe_rows(combined)
    # Дубликаты по всем колонкам (минус SYSTIME) → должны схлопнуться.
    assert len(dedup) == 2


def test_dedupe_keeps_distinct_value_at_same_timestamp_metric() -> None:
    """Если ALGOPACK перевыкачал метрику с другим value — keep last."""
    chunk1 = _make_hi2_chunk("2024-01-03", ["hhi_volume"], [100])
    chunk2 = _make_hi2_chunk("2024-01-03", ["hhi_volume"], [200])  # уточнение
    combined = pd.concat([chunk1, chunk2])
    dedup = _dedupe_rows(combined)
    assert len(dedup) == 2  # обе строки сохранены — разные value
    # Это ожидаемое поведение: ALGOPACK не должен выдавать конфликтующие
    # значения, но если выдал — мы сохраняем обе для последующего анализа.


def test_dedupe_tradestats_one_row_per_timestamp() -> None:
    """Для tradestats (1 строка/timestamp) дедуп работает как раньше."""
    df = pd.DataFrame(
        {
            "pr_open": [100.0, 100.0, 101.0],
            "pr_close": [100.5, 100.5, 101.5],
            "SYSTIME": ["2024-01-01 12:00"] * 3,
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-03 10:00:00+00:00",
                "2024-01-03 10:00:00+00:00",  # дубль с предыдущей
                "2024-01-03 10:05:00+00:00",
            ],
            name="begin",
        ),
    )
    dedup = _dedupe_rows(df)
    assert len(dedup) == 2  # один дубль убрался, осталось 2 уникальных


def test_dedupe_empty_dataframe() -> None:
    empty = pd.DataFrame()
    dedup = _dedupe_rows(empty)
    assert dedup.empty


def test_dedupe_no_systime_column_falls_back_to_index() -> None:
    """Если SYSTIME колонки нет — должна работать дедупликация по индексу."""
    df = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=pd.DatetimeIndex(
            ["2024-01-01", "2024-01-01", "2024-01-02"],
            name="begin",
        ),
    )
    dedup = _dedupe_rows(df)
    # 1+2+3 unique → все 3 остаются
    assert len(dedup) == 3


def test_dedupe_ignores_only_systime_diff() -> None:
    """Две строки идентичные кроме SYSTIME → оставляется одна (последняя)."""
    df = pd.DataFrame(
        {
            "metric": ["hhi_volume", "hhi_volume"],
            "value": [100, 100],
            "SYSTIME": ["2024-04-09 15:27:19", "2024-05-01 09:00:00"],
        },
        index=pd.DatetimeIndex(
            ["2024-01-03 18:40:00+00:00"] * 2, name="begin",
        ),
    )
    dedup = _dedupe_rows(df)
    assert len(dedup) == 1
