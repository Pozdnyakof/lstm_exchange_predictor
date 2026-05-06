"""Тесты батчированной загрузки."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graduate_work.data._batches import (
    download_in_batches,
    iter_chunks,
)


def test_iter_chunks_splits_range_into_quarters() -> None:
    chunks = iter_chunks("2020-01-01", "2021-01-01", batch_months=3)
    assert len(chunks) == 4
    assert chunks[0] == ("2020-01-01", "2020-04-01")
    assert chunks[-1] == ("2020-10-01", "2021-01-01")


def test_iter_chunks_handles_partial_tail() -> None:
    chunks = iter_chunks("2020-01-01", "2020-05-15", batch_months=3)
    assert chunks[-1][1] == "2020-05-15"


def test_iter_chunks_rejects_zero_batch() -> None:
    with pytest.raises(ValueError):
        iter_chunks("2020-01-01", "2021-01-01", batch_months=0)


def _fake_chunk_data(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "open": rng.standard_normal(len(idx)),
            "high": rng.standard_normal(len(idx)),
            "low": rng.standard_normal(len(idx)),
            "close": rng.standard_normal(len(idx)),
            "volume": rng.integers(100, 1000, len(idx)).astype(float),
        },
        index=idx,
    )


def test_download_in_batches_writes_csv_after_each_chunk(tmp_path: Path) -> None:
    target = tmp_path / "X.csv"
    calls: list[tuple[str, str]] = []

    def fetch(s: str, e: str) -> pd.DataFrame:
        calls.append((s, e))
        return _fake_chunk_data(s, e)

    df = download_in_batches(
        start="2020-01-01",
        end="2020-07-01",
        batch_months=3,
        retries=2,
        backoff_sec=0.0,
        target_path=target,
        fetch=fetch,
        label="X",
    )
    assert target.exists()
    assert len(calls) == 2
    assert not df.empty
    # Колонки сохранились.
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_download_in_batches_retries_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "X.csv"
    attempts = {"n": 0}

    def fetch(s: str, e: str) -> pd.DataFrame:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("MOEX rate-limit")
        return _fake_chunk_data(s, e)

    df = download_in_batches(
        start="2020-01-01",
        end="2020-04-01",
        batch_months=3,
        retries=4,
        backoff_sec=0.0,
        target_path=target,
        fetch=fetch,
        label="X",
    )
    assert attempts["n"] == 3
    assert not df.empty


def test_download_in_batches_raises_after_all_retries(tmp_path: Path) -> None:
    target = tmp_path / "X.csv"

    def fetch(s: str, e: str) -> pd.DataFrame:
        raise ConnectionError("MOEX is angry")

    with pytest.raises(RuntimeError, match="all 3 retries"):
        download_in_batches(
            start="2020-01-01",
            end="2020-04-01",
            batch_months=3,
            retries=3,
            backoff_sec=0.0,
            target_path=target,
            fetch=fetch,
            label="X",
        )


def test_download_in_batches_resumes_from_existing_csv(tmp_path: Path) -> None:
    target = tmp_path / "X.csv"

    # Сначала качаем первый квартал.
    download_in_batches(
        start="2020-01-01", end="2020-04-01",
        batch_months=3, retries=2, backoff_sec=0.0,
        target_path=target,
        fetch=_fake_chunk_data,
        label="X",
    )

    # Теперь зовём с расширенным диапазоном; первый чанк должен быть пропущен.
    calls: list[tuple[str, str]] = []

    def fetch(s: str, e: str) -> pd.DataFrame:
        calls.append((s, e))
        return _fake_chunk_data(s, e)

    download_in_batches(
        start="2020-01-01", end="2020-07-01",
        batch_months=3, retries=2, backoff_sec=0.0,
        target_path=target,
        fetch=fetch,
        label="X",
    )
    # Первый квартал уже был - должен быть пропущен.
    assert calls == [("2020-04-01", "2020-07-01")]
