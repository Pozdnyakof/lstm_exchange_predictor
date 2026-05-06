"""Файловое хранилище: CSV для сырых данных, Parquet для предобработанных."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_raw_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_raw_csv(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if parse_dates:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)


def load_processed(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")
