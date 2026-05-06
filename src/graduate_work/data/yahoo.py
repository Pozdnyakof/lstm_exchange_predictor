"""Клиент Yahoo Finance (через yfinance) - для котировок Brent."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Скачать дневные котировки Yahoo Finance.

    Возвращает DataFrame с колонками open/high/low/close/volume и
    DatetimeIndex в UTC. При ошибке возвращает пустой фрейм.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        msg = "yfinance not installed; run `poetry install`"
        raise RuntimeError(msg) from exc

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        logger.warning("Empty Yahoo response for %s", symbol)
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].astype("float64")
