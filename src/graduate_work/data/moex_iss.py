"""HTTP-клиент Московской биржи (MOEX ISS).

Бесплатный, без авторизации. Скачивает дневные OHLCV-свечи для акций
основного режима TQBR, а также значения индексов (IMOEX, RGBI, RTSI).
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ISS = "https://iss.moex.com/iss"
MAX_RETRIES = 5
BASE_DELAY = 1.5
PAGE_SIZE = 500


def _fetch_page(path: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(f"{ISS}/{path}.json", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning("MOEX retry %d/%d in %.1fs: %s", attempt + 1, MAX_RETRIES, delay, exc)
            time.sleep(delay)
    msg = f"MOEX ISS failed after {MAX_RETRIES} retries"
    raise RuntimeError(msg) from last_exc


def _paginated_candles(
    path: str,
    *,
    start: str,
    end: str,
    interval: int,
) -> pd.DataFrame:
    rows: list[list] = []
    columns: list[str] | None = None
    cursor = 0
    while True:
        payload = _fetch_page(
            path,
            {
                "from": start,
                "till": end,
                "interval": interval,
                "start": cursor,
            },
        )
        candles = payload.get("candles", {})
        if columns is None:
            columns = candles.get("columns", [])
        page = candles.get("data", [])
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        cursor += len(page)
        time.sleep(0.1)
    if not rows or columns is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=columns)
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.set_index("begin").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype("float64")


def fetch_ticker(
    ticker: str,
    *,
    start: str,
    end: str,
    interval: int = 24,
    board: str = "TQBR",
) -> pd.DataFrame:
    """Скачать OHLCV акции с основного режима MOEX."""
    path = f"engines/stock/markets/shares/boards/{board}/securities/{ticker}/candles"
    df = _paginated_candles(path, start=start, end=end, interval=interval)
    df["ticker"] = ticker
    return df


def fetch_index(
    code: str,
    *,
    start: str,
    end: str,
    interval: int = 24,
) -> pd.DataFrame:
    """Скачать значения индекса (IMOEX, RGBI, RTSI)."""
    path = f"engines/stock/markets/index/securities/{code}/candles"
    df = _paginated_candles(path, start=start, end=end, interval=interval)
    df["ticker"] = code
    return df
