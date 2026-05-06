"""Оркестратор загрузки: котировки тикеров MOEX, индексы, макроиндикаторы.

Тикеры и индексы качаются батчами (`_batches.download_in_batches`):
большой диапазон делится на ``download_batch_months`` куски, каждый чанк
ретраится до ``download_batch_retries`` раз с экспоненциальным backoff,
и после каждого успешного чанка CSV дописывается на диск - чтобы не
терять прогресс при rate-limit'е MOEX.

Сырые данные сохраняются как есть (1-минутки для MOEX). Ресэмпл к
``bar_minutes`` происходит ПОЗЖЕ - в `features/pipeline.py` или в
`serving/live_features.py`.
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в pyproject.toml
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import DataConfig, Paths
from . import cbr, moex_iss, storage, yahoo
from ._batches import download_in_batches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Тикеры и индексы - батчированная загрузка
# ---------------------------------------------------------------------------

def _download_one_ticker(
    ticker: str, cfg: DataConfig, paths: Paths,
) -> pd.DataFrame:
    target = paths.data_raw / "moex" / f"{ticker}.csv"

    def fetch(s: str, e: str) -> pd.DataFrame:
        return moex_iss.fetch_ticker(
            ticker, start=s, end=e, interval=cfg.moex_interval,
        )

    return download_in_batches(
        start=cfg.start_date,
        end=cfg.end_date,
        batch_months=cfg.download_batch_months,
        retries=cfg.download_batch_retries,
        backoff_sec=cfg.download_batch_backoff_sec,
        target_path=target,
        fetch=fetch,
        label=f"MOEX {ticker}",
    )


def _download_one_index(
    code: str, cfg: DataConfig, paths: Paths,
) -> pd.DataFrame:
    target = paths.data_raw / "indexes" / f"{code}.csv"

    def fetch(s: str, e: str) -> pd.DataFrame:
        return moex_iss.fetch_index(
            code, start=s, end=e, interval=cfg.moex_interval,
        )

    return download_in_batches(
        start=cfg.start_date,
        end=cfg.end_date,
        batch_months=cfg.download_batch_months,
        retries=cfg.download_batch_retries,
        backoff_sec=cfg.download_batch_backoff_sec,
        target_path=target,
        fetch=fetch,
        label=f"MOEX index {code}",
    )


def download_tickers(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    """Скачать сырые 1-минутные OHLCV для всех тикеров (батчами с retry)."""
    out: dict[str, pd.DataFrame] = {}
    bar = tqdm(cfg.tickers, desc="MOEX tickers", unit="ticker")
    for ticker in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(ticker)
        try:
            df = _download_one_ticker(ticker, cfg, paths)
        except RuntimeError as exc:
            logger.error("Skipping %s after exhausted retries: %s", ticker, exc)
            continue
        if not df.empty:
            out[ticker] = df
    return out


def download_indexes(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    codes = (cfg.base_index, *cfg.extra_indexes)
    bar = tqdm(codes, desc="MOEX indexes", unit="index")
    for code in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(code)
        try:
            df = _download_one_index(code, cfg, paths)
        except RuntimeError as exc:
            logger.error("Skipping index %s after exhausted retries: %s", code, exc)
            continue
        if not df.empty:
            out[code] = df
    return out


# ---------------------------------------------------------------------------
# Macro - объёмы небольшие, без батчирования
# ---------------------------------------------------------------------------

def _fetch_currency_step(
    cfg: DataConfig, paths: Paths, code: str,
) -> tuple[str, pd.DataFrame] | None:
    df = cbr.fetch_currency(code, cfg.start_date, cfg.end_date)
    if df.empty:
        return None
    storage.save_raw_csv(df, paths.data_raw / "macro" / f"cbr_{code.lower()}.csv")
    return f"cbr_{code.lower()}", df


def _fetch_keyrate_step(
    cfg: DataConfig, paths: Paths,
) -> tuple[str, pd.DataFrame] | None:
    df = cbr.fetch_keyrate(cfg.start_date, cfg.end_date)
    if df.empty:
        return None
    storage.save_raw_csv(df, paths.data_raw / "macro" / "cbr_keyrate.csv")
    return "cbr_keyrate", df


def _fetch_brent_step(
    cfg: DataConfig, paths: Paths,
) -> tuple[str, pd.DataFrame] | None:
    df = yahoo.fetch_yahoo(cfg.brent_symbol, cfg.start_date, cfg.end_date)
    if df.empty:
        return None
    brent_close = df[["close"]].rename(columns={"close": "brent_close"})
    storage.save_raw_csv(brent_close, paths.data_raw / "macro" / "brent.csv")
    return "brent", brent_close


def download_macro(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    steps: list[tuple[str, callable]] = [
        (f"CBR {c}", lambda c=c: _fetch_currency_step(cfg, paths, c))
        for c in cfg.cbr_currencies
    ]
    steps.append(("CBR key rate", lambda: _fetch_keyrate_step(cfg, paths)))
    steps.append((f"Yahoo {cfg.brent_symbol}", lambda: _fetch_brent_step(cfg, paths)))

    bar = tqdm(steps, desc="Macro series", unit="series")
    for label, fetch in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(label)
        result = fetch()
        if result is not None:
            key, df = result
            out[key] = df
    return out


def download_all(cfg: DataConfig, paths: Paths) -> None:
    """Полная пакетная загрузка."""
    paths.ensure()
    download_tickers(cfg, paths)
    download_indexes(cfg, paths)
    download_macro(cfg, paths)
