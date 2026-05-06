"""Оркестратор загрузки: собирает котировки тикеров MOEX, индексы,
макроиндикаторы ЦБ и нефть Brent в единое хранилище.

Каждый источник пишется в свой CSV-файл (универсальный сырой формат),
после чего модуль предобработки соберёт из них Parquet с фичами.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import DataConfig, Paths
from . import cbr, moex_iss, storage, yahoo

logger = logging.getLogger(__name__)


def _ticker_path(paths: Paths, ticker: str) -> str:
    return str(paths.data_raw / "moex" / f"{ticker}.csv")


def download_tickers(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    """Скачать дневные OHLCV для всех тикеров и сохранить в CSV."""
    out: dict[str, pd.DataFrame] = {}
    for ticker in cfg.tickers:
        logger.info("MOEX ticker %s ...", ticker)
        df = moex_iss.fetch_ticker(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.moex_interval,
        )
        if df.empty:
            logger.warning("Empty data for %s, skipping", ticker)
            continue
        storage.save_raw_csv(df, paths.data_raw / "moex" / f"{ticker}.csv")
        out[ticker] = df
    return out


def download_indexes(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for code in (cfg.base_index, *cfg.extra_indexes):
        logger.info("MOEX index %s ...", code)
        df = moex_iss.fetch_index(
            code,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.moex_interval,
        )
        if df.empty:
            logger.warning("Empty index %s", code)
            continue
        storage.save_raw_csv(df, paths.data_raw / "indexes" / f"{code}.csv")
        out[code] = df
    return out


def download_macro(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for code in cfg.cbr_currencies:
        logger.info("CBR currency %s ...", code)
        df = cbr.fetch_currency(code, cfg.start_date, cfg.end_date)
        if not df.empty:
            storage.save_raw_csv(df, paths.data_raw / "macro" / f"cbr_{code.lower()}.csv")
            out[f"cbr_{code.lower()}"] = df

    logger.info("CBR key rate ...")
    keyrate = cbr.fetch_keyrate(cfg.start_date, cfg.end_date)
    if not keyrate.empty:
        storage.save_raw_csv(keyrate, paths.data_raw / "macro" / "cbr_keyrate.csv")
        out["cbr_keyrate"] = keyrate

    logger.info("Yahoo Brent (%s) ...", cfg.brent_symbol)
    brent = yahoo.fetch_yahoo(cfg.brent_symbol, cfg.start_date, cfg.end_date)
    if not brent.empty:
        # Оставляем только close, переименуем для уникальности.
        brent_close = brent[["close"]].rename(columns={"close": "brent_close"})
        storage.save_raw_csv(brent_close, paths.data_raw / "macro" / "brent.csv")
        out["brent"] = brent_close
    return out


def download_all(cfg: DataConfig, paths: Paths) -> None:
    """Полная пакетная загрузка."""
    paths.ensure()
    download_tickers(cfg, paths)
    download_indexes(cfg, paths)
    download_macro(cfg, paths)
