"""Оркестратор загрузки: собирает котировки тикеров MOEX, индексы,
макроиндикаторы ЦБ и нефть Brent в единое хранилище.

Каждый источник пишется в свой CSV-файл (универсальный сырой формат),
после чего модуль предобработки соберёт из них Parquet с фичами.

tqdm.auto автоматически выбирает обычный/notebook прогресс-бар.
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в pyproject.toml, fallback на noop
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import DataConfig, Paths
from . import cbr, moex_iss, storage, yahoo
from .resample import resample_ohlcv

logger = logging.getLogger(__name__)


def _ticker_path(paths: Paths, ticker: str) -> str:
    return str(paths.data_raw / "moex" / f"{ticker}.csv")


def download_tickers(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    """Скачать 1-минутные OHLCV всех тикеров, ресэмплить до bar_minutes,
    сохранить в CSV (с фильтром торговой сессии MOEX)."""
    out: dict[str, pd.DataFrame] = {}
    bar = tqdm(cfg.tickers, desc="MOEX tickers", unit="ticker")
    for ticker in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(ticker)
        raw = moex_iss.fetch_ticker(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.moex_interval,
        )
        if raw.empty:
            logger.warning("Empty data for %s, skipping", ticker)
            continue
        df = resample_ohlcv(raw, cfg)
        if df.empty:
            logger.warning("After resample %s is empty, skipping", ticker)
            continue
        storage.save_raw_csv(df, paths.data_raw / "moex" / f"{ticker}.csv")
        out[ticker] = df
    return out


def download_indexes(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    codes = (cfg.base_index, *cfg.extra_indexes)
    bar = tqdm(codes, desc="MOEX indexes", unit="index")
    for code in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(code)
        raw = moex_iss.fetch_index(
            code,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.moex_interval,
        )
        if raw.empty:
            logger.warning("Empty index %s", code)
            continue
        df = resample_ohlcv(raw, cfg)
        if df.empty:
            continue
        storage.save_raw_csv(df, paths.data_raw / "indexes" / f"{code}.csv")
        out[code] = df
    return out


def _fetch_currency_step(cfg: DataConfig, paths: Paths, code: str) -> tuple[str, pd.DataFrame] | None:
    df = cbr.fetch_currency(code, cfg.start_date, cfg.end_date)
    if df.empty:
        return None
    storage.save_raw_csv(df, paths.data_raw / "macro" / f"cbr_{code.lower()}.csv")
    return f"cbr_{code.lower()}", df


def _fetch_keyrate_step(cfg: DataConfig, paths: Paths) -> tuple[str, pd.DataFrame] | None:
    df = cbr.fetch_keyrate(cfg.start_date, cfg.end_date)
    if df.empty:
        return None
    storage.save_raw_csv(df, paths.data_raw / "macro" / "cbr_keyrate.csv")
    return "cbr_keyrate", df


def _fetch_brent_step(cfg: DataConfig, paths: Paths) -> tuple[str, pd.DataFrame] | None:
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
