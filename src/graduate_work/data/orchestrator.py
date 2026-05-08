"""Оркестратор загрузки: котировки тикеров MOEX, индексы, макроиндикаторы.

Тикеры и индексы качаются батчами (`_batches.download_in_batches`):
большой диапазон делится на ``download_batch_months`` куски, каждый чанк
ретраится до ``download_batch_retries`` раз с экспоненциальным backoff,
и после каждого успешного чанка CSV дописывается на диск.

Параллельная загрузка нескольких тикеров одновременно через
``ThreadPoolExecutor`` (количество воркеров - ``download_workers``).
Сетевая работа MOEX ISS - I/O-bound, GIL не мешает.

Сырые данные сохраняются как есть (1-минутки для MOEX). Ресэмпл к
``bar_minutes`` происходит ПОЗЖЕ - в `features/pipeline.py` или в
`serving/live_features.py`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в pyproject.toml
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import DataConfig, Paths
from . import algopack as algopack_mod
from . import calendar_iss, cbr, dividends as dividends_mod, moex_iss, storage, yahoo
from ._batches import download_in_batches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Тикеры и индексы - батчированная загрузка
# ---------------------------------------------------------------------------

def _download_one_ticker(
    ticker: str, cfg: DataConfig, paths: Paths,
    *, show_chunk_progress: bool = True,
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
        show_chunk_progress=show_chunk_progress,
    )


def _download_one_index(
    code: str, cfg: DataConfig, paths: Paths,
    *, show_chunk_progress: bool = True,
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
        show_chunk_progress=show_chunk_progress,
    )


def _safe_future_result(future, item: str) -> pd.DataFrame:
    """Извлечь результат future с дружелюбным логированием ошибок."""
    try:
        return future.result()
    except RuntimeError as exc:
        logger.error("Skipping %s after exhausted retries: %s", item, exc)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected failure for %s", item)
    return pd.DataFrame()


def _run_parallel(
    items: Iterable[str],
    worker: Callable[[str], pd.DataFrame],
    *,
    workers: int,
    desc: str,
    unit: str,
) -> dict[str, pd.DataFrame]:
    """Запустить ``worker(item)`` параллельно для всех ``items``.

    Возвращает {item: df} только для тех, где df не пустой.
    Падения одной задачи не убивают остальные - логгируются и
    продолжаются.
    """
    items = list(items)
    out: dict[str, pd.DataFrame] = {}
    bar = tqdm(total=len(items), desc=desc, unit=unit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(item)
            df = _safe_future_result(future, item)
            bar.update(1)
            if not df.empty:
                out[item] = df
    bar.close()
    return out


def download_tickers(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    """Скачать сырые 1-минутные OHLCV для всех тикеров параллельно."""
    workers = max(1, cfg.download_workers)
    parallel = workers > 1

    def worker(ticker: str) -> pd.DataFrame:
        return _download_one_ticker(
            ticker, cfg, paths,
            show_chunk_progress=not parallel,
        )

    return _run_parallel(
        cfg.tickers, worker,
        workers=workers, desc="MOEX tickers", unit="ticker",
    )


def download_indexes(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    codes = (cfg.base_index, *cfg.extra_indexes)
    workers = max(1, min(cfg.download_workers, len(codes)))
    parallel = workers > 1

    def worker(code: str) -> pd.DataFrame:
        return _download_one_index(
            code, cfg, paths,
            show_chunk_progress=not parallel,
        )

    return _run_parallel(
        codes, worker,
        workers=workers, desc="MOEX indexes", unit="index",
    )


def _download_one_metal_fx(
    secid: str, cfg: DataConfig, paths: Paths,
    *, show_chunk_progress: bool = True,
) -> pd.DataFrame:
    target = paths.data_raw / "metals_fx" / f"{secid}.csv"

    def fetch(s: str, e: str) -> pd.DataFrame:
        return moex_iss.fetch_currency_metal(
            secid, start=s, end=e, interval=cfg.metals_fx_interval,
        )

    return download_in_batches(
        start=cfg.start_date,
        end=cfg.end_date,
        batch_months=cfg.download_batch_months,
        retries=cfg.download_batch_retries,
        backoff_sec=cfg.download_batch_backoff_sec,
        target_path=target,
        fetch=fetch,
        label=f"MOEX metal/FX {secid}",
        show_chunk_progress=show_chunk_progress,
    )


def download_metals_fx(cfg: DataConfig, paths: Paths) -> dict[str, pd.DataFrame]:
    codes = tuple(cfg.metals_fx_codes)
    if not codes:
        return {}
    workers = max(1, min(cfg.download_workers, len(codes)))
    parallel = workers > 1

    def worker(code: str) -> pd.DataFrame:
        return _download_one_metal_fx(
            code, cfg, paths,
            show_chunk_progress=not parallel,
        )

    return _run_parallel(
        codes, worker,
        workers=workers, desc="MOEX metals/FX", unit="instr",
    )


# ---------------------------------------------------------------------------
# ALGOPACK (платный premium-фид)
# ---------------------------------------------------------------------------

_ALGOPACK_FETCHERS: dict[str, str] = {
    # product short name -> AlgopackClient method name
    "tradestats": "tradestats",
    "orderstats": "orderstats",
    "obstats":    "obstats",
    "hi2":        "hi2",
    "futoi":      "futoi",
}


def _algopack_target(paths: Paths, product: str, secid: str) -> Path:
    return paths.data_raw / "algopack" / product / f"{secid}.csv"


def _download_one_algopack(
    client: "algopack_mod.AlgopackClient",
    secid: str,
    product: str,
    cfg: DataConfig,
    paths: Paths,
    *,
    show_chunk_progress: bool,
) -> pd.DataFrame:
    """Скачать один (ticker × product) с idempotent chunk-resume."""
    method_name = _ALGOPACK_FETCHERS[product]
    method = getattr(client, method_name)
    target = _algopack_target(paths, product, secid)

    def fetch(start: str, end: str) -> pd.DataFrame:
        kwargs: dict[str, str] = {"start": start, "end": end}
        # FUTOI не принимает market, остальные принимают.
        if product != "futoi":
            kwargs["market"] = cfg.algopack_market
        return method(secid, **kwargs)

    return download_in_batches(
        start=cfg.start_date,
        end=cfg.end_date,
        batch_months=cfg.download_batch_months,
        retries=cfg.download_batch_retries,
        backoff_sec=cfg.download_batch_backoff_sec,
        target_path=target,
        fetch=fetch,
        label=f"ALGOPACK {product} {secid}",
        show_chunk_progress=show_chunk_progress,
    )


def download_algopack(
    cfg: DataConfig,
    paths: Paths,
    *,
    token: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Скачать все (ticker × algopack_product) комбинации.

    Args:
        token: Bearer-токен ALGOPACK. Если None, берётся из переменной
            окружения ``ALGOPACK_TOKEN``. При отсутствии токена и
            непустом ``cfg.algopack_products`` поднимается ValueError.

    Returns:
        Словарь ``{(ticker, product): DataFrame}`` — только успешные.
    """
    products = tuple(cfg.algopack_products)
    if not products:
        return {}
    unknown = set(products) - set(_ALGOPACK_FETCHERS)
    if unknown:
        msg = (
            f"Неизвестные algopack-продукты: {sorted(unknown)}. "
            f"Допустимые: {sorted(_ALGOPACK_FETCHERS)}."
        )
        raise ValueError(msg)

    client = algopack_mod.AlgopackClient(
        token=token,
        request_pause=cfg.algopack_request_pause_sec,
    )

    jobs: list[tuple[str, str]] = [
        (ticker, product) for ticker in cfg.tickers for product in products
    ]
    workers = max(1, min(cfg.download_workers, len(jobs)))
    parallel = workers > 1

    def worker(job_key: str) -> pd.DataFrame:
        ticker, product = job_key.split("\x00", 1)
        return _download_one_algopack(
            client, ticker, product, cfg, paths,
            show_chunk_progress=not parallel,
        )

    # _run_parallel ключует по строкам — кодируем (ticker, product) в одну.
    encoded = [f"{t}\x00{p}" for t, p in jobs]
    raw = _run_parallel(
        encoded, worker,
        workers=workers, desc="ALGOPACK", unit="job",
    )
    return {tuple(k.split("\x00", 1)): df for k, df in raw.items()}


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


# ---------------------------------------------------------------------------
# Calendars (open ISS, без auth) — sequential, объёмы небольшие
# ---------------------------------------------------------------------------

def _save_calendar(df: pd.DataFrame, paths: Paths, name: str) -> None:
    if df.empty:
        return
    target = paths.data_raw / "calendars" / f"{name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")


# Календари, которые принимают from/till — у suspended_intraday/static-
# эндпоинтов диапазон не нужен.
_CALENDARS_WITH_DATE_RANGE = frozenset({
    "trading_days_stock", "trading_days_futures", "trading_days_currency",
    "session_stock", "session_futures", "session_currency",
    "settlecodes", "futures_expirations",
    "security_changes", "boards_history",
    "currency_settlement_shifts",
})


def _calendar_kwargs(
    key: str, cfg: DataConfig, token: str | None,
) -> dict:
    """Собрать kwargs для конкретного calendar fetcher'а."""
    kwargs: dict = {"token": token}
    if key in _CALENDARS_WITH_DATE_RANGE:
        kwargs["start"] = cfg.start_date
        kwargs["end"] = cfg.end_date
    return kwargs


def _fetch_one_calendar(key: str, kwargs: dict) -> pd.DataFrame:
    """Один календарь с try/except — exception'ы лог и пустой DataFrame."""
    fetcher = calendar_iss.CALENDAR_PRODUCTS[key]
    try:
        return fetcher(**kwargs)
    except Exception:  # noqa: BLE001 - пишем в лог и продолжаем
        logger.exception("Calendar fetch failed for %s", key)
        return pd.DataFrame()


def download_calendars(
    cfg: DataConfig,
    paths: Paths,
    *,
    products: tuple[str, ...] | None = None,
    token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Скачать календари MOEX.

    Args:
        products: подмножество ключей из ``calendar_iss.CALENDAR_PRODUCTS``.
            ``None`` = все.
        token: ALGOPACK Bearer-токен. Если ``None``, берётся из
            ``ALGOPACK_TOKEN`` env var. С токеном запросы идут на
            ``apim.moex.com`` (приоритетный rate-limit), без — на
            открытый ``iss.moex.com``.

    Returns:
        Словарь ``{product_name: DataFrame}``.
    """
    resolved_token = token if token is not None else os.environ.get("ALGOPACK_TOKEN")
    keys = (
        list(products) if products is not None
        else list(calendar_iss.CALENDAR_PRODUCTS.keys())
    )
    out: dict[str, pd.DataFrame] = {}
    bar = tqdm(keys, desc="Calendars", unit="cal")
    for key in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(key)
        df = _fetch_one_calendar(key, _calendar_kwargs(key, cfg, resolved_token))
        if not df.empty:
            _save_calendar(df, paths, key)
            out[key] = df
    bar.close()
    return out


# ---------------------------------------------------------------------------
# Dividends (open ISS, без auth) — параллельно по тикерам
# ---------------------------------------------------------------------------

def _save_dividends(df: pd.DataFrame, paths: Paths, secid: str) -> None:
    target = paths.data_raw / "dividends" / f"{secid}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")


def download_dividends(
    cfg: DataConfig,
    paths: Paths,
    *,
    token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Скачать историю дивидендов для всех тикеров параллельно.

    Args:
        token: ALGOPACK Bearer-токен; ``None`` → читается из
            ``ALGOPACK_TOKEN`` env var. С токеном — apim.moex.com.
    """
    resolved_token = token if token is not None else os.environ.get("ALGOPACK_TOKEN")
    workers = max(1, min(cfg.download_workers, len(cfg.tickers)))

    def worker(ticker: str) -> pd.DataFrame:
        df = dividends_mod.fetch_dividends(ticker, token=resolved_token)
        if not df.empty:
            _save_dividends(df, paths, ticker)
        return df

    return _run_parallel(
        cfg.tickers, worker,
        workers=workers, desc="Dividends", unit="ticker",
    )


def download_all(cfg: DataConfig, paths: Paths) -> None:
    """Полная пакетная загрузка.

    Algopack качается только при заданном ``cfg.algopack_products`` и
    переменной ``ALGOPACK_TOKEN`` — иначе тихо пропускаем (бесплатные
    эндпоинты не зависят от подписки).
    """
    paths.ensure()
    download_tickers(cfg, paths)
    download_indexes(cfg, paths)
    download_metals_fx(cfg, paths)
    download_macro(cfg, paths)
    download_calendars(cfg, paths)
    download_dividends(cfg, paths)
    if cfg.algopack_products and os.environ.get("ALGOPACK_TOKEN"):
        download_algopack(cfg, paths)
