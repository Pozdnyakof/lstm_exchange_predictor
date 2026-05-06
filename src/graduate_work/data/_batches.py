"""Утилиты батчированной загрузки с ретраем и сохранением прогресса.

MOEX ISS склонна резать соединения / rate-limit-ить при длинных
выгрузках. Поэтому большой диапазон делим на батчи по ``batch_months``
месяцев, на каждый батч даём ``retries`` попыток с экспоненциальным
backoff, и после каждого успешного батча дописываем CSV на диск.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from . import storage

logger = logging.getLogger(__name__)


def iter_chunks(start: str, end: str, batch_months: int) -> list[tuple[str, str]]:
    """Разбить [start, end] на чанки по ``batch_months`` месяцев.

    Каждый чанк - полуоткрытый интервал [chunk_start, chunk_end), границы
    выровнены по началу месяца. Последний чанк гарантированно
    заканчивается ``end``.
    """
    if batch_months <= 0:
        msg = "batch_months must be > 0"
        raise ValueError(msg)
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if s >= e:
        return []

    chunks: list[tuple[str, str]] = []
    cur = s
    while cur < e:
        nxt = (cur + pd.DateOffset(months=batch_months)).normalize()
        if nxt > e:
            nxt = e
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


def _existing_max_timestamp(path: Path) -> pd.Timestamp | None:
    """Прочитать существующий CSV (если есть), вернуть max(index)."""
    if not path.exists():
        return None
    try:
        df = storage.load_raw_csv(path)
        if df.empty:
            return None
        return pd.Timestamp(df.index.max()).tz_convert("UTC") if df.index.tz else pd.Timestamp(df.index.max(), tz="UTC")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read existing CSV %s: %s", path, exc)
        return None


def _retry_call(
    fetch: Callable[[str, str], pd.DataFrame],
    *,
    chunk_start: str,
    chunk_end: str,
    retries: int,
    backoff_sec: float,
    label: str,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fetch(chunk_start, chunk_end)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = backoff_sec * (2**attempt)
            logger.warning(
                "%s [%s..%s] attempt %d/%d failed: %s; sleeping %.1fs",
                label, chunk_start, chunk_end, attempt + 1, retries, exc, delay,
            )
            time.sleep(delay)
    msg = f"{label}: all {retries} retries failed for [{chunk_start}..{chunk_end}]"
    raise RuntimeError(msg) from last_exc


def _merge_and_save(
    accumulated: list[pd.DataFrame], target_path: Path,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    merged = pd.concat(accumulated, axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    storage.save_raw_csv(merged, target_path)
    last = pd.Timestamp(merged.index.max())
    if last.tz is None:
        last = last.tz_localize("UTC")
    return merged, last


def _process_chunk(
    chunk: tuple[str, str],
    existing_max: pd.Timestamp | None,
    *,
    fetch: Callable[[str, str], pd.DataFrame],
    retries: int,
    backoff_sec: float,
    label: str,
) -> pd.DataFrame | None:
    """Скачать один чанк, если он ещё не покрыт. Возвращает df или None."""
    chunk_start, chunk_end = chunk
    chunk_start_ts = pd.Timestamp(chunk_start, tz="UTC")
    # CSV сохраняется атомарно после каждого успешного чанка, поэтому если
    # existing_max уже зашёл за начало этого чанка - значит чанк уже скачан.
    if existing_max is not None and existing_max >= chunk_start_ts:
        logger.debug("%s: chunk [%s..%s] already covered, skip", label, chunk_start, chunk_end)
        return None
    df = _retry_call(
        fetch,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        retries=retries,
        backoff_sec=backoff_sec,
        label=label,
    )
    if df.empty:
        logger.warning("%s: empty response for [%s..%s]", label, chunk_start, chunk_end)
        return None
    return df


def download_in_batches(
    *,
    start: str,
    end: str,
    batch_months: int,
    retries: int,
    backoff_sec: float,
    target_path: Path,
    fetch: Callable[[str, str], pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    """Скачать диапазон чанками, дописывая CSV после каждого успешного батча.

    Если есть частичный файл - дозабираются только новые батчи (по
    максимальной существующей метке времени).

    ``fetch`` получает (chunk_start, chunk_end) и возвращает DataFrame
    с DatetimeIndex (UTC) и OHLCV-колонками. Дедупликация делается
    автоматически через index.
    """
    chunks = iter_chunks(start, end, batch_months)
    if not chunks:
        return pd.DataFrame()

    existing_max = _existing_max_timestamp(target_path)
    accumulated: list[pd.DataFrame] = []
    if existing_max is not None:
        accumulated.append(storage.load_raw_csv(target_path))

    bar = tqdm(chunks, desc=label, unit="chunk", leave=False)
    for chunk in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(f"{chunk[0]}..{chunk[1]}")
        df = _process_chunk(
            chunk, existing_max,
            fetch=fetch, retries=retries,
            backoff_sec=backoff_sec, label=label,
        )
        if df is None:
            continue
        accumulated.append(df)
        merged, existing_max = _merge_and_save(accumulated, target_path)
        accumulated = [merged]

    return accumulated[-1] if accumulated else pd.DataFrame()
