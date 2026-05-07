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


def _read_tail(path: Path, tail_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Cannot tail %s: %s", path, exc)
        return None


def _parse_timestamp_cell(cell: str) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(cell.strip().strip('"'))
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    return ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC")


def _csv_tail_timestamp(path: Path, tail_bytes: int = 16_384) -> pd.Timestamp | None:
    """Прочитать последнюю валидную метку времени из конца CSV.

    Не загружает весь файл - читает только хвост в ``tail_bytes``.
    Для огромных 1-минутных CSV (сотни МБ) это даёт практически
    мгновенный resume-чек вместо полного чтения.
    """
    chunk = _read_tail(path, tail_bytes)
    if chunk is None:
        return None
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    for line in reversed(lines):
        first = line.split(",", 1)[0]
        ts = _parse_timestamp_cell(first)
        if ts is not None:
            return ts
    return None


def _existing_max_timestamp(path: Path) -> pd.Timestamp | None:
    """Дешёвая проверка max(index) существующего CSV (читает только хвост)."""
    if not path.exists():
        return None
    return _csv_tail_timestamp(path)


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
    # Считаем чанк скачанным, только если в файле уже есть данные на день
    # перед chunk_end (нужен запас под выходные / последний бар интрадей).
    # Это спасает от случая, когда предыдущий запуск был прерван в
    # самом начале чанка и сохранил только пару дней - chunk_start
    # уже "пройден", но реально не скачано 99% чанка.
    coverage_threshold = pd.Timestamp(chunk_end, tz="UTC") - pd.Timedelta(days=1)
    if existing_max is not None and existing_max >= coverage_threshold:
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


def _annotate(bar: object | None, chunk: tuple[str, str]) -> None:
    if bar is not None and hasattr(bar, "set_postfix_str"):
        bar.set_postfix_str(f"{chunk[0]}..{chunk[1]}")


def _run_chunks(
    chunks: list[tuple[str, str]],
    state: "_ResumeState",
    *,
    fetch: Callable[[str, str], pd.DataFrame],
    retries: int,
    backoff_sec: float,
    label: str,
    show_chunk_progress: bool,
) -> None:
    bar = (
        tqdm(chunks, desc=label, unit="chunk", leave=False)
        if show_chunk_progress else None
    )
    iterable = bar if bar is not None else chunks
    for chunk in iterable:
        _annotate(bar, chunk)
        df = _process_chunk(
            chunk, state.existing_max,
            fetch=fetch, retries=retries,
            backoff_sec=backoff_sec, label=label,
        )
        if df is not None:
            state.append_and_save(df)


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
    show_chunk_progress: bool = True,
) -> pd.DataFrame:
    """Скачать диапазон чанками, дописывая CSV после каждого успешного батча.

    ``show_chunk_progress=False`` отключает inner tqdm-бар - удобно
    при параллельной загрузке, чтобы прогресс-бары разных воркеров
    не накладывались.
    """
    chunks = iter_chunks(start, end, batch_months)
    if not chunks:
        return pd.DataFrame()

    state = _ResumeState(target_path)
    _run_chunks(
        chunks, state,
        fetch=fetch, retries=retries, backoff_sec=backoff_sec,
        label=label, show_chunk_progress=show_chunk_progress,
    )
    return state.snapshot()


class _ResumeState:
    """Инкрементальное состояние батч-загрузки.

    Существующий CSV ЛЕНИВО загружается в память только при первом
    необходимом merge - если все чанки уже скачаны, мы не читаем
    сотни МБ CSV вообще.
    """

    def __init__(self, target_path: Path) -> None:
        self.target_path = target_path
        self.existing_max = _existing_max_timestamp(target_path)
        self._accumulated: list[pd.DataFrame] = []
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.target_path.exists():
            self._accumulated.append(storage.load_raw_csv(self.target_path))
        self._loaded = True

    def append_and_save(self, df: pd.DataFrame) -> None:
        self._ensure_loaded()
        self._accumulated.append(df)
        merged, last = _merge_and_save(self._accumulated, self.target_path)
        self._accumulated = [merged]
        self.existing_max = last

    def snapshot(self) -> pd.DataFrame:
        if self._accumulated:
            return self._accumulated[-1]
        # Ничего не докачали: либо файл уже полный, либо ничего не было.
        if self.target_path.exists():
            return storage.load_raw_csv(self.target_path)
        return pd.DataFrame()
