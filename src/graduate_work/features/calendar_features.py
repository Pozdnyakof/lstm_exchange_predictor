"""Преобразование календарей и дивидендов в bar-level фичи.

Из календарных таблиц (которые лежат в ``data/raw/calendars/``) и
дивидендов (``data/raw/dividends/``) делаем разреженные event-флаги и
непрерывные «days-to-event» признаки на сетке 5-мин баров.

Все функции принимают целевой ``target_index`` (DatetimeIndex UTC) и
возвращают DataFrame с этим же индексом.

Префиксы итоговых колонок:

* ``cal_*`` — календарные (trading day flags, session phase, дней до экспирации).
* ``div_*`` — дивидендные (days-to-ex, ex-day flag, last yield).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trading day flags
# ---------------------------------------------------------------------------

def trading_day_features(
    trading_days: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Из ``trading_days_stock`` календаря строит bar-level флаги.

    Ожидает столбцы: ``tradedate``, ``is_traded``, ``reason``
    (``H``=holiday, ``W``=weekend, ``N``=normal, ``T``=transferred).

    Возвращает:
        ``cal_is_traded`` (0/1), ``cal_is_holiday`` (0/1),
        ``cal_is_transferred`` (0/1, перенесённый рабочий день),
        ``cal_days_to_holiday`` (0..N — сколько до следующего праздника),
        ``cal_days_since_holiday`` (0..N — сколько с прошлого).
    """
    out = pd.DataFrame(index=target_index)
    out["cal_is_traded"] = 1
    out["cal_is_holiday"] = 0
    out["cal_is_transferred"] = 0
    out["cal_days_to_holiday"] = 0
    out["cal_days_since_holiday"] = 0
    if trading_days is None or trading_days.empty:
        return out
    df = trading_days.copy()
    df["tradedate"] = pd.to_datetime(df["tradedate"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["tradedate"]).sort_values("tradedate")
    if "is_traded" in df.columns:
        df["is_traded"] = df["is_traded"].astype(int)
    else:
        df["is_traded"] = 1
    if "reason" not in df.columns:
        df["reason"] = "N"
    holidays = df.loc[df["reason"] == "H", "tradedate"].to_numpy()
    transferred = set(df.loc[df["reason"] == "T", "tradedate"].to_numpy())
    by_date = df.set_index("tradedate")["is_traded"]
    target_dates = target_index.normalize()
    out["cal_is_traded"] = by_date.reindex(target_dates).fillna(1).astype(int).to_numpy()
    out["cal_is_holiday"] = (
        target_dates.to_series().isin(holidays).astype(int).to_numpy()
    )
    out["cal_is_transferred"] = [
        1 if d in transferred else 0 for d in target_dates
    ]
    if len(holidays) > 0:
        out["cal_days_to_holiday"] = _days_to_next_event(target_dates, holidays)
        out["cal_days_since_holiday"] = _days_since_last_event(target_dates, holidays)
    return out


def _normalize_to_naive_days(values) -> np.ndarray:
    """Привести dates / events к naive numpy datetime64[D] для арифметики."""
    s = pd.to_datetime(values, utc=True, errors="coerce")
    # Снимаем UTC-таймзону → naive (np.datetime64 не работает с tz-aware).
    if hasattr(s, "tz_convert"):
        s = s.tz_convert("UTC").tz_localize(None)
    elif hasattr(s, "tz") and s.tz is not None:
        s = s.tz_localize(None)
    return pd.DatetimeIndex(s).normalize().to_numpy().astype("datetime64[D]")


def _days_to_next_event(
    dates: pd.DatetimeIndex, events: np.ndarray,
) -> np.ndarray:
    """Кол-во дней до ближайшего будущего event'а (включая today)."""
    events_sorted = np.unique(_normalize_to_naive_days(events))
    out = np.zeros(len(dates), dtype=int)
    if events_sorted.size == 0:
        return out
    target = _normalize_to_naive_days(dates)
    idx = np.searchsorted(events_sorted, target, side="left")
    for i, ev_idx in enumerate(idx):
        if ev_idx >= events_sorted.size:
            out[i] = 0  # нет будущих событий
        else:
            delta_days = (events_sorted[ev_idx] - target[i]).astype("int64")
            out[i] = int(delta_days)
    return out


def _days_since_last_event(
    dates: pd.DatetimeIndex, events: np.ndarray,
) -> np.ndarray:
    """Кол-во дней с последнего прошедшего event'а."""
    events_sorted = np.unique(_normalize_to_naive_days(events))
    out = np.zeros(len(dates), dtype=int)
    if events_sorted.size == 0:
        return out
    target = _normalize_to_naive_days(dates)
    idx = np.searchsorted(events_sorted, target, side="right") - 1
    for i, ev_idx in enumerate(idx):
        if ev_idx < 0:
            out[i] = 0
        else:
            delta_days = (target[i] - events_sorted[ev_idx]).astype("int64")
            out[i] = int(delta_days)
    return out


# ---------------------------------------------------------------------------
# Dividend features
# ---------------------------------------------------------------------------

def dividend_features(
    dividends: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    last_close_price: float | None = None,
) -> pd.DataFrame:
    """Per-bar дивидендные фичи на основе ``registryclosedate``.

    Args:
        dividends: фрейм со столбцами ``registryclosedate, value, currencyid``.
        target_index: на какие бары проецировать.
        last_close_price: для расчёта ``div_yield_last``. Если None,
            yield не считается.

    Returns:
        ``div_days_to_ex`` (>=0 до записи; -1 если уже не будет),
        ``div_days_since_ex``,
        ``div_is_ex_day`` (1 ровно в день закрытия реестра),
        ``div_value_last`` (RUB, последняя выплата на момент бара),
        ``div_value_next`` (RUB, ближайшая будущая выплата если известна).
    """
    out = pd.DataFrame(index=target_index)
    out["div_days_to_ex"] = 0
    out["div_days_since_ex"] = 0
    out["div_is_ex_day"] = 0
    out["div_value_last"] = 0.0
    out["div_value_next"] = 0.0
    if dividends is None or dividends.empty:
        return out
    if "registryclosedate" not in dividends.columns:
        return out
    div = dividends.copy()
    div["registryclosedate"] = pd.to_datetime(
        div["registryclosedate"], utc=True, errors="coerce",
    ).dt.normalize()
    div = div.dropna(subset=["registryclosedate"]).sort_values("registryclosedate")
    if div.empty:
        return out

    ex_dates = div["registryclosedate"].to_numpy()
    values = div["value"].astype(float).to_numpy()
    target_dates = target_index.normalize()
    target_arr = target_dates.to_numpy()

    out["cal_days_to_holiday"] = 0  # placeholder — drop из выхода
    days_to = _days_to_next_event(target_dates, ex_dates)
    out["div_days_to_ex"] = days_to
    days_since = _days_since_last_event(target_dates, ex_dates)
    out["div_days_since_ex"] = days_since
    out["div_is_ex_day"] = (days_to == 0).astype(int) | (days_since == 0).astype(int)

    # value_last: значение события с индексом = searchsorted-right-1.
    last_idx = np.searchsorted(ex_dates, target_arr, side="right") - 1
    last_idx_clipped = np.clip(last_idx, 0, len(values) - 1)
    has_past = last_idx >= 0
    out["div_value_last"] = np.where(has_past, values[last_idx_clipped], 0.0)
    # value_next: searchsorted-left.
    next_idx = np.searchsorted(ex_dates, target_arr, side="left")
    next_idx_clipped = np.clip(next_idx, 0, len(values) - 1)
    has_future = next_idx < len(values)
    out["div_value_next"] = np.where(has_future, values[next_idx_clipped], 0.0)

    if last_close_price and last_close_price > 0:
        out["div_yield_last"] = out["div_value_last"] / last_close_price
        out["div_yield_next"] = out["div_value_next"] / last_close_price

    out = out.drop(columns=["cal_days_to_holiday"], errors="ignore")
    return out


# ---------------------------------------------------------------------------
# Futures expirations (per-ticker can be derived if needed)
# ---------------------------------------------------------------------------

def expirations_features(
    expirations: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    asset_code: str | None = None,
) -> pd.DataFrame:
    """``cal_days_to_expiration`` — дней до ближайшей экспирации.

    Если ``asset_code`` задан, фильтруем календарь до соответствующей
    серии (например, ``GOLD``, ``Si``). Иначе берём весь набор.
    """
    out = pd.DataFrame(index=target_index)
    out["cal_days_to_expiration"] = 0
    if expirations is None or expirations.empty:
        return out
    df = expirations.copy()
    if asset_code and "asset_code" in df.columns:
        df = df[df["asset_code"] == asset_code]
    date_col = next(
        (c for c in ("expiration_date", "lasttradedate", "tradedate") if c in df.columns),
        None,
    )
    if date_col is None or df.empty:
        return out
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna().dt.normalize().to_numpy()
    out["cal_days_to_expiration"] = _days_to_next_event(target_index.normalize(), dates)
    return out
