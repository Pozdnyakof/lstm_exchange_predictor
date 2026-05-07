"""Ресэмпл OHLCV-свечей до целевого таймфрейма + фильтр торговой сессии.

MOEX ISS отдаёт 1-минутки в UTC. Мы:
  1) фильтруем строки за пределами основной сессии (10:00-18:45 МСК),
  2) агрегируем до cfg.bar_minutes по правилам OHLC + sum(volume),
  3) выкидываем выходные.
"""

from __future__ import annotations

import pandas as pd

from ..config import DataConfig


def filter_moex_session(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Оставить только бары основной торговой сессии MOEX (UTC)."""
    if df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        msg = "DataFrame must have DatetimeIndex"
        raise TypeError(msg)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        df = df.set_axis(idx)
    start = pd.Timestamp(cfg.session_start_utc).time()
    end = pd.Timestamp(cfg.session_end_utc).time()
    mask = (idx.time >= start) & (idx.time <= end) & (idx.dayofweek < 5)
    return df.loc[mask]


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "ticker": "first",
}


def _resample_rule(bar_minutes: int) -> str:
    """Преобразовать минутный шаг в pandas-rule.

    Для bar_minutes >= 1440 (день) используем правило "1D" - оно явно
    привязано к календарному дню. Иначе - "Nmin".
    """
    if bar_minutes >= 1440:
        return "1D"
    return f"{bar_minutes}min"


def _select_agg(df: pd.DataFrame) -> dict[str, str]:
    """Подмножество OHLCV-агрегаций для колонок, реально присутствующих в df."""
    return {col: how for col, how in _OHLCV_AGG.items() if col in df.columns}


def resample_ohlcv(
    df: pd.DataFrame,
    cfg: DataConfig,
    *,
    apply_session_filter: bool = True,
) -> pd.DataFrame:
    """Свести 1-минутные OHLCV до cfg.bar_minutes.

    Колонки `ticker` и прочие нечисловые сохраняются (берётся first).
    Если bar_minutes == 1, возвращаем (отфильтрованный) исходник без агрегации.
    Для bar_minutes >= 1440 используется правило "1D" - один бар на
    календарный день (после session-фильтра - один бар на торговый день).
    """
    if df.empty:
        return df

    if apply_session_filter:
        df = filter_moex_session(df, cfg)
    if df.empty or cfg.bar_minutes <= 1:
        return df

    rule = _resample_rule(cfg.bar_minutes)
    agg = _select_agg(df)
    if not agg:
        return df

    resampled = df.resample(rule, label="left", closed="left").agg(agg)
    return resampled.dropna(subset=[c for c in ("open", "high", "low", "close") if c in resampled.columns])
