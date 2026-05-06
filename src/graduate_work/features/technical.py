"""Базовые технические индикаторы (§2.2 ВКР).

Состав ограничен примитивами, упомянутыми в работе: SMA/EMA различных
периодов, моментум-осциллятор RSI, историческая волатильность,
а также относительные изменения объёма.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _historic_vol(log_ret: pd.Series, window: int) -> pd.Series:
    return log_ret.rolling(window=window, min_periods=window).std()


def add_technical_indicators(
    df: pd.DataFrame,
    *,
    sma_periods: tuple[int, ...] = (5, 10, 20, 50),
    ema_periods: tuple[int, ...] = (12, 26),
    momentum_periods: tuple[int, ...] = (5, 10),
    vol_windows: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """Дополнить таблицу OHLCV производными индикаторами.

    Все индикаторы нормированы относительно цены закрытия / лог-доходности,
    чтобы признаки имели сопоставимые масштабы между тикерами.
    """
    if "close" not in df.columns:
        msg = "DataFrame must contain a 'close' column"
        raise ValueError(msg)

    out = df.copy()
    close = out["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    for p in sma_periods:
        sma = close.rolling(window=p, min_periods=p).mean()
        out[f"sma_{p}_rel"] = close / sma - 1.0
    for p in ema_periods:
        ema = close.ewm(span=p, adjust=False, min_periods=p).mean()
        out[f"ema_{p}_rel"] = close / ema - 1.0
    for p in momentum_periods:
        out[f"mom_{p}"] = close.pct_change(periods=p)
    for w in vol_windows:
        out[f"vol_{w}"] = _historic_vol(log_ret, w)

    out["rsi_14"] = _rsi(close, 14) / 100.0

    if "volume" in out.columns:
        vol = out["volume"].astype(float).replace(0.0, np.nan)
        vol_ma = vol.rolling(window=20, min_periods=20).mean()
        out["volume_rel"] = (vol / vol_ma).fillna(1.0)

    if {"high", "low", "close"}.issubset(out.columns):
        rng = (out["high"] - out["low"]).astype(float)
        out["range_rel"] = (rng / close).fillna(0.0)

    return out
