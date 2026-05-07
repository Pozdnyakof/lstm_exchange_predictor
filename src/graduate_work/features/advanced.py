"""Расширенный набор технических индикаторов и временных признаков.

Все индикаторы - стандартные (TA-Lib классика), что соответствует §1.2 ВКР:
«осцилляторы по множеству исторических окон, скользящие средние... объёмы
торгов, показатели исторической волатильности».

Состав:
- MACD (12, 26, 9) - моментум-осциллятор;
- Bollinger Bands (20, 2σ): %B и bandwidth;
- Stochastic Oscillator (14, 3): %K и %D;
- Williams %R (14);
- CCI (Commodity Channel Index, 20);
- ROC (Rate of Change) на нескольких окнах;
- OBV (On-Balance Volume) - log-нормализованный;
- MFI (Money Flow Index, 14);
- Rolling Hurst exponent (R/S analysis) - фрактальная персистентность;
- Temporal features: sin/cos часа сессии, дня недели, дня месяца.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_RADIANS = 2.0 * math.pi
# MOEX основная сессия 10:00-18:45 МСК = 07:00-15:45 UTC.
_MOEX_SESSION_START_MIN_UTC = 7 * 60
_MOEX_SESSION_LEN_MIN = 525


# ---------------------------------------------------------------------------
# Осцилляторы
# ---------------------------------------------------------------------------

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: разность EMA-fast и EMA-slow, signal-линия и гистограмма."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = (ema_fast - ema_slow) / close.replace(0.0, np.nan)
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": sig,
            "macd_hist": macd - sig,
        },
        index=close.index,
    )


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger %B и bandwidth - устойчивы к масштабу цены."""
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_pct_b": pct_b, "bb_bandwidth": bandwidth},
        index=close.index,
    )


def _stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    high = df["high"].rolling(window=window, min_periods=window).max()
    low = df["low"].rolling(window=window, min_periods=window).min()
    rng = (high - low).replace(0.0, np.nan)
    k = (df["close"] - low) / rng
    d = k.rolling(window=smooth, min_periods=smooth).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)


def _williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].rolling(window=window, min_periods=window).max()
    low = df["low"].rolling(window=window, min_periods=window).min()
    rng = (high - low).replace(0.0, np.nan)
    # Стандартная формула возвращает [-100, 0]; нормируем к [-1, 0].
    return -1.0 * (high - df["close"]) / rng


def _cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(window=window, min_periods=window).mean()
    mad = tp.rolling(window=window, min_periods=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))


def _roc(close: pd.Series, periods: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for p in periods:
        out[f"roc_{p}"] = close.pct_change(periods=p)
    return out


# ---------------------------------------------------------------------------
# Объёмные индикаторы
# ---------------------------------------------------------------------------

def _obv(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vol = df["volume"].astype(float)
    direction = np.sign(df["close"].diff()).fillna(0.0)
    obv = (vol * direction).cumsum()
    # OBV растёт неограниченно - стандартизуем относительным
    # отклонением от скользящей.
    obv_ma = obv.rolling(window=50, min_periods=50).mean()
    obv_std = obv.rolling(window=50, min_periods=50).std()
    return ((obv - obv_ma) / (obv_std + 1e-9)).fillna(0.0)


def _mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    money_flow = tp * df["volume"].astype(float)
    delta = tp.diff()
    pos_mf = money_flow.where(delta > 0, 0.0)
    neg_mf = money_flow.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(window=window, min_periods=window).sum()
    neg_sum = neg_mf.rolling(window=window, min_periods=window).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + ratio)).fillna(50.0) / 100.0


# ---------------------------------------------------------------------------
# Hurst exponent (фрактальная персистентность)
# ---------------------------------------------------------------------------

def _hurst_rs_window(values: np.ndarray) -> float:
    n = values.size
    if n < 20:
        return np.nan
    if np.isnan(values).any():
        return np.nan
    mean = np.mean(values)
    devs = np.cumsum(values - mean)
    r = float(np.ptp(devs))
    s = float(np.std(values, ddof=1))
    if s < 1e-10:
        return np.nan
    return float(np.log(r / s + 1e-10) / np.log(n))


def _rolling_hurst(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Hurst exponent через R/S analysis."""
    return returns.rolling(window=window, min_periods=window).apply(
        _hurst_rs_window,
        raw=True,
    )


# ---------------------------------------------------------------------------
# Temporal features (cyclical encoding)
# ---------------------------------------------------------------------------

def _temporal_block(index: pd.DatetimeIndex, *, market: str = "moex") -> pd.DataFrame:
    """sin/cos часа торговой сессии, дня недели, дня месяца."""
    if not isinstance(index, pd.DatetimeIndex):
        msg = "DatetimeIndex required for temporal features"
        raise TypeError(msg)
    hour = index.hour
    minute = index.minute
    dow = index.dayofweek
    dom = index.day

    if market == "moex":
        # Доля внутри сессии 07:00-15:45 UTC, нормирована [0, 1].
        utc_minute = hour * 60 + minute
        session_frac = np.clip(
            (utc_minute - _MOEX_SESSION_START_MIN_UTC) / _MOEX_SESSION_LEN_MIN,
            0.0, 1.0,
        )
    else:
        session_frac = (hour * 60 + minute) / (24 * 60)

    return pd.DataFrame(
        {
            "time_sin_daily": np.sin(_RADIANS * session_frac),
            "time_cos_daily": np.cos(_RADIANS * session_frac),
            "dow_sin": np.sin(_RADIANS * dow / 7.0),
            "dow_cos": np.cos(_RADIANS * dow / 7.0),
            "dom_sin": np.sin(_RADIANS * dom / 31.0),
            "dom_cos": np.cos(_RADIANS * dom / 31.0),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def add_advanced_indicators(
    df: pd.DataFrame,
    *,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    bb_window: int = 20,
    stoch_window: int = 14,
    cci_window: int = 20,
    williams_window: int = 14,
    roc_periods: tuple[int, ...] = (5, 10, 20),
    mfi_window: int = 14,
    hurst_window: int = 60,
    use_hurst: bool = True,
    use_temporal: bool = True,
    market: str = "moex",
) -> pd.DataFrame:
    """Дополнить таблицу с базовыми ТА расширенным набором осцилляторов
    и (опц.) Hurst-показателем + темпоральными признаками.

    На вход приходит DataFrame, уже обработанный `add_technical_indicators`
    (содержит колонки close/high/low + log_return).
    """
    out = df.copy()
    close = out["close"].astype(float)

    macd_df = _macd(close, *macd_params)
    bb_df = _bollinger(close, window=bb_window)
    stoch_df = _stochastic(out, window=stoch_window)
    williams = _williams_r(out, window=williams_window)
    cci = _cci(out, window=cci_window)
    roc_df = _roc(close, periods=roc_periods)
    obv = _obv(out)
    mfi = _mfi(out, window=mfi_window)

    for col in macd_df.columns:
        out[col] = macd_df[col]
    for col in bb_df.columns:
        out[col] = bb_df[col]
    for col in stoch_df.columns:
        out[col] = stoch_df[col]
    out["williams_r"] = williams
    out["cci_20"] = cci
    for col in roc_df.columns:
        out[col] = roc_df[col]
    out["obv_zscore"] = obv
    out["mfi_14"] = mfi

    if use_hurst and "log_return" in out.columns:
        out[f"hurst_{hurst_window}"] = _rolling_hurst(out["log_return"], window=hurst_window)

    if use_temporal:
        temporal = _temporal_block(out.index, market=market)
        for col in temporal.columns:
            out[col] = temporal[col]

    return out
