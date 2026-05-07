"""Технические индикаторы.

Базируется на наработках исследовательского проекта (см.
`src/experiment_03/features/technical.py` и `transforms.py`),
адаптировано под минималистичный пайплайн ВКР: только то, что
работает на 5-минутном таймфрейме без дополнительных данных
ордерфлоу.

Состав:
- log-доходности (1 бар)
- фрактальное дифференцирование close (FFD по Лопе де Прадо) -
  снимает тренд при сохранении долгосрочной памяти
- скользящие статистики цены: SMA / EMA расхождения, моментум
- объёмные ряды: лог-объём, z-score, отношение к скользящей
- волатильность: rolling std лог-доходностей, асимметричная
  (отдельно вверх/вниз), Garman-Klass (использует OHLC)
- NATR(14) - нормализованный ATR
- осциллятор RSI(14)
- shape бара: тени, доминирование тела, относительный диапазон
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Фрактальное дифференцирование (Lopez de Prado, "Advances in Financial ML")
# ---------------------------------------------------------------------------

def fractional_diff(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-4,
) -> pd.Series:
    """FFD - фрактально-дифференцированный ряд.

    Сохраняет долгосрочную память (в отличие от обычной первой
    разности log-цены), но устраняет нестационарность тренда.
    Параметр ``d`` ∈ (0, 1) - порядок дифференцирования; типично
    0.3-0.5. ``threshold`` отсекает хвост биномиальных весов.
    """
    values = series.astype(float).to_numpy()
    weights: list[float] = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    w_arr = np.array(weights[::-1], dtype=np.float64)
    wlen = w_arr.size
    result = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(wlen - 1, len(values)):
        win = values[i - wlen + 1:i + 1]
        if not np.isnan(win).any():
            result[i] = float(np.dot(w_arr, win))
    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# Вспомогательные индикаторы
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0) / 100.0


def _natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalized Average True Range / close."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean() / close


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass volatility estimator на одном баре."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    op = df["open"].astype(float)
    ln_hl = np.log(high / low.replace(0.0, np.nan))
    ln_co = np.log(close / op.replace(0.0, np.nan))
    return 0.5 * ln_hl**2 - (2.0 * np.log(2.0) - 1.0) * ln_co**2


# ---------------------------------------------------------------------------
# Группы признаков (каждая возвращает DataFrame и список колонок)
# ---------------------------------------------------------------------------

def _price_distance(df: pd.DataFrame, sma: tuple[int, ...], ema: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    cols: list[str] = []
    close = df["close"].astype(float)
    for p in sma:
        sma_v = close.rolling(window=p, min_periods=p).mean()
        out[f"sma_{p}_rel"] = (close / sma_v - 1.0)
        cols.append(f"sma_{p}_rel")
    for p in ema:
        ema_v = close.ewm(span=p, adjust=False, min_periods=p).mean()
        out[f"ema_{p}_rel"] = (close / ema_v - 1.0)
        cols.append(f"ema_{p}_rel")
    return out, cols


def _momentum_block(df: pd.DataFrame, periods: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    cols: list[str] = []
    close = df["close"].astype(float)
    for p in periods:
        out[f"mom_{p}"] = close.pct_change(periods=p)
        out[f"logret_{p}"] = np.log(close / close.shift(p))
        cols.extend([f"mom_{p}", f"logret_{p}"])
    return out, cols


def _volatility_block(
    log_ret: pd.Series,
    windows: tuple[int, ...],
) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=log_ret.index)
    cols: list[str] = []
    pos = log_ret.clip(lower=0.0)
    neg = log_ret.clip(upper=0.0)
    for w in windows:
        out[f"vol_{w}"] = log_ret.rolling(window=w, min_periods=w).std()
        rv_up = pos.rolling(window=w, min_periods=w).std()
        rv_dn = neg.rolling(window=w, min_periods=w).std()
        out[f"rv_up_{w}"] = rv_up
        out[f"rv_dn_{w}"] = rv_dn
        out[f"vol_asym_{w}"] = (rv_dn - rv_up) / (rv_up + rv_dn + 1e-9)
        cols.extend([f"vol_{w}", f"rv_up_{w}", f"rv_dn_{w}", f"vol_asym_{w}"])
    return out, cols


def _volume_block(
    df: pd.DataFrame,
    windows: tuple[int, ...],
) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    cols: list[str] = []
    if "volume" not in df.columns:
        return out, cols
    vol = df["volume"].astype(float).replace(0.0, np.nan)
    out["volume_log"] = np.log(vol + 1.0)
    cols.append("volume_log")
    for w in windows:
        m = vol.rolling(window=w, min_periods=w).mean()
        s = vol.rolling(window=w, min_periods=w).std()
        out[f"volume_rel_{w}"] = (vol / m).fillna(1.0)
        out[f"volume_zscore_{w}"] = ((vol - m) / (s + 1e-9)).fillna(0.0)
        cols.extend([f"volume_rel_{w}", f"volume_zscore_{w}"])
    return out, cols


def _bar_shape(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return out, []
    op = df["open"].astype(float)
    cl = df["close"].astype(float)
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    body_top = np.maximum(op, cl)
    body_bot = np.minimum(op, cl)
    out["shadow_upper"] = (hi - body_top) / cl.replace(0.0, np.nan)
    out["shadow_lower"] = (body_bot - lo) / cl.replace(0.0, np.nan)
    out["body_dominance"] = (cl - op).abs() / (hi - lo + 1e-9)
    out["range_rel"] = (hi - lo) / cl.replace(0.0, np.nan)
    return out, ["shadow_upper", "shadow_lower", "body_dominance", "range_rel"]


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def add_technical_indicators(
    df: pd.DataFrame,
    *,
    sma_periods: tuple[int, ...] = (5, 10, 20, 50),
    ema_periods: tuple[int, ...] = (12, 26),
    momentum_periods: tuple[int, ...] = (3, 6, 12),
    vol_windows: tuple[int, ...] = (12, 24, 48),
    fracdiff_d: float = 0.4,
    use_fracdiff: bool = True,
) -> pd.DataFrame:
    """Дополнить таблицу OHLCV производными индикаторами.

    На вход подаются 5-минутные бары (после ресэмпла из 1-мин).
    Параметры окон подобраны под этот таймфрейм: 12 баров = 1 час,
    48 баров = 4 часа, 50 SMA ≈ половина торговой сессии.
    """
    if "close" not in df.columns:
        msg = "DataFrame must contain a 'close' column"
        raise ValueError(msg)

    out = df.copy()
    close = out["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    # Фрактальная дифференциация цены - стационарный, но информативный признак.
    if use_fracdiff:
        out["fracdiff_close"] = fractional_diff(close, d=fracdiff_d)

    pieces: list[tuple[pd.DataFrame, list[str]]] = [
        _price_distance(out, sma_periods, ema_periods),
        _momentum_block(out, momentum_periods),
        _volatility_block(log_ret, vol_windows),
        _volume_block(out, vol_windows),
        _bar_shape(out),
    ]
    for piece_df, _ in pieces:
        for col in piece_df.columns:
            out[col] = piece_df[col]

    out["rsi_14"] = _rsi(close, 14)
    out["natr_14"] = _natr(out, 14)
    out["volatility_gk"] = _garman_klass(out)

    return out
