"""Метрики результативности торговой стратегии."""

from __future__ import annotations

import math

import pandas as pd

# Справочное число торговых дней в году для дневной частоты equity.
# При intraday-equity используйте `bars_per_year=` явно.
TRADING_DAYS = 252


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: float = TRADING_DAYS,
) -> float:
    """Annualised Sharpe.

    ``periods_per_year`` - число шагов equity в году. Для дневной серии
    это ~252; для 5-минутных баров MOEX ≈ 252 * 105 = 26460.
    """
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    sigma = float(excess.std(ddof=0))
    if sigma <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gains = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(equity: pd.Series, *, periods_per_year: float = TRADING_DAYS) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    n = len(equity)
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    periods_per_year: float = TRADING_DAYS,
) -> dict[str, float]:
    """Сводный набор метрик.

    ``periods_per_year`` важен, если equity сэмплирован чаще дня (например,
    по 5-минутным барам). Для 5-мин MOEX-сессии ≈ 26460.
    """
    daily = equity.pct_change().fillna(0.0) if not equity.empty else pd.Series(dtype=float)
    return {
        "sharpe": sharpe_ratio(daily, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity, periods_per_year=periods_per_year),
        "n_trades": int(len(trades)),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
    }
