"""Метрики результативности торговой стратегии."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, *, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    sigma = float(excess.std(ddof=0))
    if sigma <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma * math.sqrt(TRADING_DAYS))


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


def annualized_return(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    n = len(equity)
    years = n / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    daily = equity.pct_change().fillna(0.0) if not equity.empty else pd.Series(dtype=float)
    return {
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity),
        "n_trades": int(len(trades)),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
    }
