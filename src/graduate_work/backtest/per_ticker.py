"""Per-ticker бэктест: изолированный прогон стратегии по каждому тикеру.

Используется для оценки, на каких именно активах модель показывает
устойчивое предсказательное преимущество. Капитал делится поровну
между тикерами; каждый тикер получает однотикерный движок.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ..config import TradingConfig
from .engine import run_backtest
from .metrics import compute_metrics


def run_per_ticker_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: TradingConfig,
    *,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """Запустить бэктест отдельно по каждому тикеру.

    ``periods_per_year`` - используется для аннуализации Sharpe в каждом
    под-бэктесте. Для 5-минутных баров MOEX-сессии ≈ 26460.
    """
    if signals.empty or prices.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "n_trades", "total_return", "sharpe",
                "max_drawdown", "win_rate", "profit_factor", "final_equity",
            ],
        )

    tickers = sorted(prices["ticker"].unique().tolist())
    if not tickers:
        return pd.DataFrame()

    per_ticker_capital = cfg.initial_capital / len(tickers)
    sub_cfg = replace(cfg, initial_capital=per_ticker_capital, max_positions=1)

    rows: list[dict] = []
    for ticker in tickers:
        s = signals[signals["ticker"] == ticker].copy()
        p = prices[prices["ticker"] == ticker].copy()
        if p.empty:
            continue
        bt = run_backtest(s, p, sub_cfg)
        metrics = compute_metrics(bt.equity, bt.trades, periods_per_year=periods_per_year)
        rows.append({"ticker": ticker, **metrics})

    return pd.DataFrame(rows)
