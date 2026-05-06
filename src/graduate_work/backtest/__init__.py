"""Модуль 4: ретроспективное тестирование."""

from .engine import BacktestResult, run_backtest
from .metrics import compute_metrics
from .per_ticker import run_per_ticker_backtest
from .random_portfolios import RandomPortfolioReport, run_random_portfolios

__all__ = [
    "BacktestResult",
    "RandomPortfolioReport",
    "compute_metrics",
    "run_backtest",
    "run_per_ticker_backtest",
    "run_random_portfolios",
]
