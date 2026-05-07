"""Тестирование стратегии против ансамбля случайных портфелей.

Реализует методологию §1.3 ВКР: генерируется множество стохастических
агентов, каждый из которых случайно покупает / продаёт активы при тех же
инфраструктурных ограничениях (комиссии, размер портфеля, средний срок
удержания), что и тестируемая стратегия. Распределение конечных
доходностей формирует эталонный шум; решение признаётся статистически
значимым только при превышении 3σ от среднего этого распределения.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import TradingConfig


@dataclass
class RandomPortfolioReport:
    final_equities: np.ndarray
    mean: float
    std: float
    sigma_threshold: float
    threshold_value: float       # mean + sigma_threshold * std
    strategy_final: float
    strategy_z_score: float
    is_significant: bool
    final_returns: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(prices.index, pd.DatetimeIndex):
        msg = "prices index must be DatetimeIndex"
        raise TypeError(msg)
    return pd.DatetimeIndex(sorted(prices.index.unique()))


def _simulate_one(
    rng: np.random.Generator,
    *,
    calendar: pd.DatetimeIndex,
    tickers: list[str],
    lookup: dict[tuple[pd.Timestamp, str], float],
    cfg: TradingConfig,
    avg_horizon: int,
    trade_probability: float,
) -> float:
    """Один запуск случайного агента; возвращает конечный капитал."""
    cost_open = cfg.commission_rate + cfg.slippage_rate
    cost_close = cfg.commission_rate + cfg.slippage_rate
    cash = cfg.initial_capital
    positions: list[dict] = []

    for day in calendar:
        # 1) Закрываем созревшие позиции.
        still: list[dict] = []
        for pos in positions:
            if pos["close_date"] <= day:
                price = lookup.get((day, pos["ticker"]), pos["entry_price"])
                gross = pos["quantity"] * price
                cash += gross - gross * cost_close
            else:
                still.append(pos)
        positions = still

        # 2) Случайным образом, с заданной вероятностью, открываем позицию.
        if len(positions) >= cfg.max_positions:
            continue
        if rng.random() >= trade_probability:
            continue
        ticker = tickers[int(rng.integers(0, len(tickers)))]
        price = lookup.get((day, ticker))
        if price is None or price <= 0:
            continue

        free_slots = cfg.max_positions - len(positions)
        budget = cash / max(free_slots, 1)
        if budget <= 0:
            continue
        fees_in = budget * cost_open
        qty = (budget - fees_in) / price
        if qty <= 0:
            continue

        cash -= budget
        # Срок удержания случайный в окрестности среднего по стратегии.
        h = max(int(rng.integers(max(avg_horizon - 2, 1), max(avg_horizon + 3, 2))), 1)
        close_date = day + pd.Timedelta(days=h)
        future = calendar[calendar >= close_date]
        if len(future) == 0:
            cash += budget
            continue
        positions.append(
            {
                "open_date": day,
                "close_date": future[0],
                "ticker": ticker,
                "entry_price": price,
                "quantity": qty,
            },
        )

    # 3) Ликвидируем оставшиеся позиции по последнему close.
    if positions:
        last = calendar[-1]
        for pos in positions:
            price = lookup.get((last, pos["ticker"]), pos["entry_price"])
            gross = pos["quantity"] * price
            cash += gross - gross * cost_close
    return cash


def run_random_portfolios(
    prices: pd.DataFrame,
    cfg: TradingConfig,
    *,
    avg_horizon: int = 5,
    trade_probability: float = 0.05,
    strategy_final: float = 0.0,
    n_portfolios: int | None = None,
    seed: int = 42,
) -> RandomPortfolioReport:
    """Вернуть распределение случайных портфелей и оценку значимости."""
    n = n_portfolios or cfg.n_random_portfolios
    if prices.empty:
        return RandomPortfolioReport(
            final_equities=np.zeros(0),
            mean=0.0,
            std=0.0,
            sigma_threshold=cfg.sigma_threshold,
            threshold_value=0.0,
            strategy_final=strategy_final,
            strategy_z_score=0.0,
            is_significant=False,
        )

    calendar = _trading_calendar(prices)
    tickers = sorted(prices["ticker"].unique().tolist())
    lookup: dict[tuple[pd.Timestamp, str], float] = {
        (pd.Timestamp(ts), str(t)): float(c)
        for ts, t, c in zip(prices.index, prices["ticker"], prices["close"], strict=True)
    }

    rng = np.random.default_rng(seed)
    finals = np.empty(n, dtype=np.float64)
    for i in tqdm(range(n), desc="Random portfolios", unit="agent"):
        finals[i] = _simulate_one(
            rng,
            calendar=calendar,
            tickers=tickers,
            lookup=lookup,
            cfg=cfg,
            avg_horizon=avg_horizon,
            trade_probability=trade_probability,
        )

    mu = float(finals.mean())
    sigma = float(finals.std(ddof=0))
    threshold = mu + cfg.sigma_threshold * sigma
    z = (strategy_final - mu) / sigma if sigma > 1e-9 else 0.0
    final_returns = finals / cfg.initial_capital - 1.0
    return RandomPortfolioReport(
        final_equities=finals,
        mean=mu,
        std=sigma,
        sigma_threshold=cfg.sigma_threshold,
        threshold_value=threshold,
        strategy_final=strategy_final,
        strategy_z_score=z,
        is_significant=strategy_final >= threshold,
        final_returns=final_returns,
    )
