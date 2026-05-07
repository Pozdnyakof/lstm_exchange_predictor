"""Событийно-ориентированный бэктест-движок.

Допущения, упрощающие реализацию (соответствуют дипломному прототипу):
    - Дневной таймфрейм; вход и выход - по close-цене того же дня.
    - Сделка длится ровно ``horizon`` дней - выходим в close через h дней.
    - Капитал делится поровну между активными позициями.
    - Комиссия и проскальзывание - линейные проценты, применяются к
      каждой стороне сделки.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import TradingConfig


@dataclass
class Trade:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    ticker: str
    horizon: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _price_lookup(prices: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    """Построить плоский словарь (date, ticker) -> close для O(1) доступа."""
    lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for ts, ticker, close in zip(prices.index, prices["ticker"], prices["close"], strict=True):
        lookup[(pd.Timestamp(ts), str(ticker))] = float(close)
    return lookup


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: TradingConfig,
) -> BacktestResult:
    """Прокатить сигналы по историческому окну тестовой выборки.

    ``signals`` - выход :class:`SignalGenerator` (BUY/HOLD).
    ``prices``  - сводная таблица close по тикерам, индекс - DatetimeIndex.
    """
    if signals.empty or prices.empty:
        return BacktestResult(
            equity=pd.Series([cfg.initial_capital]),
            trades=pd.DataFrame(),
        )

    signals = signals.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)

    prices = prices.copy()
    if not isinstance(prices.index, pd.DatetimeIndex):
        msg = "prices DataFrame must have a DatetimeIndex"
        raise TypeError(msg)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    else:
        prices.index = prices.index.tz_convert("UTC")

    lookup = _price_lookup(prices)
    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    # Map timestamp -> integer position in trading_days. Используется для
    # строгого "horizon в барах": exit_idx = entry_idx + horizon.
    bar_index: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(trading_days)}

    equity = cfg.initial_capital
    cash = equity
    open_positions: list[dict] = []
    daily_equity: dict[pd.Timestamp, float] = {}
    trades: list[Trade] = []

    cost_open = cfg.commission_rate + cfg.slippage_rate
    cost_close = cfg.commission_rate + cfg.slippage_rate

    buy_signals = signals[signals["action"] == "BUY"]
    grouped = dict(tuple(buy_signals.groupby("timestamp", sort=True)))

    for day in tqdm(trading_days, desc="Backtest", unit="bar", leave=False):
        # 1) Закрываем позиции, у которых сегодня дата выхода.
        still_open: list[dict] = []
        for pos in open_positions:
            if pos["close_date"] == day:
                exit_price = lookup.get((day, pos["ticker"]))
                if exit_price is None:
                    # Нет данных по этому тикеру в этом баре - продлеваем
                    # на СЛЕДУЮЩИЙ БАР календаря (не на сутки!). При
                    # 5-минутном таймфрейме старая логика days=1 добавляла
                    # 24 часа stale-экспозиции, искажая PnL.
                    cur_idx = bar_index.get(day)
                    if cur_idx is None or cur_idx + 1 >= len(trading_days):
                        # Конец тестового окна - закрываем по entry-цене
                        # (нулевой PnL), чтобы не тащить позицию вечно.
                        exit_price = pos["entry_price"]
                    else:
                        pos["close_date"] = trading_days[cur_idx + 1]
                        still_open.append(pos)
                        continue
                gross = pos["quantity"] * exit_price
                fees = gross * cost_close
                proceeds = gross - fees
                cash += proceeds
                pnl = proceeds - pos["invested"]
                ret_pct = pnl / pos["invested"] if pos["invested"] > 0 else 0.0
                trades.append(
                    Trade(
                        open_date=pos["open_date"],
                        close_date=day,
                        ticker=pos["ticker"],
                        horizon=pos["horizon"],
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        quantity=pos["quantity"],
                        pnl=pnl,
                        return_pct=ret_pct,
                    ),
                )
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2) Открываем новые позиции по сигналам сегодняшней даты.
        day_signals = grouped.get(day)
        if day_signals is not None and not day_signals.empty:
            # Per-ticker uniqueness: не открываем второй вход на тот же
            # тикер, если по нему уже есть открытая позиция.
            held_tickers = {str(p["ticker"]) for p in open_positions}
            free_slots = cfg.max_positions - len(open_positions)
            if free_slots > 0:
                fresh_signals = day_signals[~day_signals["ticker"].astype(str).isin(held_tickers)]
                top = fresh_signals.sort_values("mean", ascending=False).head(free_slots)
                # Капитал делим равномерно между новыми входами.
                budget = cash / max(free_slots, 1)
                for _, row in top.iterrows():
                    price = lookup.get((day, row["ticker"]))
                    if price is None or price <= 0:
                        continue
                    invest = min(cash, budget)
                    if invest <= 0:
                        continue
                    fees_in = invest * cost_open
                    qty = (invest - fees_in) / price
                    if qty <= 0:
                        continue
                    cash -= invest
                    # Horizon выражен в БАРАХ - закрываем позицию через
                    # row["horizon"] баров от текущего entry-бара.
                    entry_idx = bar_index.get(day)
                    horizon_bars = int(row["horizon"])
                    if entry_idx is None or entry_idx + horizon_bars >= len(trading_days):
                        # Прогноз уходит за хвост тестового окна - откат.
                        cash += invest
                        continue
                    close_date = trading_days[entry_idx + horizon_bars]
                    open_positions.append(
                        {
                            "open_date": day,
                            "close_date": close_date,
                            "ticker": str(row["ticker"]),
                            "horizon": int(row["horizon"]),
                            "entry_price": price,
                            "quantity": qty,
                            "invested": invest,
                        },
                    )

        # 3) Mark-to-market оценка капитала на конец дня.
        portfolio_value = cash
        for pos in open_positions:
            mtm_price = lookup.get((day, pos["ticker"]), pos["entry_price"])
            portfolio_value += pos["quantity"] * mtm_price
        equity = portfolio_value
        daily_equity[day] = equity

    # 4) Закрываем все хвостовые позиции по последнему доступному close.
    if open_positions:
        last_day = trading_days[-1]
        for pos in open_positions:
            exit_price = lookup.get((last_day, pos["ticker"]), pos["entry_price"])
            gross = pos["quantity"] * exit_price
            fees = gross * cost_close
            proceeds = gross - fees
            cash += proceeds
            pnl = proceeds - pos["invested"]
            ret_pct = pnl / pos["invested"] if pos["invested"] > 0 else 0.0
            trades.append(
                Trade(
                    open_date=pos["open_date"],
                    close_date=last_day,
                    ticker=pos["ticker"],
                    horizon=pos["horizon"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    quantity=pos["quantity"],
                    pnl=pnl,
                    return_pct=ret_pct,
                ),
            )
        daily_equity[last_day] = cash

    equity_series = pd.Series(daily_equity, name="equity").sort_index()
    daily_returns = equity_series.pct_change().fillna(0.0)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(equity=equity_series, trades=trades_df, daily_returns=daily_returns)


# --------------------------------------------------------------------------
# Утилита: подготовка цен для движка
# --------------------------------------------------------------------------

def prices_from_full_frame(full: pd.DataFrame) -> pd.DataFrame:
    """Срезать из таблицы фич минимальный набор колонок (close + ticker).

    Индекс должен оставаться DatetimeIndex (timestamp).
    """
    needed = {"close", "ticker"}
    if not needed.issubset(full.columns):
        msg = f"full_frame must contain columns {needed}"
        raise ValueError(msg)
    out = full[["close", "ticker"]].copy()
    return out
