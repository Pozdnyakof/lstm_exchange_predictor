"""Бэктест-движок с signal-driven exit.

Отличие от :func:`run_backtest`: позиция не закрывается по фиксированному
горизонту, а закрывается по close-сигналу (выход consensus-фильтра).
Жёсткий потолок на длительность удержания — ``max_hold_bars`` —
гарантирует, что позиция не зависнет навсегда, если close-сигнал так
и не сработает.

Возвращаемая структура (`BacktestResult` с теми же полями `equity`,
`trades`) намеренно совпадает с :class:`run_backtest`, чтобы тот же
:func:`compute_metrics` посчитал sharpe / max_dd / win_rate без
костылей.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import TradingConfig
from .engine import BacktestResult, Trade, _open_lookup, _price_lookup

logger = logging.getLogger(__name__)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        msg = "prices DataFrame must have a DatetimeIndex"
        raise TypeError(msg)
    out = prices.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _grouped_decisions(decisions: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    decisions = decisions.copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True)
    return dict(tuple(decisions.groupby("timestamp", sort=True)))


def _close_existing(
    *, day: pd.Timestamp,
    open_positions: list[dict[str, Any]],
    decisions_today: pd.DataFrame | None,
    open_lookup: dict[tuple[pd.Timestamp, str], float],
    bar_index: dict[pd.Timestamp, int],
    trading_days: pd.DatetimeIndex,
    cost_close: float,
    max_hold_bars: int,
    cash_in: float,
    trades: list[Trade],
) -> tuple[float, list[dict[str, Any]]]:
    """Обработать close-условия: signal-driven exit + max-hold fallback."""
    cash = cash_in
    cur_idx = bar_index.get(day)
    still_open: list[dict[str, Any]] = []
    close_lookup: dict[str, bool] = {}
    if decisions_today is not None and not decisions_today.empty:
        close_lookup = dict(zip(
            decisions_today["ticker"].astype(str),
            decisions_today["close_long"].astype(bool),
            strict=False,
        ))

    for pos in open_positions:
        ticker = str(pos["ticker"])
        bars_held = (
            cur_idx - pos["entry_bar_idx"] if cur_idx is not None else 0
        )
        close_signal = close_lookup.get(ticker, False)
        force_max = bars_held >= max_hold_bars
        if not (close_signal or force_max):
            still_open.append(pos)
            continue
        # Exit по open следующего бара (так же, как у вход).
        if cur_idx is None or cur_idx + 1 >= len(trading_days):
            # Конец окна — закрываем по entry-цене (нулевой PnL).
            exit_bar = day
            exit_price = pos["entry_price"]
        else:
            exit_bar = trading_days[cur_idx + 1]
            exit_price = open_lookup.get((exit_bar, ticker))
            if exit_price is None:
                # Цены нет на следующем баре — пробуем через ещё один.
                if cur_idx + 2 < len(trading_days):
                    exit_bar = trading_days[cur_idx + 2]
                    exit_price = open_lookup.get((exit_bar, ticker))
            if exit_price is None:
                # Вторая попытка тоже пуста — оставляем позицию открытой.
                still_open.append(pos)
                continue
        gross = pos["quantity"] * exit_price
        fees = gross * cost_close
        proceeds = gross - fees
        cash += proceeds
        pnl = proceeds - pos["invested"]
        ret_pct = pnl / pos["invested"] if pos["invested"] > 0 else 0.0
        trades.append(Trade(
            open_date=pos["open_date"],
            close_date=exit_bar,
            ticker=ticker,
            horizon=int(bars_held),
            entry_price=pos["entry_price"],
            exit_price=float(exit_price),
            quantity=pos["quantity"],
            pnl=pnl,
            return_pct=ret_pct,
        ))
    return cash, still_open


def _open_new(
    *, day: pd.Timestamp,
    open_positions: list[dict[str, Any]],
    decisions_today: pd.DataFrame | None,
    cfg: TradingConfig,
    open_lookup: dict[tuple[pd.Timestamp, str], float],
    bar_index: dict[pd.Timestamp, int],
    trading_days: pd.DatetimeIndex,
    cost_open: float,
    cash_in: float,
) -> tuple[float, list[dict[str, Any]]]:
    if decisions_today is None or decisions_today.empty:
        return cash_in, open_positions
    cash = cash_in
    held = {str(p["ticker"]) for p in open_positions}
    free_slots = cfg.max_positions - len(open_positions)
    if free_slots <= 0:
        return cash, open_positions
    # Только бары с open_long и без активной позиции по тому же тикеру.
    fresh = decisions_today[decisions_today["open_long"]]
    fresh = fresh[~fresh["ticker"].astype(str).isin(held)]
    if fresh.empty:
        return cash, open_positions
    # Сортируем по убыванию p_long — приоритет уверенным сигналам.
    top = fresh.sort_values("p_long", ascending=False).head(free_slots)

    cur_idx = bar_index.get(day)
    if cur_idx is None or cur_idx + 1 >= len(trading_days):
        return cash, open_positions
    entry_bar = trading_days[cur_idx + 1]

    budget = cash / max(free_slots, 1)
    new_positions = list(open_positions)
    for _, row in top.iterrows():
        ticker = str(row["ticker"])
        price = open_lookup.get((entry_bar, ticker))
        if price is None or price <= 0:
            continue
        invest = min(cash, budget)
        if invest <= 0:
            break
        fees_in = invest * cost_open
        qty = (invest - fees_in) / price
        if qty <= 0:
            continue
        cash -= invest
        new_positions.append({
            "ticker": ticker,
            "open_date": entry_bar,
            "entry_bar_idx": cur_idx + 1,
            "entry_price": float(price),
            "quantity": float(qty),
            "invested": float(invest),
        })
    return cash, new_positions


def _mark_to_market(
    open_positions: list[dict[str, Any]],
    cash: float,
    day: pd.Timestamp,
    close_lookup: dict[tuple[pd.Timestamp, str], float],
) -> float:
    equity = cash
    for pos in open_positions:
        px = close_lookup.get((day, str(pos["ticker"])))
        if px is None:
            equity += pos["invested"]
            continue
        equity += pos["quantity"] * px
    return equity


def run_consensus_backtest(
    decisions: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: TradingConfig,
    *,
    max_hold_bars: int,
) -> BacktestResult:
    """Прокатить consensus-decisions через signal-driven event loop.

    Args:
        decisions: выход :func:`apply_consensus_thresholds` со столбцами
            ``open_long`` и ``close_long``. Должен содержать столбец
            ``p_long`` для ранжирования сигналов на open.
        prices: long-формат с колонками ``open``, ``close``, ``ticker``,
            индекс ``DatetimeIndex`` (UTC).
        cfg: ``TradingConfig`` (используются commission_rate, slippage_rate,
            max_positions, initial_capital).
        max_hold_bars: жёсткий потолок длительности удержания. Если
            close-сигнал не сработал — выход через ``max_hold_bars``
            баров после entry.
    """
    if decisions.empty or prices.empty:
        return BacktestResult(
            equity=pd.Series([cfg.initial_capital]),
            trades=pd.DataFrame(),
        )
    if max_hold_bars <= 0:
        msg = f"max_hold_bars must be positive, got {max_hold_bars}"
        raise ValueError(msg)

    prices = _normalize_prices(prices)
    close_lookup = _price_lookup(prices)
    open_lookup = _open_lookup(prices)

    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    bar_index = {ts: i for i, ts in enumerate(trading_days)}

    cost_open = cfg.commission_rate + cfg.slippage_rate
    cost_close = cfg.commission_rate + cfg.slippage_rate

    grouped = _grouped_decisions(decisions)

    cash = cfg.initial_capital
    open_positions: list[dict[str, Any]] = []
    trades: list[Trade] = []
    daily_equity: dict[pd.Timestamp, float] = {}

    for day in tqdm(trading_days, desc="Consensus BT", unit="bar", leave=False):
        decisions_today = grouped.get(day)
        cash, open_positions = _close_existing(
            day=day,
            open_positions=open_positions,
            decisions_today=decisions_today,
            open_lookup=open_lookup,
            bar_index=bar_index,
            trading_days=trading_days,
            cost_close=cost_close,
            max_hold_bars=max_hold_bars,
            cash_in=cash,
            trades=trades,
        )
        cash, open_positions = _open_new(
            day=day,
            open_positions=open_positions,
            decisions_today=decisions_today,
            cfg=cfg,
            open_lookup=open_lookup,
            bar_index=bar_index,
            trading_days=trading_days,
            cost_open=cost_open,
            cash_in=cash,
        )
        daily_equity[day] = _mark_to_market(open_positions, cash, day, close_lookup)

    # Принудительный close всех остатков по последнему close (нужно для
    # корректного финального equity и win_rate в метриках).
    if open_positions and trading_days.size > 0:
        last_day = trading_days[-1]
        for pos in open_positions:
            px = close_lookup.get((last_day, str(pos["ticker"])))
            if px is None:
                continue
            gross = pos["quantity"] * px
            proceeds = gross * (1.0 - cost_close)
            cash += proceeds
            pnl = proceeds - pos["invested"]
            ret_pct = pnl / pos["invested"] if pos["invested"] > 0 else 0.0
            trades.append(Trade(
                open_date=pos["open_date"],
                close_date=last_day,
                ticker=str(pos["ticker"]),
                horizon=int(bar_index[last_day] - pos["entry_bar_idx"]),
                entry_price=pos["entry_price"],
                exit_price=float(px),
                quantity=pos["quantity"],
                pnl=pnl,
                return_pct=ret_pct,
            ))
        daily_equity[last_day] = cash

    equity = pd.Series(daily_equity, name="equity").sort_index()
    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
    )
    return BacktestResult(equity=equity, trades=trades_df)
