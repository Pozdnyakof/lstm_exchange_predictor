"""Событийно-ориентированный бэктест-движок.

Допущения, упрощающие реализацию (соответствуют дипломному прототипу):
    - Дневной таймфрейм; вход и выход - по close-цене того же дня.
    - Сделка длится ровно ``horizon`` дней - выходим в close через h дней
      (если не сработал stop_loss / profit_target раньше).
    - Капитал делится поровну между активными позициями (sizing_mode
      "equal_split", legacy) либо по фиксированной доле / Kelly-доле
      (см. :class:`TradingConfig`).
    - Комиссия и проскальзывание - линейные проценты, применяются к
      каждой стороне сделки.

## Sizing modes

- ``"equal_split"`` — старое поведение: ``budget = cash / free_slots``.
- ``"fixed_frac"``  — ``budget = initial_capital * position_size_fraction``.
- ``"signal_kelly"`` — берёт ``size_fraction`` из колонки сигнала
  (precomputed Kelly, см. :func:`graduate_work.model.kelly_sizing.signal_kelly_size`),
  fallback к ``position_size_fraction`` если колонки нет.

Все режимы дополнительно ограничены ``max_position_size_fraction *
initial_capital`` и текущим cash.

## SL / PT

При ``stop_loss_pct > 0`` или ``profit_target_pct > 0`` engine ищет
intra-bar exit на каждом баре от entry+1 до horizon-exit. Для этого
``prices`` должен содержать колонки ``high`` и ``low``. Внутри одного
бара приоритет: SL > PT > horizon-close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    exit_reason: str = "horizon"  # horizon | stop_loss | profit_target | end_of_test


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# --------------------------------------------------------------------------
# Price lookups
# --------------------------------------------------------------------------

@dataclass
class _PriceLookups:
    """Все 4 OHLC-словаря для O(1) доступа в горячем цикле."""

    close: dict[tuple[pd.Timestamp, str], float]
    open: dict[tuple[pd.Timestamp, str], float]
    high: dict[tuple[pd.Timestamp, str], float] | None
    low: dict[tuple[pd.Timestamp, str], float] | None


def _column_lookup(
    prices: pd.DataFrame, col: str,
) -> dict[tuple[pd.Timestamp, str], float]:
    out: dict[tuple[pd.Timestamp, str], float] = {}
    for ts, ticker, val in zip(
        prices.index, prices["ticker"], prices[col], strict=True,
    ):
        out[(pd.Timestamp(ts), str(ticker))] = float(val)
    return out


def _build_lookups(prices: pd.DataFrame) -> _PriceLookups:
    close = _column_lookup(prices, "close")
    op = _column_lookup(prices, "open") if "open" in prices.columns else close
    high = _column_lookup(prices, "high") if "high" in prices.columns else None
    low = _column_lookup(prices, "low") if "low" in prices.columns else None
    return _PriceLookups(close=close, open=op, high=high, low=low)


# Backward-compat wrappers (used by consensus_engine).
def _price_lookup(prices: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    return _column_lookup(prices, "close")


def _open_lookup(prices: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    if "open" not in prices.columns:
        return _column_lookup(prices, "close")
    return _column_lookup(prices, "open")


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def _budget_for_signal(
    row: pd.Series,
    cfg: TradingConfig,
    cash: float,
    free_slots: int,
) -> float:
    """Бюджет одного входа в зависимости от ``cfg.sizing_mode``."""
    cap = cfg.initial_capital * cfg.max_position_size_fraction
    if cfg.sizing_mode == "equal_split":
        budget = cash / max(free_slots, 1)
    elif cfg.sizing_mode == "fixed_frac":
        budget = cfg.initial_capital * cfg.position_size_fraction
    elif cfg.sizing_mode == "signal_kelly":
        # signal-row колонка size_fraction задаётся пользователем
        # (precomputed Kelly), fallback к фиксированной доле.
        if "size_fraction" in row.index and pd.notna(row["size_fraction"]):
            budget = cfg.initial_capital * float(row["size_fraction"])
        else:
            budget = cfg.initial_capital * cfg.position_size_fraction
    else:
        msg = f"unknown sizing_mode: {cfg.sizing_mode!r}"
        raise ValueError(msg)
    return min(cash, budget, cap)


# --------------------------------------------------------------------------
# Position close — общий путь (фиксация PnL и cash)
# --------------------------------------------------------------------------

def _close_position(
    pos: dict[str, Any],
    exit_price: float,
    close_day: pd.Timestamp,
    cost_close: float,
    cash: float,
    trades: list[Trade],
    reason: str,
) -> float:
    """Зафиксировать exit, добавить trade, вернуть обновлённый cash."""
    gross = pos["quantity"] * exit_price
    fees = gross * cost_close
    proceeds = gross - fees
    cash += proceeds
    pnl = proceeds - pos["invested"]
    ret_pct = pnl / pos["invested"] if pos["invested"] > 0 else 0.0
    trades.append(
        Trade(
            open_date=pos["open_date"],
            close_date=close_day,
            ticker=pos["ticker"],
            horizon=pos["horizon"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            quantity=pos["quantity"],
            pnl=pnl,
            return_pct=ret_pct,
            exit_reason=reason,
        ),
    )
    return cash


# --------------------------------------------------------------------------
# Per-day phases
# --------------------------------------------------------------------------

def _try_intrabar_exit(
    pos: dict[str, Any],
    day: pd.Timestamp,
    lookups: _PriceLookups,
    cfg: TradingConfig,
) -> tuple[float | None, str]:
    """SL/PT-проверка внутри бара ``day``. Возвращает (exit_price, reason).

    Если ни один триггер не сработал → (None, "").
    Приоритет: SL > PT (консервативно — assume worst-fill).
    """
    if cfg.stop_loss_pct <= 0 and cfg.profit_target_pct <= 0:
        return None, ""
    if lookups.high is None or lookups.low is None:
        return None, ""
    key = (day, pos["ticker"])
    low = lookups.low.get(key)
    high = lookups.high.get(key)
    if low is None or high is None:
        return None, ""
    if cfg.stop_loss_pct > 0:
        sl_price = pos["entry_price"] * (1.0 - cfg.stop_loss_pct)
        if low <= sl_price:
            return sl_price, "stop_loss"
    if cfg.profit_target_pct > 0:
        pt_price = pos["entry_price"] * (1.0 + cfg.profit_target_pct)
        if high >= pt_price:
            return pt_price, "profit_target"
    return None, ""


def _close_due_or_intrabar(
    open_positions: list[dict[str, Any]],
    day: pd.Timestamp,
    bar_index: dict[pd.Timestamp, int],
    trading_days: pd.DatetimeIndex,
    lookups: _PriceLookups,
    cfg: TradingConfig,
    cost_close: float,
    cash: float,
    trades: list[Trade],
) -> tuple[list[dict[str, Any]], float]:
    """Закрыть позиции с close_date==day и/или сработавшим SL/PT.

    Возвращает (still_open, cash).
    """
    still_open: list[dict[str, Any]] = []
    for pos in open_positions:
        # 1) intra-bar SL/PT (только начиная со дня ПОСЛЕ entry)
        if day > pos["open_date"]:
            exit_price, reason = _try_intrabar_exit(pos, day, lookups, cfg)
            if exit_price is not None:
                cash = _close_position(
                    pos, exit_price, day, cost_close, cash, trades, reason,
                )
                continue
        # 2) horizon-close
        if pos["close_date"] == day:
            exit_price = lookups.open.get((day, pos["ticker"]))
            if exit_price is None:
                cur_idx = bar_index.get(day)
                if cur_idx is None or cur_idx + 1 >= len(trading_days):
                    exit_price = pos["entry_price"]
                else:
                    pos["close_date"] = trading_days[cur_idx + 1]
                    still_open.append(pos)
                    continue
            cash = _close_position(
                pos, exit_price, day, cost_close, cash, trades, "horizon",
            )
        else:
            still_open.append(pos)
    return still_open, cash


def _open_new_positions(
    day_signals: pd.DataFrame,
    day: pd.Timestamp,
    bar_index: dict[pd.Timestamp, int],
    trading_days: pd.DatetimeIndex,
    lookups: _PriceLookups,
    cfg: TradingConfig,
    cost_open: float,
    open_positions: list[dict[str, Any]],
    cash: float,
) -> float:
    """Открыть новые позиции по сигналам дня. Возвращает обновлённый cash."""
    if day_signals is None or day_signals.empty:
        return cash
    held_tickers = {str(p["ticker"]) for p in open_positions}
    free_slots = cfg.max_positions - len(open_positions)
    if free_slots <= 0:
        return cash
    fresh = day_signals[~day_signals["ticker"].astype(str).isin(held_tickers)]
    top = fresh.sort_values("mean", ascending=False).head(free_slots)
    for _, row in top.iterrows():
        horizon_bars = int(row["horizon"])
        entry_idx = bar_index.get(day)
        if (
            entry_idx is None
            or entry_idx + 1 + horizon_bars >= len(trading_days)
        ):
            continue
        entry_bar = trading_days[entry_idx + 1]
        exit_bar = trading_days[entry_idx + 1 + horizon_bars]
        price = lookups.open.get((entry_bar, row["ticker"]))
        if price is None or price <= 0:
            continue
        invest = _budget_for_signal(row, cfg, cash, free_slots)
        if invest <= 0:
            continue
        fees_in = invest * cost_open
        qty = (invest - fees_in) / price
        if qty <= 0:
            continue
        cash -= invest
        open_positions.append(
            {
                "open_date": entry_bar,
                "close_date": exit_bar,
                "ticker": str(row["ticker"]),
                "horizon": horizon_bars,
                "entry_price": price,
                "quantity": qty,
                "invested": invest,
            },
        )
    return cash


def _mark_to_market(
    cash: float,
    open_positions: list[dict[str, Any]],
    day: pd.Timestamp,
    lookups: _PriceLookups,
) -> float:
    portfolio_value = cash
    for pos in open_positions:
        mtm_price = lookups.close.get((day, pos["ticker"]), pos["entry_price"])
        portfolio_value += pos["quantity"] * mtm_price
    return portfolio_value


def _close_tail_positions(
    open_positions: list[dict[str, Any]],
    last_day: pd.Timestamp,
    lookups: _PriceLookups,
    cost_close: float,
    cash: float,
    trades: list[Trade],
) -> float:
    for pos in open_positions:
        exit_price = lookups.close.get((last_day, pos["ticker"]), pos["entry_price"])
        cash = _close_position(
            pos, exit_price, last_day, cost_close, cash, trades, "end_of_test",
        )
    return cash


# --------------------------------------------------------------------------
# Public entry-point
# --------------------------------------------------------------------------

def _normalize_inputs(
    signals: pd.DataFrame, prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return signals, prices


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: TradingConfig,
) -> BacktestResult:
    """Прокатить сигналы по историческому окну тестовой выборки.

    ``signals`` - выход :class:`SignalGenerator` (BUY/HOLD).
    ``prices``  - сводная таблица OHLC по тикерам, индекс - DatetimeIndex.
    """
    if signals.empty or prices.empty:
        return BacktestResult(
            equity=pd.Series([cfg.initial_capital]),
            trades=pd.DataFrame(),
        )

    signals, prices = _normalize_inputs(signals, prices)
    lookups = _build_lookups(prices)
    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    bar_index: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(trading_days)}

    cash = cfg.initial_capital
    open_positions: list[dict[str, Any]] = []
    daily_equity: dict[pd.Timestamp, float] = {}
    trades: list[Trade] = []

    cost_open = cfg.commission_rate + cfg.slippage_rate
    cost_close = cfg.commission_rate + cfg.slippage_rate

    buy_signals = signals[signals["action"] == "BUY"]
    grouped = dict(tuple(buy_signals.groupby("timestamp", sort=True)))

    for day in tqdm(trading_days, desc="Backtest", unit="bar", leave=False):
        open_positions, cash = _close_due_or_intrabar(
            open_positions, day, bar_index, trading_days, lookups, cfg,
            cost_close, cash, trades,
        )
        cash = _open_new_positions(
            grouped.get(day), day, bar_index, trading_days, lookups, cfg,
            cost_open, open_positions, cash,
        )
        daily_equity[day] = _mark_to_market(cash, open_positions, day, lookups)

    if open_positions:
        last_day = trading_days[-1]
        cash = _close_tail_positions(
            open_positions, last_day, lookups, cost_close, cash, trades,
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
    """Срезать из таблицы фич минимальный набор колонок (OHLC + ticker).

    Индекс остаётся DatetimeIndex (timestamp). ``open`` нужен engine'у
    для входа/выхода на open[t+1] — устранение look-ahead bias.
    ``high``/``low`` — для intra-bar SL/PT (если включены в TradingConfig).
    """
    needed = {"close", "ticker"}
    if not needed.issubset(full.columns):
        msg = f"full_frame must contain columns {needed}"
        raise ValueError(msg)
    keep: list[str] = []
    for col in ("open", "high", "low", "close", "ticker"):
        if col in full.columns:
            keep.append(col)
    out = full[keep].copy()
    return out
