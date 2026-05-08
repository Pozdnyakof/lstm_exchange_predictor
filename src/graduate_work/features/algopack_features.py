"""Feature engineering поверх ALGOPACK SuperCandles.

Из трёх 5-мин ALGOPACK-таблиц (TradeStats, OrderStats, OBStats)
строим **microstructure features** — то, чего не хватало модели для
предсказания short-term direction. Все они хорошо документированы
в академической литературе как сильные интрадей-предикторы:

* **Order flow imbalance (OFI)** — Cont, Kukanov & Stoikov (2014).
* **Aggressive trade imbalance** — Hasbrouck (1991), Lee-Ready (1991).
* **Spread/microprice** — Stoikov (2018).
* **Cancel/order ratio** — Hautsch (2012).

Каждая фича — float, нормированная или нативная. Загружаются с диска
(куда их сохранил ALGOPACK-фетчер) и приводятся к нашему 5-мин bar-grid'у
тикера через outer-merge на (timestamp, ticker).

Полный список добавляемых колонок (префиксы `aps_*` = ALGOPACK Stats):

| group | column | source | смысл |
|---|---|---|---|
| trade | aps_vol_imb         | TradeStats | (vol_b - vol_s) / vol — aggressive imbalance |
| trade | aps_val_imb         | TradeStats | (val_b - val_s) / val |
| trade | aps_trades_imb      | TradeStats | (trades_b - trades_s) / trades |
| trade | aps_disb            | TradeStats | сырой disbalance индикатор |
| trade | aps_vwap_premium    | TradeStats | (pr_vwap_b - pr_vwap_s) / pr_vwap |
| order | aps_put_vol_imb     | OrderStats | (put_vol_b - put_vol_s) / put_vol |
| order | aps_put_orders_imb  | OrderStats | (put_orders_b - put_orders_s) / put_orders |
| order | aps_cancel_ratio    | OrderStats | cancel_orders / (put + cancel) |
| order | aps_cancel_imb      | OrderStats | (cancel_orders_b - cancel_orders_s) / cancel_orders |
| order | aps_order_to_trade  | OrderStats+TradeStats | put_orders / trades — liquidity supply density |
| book  | aps_spread_bbo_bp   | OBStats | spread_bbo / mid_price (в б.п.) |
| book  | aps_spread_lv10_bp  | OBStats | spread_lv10 / mid_price |
| book  | aps_imb_vol_bbo     | OBStats | imbalance_vol_bbo (уже [-1, 1]) |
| book  | aps_imb_val_bbo     | OBStats | imbalance_val_bbo |
| book  | aps_levels_imb      | OBStats | (levels_b - levels_s) / (levels_b + levels_s) |
| book  | aps_depth_imb_1mio  | OBStats | (vwap_b_1mio - vwap_s_1mio) / vwap |

Все денормированы fail-safe (деление на ноль возвращает 0).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Деление с заменой 0/NaN в знаменателе на NaN (потом fillna(0))."""
    out = num / den.where(den.abs() > _EPS, np.nan)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Per-product feature builders
# ---------------------------------------------------------------------------

def tradestats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Из TradeStats строит aggressive imbalance + vwap-premium."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    vol = df.get("vol", pd.Series(dtype=float)).astype(float)
    val = df.get("val", pd.Series(dtype=float)).astype(float)
    trades = df.get("trades", pd.Series(dtype=float)).astype(float)
    out["aps_vol_imb"] = _safe_div(
        df["vol_b"].astype(float) - df["vol_s"].astype(float), vol,
    )
    out["aps_val_imb"] = _safe_div(
        df["val_b"].astype(float) - df["val_s"].astype(float), val,
    )
    out["aps_trades_imb"] = _safe_div(
        df["trades_b"].astype(float) - df["trades_s"].astype(float), trades,
    )
    if "disb" in df.columns:
        out["aps_disb"] = df["disb"].astype(float).fillna(0.0)
    if {"pr_vwap", "pr_vwap_b", "pr_vwap_s"} <= set(df.columns):
        vwap = df["pr_vwap"].astype(float)
        diff = df["pr_vwap_b"].astype(float) - df["pr_vwap_s"].astype(float)
        out["aps_vwap_premium"] = _safe_div(diff, vwap)
    return out


def orderstats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Из OrderStats: order-flow imbalance, cancel ratio."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    put_vol = df.get("put_vol", pd.Series(dtype=float)).astype(float)
    put_orders = df.get("put_orders", pd.Series(dtype=float)).astype(float)
    cancel_orders = df.get("cancel_orders", pd.Series(dtype=float)).astype(float)
    out["aps_put_vol_imb"] = _safe_div(
        df["put_vol_b"].astype(float) - df["put_vol_s"].astype(float), put_vol,
    )
    out["aps_put_orders_imb"] = _safe_div(
        df["put_orders_b"].astype(float) - df["put_orders_s"].astype(float),
        put_orders,
    )
    out["aps_cancel_ratio"] = _safe_div(
        cancel_orders, put_orders + cancel_orders,
    )
    if {"cancel_orders_b", "cancel_orders_s"} <= set(df.columns):
        out["aps_cancel_imb"] = _safe_div(
            df["cancel_orders_b"].astype(float)
            - df["cancel_orders_s"].astype(float),
            cancel_orders,
        )
    return out


def obstats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Из OBStats: спрэды, order-book imbalance, levels asymmetry."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    # Mid price — для спрэдов в bp.
    mid = None
    if "mid_price" in df.columns:
        mid = df["mid_price"].astype(float)
    elif {"vwap_b", "vwap_s"} <= set(df.columns):
        mid = 0.5 * (df["vwap_b"].astype(float) + df["vwap_s"].astype(float))

    if "spread_bbo" in df.columns:
        s = df["spread_bbo"].astype(float)
        out["aps_spread_bbo_bp"] = (
            _safe_div(s, mid) * 1e4 if mid is not None else s.fillna(0.0)
        )
    if "spread_lv10" in df.columns:
        s = df["spread_lv10"].astype(float)
        out["aps_spread_lv10_bp"] = (
            _safe_div(s, mid) * 1e4 if mid is not None else s.fillna(0.0)
        )
    if "imbalance_vol_bbo" in df.columns:
        out["aps_imb_vol_bbo"] = df["imbalance_vol_bbo"].astype(float).fillna(0.0)
    if "imbalance_val_bbo" in df.columns:
        out["aps_imb_val_bbo"] = df["imbalance_val_bbo"].astype(float).fillna(0.0)
    if {"levels_b", "levels_s"} <= set(df.columns):
        lb = df["levels_b"].astype(float)
        ls = df["levels_s"].astype(float)
        out["aps_levels_imb"] = _safe_div(lb - ls, lb + ls)
    if {"vwap_b_1mio", "vwap_s_1mio"} <= set(df.columns) and mid is not None:
        diff = df["vwap_b_1mio"].astype(float) - df["vwap_s_1mio"].astype(float)
        out["aps_depth_imb_1mio"] = _safe_div(diff, mid)
    return out


# ---------------------------------------------------------------------------
# Cross-product helper
# ---------------------------------------------------------------------------

def order_to_trade_ratio(
    orderstats: pd.DataFrame, tradestats: pd.DataFrame,
) -> pd.Series:
    """Outer-merge orderstats и tradestats по индексу + put_orders / trades."""
    if orderstats.empty or tradestats.empty:
        return pd.Series(dtype=float)
    merged = orderstats[["put_orders"]].join(
        tradestats[["trades"]], how="outer",
    ).fillna(0.0)
    return _safe_div(
        merged["put_orders"].astype(float),
        merged["trades"].astype(float),
    ).rename("aps_order_to_trade")


def build_algopack_features(
    *,
    tradestats: pd.DataFrame | None = None,
    orderstats: pd.DataFrame | None = None,
    obstats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Собрать все доступные ALGOPACK-фичи в один frame.

    Каждый аргумент — DataFrame с UTC-индексом (как из
    :class:`AlgopackClient`). Отсутствующие источники тихо
    пропускаются.

    Возвращает frame проиндексированный объединением всех индексов,
    NaN-значения после outer-merge заполняются нулями (микроструктурный
    сигнал = 0 при отсутствии данных, что трактуется как "нейтрально").
    """
    parts: list[pd.DataFrame] = []
    if tradestats is not None and not tradestats.empty:
        parts.append(tradestats_features(tradestats))
    if orderstats is not None and not orderstats.empty:
        parts.append(orderstats_features(orderstats))
    if obstats is not None and not obstats.empty:
        parts.append(obstats_features(obstats))
    if (
        tradestats is not None and orderstats is not None
        and not tradestats.empty and not orderstats.empty
    ):
        ratio = order_to_trade_ratio(orderstats, tradestats)
        if not ratio.empty:
            parts.append(ratio.to_frame())

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    return out.fillna(0.0)
