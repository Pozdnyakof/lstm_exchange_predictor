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
    """Aggressive imbalance + intraday vol + price change.

    Использует все signal-богатые колонки TradeStats:
    vol/val/trades (totals), {vol,val,trades}_{b,s} (split),
    disb, pr_vwap, pr_vwap_{b,s}, pr_std, pr_change.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    vol = df.get("vol", pd.Series(dtype=float)).astype(float)
    val = df.get("val", pd.Series(dtype=float)).astype(float)
    trades = df.get("trades", pd.Series(dtype=float)).astype(float)

    # Aggressive imbalance: volume / value / trades.
    out["aps_vol_imb"] = _safe_div(
        df["vol_b"].astype(float) - df["vol_s"].astype(float), vol,
    )
    out["aps_val_imb"] = _safe_div(
        df["val_b"].astype(float) - df["val_s"].astype(float), val,
    )
    out["aps_trades_imb"] = _safe_div(
        df["trades_b"].astype(float) - df["trades_s"].astype(float), trades,
    )
    # Avg trade size by side — крупные институциональные следы.
    if {"vol_b", "trades_b"} <= set(df.columns):
        out["aps_avg_size_b"] = _safe_div(
            df["vol_b"].astype(float), df["trades_b"].astype(float),
        )
    if {"vol_s", "trades_s"} <= set(df.columns):
        out["aps_avg_size_s"] = _safe_div(
            df["vol_s"].astype(float), df["trades_s"].astype(float),
        )

    if "disb" in df.columns:
        out["aps_disb"] = df["disb"].astype(float).fillna(0.0)
    if {"pr_vwap", "pr_vwap_b", "pr_vwap_s"} <= set(df.columns):
        vwap = df["pr_vwap"].astype(float)
        diff = df["pr_vwap_b"].astype(float) - df["pr_vwap_s"].astype(float)
        out["aps_vwap_premium"] = _safe_div(diff, vwap)

    # Intraday realized vol: pr_std нормированный на среднюю цену.
    if {"pr_std", "pr_vwap"} <= set(df.columns):
        out["aps_intra_vol_bp"] = _safe_div(
            df["pr_std"].astype(float), df["pr_vwap"].astype(float),
        ) * 1e4
    elif "pr_std" in df.columns:
        out["aps_intra_vol"] = df["pr_std"].astype(float).fillna(0.0)

    # Per-bar price change as fraction of vwap.
    if "pr_change" in df.columns:
        if "pr_vwap" in df.columns:
            out["aps_pr_change_bp"] = _safe_div(
                df["pr_change"].astype(float), df["pr_vwap"].astype(float),
            ) * 1e4
        else:
            out["aps_pr_change"] = df["pr_change"].astype(float).fillna(0.0)

    return out


def orderstats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор: put / cancel × {vol, val, orders} × {b, s} + vwap."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    cols = set(df.columns)

    # ---- placed orders ----
    put_vol = df.get("put_vol", pd.Series(dtype=float)).astype(float)
    put_val = df.get("put_val", pd.Series(dtype=float)).astype(float)
    put_orders = df.get("put_orders", pd.Series(dtype=float)).astype(float)

    if {"put_vol_b", "put_vol_s"} <= cols:
        out["aps_put_vol_imb"] = _safe_div(
            df["put_vol_b"].astype(float) - df["put_vol_s"].astype(float), put_vol,
        )
    if {"put_val_b", "put_val_s"} <= cols:
        out["aps_put_val_imb"] = _safe_div(
            df["put_val_b"].astype(float) - df["put_val_s"].astype(float), put_val,
        )
    if {"put_orders_b", "put_orders_s"} <= cols:
        out["aps_put_orders_imb"] = _safe_div(
            df["put_orders_b"].astype(float) - df["put_orders_s"].astype(float),
            put_orders,
        )
    # VWAP-премия размещённых: где покупатели агрессивнее ставят?
    if {"put_vwap_b", "put_vwap_s", "put_vwap"} <= cols:
        vwap = df["put_vwap"].astype(float)
        diff = df["put_vwap_b"].astype(float) - df["put_vwap_s"].astype(float)
        out["aps_put_vwap_premium"] = _safe_div(diff, vwap)
    # Avg placed size: насколько крупные заявки размещаются.
    if {"put_vol", "put_orders"} <= cols:
        out["aps_put_avg_size"] = _safe_div(put_vol, put_orders)

    # ---- cancellations ----
    cancel_vol = df.get("cancel_vol", pd.Series(dtype=float)).astype(float)
    cancel_val = df.get("cancel_val", pd.Series(dtype=float)).astype(float)
    cancel_orders = df.get("cancel_orders", pd.Series(dtype=float)).astype(float)

    if {"cancel_vol_b", "cancel_vol_s"} <= cols:
        out["aps_cancel_vol_imb"] = _safe_div(
            df["cancel_vol_b"].astype(float)
            - df["cancel_vol_s"].astype(float),
            cancel_vol,
        )
    if {"cancel_val_b", "cancel_val_s"} <= cols:
        out["aps_cancel_val_imb"] = _safe_div(
            df["cancel_val_b"].astype(float)
            - df["cancel_val_s"].astype(float),
            cancel_val,
        )
    if {"cancel_orders_b", "cancel_orders_s"} <= cols:
        out["aps_cancel_orders_imb"] = _safe_div(
            df["cancel_orders_b"].astype(float)
            - df["cancel_orders_s"].astype(float),
            cancel_orders,
        )
    if {"cancel_vwap_b", "cancel_vwap_s", "cancel_vwap"} <= cols:
        vwap = df["cancel_vwap"].astype(float)
        diff = df["cancel_vwap_b"].astype(float) - df["cancel_vwap_s"].astype(float)
        out["aps_cancel_vwap_premium"] = _safe_div(diff, vwap)

    # Cancel ratio (объёмы и orders).
    out["aps_cancel_ratio_orders"] = _safe_div(
        cancel_orders, put_orders + cancel_orders,
    )
    out["aps_cancel_ratio_vol"] = _safe_div(
        cancel_vol, put_vol + cancel_vol,
    )
    # Asymmetric cancel intensity: насколько одна сторона отменяется чаще.
    if {"cancel_orders_b", "cancel_orders_s",
        "put_orders_b", "put_orders_s"} <= cols:
        out["aps_cancel_rate_b"] = _safe_div(
            df["cancel_orders_b"].astype(float),
            df["put_orders_b"].astype(float),
        )
        out["aps_cancel_rate_s"] = _safe_div(
            df["cancel_orders_s"].astype(float),
            df["put_orders_s"].astype(float),
        )

    return out


def obstats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полная схема OBStats: спрэды (bbo/lv10/1mio), 4 типа imbalance,
    resting depth по сторонам (vol_b/s, val_b/s) и levels asymmetry.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    cols = set(df.columns)

    # Mid price — для всех bp-нормализаций.
    mid: pd.Series | None = None
    if "mid_price" in cols:
        mid = df["mid_price"].astype(float)
    elif {"vwap_b", "vwap_s"} <= cols:
        mid = 0.5 * (df["vwap_b"].astype(float) + df["vwap_s"].astype(float))

    # ---- Spreads ----
    for spread_col, out_col in [
        ("spread_bbo",  "aps_spread_bbo_bp"),
        ("spread_lv10", "aps_spread_lv10_bp"),
        ("spread_1mio", "aps_spread_1mio_bp"),
    ]:
        if spread_col in cols:
            s = df[spread_col].astype(float)
            out[out_col] = (
                _safe_div(s, mid) * 1e4 if mid is not None else s.fillna(0.0)
            )

    # ---- Imbalance (4 варианта: BBO + full-book × volume/value) ----
    for src, dst in [
        ("imbalance_vol_bbo", "aps_imb_vol_bbo"),
        ("imbalance_val_bbo", "aps_imb_val_bbo"),
        ("imbalance_vol",     "aps_imb_vol_full"),
        ("imbalance_val",     "aps_imb_val_full"),
    ]:
        if src in cols:
            out[dst] = df[src].astype(float).fillna(0.0)

    # ---- Levels asymmetry ----
    if {"levels_b", "levels_s"} <= cols:
        lb = df["levels_b"].astype(float)
        ls = df["levels_s"].astype(float)
        out["aps_levels_imb"] = _safe_div(lb - ls, lb + ls)
        out["aps_levels_total"] = (lb + ls).fillna(0.0)

    # ---- Resting depth (vol_b/s, val_b/s) — это РАЗНЫЕ от агрессивных
    # из TradeStats: тут стоящие в стакане лимитные заявки.
    if {"vol_b", "vol_s"} <= cols:
        vb = df["vol_b"].astype(float)
        vs = df["vol_s"].astype(float)
        out["aps_depth_vol_imb"] = _safe_div(vb - vs, vb + vs)
    if {"val_b", "val_s"} <= cols:
        ab = df["val_b"].astype(float)
        as_ = df["val_s"].astype(float)
        out["aps_depth_val_imb"] = _safe_div(ab - as_, ab + as_)

    # ---- $1M-effective spreads ----
    if {"vwap_b_1mio", "vwap_s_1mio"} <= cols and mid is not None:
        diff = df["vwap_b_1mio"].astype(float) - df["vwap_s_1mio"].astype(float)
        out["aps_depth_imb_1mio"] = _safe_div(diff, mid)
        # Стоимость прохода $1M в bid/ask — отдельно как proxy на market impact.
        out["aps_impact_b_1mio_bp"] = _safe_div(
            df["vwap_b_1mio"].astype(float) - mid, mid,
        ) * 1e4
        out["aps_impact_s_1mio_bp"] = _safe_div(
            mid - df["vwap_s_1mio"].astype(float), mid,
        ) * 1e4

    # ---- Resting VWAP premium (vwap_b - vwap_s normalized) ----
    if {"vwap_b", "vwap_s"} <= cols and mid is not None:
        diff = df["vwap_b"].astype(float) - df["vwap_s"].astype(float)
        out["aps_resting_vwap_premium_bp"] = _safe_div(diff, mid) * 1e4

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


def hi2_features(
    hi2: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Daily Herfindahl-индекс концентрации торгов → bar-level через ffill.

    Высокий HI2 = торговля сконцентрирована в немногих игроках/сделках,
    что часто предшествует резким движениям. Низкий HI2 = равномерное
    распределение = более стабильный регим.

    Колонки HI2: ``tradedate, secid, metric, value, reference``.
    Метрик может быть несколько (hhi_aggressive, hhi_passive, ...) —
    pivot'им по metric, ffill на bar-grid.
    """
    out = pd.DataFrame(index=target_index)
    if hi2 is None or hi2.empty or "tradedate" not in hi2.columns:
        return out
    df = hi2.copy()
    df["tradedate"] = pd.to_datetime(df["tradedate"], utc=True, errors="coerce")
    df = df.dropna(subset=["tradedate"])
    if "metric" not in df.columns or "value" not in df.columns:
        return out
    pivot = df.pivot_table(
        index="tradedate", columns="metric", values="value", aggfunc="last",
    )
    aligned = pivot.reindex(target_index, method="ffill")
    for metric_name in aligned.columns:
        col = f"aps_hi2_{str(metric_name).lower()}"
        out[col] = aligned[metric_name].astype(float).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Schema diagnostic
# ---------------------------------------------------------------------------

def print_schema(
    *,
    tradestats: pd.DataFrame | None = None,
    orderstats: pd.DataFrame | None = None,
    obstats: pd.DataFrame | None = None,
    hi2: pd.DataFrame | None = None,
) -> None:
    """Печатает реальные колонки каждого ALGOPACK-источника.

    Полезно сравнить с :data:`PRODUCTS` schema из docs — если
    каких-то колонок нет, увидим явно. Если есть extra колонки —
    можно их подключить.
    """
    sources = {
        "TradeStats": tradestats, "OrderStats": orderstats,
        "OBStats":    obstats,    "HI2":        hi2,
    }
    for name, df in sources.items():
        if df is None:
            continue
        if df.empty:
            print(f"[{name}] EMPTY")
            continue
        print(f"[{name}] {len(df):,} rows × {len(df.columns)} columns:")
        print(f"  cols = {list(df.columns)}")
        if hasattr(df.index, "min") and len(df) > 0:
            print(f"  range = {df.index.min()} .. {df.index.max()}")


def build_algopack_features(
    *,
    tradestats: pd.DataFrame | None = None,
    orderstats: pd.DataFrame | None = None,
    obstats: pd.DataFrame | None = None,
    hi2: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Собрать все доступные ALGOPACK-фичи в один frame.

    Каждый аргумент — DataFrame с UTC-индексом (как из
    :class:`AlgopackClient`). Отсутствующие источники тихо
    пропускаются. HI2 daily-данные ffill'ятся на bar-grid через
    :func:`hi2_features` (требует non-empty target_index).
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

    # HI2 в конце — отдельной веткой через target_index выходного frame.
    if hi2 is not None and not hi2.empty:
        target_idx = pd.DatetimeIndex(out.index)
        if len(target_idx) > 0:
            hi2_block = hi2_features(hi2, target_idx)
            if not hi2_block.empty:
                out = pd.concat([out, hi2_block], axis=1)

    return out.fillna(0.0)
