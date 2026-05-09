"""FUTOI features: Open Interest по типу участника (физ./юр.).

ALGOPACK FUTOI endpoint:
  ``https://apim.moex.com/iss/analyticalproducts/futoi/securities/{secid}.json``

**Доступный диапазон**: с 2024-10-01 (виден в meta-блоке `futoi.dates`).
Запросы за более ранний период возвращают empty.

**Схема (long-form, R-0053 inspection)**:
  ``sess_id, seqnum, tradedate, tradetime, ticker, clgroup,
   pos, pos_long, pos_short, pos_long_num, pos_short_num,
   systime, trade_session_date``

  - ``clgroup``: ``YUR`` (юрлица — фонды/банки) или ``FIZ`` (физлица — розница)
  - ``pos``: signed net positions (positive=net long; negative=net short)
  - ``pos_long``, ``pos_short``: gross позиции в каждом направлении
  - ``pos_long_num``, ``pos_short_num``: число открытых контрактов

На один timestamp — ДВЕ строки (YUR + FIZ). Pivot по clgroup даёт
wide format. После pivot строим quant-фичи:

- ``futoi_yur_pos`` / ``futoi_fiz_pos`` — net positioning per group
- ``futoi_yur_imbalance`` / ``futoi_fiz_imbalance`` — long/total ratio
- ``futoi_smart_divergence`` = ``yur_pos - fiz_pos`` (smart vs crowd)
- ``futoi_smart_imbalance_diff`` = ``yur_imb - fiz_imb``
- 1d / 5d Δ для каждого

Daily aggregation → bar-grid через ffill.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Перпы → spot-mapping. Используется в notebook для поиска FUTOI-файлов.
FUTURES_TO_SPOT: dict[str, str] = {
    "SBERF": "SBER",
    "GAZPF": "GAZP",
    "USDRUBF": "USDRUB",
    "CNYRUBF": "CNYRUB",
    "IMOEXF": "IMOEX",
    "GLDRUBF": "GLDRUB",
    "EURRUBF": "EURRUB",
}

# Группы участников — ровно эти ключи в FUTOI ALGOPACK.
_CLGROUPS = ("YUR", "FIZ")
_PIVOT_VALUES = ("pos", "pos_long", "pos_short", "pos_long_num", "pos_short_num")


def _safe_imbalance(long_v: pd.Series, short_v: pd.Series) -> pd.Series:
    """long / (long + abs(short)). Если short хранится как отрицательное —
    abs() делает формулу симметричной. NaN/0/0 → 0."""
    short_abs = short_v.abs()
    total = (long_v + short_abs).replace(0.0, np.nan)
    return ((long_v - short_abs) / total).fillna(0.0)


def _pivot_to_daily_panel(futoi_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form FUTOI → daily wide panel.

    Aggregate всех 5-min строк за день в одну (закрытие сессии). Pivot
    по clgroup даёт колонки типа ``pos_YUR``, ``pos_FIZ``, ...
    """
    if "tradedate" not in futoi_df.columns or "clgroup" not in futoi_df.columns:
        return pd.DataFrame()
    # reset_index убирает потенциальный конфликт "tradedate" как
    # одновременно index-name и column-name (из normalize_supercandle_index).
    df = futoi_df.reset_index(drop=True).copy()
    df["tradedate"] = pd.to_datetime(df["tradedate"], utc=True, errors="coerce")
    df = df.dropna(subset=["tradedate"])
    if df.empty:
        return pd.DataFrame()
    # Берём last-row каждого дня каждой группы (close-of-session positioning).
    if "tradetime" in df.columns:
        df = df.sort_values(["tradedate", "clgroup", "tradetime"])
    daily = df.groupby(["tradedate", "clgroup"], as_index=False).last()
    # Pivot к (tradedate × {pos_YUR, pos_FIZ, pos_long_YUR, pos_long_FIZ, ...}).
    available = [c for c in _PIVOT_VALUES if c in daily.columns]
    if not available:
        return pd.DataFrame()
    wide = daily.pivot(
        index="tradedate", columns="clgroup", values=available,
    )
    # MultiIndex columns → flat: e.g. ('pos', 'YUR') → 'pos_YUR'.
    wide.columns = [f"{val}_{grp}" for val, grp in wide.columns]
    return wide.sort_index()


def _add_group_features(
    out: pd.DataFrame, panel: pd.DataFrame, group: str,
) -> bool:
    """Записать pos / imbalance / Δ-фичи для одной группы (YUR или FIZ).

    Возвращает True, если pos_{group} был доступен и фичи добавлены.
    """
    pos_col = f"pos_{group}"
    if pos_col not in panel.columns:
        return False
    g = group.lower()
    pos = panel[pos_col].astype(float)
    out[f"futoi_{g}_pos"] = pos
    out[f"futoi_{g}_pos_d1"] = pos.diff(1).fillna(0.0)
    out[f"futoi_{g}_pos_d5"] = pos.diff(5).fillna(0.0)
    long_col, short_col = f"pos_long_{group}", f"pos_short_{group}"
    if {long_col, short_col} <= set(panel.columns):
        out[f"futoi_{g}_imbalance"] = _safe_imbalance(
            panel[long_col].astype(float), panel[short_col].astype(float),
        )
    return True


def _add_smart_divergence(out: pd.DataFrame) -> None:
    """smart_divergence = yur_pos − fiz_pos; smart_imbalance_diff = yur_imb − fiz_imb."""
    if "futoi_yur_pos" in out.columns and "futoi_fiz_pos" in out.columns:
        out["futoi_smart_divergence"] = out["futoi_yur_pos"] - out["futoi_fiz_pos"]
    if (
        "futoi_yur_imbalance" in out.columns
        and "futoi_fiz_imbalance" in out.columns
    ):
        out["futoi_smart_imbalance_diff"] = (
            out["futoi_yur_imbalance"] - out["futoi_fiz_imbalance"]
        )


def build_futoi_features(futoi_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form FUTOI ALGOPACK → daily quant-фичи positioning.

    Возвращает DataFrame с DatetimeIndex (UTC) и колонками:
        - ``futoi_yur_pos``, ``futoi_fiz_pos`` — net pos
        - ``futoi_yur_imbalance``, ``futoi_fiz_imbalance``
        - ``futoi_smart_divergence`` — yur_pos − fiz_pos
        - ``futoi_smart_imbalance_diff``
        - ``futoi_yur_pos_d1``, ``futoi_yur_pos_d5`` — Δ позиции
        - ``futoi_fiz_pos_d1``, ``futoi_fiz_pos_d5``

    На пустом / некорректном входе → пустой DataFrame.
    """
    if futoi_df is None or futoi_df.empty:
        return pd.DataFrame()
    panel = _pivot_to_daily_panel(futoi_df)
    if panel.empty:
        logger.warning("FUTOI: pivot produced empty panel; cols=%s",
                       list(futoi_df.columns))
        return pd.DataFrame()
    out = pd.DataFrame(index=panel.index)
    for group in _CLGROUPS:
        _add_group_features(out, panel, group)
    _add_smart_divergence(out)
    return out


def align_to_bar_grid(
    futoi_features: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """ffill daily-FUTOI features на 5-min bar-grid.

    Тот же подход, что у HI2: daily → reindex method='ffill'. На утро T
    фичи отражают closing-positioning T-1 (без look-ahead leakage).
    """
    out = pd.DataFrame(index=target_index)
    if futoi_features is None or futoi_features.empty:
        return out
    aligned = futoi_features.reindex(target_index, method="ffill")
    for col in aligned.columns:
        out[col] = aligned[col].astype(float).fillna(0.0)
    return out


__all__ = [
    "FUTURES_TO_SPOT",
    "align_to_bar_grid",
    "build_futoi_features",
]
