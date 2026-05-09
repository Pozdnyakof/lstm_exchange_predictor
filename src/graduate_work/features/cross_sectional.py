"""Cross-sectional features: rank/z-score фичей через universe тикеров.

После Sprint 3 (LightGBM h=48 даёт Sharpe 0.98) и Sprint 1 (signal AUC
max 0.64), один из путей увеличить эффект — добавить относительные
сигналы: насколько необычен imbalance / spread / order-flow данного
тикера ОТНОСИТЕЛЬНО остальных в том же баре.

Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*,
Journal of Finance — обосновали cross-sectional ranking как один из
универсальных alpha-факторов. Avellaneda & Lee (2010), *Statistical
Arbitrage in the U.S. Equities Market* — подтвердили на equities.

## Что строится

Для каждой "базовой" фичи (например, ``aps_imb_vol_bbo``):
- **rank**: per-timestamp rank внутри universe в [0, 1]
- **zscore**: per-timestamp (value - cross_mean) / cross_std
- **relative**: per-timestamp value / cross_mean (или value − cross_mean
  для центрированных фич типа imbalance)

Все три — независимые, дополняющие друг друга. Rank устойчив к outliers,
z-score чувствителен к ним, relative показывает абсолютную премию.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _safe_zscore(series: pd.Series) -> pd.Series:
    """(x - mean) / std с защитой от нулевого std."""
    mean = series.mean()
    std = series.std(ddof=0)
    if std < _EPS:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def cross_sectional_rank(
    panel: pd.DataFrame,
    *,
    method: str = "average",
) -> pd.DataFrame:
    """Per-row rank в [0, 1].

    ``panel`` — wide DataFrame: index=timestamp, columns=tickers.
    Возвращает DataFrame того же shape с rank каждого тикера ВНУТРИ
    timestamp-row (timestamp = индекс), нормированным в [0, 1].

    Если в строке только 1 не-NaN значение — rank=0.5 (нейтрал).
    """
    if panel.empty or panel.shape[1] < 2:
        return panel * 0.0
    ranks = panel.rank(axis=1, method=method, na_option="keep")
    counts = panel.notna().sum(axis=1)
    # Нормировка: rank / max_rank → [1/N, 1]. Сдвинем в [0, 1] через (rank-1)/(N-1).
    normalized = (ranks.sub(1, axis=0)).div(
        counts.sub(1, axis=0).replace(0, np.nan), axis=0,
    )
    # Пары/одиночки → 0.5 (нейтрал). Маска (N,) → (N, ncols) broadcast'ом.
    mask_df = pd.DataFrame(
        np.broadcast_to(
            (counts >= 2).to_numpy()[:, None], normalized.shape,
        ),
        index=normalized.index, columns=normalized.columns,
    )
    normalized = normalized.where(mask_df, 0.5)
    return normalized.fillna(0.5)


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-row z-score: (value - cross_mean) / cross_std.

    На строках с <2 valid → 0.0 (нет дисперсии для z-нормализации).
    """
    if panel.empty or panel.shape[1] < 2:
        return panel * 0.0
    return panel.apply(_safe_zscore, axis=1).fillna(0.0)


def cross_sectional_relative(
    panel: pd.DataFrame,
    *,
    mode: str = "ratio",
) -> pd.DataFrame:
    """Per-row relative: value vs cross_mean.

    ``mode='ratio'``    → x / mean (для положительных величин: spread, vol)
    ``mode='diff'``     → x - mean (для центрированных: imbalance ∈ [-1, 1])
    """
    if panel.empty or panel.shape[1] < 2:
        return panel * 0.0
    cross_mean = panel.mean(axis=1)
    if mode == "ratio":
        return panel.div(cross_mean.replace(0, np.nan), axis=0).fillna(1.0) - 1.0
    if mode == "diff":
        return panel.sub(cross_mean, axis=0).fillna(0.0)
    msg = f"mode must be 'ratio' or 'diff', got {mode!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Конструктор фич: long-form per-ticker → wide-panel → cross-sectional
# ---------------------------------------------------------------------------

def stack_panel(
    per_ticker: dict[str, pd.DataFrame],
    feature_col: str,
) -> pd.DataFrame:
    """Собрать (timestamp × ticker) панель из per-ticker DataFrame'ов.

    ``per_ticker``: {ticker: DataFrame} с DatetimeIndex и колонкой
    ``feature_col``. Возвращает wide DataFrame: index=timestamp,
    columns=tickers.
    """
    if not per_ticker:
        return pd.DataFrame()
    series = {}
    for ticker, df in per_ticker.items():
        if df.empty or feature_col not in df.columns:
            continue
        series[ticker] = df[feature_col].astype(float)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def _attach_cross_block(
    out: dict[str, pd.DataFrame],
    block: pd.DataFrame,
    suffix: str,
    fill_value: float,
) -> None:
    """Записать колонки одного cross-sectional блока в per-ticker DataFrame'ы."""
    for ticker in block.columns:
        if ticker not in out:
            continue
        col_name = f"{suffix}"
        out[ticker][col_name] = (
            block[ticker].reindex(out[ticker].index).fillna(fill_value)
        )


def _process_one_feature(
    feat: str,
    panel: pd.DataFrame,
    out: dict[str, pd.DataFrame],
    *,
    rank: bool, zscore: bool, relative_mode: str | None,
) -> None:
    """Посчитать rank/zscore/relative блоки для одной базовой фичи и записать."""
    if rank:
        _attach_cross_block(
            out, cross_sectional_rank(panel),
            suffix=f"{feat}_xrank", fill_value=0.5,
        )
    if zscore:
        _attach_cross_block(
            out, cross_sectional_zscore(panel),
            suffix=f"{feat}_xzscore", fill_value=0.0,
        )
    if relative_mode is not None:
        _attach_cross_block(
            out, cross_sectional_relative(panel, mode=relative_mode),
            suffix=f"{feat}_xrel", fill_value=0.0,
        )


def add_cross_sectional_features(
    per_ticker: dict[str, pd.DataFrame],
    base_features: list[str],
    *,
    rank: bool = True,
    zscore: bool = True,
    relative_mode: str | None = "ratio",
) -> dict[str, pd.DataFrame]:
    """Добавить cross-sectional колонки к каждому per-ticker DataFrame.

    Для каждой фичи в ``base_features``:
        - ``{feat}_xrank`` — cross-sectional rank в [0, 1]
        - ``{feat}_xzscore`` — z-score по universe
        - ``{feat}_xrel`` — относительное значение (если ``relative_mode``)

    Универсе = все тикеры в ``per_ticker``. Чем больше тикеров — тем
    выше качество cross-sectional сигнала. На 2-3 тикерах эффект слабый,
    на 10-15 — заметный.

    Возвращает копии ``per_ticker`` с добавленными колонками.
    """
    out: dict[str, pd.DataFrame] = {
        ticker: df.copy() for ticker, df in per_ticker.items()
    }
    for feat in base_features:
        panel = stack_panel(per_ticker, feat)
        if panel.empty:
            logger.warning("Cross-sectional: feature %s not found", feat)
            continue
        _process_one_feature(
            feat, panel, out,
            rank=rank, zscore=zscore, relative_mode=relative_mode,
        )
    return out


__all__ = [
    "add_cross_sectional_features",
    "cross_sectional_rank",
    "cross_sectional_relative",
    "cross_sectional_zscore",
    "stack_panel",
]
