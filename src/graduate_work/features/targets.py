"""Расчёт целевой переменной - нормализованной логарифмической доходности.

Для горизонта h дней:
    target_h = ln(close[t + h] / close[t]) / h

Деление на h обеспечивает математически корректное сопоставление
инвестиционных возможностей различной длительности (§1.2 ВКР).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalized_log_returns(
    close: pd.Series,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Вернуть DataFrame с одной колонкой на каждый горизонт.

    Значения в последних h строках для горизонта h будут NaN -
    они отбрасываются при формировании обучающих окон.
    """
    out = pd.DataFrame(index=close.index)
    log_close = np.log(close.astype(float))
    for h in horizons:
        future = log_close.shift(-h) - log_close
        out[f"target_h{h}"] = future / float(h)
    return out


def target_columns(horizons: tuple[int, ...]) -> list[str]:
    return [f"target_h{h}" for h in horizons]
