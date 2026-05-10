"""Kelly position sizing для сигналов с двумя порогами (Primary + Meta).

В классическом Kelly (Thorp 1962, Kelly 1956) оптимальная доля капитала
``f* = edge / variance``. Для бинарной задачи «торговать / не торговать»
с p — вероятностью прибыли и b — payoff/loss ratio:

    f* = (p · b - (1 - p)) / b

В нашей задаче нет чистой p — есть Primary (направление) и Meta
(уверенность). Используем простую proxy:

    edge_primary = max(primary - kelly_primary_floor, 0)
    edge_meta    = max(meta    - kelly_meta_floor,    0)   # или 1.0 если meta нет
    raw_kelly    = edge_primary * edge_meta * scale_to_full_kelly
    fraction     = clip(raw_kelly * kelly_scale, 0, max_position_size_fraction)

где ``scale_to_full_kelly`` = 1 / ((1 - floor)^2) — нормировка так, что при
максимальных primary=meta=1.0 raw_kelly=1.0 (полный Kelly до scale-фактора).
``kelly_scale`` — fractional Kelly (обычно 0.25–0.5; полный Kelly слишком
агрессивен и допускает огромные drawdown'ы).

Эта функция НЕ обучает ничего — это чистая трансформация сигналов в
``size_fraction`` колонку, которую затем читает backtest-engine при
``cfg.sizing_mode = "signal_kelly"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_kelly_size(
    signals: pd.DataFrame,
    *,
    primary_col: str = "mean",
    meta_col: str | None = "meta",
    kelly_scale: float = 0.5,
    kelly_primary_floor: float = 0.50,
    kelly_meta_floor: float = 0.50,
    max_position_size_fraction: float = 0.20,
) -> pd.DataFrame:
    """Добавить колонку ``size_fraction`` к ``signals``.

    ``signals`` ожидается в формате SignalGenerator: long-form с
    колонками ``timestamp, ticker, horizon, mean, action, ...``.
    ``primary_col`` — имя колонки с Primary-вероятностью (обычно "mean").
    ``meta_col``    — имя колонки с Meta-вероятностью (если не None);
                      если колонки нет — используется только Primary edge.

    Возвращает копию ``signals`` с колонкой ``size_fraction`` ∈ [0, max_frac].
    """
    out = signals.copy()
    primary = out[primary_col].astype(float).clip(lower=0.0, upper=1.0)
    edge_primary = (primary - kelly_primary_floor).clip(lower=0.0)
    norm_primary = max(1e-9, 1.0 - kelly_primary_floor)
    if meta_col is not None and meta_col in out.columns:
        meta = out[meta_col].astype(float).clip(lower=0.0, upper=1.0)
        edge_meta = (meta - kelly_meta_floor).clip(lower=0.0)
        norm_meta = max(1e-9, 1.0 - kelly_meta_floor)
        raw = (edge_primary / norm_primary) * (edge_meta / norm_meta)
    else:
        raw = edge_primary / norm_primary
    sized = (raw * kelly_scale).clip(
        lower=0.0, upper=max_position_size_fraction,
    )
    out["size_fraction"] = sized.astype(np.float32)
    return out


__all__ = ["signal_kelly_size"]
