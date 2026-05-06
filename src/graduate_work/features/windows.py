"""Нарезка скользящих окон на трёхмерные тензоры (B, T, F)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сформировать массивы X, y и временные метки.

    Возвращает:
        X     - (N, window, len(feature_cols))
        y     - (N, len(target_cols))
        idx   - (N,) numpy datetime64 - метка ПОСЛЕДНЕГО шага окна
                (момент, в который мы делаем прогноз).

    Окно с любым NaN в признаках или таргетах отбрасывается. Это
    обеспечивает корректную работу при первых n строках, где скользящие
    индикаторы ещё не определены, и в хвосте, где будущая доходность не
    наблюдается.
    """
    if df.empty:
        empty_x = np.zeros((0, window, len(feature_cols)), dtype=np.float32)
        empty_y = np.zeros((0, len(target_cols)), dtype=np.float32)
        return empty_x, empty_y, np.empty((0,), dtype="datetime64[ns]")

    df = df.sort_index()
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)
    timestamps = df.index.to_numpy()

    n = feats.shape[0]
    if n < window:
        empty_x = np.zeros((0, window, len(feature_cols)), dtype=np.float32)
        empty_y = np.zeros((0, len(target_cols)), dtype=np.float32)
        return empty_x, empty_y, np.empty((0,), dtype=timestamps.dtype)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.datetime64] = []
    for end in range(window - 1, n):
        start = end - window + 1
        x_win = feats[start : end + 1]
        y_row = targets[end]
        if np.isnan(x_win).any() or np.isnan(y_row).any():
            continue
        xs.append(x_win)
        ys.append(y_row)
        ts.append(timestamps[end])

    if not xs:
        empty_x = np.zeros((0, window, len(feature_cols)), dtype=np.float32)
        empty_y = np.zeros((0, len(target_cols)), dtype=np.float32)
        return empty_x, empty_y, np.empty((0,), dtype=timestamps.dtype)

    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
        np.array(ts, dtype=timestamps.dtype),
    )
