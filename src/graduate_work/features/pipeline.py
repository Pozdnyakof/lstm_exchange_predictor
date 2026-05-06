"""Конвейер предобработки: сырые CSV → Parquet с фичами → тензоры обучения."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig, Paths
from ..data.storage import load_raw_csv, save_processed
from .scaler import StandardScaler
from .targets import normalized_log_returns, target_columns
from .technical import OHLCV_COLUMNS, add_technical_indicators
from .windows import make_sliding_windows

logger = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    """Готовые тензоры для обучения и тестирования."""

    feature_cols: list[str]
    target_cols: list[str]
    scaler: StandardScaler
    train: dict[str, np.ndarray] = field(default_factory=dict)
    val: dict[str, np.ndarray] = field(default_factory=dict)
    test: dict[str, np.ndarray] = field(default_factory=dict)
    # Сырая таблица всего датасета (после фич + таргетов) - нужна
    # для бэктеста и веб-интерфейса.
    full_frame: pd.DataFrame | None = None

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)


# ----------------------------------------------------------------------
# Сборка таблицы фич
# ----------------------------------------------------------------------

def _load_ticker_csv(paths: Paths, ticker: str) -> pd.DataFrame:
    path = paths.data_raw / "moex" / f"{ticker}.csv"
    if not path.exists():
        msg = f"Raw data not found: {path}. Run scripts/01_download_data.py first."
        raise FileNotFoundError(msg)
    df = load_raw_csv(path)
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
    df.index.name = "timestamp"
    df["ticker"] = ticker
    return df.dropna(subset=["close"])


def _load_macro(paths: Paths, cfg: DataConfig) -> pd.DataFrame:
    """Подгружает доступные макро-ряды и приводит к дневной частоте.

    Если файлов нет (offline-режим), возвращает пустой DataFrame.
    """
    macro = pd.DataFrame()
    macro_dir = paths.data_raw / "macro"
    if not macro_dir.exists():
        return macro

    for code in cfg.cbr_currencies:
        f = macro_dir / f"cbr_{code.lower()}.csv"
        if f.exists():
            df = load_raw_csv(f)
            macro = df if macro.empty else macro.join(df, how="outer")

    keyrate_path = macro_dir / "cbr_keyrate.csv"
    if keyrate_path.exists():
        df = load_raw_csv(keyrate_path)
        macro = df if macro.empty else macro.join(df, how="outer")

    brent_path = macro_dir / "brent.csv"
    if brent_path.exists():
        df = load_raw_csv(brent_path)
        macro = df if macro.empty else macro.join(df, how="outer")

    if macro.empty:
        return macro
    macro = macro.sort_index().ffill()
    return macro


def _load_indexes(paths: Paths, cfg: DataConfig) -> pd.DataFrame:
    out = pd.DataFrame()
    idx_dir = paths.data_raw / "indexes"
    if not idx_dir.exists():
        return out
    for code in (cfg.base_index, *cfg.extra_indexes):
        f = idx_dir / f"{code}.csv"
        if not f.exists():
            continue
        df = load_raw_csv(f)
        if "close" not in df.columns:
            continue
        log_ret = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
        col = f"index_{code.lower()}_logret"
        log_ret.name = col
        out = log_ret.to_frame() if out.empty else out.join(log_ret, how="outer")
    if not out.empty:
        out = out.sort_index().fillna(0.0)
    return out


def build_feature_frame(cfg: DataConfig, paths: Paths) -> tuple[pd.DataFrame, list[str]]:
    """Объединить тикеры с макро/индексной частью, посчитать фичи и таргеты."""
    macro = _load_macro(paths, cfg)
    indexes = _load_indexes(paths, cfg)

    frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None

    for ticker in cfg.tickers:
        try:
            ohlcv = _load_ticker_csv(paths, ticker)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue

        feat = add_technical_indicators(ohlcv.drop(columns=["ticker"]))
        feat["ticker"] = ticker

        if not macro.empty:
            macro_aligned = macro.reindex(feat.index, method="ffill")
            feat = pd.concat([feat, macro_aligned], axis=1)
        if not indexes.empty:
            idx_aligned = indexes.reindex(feat.index, method="ffill").fillna(0.0)
            feat = pd.concat([feat, idx_aligned], axis=1)

        targets = normalized_log_returns(feat["close"], cfg.horizons)
        feat = pd.concat([feat, targets], axis=1)

        if feature_cols is None:
            ban = {"ticker", *target_columns(cfg.horizons), "open", "high", "low"}
            feature_cols = [c for c in feat.columns if c not in ban]

        frames.append(feat)

    if not frames:
        msg = "No raw ticker data found - run download script first"
        raise RuntimeError(msg)

    full = pd.concat(frames, axis=0).sort_index()
    return full, feature_cols  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Хронологическое разделение и нарезка окон
# ----------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить таблицу по уникальным датам индекса (без перемешивания)."""
    unique_dates = np.array(sorted(df.index.unique()))
    n = len(unique_dates)
    if n < 3:
        msg = "Not enough timestamps for chronological split"
        raise ValueError(msg)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]
    return (
        df.loc[df.index.isin(train_dates)].copy(),
        df.loc[df.index.isin(val_dates)].copy(),
        df.loc[df.index.isin(test_dates)].copy(),
    )


def _build_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    window: int,
) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    tickers: list[np.ndarray] = []
    for ticker, sub in df.groupby("ticker", sort=False):
        x, y, t = make_sliding_windows(sub, feature_cols, target_cols, window)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
        ts.append(t)
        tickers.append(np.full((x.shape[0],), ticker, dtype=object))
    if not xs:
        return {
            "x": np.zeros((0, window, len(feature_cols)), dtype=np.float32),
            "y": np.zeros((0, len(target_cols)), dtype=np.float32),
            "timestamp": np.empty((0,), dtype="datetime64[ns]"),
            "ticker": np.empty((0,), dtype=object),
        }
    return {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "timestamp": np.concatenate(ts, axis=0),
        "ticker": np.concatenate(tickers, axis=0),
    }


def build_dataset(
    cfg: DataConfig,
    paths: Paths,
    *,
    persist: bool = True,
) -> PreparedDataset:
    """Полный конвейер: загрузка → фичи → нормализация → окна."""
    full, feature_cols = build_feature_frame(cfg, paths)
    full = full.dropna(subset=feature_cols)  # отбросить строки с NaN признаками

    train_df, val_df, test_df = chronological_split(
        full, cfg.train_ratio, cfg.val_ratio,
    )

    scaler = StandardScaler()
    scaler.fit(train_df, feature_cols)
    train_df = scaler.transform(train_df)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)

    target_cols = target_columns(cfg.horizons)
    prepared = PreparedDataset(
        feature_cols=feature_cols,
        target_cols=target_cols,
        scaler=scaler,
        train=_build_arrays(train_df, feature_cols, target_cols, cfg.window_size),
        val=_build_arrays(val_df, feature_cols, target_cols, cfg.window_size),
        test=_build_arrays(test_df, feature_cols, target_cols, cfg.window_size),
        full_frame=full,
    )

    if persist:
        out_path: Path = paths.data_processed / "features.parquet"
        save_processed(full.reset_index(), out_path)
        logger.info("Saved processed features to %s", out_path)

    return prepared
