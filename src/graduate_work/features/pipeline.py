"""Конвейер предобработки: сырые CSV → Parquet с фичами → тензоры обучения."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import DataConfig, Paths, TradingConfig
from ..data.resample import resample_ohlcv
from ..data.storage import load_raw_csv, save_processed
from .advanced import add_advanced_indicators
from .scaler import StandardScaler
from .targets import (
    cost_aware_classification_labels,
    lr_columns,
    normalized_log_returns,
    target_columns,
)
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

def _load_ticker_csv(paths: Paths, ticker: str, cfg: DataConfig) -> pd.DataFrame:
    """Загрузить сырой 1-мин CSV, ресэмплить до cfg.bar_minutes,
    отфильтровать по торговой сессии MOEX."""
    path = paths.data_raw / "moex" / f"{ticker}.csv"
    if not path.exists():
        msg = f"Raw data not found: {path}. Run scripts/01_download_data.py first."
        raise FileNotFoundError(msg)
    df = load_raw_csv(path)
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
    df.index.name = "timestamp"
    df = resample_ohlcv(df, cfg)
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
    """Загрузить close-фрейм по всем индексам.

    Возвращает DataFrame с колонками `index_<code>_close` (чистая цена,
    без производных). Log-return считается ПОЗЖЕ - после выравнивания
    на сетку признаков тикера, чтобы избежать ffill'а ступенчатого
    log_return через сотни баров.
    """
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
        df = resample_ohlcv(df, cfg)
        if df.empty:
            continue
        col = f"index_{code.lower()}_close"
        close = df["close"].astype(float)
        close.name = col
        out = close.to_frame() if out.empty else out.join(close, how="outer")
    if not out.empty:
        out = out.sort_index()
    return out


def _index_log_returns(indexes: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Перевести close-фрейм индексов в log-return на сетке `target_index`.

    1) reindex(method=ffill) - выровнять цены на бары тикера;
    2) log_return = log(close[t] / close[t-1]) НА ЭТОЙ сетке;
       пропуски (между сессиями) трактуются как 0.
    """
    if indexes.empty or len(target_index) == 0:
        return pd.DataFrame(index=target_index)
    aligned = indexes.reindex(target_index, method="ffill")
    out = pd.DataFrame(index=target_index)
    for col in aligned.columns:
        # close → log-return: ровно один шаг на каждой паре баров.
        ret = np.log(aligned[col] / aligned[col].shift(1))
        # переименовываем "..._close" → "..._logret"
        ret.name = col.replace("_close", "_logret")
        out[ret.name] = ret.fillna(0.0)
    return out


def _build_targets(
    feat: pd.DataFrame,
    cfg: DataConfig,
    trading_cfg: TradingConfig | None,
) -> pd.DataFrame:
    """Построить целевые колонки по режиму.

    Regression: только `target_h{h}` = нормализованная лог-доходность.
    Classification: `target_h{h}` (бинарные сглаженные метки) +
                    `lr_h{h}` (сырая лог-доходность с костами).
    """
    if cfg.mode == "regression":
        return normalized_log_returns(feat["close"], cfg.horizons)
    # classification
    direction = "short" if cfg.swap_long_short_labels else "long"
    entry_cost = trading_cfg.commission_rate + trading_cfg.slippage_rate if trading_cfg else 0.0
    exit_cost = entry_cost
    return cost_aware_classification_labels(
        open_price=feat["open"],
        close_price=feat["close"],
        horizons=cfg.horizons,
        entry_cost=entry_cost,
        exit_cost=exit_cost,
        label_smoothing=cfg.label_smoothing,
        direction=direction,
    )


def build_feature_frame(
    cfg: DataConfig,
    paths: Paths,
    *,
    trading_cfg: TradingConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Объединить тикеры с макро/индексной частью, посчитать фичи и таргеты."""
    macro = _load_macro(paths, cfg)
    indexes = _load_indexes(paths, cfg)

    frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None

    bar = tqdm(cfg.tickers, desc="Building features", unit="ticker")
    for ticker in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(ticker)
        try:
            ohlcv = _load_ticker_csv(paths, ticker, cfg)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue

        feat = add_technical_indicators(ohlcv.drop(columns=["ticker"]))
        feat = add_advanced_indicators(feat, market="moex")
        feat["ticker"] = ticker

        if not macro.empty:
            macro_aligned = macro.reindex(feat.index, method="ffill")
            feat = pd.concat([feat, macro_aligned], axis=1)
        if not indexes.empty:
            # Сначала выравниваем цены индекса на бары тикера, потом
            # считаем log_return - чтобы избежать ffill ступенчатой
            # автокорреляции в фиче.
            idx_logret = _index_log_returns(indexes, feat.index)
            feat = pd.concat([feat, idx_logret], axis=1)

        targets = _build_targets(feat, cfg, trading_cfg)
        feat = pd.concat([feat, targets], axis=1)

        if feature_cols is None:
            ban = {
                "ticker", "open", "high", "low",
                *target_columns(cfg.horizons),
                *lr_columns(cfg.horizons),  # raw lr - не фича, нужна для калибровки
            }
            feature_cols = [c for c in feat.columns if c not in ban]

        frames.append(feat)

    if not frames:
        msg = "No raw ticker data found - run download script first"
        raise RuntimeError(msg)

    full = pd.concat(frames, axis=0).sort_index()

    # T2.1: per-ticker one-hot dummies. Один общий регрессор на нескольких
    # тикерах усредняет их поведение; dummies позволяют модели специализи-
    # роваться на каждом инструменте без обучения N отдельных моделей.
    if cfg.use_ticker_dummies:
        dummies = pd.get_dummies(full["ticker"], prefix="tid", dtype=float)
        full = pd.concat([full, dummies], axis=1)
        if feature_cols is not None:
            feature_cols = list(feature_cols) + list(dummies.columns)

    return full, feature_cols  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Хронологическое разделение и нарезка окон
# ----------------------------------------------------------------------

def _purge_tail(df: pd.DataFrame, drop_last: int) -> pd.DataFrame:
    """Удалить последние ``drop_last`` баров каждого тикера.

    Нужно для train и val, потому что target_h в этих хвостах
    подсматривает в первые ``drop_last`` баров СЛЕДУЮЩЕГО split'а.
    Без этого получается оптимистичная утечка на границе.
    """
    if drop_last <= 0 or df.empty or "ticker" not in df.columns:
        return df
    # Позиционная маска: для каждого тикера дропаем последние drop_last
    # позиций. Работает корректно при дубликатах в DatetimeIndex.
    ticker_arr = df["ticker"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for ticker in pd.unique(ticker_arr):
        positions = np.where(ticker_arr == ticker)[0]
        if len(positions) > drop_last:
            keep[positions[-drop_last:]] = False
        else:
            keep[positions] = False
    return df.iloc[keep].copy()


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    *,
    purge_horizon: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить таблицу по уникальным датам индекса (без перемешивания).

    ``purge_horizon`` - количество последних баров каждого тикера в train
    и val, которые отбрасываются, потому что их target подсматривает в
    следующий split. Передавайте ``max(horizons)``.
    """
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
    train_df = df.loc[df.index.isin(train_dates)].copy()
    val_df = df.loc[df.index.isin(val_dates)].copy()
    test_df = df.loc[df.index.isin(test_dates)].copy()
    return (
        _purge_tail(train_df, purge_horizon),
        _purge_tail(val_df, purge_horizon),
        test_df,
    )


def _build_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    window: int,
    *,
    desc: str = "windows",
) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    tickers: list[np.ndarray] = []
    groups = list(df.groupby("ticker", sort=False))
    bar = tqdm(groups, desc=desc, unit="ticker", leave=False)
    for ticker, sub in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(str(ticker))
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
    trading_cfg: TradingConfig | None = None,
) -> PreparedDataset:
    """Полный конвейер: загрузка → фичи → нормализация → окна.

    ``trading_cfg`` нужен для классификационных меток (используются
    transactions costs из него). Если None - lab-режим без костов.
    """
    full, feature_cols = build_feature_frame(cfg, paths, trading_cfg=trading_cfg)
    full = full.dropna(subset=feature_cols)  # отбросить строки с NaN признаками

    # Purge: target_h на хвосте train/val подсматривает в следующий
    # split. Дропаем последние max(horizons) баров каждого тикера.
    train_df, val_df, test_df = chronological_split(
        full, cfg.train_ratio, cfg.val_ratio,
        purge_horizon=max(cfg.horizons),
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
        train=_build_arrays(train_df, feature_cols, target_cols, cfg.window_size, desc="train windows"),
        val=_build_arrays(val_df, feature_cols, target_cols, cfg.window_size, desc="val windows"),
        test=_build_arrays(test_df, feature_cols, target_cols, cfg.window_size, desc="test windows"),
        full_frame=full,
    )

    if persist:
        out_path: Path = paths.data_processed / "features.parquet"
        save_processed(full.reset_index(), out_path)
        logger.info("Saved processed features to %s", out_path)

    return prepared
