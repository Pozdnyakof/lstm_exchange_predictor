"""Построение live-окна признаков для одного тикера.

Качает свежие свечи через MOEX ISS, строит технические индикаторы,
нормирует загруженным скейлером, возвращает тензор (1, T, F) и
последнюю наблюдённую цену close (нужна фронтенду для построения CI).

Между вызовами хранит лёгкий in-memory кэш с TTL, чтобы не дёргать
ISS на каждый клик пользователя.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DataConfig, ServingConfig
from ..data import moex_iss
from ..data.resample import resample_ohlcv
from ..features.technical import add_technical_indicators
from ..features.windows import make_sliding_windows
from .model_loader import LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class LiveWindow:
    ticker: str
    x: np.ndarray              # (1, T, F)
    last_timestamp: pd.Timestamp
    last_close: float
    history: pd.DataFrame      # последние ~60 баров OHLCV для UI-графика


class LiveFeatureBuilder:
    """Кэширующий загрузчик последнего окна для тикера."""

    def __init__(
        self,
        loaded: LoadedModel,
        data_cfg: DataConfig,
        serving_cfg: ServingConfig,
    ) -> None:
        self.loaded = loaded
        self.data_cfg = data_cfg
        self.serving_cfg = serving_cfg
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, LiveWindow]] = {}

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def get_window(self, ticker: str, *, force_refresh: bool = False) -> LiveWindow | None:
        now = time.time()
        with self._lock:
            cached = self._cache.get(ticker)
            if cached is not None and not force_refresh:
                ts, win = cached
                if now - ts < self.serving_cfg.cache_ttl_sec:
                    return win

        win = self._fetch_and_build(ticker)
        if win is None:
            return None
        with self._lock:
            self._cache[ticker] = (now, win)
        return win

    def invalidate(self, ticker: str | None = None) -> None:
        with self._lock:
            if ticker is None:
                self._cache.clear()
            else:
                self._cache.pop(ticker, None)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _date_range(self) -> tuple[str, str]:
        """Окно для live-загрузки в календарных днях.

        В одной торговой сессии MOEX (≈8.75ч) укладывается
        ⌊525 / bar_minutes⌋ баров. Нам нужно ≥ window_size баров;
        добавляем буфер для индикаторов и live_buffer_days как страховку.
        """
        today = pd.Timestamp.now(tz="UTC").normalize()
        bars_per_session = max(1, 525 // max(self.data_cfg.bar_minutes, 1))
        required_sessions = (
            (self.loaded.meta.window_size + 100) / bars_per_session
            + self.serving_cfg.live_buffer_days
        )
        # ×2 чтобы перекрыть выходные и праздники.
        days = max(int(required_sessions * 2), 14)
        start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        return start, end

    def _fetch_raw(self, ticker: str) -> pd.DataFrame | None:
        start, end = self._date_range()
        try:
            raw = moex_iss.fetch_ticker(
                ticker,
                start=start,
                end=end,
                interval=self.data_cfg.moex_interval,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MOEX live fetch failed for %s: %s", ticker, exc)
            return None
        if raw.empty:
            logger.warning("Empty MOEX response for %s", ticker)
            return None
        raw = resample_ohlcv(raw, self.data_cfg)
        return raw if not raw.empty else None

    def _build_feature_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        meta = self.loaded.meta
        feat = add_technical_indicators(
            raw.drop(columns=[c for c in ["ticker"] if c in raw.columns]),
        )
        # Колонки макро/индексов в live-режиме недоступны: заполняем 0
        # (после стандартизации это «средний наблюдённый уровень»).
        for col in meta.feature_cols:
            if col not in feat.columns:
                feat[col] = 0.0
        feat = feat[meta.feature_cols].copy()
        feat = self.loaded.scaler.transform(feat).dropna()
        # В live таргет неизвестен - подставляем нули, чтобы
        # переиспользовать make_sliding_windows.
        for col in meta.target_cols:
            feat[col] = 0.0
        return feat

    def _fetch_and_build(self, ticker: str) -> LiveWindow | None:
        raw = self._fetch_raw(ticker)
        if raw is None:
            return None
        feat = self._build_feature_frame(raw)
        meta = self.loaded.meta
        x, _y, ts = make_sliding_windows(
            feat,
            feature_cols=meta.feature_cols,
            target_cols=meta.target_cols,
            window=meta.window_size,
        )
        if x.shape[0] == 0:
            return None

        last_x = x[-1:].astype(np.float32)
        last_ts = pd.Timestamp(ts[-1])
        last_close = float(raw["close"].iloc[-1])
        history = raw[["open", "high", "low", "close", "volume"]].tail(60).copy()

        return LiveWindow(
            ticker=ticker,
            x=last_x,
            last_timestamp=last_ts,
            last_close=last_close,
            history=history,
        )
