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
    history: pd.DataFrame      # последние 60 строк OHLCV для UI-графика


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
        """Берём горизонт «сегодня - last_required_days».

        Окно в режиме обучения - 30 дней; индикаторы используют
        дополнительный буфер; добавляем live_buffer_days как страховку.
        """
        today = pd.Timestamp.now(tz="UTC").normalize()
        required = self.loaded.meta.window_size + self.serving_cfg.live_buffer_days + 50
        start = (today - pd.Timedelta(days=required * 2)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        return start, end

    def _fetch_and_build(self, ticker: str) -> LiveWindow | None:
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

        # Строим технические индикаторы (тот же конвейер, что при обучении).
        feat = add_technical_indicators(raw.drop(columns=[c for c in ["ticker"] if c in raw.columns]))
        meta = self.loaded.meta

        # Нормируем загруженным скейлером.
        missing = [c for c in meta.feature_cols if c not in feat.columns]
        if missing:
            # Колонки макро/индексов в live-режиме недоступны: заполняем 0
            # (после стандартизации это означает «средний наблюдённый уровень»).
            for col in missing:
                feat[col] = 0.0
        feat = feat[meta.feature_cols].copy()
        feat = self.loaded.scaler.transform(feat)
        feat = feat.dropna()

        # В live-режиме таргет неизвестен - подставляем фиктивные нули,
        # просто чтобы переиспользовать make_sliding_windows.
        for col in meta.target_cols:
            feat[col] = 0.0

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
