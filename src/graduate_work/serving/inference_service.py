"""Сервис live-инференса: MC Dropout по всем тикерам, формирование алертов."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import DataConfig, ServingConfig, TradingConfig
from ..training import mc_predict
from .live_features import LiveFeatureBuilder, LiveWindow
from .model_loader import LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class HorizonForecast:
    horizon: int
    mean: float
    std: float


@dataclass
class TickerForecast:
    ticker: str
    timestamp: str
    last_close: float
    best_horizon: int
    expected_return: float          # mean лучшего горизонта
    uncertainty: float              # std лучшего горизонта
    alert: bool
    alert_strength: float           # mean / std (signal-to-noise)
    all_horizons: list[HorizonForecast] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["all_horizons"] = [asdict(h) for h in self.all_horizons]
        return d


class InferenceService:
    """Кэшированный live-инференс для всех тикеров модели."""

    def __init__(
        self,
        loaded: LoadedModel,
        data_cfg: DataConfig,
        trading_cfg: TradingConfig,
        serving_cfg: ServingConfig,
        feature_builder: LiveFeatureBuilder | None = None,
    ) -> None:
        self.loaded = loaded
        self.data_cfg = data_cfg
        self.trading_cfg = trading_cfg
        self.serving_cfg = serving_cfg
        self.features = feature_builder or LiveFeatureBuilder(loaded, data_cfg, serving_cfg)
        self._lock = threading.Lock()
        self._cached: list[TickerForecast] = []
        self._cached_at: float = 0.0

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def predict_all(self, *, force_refresh: bool = False) -> list[TickerForecast]:
        now = time.time()
        with self._lock:
            if not force_refresh and self._cached and now - self._cached_at < self.serving_cfg.cache_ttl_sec:
                return list(self._cached)

        results: list[TickerForecast] = []
        for ticker in self.loaded.meta.tickers:
            forecast = self._forecast_one(ticker, force_refresh=force_refresh)
            if forecast is not None:
                results.append(forecast)
            time.sleep(self.serving_cfg.moex_request_pause)

        results.sort(key=lambda f: f.expected_return, reverse=True)
        with self._lock:
            self._cached = results
            self._cached_at = now
        return list(results)

    def predict_one(self, ticker: str, *, force_refresh: bool = False) -> TickerForecast | None:
        return self._forecast_one(ticker, force_refresh=force_refresh)

    def alerts(self) -> list[TickerForecast]:
        return [f for f in self.predict_all() if f.alert]

    def cached(self) -> list[TickerForecast]:
        with self._lock:
            return list(self._cached)

    def cached_at(self) -> float:
        return self._cached_at

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _forecast_one(self, ticker: str, *, force_refresh: bool) -> TickerForecast | None:
        win = self.features.get_window(ticker, force_refresh=force_refresh)
        if win is None:
            return None
        return self._run_mc(win)

    def _run_mc(self, win: LiveWindow) -> TickerForecast:
        mean, std = mc_predict(
            self.loaded.model, win.x,
            mc_passes=self.serving_cfg.refresh_interval_sec and 50 or 50,
            batch_size=1,
            device=str(self.loaded.device),
        )
        # mean shape (1, H), std shape (1, H)
        mean_v = mean[0]
        std_v = std[0]

        horizons = self.loaded.meta.horizons
        forecasts = [
            HorizonForecast(horizon=int(h), mean=float(m), std=float(s))
            for h, m, s in zip(horizons, mean_v, std_v)
        ]

        best_idx = int(np.argmax(mean_v))
        best_mean = float(mean_v[best_idx])
        best_std = float(std_v[best_idx])
        best_horizon = int(horizons[best_idx])

        alert_strength = best_mean / best_std if best_std > 1e-9 else 0.0
        alert = (
            best_mean >= self.trading_cfg.min_expected_return
            and best_std <= self.trading_cfg.max_uncertainty
            and alert_strength >= self.serving_cfg.alert_min_strength
        )

        history_records = self._history_records(win.history)

        return TickerForecast(
            ticker=win.ticker,
            timestamp=win.last_timestamp.isoformat(),
            last_close=win.last_close,
            best_horizon=best_horizon,
            expected_return=best_mean,
            uncertainty=best_std,
            alert=bool(alert),
            alert_strength=float(alert_strength),
            all_horizons=forecasts,
            history=history_records,
        )

    @staticmethod
    def _history_records(history: pd.DataFrame) -> list[dict]:
        if history.empty:
            return []
        out: list[dict] = []
        for ts, row in history.iterrows():
            out.append(
                {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "close": float(row["close"]),
                },
            )
        return out
