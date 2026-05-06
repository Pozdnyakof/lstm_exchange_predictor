"""Фоновый scheduler: периодически обновляет кэш предсказаний."""

from __future__ import annotations

import logging
import threading
import time

from .inference_service import InferenceService

logger = logging.getLogger(__name__)


class RefreshScheduler:
    """Простой пер-thread loop, без сторонних зависимостей.

    Реализован руками вместо APScheduler, чтобы не утяжелять стек:
    единственная задача - раз в N секунд звать ``service.predict_all()``.
    """

    def __init__(self, service: InferenceService, interval_sec: int) -> None:
        self.service = service
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="refresh-scheduler", daemon=True)
        self._thread.start()
        logger.info("Refresh scheduler started (interval=%ds)", self.interval_sec)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Refresh scheduler stopped")

    def _run(self) -> None:
        # Первое обновление сразу при старте, чтобы UI имел данные.
        self._safe_refresh()
        while not self._stop.wait(self.interval_sec):
            self._safe_refresh()

    def _safe_refresh(self) -> None:
        try:
            forecasts = self.service.predict_all(force_refresh=True)
            logger.info("Refreshed %d ticker forecasts", len(forecasts))
        except Exception:  # noqa: BLE001
            logger.exception("Background refresh failed")
