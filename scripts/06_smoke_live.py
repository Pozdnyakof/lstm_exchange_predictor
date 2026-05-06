"""Однократный вызов InferenceService без поднятия HTTP-сервера.

Полезно для отладки: проверить, что артефакт-пакет корректно загружается,
MOEX ISS отвечает, и MC Dropout даёт осмысленные прогнозы для всех тикеров.
"""

from __future__ import annotations

import json
import logging

import _bootstrap  # noqa: F401

from graduate_work.config import default_config
from graduate_work.serving import (
    InferenceService,
    LiveFeatureBuilder,
    load_artifact,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    loaded = load_artifact(cfg.paths.checkpoints, device="cpu")
    builder = LiveFeatureBuilder(loaded, cfg.data, cfg.serving)
    service = InferenceService(loaded, cfg.data, cfg.trading, cfg.serving, builder)

    forecasts = service.predict_all(force_refresh=True)
    logging.info("Got %d forecasts", len(forecasts))

    payload = [f.to_json() for f in forecasts]
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    n_alerts = sum(1 for f in forecasts if f.alert)
    logging.info("Active alerts: %d", n_alerts)


if __name__ == "__main__":
    main()
