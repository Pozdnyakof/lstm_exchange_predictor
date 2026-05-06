"""Утилиты доступа к runtime-состоянию FastAPI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, Request

from ..config import default_config
from ..serving import InferenceService

REPORT_ARTIFACTS = {
    "signals": "signals.parquet",
    "predictions": "predictions.parquet",
    "prices": "prices.parquet",
    "equity": "equity.parquet",
    "trades": "trades.parquet",
    "per_ticker": "per_ticker_metrics.parquet",
    "metrics": "metrics.json",
    "random": "random_report.json",
}


def runtime_dir() -> Path:
    return default_config().paths.data_processed / "runtime"


def load_report(name: str) -> object | None:
    path = runtime_dir() / REPORT_ARTIFACTS[name]
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def get_service(request: Request) -> InferenceService:
    state = getattr(request.app.state, "runtime", {})
    service = state.get("service") if isinstance(state, dict) else None
    if service is None:
        raise HTTPException(
            503,
            "Live inference service not initialised. "
            "Train the model and place artifacts into checkpoints/.",
        )
    return service
