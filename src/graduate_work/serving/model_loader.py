"""Загрузка артефакт-пакета обученной модели."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from ..features import StandardScaler
from ..model import build_model
from .artifact import ARTIFACT_NAME, META_NAME, SCALER_NAME, ModelMeta

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Готовая к инференсу модель + всё, что нужно для предобработки."""

    model: torch.nn.Module
    scaler: StandardScaler
    meta: ModelMeta
    device: torch.device


def load_artifact(checkpoint_dir: Path, *, device: str | None = None) -> LoadedModel:
    """Прочитать веса, скейлер и метаинформацию из ``checkpoint_dir``."""
    weights_path = checkpoint_dir / ARTIFACT_NAME
    scaler_path = checkpoint_dir / SCALER_NAME
    meta_path = checkpoint_dir / META_NAME

    for path in (weights_path, scaler_path, meta_path):
        if not path.exists():
            msg = (
                f"Artifact file not found: {path}. "
                "Run the training notebook first to produce checkpoints/."
            )
            raise FileNotFoundError(msg)

    with meta_path.open(encoding="utf-8") as f:
        meta = ModelMeta.from_json(json.load(f))
    with scaler_path.open(encoding="utf-8") as f:
        scaler = StandardScaler.from_dict(json.load(f))

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model = build_model(
        input_dim=meta.num_features,
        num_horizons=meta.num_horizons,
        cfg=meta.model_cfg(),
    )
    state = torch.load(weights_path, map_location=target_device)
    model.load_state_dict(state)
    model.to(target_device)
    model.eval()

    logger.info(
        "Loaded model artifact: %d features, %d horizons, trained at %s",
        meta.num_features, meta.num_horizons, meta.training_date,
    )
    return LoadedModel(model=model, scaler=scaler, meta=meta, device=target_device)
