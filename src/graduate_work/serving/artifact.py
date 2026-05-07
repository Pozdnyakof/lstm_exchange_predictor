"""Сериализация и десериализация артефакт-пакета модели.

Артефакт состоит из трёх файлов в `checkpoints/`:
    model_artifact.pt   - state_dict обученной сети (PyTorch .pt)
    scaler.json         - параметры StandardScaler (mean, std, columns)
    meta.json           - архитектура, фичи, горизонты, дата обучения

Пакет самодостаточен: бэкенд может загрузить модель, не имея доступа
к обучающему датасету.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from ..config import ModelConfig
from ..features import StandardScaler

ARTIFACT_NAME = "model_artifact.pt"
SCALER_NAME = "scaler.json"
META_NAME = "meta.json"


@dataclass
class ModelMeta:
    feature_cols: list[str]
    target_cols: list[str]
    horizons: list[int]
    window_size: int
    num_features: int
    num_horizons: int
    model_config: dict
    training_date: str
    tickers: list[str]
    # Режим обучения: "regression" | "classification".
    # Старые артефакты без этого поля считаются "regression" по умолчанию.
    mode: str = "regression"

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, payload: dict) -> "ModelMeta":
        return cls(**payload)

    def model_cfg(self) -> ModelConfig:
        return ModelConfig(**self.model_config)


def save_artifact(
    model: nn.Module,
    scaler: StandardScaler,
    meta: ModelMeta,
    checkpoint_dir: Path,
) -> dict[str, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / ARTIFACT_NAME
    scaler_path = checkpoint_dir / SCALER_NAME
    meta_path = checkpoint_dir / META_NAME

    torch.save(model.state_dict(), weights_path)
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta.to_json(), f, ensure_ascii=False, indent=2)

    return {"weights": weights_path, "scaler": scaler_path, "meta": meta_path}


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
