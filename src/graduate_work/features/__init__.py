"""Модуль 2: трансформация сырых данных в признаковое пространство."""

from .pipeline import (
    PreparedDataset,
    build_dataset,
    chronological_split,
)
from .scaler import StandardScaler
from .targets import normalized_log_returns
from .technical import add_technical_indicators
from .windows import make_sliding_windows

__all__ = [
    "PreparedDataset",
    "StandardScaler",
    "add_technical_indicators",
    "build_dataset",
    "chronological_split",
    "make_sliding_windows",
    "normalized_log_returns",
]
