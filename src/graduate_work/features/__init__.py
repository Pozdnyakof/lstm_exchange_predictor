"""Модуль 2: трансформация сырых данных в признаковое пространство."""

from .advanced import add_advanced_indicators
from .pipeline import (
    PreparedDataset,
    build_dataset,
    chronological_split,
)
from .scaler import StandardScaler
from .targets import (
    cost_aware_classification_labels,
    lr_columns,
    normalized_log_returns,
    target_columns,
)
from .technical import add_technical_indicators
from .windows import make_sliding_windows

__all__ = [
    "PreparedDataset",
    "StandardScaler",
    "add_advanced_indicators",
    "add_technical_indicators",
    "build_dataset",
    "chronological_split",
    "cost_aware_classification_labels",
    "lr_columns",
    "make_sliding_windows",
    "normalized_log_returns",
    "target_columns",
]
