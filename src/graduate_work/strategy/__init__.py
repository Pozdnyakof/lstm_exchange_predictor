"""Модуль 3 (часть 2): преобразование MC-прогнозов в торговые сигналы."""

from .calibration import (
    CalibratedThreshold,
    attach_actual_targets,
    calibrate_min_expected_return,
)
from .signals import SignalGenerator, build_predictions_frame

__all__ = [
    "CalibratedThreshold",
    "SignalGenerator",
    "attach_actual_targets",
    "build_predictions_frame",
    "calibrate_min_expected_return",
]
