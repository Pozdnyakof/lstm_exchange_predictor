"""Модуль 3 (часть 2): преобразование MC-прогнозов в торговые сигналы."""

from .calibration import (
    CalibratedThreshold,
    attach_actual_targets,
    attach_lr_targets,
    bayes_threshold,
    calibrate_bayes_threshold,
    calibrate_min_expected_return,
    estimate_gain_from_lr,
)
from .conformal import ConformalCalibration, ConformalSignalGenerator
from .signals import SignalGenerator, build_predictions_frame

__all__ = [
    "CalibratedThreshold",
    "ConformalCalibration",
    "ConformalSignalGenerator",
    "SignalGenerator",
    "attach_actual_targets",
    "attach_lr_targets",
    "bayes_threshold",
    "build_predictions_frame",
    "calibrate_bayes_threshold",
    "calibrate_min_expected_return",
    "estimate_gain_from_lr",
]
