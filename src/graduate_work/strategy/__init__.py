"""Модуль 3 (часть 2): преобразование MC-прогнозов в торговые сигналы."""

from .adaptive_conformal import (
    ACIState,
    AdaptiveConformalPredictor,
    DtACIPredictor,
    aci_signals_to_actions,
)
from .calibration import (
    CalibratedThreshold,
    attach_actual_targets,
    attach_lr_targets,
    bayes_threshold,
    calibrate_bayes_threshold,
    calibrate_min_expected_return,
    estimate_gain_from_lr,
    extract_lr_array,
)
from .conformal import ConformalCalibration, ConformalSignalGenerator
from .consensus import (
    ConsensusThresholds,
    apply_consensus_thresholds,
    build_consensus_frame,
    consensus_summary,
)
from .signals import SignalGenerator, build_predictions_frame
from .threshold_strategies import (
    apply_isotonic_calibration,
    fit_isotonic_per_horizon,
    max_pnl_threshold,
    signals_argmax_threshold,
    signals_per_horizon_threshold,
    top_k_threshold,
)

__all__ = [
    "ACIState",
    "AdaptiveConformalPredictor",
    "CalibratedThreshold",
    "ConformalCalibration",
    "ConformalSignalGenerator",
    "ConsensusThresholds",
    "DtACIPredictor",
    "SignalGenerator",
    "aci_signals_to_actions",
    "apply_consensus_thresholds",
    "apply_isotonic_calibration",
    "attach_actual_targets",
    "attach_lr_targets",
    "bayes_threshold",
    "build_consensus_frame",
    "build_predictions_frame",
    "calibrate_bayes_threshold",
    "calibrate_min_expected_return",
    "consensus_summary",
    "estimate_gain_from_lr",
    "extract_lr_array",
    "fit_isotonic_per_horizon",
    "max_pnl_threshold",
    "signals_argmax_threshold",
    "signals_per_horizon_threshold",
    "top_k_threshold",
]
