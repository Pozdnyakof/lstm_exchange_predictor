"""Модуль 3 (часть 1): нейросетевая архитектура."""

from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .conv_lstm import ConvLstmRegressor
from .linear_baselines import DLinear, NLinear
from .mc_dropout import MonteCarloDropout, set_mc_dropout
from .revin import RevIN
from .timexer import TimeXer


def build_model(
    input_dim: int,
    num_horizons: int,
    cfg: ModelConfig,
) -> nn.Module:
    """Собрать сеть согласно ``cfg.architecture``.

    Поддерживаемые значения:
        - ``"timexer"``    — Transformer-baseline (R-0023 / R09.M).
        - ``"conv_lstm"``  — гибридная 1D-CNN + LSTM (исходник §2.2).
        - ``"dlinear"``    — Decomposition-Linear (Zeng et al. 2023).
        - ``"nlinear"``    — Normalisation-Linear (Zeng et al. 2023).
        - ``"moment"``     — MOMENT-1 frozen encoder + trainable head.
                            Требует ``pip install momentfm``.
    """
    arch = cfg.architecture
    if arch == "timexer":
        return TimeXer(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "conv_lstm":
        return ConvLstmRegressor(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    if arch == "dlinear":
        return DLinear(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "nlinear":
        return NLinear(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "moment":
        # Лениво: импорт только при запросе, чтобы отсутствие momentfm
        # не валило весь пакет.
        from .moment_classifier import MomentClassifier  # noqa: PLC0415
        return MomentClassifier(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    msg = (
        f"Неизвестная архитектура: {arch!r} "
        "(ожидается 'timexer' | 'conv_lstm' | 'dlinear' | 'nlinear' | 'moment')"
    )
    raise ValueError(msg)


__all__ = [
    "ConvLstmRegressor",
    "DLinear",
    "MonteCarloDropout",
    "NLinear",
    "RevIN",
    "TimeXer",
    "build_model",
    "set_mc_dropout",
]
