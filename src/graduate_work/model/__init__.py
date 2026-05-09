"""Модуль 3 (часть 1): нейросетевая архитектура."""

from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .conv_lstm import ConvLstmRegressor
from .itransformer import ITransformer
from .linear_baselines import VLinear, XLinear
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
        - ``"timexer"``       — Transformer-baseline (R-0023 / R09.M).
        - ``"itransformer"``  — iTransformer (Liu et al., ICLR 2024).
                                Inverted attention поверх variate-токенов.
        - ``"conv_lstm"``     — гибридная 1D-CNN + LSTM (исходник §2.2).
        - ``"vlinear"``       — vanilla Linear (Zeng et al. 2023).
        - ``"xlinear"``       — XLinear с поддержкой exo (arXiv:2601.09237).
        - ``"moment"``        — MOMENT-1 frozen encoder + trainable head.
                                Требует ``pip install momentfm``.
    """
    arch = cfg.architecture
    if arch == "timexer":
        return TimeXer(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "itransformer":
        return ITransformer(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    if arch == "conv_lstm":
        return ConvLstmRegressor(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    if arch == "vlinear":
        return VLinear(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "xlinear":
        return XLinear(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "moment":
        # Лениво: импорт только при запросе, чтобы отсутствие momentfm
        # не валило весь пакет.
        from .moment_classifier import MomentClassifier  # noqa: PLC0415
        return MomentClassifier(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    msg = (
        f"Неизвестная архитектура: {arch!r} (ожидается 'timexer' | "
        "'itransformer' | 'conv_lstm' | 'vlinear' | 'xlinear' | 'moment')"
    )
    raise ValueError(msg)


__all__ = [
    "ConvLstmRegressor",
    "ITransformer",
    "MonteCarloDropout",
    "RevIN",
    "TimeXer",
    "VLinear",
    "XLinear",
    "build_model",
    "set_mc_dropout",
]
