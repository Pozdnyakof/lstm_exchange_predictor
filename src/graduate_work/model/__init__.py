"""Модуль 3 (часть 1): нейросетевая архитектура."""

from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .conv_lstm import ConvLstmRegressor
from .mc_dropout import MonteCarloDropout, set_mc_dropout
from .revin import RevIN
from .timexer import TimeXer


def build_model(
    input_dim: int,
    num_horizons: int,
    cfg: ModelConfig,
) -> nn.Module:
    """Собрать сеть согласно ``cfg.architecture``."""
    arch = cfg.architecture
    if arch == "timexer":
        return TimeXer(input_dim=input_dim, num_horizons=num_horizons, cfg=cfg)
    if arch == "conv_lstm":
        return ConvLstmRegressor(
            input_dim=input_dim, num_horizons=num_horizons, cfg=cfg,
        )
    msg = f"Неизвестная архитектура: {arch!r} (ожидается 'timexer' | 'conv_lstm')"
    raise ValueError(msg)


__all__ = [
    "ConvLstmRegressor",
    "MonteCarloDropout",
    "RevIN",
    "TimeXer",
    "build_model",
    "set_mc_dropout",
]
