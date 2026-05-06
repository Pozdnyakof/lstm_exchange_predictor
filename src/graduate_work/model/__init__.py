"""Модуль 3 (часть 1): нейросетевая архитектура."""

from .conv_lstm import ConvLstmRegressor
from .mc_dropout import MonteCarloDropout, set_mc_dropout

__all__ = ["ConvLstmRegressor", "MonteCarloDropout", "set_mc_dropout"]
