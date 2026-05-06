"""Обучение модели и MC-инференс."""

from .inference import mc_predict
from .trainer import Trainer, TrainingHistory, set_seed

__all__ = ["Trainer", "TrainingHistory", "mc_predict", "set_seed"]
