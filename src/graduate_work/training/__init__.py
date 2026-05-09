"""Обучение модели и MC-инференс."""

from .concurrent_ensemble import ConcurrentDeepEnsembleTrainer
from .ensemble import (
    DeepEnsembleTrainer,
    EnsembleHistory,
    ensemble_predict,
)
from .imbsam import ImbSAMOptimizer, select_minority_subset
from .inference import mc_predict
from .losses import (
    AsymmetricLossWithLogits,
    CompositeQuantLoss,
    FocalBCEWithLogits,
    HorizonMonotoneRegularizer,
    RankICLoss,
    SharpeLoss,
    WeightedBCEWithLogits,
    build_loss_fn,
    class_balanced_pos_weight,
)
from .mixup import maybe_apply_mixup, mixup_batch
from .trainer import Trainer, TrainingHistory, set_seed

__all__ = [
    "AsymmetricLossWithLogits",
    "CompositeQuantLoss",
    "ConcurrentDeepEnsembleTrainer",
    "DeepEnsembleTrainer",
    "EnsembleHistory",
    "FocalBCEWithLogits",
    "HorizonMonotoneRegularizer",
    "ImbSAMOptimizer",
    "RankICLoss",
    "SharpeLoss",
    "Trainer",
    "TrainingHistory",
    "WeightedBCEWithLogits",
    "build_loss_fn",
    "class_balanced_pos_weight",
    "ensemble_predict",
    "maybe_apply_mixup",
    "mc_predict",
    "mixup_batch",
    "select_minority_subset",
    "set_seed",
]
