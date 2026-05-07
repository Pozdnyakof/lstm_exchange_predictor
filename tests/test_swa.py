"""Тесты SWA-режима в Trainer."""

from __future__ import annotations

import numpy as np
import torch

from graduate_work.config import ModelConfig, TrainingConfig
from graduate_work.model import ConvLstmRegressor
from graduate_work.training import Trainer, set_seed


def _toy_arrays(n: int = 256, t: int = 16, f: int = 4, h: int = 2) -> dict:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, t, f)).astype(np.float32)
    y = rng.standard_normal((n, h)).astype(np.float32) * 0.01
    return {"x": x, "y": y}


def _make_model(input_dim: int, num_horizons: int) -> ConvLstmRegressor:
    cfg = ModelConfig(
        conv_channels=8, conv_kernel=3,
        lstm_hidden=16, lstm_layers=1, fc_hidden=16, dropout=0.2,
        use_revin=False,
    )
    return ConvLstmRegressor(input_dim, num_horizons, cfg)


def test_swa_trains_without_error() -> None:
    set_seed(42)
    train_arrays = _toy_arrays(n=128)
    val_arrays = _toy_arrays(n=64)
    model = _make_model(4, 2)
    cfg = TrainingConfig(
        batch_size=32, epochs=4, learning_rate=1e-3,
        weight_decay=0.0, early_stopping_patience=10,
        use_swa=True, swa_start_frac=0.5, swa_lr=1e-4,
    )
    trainer = Trainer(model, cfg, device="cpu")
    history = trainer.fit(train_arrays, val_arrays)
    # 4 эпохи прошли без падений.
    assert len(history.train_loss) == 4
    assert all(np.isfinite(history.train_loss))


def test_swa_disabled_path() -> None:
    set_seed(42)
    train_arrays = _toy_arrays(n=128)
    val_arrays = _toy_arrays(n=64)
    model = _make_model(4, 2)
    cfg = TrainingConfig(
        batch_size=32, epochs=3, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False,
    )
    trainer = Trainer(model, cfg, device="cpu")
    history = trainer.fit(train_arrays, val_arrays)
    assert len(history.train_loss) == 3


def test_swa_changes_weights_compared_to_baseline() -> None:
    """SWA-усреднённые веса должны отличаться от последнего checkpoint'a."""
    set_seed(42)
    train_arrays = _toy_arrays(n=128)
    val_arrays = _toy_arrays(n=64)

    # Baseline без SWA.
    set_seed(42)
    model_a = _make_model(4, 2)
    cfg_a = TrainingConfig(
        batch_size=32, epochs=4, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False,
    )
    Trainer(model_a, cfg_a, device="cpu").fit(train_arrays, val_arrays)

    # Тот же seed, но с SWA.
    set_seed(42)
    model_b = _make_model(4, 2)
    cfg_b = TrainingConfig(
        batch_size=32, epochs=4, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=True, swa_start_frac=0.5, swa_lr=1e-4,
    )
    Trainer(model_b, cfg_b, device="cpu").fit(train_arrays, val_arrays)

    # Веса должны различаться (хотя бы немного) - SWA усреднил последние эпохи.
    diffs = [
        torch.abs(a - b).max().item()
        for (_, a), (_, b) in zip(model_a.state_dict().items(), model_b.state_dict().items())
    ]
    assert max(diffs) > 1e-6
