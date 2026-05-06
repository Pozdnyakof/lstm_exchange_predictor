"""Тесты сети ConvLstmRegressor и MC Dropout."""

from __future__ import annotations

import numpy as np
import torch

from graduate_work.config import ModelConfig
from graduate_work.model import ConvLstmRegressor, set_mc_dropout
from graduate_work.training import mc_predict


def _make_model(input_dim: int = 8, horizons: int = 4) -> ConvLstmRegressor:
    cfg = ModelConfig(conv_channels=8, conv_kernel=3, lstm_hidden=16, lstm_layers=1, fc_hidden=16, dropout=0.5)
    return ConvLstmRegressor(input_dim, horizons, cfg)


def test_forward_shape() -> None:
    model = _make_model()
    x = torch.randn(5, 12, 8)
    out = model(x)
    assert out.shape == (5, 4)


def test_mc_dropout_creates_variability() -> None:
    model = _make_model()
    x = np.random.randn(10, 12, 8).astype(np.float32)
    mean, std = mc_predict(model, x, mc_passes=20, batch_size=4, device="cpu")
    assert mean.shape == (10, 4)
    assert std.shape == (10, 4)
    # При обученном со случайными весами Dropout=0.5 std должен быть > 0.
    assert (std > 0).any()


def test_set_mc_dropout_toggle() -> None:
    model = _make_model()
    set_mc_dropout(model, True)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert all(flags)
    set_mc_dropout(model, False)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert not any(flags)
