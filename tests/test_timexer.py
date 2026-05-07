"""Тесты TimeXer-baseline и factory build_model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graduate_work.config import ModelConfig
from graduate_work.model import (
    ConvLstmRegressor,
    TimeXer,
    build_model,
    set_mc_dropout,
)
from graduate_work.training import mc_predict


def _timexer_cfg(**overrides) -> ModelConfig:
    """Лёгкая конфигурация TimeXer для тестов."""
    base = dict(
        architecture="timexer",
        timexer_d_model=16,
        timexer_n_layers=1,
        timexer_n_heads=2,
        timexer_d_ff=32,
        timexer_patch_len=4,
        timexer_stride=2,
        timexer_seq_len=12,
        timexer_dropout=0.3,
        timexer_n_exo=0,
        fc_hidden=8,
        dropout=0.3,
        use_revin=False,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_timexer_forward_shape() -> None:
    cfg = _timexer_cfg()
    model = TimeXer(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)


def test_timexer_with_revin_runs() -> None:
    cfg = _timexer_cfg(use_revin=True)
    model = TimeXer(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)
    assert torch.isfinite(out).all()


def test_timexer_with_exo_channels() -> None:
    cfg = _timexer_cfg(timexer_n_exo=2)
    # input_dim=8: 6 эндогенных + 2 экзогенных.
    model = TimeXer(input_dim=8, num_horizons=4, cfg=cfg)
    x = torch.randn(4, 12, 8)
    out = model(x)
    assert out.shape == (4, 4)


def test_timexer_rejects_no_endo_channels() -> None:
    cfg = _timexer_cfg(timexer_n_exo=6)
    with pytest.raises(ValueError):
        TimeXer(input_dim=6, num_horizons=4, cfg=cfg)


def test_timexer_rejects_invalid_patch_grid() -> None:
    """seq_len, не нарезающийся ровно на патчи, должен падать."""
    cfg = _timexer_cfg(timexer_seq_len=11)  # (11-4)/2 != int.
    with pytest.raises(ValueError):
        TimeXer(input_dim=6, num_horizons=4, cfg=cfg)


def test_timexer_set_mc_dropout_propagates() -> None:
    """set_mc_dropout должен находить все MonteCarloDropout внутри TimeXer."""
    cfg = _timexer_cfg()
    model = TimeXer(input_dim=6, num_horizons=4, cfg=cfg)
    set_mc_dropout(model, True)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert flags  # есть MC-слои
    assert all(flags)
    set_mc_dropout(model, False)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert not any(flags)


def test_timexer_mc_predict_creates_variability() -> None:
    cfg = _timexer_cfg()
    model = TimeXer(input_dim=6, num_horizons=4, cfg=cfg)
    x = np.random.randn(10, 12, 6).astype(np.float32)
    mean, std = mc_predict(model, x, mc_passes=20, batch_size=4, device="cpu")
    assert mean.shape == (10, 4)
    assert std.shape == (10, 4)
    assert (std > 0).any()


def test_build_model_picks_timexer() -> None:
    cfg = _timexer_cfg()
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    assert isinstance(model, TimeXer)


def test_build_model_picks_conv_lstm() -> None:
    cfg = ModelConfig(
        architecture="conv_lstm",
        conv_channels=4, conv_kernel=3,
        lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.3,
    )
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    assert isinstance(model, ConvLstmRegressor)


def test_build_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError):
        build_model(
            input_dim=6, num_horizons=4,
            cfg=ModelConfig(architecture="not_a_real_arch"),
        )
