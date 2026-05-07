"""Тесты DLinear / NLinear baseline-моделей."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graduate_work.config import ModelConfig
from graduate_work.model import (
    DLinear,
    NLinear,
    build_model,
    set_mc_dropout,
)
from graduate_work.training import mc_predict


def _cfg(arch: str, **overrides) -> ModelConfig:
    base = dict(
        architecture=arch,
        linear_seq_len=12,
        linear_kernel_size=5,
        fc_hidden=8,
        dropout=0.3,
        use_revin=False,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.parametrize("arch,cls", [("dlinear", DLinear), ("nlinear", NLinear)])
def test_linear_forward_shape(arch, cls) -> None:
    cfg = _cfg(arch)
    model = cls(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["dlinear", "nlinear"])
def test_linear_with_revin_runs(arch) -> None:
    cfg = _cfg(arch, use_revin=True)
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)


def test_dlinear_rejects_even_kernel() -> None:
    cfg = _cfg("dlinear", linear_kernel_size=4)
    with pytest.raises(ValueError):
        DLinear(input_dim=6, num_horizons=4, cfg=cfg)


@pytest.mark.parametrize("arch", ["dlinear", "nlinear"])
def test_linear_mc_predict_creates_variability(arch) -> None:
    cfg = _cfg(arch)
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    x = np.random.randn(10, 12, 6).astype(np.float32)
    mean, std = mc_predict(model, x, mc_passes=20, batch_size=4, device="cpu")
    assert mean.shape == (10, 4)
    assert std.shape == (10, 4)
    # Голова содержит MonteCarloDropout — std должен быть > 0.
    assert (std > 0).any()


@pytest.mark.parametrize("arch", ["dlinear", "nlinear"])
def test_linear_set_mc_dropout_propagates(arch) -> None:
    cfg = _cfg(arch)
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    set_mc_dropout(model, True)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert flags
    assert all(flags)
    set_mc_dropout(model, False)
    flags = [m.mc_mode for m in model.modules() if hasattr(m, "mc_mode")]
    assert not any(flags)


def test_build_model_unknown_arch_raises() -> None:
    cfg = ModelConfig(architecture="not_a_thing")
    with pytest.raises(ValueError):
        build_model(input_dim=6, num_horizons=4, cfg=cfg)
