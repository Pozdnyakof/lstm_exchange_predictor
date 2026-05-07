"""Тесты VLinear (vanilla LTSF-Linear) и XLinear (arXiv:2601.09237)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graduate_work.config import ModelConfig
from graduate_work.model import (
    VLinear,
    XLinear,
    build_model,
    set_mc_dropout,
)
from graduate_work.training import mc_predict


def _cfg(arch: str, **overrides) -> ModelConfig:
    base = dict(
        architecture=arch,
        linear_seq_len=12,
        fc_hidden=8,
        dropout=0.3,
        use_revin=False,
        timexer_n_exo=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.parametrize("arch,cls", [("vlinear", VLinear), ("xlinear", XLinear)])
def test_linear_forward_shape(arch, cls) -> None:
    cfg = _cfg(arch)
    model = cls(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["vlinear", "xlinear"])
def test_linear_with_revin_runs(arch) -> None:
    cfg = _cfg(arch, use_revin=True)
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    x = torch.randn(8, 12, 6)
    out = model(x)
    assert out.shape == (8, 4)


def test_xlinear_with_exo_channels_engages_cross_mlp() -> None:
    """С n_exo>0 у XLinear должна появиться cross_mlp ветвь и forward работать."""
    cfg = _cfg("xlinear", timexer_n_exo=2)
    model = XLinear(input_dim=8, num_horizons=4, cfg=cfg)
    assert model.exo_temporal is not None
    assert model.cross_mlp is not None
    x = torch.randn(4, 12, 8)
    out = model(x)
    assert out.shape == (4, 4)


def test_xlinear_no_exo_skips_cross_mlp() -> None:
    cfg = _cfg("xlinear", timexer_n_exo=0)
    model = XLinear(input_dim=6, num_horizons=4, cfg=cfg)
    assert model.exo_temporal is None
    assert model.cross_mlp is None


def test_xlinear_rejects_no_endo_channels() -> None:
    cfg = _cfg("xlinear", timexer_n_exo=6)
    with pytest.raises(ValueError):
        XLinear(input_dim=6, num_horizons=4, cfg=cfg)


@pytest.mark.parametrize("arch", ["vlinear", "xlinear"])
def test_linear_mc_predict_creates_variability(arch) -> None:
    cfg = _cfg(arch)
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    x = np.random.randn(10, 12, 6).astype(np.float32)
    mean, std = mc_predict(model, x, mc_passes=20, batch_size=4, device="cpu")
    assert mean.shape == (10, 4)
    assert std.shape == (10, 4)
    # Голова содержит MonteCarloDropout — std должен быть > 0.
    assert (std > 0).any()


@pytest.mark.parametrize("arch", ["vlinear", "xlinear"])
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


def test_build_model_dispatch() -> None:
    """factory корректно создаёт VLinear и XLinear по строке architecture."""
    assert isinstance(build_model(6, 4, _cfg("vlinear")), VLinear)
    assert isinstance(build_model(6, 4, _cfg("xlinear")), XLinear)


def test_build_model_unknown_arch_raises() -> None:
    cfg = ModelConfig(architecture="not_a_thing")
    with pytest.raises(ValueError):
        build_model(input_dim=6, num_horizons=4, cfg=cfg)
