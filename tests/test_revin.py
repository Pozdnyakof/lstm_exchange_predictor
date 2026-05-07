"""Тесты RevIN-слоя и его интеграции в ConvLstmRegressor."""

from __future__ import annotations

import torch

from graduate_work.config import ModelConfig
from graduate_work.model import ConvLstmRegressor, RevIN


def test_revin_zero_mean_unit_std_after_normalize() -> None:
    layer = RevIN(num_features=4, affine=False)
    x = torch.randn(8, 32, 4) * 5.0 + 100.0  # сильно сдвинутый
    y = layer(x)
    # Per-instance per-feature mean ≈ 0, std ≈ 1.
    assert torch.allclose(y.mean(dim=1), torch.zeros(8, 4), atol=1e-5)
    assert torch.allclose(y.std(dim=1, unbiased=False), torch.ones(8, 4), atol=1e-2)


def test_revin_affine_parameters_are_learnable() -> None:
    layer = RevIN(num_features=4, affine=True)
    params = list(layer.parameters())
    assert len(params) == 2  # gamma, beta
    assert all(p.requires_grad for p in params)


def test_conv_lstm_with_revin_forward() -> None:
    cfg = ModelConfig(
        conv_channels=8, conv_kernel=3,
        lstm_hidden=16, lstm_layers=1, fc_hidden=16, dropout=0.3,
        use_revin=True,
    )
    model = ConvLstmRegressor(input_dim=6, num_horizons=4, cfg=cfg)
    assert model.revin is not None
    x = torch.randn(4, 20, 6) * 10.0
    out = model(x)
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()


def test_conv_lstm_without_revin_forward() -> None:
    cfg = ModelConfig(
        conv_channels=8, conv_kernel=3,
        lstm_hidden=16, lstm_layers=1, fc_hidden=16, dropout=0.3,
        use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=6, num_horizons=4, cfg=cfg)
    assert model.revin is None
    x = torch.randn(4, 20, 6)
    out = model(x)
    assert out.shape == (4, 4)
