"""Тесты iTransformer: shape, gradient flow, registry."""

from __future__ import annotations

import torch

from graduate_work.config import ModelConfig
from graduate_work.model import ITransformer, build_model


def _make_cfg(seq_len: int = 16) -> ModelConfig:
    """Маленький iTransformer для быстрого smoke-test'а."""
    return ModelConfig(
        architecture="itransformer",
        fc_hidden=16,
        dropout=0.2,
        use_revin=True,
        revin_affine=True,
        itransformer_seq_len=seq_len,
        itransformer_d_model=16,
        itransformer_n_layers=2,
        itransformer_n_heads=2,
        itransformer_d_ff=32,
        itransformer_dropout=0.2,
    )


def test_forward_shape_match() -> None:
    """T соответствует seq_len → выход (B, num_horizons)."""
    model = ITransformer(input_dim=6, num_horizons=4, cfg=_make_cfg(seq_len=16))
    x = torch.randn(8, 16, 6)
    out = model(x)
    assert out.shape == (8, 4)


def test_forward_shape_pad_when_short() -> None:
    """T < seq_len → left-pad нулями, форма выхода прежняя."""
    model = ITransformer(input_dim=6, num_horizons=4, cfg=_make_cfg(seq_len=16))
    x = torch.randn(3, 8, 6)
    out = model(x)
    assert out.shape == (3, 4)


def test_forward_shape_truncate_when_long() -> None:
    """T > seq_len → берём последние seq_len баров."""
    model = ITransformer(input_dim=6, num_horizons=4, cfg=_make_cfg(seq_len=16))
    x = torch.randn(3, 32, 6)
    out = model(x)
    assert out.shape == (3, 4)


def test_gradient_flow() -> None:
    """Градиент должен течь до variate-embedding, иначе self-attention отключён."""
    model = ITransformer(input_dim=6, num_horizons=4, cfg=_make_cfg(seq_len=16))
    x = torch.randn(4, 16, 6)
    target = torch.randn(4, 4)
    out = model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()
    # У variate-embedding linear должен быть ненулевой градиент.
    assert model.embed.proj.weight.grad is not None
    assert model.embed.proj.weight.grad.abs().sum().item() > 0


def test_input_dim_mismatch_raises() -> None:
    """Защита от ошибочной формы входа."""
    model = ITransformer(input_dim=6, num_horizons=4, cfg=_make_cfg(seq_len=16))
    bad = torch.randn(2, 16, 7)
    try:
        model(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on input_dim mismatch")


def test_build_model_registry() -> None:
    """``build_model('itransformer')`` собирает корректный класс."""
    cfg = _make_cfg()
    model = build_model(input_dim=6, num_horizons=4, cfg=cfg)
    assert isinstance(model, ITransformer)
    assert model(torch.randn(2, 16, 6)).shape == (2, 4)
