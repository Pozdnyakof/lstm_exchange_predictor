"""Тесты quant-loss функций: RankIC, Sharpe, Monotone, Composite."""

from __future__ import annotations

import torch

from graduate_work.training.losses import (
    CompositeQuantLoss,
    HorizonMonotoneRegularizer,
    RankICLoss,
    SharpeLoss,
)


def test_rankic_perfect_correlation_clipped() -> None:
    """Идеально согласованные ранги → IC=1 → clip(IC, -0.3, 0.3)=0.3 → loss = -0.3.

    Sprint 2: clip=0.3 предотвращает unbounded negative loss при memorize.
    Без clip'а loss уезжал до -1.0 и тянул composite в негатив.
    """
    loss_fn = RankICLoss(regularization=0.1, clip=0.3)
    pred = torch.linspace(0.0, 10.0, steps=32).unsqueeze(-1)
    target = torch.linspace(-5.0, 5.0, steps=32).unsqueeze(-1)
    loss = loss_fn(pred, target)
    assert abs(loss.item() - (-0.3)) < 1e-5


def test_rankic_anti_correlation_clipped() -> None:
    """Обратные ранги → IC≈-1 → clip(IC, -0.3, 0.3)=-0.3 → loss = +0.3."""
    loss_fn = RankICLoss(regularization=0.1, clip=0.3)
    pred = torch.linspace(0.0, 10.0, steps=32).unsqueeze(-1)
    target = torch.linspace(10.0, 0.0, steps=32).unsqueeze(-1)
    loss = loss_fn(pred, target)
    assert abs(loss.item() - 0.3) < 1e-5


def test_rankic_clip_validates_input() -> None:
    """clip ∉ (0, 1] запрещён."""
    for bad in (0.0, -0.1, 1.5):
        try:
            RankICLoss(clip=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on clip={bad}")


def test_rankic_gradient_flows_in_unclipped_zone() -> None:
    """Градиент по pred ненулевой когда IC не упёрся в clip (случайные данные).

    На randn(256, 2) корреляция близка к 0, далеко от clip=0.3 → grad есть.
    """
    torch.manual_seed(0)
    loss_fn = RankICLoss(regularization=1.0, clip=0.3)
    pred = torch.randn(256, 2, requires_grad=True)
    target = torch.randn(256, 2)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0


def test_rankic_zero_gradient_in_clip_zone() -> None:
    """Когда IC застряла в clip-зоне, gradient зануляется — это ИНТЕНЦИЯ
    Sprint 2: запретить дальнейшее ускользание loss в негатив.
    """
    # Идеальное anti-corr → IC=-1 → clipped в -clip → нет gradient'а.
    # Construct as leaf tensor (unsqueeze(...) returns non-leaf, поэтому
    # сначала клонируем + requires_grad_ — это даёт настоящий leaf).
    loss_fn = RankICLoss(regularization=0.1, clip=0.3)
    pred = (
        torch.linspace(0.0, 10.0, steps=32).unsqueeze(-1)
        .clone().requires_grad_(True)
    )
    target = torch.linspace(10.0, 0.0, steps=32).unsqueeze(-1)
    loss = loss_fn(pred, target)
    loss.backward()
    # Clamp blocks gradient за пределами зоны → околонулевой.
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() < 1e-3


def test_sharpe_positive_signal_negative_loss() -> None:
    """Если signal коррелирует с lr, Sharpe>0 → loss<0."""
    loss_fn = SharpeLoss(cost=0.0)
    # logits → sigmoid → signal: положительные logits = долгая позиция.
    logits = torch.tensor([[2.0], [-2.0], [2.0], [-2.0], [2.0], [-2.0]])
    lr = torch.tensor([[0.01], [-0.01], [0.02], [-0.02], [0.005], [-0.005]])
    loss = loss_fn(logits, lr)
    assert loss.item() < 0


def test_sharpe_cost_increases_loss() -> None:
    """Те же сигналы при ненулевых costs → Sharpe ниже → loss выше."""
    cheap = SharpeLoss(cost=0.0)
    expensive = SharpeLoss(cost=0.005)
    logits = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
    lr = torch.tensor([[0.01], [-0.01], [0.02], [-0.02]])
    assert expensive(logits, lr).item() > cheap(logits, lr).item()


def test_monotone_zero_when_increasing() -> None:
    """probs монотонно возрастают по горизонту → штраф 0."""
    reg = HorizonMonotoneRegularizer(weight=1.0)
    # logits возрастают → probs возрастают.
    logits = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    assert reg(logits).item() == 0.0


def test_monotone_positive_when_decreasing() -> None:
    """probs убывают по горизонту → штраф > 0."""
    reg = HorizonMonotoneRegularizer(weight=1.0)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    assert reg(logits).item() > 0.0


def test_monotone_no_op_for_single_horizon() -> None:
    """С одним горизонтом регуляризатор тождественно 0."""
    reg = HorizonMonotoneRegularizer(weight=1.0)
    assert reg(torch.randn(8, 1)).item() == 0.0


def test_composite_loss_full_signature() -> None:
    """Все 4 компоненты композитного loss работают и градиент течёт."""
    loss_fn = CompositeQuantLoss(
        bce_weight=1.0, rankic_weight=0.5,
        sharpe_weight=0.3, monotone_weight=0.1, cost=0.001,
    )
    logits = torch.randn(32, 4, requires_grad=True)
    target = torch.randint(0, 2, (32, 4)).float()
    lr = torch.randn(32, 4) * 0.01
    loss = loss_fn(logits, target, lr)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_composite_loss_without_lr_falls_back() -> None:
    """Без lr_target RankIC/Sharpe компоненты пропускаются — остаётся BCE+Monotone."""
    loss_fn = CompositeQuantLoss(
        bce_weight=1.0, rankic_weight=0.5,
        sharpe_weight=0.3, monotone_weight=0.1,
    )
    logits = torch.randn(16, 4)
    target = torch.randint(0, 2, (16, 4)).float()
    loss = loss_fn(logits, target, lr_target=None)
    assert torch.isfinite(loss)
