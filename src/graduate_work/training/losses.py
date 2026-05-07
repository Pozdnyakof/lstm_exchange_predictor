"""Loss-функции для regression и classification режимов."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import DataConfig, TrainingConfig


class WeightedBCEWithLogits(nn.Module):
    """Бинарный cross-entropy на логитах, multi-target по горизонтам.

    Принимает (B, H) логиты и (B, H) сглаженные метки в [0, 1].
    Per-element BCE, потом среднее по B*H. Опциональный per-sample вес.

    ``pos_weight`` (shape (H,)) умножает вклад positives в loss и
    предотвращает prediction collapse при дисбалансе классов: без него
    модель быстро сходится к константе ≈ P(UP) и не использует вход.
    Канонический рецепт: ``pos_weight = (1 - P(UP)) / P(UP)``.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        # register_buffer чтобы тензор автоматически уезжал на тот же
        # device, что и модуль, через .to(...). None разрешён в новых
        # PyTorch (>=1.10) и трактуется как «буфера нет».
        self.register_buffer(
            "pos_weight",
            pos_weight.float() if pos_weight is not None else None,
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none",
            pos_weight=self.pos_weight,
        )
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


class FocalBCEWithLogits(nn.Module):
    """Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Полезен при дисбалансе классов или для подавления «лёгких»
    примеров (которые модель уже хорошо классифицирует).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        focal = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * focal * bce
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


class _CallableLoss(nn.Module):
    """Тонкий wrapper, принимающий опциональные ``weights`` для совместимости."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,  # noqa: ARG002 - не используется
    ) -> torch.Tensor:
        return self.base(preds, targets)


def build_loss_fn(
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    trading_cfg=None,
) -> nn.Module:
    """Выбрать loss по режиму.

    classification + bce        → WeightedBCEWithLogits
    classification + focal      → FocalBCEWithLogits(gamma, alpha)
    regression                  → HuberLoss(delta=auto)
    """
    if data_cfg.mode == "classification":
        if trading_cfg is not None and trading_cfg.loss_objective == "focal":
            return FocalBCEWithLogits(
                gamma=trading_cfg.focal_gamma,
                alpha=trading_cfg.focal_alpha,
            )
        return WeightedBCEWithLogits()
    # regression - старый Huber
    return _CallableLoss(nn.HuberLoss(reduction="mean", delta=1.0))
