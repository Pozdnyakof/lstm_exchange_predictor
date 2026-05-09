"""ImbSAM: Imbalanced Sharpness-Aware Minimization.

Zhou et al., **ICCV 2023**, [arXiv:2308.07815](https://arxiv.org/abs/2308.07815),
референс: [cool-xuan/Imbalanced_SAM](https://github.com/cool-xuan/Imbalanced_SAM).

SAM (Foret et al., ICLR 2021) находит «плоский» минимум: вместо
``min L(w)`` оптимизирует ``min max_{||ε||≤ρ} L(w+ε)``. Доказанные
формальные generalization bounds.

ImbSAM применяет sharpness-perturbation **только к minority-class
сэмплам**. Зачем — overfit при class imbalance концентрируется именно
на minority, и общий SAM «расплачивается» за этот шум на ВСЕХ примерах.
ImbSAM локализует регуляризацию.

Алгоритм (один шаг):
1. forward+backward на minority-подмножестве → grad_min
2. ε = ρ · grad_min / ||grad_min||;  w ← w + ε
3. forward+backward на ПОЛНОМ батче (с перетурбированными w) → grad_full
4. w ← w − ε  (откатываем перетурбацию)
5. optimizer.step() с grad_full

~1.5× оверхед к обычному обучению (один лишний backward на minority).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ImbSAMOptimizer:
    """Wrapper над оптимизатором с ImbSAM-step.

    Использование (внутри Trainer._step)::

        sam = ImbSAMOptimizer(optimizer, model, rho=0.05)
        loss_value = sam.step(
            loss_fn=lambda: full_batch_loss(model(x), y, lr),
            minority_loss_fn=lambda: minority_loss(model, x_min, y_min, lr_min),
        )

    Если ``minority_loss_fn`` возвращает ``None`` (нет minority в батче) —
    деградирует в обычный backward+step без перетурбации.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        rho: float = 0.05,
        eps: float = 1e-12,
    ) -> None:
        if rho <= 0:
            msg = f"rho must be > 0, got {rho}"
            raise ValueError(msg)
        self.optimizer = optimizer
        self.model = model
        self.rho = float(rho)
        self.eps = float(eps)
        self._stash: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def _grad_norm(self) -> float:
        chunks = [
            p.grad.detach().flatten()
            for p in self.model.parameters() if p.grad is not None
        ]
        if not chunks:
            return 0.0
        return float(torch.cat(chunks).norm(p=2).item())

    @torch.no_grad()
    def _stash_perturbation(self, scale: float) -> None:
        """Сохранить ε и применить w ← w + ε."""
        self._stash = {}
        for p in self.model.parameters():
            if p.grad is None:
                continue
            eps = p.grad.detach().clone() * scale
            self._stash[id(p)] = eps
            p.add_(eps)

    @torch.no_grad()
    def _undo_stashed_perturbation(self) -> None:
        """Откатить w ← w − ε."""
        if not self._stash:
            return
        for p in self.model.parameters():
            eps = self._stash.get(id(p))
            if eps is not None:
                p.sub_(eps)
        self._stash = {}

    def _compute_minority_perturbation(
        self, minority_loss_fn: Callable[[], torch.Tensor | None],
    ) -> bool:
        """Шаг 1+2: посчитать grad по minority и применить w ← w + ε.

        Возвращает True если перетурбация была применена.
        """
        self.optimizer.zero_grad(set_to_none=True)
        minority_loss = minority_loss_fn()
        if minority_loss is None or not torch.isfinite(minority_loss).all():
            return False
        minority_loss.backward()
        gn = self._grad_norm()
        if gn <= self.eps:
            return False
        self._stash_perturbation(self.rho / (gn + self.eps))
        return True

    def step(
        self,
        *,
        loss_fn: Callable[[], torch.Tensor],
        minority_loss_fn: Callable[[], torch.Tensor | None],
        grad_clip: float = 0.0,
    ) -> float:
        """Один ImbSAM-шаг. Возвращает float полного loss'а для логов."""
        applied = self._compute_minority_perturbation(minority_loss_fn)
        # Шаг 3: full-batch forward+backward на (возможно) перетурбированных w.
        self.optimizer.zero_grad(set_to_none=True)
        full_loss = loss_fn()
        full_loss.backward()
        # Шаг 4: откат ε ДО optimizer.step.
        if applied:
            self._undo_stashed_perturbation()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        # Шаг 5: optimizer.step с full-batch grad'ом, w в исходной точке.
        self.optimizer.step()
        return float(full_loss.item())


def select_minority_subset(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    minority_label: float = 1.0,
    horizon_for_filter: int = 0,
    extras: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Выбрать minority-сэмплы по фильтру одного горизонта.

    ``horizon_for_filter`` — индекс horizon-а в target shape (B, H).
    Имеет смысл выбирать самый разбалансированный (например, h=6 при
    P(UP)=0.29 → minority = UP по этому горизонту).

    ``extras`` — опциональный словарь дополнительных тензоров (lr_target,
    весов и пр.) — все ужимаются по той же mask и возвращаются.
    """
    if minority_label >= 0.5:
        mask = target[:, horizon_for_filter] >= 0.5
    else:
        mask = target[:, horizon_for_filter] < 0.5
    sub_extras = {}
    if extras is not None:
        for k, v in extras.items():
            sub_extras[k] = v[mask] if v is not None else None
    return x[mask], target[mask], sub_extras
