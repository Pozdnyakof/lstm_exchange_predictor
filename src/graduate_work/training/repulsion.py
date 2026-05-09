"""Function-space repulsion для Repulsive Deep Ensembles.

D'Angelo & Fortuin, **NeurIPS 2021**, *Repulsive Deep Ensembles are
Bayesian* ([arXiv:2106.11642](https://arxiv.org/abs/2106.11642)).

Идея: чтобы M членов ансамбля **гарантированно** разошлись в разные
функциональные моды, добавляем к loss каждого члена штраф за высокое
RBF-схождение его предсказаний с уже обученными предшественниками.

Формула (sequential approximation simultaneous SVGD):
    L_member_i = L_data + λ · (1/i) · Σ_{j<i} k(f_i(x), f_j(x))

где k(a, b) = exp(−||a − b||² / (2 h²)). При близких предсказаниях
kernel ≈ 1 → штраф большой → градиент толкает f_i от f_j.

Bandwidth ``h`` ставим как медиану попарных расстояний (стандарт SVGD,
Liu & Wang 2016, *Stein Variational Gradient Descent*); такой
self-tuning делает метод устойчивым к масштабу выходов.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _median_bandwidth(diff_sq: torch.Tensor) -> torch.Tensor:
    """Median heuristic для RBF bandwidth.

    Liu & Wang (2016): h² = median(||a-b||²) / log(n+1).
    Делает kernel инвариантным к масштабу logits/probs, так что
    лямбду можно задать раз и она работает на любом output-range'е.
    """
    n = diff_sq.numel()
    if n <= 1:
        return torch.tensor(1.0, device=diff_sq.device)
    med = torch.median(diff_sq.detach())
    log_n = torch.log(torch.tensor(float(n + 1), device=diff_sq.device))
    return (med / log_n.clamp_min(_EPS)).clamp_min(_EPS)


def rbf_kernel_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RBF-kernel similarity между двумя prediction-тензорами.

    ``a``, ``b`` shape (B, H). Возвращает скаляр — среднее
    `exp(−||a_i − b_i||² / (2 h²))` по batch'у. Чем БОЛЬШЕ значение,
    тем БЛИЖЕ предсказания. Для repulsion-loss минимизируем именно его.
    """
    if a.shape != b.shape:
        msg = f"shape mismatch: a={a.shape}, b={b.shape}"
        raise ValueError(msg)
    diff = a - b                                  # (B, H)
    dist_sq = (diff ** 2).sum(dim=-1)             # (B,)
    h_sq = _median_bandwidth(dist_sq)
    kernel = torch.exp(-dist_sq / (2.0 * h_sq))
    return kernel.mean()


@torch.no_grad()
def _frozen_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Предсказание frozen-модели в eval-mode без gradient-tracking."""
    was_training = model.training
    model.eval()
    out = model(x)
    if was_training:
        model.train()
    return out


def functional_rbf_repulsion(
    current_preds: torch.Tensor,
    x: torch.Tensor,
    predecessors: list[nn.Module],
    *,
    weight: float = 0.1,
) -> torch.Tensor:
    """Repulsion-loss текущей модели от списка frozen-предшественников.

    Возвращает скаляр ``weight · mean_j RBF(current, prev_j(x))``. Если
    список пуст или вес 0 — возвращает нулевой скаляр (без autograd-связи).

    На время prev_j-форвардов модели держатся в eval-режиме (важно:
    их dropout и BN выключены, чтобы получить детерминированный
    sample их «функциональной» позиции, а не случайной точки).
    """
    if not predecessors or weight <= 0.0:
        return current_preds.new_zeros(())
    total = current_preds.new_zeros(())
    for prev in predecessors:
        prev_out = _frozen_predict(prev, x)
        total = total + rbf_kernel_loss(current_preds, prev_out)
    return weight * total / float(len(predecessors))
