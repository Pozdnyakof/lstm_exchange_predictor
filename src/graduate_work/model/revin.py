"""Reversible Instance Normalization (Kim et al., ICLR 2022).

https://openreview.net/forum?id=cGDAkQo1C0p

Per-instance нормализация ВХОДА сети: каждое окно (batch element)
центрируется и масштабируется по своим собственным mean/std (по
временной оси), что снимает медленный distribution shift между
периодами обучения и инференса.

В нашем пайплайне используется как input-only нормализация: модель
обучается работать со стандартизованным входом, denormalize не
применяется (выход - нормализованная лог-доходность, отдельная
концептуально шкала от исходных признаков).

Это адаптивная per-instance нормализация поверх глобального
StandardScaler. Текст ВКР §2.2 говорит про «нормализацию признакового
пространства» - RevIN дополняет глобальный шаг адаптивным.
"""

from __future__ import annotations

import torch
from torch import nn

_EPS = 1e-5


class RevIN(nn.Module):
    """Reversible per-instance normalization с learnable affine.

    Ожидает вход (B, T, F). На выходе того же шейпа - нормализованный по
    оси времени для каждого батч-элемента отдельно.
    """

    def __init__(
        self,
        num_features: int,
        *,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False) + _EPS).sqrt().detach()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x
