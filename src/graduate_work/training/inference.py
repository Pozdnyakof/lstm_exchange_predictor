"""MC Dropout инференс: многократный прямой проход и оценка неопределённости.

Возвращаемое распределение из ``mc_passes`` сэмплов используется
дальше в стратегии для двухступенчатой фильтрации сигналов (§2.2).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..model.mc_dropout import set_mc_dropout


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    mc_passes: int = 50,
    batch_size: int = 256,
    device: str | None = None,
    apply_sigmoid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть (mean, std) по ``mc_passes`` стохастическим проходам.

    Размерности:
        x     - (N, T, F)
        mean  - (N, H)
        std   - (N, H)

    ``apply_sigmoid=True`` нужен для classification-режима: модель
    выдаёт логиты, сигмоиду применяем ВНУТРИ MC-цикла на каждый
    проход (чтобы дисперсия считалась в probability-пространстве, а
    не в logit-пространстве). Для regression - False.
    """
    if x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model = model.to(target_device)
    model.eval()
    set_mc_dropout(model, True)
    try:
        tensor_x = torch.from_numpy(x).float()
        all_passes: list[np.ndarray] = []
        passes_bar = tqdm(range(mc_passes), desc="MC passes", unit="pass", leave=False)
        for _ in passes_bar:
            preds: list[np.ndarray] = []
            for start in range(0, tensor_x.shape[0], batch_size):
                batch = tensor_x[start:start + batch_size].to(target_device)
                out = model(batch)
                if apply_sigmoid:
                    out = torch.sigmoid(out)
                preds.append(out.detach().cpu().numpy())
            all_passes.append(np.concatenate(preds, axis=0))
        stacked = np.stack(all_passes, axis=0)   # (P, N, H)
    finally:
        set_mc_dropout(model, False)

    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0, ddof=0).astype(np.float32)
    return mean, std
