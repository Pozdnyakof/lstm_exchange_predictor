"""Цикл обучения PyTorch-модели регрессии лог-доходности."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    arrays: dict,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader | None:
    if arrays["x"].shape[0] == 0:
        return None
    x = torch.from_numpy(arrays["x"]).float()
    y = torch.from_numpy(arrays["y"]).float()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


class Trainer:
    """Минимальный, но самодостаточный тренер.

    Ответственности:
        - Adam + weight decay, gradient clipping;
        - Huber loss (устойчив к выбросам в финансовых рядах);
        - early stopping по валидационной потере;
        - сохранение лучшего чекпоинта на диск.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        *,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.loss_fn = nn.HuberLoss(reduction="mean", delta=1.0)

    def fit(
        self,
        train_arrays: dict,
        val_arrays: dict,
        *,
        checkpoint_path: Path | None = None,
    ) -> TrainingHistory:
        train_loader = _make_loader(train_arrays, self.cfg.batch_size, shuffle=True)
        val_loader = _make_loader(val_arrays, self.cfg.batch_size, shuffle=False)
        if train_loader is None:
            msg = "Training set is empty"
            raise ValueError(msg)

        history = TrainingHistory()
        best_state: dict | None = None
        patience = self.cfg.early_stopping_patience
        bad_epochs = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._epoch(train_loader, train=True)
            val_loss = self._epoch(val_loader, train=False) if val_loader else math.nan

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            logger.info(
                "epoch=%03d  train_loss=%.6f  val_loss=%.6f",
                epoch, train_loss, val_loss,
            )

            if not math.isnan(val_loss) and val_loss < history.best_val_loss - 1e-6:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

        return history

    def _epoch(self, loader: DataLoader | None, *, train: bool) -> float:
        if loader is None:
            return math.nan
        self.model.train(train)
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.set_grad_enabled(train):
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
        return total / max(count, 1)
