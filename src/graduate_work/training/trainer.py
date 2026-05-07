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
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm в зависимостях
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

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
        opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = opt_cls(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        # δ Huber-loss подбирается на основе распределения таргетов (см. fit()).
        # До первого fit() ставим консервативный дефолт.
        self.loss_fn = nn.HuberLoss(reduction="mean", delta=1.0)
        # CosineAnnealing-планировщик опционально (cfg.scheduler).
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

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

        # T1.2: δ Huber подбираем по масштабу таргетов. Иначе при δ=1.0
        # и таргете масштаба 1e-3 функция эффективно становится MSE, и
        # модель сходится к предсказанию нуля.
        if self.cfg.huber_delta_auto:
            y_train = train_arrays["y"]
            if y_train.size > 0:
                delta = max(2.0 * float(np.median(np.abs(y_train))), 1e-4)
                self.loss_fn = nn.HuberLoss(reduction="mean", delta=delta)
                logger.info("Auto Huber-delta: %.5g (from train target scale)", delta)

        # T2.2: CosineAnnealingLR-планировщик. Длится все epochs;
        # при early stopping просто прерывается на полпути.
        if self.cfg.scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.epochs),
                eta_min=self.cfg.learning_rate * 0.01,
            )

        history = TrainingHistory()
        ctx = _FitContext(history=history, patience=self.cfg.early_stopping_patience)
        self._init_swa(ctx)

        outer = tqdm(
            range(1, self.cfg.epochs + 1),
            desc="Training", unit="epoch",
        )
        for epoch in outer:
            train_loss = self._epoch(train_loader, train=True, epoch=epoch, phase="train")
            val_loss = (
                self._epoch(val_loader, train=False, epoch=epoch, phase="val")
                if val_loader else math.nan
            )
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            self._update_outer_bar(outer, epoch, train_loss, val_loss, history)
            self._maybe_step_swa(ctx, epoch)
            stop = self._update_best(ctx, val_loss, epoch)
            if stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        outer.close()
        # SWA имеет приоритет над best_state, если был активирован.
        if ctx.swa_active and ctx.swa_model is not None:
            self._finalize_swa(ctx)
        elif ctx.best_state is not None:
            self.model.load_state_dict(ctx.best_state)
        self._save_checkpoint(checkpoint_path)
        return history

    def _init_swa(self, ctx: _FitContext) -> None:
        if not self.cfg.use_swa:
            return
        ctx.swa_start_epoch = max(1, int(self.cfg.epochs * self.cfg.swa_start_frac))
        ctx.swa_model = AveragedModel(self.model)
        ctx.swa_scheduler = SWALR(self.optimizer, swa_lr=self.cfg.swa_lr)
        logger.info("SWA enabled, start at epoch %d, swa_lr=%.4g",
                    ctx.swa_start_epoch, self.cfg.swa_lr)

    def _maybe_step_swa(self, ctx: _FitContext, epoch: int) -> None:
        if ctx.swa_model is None or ctx.swa_scheduler is None:
            # SWA выключен → шагаем обычным cosine scheduler.
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            return
        if epoch < ctx.swa_start_epoch:
            # Пока не активен SWA - шагает основной cosine scheduler.
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            return
        # SWA-фаза: используем SWALR, обычный scheduler не трогаем.
        ctx.swa_active = True
        ctx.swa_model.update_parameters(self.model)
        ctx.swa_scheduler.step()

    def _finalize_swa(self, ctx: _FitContext) -> None:
        if ctx.swa_model is None:
            return
        # Копируем усреднённые веса в основную модель. BN-stats не
        # пересчитываем (сеть их не использует).
        averaged_state = {
            k.removeprefix("module."): v.detach().clone()
            for k, v in ctx.swa_model.state_dict().items()
            if not k.startswith("n_averaged")
        }
        self.model.load_state_dict(averaged_state, strict=False)
        logger.info("SWA finalized: averaged weights copied into model")

    @staticmethod
    def _update_outer_bar(
        bar, epoch: int, train_loss: float, val_loss: float, history: TrainingHistory,
    ) -> None:
        if not hasattr(bar, "set_postfix"):
            return
        bar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            best=f"e{history.best_epoch}={history.best_val_loss:.5f}",
        )

    def _update_best(
        self, ctx: "_FitContext", val_loss: float, epoch: int,
    ) -> bool:
        improved = (
            not math.isnan(val_loss)
            and val_loss < ctx.history.best_val_loss - 1e-6
        )
        if improved:
            ctx.history.best_val_loss = val_loss
            ctx.history.best_epoch = epoch
            ctx.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            ctx.bad_epochs = 0
            return False
        ctx.bad_epochs += 1
        return ctx.bad_epochs >= ctx.patience

    def _save_checkpoint(self, checkpoint_path: Path | None) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)

    def _epoch(
        self,
        loader: DataLoader | None,
        *,
        train: bool,
        epoch: int,
        phase: str,
    ) -> float:
        if loader is None:
            return math.nan
        self.model.train(train)
        total = 0.0
        count = 0
        bar = tqdm(
            loader,
            desc=f"  ep{epoch:02d} {phase}",
            unit="batch",
            leave=False,
        )
        for x, y in bar:
            batch_loss = self._step(x, y, train=train)
            total += batch_loss * x.shape[0]
            count += x.shape[0]
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"loss={total / max(count, 1):.5f}")
        return total / max(count, 1)

    def _step(self, x: torch.Tensor, y: torch.Tensor, *, train: bool) -> float:
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
        return float(loss.item())


@dataclass
class _FitContext:
    history: TrainingHistory
    patience: int
    bad_epochs: int = 0
    best_state: dict | None = None
    # SWA-инфраструктура: усреднённая модель + LR-планировщик.
    # Активируется только если cfg.use_swa и эпоха >= swa_start.
    swa_model: AveragedModel | None = None
    swa_scheduler: SWALR | None = None
    swa_start_epoch: int = 0
    swa_active: bool = False
