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

from ..config import DataConfig, TradingConfig, TrainingConfig
from .imbsam import ImbSAMOptimizer, select_minority_subset
from .losses import (
    CompositeQuantLoss,
    WeightedBCEWithLogits,
    build_loss_fn,
    class_balanced_pos_weight,
)
from .mixup import maybe_apply_mixup
from .repulsion import functional_rbf_repulsion

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
    lr_array: np.ndarray | None = None,
) -> DataLoader | None:
    """Сформировать DataLoader для (x, y) или (x, y, lr).

    ``lr_array`` опционален: при composite-loss он несёт сырые
    лог-доходности (B, H) для RankIC/Sharpe-компонент. Если None —
    собираем 2-tuple, как раньше.
    """
    if arrays["x"].shape[0] == 0:
        return None
    x = torch.from_numpy(arrays["x"]).float()
    y = torch.from_numpy(arrays["y"]).float()
    if lr_array is not None:
        if lr_array.shape != arrays["y"].shape:
            msg = (
                f"lr_array shape {lr_array.shape} must match y shape "
                f"{arrays['y'].shape}"
            )
            raise ValueError(msg)
        lr = torch.from_numpy(lr_array).float()
        dataset = TensorDataset(x, y, lr)
    else:
        dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
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
        data_cfg: DataConfig | None = None,
        trading_cfg: TradingConfig | None = None,
        device: str | None = None,
        repulsion_predecessors: list[nn.Module] | None = None,
        repulsion_weight: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.trading_cfg = trading_cfg
        self.device = self._resolve_device(device)
        # Repulsive Deep Ensembles (Sprint C1, D'Angelo NeurIPS 2021).
        self._repulsion_predecessors: list[nn.Module] = list(
            repulsion_predecessors or [],
        )
        self._repulsion_weight = float(repulsion_weight)
        self.model = model.to(self.device)
        self.loss_fn, self._is_classification = self._build_loss(
            data_cfg, cfg, trading_cfg,
        )
        self.optimizer = self._build_optimizer(cfg)
        self._imbsam: ImbSAMOptimizer | None = (
            ImbSAMOptimizer(self.optimizer, self.model, rho=float(cfg.imbsam_rho))
            if getattr(cfg, "use_imbsam", False) else None
        )
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loss(
        self,
        data_cfg: DataConfig | None,
        cfg: TrainingConfig,
        trading_cfg: TradingConfig | None,
    ) -> tuple[nn.Module, bool]:
        """Собрать loss + флаг classification, перенести на device."""
        if data_cfg is not None and data_cfg.mode == "classification":
            loss = build_loss_fn(data_cfg, cfg, trading_cfg)
            is_cls = True
        else:
            loss = nn.HuberLoss(reduction="mean", delta=1.0)
            is_cls = False
        return loss.to(self.device), is_cls

    def _build_optimizer(self, cfg: TrainingConfig) -> torch.optim.Optimizer:
        """AdamW/Adam над параметрами модели + loss_fn (для UW log_var)."""
        opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        params = list(self.model.parameters()) + [
            p for p in self.loss_fn.parameters() if p.requires_grad
        ]
        return opt_cls(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def _build_loaders(
        self,
        train_arrays: dict,
        val_arrays: dict,
        *,
        train_lr: np.ndarray | None,
        val_lr: np.ndarray | None,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Собрать train + val DataLoader'ы с учётом опциональных lr-arrays."""
        if isinstance(self.loss_fn, CompositeQuantLoss):
            if train_lr is None or val_lr is None:
                logger.warning(
                    "CompositeQuantLoss активен, но lr-target не передан — "
                    "RankIC/Sharpe компоненты будут пропущены.",
                )
        train_loader = _make_loader(
            train_arrays, self.cfg.batch_size, shuffle=True, lr_array=train_lr,
        )
        val_loader = _make_loader(
            val_arrays, self.cfg.batch_size, shuffle=False, lr_array=val_lr,
        )
        if train_loader is None:
            msg = "Training set is empty"
            raise ValueError(msg)
        return train_loader, val_loader

    def _init_lr_scheduler(self) -> None:
        """CosineAnnealingLR (T2.2). При early stopping прерывается."""
        if self.cfg.scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.epochs),
                eta_min=self.cfg.learning_rate * 0.01,
            )

    def fit(
        self,
        train_arrays: dict,
        val_arrays: dict,
        *,
        checkpoint_path: Path | None = None,
        train_lr: np.ndarray | None = None,
        val_lr: np.ndarray | None = None,
    ) -> TrainingHistory:
        """Обучить модель.

        ``train_lr``/``val_lr`` — опциональные сырые лог-доходности (B, H);
        нужны только для :class:`CompositeQuantLoss` (RankIC + Sharpe).
        """
        train_loader, val_loader = self._build_loaders(
            train_arrays, val_arrays, train_lr=train_lr, val_lr=val_lr,
        )
        self._auto_tune_loss(train_arrays["y"])
        self._init_lr_scheduler()

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

    def _compute_pos_weight(self, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть (p_up, pos_weight) per-horizon.

        ``trading_cfg.use_class_balanced_pos_weight=True`` →
        Class-Balanced (Cui CVPR 2019, β=0.999); иначе legacy (1-P)/P.
        """
        p_up = np.clip(y_train.mean(axis=0), 0.05, 0.95).astype(np.float32)
        if (
            self.trading_cfg is not None
            and getattr(self.trading_cfg, "use_class_balanced_pos_weight", False)
        ):
            pos_weight = class_balanced_pos_weight(
                y_train, beta=float(self.trading_cfg.class_balanced_beta),
            )
            logger.info(
                "Class-Balanced pos_weight per horizon: %s (P(UP)=%s, β=%.3f)",
                np.round(pos_weight, 3).tolist(),
                np.round(p_up, 3).tolist(),
                float(self.trading_cfg.class_balanced_beta),
            )
        else:
            pos_weight = ((1.0 - p_up) / p_up).astype(np.float32)
            logger.info(
                "Legacy pos_weight per horizon: %s (P(UP)=%s)",
                np.round(pos_weight, 3).tolist(),
                np.round(p_up, 3).tolist(),
            )
        return p_up, pos_weight

    def _set_logit_prior(self, p_up: np.ndarray) -> None:
        """Установить prior на модели для Logit Adjustment (Menon ICLR 2021).

        Активно только если у модели есть метод ``set_logit_prior``
        (iTransformer) и ``logit_adjust_tau > 0``. Бэквард-совместимо для
        TimeXer/ConvLSTM, у которых нет этой инфраструктуры.
        """
        setter = getattr(self.model, "set_logit_prior", None)
        if setter is None:
            return
        tau = float(getattr(self.model, "logit_adjust_tau", 0.0))
        if tau <= 0.0:
            return
        setter(torch.from_numpy(p_up.astype(np.float32)))
        logger.info(
            "Logit-adjust prior set: P(UP)=%s, tau=%.3f",
            np.round(p_up, 3).tolist(), tau,
        )

    def _auto_tune_loss(self, y_train: np.ndarray) -> None:
        """Настройка loss-функции по статистике train-таргетов."""
        if y_train.size == 0:
            return
        if not self._is_classification and self.cfg.huber_delta_auto:
            delta = max(2.0 * float(np.median(np.abs(y_train))), 1e-4)
            self.loss_fn = nn.HuberLoss(reduction="mean", delta=delta)
            logger.info("Auto Huber-delta: %.5g (from train target scale)", delta)
            return
        if not self._is_classification:
            return
        p_up, pos_weight = self._compute_pos_weight(y_train)
        # Logit Adjustment первым делом — независим от типа loss'а.
        self._set_logit_prior(p_up)
        # pos_weight применяется только к WeightedBCE (legacy путь).
        # ASL/Focal/Composite его не используют — у них свой механизм
        # компенсации imbalance.
        if isinstance(self.loss_fn, WeightedBCEWithLogits):
            pw_tensor = torch.tensor(pos_weight, dtype=torch.float32)
            self.loss_fn = WeightedBCEWithLogits(pos_weight=pw_tensor).to(self.device)

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
        for batch in bar:
            x, y, lr = (batch[0], batch[1], batch[2] if len(batch) > 2 else None)
            batch_loss = self._step(x, y, lr=lr, train=train)
            total += batch_loss * x.shape[0]
            count += x.shape[0]
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"loss={total / max(count, 1):.5f}")
        return total / max(count, 1)

    def _call_loss(
        self,
        preds: torch.Tensor,
        y: torch.Tensor,
        lr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Унифицированный вызов loss с учётом разных сигнатур."""
        if isinstance(self.loss_fn, CompositeQuantLoss):
            return self.loss_fn(preds, y, lr)
        if self._is_classification:
            return self.loss_fn(preds, y, None)
        return self.loss_fn(preds, y)

    def _compute_loss(
        self,
        preds: torch.Tensor,
        y: torch.Tensor,
        lr: torch.Tensor | None,
        *,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Базовый loss + опциональная repulsion (D'Angelo NeurIPS 2021)."""
        base = self._call_loss(preds, y, lr)
        if x is None or not self._repulsion_predecessors or self._repulsion_weight <= 0:
            return base
        return base + functional_rbf_repulsion(
            preds, x,
            self._repulsion_predecessors,
            weight=self._repulsion_weight,
        )

    def _maybe_mixup(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Применить Mixup (Sprint B2) если включён в cfg и train-режим."""
        alpha = float(getattr(self.cfg, "mixup_alpha", 0.0))
        p = float(getattr(self.cfg, "mixup_p", 0.0))
        if alpha <= 0 or p <= 0:
            return x, y, lr
        x_mix, y_mix, lr_mix, _ = maybe_apply_mixup(
            x, y, lr, alpha=alpha, p=p,
        )
        return x_mix, y_mix, lr_mix

    def _eval_step(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> float:
        """Forward без backward — для val-эпохи. Repulsion в val не считаем."""
        with torch.no_grad():
            preds = self.model(x)
            loss = self._compute_loss(preds, y, lr)  # без x → без repulsion
        return float(loss.item())

    def _train_step_plain(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> float:
        """Обычный train-шаг (без ImbSAM)."""
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.model(x)
        loss = self._compute_loss(preds, y, lr, x=x)
        loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def _train_step_imbsam(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> float:
        """ImbSAM-шаг (B1): perturbation на minority-сэмплах + full-batch backward."""
        h_idx = int(getattr(self.cfg, "imbsam_horizon_index", 0))
        x_min, y_min, sub = select_minority_subset(
            x, y, horizon_for_filter=h_idx,
            extras={"lr": lr} if lr is not None else None,
        )
        lr_min = sub.get("lr") if sub else None

        def loss_fn() -> torch.Tensor:
            return self._compute_loss(self.model(x), y, lr, x=x)

        def minority_loss_fn() -> torch.Tensor | None:
            if x_min.shape[0] == 0:
                return None
            # Repulsion на minority-подмножестве тоже считаем — это
            # последовательно с поведением full-batch шага.
            return self._compute_loss(self.model(x_min), y_min, lr_min, x=x_min)

        return self._imbsam.step(
            loss_fn=loss_fn,
            minority_loss_fn=minority_loss_fn,
            grad_clip=self.cfg.grad_clip,
        )

    def _step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        lr: torch.Tensor | None = None,
        train: bool,
    ) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        if lr is not None:
            lr = lr.to(self.device)
        if not train:
            return self._eval_step(x, y, lr)
        # Train: опциональный mixup, затем либо ImbSAM, либо plain.
        x, y, lr = self._maybe_mixup(x, y, lr)
        if self._imbsam is not None:
            return self._train_step_imbsam(x, y, lr)
        return self._train_step_plain(x, y, lr)


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
