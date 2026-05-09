"""ConcurrentDeepEnsembleTrainer: параллельное обучение M членов.

Все M моделей живут на GPU одновременно, DataLoader итерируется ОДИН раз
за эпоху. Для каждого батча: cached-forward всех членов (no_grad для
SVGD-targets), затем для каждого члена — fresh forward + loss с
all-pairs repulsion + backward + step.

**Когда использовать:**
- VRAM: M × (модель + activations) — для small iTransformer (d=64,
  layers=2) и batch=2048×384×53 это ~12 GB при M=5 (умещается на L4).
- RAM: 1× данных (как у sequential), плюс small overhead на M
  оптимизаторов.
- Wall-clock: ≈ M × single-model time (не быстрее), но **DataLoader
  работает 1× вместо M×** → большой выигрыш на медленных дисках.

**Главное преимущество** относительно :class:`DeepEnsembleTrainer`:
все члены тренируются ОДНОВРЕМЕННО → корректная simultaneous SVGD
(D'Angelo & Fortuin NeurIPS 2021, §4.2) вместо sequential-аппроксимации.
В sequential mode 2-й член отталкивается от 1-го frozen, 3-й от 1+2
frozen и т.д. — это смещённая оценка SVGD-update'а. В concurrent mode
все M членов на каждом шаге толкаются друг от друга в realtime, что
канонично соответствует Stein Variational Gradient Descent.

Использование::

    ens = ConcurrentDeepEnsembleTrainer(
        model_factory, training_cfg,
        ensemble_size=5,
        data_cfg=data_cfg, trading_cfg=trading_cfg,
        svgd_repulsion_weight=0.1,
    )
    ens.fit(train, val, train_lr=train_lr, val_lr=val_lr)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in deps
    def tqdm(it, **kw):  # type: ignore[no-redef]
        return it

from ..config import DataConfig, TradingConfig, TrainingConfig
from .ensemble import EnsembleHistory, ModelFactory
from .imbsam import select_minority_subset
from .losses import CompositeQuantLoss, build_loss_fn, class_balanced_pos_weight
from .mixup import maybe_apply_mixup
from .repulsion import svgd_pairwise_repulsion
from .trainer import TrainingHistory, _make_loader, set_seed

logger = logging.getLogger(__name__)


@dataclass
class _MemberState:
    """Внутреннее состояние одного члена ансамбля при concurrent-обучении."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    history: TrainingHistory = field(default_factory=TrainingHistory)
    seed: int = 0
    best_state: dict | None = None
    bad_epochs: int = 0


class ConcurrentDeepEnsembleTrainer:
    """Параллельное обучение M членов на одном GPU.

    Контракт ``model_factory(seed: int) -> nn.Module``: фабрика, как у
    :class:`DeepEnsembleTrainer`. Все построенные модели сразу
    переезжают на ``device``.

    ``svgd_repulsion_weight``: вес all-pairs RBF-репульсии между
    предсказаниями членов. 0 = обычный non-repulsive concurrent
    ensemble. 0.1-0.5 — рекомендуемый диапазон D'Angelo NeurIPS 2021.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        training_cfg: TrainingConfig,
        *,
        ensemble_size: int = 5,
        data_cfg: DataConfig | None = None,
        trading_cfg: TradingConfig | None = None,
        device: str | None = None,
        base_seed: int | None = None,
        svgd_repulsion_weight: float = 0.0,
    ) -> None:
        if ensemble_size < 2:
            msg = f"ensemble_size must be >= 2, got {ensemble_size}"
            raise ValueError(msg)
        if svgd_repulsion_weight < 0:
            msg = (
                f"svgd_repulsion_weight must be >= 0, got {svgd_repulsion_weight}"
            )
            raise ValueError(msg)

        self.model_factory = model_factory
        self.training_cfg = training_cfg
        self.ensemble_size = int(ensemble_size)
        self.data_cfg = data_cfg
        self.trading_cfg = trading_cfg
        self.svgd_repulsion_weight = float(svgd_repulsion_weight)
        self.base_seed = (
            int(base_seed) if base_seed is not None else int(training_cfg.seed)
        )
        self.device = self._resolve_device(device)
        self._is_classification = (
            data_cfg is not None and data_cfg.mode == "classification"
        )
        # Построить M членов.
        self._members: list[_MemberState] = self._build_members()

    # ------------------------------------------------------------------
    # Helpers: device resolution + member construction
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_members(self) -> list[_MemberState]:
        """Создать M моделей + per-member optimizer + loss."""
        members: list[_MemberState] = []
        for i in range(self.ensemble_size):
            seed = self.base_seed + i
            set_seed(seed)
            model = self.model_factory(seed).to(self.device)
            loss_fn = self._build_loss_for_member().to(self.device)
            optim = self._build_optimizer(model, loss_fn)
            members.append(_MemberState(
                model=model, optimizer=optim,
                loss_fn=loss_fn, seed=seed,
            ))
        logger.info(
            "Built %d concurrent ensemble members on %s "
            "(svgd_repulsion=%.3f)",
            self.ensemble_size, self.device, self.svgd_repulsion_weight,
        )
        return members

    def _build_loss_for_member(self) -> nn.Module:
        """Per-member loss. classification → build_loss_fn; иначе HuberLoss."""
        if self._is_classification:
            return build_loss_fn(self.data_cfg, self.training_cfg, self.trading_cfg)
        return nn.HuberLoss(reduction="mean", delta=1.0)

    def _build_optimizer(
        self, model: nn.Module, loss_fn: nn.Module,
    ) -> torch.optim.Optimizer:
        """AdamW/Adam с включением loss_fn-параметров (UW log_var)."""
        cfg = self.training_cfg
        opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        params = list(model.parameters()) + [
            p for p in loss_fn.parameters() if p.requires_grad
        ]
        return opt_cls(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def members(self) -> list[nn.Module]:
        """Список моделей (для совместимости с :func:`ensemble_predict`)."""
        return [m.model for m in self._members]

    def fit(
        self,
        train_arrays: dict,
        val_arrays: dict,
        *,
        checkpoint_dir: Path | None = None,
        train_lr: np.ndarray | None = None,
        val_lr: np.ndarray | None = None,
    ) -> EnsembleHistory:
        """Запустить параллельное обучение всех членов.

        ``train_lr``/``val_lr`` — опциональные сырые лог-доходности
        (для composite loss).
        """
        train_loader, val_loader = self._build_loaders(
            train_arrays, val_arrays, train_lr=train_lr, val_lr=val_lr,
        )
        # Auto-tune (logit prior + class-balanced) для всех членов.
        self._auto_tune_all(train_arrays["y"])

        epochs = max(1, int(self.training_cfg.epochs))
        patience = int(self.training_cfg.early_stopping_patience)
        bar = tqdm(range(1, epochs + 1), desc="Concurrent ensemble", unit="epoch")
        for epoch in bar:
            train_losses = self._epoch_concurrent(
                train_loader, train=True, epoch=epoch, phase="train",
            )
            val_losses = (
                self._epoch_concurrent(
                    val_loader, train=False, epoch=epoch, phase="val",
                )
                if val_loader else [math.nan] * self.ensemble_size
            )
            stop_all = self._update_histories(
                epoch, train_losses, val_losses, patience,
            )
            self._update_outer_bar(bar, epoch, train_losses, val_losses)
            if stop_all:
                logger.info(
                    "All %d members triggered early stopping at epoch %d",
                    self.ensemble_size, epoch,
                )
                break

        # Восстанавливаем best_state для каждого члена.
        for ms in self._members:
            if ms.best_state is not None:
                ms.model.load_state_dict(ms.best_state)
            ms.model.eval()

        if checkpoint_dir is not None:
            self._save_checkpoints(checkpoint_dir)

        return EnsembleHistory(
            member_histories=[m.history for m in self._members],
            seeds=[m.seed for m in self._members],
            checkpoint_paths=(
                [checkpoint_dir / f"member_{i:02d}_seed{m.seed}.pt"
                 for i, m in enumerate(self._members)]
                if checkpoint_dir is not None else []
            ),
        )

    def _build_loaders(
        self,
        train_arrays: dict, val_arrays: dict,
        *,
        train_lr: np.ndarray | None, val_lr: np.ndarray | None,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Один shared DataLoader для всех M членов."""
        # Если у любого члена composite loss — нужны lr-arrays.
        needs_lr = any(
            isinstance(m.loss_fn, CompositeQuantLoss) for m in self._members
        )
        if needs_lr and (train_lr is None or val_lr is None):
            logger.warning(
                "Composite loss активен, но train_lr/val_lr не переданы; "
                "RankIC/Sharpe будут пропущены.",
            )
        train_loader = _make_loader(
            train_arrays, self.training_cfg.batch_size,
            shuffle=True, lr_array=train_lr,
        )
        val_loader = _make_loader(
            val_arrays, self.training_cfg.batch_size,
            shuffle=False, lr_array=val_lr,
        )
        if train_loader is None:
            msg = "Training set is empty"
            raise ValueError(msg)
        return train_loader, val_loader

    def _auto_tune_all(self, y_train: np.ndarray) -> None:
        """Применить logit-adjustment prior + class-balanced pos_weight ко всем
        членам по статистике train-таргетов."""
        if y_train.size == 0 or not self._is_classification:
            return
        p_up = np.clip(y_train.mean(axis=0), 0.05, 0.95).astype(np.float32)
        # Logit Adjustment (если модель его поддерживает).
        prior_tensor = torch.from_numpy(p_up.astype(np.float32))
        for ms in self._members:
            setter = getattr(ms.model, "set_logit_prior", None)
            tau = float(getattr(ms.model, "logit_adjust_tau", 0.0))
            if setter is not None and tau > 0.0:
                setter(prior_tensor)
        logger.info(
            "Auto-tuned %d members: P(UP)=%s",
            self.ensemble_size, np.round(p_up, 3).tolist(),
        )

    # ------------------------------------------------------------------
    # Per-batch concurrent step
    # ------------------------------------------------------------------

    def _maybe_mixup(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Mixup один раз на батч; результат разделяется ВСЕМИ M членами."""
        cfg = self.training_cfg
        alpha = float(getattr(cfg, "mixup_alpha", 0.0))
        p = float(getattr(cfg, "mixup_p", 0.0))
        if alpha <= 0 or p <= 0:
            return x, y, lr
        x_mix, y_mix, lr_mix, _ = maybe_apply_mixup(x, y, lr, alpha=alpha, p=p)
        return x_mix, y_mix, lr_mix

    def _call_loss(
        self, member: _MemberState,
        preds: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Унифицированный вызов loss с учётом разных сигнатур."""
        if isinstance(member.loss_fn, CompositeQuantLoss):
            return member.loss_fn(preds, y, lr)
        if self._is_classification:
            return member.loss_fn(preds, y, None)
        return member.loss_fn(preds, y)

    @torch.no_grad()
    def _cache_other_preds(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Cached forward всех членов (no_grad) — таргеты для SVGD-репульсии.

        Каждый член переключается в eval() для детерминированного forward'а
        (dropout/BN отключены), потом возвращается в train(). Это важно:
        если оставить dropout активным в repulsion-target, kernel будет
        зашумлён и градиент репульсии станет noisy.
        """
        cached: list[torch.Tensor] = []
        for ms in self._members:
            was_training = ms.model.training
            ms.model.eval()
            cached.append(ms.model(x).detach())
            if was_training:
                ms.model.train()
        return cached

    def _step_member(
        self,
        i: int,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
        cached_others: list[torch.Tensor],
    ) -> float:
        """Один backward+step для члена i с SVGD-репульсией от остальных."""
        ms = self._members[i]
        ms.optimizer.zero_grad(set_to_none=True)
        preds_i = ms.model(x)
        base = self._call_loss(ms, preds_i, y, lr)
        if self.svgd_repulsion_weight > 0 and len(cached_others) > 1:
            others = [cached_others[j] for j in range(len(cached_others)) if j != i]
            base = base + svgd_pairwise_repulsion(
                preds_i, others, weight=self.svgd_repulsion_weight,
            )
        base.backward()
        if self.training_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                ms.model.parameters(), self.training_cfg.grad_clip,
            )
        ms.optimizer.step()
        return float(base.item())

    @torch.no_grad()
    def _eval_step_member(
        self,
        i: int,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> float:
        """Val-шаг одного члена. Без репульсии (валидация — чистый loss)."""
        ms = self._members[i]
        preds = ms.model(x)
        return float(self._call_loss(ms, preds, y, lr).item())

    def _train_batch(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> list[float]:
        """Concurrent batch: один shared mixup → cached forwards → M шагов."""
        x, y, lr = self._maybe_mixup(x, y, lr)
        cached = self._cache_other_preds(x) if self.svgd_repulsion_weight > 0 else []
        per_member_loss: list[float] = []
        for i in range(self.ensemble_size):
            per_member_loss.append(
                self._step_member(i, x, y, lr, cached),
            )
        return per_member_loss

    def _eval_batch(
        self,
        x: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None,
    ) -> list[float]:
        """Val-batch: M eval-шагов без репульсии."""
        return [
            self._eval_step_member(i, x, y, lr)
            for i in range(self.ensemble_size)
        ]

    def _epoch_concurrent(
        self,
        loader: DataLoader | None,
        *,
        train: bool,
        epoch: int,
        phase: str,
    ) -> list[float]:
        """Эпоха concurrent-обучения. Возвращает per-member средний loss."""
        if loader is None:
            return [math.nan] * self.ensemble_size
        for ms in self._members:
            ms.model.train(train)
        totals = np.zeros(self.ensemble_size, dtype=np.float64)
        count = 0
        bar = tqdm(loader, desc=f"  ep{epoch:02d} {phase}", unit="batch", leave=False)
        for batch in bar:
            x_cpu, y_cpu = batch[0], batch[1]
            lr_cpu = batch[2] if len(batch) > 2 else None
            x = x_cpu.to(self.device)
            y = y_cpu.to(self.device)
            lr = lr_cpu.to(self.device) if lr_cpu is not None else None
            losses = (
                self._train_batch(x, y, lr) if train
                else self._eval_batch(x, y, lr)
            )
            totals += np.array(losses) * x.shape[0]
            count += x.shape[0]
            if hasattr(bar, "set_postfix_str"):
                avg = totals / max(count, 1)
                bar.set_postfix_str(
                    f"loss(min/max)={avg.min():.4f}/{avg.max():.4f}",
                )
        return (totals / max(count, 1)).tolist()

    # ------------------------------------------------------------------
    # Bookkeeping: best/early-stopping per-member
    # ------------------------------------------------------------------

    def _update_histories(
        self,
        epoch: int,
        train_losses: list[float],
        val_losses: list[float],
        patience: int,
    ) -> bool:
        """Обновить TrainingHistory и best_state каждого члена. Возвращает
        True если ВСЕ члены достигли early-stop."""
        all_stopped = True
        for ms, tl, vl in zip(self._members, train_losses, val_losses):
            ms.history.train_loss.append(float(tl))
            ms.history.val_loss.append(float(vl))
            improved = (
                not math.isnan(vl)
                and vl < ms.history.best_val_loss - 1e-6
            )
            if improved:
                ms.history.best_val_loss = float(vl)
                ms.history.best_epoch = epoch
                ms.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in ms.model.state_dict().items()
                }
                ms.bad_epochs = 0
                all_stopped = False
            else:
                ms.bad_epochs += 1
                if ms.bad_epochs < patience:
                    all_stopped = False
        return all_stopped

    @staticmethod
    def _update_outer_bar(
        bar, epoch: int,
        train_losses: list[float], val_losses: list[float],
    ) -> None:
        if not hasattr(bar, "set_postfix"):
            return
        tr = np.asarray(train_losses)
        va = np.asarray(val_losses)
        bar.set_postfix(
            train_avg=f"{tr.mean():.4f}",
            val_avg=f"{va.mean():.4f}",
            val_spread=f"{va.max() - va.min():.4f}",
        )

    def _save_checkpoints(self, checkpoint_dir: Path) -> None:
        """Сохранить per-member чекпоинты + manifest."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_members = []
        for i, ms in enumerate(self._members):
            ckpt = checkpoint_dir / f"member_{i:02d}_seed{ms.seed}.pt"
            torch.save(ms.model.state_dict(), ckpt)
            manifest_members.append({
                "seed": ms.seed,
                "checkpoint": ckpt.name,
                "best_val_loss": float(ms.history.best_val_loss),
                "best_epoch": int(ms.history.best_epoch),
            })
        manifest = {
            "ensemble_size": self.ensemble_size,
            "base_seed": self.base_seed,
            "svgd_repulsion_weight": self.svgd_repulsion_weight,
            "members": manifest_members,
        }
        (checkpoint_dir / "ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Saved concurrent ensemble manifest to %s", checkpoint_dir)
