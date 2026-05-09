"""Deep Ensemble: N независимых моделей с разными сидами.

Lakshminarayanan, Pritzel, Blundell (2017),
*Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*
[arXiv:1612.01474](https://arxiv.org/abs/1612.01474).

Заменяет MC Dropout как источник эпистемической неопределённости. На
финансовых задачах daily/intraday эмпирически даёт лучшую калибровку,
особенно при class imbalance и distribution shift между train и test.

Идея:
- Обучаем M моделей на одних и тех же данных, но с разной инициализацией
  (разные seeds + разный shuffling). Каждая сходится к своему локальному
  минимуму.
- На инференсе усредняем предсказания всех моделей и считаем std по
  моделям как эпистемическую неопределённость.
- Если внутри каждой модели ещё используется MC Dropout — получаем
  дополнительную aleatoric-составляющую (но это опционально).

Использование::

    ens = DeepEnsembleTrainer(model_factory, training_cfg, ensemble_size=5)
    ens.fit(train_arrays, val_arrays, checkpoint_dir=...)
    mean, std = ens.predict(test_arrays['x'])  # std — эпистемическая

Контракт ``model_factory(seed: int) -> nn.Module``: фабрика-замыкание,
возвращающая свежую (необученную) сеть для данного seed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..config import DataConfig, TradingConfig, TrainingConfig
from .trainer import Trainer, TrainingHistory, set_seed

logger = logging.getLogger(__name__)


ModelFactory = Callable[[int], nn.Module]


@dataclass
class EnsembleHistory:
    """Метаданные обучения каждого члена ансамбля."""

    member_histories: list[TrainingHistory] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)

    @property
    def best_val_losses(self) -> list[float]:
        return [h.best_val_loss for h in self.member_histories]


class DeepEnsembleTrainer:
    """Обучает M моделей независимо и сохраняет их чекпоинты.

    Каждый член ансамбля получает уникальный seed (``base_seed + i``),
    что меняет: (1) инициализацию весов, (2) порядок батчей в DataLoader.
    Этого достаточно для расхождения локальных минимумов.

    Параллельное обучение НЕ реализовано (по умолчанию sequential), потому
    что в Colab/L4-окружении одна модель уже занимает большую часть VRAM.
    Для multi-GPU кластеров можно обернуть фабрику ``torch.nn.DataParallel``
    или запускать процессы вручную.
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
    ) -> None:
        if ensemble_size < 2:
            msg = (
                f"ensemble_size must be >= 2 for meaningful UQ, got {ensemble_size}"
            )
            raise ValueError(msg)
        self.model_factory = model_factory
        self.training_cfg = training_cfg
        self.ensemble_size = int(ensemble_size)
        self.data_cfg = data_cfg
        self.trading_cfg = trading_cfg
        self.device = device
        self.base_seed = (
            int(base_seed) if base_seed is not None else int(training_cfg.seed)
        )
        # Список обученных моделей (in-memory) — заполняется в fit().
        self.members: list[nn.Module] = []

    def fit(
        self,
        train_arrays: dict,
        val_arrays: dict,
        *,
        checkpoint_dir: Path | None = None,
        train_lr: np.ndarray | None = None,
        val_lr: np.ndarray | None = None,
    ) -> EnsembleHistory:
        """Обучить M моделей последовательно.

        ``train_lr``/``val_lr`` — опциональные сырые лог-доходности
        (передаются в каждый Trainer.fit() для composite loss).
        """
        history = EnsembleHistory()
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.ensemble_size):
            seed = self.base_seed + i
            logger.info(
                "=== Training ensemble member %d/%d (seed=%d) ===",
                i + 1, self.ensemble_size, seed,
            )
            set_seed(seed)
            model = self.model_factory(seed)
            trainer = Trainer(
                model,
                self.training_cfg,
                data_cfg=self.data_cfg,
                trading_cfg=self.trading_cfg,
                device=self.device,
            )
            ckpt = (
                checkpoint_dir / f"member_{i:02d}_seed{seed}.pt"
                if checkpoint_dir is not None else None
            )
            member_history = trainer.fit(
                train_arrays, val_arrays,
                checkpoint_path=ckpt,
                train_lr=train_lr, val_lr=val_lr,
            )
            self.members.append(trainer.model.eval())
            history.member_histories.append(member_history)
            history.seeds.append(seed)
            if ckpt is not None:
                history.checkpoint_paths.append(ckpt)

        if checkpoint_dir is not None:
            self._save_manifest(checkpoint_dir, history)
        return history

    def _save_manifest(
        self, checkpoint_dir: Path, history: EnsembleHistory,
    ) -> None:
        manifest = {
            "ensemble_size": self.ensemble_size,
            "base_seed": self.base_seed,
            "members": [
                {
                    "seed": s,
                    "checkpoint": str(p.name),
                    "best_val_loss": float(h.best_val_loss),
                    "best_epoch": int(h.best_epoch),
                }
                for s, p, h in zip(
                    history.seeds, history.checkpoint_paths,
                    history.member_histories,
                )
            ],
        }
        (checkpoint_dir / "ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved ensemble manifest to %s",
            checkpoint_dir / "ensemble_manifest.json",
        )

    def load_from_dir(
        self, checkpoint_dir: Path, *, factory_seed: int | None = None,
    ) -> None:
        """Загрузить веса всех членов из checkpoint_dir.

        Восстанавливает self.members — N моделей с подгруженными весами.
        Использует тот же ``model_factory`` для построения архитектуры;
        ``factory_seed`` если задан — используется для всех (не влияет
        на веса, влияет только на placeholder-инициализацию).
        """
        manifest_path = checkpoint_dir / "ensemble_manifest.json"
        if not manifest_path.exists():
            msg = f"Ensemble manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_device = torch.device(
            self.device if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.members = []
        for entry in manifest["members"]:
            seed = factory_seed if factory_seed is not None else int(entry["seed"])
            model = self.model_factory(seed)
            ckpt_path = checkpoint_dir / entry["checkpoint"]
            state = torch.load(ckpt_path, map_location=target_device)
            model.load_state_dict(state)
            model.to(target_device).eval()
            self.members.append(model)
        logger.info(
            "Loaded %d ensemble members from %s",
            len(self.members), checkpoint_dir,
        )


@torch.no_grad()
def _predict_one_member(
    model: nn.Module,
    tensor_x: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    apply_sigmoid: bool,
) -> np.ndarray:
    """Прогон одного члена ансамбля по всему x. Возвращает (N, H)."""
    model = model.to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, tensor_x.shape[0], batch_size):
        batch = tensor_x[start:start + batch_size].to(device)
        out = model(batch)
        if apply_sigmoid:
            out = torch.sigmoid(out)
        chunks.append(out.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def ensemble_predict(
    members: list[nn.Module],
    x: np.ndarray,
    *,
    batch_size: int = 256,
    device: str | None = None,
    apply_sigmoid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Усреднить прогнозы по членам ансамбля и вернуть (mean, std).

    ``mean`` shape (N, H), ``std`` shape (N, H). std — эпистемическая
    неопределённость (variability между членами ансамбля). Aleatoric
    оценивается отдельно (например, из MC Dropout внутри каждого члена).

    ``apply_sigmoid=True`` — для classification, чтобы дисперсия
    считалась в probability-пространстве, как в ``mc_predict``.
    """
    if not members:
        msg = "ensemble has no members; call fit() or load_from_dir() first"
        raise ValueError(msg)
    if x.shape[0] == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    target_device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tensor_x = torch.from_numpy(x).float()
    member_preds = [
        _predict_one_member(
            m, tensor_x,
            batch_size=batch_size, device=target_device,
            apply_sigmoid=apply_sigmoid,
        )
        for m in members
    ]
    stacked = np.stack(member_preds, axis=0)         # (M, N, H)
    mean = stacked.mean(axis=0).astype(np.float32)
    # Несмещённая дисперсия (ddof=1) — потому что у нас выборочная
    # оценка std по M независимым моделям.
    std = stacked.std(axis=0, ddof=1).astype(np.float32)
    return mean, std
