"""LightGBM-only training/inference pipeline для multi-ticker микроструктуры.

После Sprint 3 показа: на нашем сигнале (AUC 0.54-0.64) LightGBM
показал Sharpe 0.98 на h=48 (best). iTransformer проиграл везде.
Этот модуль — production-pipeline БЕЗ deep learning, минимум кода
максимум сигнала.

## Особенности

- **Multi-ticker training**: один model per horizon на ВСЕХ тикерах
  (universe-wide), NOT per-ticker. Это даёт ~10× данных и cross-sectional
  фичи начинают работать.
- **Cost-aware labels**: уже привязаны к costs (как везде в этом проекте).
- **Early stopping** на val для каждого горизонта независимо.
- **Sample weights по uniqueness** (опц.) — снижают overfit от
  overlapping forward windows (López de Prado AFML ch. 4).

## Использование

    pipeline = LightGBMPipeline(cfg)
    pipeline.fit(train_features, val_features, target_cols, lr_cols)
    val_preds = pipeline.predict(val_features)
    test_preds = pipeline.predict(test_features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LightGBMConfig:
    """Гипер-параметры LightGBM, общие для всех горизонтов."""

    n_estimators: int = 500
    num_leaves: int = 63
    learning_rate: float = 0.03
    min_child_samples: int = 200
    max_depth: int = -1
    reg_lambda: float = 1.0
    reg_alpha: float = 0.1
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    early_stopping_rounds: int = 30
    random_state: int = 42

    def to_lgb_params(self) -> dict[str, Any]:
        """Преобразовать в kwargs для LGBMClassifier."""
        return {
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }


@dataclass
class LightGBMHorizonResult:
    """Результат обучения для одного горизонта."""

    horizon: int
    model: lgb.LGBMClassifier
    feature_cols: list[str]
    best_iteration: int
    val_log_loss: float
    val_auc: float | None = None


@dataclass
class LightGBMPipeline:
    """Per-horizon LightGBM ансамбль.

    Не путать с :class:`DeepEnsembleTrainer` (M моделей одного горизонта).
    Тут — N моделей, по одной на горизонт. Каждая horizon-модель
    обучается независимо.
    """

    horizons: tuple[int, ...]
    feature_cols: list[str]
    cfg: LightGBMConfig = field(default_factory=LightGBMConfig)
    models: dict[int, LightGBMHorizonResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        sample_weights_train: np.ndarray | None = None,
    ) -> None:
        """Обучить N моделей (по одной на горизонт).

        ``train_df`` и ``val_df`` должны содержать колонки:
            - все ``feature_cols``
            - ``target_h{h}`` для каждого h в horizons (binary 0/1)
        Строки с NaN target автоматически отбрасываются.

        ``sample_weights_train`` — опциональные веса (López de Prado
        uniqueness weights). Shape: (len(train_df),). Передаются
        в lgb.fit.
        """
        for h in self.horizons:
            target_col = f"target_h{h}"
            if target_col not in train_df.columns:
                logger.warning("Skipping h=%d: %s not in train_df", h, target_col)
                continue
            self._fit_one_horizon(
                h, train_df, val_df, target_col,
                sample_weights_train=sample_weights_train,
            )

    def _fit_one_horizon(
        self,
        horizon: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        *,
        sample_weights_train: np.ndarray | None,
    ) -> None:
        """Обучить ОДНУ модель для горизонта h."""
        train_mask = train_df[target_col].notna()
        val_mask = val_df[target_col].notna()
        if not train_mask.any() or not val_mask.any():
            logger.warning("h=%d: empty train/val", horizon)
            return

        X_tr = train_df.loc[train_mask, self.feature_cols]
        y_tr = train_df.loc[train_mask, target_col].astype(int).to_numpy()
        X_va = val_df.loc[val_mask, self.feature_cols]
        y_va = val_df.loc[val_mask, target_col].astype(int).to_numpy()

        weights_tr = (
            sample_weights_train[train_mask.to_numpy()]
            if sample_weights_train is not None else None
        )

        model = lgb.LGBMClassifier(**self.cfg.to_lgb_params())
        callbacks = [
            lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False),
        ]
        model.fit(
            X_tr, y_tr,
            sample_weight=weights_tr,
            eval_set=[(X_va, y_va)],
            callbacks=callbacks,
        )
        # Метрики на val (для отчётности).
        val_proba = model.predict_proba(X_va)[:, 1]
        val_clipped = np.clip(val_proba, 1e-7, 1 - 1e-7)
        val_log_loss = float(
            -np.mean(y_va * np.log(val_clipped) + (1 - y_va) * np.log(1 - val_clipped)),
        )
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = (
                float(roc_auc_score(y_va, val_proba))
                if len(np.unique(y_va)) == 2 else None
            )
        except ImportError:
            val_auc = None

        self.models[horizon] = LightGBMHorizonResult(
            horizon=horizon,
            model=model,
            feature_cols=list(self.feature_cols),
            best_iteration=int(model.best_iteration_ or self.cfg.n_estimators),
            val_log_loss=val_log_loss,
            val_auc=val_auc,
        )
        logger.info(
            "h=%d: best_iter=%d, val_log_loss=%.4f, val_auc=%s",
            horizon, model.best_iteration_, val_log_loss,
            f"{val_auc:.4f}" if val_auc is not None else "n/a",
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        *,
        meta_cols: tuple[str, ...] = ("ticker",),
    ) -> pd.DataFrame:
        """Long-form predictions: (timestamp, ticker, horizon, mean, std).

        ``df`` должен содержать ``feature_cols`` + ``meta_cols``. Index
        = timestamp.

        std=0 (single model). Для UQ обернуть в bagging-ансамбль.
        """
        if not self.models:
            msg = "fit() not called yet"
            raise RuntimeError(msg)
        rows: list[pd.DataFrame] = []
        for h, result in sorted(self.models.items()):
            X = df[self.feature_cols]
            p = result.model.predict_proba(X)[:, 1]
            block = pd.DataFrame(
                {
                    "timestamp": df.index,
                    "horizon": int(h),
                    "mean": p.astype(np.float32),
                    "std": np.zeros_like(p, dtype=np.float32),
                },
            )
            for col in meta_cols:
                if col in df.columns:
                    block[col] = df[col].to_numpy()
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """Сохранить все per-horizon модели через pickle + manifest.

        LightGBM-booster save_model сохраняет ТОЛЬКО tree-структуру без
        sklearn-обёртки → predict_proba не доступен после load. Pickle
        сохраняет весь LGBMClassifier целиком, что нужно для inference.
        """
        import json
        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)
        for h, result in self.models.items():
            path = output_dir / f"lgbm_h{h}.pkl"
            with path.open("wb") as f:
                pickle.dump(result.model, f)
        manifest = {
            "horizons": list(self.horizons),
            "feature_cols": list(self.feature_cols),
            "models": [
                {
                    "horizon": h,
                    "checkpoint": f"lgbm_h{h}.pkl",
                    "best_iteration": r.best_iteration,
                    "val_log_loss": r.val_log_loss,
                    "val_auc": r.val_auc,
                }
                for h, r in sorted(self.models.items())
            ],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved LightGBM pipeline to %s", output_dir)

    @classmethod
    def load(cls, output_dir: Path) -> "LightGBMPipeline":
        """Восстановить pipeline через pickle.

        SECURITY: pickle небезопасен на untrusted данных. Применять
        только к собственным чекпоинтам.
        """
        import json
        import pickle

        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        horizons = tuple(manifest["horizons"])
        feature_cols = list(manifest["feature_cols"])
        pipeline = cls(horizons=horizons, feature_cols=feature_cols)
        for entry in manifest["models"]:
            with (output_dir / entry["checkpoint"]).open("rb") as f:
                model = pickle.load(f)  # noqa: S301 — own checkpoint
            pipeline.models[int(entry["horizon"])] = LightGBMHorizonResult(
                horizon=int(entry["horizon"]),
                model=model,
                feature_cols=feature_cols,
                best_iteration=int(entry["best_iteration"]),
                val_log_loss=float(entry["val_log_loss"]),
                val_auc=entry.get("val_auc"),
            )
        return pipeline


__all__ = [
    "LightGBMConfig",
    "LightGBMHorizonResult",
    "LightGBMPipeline",
]
