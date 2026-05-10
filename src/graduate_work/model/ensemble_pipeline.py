"""Ensemble Primary: LightGBM + CatBoost + ExtraTrees.

Sprint 2 из дорожной карты (после meta-labeling).

## Идея

Каждый GBM-алгоритм имеет свой **inductive bias**:
- LightGBM: leaf-wise growth, точные пороги
- CatBoost: ordered boosting, robust против overfitting на категориальных
- ExtraTrees: random feature splits, максимальная variance reduction

Trees решают одну задачу с разной геометрией. **Усреднение их
predictions снижает variance** примерно в N (Tukey 1959, Breiman 1996),
особенно на noisy сигнале (нашем).

## Совместимость с MetaLabelingPipeline

`EnsemblePipeline` имеет тот же интерфейс что `LightGBMPipeline`:
- ``fit(train_df, val_df) -> dict``
- ``predict(df) -> pd.DataFrame`` (long-form: timestamp, ticker, horizon, mean, std)

`std` — теперь МЕЖ-моделей disagreement (epistemic UQ). Можно использовать
для дополнительного фильтра «trade only when models agree».

## Аналогия

Если LightGBM — это «решение одного эксперта», то ensemble — это
«согласие 3 экспертов с разным образованием». На шумных сигналах
(AUC 0.55-0.65) разные эксперты ошибаются на разных кейсах →
усреднение даёт более устойчивый ответ.
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


# ---------------------------------------------------------------------------
# Per-model trainers (universal interface: fit + predict_proba)
# ---------------------------------------------------------------------------

def _train_lightgbm(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, y_va: np.ndarray,
    params: dict[str, Any],
    early_stopping_rounds: int,
):
    """LightGBM classifier with early stopping."""
    model = lgb.LGBMClassifier(**params)
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    return model


def _train_catboost(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, y_va: np.ndarray,
    params: dict[str, Any],
    early_stopping_rounds: int,
):
    """CatBoost classifier with ordered boosting."""
    import catboost as cb  # noqa: PLC0415 — lazy import

    full_params = {
        **params,
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": False,
    }
    model = cb.CatBoostClassifier(**full_params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return model


def _train_extratrees(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, y_va: np.ndarray,  # noqa: ARG001 — нет early stop
    params: dict[str, Any],
    early_stopping_rounds: int,  # noqa: ARG001
):
    """ExtraTrees from sklearn (no early stopping)."""
    from sklearn.ensemble import ExtraTreesClassifier  # noqa: PLC0415

    model = ExtraTreesClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


_TRAINERS: dict[str, Any] = {
    "lightgbm": _train_lightgbm,
    "catboost": _train_catboost,
    "extratrees": _train_extratrees,
}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class BaseModelConfig:
    """Конфиг одного базового члена ансамбля.

    ``model_type``: 'lightgbm' | 'catboost' | 'extratrees'.
    ``params``: kwargs передаваемые в model конструктор.
    ``early_stopping_rounds``: для моделей, поддерживающих early stop.
        Игнорируется для extratrees.
    """

    model_type: str
    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 30

    def __post_init__(self) -> None:
        if self.model_type not in _TRAINERS:
            msg = (
                f"unknown model_type {self.model_type!r}; "
                f"supported: {sorted(_TRAINERS)}"
            )
            raise ValueError(msg)


def default_lightgbm_config() -> BaseModelConfig:
    return BaseModelConfig(
        model_type="lightgbm",
        params={
            "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
            "min_child_samples": 200, "reg_lambda": 1.0, "reg_alpha": 0.1,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "random_state": 42, "verbosity": -1, "n_jobs": -1,
        },
    )


def default_catboost_config() -> BaseModelConfig:
    return BaseModelConfig(
        model_type="catboost",
        params={
            "iterations": 200, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3.0, "min_data_in_leaf": 200,
            "random_seed": 42, "thread_count": -1,
        },
    )


def default_extratrees_config() -> BaseModelConfig:
    return BaseModelConfig(
        model_type="extratrees",
        params={
            "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 200,
            "max_features": "sqrt", "random_state": 42, "n_jobs": -1,
        },
    )


# ---------------------------------------------------------------------------
# EnsemblePipeline
# ---------------------------------------------------------------------------

@dataclass
class EnsembleHorizonResult:
    """Результат обучения для одного горизонта в ensemble."""

    horizon: int
    base_models: list  # one trained sklearn-compatible model per BaseModelConfig
    base_types: list[str]
    feature_cols: list[str]
    val_log_loss: float
    val_auc: float | None = None


@dataclass
class EnsemblePipeline:
    """Per-horizon ансамбль: для каждого горизонта тренирует N base
    models и усредняет probabilities.

    Совместим интерфейсом с :class:`LightGBMPipeline`: можно подменить
    primary в :class:`MetaLabelingPipeline` без других изменений.

    ``base_configs`` — список из 1+ :class:`BaseModelConfig`. Каждый
    horizon обучает по одному экземпляру каждой конфигурации.
    """

    horizons: tuple[int, ...]
    feature_cols: list[str]
    base_configs: list[BaseModelConfig] = field(default_factory=list)
    models: dict[int, EnsembleHorizonResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_configs:
            self.base_configs = [
                default_lightgbm_config(),
                default_catboost_config(),
                default_extratrees_config(),
            ]

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
        """Обучить N×H base models. ``sample_weights_train`` пробрасывается
        только в LightGBM; CatBoost/ExtraTrees игнорируют (для простоты)."""
        for h in self.horizons:
            target_col = f"target_h{h}"
            if target_col not in train_df.columns:
                logger.warning("Skipping h=%d: %s not in train_df", h, target_col)
                continue
            self._fit_one_horizon(h, train_df, val_df, target_col)

    def _fit_one_horizon(
        self,
        horizon: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Обучить ВСЕХ base members для одного горизонта."""
        train_mask = train_df[target_col].notna()
        val_mask = val_df[target_col].notna()
        if not train_mask.any() or not val_mask.any():
            logger.warning("h=%d: empty train/val", horizon)
            return
        X_tr = train_df.loc[train_mask, self.feature_cols]
        y_tr = train_df.loc[train_mask, target_col].astype(int).to_numpy()
        X_va = val_df.loc[val_mask, self.feature_cols]
        y_va = val_df.loc[val_mask, target_col].astype(int).to_numpy()

        base_models = []
        base_types = []
        for cfg in self.base_configs:
            trainer = _TRAINERS[cfg.model_type]
            model = trainer(
                X_tr, y_tr, X_va, y_va,
                params=cfg.params,
                early_stopping_rounds=cfg.early_stopping_rounds,
            )
            base_models.append(model)
            base_types.append(cfg.model_type)

        # Ensemble val metrics (averaged probas)
        val_proba = self._ensemble_predict_proba(base_models, X_va)
        val_clipped = np.clip(val_proba, 1e-7, 1 - 1e-7)
        val_log_loss = float(
            -np.mean(y_va * np.log(val_clipped) + (1 - y_va) * np.log(1 - val_clipped)),
        )
        try:
            from sklearn.metrics import roc_auc_score  # noqa: PLC0415
            val_auc = (
                float(roc_auc_score(y_va, val_proba))
                if len(np.unique(y_va)) == 2 else None
            )
        except ImportError:
            val_auc = None

        self.models[horizon] = EnsembleHorizonResult(
            horizon=horizon,
            base_models=base_models,
            base_types=base_types,
            feature_cols=list(self.feature_cols),
            val_log_loss=val_log_loss,
            val_auc=val_auc,
        )
        logger.info(
            "h=%d ensemble [%s]: val_log_loss=%.4f, val_auc=%s",
            horizon, ",".join(base_types), val_log_loss,
            f"{val_auc:.4f}" if val_auc is not None else "n/a",
        )

    @staticmethod
    def _ensemble_predict_proba(
        base_models: list, X: pd.DataFrame,
    ) -> np.ndarray:
        """Среднее positive-class probability по всем base members."""
        probas = np.stack(
            [m.predict_proba(X)[:, 1] for m in base_models], axis=0,
        )
        return probas.mean(axis=0)

    @staticmethod
    def _ensemble_proba_with_std(
        base_models: list, X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (mean, std) positive-class probability через ансамбль.

        std — disagreement между базовыми моделями (epistemic-style UQ).
        """
        probas = np.stack(
            [m.predict_proba(X)[:, 1] for m in base_models], axis=0,
        )
        mean = probas.mean(axis=0)
        # ddof=0: для маленьких N (3 модели) выборочная дисперсия
        # переоценивает variance, биасированная даёт более стабильные числа.
        std = probas.std(axis=0, ddof=0)
        return mean, std

    # ------------------------------------------------------------------
    # Inference (compatible with LightGBMPipeline.predict signature)
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        *,
        meta_cols: tuple[str, ...] = ("ticker",),
    ) -> pd.DataFrame:
        """Long-form predictions: (timestamp, ticker, horizon, mean, std).

        ``mean`` — averaged probability across base models.
        ``std`` — disagreement (epistemic UQ).
        """
        if not self.models:
            msg = "fit() not called yet"
            raise RuntimeError(msg)
        rows: list[pd.DataFrame] = []
        for h, result in sorted(self.models.items()):
            X = df[self.feature_cols]
            mean, std = self._ensemble_proba_with_std(result.base_models, X)
            block = pd.DataFrame({
                "timestamp": df.index,
                "horizon": int(h),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            })
            for col in meta_cols:
                if col in df.columns:
                    block[col] = df[col].to_numpy()
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """Сохранить все per-horizon ensembles через pickle + manifest.

        Каждая base-модель пиклится отдельно: ``ens_h{h}_m{i}_{type}.pkl``.
        Манифест хранит порядок и тип каждой базы — после load() мы должны
        ВОССТАНОВИТЬ predict_proba над теми же фичами в том же порядке.

        SECURITY: pickle небезопасен на untrusted данных. Применять
        только к собственным чекпоинтам.
        """
        import json
        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)
        models_meta: list[dict[str, Any]] = []
        for h, result in sorted(self.models.items()):
            base_files: list[str] = []
            for i, (model, mtype) in enumerate(
                zip(result.base_models, result.base_types, strict=True),
            ):
                fname = f"ens_h{h}_m{i}_{mtype}.pkl"
                with (output_dir / fname).open("wb") as f:
                    pickle.dump(model, f)
                base_files.append(fname)
            models_meta.append({
                "horizon": int(h),
                "base_files": base_files,
                "base_types": list(result.base_types),
                "val_log_loss": float(result.val_log_loss),
                "val_auc": (
                    None if result.val_auc is None else float(result.val_auc)
                ),
            })
        manifest = {
            "horizons": list(self.horizons),
            "feature_cols": list(self.feature_cols),
            "base_configs": [
                {
                    "model_type": cfg.model_type,
                    "params": cfg.params,
                    "early_stopping_rounds": cfg.early_stopping_rounds,
                }
                for cfg in self.base_configs
            ],
            "models": models_meta,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Saved EnsemblePipeline to %s", output_dir)

    @classmethod
    def load(cls, output_dir: Path) -> "EnsemblePipeline":
        """Восстановить pipeline из директории, созданной :meth:`save`."""
        import json
        import pickle

        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        base_configs = [
            BaseModelConfig(
                model_type=str(c["model_type"]),
                params=dict(c.get("params", {})),
                early_stopping_rounds=int(c.get("early_stopping_rounds", 30)),
            )
            for c in manifest.get("base_configs", [])
        ]
        pipeline = cls(
            horizons=tuple(manifest["horizons"]),
            feature_cols=list(manifest["feature_cols"]),
            base_configs=base_configs,
        )
        for entry in manifest["models"]:
            base_models = []
            for fname in entry["base_files"]:
                with (output_dir / fname).open("rb") as f:
                    base_models.append(pickle.load(f))  # noqa: S301 — own checkpoint
            h = int(entry["horizon"])
            pipeline.models[h] = EnsembleHorizonResult(
                horizon=h,
                base_models=base_models,
                base_types=list(entry["base_types"]),
                feature_cols=list(manifest["feature_cols"]),
                val_log_loss=float(entry["val_log_loss"]),
                val_auc=(
                    None if entry.get("val_auc") is None
                    else float(entry["val_auc"])
                ),
            )
        return pipeline


__all__ = [
    "BaseModelConfig",
    "EnsembleHorizonResult",
    "EnsemblePipeline",
    "default_catboost_config",
    "default_extratrees_config",
    "default_lightgbm_config",
]
