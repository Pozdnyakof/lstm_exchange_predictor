"""Тесты EnsemblePipeline: LightGBM + CatBoost + ExtraTrees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.model.ensemble_pipeline import (
    BaseModelConfig,
    EnsemblePipeline,
    default_catboost_config,
    default_extratrees_config,
    default_lightgbm_config,
)


def _make_panel(n: int = 1000, h_list=(6, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетика с реальным сигналом: target=1 если f1+f2 > 0."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
            "ticker": ["SBER"] * n,
        },
        index=idx,
    )
    for h in h_list:
        signal = df["f1"] + df["f2"] + 0.3 * rng.standard_normal(n)
        df[f"target_h{h}"] = (signal > 0).astype(int)
    return df.iloc[: int(0.7 * n)], df.iloc[int(0.7 * n):]


# ---------------------------------------------------------------------------
# BaseModelConfig
# ---------------------------------------------------------------------------

def test_unknown_model_type_raises() -> None:
    try:
        BaseModelConfig(model_type="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad model_type")


def test_default_configs_are_valid() -> None:
    """Default configs всех 3 типов создаются без ошибок."""
    for fn in (default_lightgbm_config, default_catboost_config, default_extratrees_config):
        cfg = fn()
        assert cfg.model_type in {"lightgbm", "catboost", "extratrees"}


# ---------------------------------------------------------------------------
# EnsemblePipeline
# ---------------------------------------------------------------------------

def test_ensemble_default_uses_three_models() -> None:
    """Без явного base_configs → используется 3 default'a (LGB+CB+ET)."""
    pipeline = EnsemblePipeline(
        horizons=(6,), feature_cols=["f1", "f2", "f3"],
    )
    types = {cfg.model_type for cfg in pipeline.base_configs}
    assert types == {"lightgbm", "catboost", "extratrees"}


def test_ensemble_fits_all_horizons() -> None:
    """fit() обучает по 3 base models на каждый горизонт."""
    train, val = _make_panel(n=400, h_list=(6, 12))
    # Сократим params для скорости теста
    fast_cfgs = [
        BaseModelConfig(
            model_type="lightgbm",
            params={"n_estimators": 20, "num_leaves": 7,
                    "learning_rate": 0.1, "verbosity": -1, "n_jobs": -1},
            early_stopping_rounds=5,
        ),
        BaseModelConfig(
            model_type="extratrees",
            params={"n_estimators": 20, "max_depth": 4, "n_jobs": -1, "random_state": 0},
        ),
    ]
    pipeline = EnsemblePipeline(
        horizons=(6, 12),
        feature_cols=["f1", "f2", "f3"],
        base_configs=fast_cfgs,
    )
    pipeline.fit(train, val)
    assert set(pipeline.models.keys()) == {6, 12}
    for h in (6, 12):
        result = pipeline.models[h]
        assert len(result.base_models) == 2
        assert result.base_types == ["lightgbm", "extratrees"]


def test_ensemble_finds_synthetic_signal() -> None:
    """На синтетике AUC > 0.7 (сильный сигнал)."""
    train, val = _make_panel(n=2000, h_list=(6,))
    pipeline = EnsemblePipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        base_configs=[
            BaseModelConfig(
                model_type="lightgbm",
                params={"n_estimators": 50, "num_leaves": 15,
                        "learning_rate": 0.05, "verbosity": -1, "n_jobs": -1},
                early_stopping_rounds=10,
            ),
            BaseModelConfig(
                model_type="extratrees",
                params={"n_estimators": 50, "max_depth": 6, "n_jobs": -1, "random_state": 0},
            ),
        ],
    )
    pipeline.fit(train, val)
    assert pipeline.models[6].val_auc is not None
    assert pipeline.models[6].val_auc > 0.7


def test_ensemble_predict_returns_long_form_with_std() -> None:
    """predict() — long-form, std отражает disagreement."""
    train, val = _make_panel(n=400, h_list=(6,))
    pipeline = EnsemblePipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        base_configs=[
            BaseModelConfig(
                model_type="lightgbm",
                params={"n_estimators": 20, "verbosity": -1, "n_jobs": -1},
                early_stopping_rounds=5,
            ),
            BaseModelConfig(
                model_type="extratrees",
                params={"n_estimators": 20, "max_depth": 4, "n_jobs": -1, "random_state": 0},
            ),
        ],
    )
    pipeline.fit(train, val)
    preds = pipeline.predict(val)
    expected = {"timestamp", "horizon", "mean", "std", "ticker"}
    assert expected.issubset(preds.columns)
    # std неотрицательный, mean в [0, 1]
    assert (preds["mean"] >= 0).all() and (preds["mean"] <= 1).all()
    assert (preds["std"] >= 0).all()


def test_ensemble_predict_before_fit_raises() -> None:
    pipeline = EnsemblePipeline(
        horizons=(6,), feature_cols=["f1"],
    )
    df = pd.DataFrame(
        {"f1": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    try:
        pipeline.predict(df)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_ensemble_disagreement_is_zero_for_identical_models() -> None:
    """Если все base modles идентичны → std=0."""
    train, val = _make_panel(n=300, h_list=(6,))
    same_cfg = BaseModelConfig(
        model_type="lightgbm",
        params={"n_estimators": 10, "num_leaves": 7,
                "verbosity": -1, "n_jobs": -1, "random_state": 42},
        early_stopping_rounds=3,
    )
    pipeline = EnsemblePipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        base_configs=[same_cfg, same_cfg],  # 2 одинаковых
    )
    pipeline.fit(train, val)
    preds = pipeline.predict(val)
    # Идентичные модели → одинаковые predictions → std ≈ 0
    assert (preds["std"] < 1e-6).all()
