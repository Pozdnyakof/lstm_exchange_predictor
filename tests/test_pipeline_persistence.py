"""Тесты save/load для EnsemblePipeline и MetaLabelingPipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graduate_work.model.ensemble_pipeline import (
    BaseModelConfig,
    EnsemblePipeline,
)
from graduate_work.model.lgbm_pipeline import LightGBMConfig
from graduate_work.model.meta_labeling import MetaLabelingPipeline


def _make_panel(n: int = 400, h_list=(6,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетика с реальным сигналом + lr_h{h} для meta-labeling."""
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
        df[f"lr_h{h}"] = 0.005 * signal + 0.001 * rng.standard_normal(n)
    cut = int(0.7 * n)
    return df.iloc[:cut], df.iloc[cut:]


# ---------------------------------------------------------------------------
# EnsemblePipeline.save/load
# ---------------------------------------------------------------------------

def test_ensemble_save_load_roundtrip(tmp_path: Path) -> None:
    """save → load → predict выдаёт идентичные предсказания."""
    train, val = _make_panel(n=400, h_list=(6,))
    fast_cfgs = [
        BaseModelConfig(
            model_type="lightgbm",
            params={"n_estimators": 20, "num_leaves": 7,
                    "verbosity": -1, "n_jobs": -1, "random_state": 42},
            early_stopping_rounds=5,
        ),
        BaseModelConfig(
            model_type="extratrees",
            params={"n_estimators": 20, "max_depth": 4,
                    "n_jobs": -1, "random_state": 42},
        ),
    ]
    pipeline = EnsemblePipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        base_configs=fast_cfgs,
    )
    pipeline.fit(train, val)
    preds_before = pipeline.predict(val)

    save_dir = tmp_path / "ensemble_ckpt"
    pipeline.save(save_dir)
    assert (save_dir / "manifest.json").exists()

    restored = EnsemblePipeline.load(save_dir)
    preds_after = restored.predict(val)

    np.testing.assert_array_equal(
        preds_before["mean"].to_numpy(),
        preds_after["mean"].to_numpy(),
    )
    np.testing.assert_array_equal(
        preds_before["std"].to_numpy(),
        preds_after["std"].to_numpy(),
    )


def test_ensemble_load_preserves_base_types(tmp_path: Path) -> None:
    """После load порядок и типы базовых моделей совпадают."""
    train, val = _make_panel(n=300, h_list=(6,))
    cfgs = [
        BaseModelConfig(
            model_type="lightgbm",
            params={"n_estimators": 10, "verbosity": -1, "n_jobs": -1},
            early_stopping_rounds=3,
        ),
        BaseModelConfig(
            model_type="extratrees",
            params={"n_estimators": 10, "max_depth": 3, "n_jobs": -1},
        ),
    ]
    pipeline = EnsemblePipeline(
        horizons=(6,), feature_cols=["f1", "f2", "f3"], base_configs=cfgs,
    )
    pipeline.fit(train, val)

    save_dir = tmp_path / "ck"
    pipeline.save(save_dir)
    restored = EnsemblePipeline.load(save_dir)
    assert restored.models[6].base_types == ["lightgbm", "extratrees"]
    assert restored.feature_cols == ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# MetaLabelingPipeline.save/load
# ---------------------------------------------------------------------------

def test_meta_labeling_save_load_roundtrip(tmp_path: Path) -> None:
    """save → load → predict даёт те же primary+meta предсказания."""
    train, val = _make_panel(n=400, h_list=(6,))
    fast_cfg = LightGBMConfig(
        n_estimators=20, num_leaves=7, learning_rate=0.1,
        early_stopping_rounds=5,
    )
    primary_features = ["f1", "f2", "f3"]
    meta_features = ["f1", "primary_h6"]
    pipeline = MetaLabelingPipeline(
        horizons=(6,),
        primary_features=primary_features,
        meta_features=meta_features,
        primary_cfg=fast_cfg,
        meta_cfg=fast_cfg,
        cost_per_trade=0.001,
        profit_multiplier=2.0,
        n_oof_splits=2,
    )
    pipeline.fit(train, val)
    primary_before, meta_before = pipeline.predict(val)

    save_dir = tmp_path / "meta_ckpt"
    pipeline.save(save_dir)
    assert (save_dir / "meta_labeling.json").exists()
    assert (save_dir / "primary" / "manifest.json").exists()
    assert (save_dir / "meta" / "manifest.json").exists()

    restored = MetaLabelingPipeline.load(save_dir)
    primary_after, meta_after = restored.predict(val)
    np.testing.assert_array_equal(
        primary_before["mean"].to_numpy(),
        primary_after["mean"].to_numpy(),
    )
    np.testing.assert_array_equal(
        meta_before["mean"].to_numpy(),
        meta_after["mean"].to_numpy(),
    )


def test_meta_labeling_save_before_fit_raises(tmp_path: Path) -> None:
    pipeline = MetaLabelingPipeline(
        horizons=(6,),
        primary_features=["f1"],
        meta_features=["f1"],
        primary_cfg=LightGBMConfig(),
        meta_cfg=LightGBMConfig(),
    )
    try:
        pipeline.save(tmp_path / "x")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")
