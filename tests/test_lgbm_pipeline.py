"""Тесты LightGBMPipeline: fit, predict, save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.model.lgbm_pipeline import (
    LightGBMConfig,
    LightGBMPipeline,
)


def _make_synthetic_panel(n: int = 1000, h_list=(6, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетический panel с реальным сигналом: target=1 если f1+f2 > 0."""
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
        # Сигнал — линейная комбинация фич с шумом.
        signal = df["f1"] + df["f2"] + 0.3 * rng.standard_normal(n)
        df[f"target_h{h}"] = (signal > 0).astype(int)
    train_df = df.iloc[: int(0.7 * n)]
    val_df = df.iloc[int(0.7 * n):]
    return train_df, val_df


def test_pipeline_fits_all_horizons() -> None:
    """fit() обучает по одной модели на каждый горизон."""
    train_df, val_df = _make_synthetic_panel(n=500, h_list=(6, 12))
    pipeline = LightGBMPipeline(
        horizons=(6, 12),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=20, early_stopping_rounds=5),
    )
    pipeline.fit(train_df, val_df)
    assert set(pipeline.models.keys()) == {6, 12}
    for h, result in pipeline.models.items():
        assert result.horizon == h
        assert result.best_iteration > 0


def test_pipeline_finds_synthetic_signal() -> None:
    """На сильном синтетическом сигнале AUC должен быть значимо > 0.5."""
    train_df, val_df = _make_synthetic_panel(n=2000, h_list=(6,))
    pipeline = LightGBMPipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=100, early_stopping_rounds=10),
    )
    pipeline.fit(train_df, val_df)
    assert pipeline.models[6].val_auc is not None
    assert pipeline.models[6].val_auc > 0.7  # сигнал f1+f2>0 явный


def test_pipeline_predict_returns_long_form() -> None:
    """predict() возвращает long-form (timestamp, ticker, horizon, mean, std)."""
    train_df, val_df = _make_synthetic_panel(n=500, h_list=(6, 12))
    pipeline = LightGBMPipeline(
        horizons=(6, 12),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=20, early_stopping_rounds=5),
    )
    pipeline.fit(train_df, val_df)
    preds = pipeline.predict(val_df)
    expected = {"timestamp", "horizon", "mean", "std", "ticker"}
    assert expected.issubset(preds.columns)
    assert len(preds) == len(val_df) * 2  # 2 horizons


def test_pipeline_predict_before_fit_raises() -> None:
    pipeline = LightGBMPipeline(horizons=(6,), feature_cols=["f1"])
    df = pd.DataFrame({"f1": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="5min"))
    try:
        pipeline.predict(df)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_pipeline_save_load_roundtrip(tmp_path) -> None:
    """save() + load() сохраняет manifest и веса; predict работает."""
    train_df, val_df = _make_synthetic_panel(n=500, h_list=(6,))
    pipeline = LightGBMPipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=30, early_stopping_rounds=5),
    )
    pipeline.fit(train_df, val_df)
    original_preds = pipeline.predict(val_df)

    pipeline.save(tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "lgbm_h6.pkl").exists()

    loaded = LightGBMPipeline.load(tmp_path)
    assert loaded.horizons == (6,)
    assert loaded.feature_cols == ["f1", "f2", "f3"]
    loaded_preds = loaded.predict(val_df)
    # Predictions должны совпадать с оригиналом (детерминированные).
    np.testing.assert_allclose(
        original_preds["mean"].values, loaded_preds["mean"].values,
        rtol=1e-5,
    )


def test_pipeline_with_sample_weights() -> None:
    """sample_weights передаются в LightGBM без ошибок."""
    train_df, val_df = _make_synthetic_panel(n=500, h_list=(6,))
    weights = np.ones(len(train_df), dtype=np.float32) * 0.5
    pipeline = LightGBMPipeline(
        horizons=(6,),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=20, early_stopping_rounds=5),
    )
    pipeline.fit(train_df, val_df, sample_weights_train=weights)
    assert 6 in pipeline.models


def test_pipeline_skips_horizon_without_target() -> None:
    """Если target_h{h} отсутствует — горизонт пропускается."""
    train_df, val_df = _make_synthetic_panel(n=300, h_list=(6,))
    # Просим обучить h=12, которого нет в данных.
    pipeline = LightGBMPipeline(
        horizons=(6, 12),
        feature_cols=["f1", "f2", "f3"],
        cfg=LightGBMConfig(n_estimators=15, early_stopping_rounds=5),
    )
    pipeline.fit(train_df, val_df)
    assert 6 in pipeline.models
    assert 12 not in pipeline.models  # пропущен
