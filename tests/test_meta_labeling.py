"""Тесты meta-labeling: OOF + meta targets + end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.model.lgbm_pipeline import LightGBMConfig
from graduate_work.model.meta_labeling import (
    MetaLabelingPipeline,
    add_primary_predictions_wide,
    build_meta_targets,
    compute_lgbm_oof_per_horizon,
)


def _make_panel(n: int = 1000, h_list=(6, 12)) -> pd.DataFrame:
    """Синтетический panel с реальным сигналом + lr-targets."""
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
        # Synthetic log returns: signal scale + noise
        df[f"lr_h{h}"] = (
            signal * 0.005 + 0.001 * rng.standard_normal(n)
        ).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# build_meta_targets
# ---------------------------------------------------------------------------

def test_meta_target_is_one_when_lr_exceeds_cost() -> None:
    df = pd.DataFrame({
        "lr_h6": [0.005, 0.001, -0.002, 0.010, 0.0],
    }, index=pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"))
    out = build_meta_targets(
        df, horizons=(6,),
        cost_per_trade=0.001, profit_multiplier=2.0,  # threshold = 0.002
    )
    expected = [1.0, 0.0, 0.0, 1.0, 0.0]
    assert list(out["meta_target_h6"]) == expected


def test_meta_target_is_nan_when_lr_is_nan() -> None:
    df = pd.DataFrame({
        "lr_h6": [0.005, np.nan, 0.001],
    }, index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"))
    out = build_meta_targets(df, horizons=(6,), cost_per_trade=0.001)
    assert pd.isna(out["meta_target_h6"].iloc[1])


def test_meta_target_skips_horizons_without_lr() -> None:
    df = pd.DataFrame({"lr_h6": [0.005]}, index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"))
    out = build_meta_targets(df, horizons=(6, 12, 24))
    assert "meta_target_h6" in out.columns
    assert "meta_target_h12" not in out.columns


# ---------------------------------------------------------------------------
# compute_lgbm_oof_per_horizon
# ---------------------------------------------------------------------------

def test_oof_returns_dataframe_with_correct_columns() -> None:
    df = _make_panel(n=500, h_list=(6, 12))
    oof = compute_lgbm_oof_per_horizon(
        df, feature_cols=["f1", "f2", "f3"], horizons=(6, 12),
        cfg=LightGBMConfig(n_estimators=10, early_stopping_rounds=5),
        n_splits=3,
    )
    assert "primary_h6" in oof.columns
    assert "primary_h12" in oof.columns
    assert len(oof) == len(df)


def test_oof_first_fold_is_nan_no_oof_for_initial_data() -> None:
    """TimeSeriesSplit: первый fold обучается на начале → нет OOF
    для самых первых строк."""
    df = _make_panel(n=500, h_list=(6,))
    oof = compute_lgbm_oof_per_horizon(
        df, feature_cols=["f1", "f2", "f3"], horizons=(6,),
        cfg=LightGBMConfig(n_estimators=5, early_stopping_rounds=3),
        n_splits=3,
    )
    # Первая порция должна быть NaN (использовалась только для обучения).
    # С n_splits=3 первый train fold = 1/4 данных, первые ~125 строк.
    assert oof["primary_h6"].iloc[:50].isna().all()
    # Последняя часть точно не NaN.
    assert oof["primary_h6"].iloc[-50:].notna().any()


def test_oof_predictions_in_unit_interval() -> None:
    df = _make_panel(n=500, h_list=(6,))
    oof = compute_lgbm_oof_per_horizon(
        df, feature_cols=["f1", "f2", "f3"], horizons=(6,),
        cfg=LightGBMConfig(n_estimators=10, early_stopping_rounds=5),
        n_splits=3,
    )
    valid = oof["primary_h6"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


# ---------------------------------------------------------------------------
# add_primary_predictions_wide
# ---------------------------------------------------------------------------

def test_add_primary_predictions_wide_merges_correctly() -> None:
    """Long-form primary preds → wide-merge in df."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "feat1": [1.0, 2.0, 3.0],
    }, index=pd.to_datetime(
        ["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:00"], utc=True,
    ))
    df.index.name = "begin"
    primary_long = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:00"],
            utc=True,
        ),
        "ticker": ["A", "A", "B"],
        "horizon": [6, 6, 6],
        "mean": [0.6, 0.7, 0.5],
        "std": [0.0, 0.0, 0.0],
    })
    out = add_primary_predictions_wide(df, primary_long, horizons=(6,))
    assert "primary_h6" in out.columns
    # Verify merged values (timestamps have UTC, do search by index)
    a_first = out[(out["ticker"] == "A")].sort_index().iloc[0]["primary_h6"]
    assert abs(a_first - 0.6) < 1e-6


def test_add_primary_predictions_wide_handles_empty() -> None:
    df = pd.DataFrame({"ticker": ["A"], "feat": [1.0]},
                     index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"))
    out = add_primary_predictions_wide(df, pd.DataFrame(), horizons=(6,))
    assert "primary_h6" in out.columns
    assert out["primary_h6"].isna().all()


# ---------------------------------------------------------------------------
# MetaLabelingPipeline end-to-end
# ---------------------------------------------------------------------------

def test_meta_pipeline_fits_and_predicts() -> None:
    """End-to-end: fit on train+val, predict on test."""
    train = _make_panel(n=500, h_list=(6,))
    val = _make_panel(n=100, h_list=(6,))
    test = _make_panel(n=100, h_list=(6,))
    pipeline = MetaLabelingPipeline(
        horizons=(6,),
        primary_features=["f1", "f2", "f3"],
        meta_features=["f1", "primary_h6"],  # meta использует только эту фичу + primary
        primary_cfg=LightGBMConfig(n_estimators=20, early_stopping_rounds=5),
        meta_cfg=LightGBMConfig(n_estimators=20, early_stopping_rounds=5),
        cost_per_trade=0.001,
        n_oof_splits=3,
    )
    summary = pipeline.fit(train, val)
    assert "primary" in summary
    assert "meta" in summary
    assert 6 in summary["primary"]
    assert 6 in summary["meta"]

    primary_preds, meta_preds = pipeline.predict(test)
    assert len(primary_preds) == len(test)
    assert len(meta_preds) == len(test)
    # Both probabilities in [0, 1]
    assert (primary_preds["mean"] >= 0).all()
    assert (primary_preds["mean"] <= 1).all()
    assert (meta_preds["mean"] >= 0).all()
    assert (meta_preds["mean"] <= 1).all()


def test_meta_pipeline_predict_before_fit_raises() -> None:
    pipeline = MetaLabelingPipeline(
        horizons=(6,),
        primary_features=["f1"],
        meta_features=["primary_h6"],
        primary_cfg=LightGBMConfig(n_estimators=5),
        meta_cfg=LightGBMConfig(n_estimators=5),
    )
    df = _make_panel(n=10, h_list=(6,))
    try:
        pipeline.predict(df)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")
