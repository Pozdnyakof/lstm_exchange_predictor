"""Тесты serving-слоя: артефакт-пакет, live-features (с моком MOEX), инференс."""

from __future__ import annotations

import dataclasses as _dc
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from graduate_work.config import (
    DataConfig,
    ModelConfig,
    ServingConfig,
    TradingConfig,
)
from graduate_work.features import StandardScaler
from graduate_work.model import ConvLstmRegressor
from graduate_work.serving import (
    InferenceService,
    LiveFeatureBuilder,
    LoadedModel,
    ModelMeta,
    load_artifact,
    save_artifact,
)
from graduate_work.serving.artifact import now_iso


def _build_meta(feature_cols: list[str], horizons: tuple[int, ...]) -> ModelMeta:
    cfg = ModelConfig(
        conv_channels=8, conv_kernel=3,
        lstm_hidden=16, lstm_layers=1,
        fc_hidden=16, dropout=0.3,
    )
    return ModelMeta(
        feature_cols=feature_cols,
        target_cols=[f"target_h{h}" for h in horizons],
        horizons=list(horizons),
        window_size=20,
        num_features=len(feature_cols),
        num_horizons=len(horizons),
        model_config=_dc.asdict(cfg),
        training_date=now_iso(),
        tickers=["SBER", "GAZP"],
    )


def _make_dummy_model(meta: ModelMeta) -> ConvLstmRegressor:
    return ConvLstmRegressor(
        input_dim=meta.num_features,
        num_horizons=meta.num_horizons,
        cfg=meta.model_cfg(),
    )


def test_save_and_load_artifact(tmp_path: Path) -> None:
    feature_cols = ["log_return", "sma_5_rel", "rsi_14"]
    meta = _build_meta(feature_cols, horizons=(1, 5))
    model = _make_dummy_model(meta)
    scaler = StandardScaler()
    df = pd.DataFrame({c: np.random.randn(100) for c in feature_cols})
    scaler.fit(df, feature_cols)

    save_artifact(model, scaler, meta, tmp_path)
    loaded = load_artifact(tmp_path, device="cpu")

    assert loaded.meta.feature_cols == feature_cols
    assert loaded.meta.num_horizons == 2
    # Веса должны совпасть до байта.
    for k, v in model.state_dict().items():
        assert torch.allclose(v, loaded.model.state_dict()[k])


def _fake_moex_response(rows: int = 5000, *, ticker: str = "SBER") -> pd.DataFrame:
    """Имитация ответа MOEX ISS: 1-минутные бары в окне торговой сессии.

    Генерим непрерывный диапазон, потом оставляем только пн-пт 07:00-15:45 UTC.
    """
    idx = pd.date_range("2024-01-02 07:00", periods=rows, freq="1min", tz="UTC")
    mask = (
        (idx.dayofweek < 5)
        & (idx.time >= pd.Timestamp("07:00").time())
        & (idx.time <= pd.Timestamp("15:45").time())
    )
    idx = idx[mask]
    n = len(idx)
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(n).cumsum() * 0.05
    return pd.DataFrame(
        {
            "open": close + rng.standard_normal(n) * 0.01,
            "high": close + rng.uniform(0.01, 0.1, n),
            "low": close - rng.uniform(0.01, 0.1, n),
            "close": close,
            "volume": rng.integers(100, 500, n).astype(float),
            "ticker": ticker,
        },
        index=idx,
    )


def test_live_feature_builder_window_shape(tmp_path: Path) -> None:
    feature_cols = ["log_return", "sma_5_rel", "rsi_14"]
    meta = _build_meta(feature_cols, horizons=(1, 5))
    model = _make_dummy_model(meta)
    scaler = StandardScaler()
    df = pd.DataFrame({c: np.random.randn(200) for c in feature_cols})
    scaler.fit(df, feature_cols)

    loaded = LoadedModel(
        model=model.eval(),
        scaler=scaler,
        meta=meta,
        device=torch.device("cpu"),
    )
    data_cfg = DataConfig()
    serving_cfg = ServingConfig(cache_ttl_sec=10, moex_request_pause=0.0)
    builder = LiveFeatureBuilder(loaded, data_cfg, serving_cfg)

    fake = _fake_moex_response(rows=5000)
    with patch("graduate_work.serving.live_features.moex_iss.fetch_ticker", return_value=fake):
        win = builder.get_window("SBER", force_refresh=True)

    assert win is not None
    assert win.x.shape == (1, meta.window_size, meta.num_features)
    assert win.last_close > 0


def test_inference_service_returns_forecast(tmp_path: Path) -> None:
    feature_cols = ["log_return", "sma_5_rel", "rsi_14"]
    meta = _build_meta(feature_cols, horizons=(1, 5))
    scaler = StandardScaler()
    df = pd.DataFrame({c: np.random.randn(200) for c in feature_cols})
    scaler.fit(df, feature_cols)

    loaded = LoadedModel(
        model=_make_dummy_model(meta).eval(),
        scaler=scaler,
        meta=meta,
        device=torch.device("cpu"),
    )
    data_cfg = DataConfig()
    serving_cfg = ServingConfig(cache_ttl_sec=10, moex_request_pause=0.0)
    trading_cfg = TradingConfig(
        min_expected_return=0.0001, max_uncertainty=10.0,
    )
    service = InferenceService(loaded, data_cfg, trading_cfg, serving_cfg)

    fakes = {
        "SBER": _fake_moex_response(rows=5000, ticker="SBER"),
        "GAZP": _fake_moex_response(rows=5000, ticker="GAZP"),
    }
    with patch(
        "graduate_work.serving.live_features.moex_iss.fetch_ticker",
        side_effect=lambda ticker, **_: fakes[ticker],
    ):
        forecasts = service.predict_all(force_refresh=True)

    assert len(forecasts) == 2
    tickers = {f.ticker for f in forecasts}
    assert tickers == {"SBER", "GAZP"}
    for f in forecasts:
        assert len(f.all_horizons) == meta.num_horizons
        assert f.uncertainty >= 0.0


@pytest.mark.parametrize("missing", ["model_artifact.pt", "scaler.json", "meta.json"])
def test_load_artifact_raises_on_missing_file(tmp_path: Path, missing: str) -> None:
    feature_cols = ["a", "b"]
    meta = _build_meta(feature_cols, horizons=(1,))
    model = _make_dummy_model(meta)
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({c: np.random.randn(50) for c in feature_cols}), feature_cols)
    save_artifact(model, scaler, meta, tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path, device="cpu")
