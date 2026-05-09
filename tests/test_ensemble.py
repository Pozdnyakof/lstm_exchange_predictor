"""Тесты Deep Ensemble: член-ансамбля диверсифицируется по сидам, predict
агрегирует корректно."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from graduate_work.config import (
    DataConfig,
    ModelConfig,
    TradingConfig,
    TrainingConfig,
)
from graduate_work.training import DeepEnsembleTrainer, ensemble_predict


class _TinyMLP(nn.Module):
    """Минимальная сеть, чтобы тест не зависел от больших архитектур."""

    def __init__(self, input_dim: int, num_horizons: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * 8, 16),
            nn.GELU(),
            nn.Linear(16, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_arrays(n: int = 64, t: int = 8, f: int = 4, h: int = 2) -> dict:
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((n, t, f)).astype(np.float32),
        "y": (rng.standard_normal((n, h)) > 0).astype(np.float32),
        "timestamp": np.arange(n).astype("datetime64[ns]"),
        "ticker": np.array(["TST"] * n, dtype=object),
    }


def test_ensemble_size_validation() -> None:
    """ensemble_size < 2 запрещён — иначе нет смысла в UQ."""
    factory = lambda seed: _TinyMLP(4, 2, seed)
    try:
        DeepEnsembleTrainer(factory, TrainingConfig(epochs=1), ensemble_size=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError on ensemble_size=1")


def test_ensemble_predict_shape() -> None:
    """``ensemble_predict`` даёт (N, H) для mean/std."""
    members = [_TinyMLP(4, 2, seed=s) for s in (1, 2, 3)]
    x = np.random.randn(10, 8, 4).astype(np.float32)
    mean, std = ensemble_predict(
        members, x, batch_size=4, device="cpu", apply_sigmoid=True,
    )
    assert mean.shape == (10, 2)
    assert std.shape == (10, 2)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()


def test_ensemble_predict_diverges_across_seeds() -> None:
    """3 модели с разными сидами → std > 0 хотя бы где-то."""
    members = [_TinyMLP(4, 2, seed=s) for s in (1, 7, 42)]
    x = np.random.randn(20, 8, 4).astype(np.float32)
    _, std = ensemble_predict(members, x, device="cpu")
    assert (std > 0).all()


def test_ensemble_fit_runs_end_to_end(tmp_path) -> None:
    """Полный fit() ансамбля: M моделей сохраняются в чекпоинты + manifest."""
    arrays = _make_arrays(n=32, t=8, f=4, h=2)
    factory = lambda seed: _TinyMLP(4, 2, seed)
    ens = DeepEnsembleTrainer(
        factory,
        TrainingConfig(
            epochs=1, batch_size=8, mc_passes=2,
            use_swa=False, scheduler="none",
        ),
        ensemble_size=2,
        data_cfg=DataConfig(mode="classification", horizons=(1, 2)),
        trading_cfg=TradingConfig(),
        device="cpu",
        base_seed=0,
    )
    history = ens.fit(arrays, arrays, checkpoint_dir=tmp_path)
    assert len(ens.members) == 2
    assert len(history.checkpoint_paths) == 2
    assert (tmp_path / "ensemble_manifest.json").exists()


def test_ensemble_load_from_dir(tmp_path) -> None:
    """Сохранённый ансамбль загружается без ошибок и predict работает."""
    arrays = _make_arrays(n=32, t=8, f=4, h=2)
    factory = lambda seed: _TinyMLP(4, 2, seed)
    base_cfg = dict(
        training_cfg=TrainingConfig(
            epochs=1, batch_size=8, mc_passes=2,
            use_swa=False, scheduler="none",
        ),
        ensemble_size=2,
        data_cfg=DataConfig(mode="classification", horizons=(1, 2)),
        trading_cfg=TradingConfig(),
        device="cpu",
        base_seed=0,
    )
    ens = DeepEnsembleTrainer(factory, **base_cfg)
    ens.fit(arrays, arrays, checkpoint_dir=tmp_path)

    fresh = DeepEnsembleTrainer(factory, **base_cfg)
    fresh.load_from_dir(tmp_path)
    mean, std = ensemble_predict(
        fresh.members, arrays["x"], device="cpu", apply_sigmoid=True,
    )
    assert mean.shape == (32, 2)
    assert (std >= 0).all()
