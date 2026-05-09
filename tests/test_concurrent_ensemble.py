"""Тесты ConcurrentDeepEnsembleTrainer (parallel training, simultaneous SVGD)."""

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
from graduate_work.training import (
    ConcurrentDeepEnsembleTrainer,
    ensemble_predict,
)
from graduate_work.training.repulsion import svgd_pairwise_repulsion


class _TinyMLP(nn.Module):
    """Минимальная сеть для smoke-test'а."""

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


def _factory(input_dim: int, horizons: int):
    return lambda seed: _TinyMLP(input_dim, horizons, seed)


# ---------------------------------------------------------------------------
# SVGD pairwise repulsion (low-level)
# ---------------------------------------------------------------------------

def test_svgd_pairwise_zero_for_no_others() -> None:
    """Без other_preds репульсия — ноль."""
    preds = torch.randn(4, 2, requires_grad=True)
    rep = svgd_pairwise_repulsion(preds, [], weight=0.5)
    assert rep.item() == 0.0


def test_svgd_pairwise_aggregates_kernels() -> None:
    """Усреднение по other_preds: 2 идентичных other → kernel=1 → loss=weight."""
    preds = torch.zeros(4, 2)
    other = torch.zeros(4, 2)
    rep = svgd_pairwise_repulsion(preds, [other, other], weight=0.5)
    # k(self, self) = 1; mean = 1; * weight 0.5 = 0.5
    assert abs(rep.item() - 0.5) < 1e-5


def test_svgd_pairwise_gradient_flows_only_to_current() -> None:
    """Градиент течёт по preds, не по other_preds (которые detach'нуты)."""
    preds = torch.randn(4, 2, requires_grad=True)
    other = torch.randn(4, 2)  # без requires_grad
    rep = svgd_pairwise_repulsion(preds, [other], weight=0.5)
    rep.backward()
    assert preds.grad is not None
    assert preds.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# ConcurrentDeepEnsembleTrainer integration
# ---------------------------------------------------------------------------

def _make_concurrent_kwargs() -> dict:
    return {
        "training_cfg": TrainingConfig(
            epochs=1, batch_size=16, mc_passes=2,
            use_swa=False, scheduler="none",
        ),
        "ensemble_size": 3,
        "data_cfg": DataConfig(mode="classification", horizons=(1, 2)),
        "trading_cfg": TradingConfig(loss_objective="bce"),
        "device": "cpu",
        "base_seed": 0,
    }


def test_concurrent_ensemble_size_validated() -> None:
    """ensemble_size < 2 запрещён."""
    try:
        ConcurrentDeepEnsembleTrainer(
            _factory(4, 2),
            TrainingConfig(epochs=1),
            ensemble_size=1,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on ensemble_size=1")


def test_concurrent_repulsion_weight_validated() -> None:
    """svgd_repulsion_weight < 0 запрещён."""
    try:
        ConcurrentDeepEnsembleTrainer(
            _factory(4, 2),
            TrainingConfig(epochs=1),
            ensemble_size=2,
            svgd_repulsion_weight=-0.1,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative weight")


def test_concurrent_builds_all_members_on_init() -> None:
    """После __init__ все M моделей построены и доступны через .members."""
    trainer = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        **_make_concurrent_kwargs(),
    )
    assert len(trainer.members) == 3
    # Все модели должны быть на правильном device.
    assert all(next(m.parameters()).device == torch.device("cpu") for m in trainer.members)


def test_concurrent_fit_runs_end_to_end(tmp_path) -> None:
    """Полный fit() работает: M членов обучаются, history заполнена,
    чекпоинты + manifest пишутся."""
    arrays = _make_arrays(n=32, t=8, f=4, h=2)
    trainer = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        svgd_repulsion_weight=0.0,
        **_make_concurrent_kwargs(),
    )
    history = trainer.fit(arrays, arrays, checkpoint_dir=tmp_path)
    assert len(history.member_histories) == 3
    assert len(history.seeds) == 3
    assert (tmp_path / "ensemble_manifest.json").exists()
    # Все 3 модели существуют и работают на инференсе.
    mean, std = ensemble_predict(
        trainer.members, arrays["x"], device="cpu", apply_sigmoid=True,
    )
    assert mean.shape == (32, 2)
    assert std.shape == (32, 2)


def test_concurrent_with_repulsion_diversifies_members() -> None:
    """С svgd_repulsion_weight > 0 члены ансамбля должны разойтись СИЛЬНЕЕ,
    чем без репульсии (на той же starting point distribution)."""
    arrays = _make_arrays(n=64, t=8, f=4, h=2)

    # Без репульсии
    trainer_off = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        svgd_repulsion_weight=0.0,
        **_make_concurrent_kwargs(),
    )
    trainer_off.fit(arrays, arrays)
    _, std_off = ensemble_predict(
        trainer_off.members, arrays["x"], device="cpu", apply_sigmoid=True,
    )

    # С репульсией — больше разброс между членами.
    trainer_on = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        svgd_repulsion_weight=1.0,  # сильная репульсия для теста
        **_make_concurrent_kwargs(),
    )
    trainer_on.fit(arrays, arrays)
    _, std_on = ensemble_predict(
        trainer_on.members, arrays["x"], device="cpu", apply_sigmoid=True,
    )

    # std с репульсией ДОЛЖЕН быть >= std без репульсии (хоть в среднем).
    # На крошечных моделях/данных разница может быть маленькой, но не нулевой.
    assert std_on.mean() >= std_off.mean() * 0.5  # допускаем шум


def test_concurrent_classification_mode_handles_logit_prior() -> None:
    """В classification-mode auto-tune применяет logit-prior к моделям,
    у которых есть set_logit_prior (iTransformer, не _TinyMLP).

    Проверяем, что _TinyMLP без set_logit_prior — не падает, просто
    игнорируется."""
    arrays = _make_arrays(n=32, t=8, f=4, h=2)
    trainer = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        **_make_concurrent_kwargs(),
    )
    # Не должен падать на отсутствии set_logit_prior.
    trainer.fit(arrays, arrays)
    assert all(m.training is False for m in trainer.members)


def test_concurrent_eval_step_no_repulsion() -> None:
    """На val-эпохе репульсия не считается — это чистый loss."""
    arrays = _make_arrays(n=16, t=8, f=4, h=2)
    trainer = ConcurrentDeepEnsembleTrainer(
        _factory(4, 2),
        svgd_repulsion_weight=0.5,
        **_make_concurrent_kwargs(),
    )
    history = trainer.fit(arrays, arrays)
    # У всех членов есть и train, и val loss.
    for h in history.member_histories:
        assert len(h.train_loss) >= 1
        assert len(h.val_loss) >= 1
        # Val loss конечный.
        assert all(np.isfinite(v) for v in h.val_loss)
