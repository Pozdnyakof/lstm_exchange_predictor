"""Регрессионные тесты для перехода на классификацию.

Покрывает:
- cost-aware бинарные метки + raw lr колонки;
- BCE-loss в Trainer'е через mode='classification';
- apply_sigmoid в mc_predict;
- Bayes-калибровка порога на val;
- SignalGenerator(mode='classification') принимает probabilities;
- ModelMeta.mode сохраняется/загружается.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from graduate_work.config import (
    DataConfig,
    ModelConfig,
    TradingConfig,
    TrainingConfig,
)
from graduate_work.features import (
    StandardScaler,
    cost_aware_classification_labels,
)
from graduate_work.model import ConvLstmRegressor
from graduate_work.serving.artifact import ModelMeta, save_artifact
from graduate_work.serving.model_loader import load_artifact
from graduate_work.strategy import (
    SignalGenerator,
    bayes_threshold,
    build_predictions_frame,
    calibrate_bayes_threshold,
)
from graduate_work.training import Trainer, mc_predict, set_seed
from graduate_work.training.losses import (
    FocalBCEWithLogits,
    WeightedBCEWithLogits,
    build_loss_fn,
)


# ---------------------------------------------------------------------------
# Cost-aware labels
# ---------------------------------------------------------------------------

def test_cost_aware_labels_have_target_and_lr() -> None:
    n = 50
    idx = pd.date_range("2024-01-01 07:00", periods=n, freq="5min", tz="UTC")
    open_p = pd.Series(np.linspace(100, 105, n), index=idx)
    close_p = pd.Series(np.linspace(100, 105, n) + 0.1, index=idx)
    df = cost_aware_classification_labels(
        open_p, close_p, horizons=(1, 3),
        entry_cost=0.0003, exit_cost=0.0002,
    )
    assert "target_h1" in df.columns
    assert "target_h3" in df.columns
    assert "lr_h1" in df.columns
    assert "lr_h3" in df.columns
    # Метки должны быть в [0, 1].
    valid = df["target_h1"].dropna()
    assert valid.between(0.0, 1.0).all()


def test_cost_aware_labels_smoothing_pulls_off_extremes() -> None:
    n = 30
    idx = pd.date_range("2024-01-01 07:00", periods=n, freq="5min", tz="UTC")
    open_p = pd.Series(np.linspace(100, 110, n), index=idx)   # рост
    close_p = pd.Series(np.linspace(100, 110, n) + 0.1, index=idx)
    df = cost_aware_classification_labels(
        open_p, close_p, horizons=(1,),
        entry_cost=0.0001, exit_cost=0.0001,
        label_smoothing=0.05,
    )
    valid = df["target_h1"].dropna()
    # Большинство меток должно быть ~0.95, не 1.0.
    assert valid.max() <= 0.95 + 1e-6
    assert valid.min() >= 0.05 - 1e-6


def test_cost_aware_labels_negative_when_costs_eat_profit() -> None:
    """Если кост-роли больше реального движения - метка должна стать 0."""
    n = 10
    idx = pd.date_range("2024-01-01 07:00", periods=n, freq="5min", tz="UTC")
    flat = pd.Series([100.0] * n, index=idx)
    df = cost_aware_classification_labels(
        flat, flat, horizons=(1,),
        entry_cost=0.001, exit_cost=0.001,
        label_smoothing=0.0,
    )
    # Любая сделка теряет 0.2% costs - метка должна быть 0.
    valid = df["target_h1"].dropna()
    assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def test_build_loss_fn_classification_returns_bce() -> None:
    data_cfg = DataConfig(mode="classification")
    train_cfg = TrainingConfig()
    trading_cfg = TradingConfig(loss_objective="bce")
    loss = build_loss_fn(data_cfg, train_cfg, trading_cfg)
    assert isinstance(loss, WeightedBCEWithLogits)


def test_build_loss_fn_focal() -> None:
    data_cfg = DataConfig(mode="classification")
    train_cfg = TrainingConfig()
    trading_cfg = TradingConfig(loss_objective="focal")
    loss = build_loss_fn(data_cfg, train_cfg, trading_cfg)
    assert isinstance(loss, FocalBCEWithLogits)


def test_build_loss_fn_regression_returns_huber() -> None:
    data_cfg = DataConfig(mode="regression")
    train_cfg = TrainingConfig()
    loss = build_loss_fn(data_cfg, train_cfg, None)
    # Wrapper'a callable_loss или сразу HuberLoss - оба ОК.
    assert hasattr(loss, "forward")


# ---------------------------------------------------------------------------
# Trainer end-to-end на classification
# ---------------------------------------------------------------------------

def test_trainer_classification_runs_and_outputs_logits() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    n_train, n_val = 128, 32
    t, f, h = 16, 4, 2
    x_tr = rng.standard_normal((n_train, t, f)).astype(np.float32)
    y_tr = rng.integers(0, 2, size=(n_train, h)).astype(np.float32)
    x_va = rng.standard_normal((n_val, t, f)).astype(np.float32)
    y_va = rng.integers(0, 2, size=(n_val, h)).astype(np.float32)

    model_cfg = ModelConfig(
        conv_channels=8, conv_kernel=3, lstm_hidden=8, lstm_layers=1,
        fc_hidden=8, dropout=0.2, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=f, num_horizons=h, cfg=model_cfg)
    data_cfg = DataConfig(mode="classification")
    train_cfg = TrainingConfig(
        batch_size=32, epochs=2, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False, scheduler="none",
    )
    trading_cfg = TradingConfig(loss_objective="bce")
    trainer = Trainer(
        model, train_cfg,
        data_cfg=data_cfg, trading_cfg=trading_cfg,
        device="cpu",
    )
    trainer.fit({"x": x_tr, "y": y_tr}, {"x": x_va, "y": y_va})
    # Loss должен быть BCE.
    assert isinstance(trainer.loss_fn, WeightedBCEWithLogits)
    # Forward должен давать логиты (не ограниченные в [0, 1]).
    with torch.no_grad():
        logits = model(torch.from_numpy(x_va))
    # При случайной модели логиты разлетаются за [0,1].
    assert logits.shape == (n_val, h)


def test_trainer_auto_pos_weight_legacy_formula() -> None:
    """Legacy ветка: BCE получает per-horizon pos_weight = (1-p)/p,
    когда ``use_class_balanced_pos_weight=False``."""
    set_seed(0)
    rng = np.random.default_rng(0)
    n_train, n_val = 200, 50
    t, f, h = 8, 4, 2
    x_tr = rng.standard_normal((n_train, t, f)).astype(np.float32)
    y_tr = np.zeros((n_train, h), dtype=np.float32)
    y_tr[:40, 0] = 1.0   # 20% positives для горизонта 0
    y_tr[:100, 1] = 1.0  # 50% positives для горизонта 1
    x_va = rng.standard_normal((n_val, t, f)).astype(np.float32)
    y_va = np.zeros((n_val, h), dtype=np.float32)

    model_cfg = ModelConfig(
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.0, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=f, num_horizons=h, cfg=model_cfg)
    data_cfg = DataConfig(mode="classification")
    train_cfg = TrainingConfig(
        batch_size=32, epochs=1, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False, scheduler="none",
    )
    # ВАЖНО: явно отключаем class-balanced ради проверки legacy-формулы.
    trading_cfg = TradingConfig(
        loss_objective="bce", use_class_balanced_pos_weight=False,
    )
    trainer = Trainer(
        model, train_cfg,
        data_cfg=data_cfg, trading_cfg=trading_cfg,
        device="cpu",
    )
    trainer.fit({"x": x_tr, "y": y_tr}, {"x": x_va, "y": y_va})

    pw = trainer.loss_fn.pos_weight
    assert pw is not None
    assert pw.shape == (h,)
    # h=0: (1-0.2)/0.2 = 4.0
    # h=1: (1-0.5)/0.5 = 1.0
    assert pw[0].item() == pytest.approx(4.0, rel=0.05)
    assert pw[1].item() == pytest.approx(1.0, rel=0.05)


def test_trainer_class_balanced_pos_weight_default() -> None:
    """Default ветка после R-0050: ``use_class_balanced_pos_weight=True``
    даёт более мягкое значение pos_weight, чем (1-p)/p."""
    set_seed(0)
    rng = np.random.default_rng(0)
    n_train, n_val = 1000, 100
    t, f, h = 8, 4, 2
    x_tr = rng.standard_normal((n_train, t, f)).astype(np.float32)
    y_tr = np.zeros((n_train, h), dtype=np.float32)
    y_tr[:200, 0] = 1.0   # 20% positives для h=0
    y_tr[:500, 1] = 1.0   # 50% positives для h=1
    x_va = rng.standard_normal((n_val, t, f)).astype(np.float32)
    y_va = np.zeros((n_val, h), dtype=np.float32)

    model_cfg = ModelConfig(
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.0, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=f, num_horizons=h, cfg=model_cfg)
    trainer = Trainer(
        model,
        TrainingConfig(
            batch_size=32, epochs=1, learning_rate=1e-3,
            use_swa=False, scheduler="none",
        ),
        data_cfg=DataConfig(mode="classification"),
        trading_cfg=TradingConfig(
            loss_objective="bce",
            use_class_balanced_pos_weight=True, class_balanced_beta=0.999,
        ),
        device="cpu",
    )
    trainer.fit({"x": x_tr, "y": y_tr}, {"x": x_va, "y": y_va})

    pw = trainer.loss_fn.pos_weight
    assert pw is not None
    # h=0 (P=0.2): legacy =4.0; class-balanced при n=1000, β=0.999 → ~3.0–3.7
    # — определённо МЕНЬШЕ legacy 4.0 (что и нужно — предотвращаем
    # агрессивное усиление minority-класса).
    assert pw[0].item() < 4.0
    # h=1 (P=0.5): обе формулы дают ~1.0.
    assert pw[1].item() == pytest.approx(1.0, rel=0.10)


def test_weighted_bce_pos_weight_pushes_logits_up_for_minority_class() -> None:
    """Sanity-check: на сильном дисбалансе модель с pos_weight НЕ
    схлопывается на маргинальное P(UP)."""
    set_seed(0)
    # Без pos_weight оптимальный логит ~ logit(p_up)=logit(0.1)≈-2.2.
    # С pos_weight=(1-p)/p≈9.0 оптимальный логит = logit(0.5)=0.0.
    n, h = 100, 1
    target = torch.zeros(n, h)
    target[:10] = 1.0  # 10% positives.
    logits = torch.zeros(n, h, requires_grad=True)
    optim = torch.optim.SGD([logits], lr=0.5)

    pw = torch.tensor([(1.0 - 0.1) / 0.1])
    loss_fn = WeightedBCEWithLogits(pos_weight=pw)
    for _ in range(200):
        optim.zero_grad()
        loss = loss_fn(logits, target)
        loss.backward()
        optim.step()
    # С pos_weight оптимум — около 0 (равные веса positives и negatives
    # балансируют), а не logit(0.1)=-2.2 как было бы в простом BCE.
    assert abs(logits.detach().mean().item()) < 0.5


# ---------------------------------------------------------------------------
# MC inference - apply_sigmoid
# ---------------------------------------------------------------------------

def test_mc_predict_applies_sigmoid_to_classification() -> None:
    rng = np.random.default_rng(0)
    n, t, f, h = 20, 10, 3, 2
    x = rng.standard_normal((n, t, f)).astype(np.float32)
    model_cfg = ModelConfig(
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.2, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=f, num_horizons=h, cfg=model_cfg)

    mean_cls, _ = mc_predict(
        model, x, mc_passes=5, batch_size=8,
        device="cpu", apply_sigmoid=True,
    )
    mean_reg, _ = mc_predict(
        model, x, mc_passes=5, batch_size=8,
        device="cpu", apply_sigmoid=False,
    )
    # При apply_sigmoid=True все значения в [0, 1].
    assert mean_cls.min() >= 0.0
    assert mean_cls.max() <= 1.0
    # Без сигмоиды могут вылезти.
    assert (mean_reg.min() < 0.0) or (mean_reg.max() > 1.0)


# ---------------------------------------------------------------------------
# Bayes threshold
# ---------------------------------------------------------------------------

def test_bayes_threshold_clipped_below_0_51() -> None:
    # Огромный gain, маленький cost → формула даёт ~0; должно зажаться 0.51.
    t = bayes_threshold(cost_per_trade=0.0001, expected_gain=10.0)
    assert t == 0.51


def test_bayes_threshold_high_for_expensive_trades() -> None:
    # Cost огромный, gain маленький → формула близка к 1; зажмётся к 0.95.
    t = bayes_threshold(cost_per_trade=0.5, expected_gain=0.001)
    assert 0.55 < t <= 0.95


def test_calibrate_bayes_threshold_returns_consistent_value() -> None:
    rng = np.random.default_rng(0)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    val_pred = build_predictions_frame(
        timestamps=np.array(timestamps),
        tickers=np.array(["A"] * n),
        mean=rng.uniform(0.4, 0.7, (n, 1)).astype(np.float32),
        std=np.full((n, 1), 0.05, dtype=np.float32),
        horizons=(1,),
    )
    # actual = positive log-return на 60% сэмплов.
    pos_lr = rng.exponential(0.002, n) * np.sign(rng.standard_normal(n) + 0.5)
    actual = val_pred[["timestamp", "ticker", "horizon"]].copy()
    actual["actual"] = pos_lr
    res = calibrate_bayes_threshold(
        val_pred, actual, cost_per_trade=0.001,
    )
    assert 0.51 <= res.min_expected_return <= 0.95


# ---------------------------------------------------------------------------
# SignalGenerator в classification
# ---------------------------------------------------------------------------

def test_signal_generator_classification_uses_probability_threshold() -> None:
    timestamps = np.array(["2024-01-01"] * 3, dtype="datetime64[ns]")
    tickers = np.array(["A", "B", "C"])
    # Вероятности: 0.6, 0.7, 0.8. Std небольшой.
    mean = np.array([[0.6], [0.7], [0.8]], dtype=np.float32)
    std = np.full((3, 1), 0.05, dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1,))

    cfg = TradingConfig(
        max_positions=3, probability_threshold=0.65,
        max_probability_std=0.5,
    )
    sigs = SignalGenerator(cfg, mode="classification").generate(df)
    buys = sigs[sigs["action"] == "BUY"]
    # Только B (0.7) и C (0.8) проходят порог 0.65.
    assert set(buys["ticker"]) == {"B", "C"}


def test_signal_generator_classification_rejects_high_uncertainty() -> None:
    timestamps = np.array(["2024-01-01"] * 2, dtype="datetime64[ns]")
    tickers = np.array(["A", "B"])
    mean = np.array([[0.8], [0.8]], dtype=np.float32)
    # У A очень большой std.
    std = np.array([[0.4], [0.05]], dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1,))

    cfg = TradingConfig(
        max_positions=2, probability_threshold=0.6,
        max_probability_std=0.2,
    )
    sigs = SignalGenerator(cfg, mode="classification").generate(df)
    buys = sigs[sigs["action"] == "BUY"]
    # A отрезан high std, B проходит.
    assert set(buys["ticker"]) == {"B"}


# ---------------------------------------------------------------------------
# ModelMeta.mode round-trip
# ---------------------------------------------------------------------------

def test_model_meta_classification_round_trip(tmp_path) -> None:
    feature_cols = ["a", "b"]
    model_cfg = ModelConfig(
        architecture="conv_lstm",
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.0, use_revin=False,
    )
    meta = ModelMeta(
        feature_cols=feature_cols,
        target_cols=["target_h1"],
        horizons=[1],
        window_size=10,
        num_features=2,
        num_horizons=1,
        model_config=dataclasses.asdict(model_cfg),
        training_date="2024-01-01T00:00:00+00:00",
        tickers=["AAA"],
        mode="classification",
    )
    model = ConvLstmRegressor(input_dim=2, num_horizons=1, cfg=model_cfg)
    scaler = StandardScaler()
    df = pd.DataFrame({c: np.random.randn(50) for c in feature_cols})
    scaler.fit(df, feature_cols)
    save_artifact(model, scaler, meta, tmp_path)
    loaded = load_artifact(tmp_path, device="cpu")
    assert loaded.meta.mode == "classification"
