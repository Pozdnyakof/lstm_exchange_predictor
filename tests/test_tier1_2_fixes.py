"""Регрессионные тесты для исправлений Tier 1 + Tier 2.

Покрывают:
- T1.1: rollover на следующий БАР, не сутки.
- T1.2: auto Huber delta.
- T1.4: калибровка порога на val.
- T2.1: per-ticker dummies в фичах.
- T2.2: AdamW + cosine scheduler.
- T2.3: Bonferroni correction в SignalGenerator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from graduate_work.backtest.engine import run_backtest
from graduate_work.config import (
    DataConfig,
    ModelConfig,
    Paths,
    TradingConfig,
    TrainingConfig,
)
from graduate_work.features.pipeline import build_feature_frame
from graduate_work.model import ConvLstmRegressor
from graduate_work.strategy import (
    SignalGenerator,
    attach_actual_targets,
    build_predictions_frame,
    calibrate_min_expected_return,
)
from graduate_work.training import Trainer, set_seed


# ---------------------------------------------------------------------------
# T1.1 — rollover на бар, не сутки
# ---------------------------------------------------------------------------

def test_engine_rollover_uses_next_bar_not_day() -> None:
    """Если на close_date нет цены - продлеваем на 1 БАР, не на сутки.

    После фикса entry-on-next-open: сигнал bar 0 → entry bar 1 →
    запланированный exit на bar 1+5=6. Если в bar 6 цены нет, движок
    откатит на bar 7 (1 бар, не сутки).
    """
    n = 30
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {"close": np.linspace(100.0, 110.0, n), "ticker": "TST"},
        index=idx,
    )
    # Удаляем цену в bar 6 - именно где должен сработать exit.
    prices = prices.drop(idx[6])
    cfg = TradingConfig(
        initial_capital=100_000.0, max_positions=1,
        commission_rate=0.0, slippage_rate=0.0,
    )
    signals = pd.DataFrame(
        [{"timestamp": idx[0], "ticker": "TST",
          "horizon": 5, "mean": 0.01, "std": 0.001, "action": "BUY"}],
    )
    bt = run_backtest(signals, prices, cfg)
    assert len(bt.trades) == 1
    trade = bt.trades.iloc[0]
    assert trade["close_date"] in prices.index
    # bar 6 удалён → rollover ровно на bar 7.
    assert trade["close_date"] == idx[7]


# ---------------------------------------------------------------------------
# T1.2 — Auto Huber delta
# ---------------------------------------------------------------------------

def test_trainer_uses_auto_huber_delta_when_enabled() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    # Целевая шкала ~ 1e-3.
    y = rng.standard_normal((128, 2)).astype(np.float32) * 1e-3
    train_arrays = {
        "x": rng.standard_normal((128, 16, 4)).astype(np.float32),
        "y": y,
    }
    val_arrays = {
        "x": rng.standard_normal((32, 16, 4)).astype(np.float32),
        "y": rng.standard_normal((32, 2)).astype(np.float32) * 1e-3,
    }

    model_cfg = ModelConfig(
        conv_channels=8, conv_kernel=3, lstm_hidden=8, lstm_layers=1,
        fc_hidden=8, dropout=0.0, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=4, num_horizons=2, cfg=model_cfg)
    cfg = TrainingConfig(
        batch_size=32, epochs=2, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False,
        huber_delta_auto=True, optimizer="adam", scheduler="none",
    )
    trainer = Trainer(model, cfg, device="cpu")
    trainer.fit(train_arrays, val_arrays)
    # После fit-а delta должна быть существенно меньше 1.0
    # (≈ 2 × median(|y|) ≈ 1e-3 для нашего y).
    assert trainer.loss_fn.delta < 0.1
    assert trainer.loss_fn.delta >= 1e-4


def test_trainer_uses_adamw_when_configured() -> None:
    model_cfg = ModelConfig(
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.0, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=4, num_horizons=2, cfg=model_cfg)
    cfg = TrainingConfig(optimizer="adamw")
    trainer = Trainer(model, cfg, device="cpu")
    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_trainer_uses_cosine_scheduler_when_configured() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    train_arrays = {
        "x": rng.standard_normal((64, 16, 4)).astype(np.float32),
        "y": rng.standard_normal((64, 2)).astype(np.float32) * 1e-3,
    }
    val_arrays = {
        "x": rng.standard_normal((16, 16, 4)).astype(np.float32),
        "y": rng.standard_normal((16, 2)).astype(np.float32) * 1e-3,
    }
    model_cfg = ModelConfig(
        conv_channels=4, conv_kernel=3, lstm_hidden=4, lstm_layers=1,
        fc_hidden=4, dropout=0.0, use_revin=False,
    )
    model = ConvLstmRegressor(input_dim=4, num_horizons=2, cfg=model_cfg)
    cfg = TrainingConfig(
        batch_size=16, epochs=4, learning_rate=1e-3,
        early_stopping_patience=10, use_swa=False,
        scheduler="cosine",
    )
    trainer = Trainer(model, cfg, device="cpu")
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.fit(train_arrays, val_arrays)
    final_lr = trainer.optimizer.param_groups[0]["lr"]
    # CosineAnnealing должен снизить LR.
    assert final_lr < initial_lr


# ---------------------------------------------------------------------------
# T1.4 — калибровка порога
# ---------------------------------------------------------------------------

def test_calibrate_threshold_returns_fallback_on_empty() -> None:
    res = calibrate_min_expected_return(
        pd.DataFrame(), pd.DataFrame(), cost_per_trade=0.001,
    )
    assert res.n_val_signals == 0
    assert res.min_expected_return >= 0.0


def test_calibrate_threshold_picks_above_cost_floor() -> None:
    """Если в val есть сигналы со значимым edge - порог должен их пропускать."""
    rng = np.random.default_rng(42)
    n = 1000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    horizons = (1, 3)
    val_pred = build_predictions_frame(
        timestamps=np.array(timestamps),
        tickers=np.array(["A"] * n),
        mean=rng.standard_normal((n, len(horizons))).astype(np.float32) * 0.001,
        std=np.full((n, len(horizons)), 0.001, dtype=np.float32),
        horizons=horizons,
    )
    # actual ≈ mean + noise (correlated → edge есть).
    actual = val_pred[["timestamp", "ticker", "horizon", "mean"]].copy()
    actual["actual"] = val_pred["mean"] + rng.standard_normal(len(val_pred)) * 0.0005
    actual = actual.drop(columns=["mean"])
    res = calibrate_min_expected_return(
        val_pred, actual, cost_per_trade=0.0002,
    )
    assert res.min_expected_return >= 0.0
    assert res.n_val_signals > 0


def test_attach_actual_targets_long_format() -> None:
    val = {
        "y": np.array([[0.001, 0.002], [-0.001, 0.003]], dtype=np.float32),
        "timestamp": np.array(["2024-01-01", "2024-01-01 00:05"], dtype="datetime64[ns]"),
        "ticker": np.array(["A", "B"]),
    }
    df = attach_actual_targets(val, horizons=(1, 5))
    assert len(df) == 4
    assert set(df.columns) == {"timestamp", "ticker", "horizon", "actual"}


# ---------------------------------------------------------------------------
# T2.1 — per-ticker dummies
# ---------------------------------------------------------------------------

def test_pipeline_adds_ticker_dummies(tmp_path) -> None:
    """build_feature_frame с use_ticker_dummies=True добавляет tid_-колонки."""
    # Синтезируем минимальное хранилище для двух тикеров.
    paths = Paths(
        project_root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        checkpoints=tmp_path / "checkpoints",
    )
    paths.ensure()
    moex_dir = paths.data_raw / "moex"
    moex_dir.mkdir(parents=True, exist_ok=True)
    # 1-минутные бары сессии для двух тикеров.
    n = 600
    idx = pd.date_range("2024-01-02 07:00", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    for ticker in ("TKA", "TKB"):
        close = 100 + rng.standard_normal(n).cumsum() * 0.05
        df = pd.DataFrame(
            {
                "open": close + 0.01, "high": close + 0.05,
                "low": close - 0.05, "close": close,
                "volume": rng.integers(100, 500, n).astype(float),
            },
            index=idx,
        )
        df.to_csv(moex_dir / f"{ticker}.csv")

    data_cfg = DataConfig(
        tickers=("TKA", "TKB"),
        bar_minutes=5,
        horizons=(1, 3),
        window_size=10,
        use_ticker_dummies=True,
    )
    full, feature_cols = build_feature_frame(data_cfg, paths)
    assert "tid_TKA" in full.columns
    assert "tid_TKB" in full.columns
    assert "tid_TKA" in feature_cols
    assert "tid_TKB" in feature_cols


# ---------------------------------------------------------------------------
# T2.3 — Bonferroni argmax correction
# ---------------------------------------------------------------------------

def test_signal_generator_applies_horizon_correction() -> None:
    """При нескольких горизонтах порог умножается на correction_factor."""
    timestamps = np.array(["2024-01-01"] * 4, dtype="datetime64[ns]")
    tickers = np.array(["A", "B", "C", "D"])
    # mean = 0.001 для всех, std маленький, 4 горизонта.
    mean = np.full((4, 4), 0.001, dtype=np.float32)
    std = np.full((4, 4), 0.0001, dtype=np.float32)
    df = build_predictions_frame(timestamps, tickers, mean, std, horizons=(1, 3, 6, 12))

    # Без коррекции: T_eff = 0.0005, mean=0.001 ≥ 0.0005 → BUY.
    cfg_no_corr = TradingConfig(
        max_positions=4, min_expected_return=0.0005, max_uncertainty=1.0,
        horizon_argmax_correction=1.0,
    )
    sig_no = SignalGenerator(cfg_no_corr).generate(df)
    n_buy_no = (sig_no["action"] == "BUY").sum()

    # С коррекцией 3.0: T_eff = 0.0015, mean=0.001 < 0.0015 → HOLD.
    cfg_corr = TradingConfig(
        max_positions=4, min_expected_return=0.0005, max_uncertainty=1.0,
        horizon_argmax_correction=3.0,
    )
    sig_corr = SignalGenerator(cfg_corr).generate(df)
    n_buy_corr = (sig_corr["action"] == "BUY").sum()

    assert n_buy_no > n_buy_corr   # коррекция ужесточает фильтр.
