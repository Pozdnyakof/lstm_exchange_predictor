"""Тесты Adaptive Conformal Inference (Gibbs-Candès 2021)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graduate_work.strategy import (
    AdaptiveConformalPredictor,
    aci_signals_to_actions,
)


def _make_calib_frames(n: int = 100, h_list=(6, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетические predictions/targets с временной осью UTC."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rows_pred, rows_act = [], []
    for ts in timestamps:
        for h in h_list:
            mean = float(np.clip(rng.normal(0.55, 0.1), 0, 1))
            actual = float(rng.binomial(1, 0.5))
            rows_pred.append(
                {"timestamp": ts, "ticker": "TST", "horizon": h,
                 "mean": mean, "std": 0.05},
            )
            rows_act.append(
                {"timestamp": ts, "ticker": "TST", "horizon": h, "actual": actual},
            )
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_act)


def test_calibrate_initialises_state_per_horizon() -> None:
    """После calibrate() в state_summary должна быть строка на каждый горизонт."""
    aci = AdaptiveConformalPredictor(target_alpha=0.1)
    pred, act = _make_calib_frames(n=80, h_list=(6, 12, 24))
    aci.calibrate(pred, act)
    summary = aci.state_summary
    assert sorted(summary["horizon"].tolist()) == [6, 12, 24]
    assert (summary["alpha"] == 0.1).all()


def test_threshold_in_unit_interval() -> None:
    """threshold всегда в [0, 1]."""
    aci = AdaptiveConformalPredictor(target_alpha=0.1)
    pred, act = _make_calib_frames(n=60)
    aci.calibrate(pred, act)
    for h in (6, 12):
        thr = aci.threshold(h)
        assert 0.0 <= thr <= 1.0


def test_alpha_increases_when_no_miscoverage() -> None:
    """Покрытие выше target → α растёт (Gibbs-Candès: новый α = α + γ(α* - err)).

    err=0 всегда → empirical_miscov=0 < target_α=0.1 → есть headroom →
    α растёт → threshold↑ → меньше BUY'ев, но каждый более уверенный.
    """
    aci = AdaptiveConformalPredictor(target_alpha=0.1, gamma=0.05)
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    for _ in range(50):
        aci.update(predicted_prob=0.99, actual=1.0, horizon=6)
    state = aci.state_summary
    alpha_h6 = float(state[state["horizon"] == 6]["alpha"].iloc[0])
    assert alpha_h6 > 0.1


def test_alpha_decreases_when_miscoverage_high() -> None:
    """Все позитивы miscovered → α убывает → threshold↓ → больше BUY'ев."""
    aci = AdaptiveConformalPredictor(target_alpha=0.1, gamma=0.05)
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    for _ in range(50):
        aci.update(predicted_prob=0.0, actual=1.0, horizon=6)
    alpha_h6 = float(aci.state_summary[aci.state_summary["horizon"] == 6]["alpha"].iloc[0])
    assert alpha_h6 < 0.1


def test_alpha_clipped_to_bounds() -> None:
    """α не выходит за [alpha_min, alpha_max]."""
    aci = AdaptiveConformalPredictor(
        target_alpha=0.1, gamma=1.0,            # огромный шаг → быстрая адаптация
        alpha_min=0.01, alpha_max=0.4,
    )
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    # Постоянно miscovered → α упрётся в alpha_min (becoming less strict).
    for _ in range(30):
        aci.update(predicted_prob=0.0, actual=1.0, horizon=6)
    summary = aci.state_summary
    assert summary["alpha"].max() <= 0.4 + 1e-9
    assert summary["alpha"].min() >= 0.01 - 1e-9


def test_replay_returns_signals() -> None:
    """``replay`` отдаёт фрейм с threshold/alpha/signal/miscovered колонками."""
    aci = AdaptiveConformalPredictor(target_alpha=0.1, gamma=0.005)
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    test_pred, test_act = _make_calib_frames(n=20)
    out = aci.replay(test_pred, test_act)
    for col in ("threshold", "alpha", "signal", "miscovered"):
        assert col in out.columns
    assert ((out["signal"] == 0) | (out["signal"] == 1)).all()


def test_aci_signals_to_actions_top_k_per_timestamp() -> None:
    """С 1 тикером top-k = всегда BUY если signal=1, иначе HOLD."""
    aci = AdaptiveConformalPredictor(target_alpha=0.1)
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    test_pred, test_act = _make_calib_frames(n=20)
    replayed = aci.replay(test_pred, test_act)
    actions = aci_signals_to_actions(replayed, max_positions=5)
    assert "action" in actions.columns
    assert set(actions["action"].unique()) <= {"BUY", "HOLD"}
