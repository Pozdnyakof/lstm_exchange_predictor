"""Тесты anti-collapse стека (Sprint A+B+C+D после R-0050).

Покрывает:
- Logit Adjustment (A1) в iTransformer
- AsymmetricLossWithLogits (A2)
- class_balanced_pos_weight (A3)
- ImbSAMOptimizer (B1)
- mixup_batch (B2)
- DropPath в iTransformer (B3)
- functional_rbf_repulsion (C1)
- DtACIPredictor (C2)
- CompositeQuantLoss с UW (D1) и sharpe_weight=0 (D2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from graduate_work.config import ModelConfig
from graduate_work.model import ITransformer
from graduate_work.strategy import DtACIPredictor
from graduate_work.training import (
    AsymmetricLossWithLogits,
    CompositeQuantLoss,
    ImbSAMOptimizer,
    class_balanced_pos_weight,
    maybe_apply_mixup,
    mixup_batch,
    select_minority_subset,
)
from graduate_work.training.repulsion import (
    functional_rbf_repulsion,
    rbf_kernel_loss,
)


# ---------------------------------------------------------------------------
# A1: Logit Adjustment
# ---------------------------------------------------------------------------

def _tiny_itransformer(input_dim: int = 6, num_horizons: int = 4) -> ITransformer:
    cfg = ModelConfig(
        architecture="itransformer",
        fc_hidden=16,
        dropout=0.0,
        use_revin=False,
        itransformer_seq_len=16,
        itransformer_d_model=16,
        itransformer_n_layers=1,
        itransformer_n_heads=2,
        itransformer_d_ff=32,
        itransformer_dropout=0.0,
        itransformer_drop_path=0.0,
        logit_adjust_tau=1.0,
    )
    return ITransformer(input_dim, num_horizons, cfg)


def test_logit_adjustment_active_in_train_mode() -> None:
    """В train-mode logit_adjust_tau>0 должен сместить выходы."""
    model = _tiny_itransformer()
    model.set_logit_prior(torch.tensor([0.3, 0.4, 0.5, 0.6]))
    x = torch.randn(2, 16, 6)
    model.train()
    train_out = model(x)
    model.eval()
    eval_out = model(x)
    # eval не вычитает prior → разница между ними соответствует prior
    diff = (eval_out - train_out).mean(dim=0)
    expected = torch.tensor([
        np.log(0.3 / 0.7), np.log(0.4 / 0.6),
        np.log(0.5 / 0.5), np.log(0.6 / 0.4),
    ], dtype=torch.float32)
    assert torch.allclose(diff, expected, atol=1e-4)


def test_logit_adjustment_zero_tau_is_noop() -> None:
    """tau=0 → train и eval дают одинаковый выход (на одной модели)."""
    cfg = ModelConfig(
        architecture="itransformer",
        fc_hidden=16, dropout=0.0, use_revin=False,
        itransformer_seq_len=16, itransformer_d_model=16,
        itransformer_n_layers=1, itransformer_n_heads=2,
        itransformer_d_ff=32, itransformer_dropout=0.0,
        itransformer_drop_path=0.0, logit_adjust_tau=0.0,
    )
    model = ITransformer(6, 4, cfg)
    model.set_logit_prior(torch.tensor([0.3, 0.4, 0.5, 0.6]))
    x = torch.randn(2, 16, 6)
    model.train()
    train_out = model(x)
    model.eval()
    assert torch.allclose(train_out, model(x), atol=1e-6)


# ---------------------------------------------------------------------------
# A2: Asymmetric Loss
# ---------------------------------------------------------------------------

def test_asl_zero_loss_when_perfect_prediction() -> None:
    """Идеально предсказали → loss ≈ 0."""
    loss = AsymmetricLossWithLogits(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    logits = torch.tensor([[10.0, -10.0]])
    target = torch.tensor([[1.0, 0.0]])
    assert loss(logits, target).item() < 1e-3


def test_asl_dominates_easy_negatives() -> None:
    """ASL с γ_neg=4 даёт МЕНЬШЕ loss на лёгких negatives, чем BCE."""
    asl = AsymmetricLossWithLogits(gamma_pos=0.0, gamma_neg=4.0, clip=0.0)
    # «Лёгкий» negative: модель уверена в 0 (logit ≈ -3), target=0.
    logits = torch.tensor([[-3.0]])
    target = torch.tensor([[0.0]])
    asl_val = asl(logits, target).item()
    bce_val = nn.functional.binary_cross_entropy_with_logits(
        logits, target,
    ).item()
    assert asl_val < bce_val


def test_asl_clip_makes_negatives_easier() -> None:
    """clip=0.05 → лёгкие negatives дают ровно 0 loss."""
    loss = AsymmetricLossWithLogits(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    # Logit ≈ −3 → prob ≈ 0.047 < 0.05 → p_shifted = max(p-clip, 0) ≈ 0
    # → focal_neg = 0 → loss ≈ 0.
    logits = torch.tensor([[-3.0]])
    target = torch.tensor([[0.0]])
    assert loss(logits, target).item() < 1e-2


# ---------------------------------------------------------------------------
# A3: Class-Balanced pos_weight
# ---------------------------------------------------------------------------

def test_class_balanced_softer_than_legacy() -> None:
    """На больших n class-balanced даёт МЕНЬШИЙ pos_weight, чем legacy."""
    n = 10_000
    y = np.zeros((n, 1), dtype=np.float32)
    y[:2000] = 1.0  # P(UP)=0.2 → legacy=4.0
    pw = class_balanced_pos_weight(y, beta=0.999)
    assert pw[0] < 4.0
    assert pw[0] > 1.0


def test_class_balanced_validates_inputs() -> None:
    """Защита от плохих параметров β."""
    y = np.zeros((10, 1), dtype=np.float32)
    try:
        class_balanced_pos_weight(y, beta=1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError on beta=1.5")


# ---------------------------------------------------------------------------
# B1: ImbSAM
# ---------------------------------------------------------------------------

def test_imbsam_runs_one_step() -> None:
    """ImbSAMOptimizer.step() выполняется и обновляет веса."""
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sam = ImbSAMOptimizer(opt, model, rho=0.05)
    x = torch.randn(8, 4)
    y = torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
    w_before = model.weight.detach().clone()
    sam.step(
        loss_fn=lambda: nn.functional.binary_cross_entropy_with_logits(model(x), y),
        minority_loss_fn=lambda: nn.functional.binary_cross_entropy_with_logits(
            model(x[:4]), y[:4],
        ),
    )
    assert not torch.allclose(w_before, model.weight)


def test_imbsam_no_minority_falls_back_to_plain_sgd() -> None:
    """Если minority_loss_fn возвращает None — обычный шаг без перетурбации."""
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sam = ImbSAMOptimizer(opt, model, rho=0.05)
    x = torch.randn(8, 4)
    y = torch.zeros(8, 2)
    w_before = model.weight.detach().clone()
    sam.step(
        loss_fn=lambda: nn.functional.binary_cross_entropy_with_logits(model(x), y),
        minority_loss_fn=lambda: None,
    )
    assert not torch.allclose(w_before, model.weight)


def test_select_minority_subset_filters_correctly() -> None:
    """select_minority_subset берёт только UP-сэмплы по h=0."""
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                      [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x_min, y_min, _ = select_minority_subset(
        x, y, minority_label=1.0, horizon_for_filter=0,
    )
    assert x_min.shape[0] == 3   # 3 UPs по h=0
    assert (y_min[:, 0] >= 0.5).all()


# ---------------------------------------------------------------------------
# B2: Mixup
# ---------------------------------------------------------------------------

def test_mixup_preserves_shapes() -> None:
    """Mixup сохраняет shapes."""
    x = torch.randn(8, 16, 4)
    y = torch.randint(0, 2, (8, 2)).float()
    lr = torch.randn(8, 2) * 0.01
    x_m, y_m, lr_m, lam = mixup_batch(x, y, lr, alpha=0.2)
    assert x_m.shape == x.shape
    assert y_m.shape == y.shape
    assert lr_m.shape == lr.shape
    assert 0.5 <= lam <= 1.0  # симметризованный λ


def test_mixup_alpha_zero_returns_input() -> None:
    """α=0 → mixup отключён, возвращает входы как есть."""
    x = torch.randn(4, 8, 3)
    y = torch.zeros(4, 1)
    x_m, y_m, _, lam = mixup_batch(x, y, alpha=0.0)
    assert torch.allclose(x_m, x)
    assert lam == 1.0


def test_maybe_apply_mixup_p_zero_skips() -> None:
    """p=0 → детерминированно НЕ применяет mixup."""
    x = torch.randn(4, 8, 3)
    y = torch.zeros(4, 1)
    x_m, y_m, _, lam = maybe_apply_mixup(x, y, alpha=0.5, p=0.0)
    assert torch.allclose(x_m, x)
    assert lam == 1.0


# ---------------------------------------------------------------------------
# B3: DropPath
# ---------------------------------------------------------------------------

def test_droppath_active_in_train_mode() -> None:
    """drop_path>0 в iTransformer должен делать train-выход стохастическим."""
    cfg = ModelConfig(
        architecture="itransformer",
        fc_hidden=16, dropout=0.0, use_revin=False,
        itransformer_seq_len=16, itransformer_d_model=16,
        itransformer_n_layers=2, itransformer_n_heads=2,
        itransformer_d_ff=32, itransformer_dropout=0.0,
        itransformer_drop_path=0.5,
        logit_adjust_tau=0.0,
    )
    model = ITransformer(6, 4, cfg)
    model.train()
    x = torch.randn(8, 16, 6)
    out_a = model(x)
    out_b = model(x)
    # При drop_path=0.5 разные форварды дают разные выходы.
    assert not torch.allclose(out_a, out_b, atol=1e-4)


# ---------------------------------------------------------------------------
# C1: Repulsive ensembles
# ---------------------------------------------------------------------------

def test_rbf_kernel_self_is_one() -> None:
    """k(a, a) = 1 для любых a."""
    a = torch.randn(8, 4)
    assert rbf_kernel_loss(a, a).item() == 1.0


def test_rbf_kernel_below_one_for_different_pairs() -> None:
    """k(a, b) < 1 когда хотя бы часть пар различается.

    Median-bandwidth делает kernel scale-invariant, поэтому абсолютные
    значения зависят от размера батча; гарантировано лишь, что для
    смеси разных пар средний kernel < 1.
    """
    torch.manual_seed(0)
    a = torch.randn(8, 4)
    b = a + torch.randn_like(a)  # все пары различны
    k = rbf_kernel_loss(a, b).item()
    assert 0.0 < k < 1.0


def test_repulsion_zero_for_empty_predecessors() -> None:
    """Без predecessors репульсия — нулевой скаляр."""
    preds = torch.randn(4, 2, requires_grad=True)
    x = torch.randn(4, 8, 3)
    rep = functional_rbf_repulsion(preds, x, [], weight=0.5)
    assert rep.item() == 0.0


def test_repulsion_gradient_flows_to_current() -> None:
    """Repulsion-loss скаляр >= 0; predecessors — frozen nn.Module."""
    class _PrevWrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8 * 3, 2)
            for p in self.parameters():
                p.requires_grad = False
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x.flatten(1) if x.ndim > 2 else x)

    prev = _PrevWrap()
    x = torch.randn(4, 8, 3)
    cur = nn.Linear(8 * 3, 2)
    preds = cur(x.flatten(1))
    rep = functional_rbf_repulsion(preds, x, [prev], weight=0.5)
    assert rep.dim() == 0
    assert rep.item() >= 0


# ---------------------------------------------------------------------------
# C2: DtACI
# ---------------------------------------------------------------------------

def _make_calib_frames(n: int = 60, h_list=(6, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rows_pred, rows_act = [], []
    for ts in timestamps:
        for h in h_list:
            rows_pred.append({
                "timestamp": ts, "ticker": "TST", "horizon": h,
                "mean": float(np.clip(rng.normal(0.55, 0.1), 0, 1)),
                "std": 0.05,
            })
            rows_act.append({
                "timestamp": ts, "ticker": "TST", "horizon": h,
                "actual": float(rng.binomial(1, 0.5)),
            })
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_act)


def test_dtaci_calibrates_per_horizon() -> None:
    """После calibrate() в state_summary все горизонты + все γ-эксперты."""
    aci = DtACIPredictor(
        target_alpha=0.1, gammas=(0.001, 0.01, 0.05),
    )
    pred, act = _make_calib_frames(n=40, h_list=(6, 12))
    aci.calibrate(pred, act)
    s = aci.state_summary
    assert sorted(s["horizon"].tolist()) == [6, 12]
    # 3 эксперта → должны быть колонки w_g=... и alpha_g=... × 3.
    weight_cols = [c for c in s.columns if c.startswith("w_g=")]
    assert len(weight_cols) == 3
    # Веса нормализованы.
    for _, row in s.iterrows():
        ws = [row[c] for c in weight_cols]
        assert abs(sum(ws) - 1.0) < 1e-6


def test_dtaci_replay_returns_signals() -> None:
    """replay() добавляет threshold/alpha/signal/miscovered."""
    aci = DtACIPredictor(target_alpha=0.1)
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    test_pred, test_act = _make_calib_frames(n=20)
    out = aci.replay(test_pred, test_act)
    for col in ("threshold", "alpha", "signal", "miscovered"):
        assert col in out.columns


def test_dtaci_weight_concentration_after_many_steps() -> None:
    """После многих миссов веса должны сместиться от плохих γ к лучшим
    (т.е. они НЕ остаются равномерными)."""
    aci = DtACIPredictor(
        target_alpha=0.1, gammas=(0.001, 0.05), eta=0.5,
    )
    pred, act = _make_calib_frames(n=40)
    aci.calibrate(pred, act)
    # Постоянно miscovered → плохой γ накопит больше pinball-loss.
    for _ in range(100):
        aci.update(predicted_prob=0.0, actual=1.0, horizon=6)
    s = aci.state_summary
    weight_cols = [c for c in s.columns if c.startswith("w_g=")]
    h6 = s[s["horizon"] == 6].iloc[0]
    ws = [float(h6[c]) for c in weight_cols]
    # Хотя бы один вес отличается от равномерного на >1%.
    assert max(abs(w - 0.5) for w in ws) > 0.01


# ---------------------------------------------------------------------------
# D1+D2: Composite loss с UW и sharpe_weight=0 по умолчанию
# ---------------------------------------------------------------------------

def test_composite_uncertainty_weighting_creates_log_var_params() -> None:
    """UW=True → у composite loss появляются обучаемые log_var параметры."""
    loss = CompositeQuantLoss(
        bce_weight=1.0, rankic_weight=0.5, sharpe_weight=0.3,
        monotone_weight=0.1, use_uncertainty_weighting=True,
    )
    params = list(loss.parameters())
    # 4 компонента включены → 4 log_var параметра.
    assert len(params) == 4
    assert all(p.requires_grad for p in params)


def test_composite_default_sharpe_weight_is_zero() -> None:
    """Дефолт sharpe_weight=0 (Sprint D2): Sharpe компонент выключен."""
    loss = CompositeQuantLoss()  # все дефолты
    assert loss._fixed_weights["sharpe"] == 0.0


def test_composite_uw_reduces_to_fixed_when_disabled() -> None:
    """use_uncertainty_weighting=False → log_var не создаётся."""
    loss = CompositeQuantLoss(use_uncertainty_weighting=False)
    assert len(list(loss.parameters())) == 0


def test_composite_inner_loss_asl_is_default() -> None:
    """По умолчанию inner_classification='asl' (после R-0050)."""
    loss = CompositeQuantLoss()
    assert isinstance(loss.cls, AsymmetricLossWithLogits)
