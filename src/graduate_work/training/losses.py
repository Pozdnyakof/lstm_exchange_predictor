"""Loss-функции для regression и classification режимов.

Включает quant-специализированные loss'ы (RankIC, Sharpe, Monotone)
и композитный wrapper, прямо оптимизирующий метрики, которые
используются в торговой стратегии.

Литература:
- Microsoft Qlib (qlib.contrib.loss): RankIC и пр.
- Lim, Zohren, Roberts (2019), *Enhancing Time Series Momentum Strategies
  with Deep Neural Networks* — Sharpe loss.
- Lopez de Prado (2018), *Advances in Financial ML* ch. 16.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import DataConfig, TradingConfig, TrainingConfig

logger = logging.getLogger(__name__)
_EPS = 1e-8


class WeightedBCEWithLogits(nn.Module):
    """Бинарный cross-entropy на логитах, multi-target по горизонтам.

    Принимает (B, H) логиты и (B, H) сглаженные метки в [0, 1].
    Per-element BCE, потом среднее по B*H. Опциональный per-sample вес.

    ``pos_weight`` (shape (H,)) умножает вклад positives в loss и
    предотвращает prediction collapse при дисбалансе классов: без него
    модель быстро сходится к константе ≈ P(UP) и не использует вход.
    Канонический рецепт: ``pos_weight = (1 - P(UP)) / P(UP)``.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        # register_buffer чтобы тензор автоматически уезжал на тот же
        # device, что и модуль, через .to(...). None разрешён в новых
        # PyTorch (>=1.10) и трактуется как «буфера нет».
        self.register_buffer(
            "pos_weight",
            pos_weight.float() if pos_weight is not None else None,
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none",
            pos_weight=self.pos_weight,
        )
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


class FocalBCEWithLogits(nn.Module):
    """Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Полезен при дисбалансе классов или для подавления «лёгких»
    примеров (которые модель уже хорошо классифицирует).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        focal = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * focal * bce
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


class AsymmetricLossWithLogits(nn.Module):
    """Asymmetric Loss (Ridnik & Ben-Baruch, ICCV 2021,
    [arXiv:2009.14119](https://arxiv.org/abs/2009.14119)).

    Раздельные γ⁺/γ⁻ + probability shift ``clip``. Доказанно SOTA на
    extreme negative dominance — точно наш случай (P(UP)=0.29 для h=6).

    Формула:
        L⁺ = (1 - p)^γ⁺ · log(p)
        L⁻ = p_shifted^γ⁻ · log(1 - p_shifted),  p_shifted = max(p - clip, 0)
        L  = -mean(y · L⁺ + (1 - y) · L⁻)

    γ⁻ > γ⁺ → сильнее подавляет «лёгкие» negatives, которые доминируют
    градиент при class imbalance и тащат модель в predict-the-prior
    collapse.
    """

    def __init__(
        self,
        *,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0:
            msg = (
                f"gamma_pos/gamma_neg must be non-negative; "
                f"got pos={gamma_pos}, neg={gamma_neg}"
            )
            raise ValueError(msg)
        if not 0.0 <= clip < 1.0:
            msg = f"clip must be in [0, 1), got {clip}"
            raise ValueError(msg)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        # log(p) и log(1 - p_shifted) с защитой от 0.
        log_pos = torch.log(prob.clamp_min(_EPS))
        if self.clip > 0:
            prob_neg = (prob - self.clip).clamp_min(0.0)
        else:
            prob_neg = prob
        log_neg = torch.log((1.0 - prob_neg).clamp_min(_EPS))
        # Focal-веса с раздельными γ.
        focal_pos = (1.0 - prob).pow(self.gamma_pos) if self.gamma_pos > 0 else 1.0
        focal_neg = prob_neg.pow(self.gamma_neg) if self.gamma_neg > 0 else 1.0
        loss = -(target * focal_pos * log_pos + (1.0 - target) * focal_neg * log_neg)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


def class_balanced_pos_weight(
    y_train: np.ndarray,
    *,
    beta: float = 0.999,
) -> np.ndarray:
    """Class-Balanced pos_weight (Cui et al., CVPR 2019,
    [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)).

    Заменяет наивный ``(1 - P) / P``, который на больших n агрессивно
    усиливает minority и провоцирует prediction collapse. Использует
    «эффективное число сэмплов»: w_class ∝ (1 - β) / (1 - β^n_class).

    Возвращает per-horizon ``pos_weight = w_pos / w_neg`` shape (H,).
    Каждое значение нормировано так, что сумма w_pos + w_neg = 2 в
    каждом горизонте, чтобы общий масштаб loss оставался стабильным.
    """
    if not 0.0 < beta < 1.0:
        msg = f"beta must be in (0, 1), got {beta}"
        raise ValueError(msg)
    if y_train.ndim != 2:
        msg = f"y_train must be 2D (N, H), got shape {y_train.shape}"
        raise ValueError(msg)
    n_pos = (y_train >= 0.5).sum(axis=0).astype(np.float64)
    n_neg = (y_train < 0.5).sum(axis=0).astype(np.float64)
    # Эффективное число: при n→∞, eff_num → 1/(1-β); при n=0 → 0.
    eff_pos = 1.0 - np.power(beta, np.maximum(n_pos, 1.0))
    eff_neg = 1.0 - np.power(beta, np.maximum(n_neg, 1.0))
    w_pos = (1.0 - beta) / np.maximum(eff_pos, 1e-12)
    w_neg = (1.0 - beta) / np.maximum(eff_neg, 1e-12)
    # Нормируем так, чтобы w_pos + w_neg = 2 для сохранения масштаба.
    total = w_pos + w_neg
    w_pos = 2.0 * w_pos / np.maximum(total, 1e-12)
    w_neg = 2.0 * w_neg / np.maximum(total, 1e-12)
    return (w_pos / np.maximum(w_neg, 1e-12)).astype(np.float32)


class _CallableLoss(nn.Module):
    """Тонкий wrapper, принимающий опциональные ``weights`` для совместимости."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,  # noqa: ARG002 - не используется
    ) -> torch.Tensor:
        return self.base(preds, targets)


# ---------------------------------------------------------------------------
# Quant-специализированные loss'ы
# ---------------------------------------------------------------------------


def _soft_rank(values: torch.Tensor, *, regularization: float = 1.0) -> torch.Tensor:
    """Дифференцируемая soft-rank через softmax-сравнение пар.

    ``values`` shape (N,). Возвращает (N,) ranks в [0, N-1] в softmax-смысле:
    rank_i = sum_j sigmoid((v_i - v_j) / reg). При reg → 0 это hard-rank,
    при reg большом — почти константа.

    Сложность O(N²) — приемлемо для batch (B≤4096) и одного horizon'а
    за раз. Для огромных батчей перейти на pairwise loss.
    """
    diff = values.unsqueeze(0) - values.unsqueeze(1)   # (N, N): row j minus col i
    return torch.sigmoid(diff / regularization).sum(dim=0)


class RankICLoss(nn.Module):
    """Negative Spearman-style rank-IC между ``pred`` и ``target``.

    На каждом горизонте отдельно: rank-correlation сводится к Pearson на
    soft-ranks (у вас гладкая, дифференцируемая версия). Целевая метрика
    cross-sectional ranking (Microsoft Qlib).

    Loss = -mean_h pearson(soft_rank(pred_h), soft_rank(target_h)).

    ``regularization`` контролирует «жёсткость» ранжирования; 1.0 — стандарт
    из torchsort/Blondel et al. (2020).
    """

    def __init__(self, regularization: float = 1.0) -> None:
        super().__init__()
        if regularization <= 0:
            msg = f"regularization must be positive, got {regularization}"
            raise ValueError(msg)
        self.regularization = float(regularization)

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp_min(_EPS)
        return (a * b).sum() / denom

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        if pred.dim() != 2 or pred.shape != target.shape:
            msg = f"shapes mismatch: pred={pred.shape}, target={target.shape}"
            raise ValueError(msg)
        n_h = pred.shape[1]
        ic_per_h = []
        for h in range(n_h):
            p = _soft_rank(pred[:, h], regularization=self.regularization)
            t = _soft_rank(target[:, h], regularization=self.regularization)
            ic_per_h.append(self._pearson(p, t))
        return -torch.stack(ic_per_h).mean()


class SharpeLoss(nn.Module):
    """Дифференцируемая negative Sharpe-ratio loss.

    Lim-Zohren-Roberts (2019) формулировка: ``signal_t · ret_t − cost``
    интерпретируется как PnL шага, и оптимизируется ``-mean(PnL) /
    std(PnL)``.

    На вход — ``logits`` (B, H) или (B, 1) и ``lr`` (B, H) сырых
    лог-доходностей с тем же шейпом. Сигнал получается как
    ``2·sigmoid(logits) - 1 ∈ (-1, 1)`` для бинарной директивности.

    ``cost`` — ожидаемые транзакционные costs за round-trip
    (commission+slippage), вычитаются из каждой ставки. Без них Sharpe
    переоценивает realistичность стратегии.
    """

    def __init__(self, cost: float = 0.0) -> None:
        super().__init__()
        self.cost = float(cost)

    def forward(
        self, logits: torch.Tensor, lr: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != lr.shape:
            msg = f"shapes mismatch: logits={logits.shape}, lr={lr.shape}"
            raise ValueError(msg)
        signal = 2.0 * torch.sigmoid(logits) - 1.0      # (B, H), in (-1, 1)
        pnl = signal * lr - self.cost * signal.abs()    # cost пропорционален размеру ставки
        # Усреднённый по горизонтам Sharpe: считаем по каждой колонке,
        # потом mean. Так per-horizon шум не давит на общий signal.
        n_h = pnl.shape[1]
        sharpes = []
        for h in range(n_h):
            r = pnl[:, h]
            mean = r.mean()
            std = r.std(unbiased=False).clamp_min(_EPS)
            sharpes.append(mean / std)
        return -torch.stack(sharpes).mean()


class HorizonMonotoneRegularizer(nn.Module):
    """Штраф за нарушение монотонности `P(up)` по горизонтам.

    Если горизонты упорядочены ``h₁ < h₂ < ... < h_H``, ожидаем, что
    кумулятивная вероятность роста на длинном горизонте не меньше, чем
    на коротком (для большинства тикеров). Штраф:

        L = mean(relu(prob_h_i - prob_h_{i+1})²)

    Применяется на ВЕРОЯТНОСТЯХ (после sigmoid), не на logits, чтобы
    штраф был в правильной шкале.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        if weight < 0:
            msg = f"weight must be non-negative, got {weight}"
            raise ValueError(msg)
        self.weight = float(weight)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.new_zeros(())
        probs = torch.sigmoid(logits)                # (B, H)
        diff = probs[:, :-1] - probs[:, 1:]          # >0 = нарушение
        violation = F.relu(diff) ** 2
        return self.weight * violation.mean()


def _build_inner_classification_loss(
    name: str, *, pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    """Внутренний classification-loss для CompositeQuantLoss.

    Принимает 'bce' | 'focal' | 'asl' и собирает соответствующий
    модуль. Этот хелпер избавляет :class:`CompositeQuantLoss` от знаний
    про конкретные классы.
    """
    if name == "asl":
        return AsymmetricLossWithLogits()
    if name == "focal":
        return FocalBCEWithLogits()
    return WeightedBCEWithLogits(pos_weight=pos_weight)


class CompositeQuantLoss(nn.Module):
    """Взвешенная комбинация InnerCls + RankIC + Sharpe + Monotone.

    Loss = α·InnerCls(logits, target) +
           β·RankIC(logits, lr_target) +
           γ·Sharpe(logits, lr_target) +
           δ·MonotoneReg(logits)

    InnerCls = BCE | Focal | ASL (выбирается через ``inner_classification``).
    После R-0050 collapse-эпизода **по умолчанию ASL** — у BCE trivial-
    минимум на prior'е совпадает с минимумом RankIC и Sharpe, что и
    вызвало коллапс ансамбля в один minimum. ASL смещает minimum
    InnerCls с prior'а (Ridnik & Ben-Baruch ICCV 2021).

    **Uncertainty Weighting** (Kendall et al., CVPR 2018,
    [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)): если
    ``use_uncertainty_weighting=True``, веса заменяются на обучаемые
    ``1/(2σᵢ²)`` с регуляризатором ``log σᵢ``. Это автоматически
    балансирует разномасштабные компоненты без ручного подбора весов.
    """

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        rankic_weight: float = 0.5,
        sharpe_weight: float = 0.0,
        monotone_weight: float = 0.1,
        cost: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        inner_classification: str = "asl",
        use_uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self._fixed_weights = {
            "bce": float(bce_weight),
            "rankic": float(rankic_weight),
            "sharpe": float(sharpe_weight),
            "monotone": float(monotone_weight),
        }
        self.use_uncertainty_weighting = bool(use_uncertainty_weighting)
        # Per-component log-variance (обучаемые), если включен UW.
        # Иначе отсутствуют — используется фиксированный масштаб.
        if self.use_uncertainty_weighting:
            self.log_var = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(()))
                for k, w in self._fixed_weights.items() if w > 0
            })
        self.cls = _build_inner_classification_loss(
            inner_classification, pos_weight=pos_weight,
        )
        self.rankic = RankICLoss()
        self.sharpe = SharpeLoss(cost=cost)
        self.monotone = HorizonMonotoneRegularizer(weight=1.0)

    def _scale(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Масштабирование одной компоненты loss'а.

        Без UW: умножение на фиксированный вес из ``_fixed_weights``.
        С UW: ``L/(2σ²) + log σ`` — Kendall 2018 формула, где σ²=exp(log_var).
        """
        if not self.use_uncertainty_weighting or name not in self.log_var:
            return self._fixed_weights[name] * value
        log_var = self.log_var[name]
        return 0.5 * torch.exp(-log_var) * value + 0.5 * log_var

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lr_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._scale("bce", self.cls(logits, target))
        if lr_target is not None and self._fixed_weights["rankic"] > 0:
            loss = loss + self._scale("rankic", self.rankic(logits, lr_target))
        if lr_target is not None and self._fixed_weights["sharpe"] > 0:
            loss = loss + self._scale("sharpe", self.sharpe(logits, lr_target))
        if self._fixed_weights["monotone"] > 0:
            loss = loss + self._scale("monotone", self.monotone(logits))
        return loss


def _build_classification_loss(trading_cfg: TradingConfig | None) -> nn.Module:
    """Под-фабрика classification-loss по ``trading_cfg.loss_objective``."""
    objective = trading_cfg.loss_objective if trading_cfg is not None else "bce"
    if objective == "focal":
        return FocalBCEWithLogits(
            gamma=trading_cfg.focal_gamma,
            alpha=trading_cfg.focal_alpha,
        )
    if objective == "asl":
        return AsymmetricLossWithLogits(
            gamma_pos=trading_cfg.asl_gamma_pos,
            gamma_neg=trading_cfg.asl_gamma_neg,
            clip=trading_cfg.asl_clip,
        )
    if objective == "composite":
        cost = (
            2.0 * (trading_cfg.commission_rate + trading_cfg.slippage_rate)
            if trading_cfg is not None else 0.0
        )
        return CompositeQuantLoss(
            bce_weight=trading_cfg.composite_bce_weight,
            rankic_weight=trading_cfg.composite_rankic_weight,
            sharpe_weight=trading_cfg.composite_sharpe_weight,
            monotone_weight=trading_cfg.composite_monotone_weight,
            cost=cost,
            use_uncertainty_weighting=trading_cfg.composite_uncertainty_weighting,
            inner_classification=trading_cfg.composite_inner_loss,
        )
    return WeightedBCEWithLogits()


def build_loss_fn(
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    trading_cfg: TradingConfig | None = None,
) -> nn.Module:
    """Выбрать loss по режиму.

    classification + bce        → WeightedBCEWithLogits
    classification + focal      → FocalBCEWithLogits
    classification + asl        → AsymmetricLossWithLogits (Ridnik 2021)
    classification + composite  → CompositeQuantLoss
    regression                  → HuberLoss(delta=auto)
    """
    if data_cfg.mode == "classification":
        return _build_classification_loss(trading_cfg)
    return _CallableLoss(nn.HuberLoss(reduction="mean", delta=1.0))
