"""Adaptive Conformal Inference (Gibbs & Candès, 2021).

[arXiv:2106.00170](https://arxiv.org/abs/2106.00170),
*Adaptive Conformal Inference Under Distribution Shift*.

Ключевое отличие от split-conformal в `conformal.py`: вместо одного
фиксированного квантиля на val, ACI **адаптирует уровень α онлайн**
по фактическому покрытию на стриме недавних предсказаний:

    α_{t+1} = α_t + γ · (target_α − empirical_miscoverage_t)

где ``empirical_miscoverage_t = mean over recent window of
1{actual NOT in conformal_set}``, ``γ`` — learning rate.

Это даёт **формальную гарантию долговременного покрытия**
``1 - target_α`` при любом распределении дрифтов (стационарном или нет).
Для финансов это критично — режимы рынка меняются, и фиксированный
порог `0.6` или `q=0.7` со временем теряет калибровку.

Использование::

    aci = AdaptiveConformalPredictor(target_alpha=0.1, gamma=0.005)
    aci.calibrate(val_predictions, val_targets)  # warm-start с split-conformal
    # Online inference: после каждого нового предсказания вызываем update().
    for t, row in test_predictions.iterrows():
        signal = aci.predict_signal(row)         # ACTION + threshold
        # ... после реализации actual_t:
        aci.update(predicted=row['mean'], actual=actual_t)

В нашем пайплайне для backtest (offline) удобнее вычислить
адаптивные пороги ``threshold_t`` по всему стриму сразу через
:meth:`replay`, не вмешиваясь в сторонний код.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conformal import ConformalSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class ACIState:
    """Состояние adaptive conformal: текущий α и история miscoverage."""

    alpha: float
    miscoverage_count: int = 0
    total_count: int = 0

    @property
    def empirical_miscoverage(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.miscoverage_count / self.total_count


class AdaptiveConformalPredictor:
    """Online ACI поверх directional conformal-калибровки.

    Каждый шаг (по времени бара) обновляет α в зависимости от того,
    попало ли реальное направление в conformal set:

    - target_alpha — желаемая долгосрочная miscoverage rate (0.1 = 90% покрытие);
    - gamma        — learning rate для α (0.005 — рекомендация paper'а);
    - alpha_min    — нижняя граница α (по умолчанию 0.001) — защищает от
                     схлопывания threshold в 1 (никаких сделок);
    - alpha_max    — верхняя граница (0.5) — защищает от деградации в
                     unconditional baseline.

    State хранится per-horizon — для каждого горизонта своё α.
    """

    def __init__(
        self,
        *,
        target_alpha: float = 0.1,
        gamma: float = 0.005,
        alpha_min: float = 0.001,
        alpha_max: float = 0.5,
    ) -> None:
        if not 0.0 < target_alpha < 1.0:
            msg = f"target_alpha must be in (0, 1), got {target_alpha}"
            raise ValueError(msg)
        if not 0.0 < alpha_min < alpha_max < 1.0:
            msg = (
                f"need 0 < alpha_min < alpha_max < 1; "
                f"got min={alpha_min}, max={alpha_max}"
            )
            raise ValueError(msg)
        self.target_alpha = float(target_alpha)
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        # Per-horizon state (заполняется в calibrate()).
        self._states: dict[int, ACIState] = {}
        # Per-horizon scores из calibration set, нужны для quantile-lookup
        # на каждом adaptive шаге.
        self._scores: dict[int, np.ndarray] = {}

    @property
    def state_summary(self) -> pd.DataFrame:
        """Текущее состояние: α и empirical miscoverage по горизонтам."""
        rows = [
            {
                "horizon": h,
                "alpha": s.alpha,
                "empirical_miscoverage": s.empirical_miscoverage,
                "n_observed": s.total_count,
            }
            for h, s in self._states.items()
        ]
        return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

    @staticmethod
    def _scores_for_horizon(sub: pd.DataFrame) -> np.ndarray:
        """Достать conformal-скоры из под-фрейма одного горизонта.

        Резервно — если позитивов нет, берём 1-prob по всем; если совсем
        пусто — fallback в равномерный 0.5 чтобы threshold() не падал.
        """
        positives = sub[sub["actual"] >= 0.5]
        source = positives if not positives.empty else sub
        scores = 1.0 - source["mean"].to_numpy(dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return np.array([0.5], dtype=np.float64)
        return scores

    def calibrate(
        self,
        val_predictions: pd.DataFrame,
        val_targets: pd.DataFrame,
    ) -> None:
        """Warm-start: берём directional conformal scores с val'а
        как baseline distribution для последующих online-обновлений."""
        if val_predictions.empty or val_targets.empty:
            msg = "Cannot calibrate ACI on empty validation data"
            raise ValueError(msg)
        merged = val_predictions.merge(
            val_targets, on=["timestamp", "ticker", "horizon"], how="inner",
        )
        if merged.empty or "actual" not in merged.columns:
            msg = "Validation merge produced empty frame; check column names"
            raise ValueError(msg)

        for h, sub in merged.groupby("horizon"):
            self._scores[int(h)] = self._scores_for_horizon(sub)
            self._states[int(h)] = ACIState(alpha=self.target_alpha)

        logger.info(
            "ACI calibrated on %d horizons; warm-start alpha=%.3f",
            len(self._states), self.target_alpha,
        )

    def threshold(self, horizon: int) -> float:
        """Текущий probability-threshold для данного горизонта.

        Threshold = 1 - quantile(scores, 1 - α_t). Здесь scores = 1 - prob
        для validation positives. Соотношение: ``α↑ → quantile↓ → q↓ →
        threshold↑`` (становится строже, меньше BUY'ев). Это директивная
        интерпретация: высокий target_α = «допускаем больше пропусков
        positives ради меньшего числа ложных срабатываний».
        """
        if horizon not in self._states:
            msg = f"horizon {horizon} not calibrated; call calibrate() first"
            raise KeyError(msg)
        state = self._states[horizon]
        scores = self._scores[horizon]
        n = scores.size
        # (1 - α)·(n+1)/n — стандартная split-conformal коррекция конечной
        # выборки. Защищаемся клипом, чтобы level не вышел за 1.
        level = min(1.0, (1.0 - state.alpha) * (1 + 1 / max(n, 1)))
        q = float(np.quantile(scores, level))
        return float(np.clip(1.0 - q, 0.0, 1.0))

    def update(
        self,
        *,
        predicted_prob: float,
        actual: float,
        horizon: int,
    ) -> None:
        """Online-обновление α для одного горизонта.

        ``predicted_prob`` — выход модели (вероятность UP),
        ``actual`` — реализованная сглаженная метка (0/1 либо в
        [0, 1] при label smoothing).

        Conformal set принимает, что P(UP) ≥ threshold(h) ↔ сигнал BUY.
        Miscoverage = 1, если actual=1 но threshold(h) > predicted_prob
        (положительный класс был, но мы его не предсказали).
        """
        if horizon not in self._states:
            msg = f"horizon {horizon} not calibrated"
            raise KeyError(msg)
        state = self._states[horizon]
        threshold = self.threshold(horizon)
        # Directional miscoverage: positive class missed.
        positive = float(actual) >= 0.5
        miscovered = positive and (float(predicted_prob) < threshold)
        state.total_count += 1
        if miscovered:
            state.miscoverage_count += 1
        # Gibbs-Candès update rule.
        new_alpha = state.alpha + self.gamma * (
            self.target_alpha - (1.0 if miscovered else 0.0)
        )
        state.alpha = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))

    def _replay_step(self, row, *, idx: int, buffers: dict) -> None:
        """Обработать одну строку replay-стрима в-place.

        ``buffers`` — словарь с numpy-массивами ``thresholds/alphas/
        signals/miscov``, длиной равной общему числу строк.
        """
        h = int(row.horizon)
        if h not in self._states:
            buffers["thresholds"][idx] = 1.0
            buffers["alphas"][idx] = self.target_alpha
            return
        thr = self.threshold(h)
        buffers["thresholds"][idx] = thr
        buffers["alphas"][idx] = self._states[h].alpha
        signal = int(float(row.mean) > thr)
        buffers["signals"][idx] = signal
        actual = getattr(row, "actual", float("nan"))
        if np.isnan(actual):
            return
        self.update(
            predicted_prob=float(row.mean), actual=float(actual), horizon=h,
        )
        if int(actual >= 0.5) == 1 and signal == 0:
            buffers["miscov"][idx] = 1

    def replay(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Offline-симуляция ACI на отсортированном по времени потоке.

        Возвращает копию ``predictions`` с добавленными колонками:
        ``threshold``, ``alpha``, ``signal`` (1 если ``mean > threshold``,
        иначе 0), ``miscovered`` (1 если actual=1 и signal=0).
        """
        if predictions.empty:
            return predictions.assign(
                threshold=pd.Series(dtype=float),
                alpha=pd.Series(dtype=float),
                signal=pd.Series(dtype=int),
                miscovered=pd.Series(dtype=int),
            )
        merged = predictions.merge(
            actuals[["timestamp", "ticker", "horizon", "actual"]],
            on=["timestamp", "ticker", "horizon"],
            how="left",
        ).sort_values(["timestamp", "ticker", "horizon"]).reset_index(drop=True)
        n = len(merged)
        buffers = {
            "thresholds": np.zeros(n, dtype=np.float64),
            "alphas": np.zeros(n, dtype=np.float64),
            "signals": np.zeros(n, dtype=np.int8),
            "miscov": np.zeros(n, dtype=np.int8),
        }
        for i, row in enumerate(merged.itertuples(index=False)):
            self._replay_step(row, idx=i, buffers=buffers)
        merged["threshold"] = buffers["thresholds"]
        merged["alpha"] = buffers["alphas"]
        merged["signal"] = buffers["signals"]
        merged["miscovered"] = buffers["miscov"]
        return merged


def aci_signals_to_actions(
    aci_frame: pd.DataFrame,
    *,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Конвертировать вывод :meth:`AdaptiveConformalPredictor.replay`
    в формат, совместимый с :func:`backtest.run_backtest`.

    Берём best-horizon argmax по ``mean`` для каждой пары
    (timestamp, ticker), оставляем top-``max_positions`` тикеров с
    ``signal=1`` за timestamp как BUY, остальные — HOLD. Так получается
    тот же интерфейс, что у :class:`ConformalSignalGenerator.generate`.
    """
    cols = ["timestamp", "ticker", "horizon", "mean", "std", "action", "signal"]
    if aci_frame.empty:
        return pd.DataFrame(columns=cols)
    # Best horizon per (timestamp, ticker) — argmax по mean (как в conformal.py).
    idx = (
        aci_frame.groupby(["timestamp", "ticker"])["mean"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best = aci_frame.loc[idx].reset_index(drop=True)

    sessions: list[pd.DataFrame] = []
    for _ts, day in best.groupby("timestamp", sort=True):
        day = day.copy()
        day = day.sort_values("mean", ascending=False)
        day["action"] = "HOLD"
        top = day.head(int(max_positions)).copy()
        buy_mask = top["signal"] == 1
        day.loc[top.index[buy_mask], "action"] = "BUY"
        sessions.append(day[cols])
    if not sessions:
        return pd.DataFrame(columns=cols)
    return pd.concat(sessions, ignore_index=True)


class DtACIPredictor:
    """DtACI: Dynamically-tuned Adaptive Conformal Inference.

    Gibbs & Candès, **JMLR 2024**, *Conformal Inference for Online
    Prediction with Arbitrary Distribution Shifts*
    ([arXiv:2208.08401](https://arxiv.org/abs/2208.08401)).

    Расширение ACI на «grid γ-кандидатов»: вместо одного фиксированного
    learning rate γ держим K разных γ_k и онлайн-выбираем взвешенную
    комбинацию через exponential weighted averaging. При mis-specified
    γ (как было в R-0050: фиксированный γ=0.005 → α упёрлось в 0.5)
    DtACI автоматически переключается на γ, дающий лучшее покрытие.

    На каждом шаге:
    1. Каждый эксперт k поддерживает свой α_t^(k) по правилу ACI с γ_k.
    2. После наблюдения actual: считаем pinball loss каждого эксперта.
    3. Веса экспертов w_k ∝ exp(−η · cumulative_loss_k).
    4. Финальное α = Σ_k w_k · α_t^(k).

    Per-horizon — каждый горизонт держит свой набор экспертов.
    """

    def __init__(
        self,
        *,
        target_alpha: float = 0.1,
        gammas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
        eta: float = 0.1,
        alpha_min: float = 0.001,
        alpha_max: float = 0.5,
    ) -> None:
        if not 0.0 < target_alpha < 1.0:
            msg = f"target_alpha must be in (0, 1), got {target_alpha}"
            raise ValueError(msg)
        if not gammas:
            msg = "gammas must be non-empty"
            raise ValueError(msg)
        if any(g <= 0 for g in gammas):
            msg = f"all gammas must be > 0, got {gammas}"
            raise ValueError(msg)
        if eta <= 0:
            msg = f"eta must be > 0, got {eta}"
            raise ValueError(msg)
        self.target_alpha = float(target_alpha)
        self.gammas = tuple(float(g) for g in gammas)
        self.eta = float(eta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        # Per-horizon state: alphas[horizon] = list[float] длины K (по экспертам);
        # weights[horizon] = list[float] длины K; cumulative_loss[horizon] = ...;
        # scores[horizon] = ndarray (warm-start scores).
        self._alphas: dict[int, list[float]] = {}
        self._weights: dict[int, list[float]] = {}
        self._cum_loss: dict[int, list[float]] = {}
        self._scores: dict[int, np.ndarray] = {}
        self._n_observed: dict[int, int] = {}

    @property
    def state_summary(self) -> pd.DataFrame:
        """DataFrame: для каждого горизонта — текущая комбинированная α
        и веса экспертов."""
        rows = []
        for h in sorted(self._alphas):
            alpha_combined = self._combined_alpha(h)
            row = {
                "horizon": h,
                "alpha_combined": alpha_combined,
                "n_observed": self._n_observed[h],
            }
            for k, gamma in enumerate(self.gammas):
                row[f"alpha_g={gamma:.3f}"] = self._alphas[h][k]
                row[f"w_g={gamma:.3f}"] = self._weights[h][k]
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _scores_for_horizon(sub: pd.DataFrame) -> np.ndarray:
        positives = sub[sub["actual"] >= 0.5]
        source = positives if not positives.empty else sub
        scores = 1.0 - source["mean"].to_numpy(dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return np.array([0.5], dtype=np.float64)
        return scores

    def calibrate(
        self,
        val_predictions: pd.DataFrame,
        val_targets: pd.DataFrame,
    ) -> None:
        """Warm-start с val-данных: одна и та же conformal-база для всех экспертов."""
        if val_predictions.empty or val_targets.empty:
            msg = "Cannot calibrate DtACI on empty validation data"
            raise ValueError(msg)
        merged = val_predictions.merge(
            val_targets, on=["timestamp", "ticker", "horizon"], how="inner",
        )
        if merged.empty or "actual" not in merged.columns:
            msg = "Validation merge produced empty frame; check column names"
            raise ValueError(msg)
        k = len(self.gammas)
        for h, sub in merged.groupby("horizon"):
            self._scores[int(h)] = self._scores_for_horizon(sub)
            self._alphas[int(h)] = [self.target_alpha] * k
            self._weights[int(h)] = [1.0 / k] * k
            self._cum_loss[int(h)] = [0.0] * k
            self._n_observed[int(h)] = 0
        logger.info(
            "DtACI calibrated on %d horizons; gammas=%s",
            len(self._scores), self.gammas,
        )

    def _combined_alpha(self, horizon: int) -> float:
        """Взвешенная комбинация α по экспертам (нормализованные веса)."""
        ws = np.asarray(self._weights[horizon])
        alphas = np.asarray(self._alphas[horizon])
        norm = ws.sum()
        if norm <= 1e-12:
            return float(self.target_alpha)
        return float((ws * alphas).sum() / norm)

    def _expert_threshold(self, horizon: int, alpha: float) -> float:
        """Threshold под конкретное α (как в обычном ACI)."""
        scores = self._scores[horizon]
        n = scores.size
        level = min(1.0, (1.0 - alpha) * (1 + 1 / max(n, 1)))
        q = float(np.quantile(scores, level))
        return float(np.clip(1.0 - q, 0.0, 1.0))

    def threshold(self, horizon: int) -> float:
        """Threshold по комбинированной α."""
        if horizon not in self._alphas:
            msg = f"horizon {horizon} not calibrated"
            raise KeyError(msg)
        return self._expert_threshold(horizon, self._combined_alpha(horizon))

    def _pinball_loss(
        self, alpha: float, miscovered: int,
    ) -> float:
        """Pinball loss для онлайн-обучения весов экспертов.

        ``alpha · (1 - miscovered)`` если covered (хотим α малым),
        ``(1 - alpha) · miscovered`` если miscovered (хотим α меньше
        чтобы быть готовым к промахам — но здесь мы не покрыли).
        Стандарт DtACI Gibbs-Candès JMLR 2024 §3.
        """
        return float(alpha * (1 - miscovered) + (1.0 - alpha) * miscovered)

    def update(
        self,
        *,
        predicted_prob: float,
        actual: float,
        horizon: int,
    ) -> None:
        """Online-update всех экспертов и весов."""
        if horizon not in self._alphas:
            msg = f"horizon {horizon} not calibrated"
            raise KeyError(msg)
        positive = float(actual) >= 0.5
        self._n_observed[horizon] += 1
        for k, gamma in enumerate(self.gammas):
            alpha_k = self._alphas[horizon][k]
            thr_k = self._expert_threshold(horizon, alpha_k)
            miscov = int(positive and (float(predicted_prob) < thr_k))
            # ACI rule
            new_alpha = alpha_k + gamma * (self.target_alpha - miscov)
            self._alphas[horizon][k] = float(
                np.clip(new_alpha, self.alpha_min, self.alpha_max),
            )
            # Per-expert pinball loss → cumulative → exp-weights.
            self._cum_loss[horizon][k] += self._pinball_loss(alpha_k, miscov)
        # Renormalize веса как softmax по −η · cum_loss (стандарт EWA).
        cum = np.asarray(self._cum_loss[horizon])
        log_w = -self.eta * (cum - cum.min())  # shift для numerical stability
        w = np.exp(log_w)
        w = w / max(w.sum(), 1e-12)
        self._weights[horizon] = w.tolist()

    def _replay_step(self, row, *, idx: int, buffers: dict) -> None:
        """Один шаг replay-стрима в-place, аналог ACI._replay_step."""
        h = int(row.horizon)
        if h not in self._alphas:
            buffers["thresholds"][idx] = 1.0
            buffers["alphas"][idx] = self.target_alpha
            return
        thr = self.threshold(h)
        buffers["thresholds"][idx] = thr
        buffers["alphas"][idx] = self._combined_alpha(h)
        signal = int(float(row.mean) > thr)
        buffers["signals"][idx] = signal
        actual = getattr(row, "actual", float("nan"))
        if np.isnan(actual):
            return
        self.update(
            predicted_prob=float(row.mean), actual=float(actual), horizon=h,
        )
        if int(actual >= 0.5) == 1 and signal == 0:
            buffers["miscov"][idx] = 1

    def replay(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Offline-симуляция аналогично ACI.replay, с DtACI-обновлением."""
        if predictions.empty:
            return predictions.assign(
                threshold=pd.Series(dtype=float),
                alpha=pd.Series(dtype=float),
                signal=pd.Series(dtype=int),
                miscovered=pd.Series(dtype=int),
            )
        merged = predictions.merge(
            actuals[["timestamp", "ticker", "horizon", "actual"]],
            on=["timestamp", "ticker", "horizon"], how="left",
        ).sort_values(["timestamp", "ticker", "horizon"]).reset_index(drop=True)
        n = len(merged)
        buffers = {
            "thresholds": np.zeros(n, dtype=np.float64),
            "alphas": np.zeros(n, dtype=np.float64),
            "signals": np.zeros(n, dtype=np.int8),
            "miscov": np.zeros(n, dtype=np.int8),
        }
        for i, row in enumerate(merged.itertuples(index=False)):
            self._replay_step(row, idx=i, buffers=buffers)
        merged["threshold"] = buffers["thresholds"]
        merged["alpha"] = buffers["alphas"]
        merged["signal"] = buffers["signals"]
        merged["miscovered"] = buffers["miscov"]
        return merged


__all__ = [
    "ACIState",
    "AdaptiveConformalPredictor",
    "DtACIPredictor",
    "aci_signals_to_actions",
]


# Re-export для удобства: split-conformal как warm-start.
SplitConformal = ConformalSignalGenerator
