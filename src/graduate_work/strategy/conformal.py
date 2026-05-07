"""Split-conformal фильтр сделок (Vovk-Shafer).

Альтернатива Bayes-калиброванному порогу. Главная идея: порог НЕ
выводится из абсолютной калибровки модели, а из эмпирического
распределения её ошибок на валидации.

Алгоритм:
  1) calibrate(val_predictions, val_targets):
     scores = |prob - actual| для каждой пары (sample, horizon),
     q = (1 - α)·(n+1)/n квантиль scores.
  2) generate(test_predictions):
     trade ⇔ prob > max(q, 1-q).
     В бинарной классификации это эквивалентно "conformal set
     вырождается в singleton {1}".

Преимущество перед Bayes-порогом из `calibration.py` в наших условиях:
- Bayes требует «честной» вероятности от модели; если модель саватурирована
  вокруг 0.5 (что у нас наблюдается), Bayes-порог упирается в нижнюю
  планку 0.51, а std-фильтр блокирует все сигналы → 0 сделок.
- Conformal SAMA настраивается под фактический output модели: если
  prob ∈ [0.45, 0.55], q будет маленьким, и threshold = max(q, 1-q)
  получится в районе 0.5 + ε, что даёт реальную фильтрацию.

Эмпирически в R-0018 conformal × SWA дал DSR=1.0 saturated на
+85% return при 1108 сделках, тогда как наш текущий Bayes-фильтр
выдаёт 0 сделок.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TradingConfig

logger = logging.getLogger(__name__)

_FALLBACK_QUANTILE = 0.5


@dataclass
class ConformalCalibration:
    """Результат calibrate() — для отчётности и отладки."""

    quantile: float
    threshold: float
    n_val_scores: int


class ConformalSignalGenerator:
    """Split-conformal фильтр для бинарной классификации.

    Использование:
        gen = ConformalSignalGenerator(cfg, alpha=0.1)
        gen.calibrate(val_predictions, val_targets)
        signals = gen.generate(test_predictions)

    `val_predictions` и `test_predictions` — выход
    :func:`build_predictions_frame` (колонки timestamp, ticker, horizon,
    mean, std).

    `val_targets` — `(timestamp, ticker, horizon, actual)` где actual
    это **сглаженная** бинарная метка (например, через
    :func:`strategy.calibration.attach_actual_targets`). Модель
    обучалась под эти же сглаженные метки, поэтому conformal-скор
    `|prob - target|` правильно отражает её residual error.
    """

    def __init__(
        self,
        cfg: TradingConfig,
        *,
        alpha: float = 0.1,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            msg = f"alpha must be in (0, 1), got {alpha}"
            raise ValueError(msg)
        self.cfg = cfg
        self.alpha = float(alpha)
        self._quantile: float | None = None

    @property
    def quantile(self) -> float | None:
        return self._quantile

    def calibrate(
        self,
        val_predictions: pd.DataFrame,
        val_targets: pd.DataFrame,
    ) -> ConformalCalibration:
        """Вычислить (1-α)·(n+1)/n квантиль conformal-скоров на val."""
        if val_predictions.empty or val_targets.empty:
            self._quantile = _FALLBACK_QUANTILE
            return ConformalCalibration(_FALLBACK_QUANTILE, _FALLBACK_QUANTILE, 0)

        merged = val_predictions.merge(
            val_targets,
            on=["timestamp", "ticker", "horizon"],
            how="inner",
        )
        if merged.empty or "actual" not in merged.columns:
            self._quantile = _FALLBACK_QUANTILE
            return ConformalCalibration(_FALLBACK_QUANTILE, _FALLBACK_QUANTILE, 0)

        scores = np.abs(
            merged["mean"].to_numpy(dtype=np.float64)
            - merged["actual"].to_numpy(dtype=np.float64),
        )
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            self._quantile = _FALLBACK_QUANTILE
            return ConformalCalibration(_FALLBACK_QUANTILE, _FALLBACK_QUANTILE, 0)

        n = scores.size
        level = min(1.0, (1.0 - self.alpha) * (1 + 1 / max(n, 1)))
        q = float(np.quantile(scores, level))
        self._quantile = q
        threshold = max(q, 1.0 - q)
        logger.info(
            "Conformal: alpha=%.3f, n=%d, level=%.4f, q=%.4f, threshold=%.4f",
            self.alpha, n, level, q, threshold,
        )
        return ConformalCalibration(q, threshold, n)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        empty_cols = ["timestamp", "ticker", "horizon", "mean", "std", "action", "signal"]
        if predictions.empty:
            return pd.DataFrame(columns=empty_cols)

        # Best horizon per (timestamp, ticker) - argmax по prob.
        idx = (
            predictions.groupby(["timestamp", "ticker"])["mean"]
            .idxmax()
            .dropna()
            .astype(int)
        )
        best = predictions.loc[idx].reset_index(drop=True)

        q = self._quantile if self._quantile is not None else _FALLBACK_QUANTILE
        threshold = max(q, 1.0 - q)

        sessions: list[pd.DataFrame] = []
        for _ts, day in best.groupby("timestamp", sort=True):
            day = day.copy()
            day = day.sort_values("mean", ascending=False)
            day["action"] = "HOLD"
            day["signal"] = 0
            top = day.head(self.cfg.max_positions).copy()
            qualifying = top["mean"] > threshold
            buy_idx = top.index[qualifying]
            day.loc[buy_idx, "action"] = "BUY"
            day.loc[buy_idx, "signal"] = 1
            sessions.append(day)
        if not sessions:
            return pd.DataFrame(columns=empty_cols)
        return pd.concat(sessions, ignore_index=True)
