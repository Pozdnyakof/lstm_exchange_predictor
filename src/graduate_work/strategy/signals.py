"""Двухступенчатый фильтр торговых сигналов.

Поддерживает оба режима из §1.2 ВКР через один интерфейс:

* **regression**: фильтр по mean (порог `min_expected_return`) + std-cap
  (`max_uncertainty`). При всех отрицательных mean → HOLD.
* **classification**: фильтр по probability (порог `probability_threshold`,
  калибруется через Bayes на val) + std-cap (`max_probability_std`).
  При всех вероятностях ниже порога → HOLD.

В обоих случаях:
- ранжируем активы по mean / probability;
- берём top-K по `max_positions`;
- учитываем Šidák-коррекцию при выборе argmax-горизонта (§3.4 ВКР).

Унифицированный формат вывода:
    timestamp, ticker, horizon, mean, std, action, signal

где ``signal`` ∈ {-1, 0, +1} (long / hold / short).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TradingConfig


def build_predictions_frame(
    timestamps: np.ndarray,
    tickers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Развернуть тензоры (N, H) в длинный фрейм (N×H, _)."""
    n, h = mean.shape
    if h != len(horizons):
        msg = f"mean has {h} horizons but cfg has {len(horizons)}"
        raise ValueError(msg)
    rows: list[dict] = []
    for i in range(n):
        for j, hz in enumerate(horizons):
            rows.append(
                {
                    "timestamp": timestamps[i],
                    "ticker": str(tickers[i]),
                    "horizon": int(hz),
                    "mean": float(mean[i, j]),
                    "std": float(std[i, j]),
                },
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


class SignalGenerator:
    """Двухступенчатый фильтр сигналов.

    Поведение зависит от ``mode`` (regression / classification),
    задаётся аргументом конструктора. По умолчанию - regression
    (для обратной совместимости со старыми тестами).
    """

    def __init__(self, cfg: TradingConfig, *, mode: str = "regression") -> None:
        if mode not in ("regression", "classification"):
            msg = f"Unknown mode: {mode}"
            raise ValueError(msg)
        self.cfg = cfg
        self.mode = mode

    # ------------------------------------------------------------------
    # Вспомогательное: argmax по горизонту
    # ------------------------------------------------------------------
    def best_per_ticker(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Для каждого (timestamp, ticker) - горизонт с наибольшим mean."""
        if predictions.empty:
            return predictions.assign(action="HOLD", signal=0)
        idx = (
            predictions.groupby(["timestamp", "ticker"])["mean"]
            .idxmax()
            .dropna()
            .astype(int)
        )
        best = predictions.loc[idx].reset_index(drop=True)
        return best

    # ------------------------------------------------------------------
    # Эффективные пороги (с argmax-коррекцией)
    # ------------------------------------------------------------------
    def _effective_thresholds(self, n_horizons: int) -> tuple[float, float]:
        """Вернуть (порог mean/probability, порог std)."""
        if self.mode == "classification":
            base = self.cfg.probability_threshold
            std_cap = self.cfg.max_probability_std
        else:
            base = self.cfg.min_expected_return
            std_cap = self.cfg.max_uncertainty
        return self._apply_horizon_correction(base, n_horizons), std_cap

    def _apply_horizon_correction(self, base: float, n_horizons: int) -> float:
        if n_horizons <= 1:
            return base
        if self.mode == "classification":
            # Šidák / Bonferroni для probability-порога:
            # при argmax из N горизонтов FWER компенсируется так, чтобы
            # T_eff^N = base ⇒ T_eff = base^(1/N).
            mode = self.cfg.selection_correction
            if mode == "sidak":
                return float(base ** (1.0 / n_horizons))
            if mode == "bonferroni":
                tail = max(1.0 - base, 0.0)
                return float(1.0 - tail / n_horizons)
            return base
        # regression: множитель из конфига
        return base * self.cfg.horizon_argmax_correction

    # ------------------------------------------------------------------
    # Главная процедура
    # ------------------------------------------------------------------
    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        empty_cols = ["timestamp", "ticker", "horizon", "mean", "std", "action", "signal"]
        if predictions.empty:
            return pd.DataFrame(columns=empty_cols)

        best = self.best_per_ticker(predictions)
        n_h = predictions["horizon"].nunique()
        thr_mean, thr_std = self._effective_thresholds(n_h)

        sessions: list[pd.DataFrame] = []
        for _ts, day in best.groupby("timestamp", sort=True):
            day = day.copy()
            # Глобальный фильтр: если нет ни одного "позитивного" - HOLD.
            if self.mode == "regression":
                global_ok = (day["mean"] > 0.0).any()
            else:
                # classification: нужна хотя бы одна вероятность > порога.
                global_ok = (day["mean"] > thr_mean).any()
            if not global_ok:
                day["action"] = "HOLD"
                day["signal"] = 0
                sessions.append(day)
                continue
            # Шаг 2: ранжируем top-K и проверяем std-cap.
            day = day.sort_values("mean", ascending=False)
            day["action"] = "HOLD"
            day["signal"] = 0
            top = day.head(self.cfg.max_positions).copy()
            qualifying = (top["mean"] >= thr_mean) & (top["std"] <= thr_std)
            buy_idx = top.index[qualifying]
            day.loc[buy_idx, "action"] = "BUY"
            day.loc[buy_idx, "signal"] = 1
            sessions.append(day)
        if not sessions:
            return pd.DataFrame(columns=empty_cols)
        return pd.concat(sessions, ignore_index=True)
