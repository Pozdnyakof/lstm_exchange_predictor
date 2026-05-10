"""Meta-labeling: вторичный classifier поверх Primary для фильтрации сделок.

Lopez de Prado, *Advances in Financial ML* (2018), глава 3.

## Идея

Primary (LightGBM на всех 90+ признаках) отвечает на вопрос:
   "Цена пойдёт вверх или вниз?"

Meta (отдельный classifier на ~10 признаках, включая Primary's prediction)
отвечает на вопрос:
   "Стоит ли действовать на этом конкретном сигнале Primary?"

Meta получает на вход:
   - Primary's prediction (как фичу)
   - Контекст: волатильность, спред, режим (HI2), время суток
И выдаёт probability того, что **трейд закроется в плюс**.

Финальный сигнал: торгуем если Primary > T1 AND Meta > T2.

## Зачем это работает

- Primary имеет конфликт: BCE-loss минимизируется и при хорошем
  ranking, и при выходе на prior. Meta развязывает эти задачи —
  ранжирование остаётся за Primary, **бинарное решение** (trade/skip)
  делает Meta.
- Meta видит meta-target = "трейд был прибыльным" (бинарный),
  что НЕ совпадает с Primary's target = "цена пошла вверх" (тоже
  бинарный, но шире — включает не-прибыльные движения < cost).
- Mета учится на ОШИБКАХ Primary: где Primary говорит "BUY", но
  это плохая идея (низкая ликвидность, режим high-vol).

## Pipeline

1. **OOF Primary**: 5-fold TimeSeriesSplit на train → out-of-sample
   primary predictions для всего train.
2. **Final Primary**: обучается на полном train (для inference).
3. **Meta train**: train_df + primary_h{h} (OOF) + meta_target_h{h}
   (= 1 если lr > 2·cost, иначе 0).
4. **Meta val**: val_df + primary predictions Final Primary + meta_target.
5. **Meta**: LightGBM на meta features → predicts meta_target.
6. **Inference**: Primary → Meta → final signal where Meta > threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .lgbm_pipeline import LightGBMConfig, LightGBMPipeline

logger = logging.getLogger(__name__)


def compute_lgbm_oof_per_horizon(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    horizons: tuple[int, ...],
    *,
    cfg: LightGBMConfig | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """OOF predictions через TimeSeriesSplit. Возвращает DataFrame с
    индексом train_df и колонками ``primary_h{h}`` per horizon.

    TimeSeriesSplit обеспечивает строгую chronological-разбивку:
    fold k обучается ТОЛЬКО на данных ДО fold k. Это важно для
    финансов, где случайный shuffle даёт look-ahead через
    автокорреляцию между близкими барами.

    Первые ~1/(n_splits+1) баров остаются БЕЗ OOF (первый фолд
    использует их для обучения, не для предсказания). Эти строки
    получают NaN в выводе и должны фильтроваться вызывающей стороной.
    """
    if cfg is None:
        cfg = LightGBMConfig()
    n = len(train_df)
    # Используем numpy-массивы для записи по позициям. ``out.loc[idx, col] = ...``
    # ломается на multi-ticker DataFrame: один timestamp матчит несколько
    # строк (по одной на тикер), и broadcast'инг скаляра/массива на N×k
    # позиций даёт length mismatch.
    out_arrays: dict[str, np.ndarray] = {
        f"primary_h{h}": np.full(n, np.nan, dtype=np.float32)
        for h in horizons
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(np.arange(n))):
        tr_sub = train_df.iloc[tr_idx]
        te_sub = train_df.iloc[te_idx]
        for h in horizons:
            _fit_oof_one_fold_horizon(
                tr_sub, te_sub, te_idx, h, feature_cols, cfg,
                out_arrays[f"primary_h{h}"],
            )
        logger.info(
            "OOF fold %d/%d done (train=%d, test=%d)",
            fold_idx + 1, n_splits, len(tr_sub), len(te_sub),
        )
    # Собираем итоговый DataFrame с СОХРАНЕНИЕМ оригинального индекса
    # (чтобы downstream код мог join'ить по timestamp+ticker).
    return pd.DataFrame(out_arrays, index=train_df.index)


def _fit_oof_one_fold_horizon(
    tr_sub: pd.DataFrame,
    te_sub: pd.DataFrame,
    te_positions: np.ndarray,
    horizon: int,
    feature_cols: list[str],
    cfg: LightGBMConfig,
    out_buffer: np.ndarray,
) -> None:
    """Обучить LGBM на tr_sub, предсказать на te_sub, записать в out_buffer
    по позициям te_positions[mask_te]."""
    target_col = f"target_h{horizon}"
    if target_col not in tr_sub.columns:
        return
    mask_tr = tr_sub[target_col].notna().to_numpy()
    mask_te = te_sub[target_col].notna().to_numpy()
    if not mask_tr.any() or not mask_te.any():
        return
    # .iloc безопасен на дубликатах индекса (multi-ticker), .loc нет.
    X_tr = tr_sub.iloc[mask_tr][feature_cols]
    y_tr = tr_sub.iloc[mask_tr][target_col].astype(int).to_numpy()
    X_te = te_sub.iloc[mask_te][feature_cols]
    model = lgb.LGBMClassifier(**cfg.to_lgb_params())
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_te)[:, 1]
    positions_in_full = te_positions[mask_te]
    out_buffer[positions_in_full] = preds.astype(np.float32)


def build_meta_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    cost_per_trade: float = 0.001,
    profit_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Добавить meta_target_h{h} колонки.

    meta_target = 1 если ``lr_h{h} > profit_multiplier · cost_per_trade``.
    Иначе 0. NaN там, где lr недоступен.

    profit_multiplier=2 — заходим только если ожидаемый профит >= 2× costs
    (margin of safety). Можно ужать до 1.0 для более «жадного» режима.
    """
    out = df.copy()
    threshold = float(profit_multiplier * cost_per_trade)
    for h in horizons:
        lr_col = f"lr_h{h}"
        meta_col = f"meta_target_h{h}"
        if lr_col not in df.columns:
            continue
        lr = df[lr_col]
        meta_target = (lr > threshold).astype(np.float32)
        meta_target[lr.isna()] = np.nan
        out[meta_col] = meta_target
    return out


def add_primary_predictions_wide(
    df: pd.DataFrame,
    primary_long: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    timestamp_col: str = "timestamp",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Long-form primary predictions → wide-merge в df.

    ``df`` имеет DatetimeIndex и колонку ``ticker``.
    ``primary_long`` имеет колонки ``[timestamp, ticker, horizon, mean]``
    (формат :meth:`LightGBMPipeline.predict`).

    Возвращает копию ``df`` с добавленными ``primary_h{h}`` колонками.
    Для строк без матча — NaN.
    """
    if primary_long.empty:
        out = df.copy()
        for h in horizons:
            out[f"primary_h{h}"] = np.nan
        return out
    primary_wide = primary_long.pivot_table(
        index=[timestamp_col, ticker_col], columns="horizon", values="mean",
    ).reset_index()
    primary_wide.columns = [
        timestamp_col, ticker_col,
        *[f"primary_h{int(h)}" for h in primary_wide.columns[2:]],
    ]
    primary_wide[timestamp_col] = pd.to_datetime(
        primary_wide[timestamp_col], utc=True,
    )
    df_reset = df.reset_index()
    if "begin" in df_reset.columns:
        df_reset = df_reset.rename(columns={"begin": timestamp_col})
    elif df_reset.columns[0] != timestamp_col:
        df_reset = df_reset.rename(columns={df_reset.columns[0]: timestamp_col})
    df_reset[timestamp_col] = pd.to_datetime(df_reset[timestamp_col], utc=True)
    merged = df_reset.merge(
        primary_wide, on=[timestamp_col, ticker_col], how="left",
    )
    return merged.set_index(timestamp_col)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class MetaLabelingPipeline:
    """Two-stage tree-based pipeline: Primary + Meta.

    После fit'а имеет два LightGBMPipeline объекта (primary, meta).
    Predict возвращает оба набора предсказаний. Стратегия использует
    их вместе: trade if primary > T1 AND meta > T2.
    """

    horizons: tuple[int, ...]
    primary_features: list[str]
    meta_features: list[str]
    primary_cfg: LightGBMConfig
    meta_cfg: LightGBMConfig
    cost_per_trade: float = 0.001
    profit_multiplier: float = 2.0
    n_oof_splits: int = 5

    primary: LightGBMPipeline | None = None
    meta: LightGBMPipeline | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict[str, dict]:
        """Train Primary (full train) + Meta (на OOF train + val).

        Возвращает summary: {primary: {...}, meta: {...}} с per-horizon
        метриками для отчёта.
        """
        # Step 1: OOF predictions for primary (для обучения meta)
        logger.info("Step 1: Computing primary OOF on train (%d rows, %d folds)",
                    len(train_df), self.n_oof_splits)
        oof = compute_lgbm_oof_per_horizon(
            train_df, self.primary_features, self.horizons,
            cfg=self.primary_cfg, n_splits=self.n_oof_splits,
        )

        # Step 2: Train final Primary on full train (для inference)
        logger.info("Step 2: Training final Primary on full train")
        self.primary = LightGBMPipeline(
            horizons=self.horizons,
            feature_cols=self.primary_features,
            cfg=self.primary_cfg,
        )
        self.primary.fit(train_df, val_df)

        # Step 3: Build meta train dataset
        logger.info("Step 3: Building meta train (OOF + meta targets)")
        meta_train = train_df.copy()
        for h in self.horizons:
            meta_train[f"primary_h{h}"] = oof[f"primary_h{h}"]
        meta_train = build_meta_targets(
            meta_train, self.horizons,
            cost_per_trade=self.cost_per_trade,
            profit_multiplier=self.profit_multiplier,
        )
        # Подменяем target_h{h} на meta_target_h{h} — LightGBMPipeline
        # ожидает target_h{h} для обучения, нам надо чтобы он учил
        # meta-задачу (прибыльность сделки), а не direction.
        meta_train_for_pipeline = meta_train.copy()
        for h in self.horizons:
            meta_train_for_pipeline[f"target_h{h}"] = meta_train[f"meta_target_h{h}"]

        # Step 4: Build meta val dataset
        logger.info("Step 4: Building meta val (using final Primary preds)")
        val_primary_preds = self.primary.predict(val_df)
        meta_val = add_primary_predictions_wide(
            val_df, val_primary_preds, self.horizons,
        )
        meta_val = build_meta_targets(
            meta_val, self.horizons,
            cost_per_trade=self.cost_per_trade,
            profit_multiplier=self.profit_multiplier,
        )
        meta_val_for_pipeline = meta_val.copy()
        for h in self.horizons:
            meta_val_for_pipeline[f"target_h{h}"] = meta_val[f"meta_target_h{h}"]

        # Step 5: Train Meta
        logger.info("Step 5: Training Meta on OOF train + val (%d feats)",
                    len(self.meta_features))
        # Drop rows with NaN OOF (первые 1/(n+1) баров)
        train_mask = meta_train_for_pipeline[
            [f"primary_h{h}" for h in self.horizons]
        ].notna().all(axis=1)
        meta_train_clean = meta_train_for_pipeline[train_mask]
        logger.info("  Meta train after OOF NaN drop: %d rows", len(meta_train_clean))
        self.meta = LightGBMPipeline(
            horizons=self.horizons,
            feature_cols=self.meta_features,
            cfg=self.meta_cfg,
        )
        self.meta.fit(meta_train_clean, meta_val_for_pipeline)

        # Build summary
        summary = {
            "primary": {
                h: {
                    "best_iter": r.best_iteration,
                    "val_log_loss": r.val_log_loss,
                    "val_auc": r.val_auc,
                }
                for h, r in self.primary.models.items()
            },
            "meta": {
                h: {
                    "best_iter": r.best_iteration,
                    "val_log_loss": r.val_log_loss,
                    "val_auc": r.val_auc,
                }
                for h, r in self.meta.models.items()
            },
        }
        return summary

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Сделать предсказания обеих моделей.

        Возвращает (primary_long, meta_long) — оба в long-form
        (timestamp, ticker, horizon, mean, std), как у :meth:`LightGBMPipeline.predict`.
        """
        if self.primary is None or self.meta is None:
            msg = "fit() not called yet"
            raise RuntimeError(msg)
        primary_preds = self.primary.predict(df)
        # Build meta features by adding primary preds wide
        df_with_primary = add_primary_predictions_wide(
            df, primary_preds, self.horizons,
        )
        # Replace ticker column (lost in reset_index)
        meta_preds = self.meta.predict(df_with_primary)
        return primary_preds, meta_preds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """Сохранить Primary + Meta + конфиг в одну директорию.

        Layout::

            output_dir/
              primary/   ← LightGBMPipeline.save()
              meta/      ← LightGBMPipeline.save()
              meta_labeling.json   ← features, cost_per_trade, etc
        """
        import json

        if self.primary is None or self.meta is None:
            msg = "fit() not called yet — nothing to save"
            raise RuntimeError(msg)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.primary.save(output_dir / "primary")
        self.meta.save(output_dir / "meta")
        manifest = {
            "horizons": list(self.horizons),
            "primary_features": list(self.primary_features),
            "meta_features": list(self.meta_features),
            "cost_per_trade": float(self.cost_per_trade),
            "profit_multiplier": float(self.profit_multiplier),
            "n_oof_splits": int(self.n_oof_splits),
        }
        (output_dir / "meta_labeling.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved MetaLabelingPipeline to %s", output_dir)

    @classmethod
    def load(cls, output_dir: Path) -> "MetaLabelingPipeline":
        """Восстановить pipeline из директории, созданной :meth:`save`.

        Configs LightGBM не сохраняются отдельно — они нужны только для
        обучения, в восстановленном пайплайне используются default'ы.
        """
        import json

        manifest = json.loads(
            (output_dir / "meta_labeling.json").read_text(encoding="utf-8"),
        )
        primary = LightGBMPipeline.load(output_dir / "primary")
        meta = LightGBMPipeline.load(output_dir / "meta")
        pipeline = cls(
            horizons=tuple(manifest["horizons"]),
            primary_features=list(manifest["primary_features"]),
            meta_features=list(manifest["meta_features"]),
            primary_cfg=LightGBMConfig(),
            meta_cfg=LightGBMConfig(),
            cost_per_trade=float(manifest["cost_per_trade"]),
            profit_multiplier=float(manifest["profit_multiplier"]),
            n_oof_splits=int(manifest["n_oof_splits"]),
        )
        pipeline.primary = primary
        pipeline.meta = meta
        return pipeline


def _merge_primary_meta_lr(
    val_primary: pd.DataFrame,
    val_meta: pd.DataFrame,
    val_lr_targets: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Merge primary + meta + lr per (timestamp, ticker) на одном горизонте."""
    h_prim = val_primary[val_primary["horizon"] == horizon][
        ["timestamp", "ticker", "horizon", "mean"]
    ].rename(columns={"mean": "primary"})
    h_meta = val_meta[val_meta["horizon"] == horizon][
        ["timestamp", "ticker", "horizon", "mean"]
    ].rename(columns={"mean": "meta"})
    h_lr = val_lr_targets[val_lr_targets["horizon"] == horizon][
        ["timestamp", "ticker", "horizon", "actual"]
    ].rename(columns={"actual": "lr"})
    if h_prim.empty or h_meta.empty or h_lr.empty:
        return pd.DataFrame()
    return h_prim.merge(
        h_meta, on=["timestamp", "ticker", "horizon"], how="inner",
    ).merge(h_lr, on=["timestamp", "ticker", "horizon"], how="inner")


def _evaluate_threshold_pair(
    merged: pd.DataFrame,
    T_prim: float,
    T_meta_abs: float,
    pct: float,
    cost_per_trade: float,
) -> dict:
    """Применить (T_prim, T_meta_abs) к merged val и вернуть метрики."""
    mask = (merged["primary"] > T_prim) & (merged["meta"] > T_meta_abs)
    n = int(mask.sum())
    if n == 0:
        return {
            "T_prim": T_prim, "meta_pct": pct, "T_meta_abs": T_meta_abs,
            "n_trades": 0, "mean_pnl": float("nan"),
        }
    mean_pnl = float(merged.loc[mask, "lr"].mean() - cost_per_trade)
    return {
        "T_prim": T_prim, "meta_pct": pct, "T_meta_abs": T_meta_abs,
        "n_trades": n, "mean_pnl": mean_pnl,
    }


def joint_max_pnl_thresholds(
    val_primary: pd.DataFrame,
    val_meta: pd.DataFrame,
    val_lr_targets: pd.DataFrame,
    horizon: int,
    *,
    primary_thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65),
    meta_percentiles: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0),
    cost_per_trade: float = 0.001,
    min_trades: int = 50,
) -> tuple[float, float, list[dict]]:
    """Найти (T_prim, T_meta) maximizing mean(lr - cost) on val для horizon.

    **Sprint 1.5 fix**: meta-distribution не достигает абсолютного 0.5 на
    коротких горизонтах (для h=6 mean meta = 0.157). Поэтому T_meta задаём
    через **percentile** val-meta-распределения. T_meta_pct=10 значит
    «топ-10% самых уверенных meta-предсказаний», что **гарантирует**
    непустой n_buy на любом горизонте.

    Возвращает (best_T_prim, best_T_meta_abs, sweep_table). T_meta_abs —
    абсолютный порог, рассчитанный из val-meta-распределения.

    ``min_trades`` — минимальное число сделок на val для валидной точки.
    Если нет ни одной комбинации с min_trades — fallback к лучшей по PnL.
    """
    merged = _merge_primary_meta_lr(
        val_primary, val_meta, val_lr_targets, horizon,
    )
    if merged.empty:
        return float(primary_thresholds[0]), float("nan"), []

    sweep: list[dict] = []
    best_idx = None
    best_pnl = -np.inf
    fallback_idx = None
    fallback_pnl = -np.inf
    for T_prim in primary_thresholds:
        for pct in meta_percentiles:
            T_meta_abs = float(np.percentile(merged["meta"], 100.0 - pct))
            row = _evaluate_threshold_pair(
                merged, T_prim, T_meta_abs, pct, cost_per_trade,
            )
            sweep.append(row)
            pnl = row["mean_pnl"]
            if not np.isfinite(pnl):
                continue
            if pnl > fallback_pnl:
                fallback_pnl = pnl
                fallback_idx = len(sweep) - 1
            if row["n_trades"] >= min_trades and pnl > best_pnl:
                best_pnl = pnl
                best_idx = len(sweep) - 1

    if best_idx is None:
        if fallback_idx is None:
            return float(primary_thresholds[0]), float("nan"), sweep
        chosen = sweep[fallback_idx]
        logger.warning(
            "joint_max_pnl: нет комбинаций >= %d trades, fallback к best PnL: "
            "T_prim=%.3f, T_meta=%.3f",
            min_trades, chosen["T_prim"], chosen["T_meta_abs"],
        )
        return float(chosen["T_prim"]), float(chosen["T_meta_abs"]), sweep
    chosen = sweep[best_idx]
    return float(chosen["T_prim"]), float(chosen["T_meta_abs"]), sweep


__all__ = [
    "MetaLabelingPipeline",
    "add_primary_predictions_wide",
    "build_meta_targets",
    "compute_lgbm_oof_per_horizon",
    "joint_max_pnl_thresholds",
]
