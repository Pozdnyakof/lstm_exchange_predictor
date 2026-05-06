"""Офлайн-отчёт о тестировании (§3.4 ВКР).

Прогоняет обученную модель через тестовый период:
    - MC Dropout инференс;
    - двухступенчатый фильтр сигналов;
    - aggregate бэктест;
    - per-ticker бэктест;
    - random monkeys (3σ-критерий).

Артефакты сохраняются в data/processed/runtime/ и потребляются
страницей /report веб-интерфейса.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from graduate_work.backtest import (
    compute_metrics,
    run_backtest,
    run_per_ticker_backtest,
    run_random_portfolios,
)
from graduate_work.backtest.engine import prices_from_full_frame
from graduate_work.config import default_config
from graduate_work.data.storage import load_processed
from graduate_work.features import build_dataset
from graduate_work.serving import load_artifact
from graduate_work.strategy import SignalGenerator, build_predictions_frame
from graduate_work.training import mc_predict


def _save_runtime(runtime: Path, name: str, payload: dict) -> None:
    with (runtime / name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    runtime = cfg.paths.data_processed / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    # 1) Загрузить артефакт-пакет (модель + scaler + meta).
    loaded = load_artifact(cfg.paths.checkpoints, device="cpu")
    logging.info(
        "Loaded model: %d features, %d horizons, trained at %s",
        loaded.meta.num_features, loaded.meta.num_horizons, loaded.meta.training_date,
    )

    # 2) Восстановить тестовую выборку (точно так же, как при обучении).
    prepared = build_dataset(cfg.data, cfg.paths, persist=False)
    test = prepared.test
    if test["x"].shape[0] == 0:
        msg = "Test split is empty - проверьте train_ratio/val_ratio"
        raise RuntimeError(msg)

    # 3) MC Dropout инференс.
    mean, std = mc_predict(
        loaded.model, test["x"],
        mc_passes=cfg.training.mc_passes,
        batch_size=cfg.training.batch_size,
        device="cpu",
    )

    horizons = tuple(loaded.meta.horizons)
    predictions = build_predictions_frame(
        timestamps=test["timestamp"],
        tickers=test["ticker"],
        mean=mean,
        std=std,
        horizons=horizons,
    )

    # 4) Двухступенчатый фильтр.
    signals = SignalGenerator(cfg.trading).generate(predictions)

    # 5) Цены тестового периода + buffer на хвостовые позиции.
    full_path = cfg.paths.data_processed / "features.parquet"
    full = load_processed(full_path)
    if "timestamp" in full.columns:
        full = full.set_index("timestamp")
    full.index = pd.to_datetime(full.index, utc=True)
    test_start = pd.to_datetime(min(test["timestamp"]), utc=True)
    test_end = pd.to_datetime(max(test["timestamp"]), utc=True) + pd.Timedelta(days=max(horizons))
    test_prices = prices_from_full_frame(
        full.loc[(full.index >= test_start) & (full.index <= test_end)],
    )

    # 6) Aggregate бэктест.
    bt = run_backtest(signals, test_prices, cfg.trading)
    metrics = compute_metrics(bt.equity, bt.trades)
    logging.info("Aggregate metrics: %s", metrics)

    # 7) Per-ticker бэктест.
    per_ticker = run_per_ticker_backtest(signals, test_prices, cfg.trading)
    logging.info(
        "Per-ticker: %d рядов, профит-тикеров %d/%d",
        len(per_ticker),
        int((per_ticker["total_return"] > 0).sum()) if not per_ticker.empty else 0,
        len(per_ticker),
    )

    # 8) Random monkeys.
    avg_h = (
        int(round(bt.trades["horizon"].mean()))
        if not bt.trades.empty else int(np.mean(horizons))
    )
    random_report = run_random_portfolios(
        test_prices,
        cfg.trading,
        avg_horizon=avg_h,
        strategy_final=metrics["final_equity"],
        seed=cfg.training.seed,
    )
    logging.info(
        "Random monkeys: mean=%.2f std=%.2f threshold=%.2f strategy=%.2f z=%.2f sig=%s",
        random_report.mean, random_report.std, random_report.threshold_value,
        random_report.strategy_final, random_report.strategy_z_score,
        random_report.is_significant,
    )

    # 9) Сохранить артефакты для страницы /report.
    signals.to_parquet(runtime / "signals.parquet", index=False)
    predictions.to_parquet(runtime / "predictions.parquet", index=False)
    test_prices.reset_index().rename(columns={"index": "timestamp"}).to_parquet(
        runtime / "prices.parquet", index=False,
    )
    bt.equity.rename("equity").reset_index().rename(columns={"index": "timestamp"}).to_parquet(
        runtime / "equity.parquet", index=False,
    )
    bt.trades.to_parquet(runtime / "trades.parquet", index=False)
    per_ticker.to_parquet(runtime / "per_ticker_metrics.parquet", index=False)

    _save_runtime(runtime, "metrics.json", metrics)
    _save_runtime(
        runtime, "random_report.json",
        {
            "mean": random_report.mean,
            "std": random_report.std,
            "sigma_threshold": random_report.sigma_threshold,
            "threshold_value": random_report.threshold_value,
            "strategy_final": random_report.strategy_final,
            "strategy_z_score": random_report.strategy_z_score,
            "is_significant": bool(random_report.is_significant),
            "initial_capital": cfg.trading.initial_capital,
            "final_returns": random_report.final_returns.tolist(),
        },
    )
    logging.info("Saved runtime artifacts to %s", runtime)


if __name__ == "__main__":
    main()
