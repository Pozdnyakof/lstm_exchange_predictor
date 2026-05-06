"""Headless-обучение и сохранение артефакт-пакета.

Альтернатива блокноту notebooks/training_pipeline.ipynb для CI / serverless
сценариев. Сам блокнот - основной режим для §3.2 защиты, скрипт повторяет
ровно те же шаги без графиков.
"""

from __future__ import annotations

import dataclasses as _dc
import logging

import _bootstrap  # noqa: F401

from graduate_work.config import default_config
from graduate_work.features import build_dataset
from graduate_work.model import ConvLstmRegressor
from graduate_work.serving import ModelMeta, save_artifact
from graduate_work.serving.artifact import now_iso
from graduate_work.training import Trainer, set_seed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    set_seed(cfg.training.seed)

    prepared = build_dataset(cfg.data, cfg.paths, persist=True)
    logging.info(
        "Dataset: features=%d, horizons=%d, train=%d val=%d test=%d",
        prepared.num_features, len(cfg.data.horizons),
        prepared.train["x"].shape[0],
        prepared.val["x"].shape[0],
        prepared.test["x"].shape[0],
    )

    model = ConvLstmRegressor(
        input_dim=prepared.num_features,
        num_horizons=len(cfg.data.horizons),
        cfg=cfg.model,
    )
    trainer = Trainer(model, cfg.training)
    history = trainer.fit(prepared.train, prepared.val)
    logging.info(
        "Training done: best_epoch=%d val_loss=%.6f",
        history.best_epoch, history.best_val_loss,
    )

    meta = ModelMeta(
        feature_cols=list(prepared.feature_cols),
        target_cols=list(prepared.target_cols),
        horizons=list(cfg.data.horizons),
        window_size=cfg.data.window_size,
        num_features=prepared.num_features,
        num_horizons=len(cfg.data.horizons),
        model_config=_dc.asdict(cfg.model),
        training_date=now_iso(),
        tickers=list(cfg.data.tickers),
    )
    paths = save_artifact(model, prepared.scaler, meta, cfg.paths.checkpoints)
    for k, p in paths.items():
        logging.info("artifact %s: %s (%.1f KB)", k, p, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
