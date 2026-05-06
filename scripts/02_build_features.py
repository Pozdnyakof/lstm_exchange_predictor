"""Собрать признаковую таблицу из data/raw/ и сохранить в data/processed/."""

from __future__ import annotations

import logging
import pickle

import _bootstrap  # noqa: F401

from graduate_work.config import default_config
from graduate_work.features import build_dataset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    prepared = build_dataset(cfg.data, cfg.paths, persist=True)

    out = cfg.paths.data_processed / "prepared_dataset.pkl"
    with out.open("wb") as f:
        pickle.dump(
            {
                "feature_cols": prepared.feature_cols,
                "target_cols": prepared.target_cols,
                "scaler": prepared.scaler.to_dict(),
                "train": prepared.train,
                "val": prepared.val,
                "test": prepared.test,
            },
            f,
        )
    logging.info("Saved prepared dataset to %s", out)
    logging.info(
        "Shapes: train=%s val=%s test=%s",
        prepared.train["x"].shape,
        prepared.val["x"].shape,
        prepared.test["x"].shape,
    )


if __name__ == "__main__":
    main()
