"""Возобновить прерванную загрузку биржевых данных.

Делает три вещи:
    1. Если в `data/raw/` (на верхнем уровне) лежат CSV тикеров - переносит
       их в `data/raw/moex/` (целевая папка orchestrator-а).
    2. Запускает `download_all`. Внутренний механизм
       `_batches.download_in_batches` сам:
         - прочитает хвост каждого существующего CSV (без полной загрузки),
         - определит max-метку времени,
         - пропустит уже скачанные кварталы,
         - дозабирает оставшиеся.
       На каждый чанк - `download_batch_retries` попыток с
       экспоненциальным backoff.
    3. Печатает сводку: какие тикеры были полностью готовы, какие
       докачаны, какие не удалось добрать.
"""

from __future__ import annotations

import logging
import shutil

import _bootstrap  # noqa: F401

from graduate_work.config import default_config
from graduate_work.data.orchestrator import download_all


def _migrate_top_level_csvs(cfg) -> int:
    """Перенести data/raw/<TICKER>.csv в data/raw/moex/<TICKER>.csv.

    CSV не читаем - просто проверяем имя файла и перемещаем.
    Если в обоих местах одноимённые файлы - оставляем уже лежащий
    в moex/ (он, скорее всего, актуальнее), а верхнеуровневый удаляем.
    """
    raw = cfg.paths.data_raw
    moex = raw / "moex"
    moex.mkdir(parents=True, exist_ok=True)

    moved = 0
    for ticker in cfg.data.tickers:
        src = raw / f"{ticker}.csv"
        dst = moex / f"{ticker}.csv"
        if not src.exists():
            continue
        if dst.exists():
            logging.warning(
                "Both %s and %s exist; keeping moex/ version, removing top-level.",
                src, dst,
            )
            src.unlink()
            continue
        shutil.move(str(src), str(dst))
        moved += 1
        logging.info("Moved %s -> %s", src, dst)
    return moved


def _summarise(cfg) -> None:
    moex = cfg.paths.data_raw / "moex"
    if not moex.exists():
        return
    present = sorted(p.stem for p in moex.glob("*.csv"))
    missing = [t for t in cfg.data.tickers if t not in present]
    logging.info("Tickers present (%d): %s", len(present), ", ".join(present))
    if missing:
        logging.warning("Tickers still missing (%d): %s", len(missing), ", ".join(missing))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    logging.info(
        "Resume target: %s tickers, period %s..%s, batch=%d months, retries=%d",
        len(cfg.data.tickers), cfg.data.start_date, cfg.data.end_date,
        cfg.data.download_batch_months, cfg.data.download_batch_retries,
    )

    moved = _migrate_top_level_csvs(cfg)
    if moved:
        logging.info("Migrated %d CSV(s) from data/raw/ to data/raw/moex/", moved)

    download_all(cfg.data, cfg.paths)
    _summarise(cfg)


if __name__ == "__main__":
    main()
