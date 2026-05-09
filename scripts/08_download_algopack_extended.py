"""Расширенная загрузка ALGOPACK — 14 ликвидных тикеров + 6 фьючерсов.

Используется как одноразовый скрипт перед развёртыванием LightGBM-only
пайплайна с cross-sectional фичами.

Тикеры (top-14 ликвидных MOEX equities):
  SBER, VTBR, GAZP, LKOH, GMKN, ROSN, NVTK, MGNT,
  TATN, MTSS, MOEX, NLMK, CHMF, ALRS

Algopack-продукты для equities:
  - tradestats — 5-min split buy/sell
  - orderstats — placed/cancel orders
  - obstats — order-book imbalance
  - hi2 — daily Herfindahl

Перпы для FUTOI (отдельный проход, futoi не принимает tickers):
  SBERF, GAZPF, USDRUBF, CNYRUBF, IMOEXF, GLDRUBF

Запуск:
  cd graduate_work
  python scripts/08_download_algopack_extended.py
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Bootstrap
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Загружаем .env (ALGOPACK_TOKEN)
load_dotenv(_REPO / ".env")
if "ALGOPACK_TOKEN" not in os.environ:
    raise RuntimeError(
        f"ALGOPACK_TOKEN не найден в окружении. Проверь {_REPO / '.env'}",
    )

from graduate_work.config import default_config  # noqa: E402
from graduate_work.data.orchestrator import download_algopack  # noqa: E402

logger = logging.getLogger(__name__)


# Тикеры выбраны по ликвидности MOEX 2024 + наличию ALGOPACK-покрытия с 2020.
# SBER+VTBR уже скачены — добавляем 12 новых.
TICKERS_EQUITIES = (
    "SBER", "VTBR",                            # уже было — переcкачиваем для consistency
    "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",   # топ-капитализация
    "MGNT", "TATN", "MTSS", "MOEX",            # ритейл/телеком/биржа
    "NLMK", "CHMF", "ALRS",                    # металлурги + ALRS (золото/алмазы)
)

# Algopack-продукты для equities (без FUTOI — он только для futures)
EQUITY_PRODUCTS = ("tradestats", "orderstats", "obstats", "hi2")

# Перпы для FUTOI — daily, мало данных, но critical для positioning
FUTURES_FOR_FUTOI = ("SBERF", "GAZPF", "USDRUBF", "CNYRUBF", "IMOEXF", "GLDRUBF")


def _run_phase_equities(base_cfg) -> None:
    """Phase 1: 14 equities × {tradestats, orderstats, obstats, hi2}."""
    eq_cfg = dataclasses.replace(
        base_cfg.data,
        tickers=TICKERS_EQUITIES,
        algopack_products=EQUITY_PRODUCTS,
        algopack_market="eq",
        download_workers=5,
    )
    logger.info(
        "=== Phase 1: equities × algopack ===\n"
        "  tickers: %d, products: %s, date range: %s .. %s",
        len(eq_cfg.tickers), eq_cfg.algopack_products,
        eq_cfg.start_date, eq_cfg.end_date,
    )
    results = download_algopack(eq_cfg, base_cfg.paths)
    logger.info("Phase 1 done: %d combinations", len(results))


def _run_phase_futures(base_cfg) -> None:
    """Phase 2: 6 futures × FUTOI с 2024-10-01.

    FUTOI endpoint имеет данные ТОЛЬКО с 2024-10-01 (meta `futoi.dates`,
    R-0053). Запросы за более ранний период возвращают empty.
    """
    fo_cfg = dataclasses.replace(
        base_cfg.data,
        tickers=FUTURES_FOR_FUTOI,
        algopack_products=("futoi",),
        algopack_market="fo",
        download_workers=3,
        start_date="2024-10-01",
    )
    logger.info(
        "=== Phase 2: futures × FUTOI ===\n"
        "  futures: %d, date range: %s .. %s",
        len(fo_cfg.tickers), fo_cfg.start_date, fo_cfg.end_date,
    )
    results = download_algopack(fo_cfg, base_cfg.paths)
    logger.info("Phase 2 done: %d futures with FUTOI", len(results))


def _print_summary(base_cfg) -> None:
    """Final summary: file count + size by product."""
    raw_dir = base_cfg.paths.data_raw / "algopack"
    if not raw_dir.exists():
        return
    for product in (*EQUITY_PRODUCTS, "futoi"):
        p = raw_dir / product
        if not p.exists():
            logger.info("  %s: 0 файлов (нет директории)", product)
            continue
        files = list(p.glob("*.csv"))
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        logger.info("  %s: %d файлов, %.1f MB", product, len(files), size_mb)
    logger.info("Готово. Можно копировать в Drive: %s", raw_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    base_cfg = default_config()
    _run_phase_equities(base_cfg)
    _run_phase_futures(base_cfg)
    _print_summary(base_cfg)


if __name__ == "__main__":
    main()
