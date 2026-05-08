"""Полная локальная загрузка всех ALGOPACK + Calendars + Dividends.

После запуска:

* ``data/raw/algopack/<product>/<ticker>.csv`` — SuperCandles (TradeStats /
  OrderStats / OBStats), а так же FUTOI / HI2 / MegaAlerts если включены.
* ``data/raw/calendars/<product>.csv`` — 12 календарных продуктов:
  trading_days_{stock,futures,currency}, session_{stock,futures,currency},
  suspended_planned, settlecodes, security_changes, boards_history,
  futures_expirations, currency_settlement_shifts.
* ``data/raw/dividends/<ticker>.csv`` — история дивидендов по каждому
  тикеру (открытое ISS, без auth).

Подгрузка идёт батчами по cfg.download_batch_months месяцев — при сбое
сети возобновится с последнего успешного куска (idempotent).

Использование (локально или в Colab):

    export ALGOPACK_TOKEN="<your-bearer-token>"
    python scripts/08_download_algopack.py                    # всё подряд
    python scripts/08_download_algopack.py --skip-supercandles # только календари+дивиденды
    python scripts/08_download_algopack.py --tickers SBER GAZP --products tradestats

После завершения папка ``data/raw/`` готова к загрузке на Google Drive
(в `MyDrive/lstm_exchange/data/raw/Algopack/` пользователь кладёт
вручную).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

# Делаем скрипт запускаемым из любой директории.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graduate_work.config import default_config  # noqa: E402
from graduate_work.data.orchestrator import (  # noqa: E402
    download_algopack,
    download_calendars,
    download_dividends,
)

logger = logging.getLogger("scripts.algopack")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--products", nargs="+",
        default=["tradestats", "orderstats", "obstats"],
        help=(
            "ALGOPACK SuperCandles-продукты "
            "(default: tradestats orderstats obstats; "
            "ещё доступны: hi2 futoi)."
        ),
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Тикеры (default: cfg.data.tickers).",
    )
    parser.add_argument(
        "--market", default="eq", choices=["eq", "fo", "fx"],
        help="Рынок: eq (акции) | fo (фьючерсы) | fx (валюта).",
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--skip-supercandles", action="store_true",
        help="Не качать ALGOPACK-продукты (только календари + дивиденды).",
    )
    parser.add_argument(
        "--skip-calendars", action="store_true",
        help="Не качать календари MOEX.",
    )
    parser.add_argument(
        "--skip-dividends", action="store_true",
        help="Не качать дивиденды.",
    )
    return parser.parse_args()


def _build_data_cfg(args: argparse.Namespace, base_cfg):
    overrides: dict = {
        "algopack_products": tuple(args.products),
        "algopack_market": args.market,
    }
    if args.tickers:
        overrides["tickers"] = tuple(args.tickers)
    if args.start:
        overrides["start_date"] = args.start
    if args.end:
        overrides["end_date"] = args.end
    return dataclasses.replace(base_cfg.data, **overrides)


def _run_supercandles(args, data_cfg, paths) -> int:
    if args.skip_supercandles:
        logger.info("--skip-supercandles → пропускаем ALGOPACK SuperCandles")
        return 0
    if not os.environ.get("ALGOPACK_TOKEN"):
        logger.error(
            "ALGOPACK_TOKEN не задан. Получите Bearer-токен в MOEX "
            "DataShop / Passport и экспортируйте перед запуском.",
        )
        return 1
    logger.info(
        "ALGOPACK: %d продуктов × %d тикеров × range %s..%s",
        len(data_cfg.algopack_products), len(data_cfg.tickers),
        data_cfg.start_date, data_cfg.end_date,
    )
    out = download_algopack(data_cfg, paths)
    logger.info("Скачано (ticker, product) пар: %d", len(out))
    for (ticker, product), df in sorted(out.items()):
        logger.info("  %s/%s -> %d строк", product, ticker, len(df))
    return 0


def _run_calendars(args, data_cfg, paths) -> None:
    if args.skip_calendars:
        logger.info("--skip-calendars → пропускаем календари")
        return
    out = download_calendars(data_cfg, paths)
    logger.info("Календарей скачано: %d", len(out))
    for name, df in sorted(out.items()):
        logger.info("  %s -> %d строк", name, len(df))


def _run_dividends(args, data_cfg, paths) -> None:
    if args.skip_dividends:
        logger.info("--skip-dividends → пропускаем дивиденды")
        return
    out = download_dividends(data_cfg, paths)
    logger.info("Дивиденды скачаны для %d тикеров", len(out))
    for ticker, df in sorted(out.items()):
        logger.info("  %s -> %d записей", ticker, len(df))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = _parse_args()
    base_cfg = default_config()
    data_cfg = _build_data_cfg(args, base_cfg)
    base_cfg.paths.ensure()

    rc = _run_supercandles(args, data_cfg, base_cfg.paths)
    if rc:
        return rc
    _run_calendars(args, data_cfg, base_cfg.paths)
    _run_dividends(args, data_cfg, base_cfg.paths)
    logger.info("Готово. Папка data/raw/ готова к загрузке на Google Drive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
