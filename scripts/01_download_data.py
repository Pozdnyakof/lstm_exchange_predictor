"""Скачать OHLCV тикеров MOEX, индексы IMOEX/RGBI/RTSI, макропоказатели ЦБ
и котировки Brent с Yahoo Finance в data/raw/.
"""

from __future__ import annotations

import logging

import _bootstrap  # noqa: F401  (sys.path hack)

from graduate_work.config import default_config
from graduate_work.data.orchestrator import download_all


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = default_config()
    download_all(cfg.data, cfg.paths)


if __name__ == "__main__":
    main()
