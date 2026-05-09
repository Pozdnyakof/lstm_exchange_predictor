"""HTTP-клиент MOEX ALGOPACK API.

ALGOPACK — премиум-фид MOEX с микроструктурными данными,
недоступными в бесплатном ISS:

* **TradeStats** (eq/fo/fx, 5-мин) — agressive buy/sell разбивка
  объёма и цены, VWAP по сторонам, disbalance.
* **OrderStats** (eq/fx, 5-мин) — flow выставленных и снятых заявок
  по сторонам.
* **OBStats** (eq/fo/fx, 5-мин) — спрэды (bbo/lv10/1mio), order-book
  imbalance, levels b/s.
* **FUTOI** (5-мин) — open interest по физ/юр лицам.
* **HI2** (daily) — индекс концентрации Herfindahl.
* **MegaAlerts** (1-мин с 2024) — anomaly detection.

Архив доступен с 2020-01.

Аутентификация: Bearer token из MOEX DataShop / Passport, передаётся
в HTTP-header. Базовый URL — ``apim.moex.com`` (НЕ ``iss.moex.com``).

Все методы возвращают ``pd.DataFrame`` с UTC-timestamp в индексе.
Pagination прозрачная (10k строк на ticker-scoped запрос).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ALGOPACK_BASE = "https://apim.moex.com/iss"
PAGE_SIZE_TICKER = 10_000
PAGE_SIZE_MARKET = 50_000
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF = 2.0


@dataclass(frozen=True)
class AlgopackProduct:
    """Описание endpoint-а в каталоге ALGOPACK."""

    name: str
    path: str  # без `.json` и без secid
    market: str  # eq | fo | fx
    granularity: str  # 5min | 1min | 1day
    ticker_scoped: bool


PRODUCTS = {
    "tradestats_eq": AlgopackProduct(
        "tradestats_eq", "datashop/algopack/eq/tradestats", "eq", "5min", True,
    ),
    "tradestats_fo": AlgopackProduct(
        "tradestats_fo", "datashop/algopack/fo/tradestats", "fo", "5min", True,
    ),
    "tradestats_fx": AlgopackProduct(
        "tradestats_fx", "datashop/algopack/fx/tradestats", "fx", "5min", True,
    ),
    "orderstats_eq": AlgopackProduct(
        "orderstats_eq", "datashop/algopack/eq/orderstats", "eq", "5min", True,
    ),
    "orderstats_fx": AlgopackProduct(
        "orderstats_fx", "datashop/algopack/fx/orderstats", "fx", "5min", True,
    ),
    "obstats_eq": AlgopackProduct(
        "obstats_eq", "datashop/algopack/eq/obstats", "eq", "5min", True,
    ),
    "obstats_fo": AlgopackProduct(
        "obstats_fo", "datashop/algopack/fo/obstats", "fo", "5min", True,
    ),
    "obstats_fx": AlgopackProduct(
        "obstats_fx", "datashop/algopack/fx/obstats", "fx", "5min", True,
    ),
    "futoi": AlgopackProduct(
        "futoi", "analyticalproducts/futoi/securities", "fo", "5min", True,
    ),
    "hi2_eq": AlgopackProduct(
        "hi2_eq", "datashop/algopack/eq/hi2", "eq", "1day", True,
    ),
    "megaalerts_eq": AlgopackProduct(
        "megaalerts_eq", "datashop/algopack/eq/alerts", "eq", "1min", False,
    ),
}


class AlgopackError(RuntimeError):
    """Поднимается при неустранимой ошибке API (после исчерпания ретраев)."""


class AlgopackClient:
    """HTTP-клиент с автопагинацией и ретраями.

    Использование::

        client = AlgopackClient()  # подхватит ALGOPACK_TOKEN из env
        df = client.tradestats('SBER', start='2024-01-01', end='2024-01-31')
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = ALGOPACK_BASE,
        request_pause: float = 0.12,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_sec: float = DEFAULT_BACKOFF,
    ) -> None:
        resolved = token or os.environ.get("ALGOPACK_TOKEN")
        if not resolved:
            msg = (
                "ALGOPACK_TOKEN не задан: передайте token=... в конструктор "
                "или установите переменную окружения."
            )
            raise ValueError(msg)
        self.base_url = base_url.rstrip("/")
        self.request_pause = float(request_pause)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff_sec = float(backoff_sec)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {resolved}",
            "Accept": "application/json",
            "User-Agent": "graduate-work-algopack/1.0",
        })

    # ------------------------------------------------------------------
    # Низкоуровневые helpers
    # ------------------------------------------------------------------

    def _fetch_one_page(self, path: str, params: dict) -> dict:
        """Один HTTP-запрос с экспоненциальным backoff на 5xx и 429."""
        url = f"{self.base_url}/{path}.json"
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 503):
                    delay = self.backoff_sec * (2 ** attempt)
                    logger.warning(
                        "ALGOPACK %s %d, retry %d/%d in %.1fs",
                        url, resp.status_code, attempt + 1, self.retries, delay,
                    )
                    time.sleep(delay)
                    continue
                if resp.status_code == 401:
                    msg = (
                        "ALGOPACK 401 Unauthorized — проверьте корректность "
                        "ALGOPACK_TOKEN и активность подписки."
                    )
                    raise AlgopackError(msg)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                delay = self.backoff_sec * (2 ** attempt)
                logger.warning(
                    "ALGOPACK request error (attempt %d/%d) in %.1fs: %s",
                    attempt + 1, self.retries, delay, exc,
                )
                time.sleep(delay)
        msg = f"ALGOPACK {url} failed after {self.retries} retries"
        raise AlgopackError(msg) from last_exc

    def _iter_pages(self, path: str, base_params: dict):
        """Generator: yields (columns, data_rows) на каждой странице.

        Останавливается ТОЛЬКО на пустой странице — ALGOPACK имеет
        server-side hard cap ~1000 строк, поэтому проверка
        ``len(page) < page_size`` здесь не работает (всегда true на
        первой же странице, теряем 90%+ данных).
        """
        cursor = 0
        while True:
            payload = self._fetch_one_page(
                path, {**base_params, "start": cursor},
            )
            block = _extract_data_block(payload)
            if block is None:
                return
            page = block.get("data", [])
            cols = block.get("columns", [])
            if not page:
                return
            yield cols, page
            cursor += len(page)
            time.sleep(self.request_pause)

    def _paginate(
        self,
        path: str,
        params: dict,
        *,
        page_size: int,
    ) -> pd.DataFrame:
        """Скачать все страницы endpoint-а и склеить в один DataFrame."""
        base_params = {**params, "limit": page_size}
        rows: list[list] = []
        columns: list[str] | None = None
        for cols, page in self._iter_pages(path, base_params):
            if columns is None:
                columns = list(cols)
            rows.extend(page)
        if not rows or columns is None:
            return pd.DataFrame(columns=columns or [])
        return pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------------
    # Per-product методы
    # ------------------------------------------------------------------

    def _ticker_endpoint(
        self,
        product: AlgopackProduct,
        secid: str,
        *,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        params: dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["till"] = end
        path = f"{product.path}/{secid}"
        df = self._paginate(path, params, page_size=PAGE_SIZE_TICKER)
        return _normalize_supercandle_index(df)

    def tradestats(
        self,
        secid: str,
        *,
        market: str = "eq",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """5-мин TradeStats: aggressive buy/sell разбивка объёма и цены."""
        return self._ticker_endpoint(
            PRODUCTS[f"tradestats_{market}"],
            secid, start=start, end=end,
        )

    def orderstats(
        self,
        secid: str,
        *,
        market: str = "eq",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """5-мин OrderStats: поток выставленных и снятых заявок."""
        return self._ticker_endpoint(
            PRODUCTS[f"orderstats_{market}"],
            secid, start=start, end=end,
        )

    def obstats(
        self,
        secid: str,
        *,
        market: str = "eq",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """5-мин OBStats: спрэды и order-book imbalance."""
        return self._ticker_endpoint(
            PRODUCTS[f"obstats_{market}"],
            secid, start=start, end=end,
        )

    def hi2(
        self,
        secid: str,
        *,
        market: str = "eq",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Daily HI2 (Herfindahl): индекс концентрации торгов."""
        return self._ticker_endpoint(
            PRODUCTS[f"hi2_{market}"],
            secid, start=start, end=end,
        )

    def futoi(
        self,
        secid: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """5-мин Futures Open Interest по физ/юр лицам."""
        return self._ticker_endpoint(
            PRODUCTS["futoi"], secid, start=start, end=end,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_data_block(payload: dict) -> dict | None:
    """ALGOPACK возвращает {"<metric>": {"columns": [...], "data": [[...]]}}.

    Имя ключа метрики варьируется (tradestats / orderstats / obstats / ...).
    Возвращаем первый dict-блок с ключом ``data``.
    """
    if not isinstance(payload, dict):
        return None
    for value in payload.values():
        if isinstance(value, dict) and "data" in value:
            return value
    return None


def _normalize_supercandle_index(df: pd.DataFrame) -> pd.DataFrame:
    """Собрать tradedate+tradetime в UTC-индекс ``begin``.

    SuperCandle-таблицы (TradeStats/OrderStats/OBStats) приходят с
    раздельными ``tradedate`` и ``tradetime`` колонками. Объединяем
    их в timezone-aware DatetimeIndex для прямой стыковки с нашим
    bar-grid'ом. Если в DataFrame нет ``tradedate``, возвращаем как
    есть (например, HI2 имеет только ``tradedate``).
    """
    if df.empty:
        return df
    if "tradedate" not in df.columns:
        return df
    if "tradetime" in df.columns:
        ts = pd.to_datetime(
            df["tradedate"].astype(str) + " " + df["tradetime"].astype(str),
            utc=True, errors="coerce",
        )
    else:
        ts = pd.to_datetime(df["tradedate"], utc=True, errors="coerce")
    out = df.copy()
    out.index = ts
    out.index.name = "begin"
    out = out.dropna(how="all").sort_index()
    return out
