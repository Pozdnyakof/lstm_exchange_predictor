"""Календари MOEX (открытое ISS, без auth).

Лежат под `iss.moex.com/iss/calendars/...` — отдельная подсистема, НЕ
gated подпиской ALGOPACK. Все эндпоинты возвращают стандартный
ISS-конверт ``{"<key>": {"columns": [...], "data": [[...]]}}``.

Покрывает 13 разных календарей. В нашей задаче (intraday-классификация
российских акций) полезны:

* **trading_days** — флаги торговый/выходной/праздник/перенос с разделением
  по рынкам stock/futures/currency.
* **session_schedule** — внутри-дневное расписание: открытие, закрытие,
  аукционы, перерывы (даёт точные `time_from`/`time_till` для каждого дня).
* **suspended_planned** / **suspended_intraday** — приостановки торгов
  по конкретным тикерам.
* **futures_expirations** — даты исполнения фьючерсов / опционов.
* **stock_changes** / **currency_changes** — изменения атрибутов
  (тикер, номинал, листинг).

Дивиденды лежат в **другом** месте — `iss/securities/{secid}/dividends`,
обрабатываются в `data/dividends.py`.

Все методы возвращают `pd.DataFrame`. Параметры запроса (start/end)
универсальные — `from`/`till` в формате YYYY-MM-DD.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ISS_BASE = "https://iss.moex.com/iss"          # open, без auth
APIM_BASE = "https://apim.moex.com/iss"        # авторизованный (вышеприоритетный rate-limit)
PAGE_SIZE = 1000  # размер страницы для всех calendar endpoints
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF = 1.5


def _resolve_endpoint(token: str | None) -> tuple[str, dict[str, str]]:
    """Выбрать base URL + headers в зависимости от наличия токена.

    С токеном — отправляем на ``apim.moex.com`` с Bearer-header (это
    уже путь с приоритетным rate-limit'ом, действующий и для
    «открытых» календарей/дивидендов). Без токена — обычный
    ``iss.moex.com``.
    """
    if token:
        return APIM_BASE, {"Authorization": f"Bearer {token}"}
    return ISS_BASE, {}


def _request_with_retry(
    url: str,
    params: dict,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_sec: float = DEFAULT_BACKOFF,
    headers: dict[str, str] | None = None,
) -> dict:
    """GET с экспоненциальным backoff. Возвращает JSON."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code in (429, 503):
                delay = backoff_sec * (2 ** attempt)
                logger.warning(
                    "ISS %s status %d, retry %d/%d in %.1fs",
                    url, resp.status_code, attempt + 1, retries, delay,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            delay = backoff_sec * (2 ** attempt)
            logger.warning(
                "ISS error attempt %d/%d in %.1fs: %s",
                attempt + 1, retries, delay, exc,
            )
            time.sleep(delay)
    msg = f"ISS request failed after {retries} retries: {url}"
    raise RuntimeError(msg) from last_exc


def _extract_block(payload: dict, key: str | None) -> dict | None:
    """Из ISS-ответа достать первый dict с колонкой ``data``.

    Если ``key`` указан — ищем именно его. Иначе берём первый
    подходящий блок (имена варьируются: ``calendars``, ``session``,
    ``securities``, ...).
    """
    if not isinstance(payload, dict):
        return None
    if key and key in payload and isinstance(payload[key], dict):
        return payload[key]
    for value in payload.values():
        if isinstance(value, dict) and "data" in value:
            return value
    return None


def _iter_pages(
    url: str,
    params: dict,
    headers: dict[str, str] | None,
    block_key: str | None,
    page_size: int,
):
    """Generator: yields (columns, data_rows) на каждой странице."""
    cursor = 0
    while True:
        payload = _request_with_retry(
            url, {**params, "start": cursor}, headers=headers,
        )
        block = _extract_block(payload, block_key)
        if block is None:
            return
        page = block.get("data", [])
        cols = block.get("columns", [])
        yield cols, page
        if len(page) < page_size:
            return
        cursor += len(page)
        time.sleep(0.05)


def _paginate(
    path: str,
    *,
    params: dict | None = None,
    block_key: str | None = None,
    page_size: int = PAGE_SIZE,
    token: str | None = None,
) -> pd.DataFrame:
    """Пагинированный fetch + concat в один DataFrame.

    Если ``token`` задан — запросы идут на ``apim.moex.com`` с
    Bearer-header (приоритетный rate-limit), иначе на открытый
    ``iss.moex.com``.
    """
    base, headers = _resolve_endpoint(token)
    url = f"{base}/{path}.json"
    rows: list[list] = []
    columns: list[str] | None = None
    for cols, page in _iter_pages(url, params or {}, headers, block_key, page_size):
        if columns is None:
            columns = list(cols)
        if page:
            rows.extend(page)
    if not rows or columns is None:
        return pd.DataFrame(columns=columns or [])
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Per-product fetchers
# ---------------------------------------------------------------------------

def _date_range_params(start: str | None, end: str | None) -> dict[str, str]:
    """Утилита: ``from``/``till`` если переданы."""
    params: dict[str, str] = {}
    if start:
        params["from"] = start
    if end:
        params["till"] = end
    return params


def fetch_trading_days(
    *,
    market: str = "stock",
    start: str | None = None,
    end: str | None = None,
    include_weekends: bool = True,
    token: str | None = None,
) -> pd.DataFrame:
    """Календарь торговых дней.

    Args:
        market: ``stock`` | ``futures`` | ``currency`` | ``""`` (пусто=все).
        start: начало, YYYY-MM-DD.
        end: конец, YYYY-MM-DD.
        include_weekends: добавляет ``show_all_days=1`` — иначе вернёт
            только ``off_days``.
        token: ALGOPACK Bearer-токен; если задан, запрос идёт на
            apim.moex.com с приоритетным rate-limit'ом.
    """
    path = "calendars" if not market else f"calendars/{market}"
    params = _date_range_params(start, end)
    if include_weekends:
        params["show_all_days"] = "1"
    return _paginate(path, params=params, token=token)


def fetch_session_schedule(
    *,
    market: str = "stock",
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Внутри-дневное расписание (опен/клоуз аукционы, перерывы)."""
    path = f"calendars/{market}/session"
    return _paginate(path, params=_date_range_params(start, end), token=token)


def fetch_suspended_planned(
    *,
    secid: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Запланированные приостановки торгов (current year + 3y forward).

    Если ``secid`` не задан — общий список по всем тикерам.
    """
    path = "calendars/stock/securities/suspended/details"
    params: dict[str, str] = {}
    if secid:
        params["securities"] = secid
    return _paginate(path, params=params, token=token)


def fetch_suspended_intraday(*, token: str | None = None) -> pd.DataFrame:
    """Сегодняшние внутри-дневные приостановки. Очищается ежедневно."""
    return _paginate("calendars/stock/session/suspended", token=token)


def fetch_settlecodes(
    *,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Календарь settlement-кодов (T+0/T+1/T+2 → даты расчёта)."""
    path = "calendars/stock/session/settlecodes"
    return _paginate(path, params=_date_range_params(start, end), token=token)


def fetch_security_attribute_changes(
    *,
    secid: str | None = None,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Лог изменений атрибутов бумаг (тикер, номинал, листинг)."""
    path = "calendars/stock/securities/changes"
    params = _date_range_params(start, end)
    if secid:
        params["securities"] = secid
    return _paginate(path, params=params, token=token)


def fetch_security_boards_history(
    *,
    secid: str | None = None,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """История режимов торгов (TQBR/TQOB/...) по дням и бумагам."""
    path = "calendars/stock/securities/boards"
    params = _date_range_params(start, end)
    if secid:
        params["securities"] = secid
    return _paginate(path, params=params, token=token)


def fetch_futures_expirations(
    *,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Календарь экспираций фьючерсов и опционов FORTS."""
    path = "calendars/futures/securities"
    return _paginate(path, params=_date_range_params(start, end), token=token)


def fetch_currency_settlement_shifts(
    *,
    secid: str | None = None,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Сдвиги settlement-дат на валютном рынке (праздничные переносы)."""
    path = "calendars/currency/securities"
    params = _date_range_params(start, end)
    if secid:
        params["securities"] = secid
    return _paginate(path, params=params, token=token)


CALENDAR_PRODUCTS = {
    "trading_days_stock": lambda **kw: fetch_trading_days(market="stock", **kw),
    "trading_days_futures": lambda **kw: fetch_trading_days(market="futures", **kw),
    "trading_days_currency": lambda **kw: fetch_trading_days(market="currency", **kw),
    "session_stock": lambda **kw: fetch_session_schedule(market="stock", **kw),
    "session_futures": lambda **kw: fetch_session_schedule(market="futures", **kw),
    "session_currency": lambda **kw: fetch_session_schedule(market="currency", **kw),
    "suspended_planned": lambda **kw: fetch_suspended_planned(**kw),
    "settlecodes": lambda **kw: fetch_settlecodes(**kw),
    "security_changes": lambda **kw: fetch_security_attribute_changes(**kw),
    "boards_history": lambda **kw: fetch_security_boards_history(**kw),
    "futures_expirations": lambda **kw: fetch_futures_expirations(**kw),
    "currency_settlement_shifts": lambda **kw: fetch_currency_settlement_shifts(**kw),
}
