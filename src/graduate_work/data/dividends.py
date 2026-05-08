"""Истории дивидендов с открытого ISS — `iss/securities/{secid}/dividends`.

Не требует подписки ALGOPACK. Возвращает all-time историю выплат:
дату закрытия реестра (record date), value, currencyid.

Формат, который мы делаем для downstream-фичей:

    secid, registryclosedate (UTC datetime), value, currencyid

`registryclosedate` это **ex-dividend cutoff** (день после которого
держатель не получит выплату). На бар-уровне используется
`days_to_ex_date` и `is_ex_date_today` как event-feature.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .calendar_iss import _request_with_retry, _resolve_endpoint

logger = logging.getLogger(__name__)


def fetch_dividends(secid: str, *, token: str | None = None) -> pd.DataFrame:
    """Историю дивидендов по тикеру.

    Args:
        secid: тикер.
        token: ALGOPACK Bearer-токен; если задан, запрос идёт на
            apim.moex.com с приоритетным rate-limit'ом.

    Returns:
        DataFrame со столбцами ``secid, registryclosedate, value, currencyid``.
        ``registryclosedate`` — pandas Timestamp (UTC).
    """
    base, headers = _resolve_endpoint(token)
    url = f"{base}/securities/{secid}/dividends.json"
    payload = _request_with_retry(url, params={}, headers=headers)
    block = payload.get("dividends")
    if not isinstance(block, dict):
        return pd.DataFrame()
    columns = block.get("columns", [])
    rows = block.get("data", [])
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows, columns=columns)
    if "registryclosedate" in df.columns:
        df["registryclosedate"] = pd.to_datetime(
            df["registryclosedate"], utc=True, errors="coerce",
        )
    return df.sort_values("registryclosedate") if "registryclosedate" in df.columns else df


def fetch_dividends_batch(
    secids: list[str],
    *,
    request_pause: float = 0.1,
    token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Подряд (sequential, без параллелизма — слабый rate-limit ISS).

    Возвращает dict {secid: DataFrame}, пустые тоже включены.
    """
    out: dict[str, pd.DataFrame] = {}
    for secid in secids:
        try:
            out[secid] = fetch_dividends(secid, token=token)
        except (RuntimeError, requests.RequestException) as exc:
            logger.warning("dividends fetch failed for %s: %s", secid, exc)
            out[secid] = pd.DataFrame()
        time.sleep(request_pause)
    return out
