"""Клиент API Центрального банка РФ.

Курсы валют - XML endpoint cbr.ru/scripts/XML_dynamic.asp.
Ключевая ставка - cbr.ru/scripts/xml_keyrate.asp.

XML от ЦБ изредка приходит битым (HTML-error page вместо ответа,
обрезанный/перекодированный байтовый поток). Поэтому парсинг
завёрнут в `_robust_parse_xml`: при сбое сырой ответ дампится в
data/raw/macro/_failed/ для последующего разбора, а функция-обёртка
возвращает пустой DataFrame, чтобы остальной конвейер мог продолжить.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DYN_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
# Старый xml_keyrate.asp деактивирован ЦБ (404 с осени 2025). Современный
# рабочий канал - SOAP-веб-сервис DailyInfoWebServ.asmx, операция KeyRate.
KEYRATE_SOAP_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
KEYRATE_SOAP_ENVELOPE = (
    '<?xml version="1.0" encoding="utf-8"?>'
    "<soap:Envelope"
    ' xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"'
    ' xmlns:w="http://web.cbr.ru/">'
    "<soap:Body><w:KeyRate>"
    "<w:fromDate>{s}T00:00:00</w:fromDate>"
    "<w:ToDate>{e}T00:00:00</w:ToDate>"
    "</w:KeyRate></soap:Body></soap:Envelope>"
)

CURRENCY_CODES: dict[str, str] = {
    # См. https://www.cbr.ru/scripts/XML_val.asp?d=0
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
}


def _parse_cbr_date(s: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(datetime.strptime(s, "%d.%m.%Y").replace(tzinfo=timezone.utc))
    except (ValueError, TypeError):
        return None


def _format_cbr_date(s: str) -> str:
    return pd.Timestamp(s).strftime("%d/%m/%Y")


# ---------------------------------------------------------------------------
# Robust XML parsing
# ---------------------------------------------------------------------------

def _dump_failed_response(content: bytes, label: str) -> None:
    """Сохранить сырой ответ для оффлайн-разбора."""
    fail_dir = Path.cwd() / "data" / "raw" / "macro" / "_failed"
    try:
        fail_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = fail_dir / f"{label}_{ts}.xml"
        out.write_bytes(content)
        logger.warning("Saved bad CBR response to %s", out)
    except OSError:
        logger.exception("Could not dump bad CBR response")


def _try_parse(content: bytes) -> ET.Element | None:
    try:
        return ET.fromstring(content)
    except ET.ParseError:
        # Иногда CBR отдаёт XML без корректного объявления encoding,
        # но с реальным windows-1251 содержимым. Пробуем перекодировать
        # и убрать XML-декларацию (т.к. передаём str).
        try:
            text = content.decode("windows-1251", errors="replace")
            text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text).strip()
            return ET.fromstring(text)
        except (ET.ParseError, UnicodeDecodeError):
            return None


def _robust_parse_xml(content: bytes, label: str) -> ET.Element | None:
    """Распарсить XML с двумя попытками; на сбой - дамп файла + None."""
    root = _try_parse(content)
    if root is None:
        _dump_failed_response(content, label)
    return root


# ---------------------------------------------------------------------------
# Currency rates
# ---------------------------------------------------------------------------

def _record_to_currency_row(record: ET.Element) -> dict | None:
    date_attr = record.attrib.get("Date", "")
    parsed = _parse_cbr_date(date_attr)
    if parsed is None:
        return None
    nominal = float((record.findtext("Nominal") or "1").replace(",", "."))
    value = float((record.findtext("Value") or "0").replace(",", ".")) / max(nominal, 1.0)
    return {"date": parsed, "value": value}


def fetch_currency(code: str, start: str, end: str) -> pd.DataFrame:
    """Курс валюты ЦБ РФ для указанного диапазона дат."""
    if code not in CURRENCY_CODES:
        msg = f"Unknown CBR currency code: {code}"
        raise ValueError(msg)
    params = {
        "date_req1": _format_cbr_date(start),
        "date_req2": _format_cbr_date(end),
        "VAL_NM_RQ": CURRENCY_CODES[code],
    }
    try:
        resp = requests.get(DYN_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("CBR currency %s HTTP failed: %s", code, exc)
        return pd.DataFrame(columns=["value"])

    root = _robust_parse_xml(resp.content, f"currency_{code.lower()}")
    if root is None:
        logger.warning("CBR currency %s: XML parse failed, returning empty", code)
        return pd.DataFrame(columns=["value"])

    rows = [r for r in (_record_to_currency_row(rec) for rec in root.findall("Record")) if r]
    if not rows:
        return pd.DataFrame(columns=["value"])
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df.rename(columns={"value": f"cbr_{code.lower()}"})


# ---------------------------------------------------------------------------
# Key rate
# ---------------------------------------------------------------------------

def _kr_fields(element: ET.Element) -> dict | None:
    """Распаковать дочерние теги <DT> и <Rate> из узла <KR>.

    Имена тегов могут быть в namespace вида '{http://web.cbr.ru/}DT',
    поэтому используем .split('}')[-1] для извлечения локальной части.
    """
    dt = rate = None
    for ch in element:
        local = ch.tag.split("}")[-1]
        if local == "DT" and ch.text:
            dt = ch.text
        elif local == "Rate" and ch.text:
            rate = ch.text
    if not (dt and rate):
        return None
    parsed = pd.to_datetime(dt, utc=True, errors="coerce")
    if parsed is pd.NaT:
        return None
    try:
        rate_value = float(rate.replace(",", "."))
    except ValueError:
        return None
    return {"date": parsed, "key_rate": rate_value}


def _parse_keyrate_soap(content: bytes) -> list[dict]:
    root = _robust_parse_xml(content, "keyrate_soap")
    if root is None:
        return []
    rows: list[dict] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "KR":
            continue
        rec = _kr_fields(el)
        if rec is not None:
            rows.append(rec)
    return rows


def _format_iso_date(s: str) -> str:
    """ISO-формат YYYY-MM-DD для SOAP-конверта."""
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def fetch_keyrate(start: str, end: str) -> pd.DataFrame:
    """Ключевая ставка ЦБ РФ через SOAP-сервис DailyInfoWebServ.asmx.

    Старый XML endpoint xml_keyrate.asp ЦБ деактивировал (HTTP 404).
    SOAP-сервис стабилен и используется банком для интеграционных
    клиентов. Запрос - POST с XML-конвертом, операция KeyRate.
    """
    body = KEYRATE_SOAP_ENVELOPE.format(
        s=_format_iso_date(start),
        e=_format_iso_date(end),
    ).encode("utf-8")
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": "http://web.cbr.ru/KeyRate",
    }
    try:
        resp = requests.post(KEYRATE_SOAP_URL, data=body, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("CBR keyrate SOAP HTTP failed: %s", exc)
        return pd.DataFrame(columns=["key_rate"])

    rows = _parse_keyrate_soap(resp.content)
    if not rows:
        logger.warning("CBR keyrate: SOAP вернул 0 записей, возвращаем пусто")
        return pd.DataFrame(columns=["key_rate"])
    return pd.DataFrame(rows).set_index("date").sort_index()
