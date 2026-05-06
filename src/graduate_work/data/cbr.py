"""Клиент API Центрального банка РФ.

Курсы валют - XML endpoint cbr.ru/scripts/XML_dynamic.asp.
Ключевая ставка - таблица из XML_keyrate.asp.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DYN_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
KEYRATE_URL = "https://www.cbr.ru/scripts/xml_keyrate.asp"

CURRENCY_CODES: dict[str, str] = {
    # См. https://www.cbr.ru/scripts/XML_val.asp?d=0
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
}


def _parse_cbr_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s, "%d.%m.%Y").replace(tzinfo=timezone.utc))


def _format_cbr_date(s: str) -> str:
    return pd.Timestamp(s).strftime("%d/%m/%Y")


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
    resp = requests.get(DYN_URL, params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    rows: list[dict] = []
    for record in root.findall("Record"):
        date_attr = record.attrib.get("Date", "")
        nominal_text = (record.findtext("Nominal") or "1").replace(",", ".")
        value_text = (record.findtext("Value") or "0").replace(",", ".")
        nominal = float(nominal_text)
        value = float(value_text) / max(nominal, 1.0)
        rows.append({"date": _parse_cbr_date(date_attr), "value": value})
    if not rows:
        return pd.DataFrame(columns=["value"])
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df.rename(columns={"value": f"cbr_{code.lower()}"})
    return df


def fetch_keyrate(start: str, end: str) -> pd.DataFrame:
    """Ключевая ставка ЦБ РФ."""
    params = {
        "DT": _format_cbr_date(start),
        "DT1": _format_cbr_date(end),
    }
    resp = requests.get(KEYRATE_URL, params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    rows: list[dict] = []
    for record in root.findall("KR"):
        date_attr = record.attrib.get("DT", "")
        rate_text = (record.findtext("Rate") or "0").replace(",", ".")
        rows.append({"date": _parse_cbr_date(date_attr), "key_rate": float(rate_text)})
    if not rows:
        return pd.DataFrame(columns=["key_rate"])
    return pd.DataFrame(rows).set_index("date").sort_index()
