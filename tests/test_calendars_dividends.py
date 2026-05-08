"""Тесты calendar_iss + dividends + calendar_features."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from graduate_work.data import calendar_iss, dividends as dividends_mod
from graduate_work.features.calendar_features import (
    _days_since_last_event,
    _days_to_next_event,
    dividend_features,
    expirations_features,
    trading_day_features,
)


# ---------------------------------------------------------------------------
# calendar_iss._extract_block / _paginate basics
# ---------------------------------------------------------------------------

def test_extract_block_finds_first_with_data() -> None:
    payload = {
        "metadata": {"x": 1},
        "calendars": {"columns": ["a"], "data": [[1]]},
    }
    block = calendar_iss._extract_block(payload, None)
    assert block["data"] == [[1]]


def test_extract_block_explicit_key_priority() -> None:
    payload = {
        "session": {"columns": ["a"], "data": [[1]]},
        "calendars": {"columns": ["b"], "data": [[2]]},
    }
    block = calendar_iss._extract_block(payload, "calendars")
    assert block["columns"] == ["b"]


def test_paginate_handles_short_first_page() -> None:
    """Если первая страница меньше page_size — после неё запросов нет."""
    fake_payload = {
        "calendars": {
            "columns": ["tradedate", "is_traded"],
            "data": [["2024-01-15", 1], ["2024-01-16", 1]],
        },
    }
    with patch.object(calendar_iss, "_request_with_retry", return_value=fake_payload):
        df = calendar_iss._paginate("calendars", page_size=1000)
    assert len(df) == 2
    assert list(df.columns) == ["tradedate", "is_traded"]


def test_paginate_returns_empty_columns_when_no_data() -> None:
    fake = {"calendars": {"columns": ["a"], "data": []}}
    with patch.object(calendar_iss, "_request_with_retry", return_value=fake):
        df = calendar_iss._paginate("calendars")
    assert df.empty
    assert list(df.columns) == ["a"]


def test_calendar_products_registry_has_expected_keys() -> None:
    expected = {
        "trading_days_stock", "trading_days_futures", "trading_days_currency",
        "session_stock", "session_futures", "session_currency",
        "suspended_planned", "settlecodes", "security_changes",
        "boards_history", "futures_expirations",
        "currency_settlement_shifts",
    }
    assert expected <= set(calendar_iss.CALENDAR_PRODUCTS)


# ---------------------------------------------------------------------------
# fetch_trading_days adds show_all_days param when requested
# ---------------------------------------------------------------------------

def test_fetch_trading_days_passes_show_all_days() -> None:
    captured: dict = {}

    def fake_paginate(path, *, params=None, **_kw):
        captured["path"] = path
        captured["params"] = params
        return pd.DataFrame()

    with patch.object(calendar_iss, "_paginate", side_effect=fake_paginate):
        calendar_iss.fetch_trading_days(
            market="stock", start="2024-01-01", end="2024-01-31",
            include_weekends=True,
        )
    assert captured["path"] == "calendars/stock"
    assert captured["params"]["show_all_days"] == "1"
    assert captured["params"]["from"] == "2024-01-01"


def test_resolve_endpoint_routes_by_token() -> None:
    """С токеном → apim.moex.com + Bearer-header; без токена → iss.moex.com."""
    base, headers = calendar_iss._resolve_endpoint("dummy-token")
    assert base == calendar_iss.APIM_BASE
    assert headers == {"Authorization": "Bearer dummy-token"}

    base, headers = calendar_iss._resolve_endpoint(None)
    assert base == calendar_iss.ISS_BASE
    assert headers == {}


def test_paginate_uses_apim_when_token_set() -> None:
    """End-to-end: токен прокидывается через _paginate в URL запроса."""
    captured_urls: list[str] = []
    captured_headers: list[dict] = []

    def fake_request(url, params, **kw):
        captured_urls.append(url)
        captured_headers.append(kw.get("headers"))
        return {"x": {"columns": ["a"], "data": []}}

    with patch.object(calendar_iss, "_request_with_retry", side_effect=fake_request):
        calendar_iss._paginate("calendars/stock", token="abc123")

    assert captured_urls[0].startswith("https://apim.moex.com/")
    assert captured_headers[0] == {"Authorization": "Bearer abc123"}


def test_paginate_uses_iss_when_no_token() -> None:
    captured_urls: list[str] = []
    captured_headers: list[dict] = []

    def fake_request(url, params, **kw):
        captured_urls.append(url)
        captured_headers.append(kw.get("headers"))
        return {"x": {"columns": ["a"], "data": []}}

    with patch.object(calendar_iss, "_request_with_retry", side_effect=fake_request):
        calendar_iss._paginate("calendars/stock", token=None)

    assert captured_urls[0].startswith("https://iss.moex.com/")
    # Без токена headers либо None, либо пустой dict.
    assert not captured_headers[0]


def test_fetch_dividends_uses_apim_when_token_set() -> None:
    captured_url: list[str] = []
    captured_headers: list[dict] = []

    def fake_request(url, params, **kw):
        captured_url.append(url)
        captured_headers.append(kw.get("headers"))
        return {"dividends": {"columns": [], "data": []}}

    with patch.object(dividends_mod, "_request_with_retry", side_effect=fake_request):
        dividends_mod.fetch_dividends("SBER", token="zzz")

    assert "apim.moex.com" in captured_url[0]
    assert captured_headers[0] == {"Authorization": "Bearer zzz"}


# ---------------------------------------------------------------------------
# dividends fetcher
# ---------------------------------------------------------------------------

def test_fetch_dividends_parses_payload() -> None:
    payload = {
        "dividends": {
            "columns": ["secid", "isin", "registryclosedate", "value", "currencyid"],
            "data": [
                ["SBER", "RU000", "2023-05-08", 25.0, "RUB"],
                ["SBER", "RU000", "2024-05-10", 33.3, "RUB"],
            ],
        },
    }
    with patch.object(dividends_mod, "_request_with_retry", return_value=payload):
        df = dividends_mod.fetch_dividends("SBER")
    assert len(df) == 2
    assert df["value"].iloc[1] == pytest.approx(33.3)
    assert isinstance(df["registryclosedate"].iloc[0], pd.Timestamp)
    assert df["registryclosedate"].iloc[0].tz is not None


def test_fetch_dividends_empty_payload_returns_empty() -> None:
    payload = {"dividends": {"columns": [], "data": []}}
    with patch.object(dividends_mod, "_request_with_retry", return_value=payload):
        df = dividends_mod.fetch_dividends("SBER")
    assert df.empty


# ---------------------------------------------------------------------------
# Helper: _days_to_next_event / _days_since_last_event
# ---------------------------------------------------------------------------

def test_days_to_next_event() -> None:
    target = pd.DatetimeIndex([
        "2024-01-01", "2024-01-05", "2024-01-10",
    ], tz="UTC")
    events = pd.to_datetime(["2024-01-08", "2024-02-23"]).to_numpy()
    out = _days_to_next_event(target, events)
    # 01-01 → 7 дней до 01-08; 01-05 → 3; 01-10 → 02-23 = 44.
    assert out[0] == 7
    assert out[1] == 3
    assert out[2] == 44


def test_days_since_last_event_handles_no_past() -> None:
    target = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
    events = pd.to_datetime(["2024-12-31"]).to_numpy()
    out = _days_since_last_event(target, events)
    # Нет прошедших событий → 0.
    assert out[0] == 0


# ---------------------------------------------------------------------------
# trading_day_features
# ---------------------------------------------------------------------------

def test_trading_day_features_marks_holidays() -> None:
    cal = pd.DataFrame({
        "tradedate": ["2024-01-01", "2024-01-02", "2024-01-08"],
        "is_traded": [0, 0, 1],
        "reason":    ["H", "W", "N"],
    })
    target = pd.date_range("2024-01-01 10:00", periods=3, freq="1D", tz="UTC")
    out = trading_day_features(cal, target)
    assert out["cal_is_holiday"].iloc[0] == 1   # 2024-01-01 holiday
    assert out["cal_is_holiday"].iloc[2] == 0   # 2024-01-08 normal
    assert out["cal_is_traded"].iloc[2] == 1


def test_trading_day_features_empty_safe() -> None:
    target = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    out = trading_day_features(pd.DataFrame(), target)
    # При пустом календаре все дни помечаем как торговые (fallback).
    assert (out["cal_is_traded"] == 1).all()
    assert (out["cal_is_holiday"] == 0).all()


# ---------------------------------------------------------------------------
# dividend_features
# ---------------------------------------------------------------------------

def test_dividend_features_days_to_ex_and_value() -> None:
    div = pd.DataFrame({
        "secid": ["SBER", "SBER"],
        "registryclosedate": pd.to_datetime(
            ["2024-05-10", "2025-05-10"], utc=True,
        ),
        "value": [25.0, 33.3],
    })
    target = pd.date_range("2024-05-01", periods=12, freq="1D", tz="UTC")
    out = dividend_features(div, target, last_close_price=300.0)
    # 2024-05-01 → 9 дней до 05-10.
    assert out["div_days_to_ex"].iloc[0] == 9
    # 2024-05-10 — ex-day.
    assert out["div_is_ex_day"].iloc[9] == 1
    assert out["div_value_next"].iloc[0] == pytest.approx(25.0)
    # После 2024-05-10 next становится 2025-05-10 (33.3).
    assert out["div_value_next"].iloc[10] == pytest.approx(33.3)
    assert out["div_value_last"].iloc[11] == pytest.approx(25.0)
    assert "div_yield_last" in out.columns
    assert out["div_yield_last"].iloc[11] == pytest.approx(25.0 / 300.0)


def test_dividend_features_empty_returns_zeros() -> None:
    target = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    out = dividend_features(pd.DataFrame(), target)
    assert (out["div_value_last"] == 0).all()
    assert (out["div_is_ex_day"] == 0).all()


def test_expirations_features_filters_by_asset_code() -> None:
    expir = pd.DataFrame({
        "asset_code": ["GOLD", "SBER", "GOLD"],
        "expiration_date": ["2024-06-15", "2024-07-15", "2024-09-15"],
    })
    target = pd.date_range("2024-06-01", periods=3, freq="1D", tz="UTC")
    out = expirations_features(expir, target, asset_code="GOLD")
    # 2024-06-01 → 14 дней до 06-15.
    assert out["cal_days_to_expiration"].iloc[0] == 14
