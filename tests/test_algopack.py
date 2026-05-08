"""Тесты ALGOPACK-клиента и feature-engineering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from graduate_work.data.algopack import (
    PRODUCTS,
    AlgopackClient,
    AlgopackError,
    _extract_data_block,
    _normalize_supercandle_index,
)
from graduate_work.features.algopack_features import (
    build_algopack_features,
    obstats_features,
    orderstats_features,
    order_to_trade_ratio,
    tradestats_features,
)


# ---------------------------------------------------------------------------
# AlgopackClient — auth & request shape
# ---------------------------------------------------------------------------

def test_client_requires_token_or_env(monkeypatch) -> None:
    monkeypatch.delenv("ALGOPACK_TOKEN", raising=False)
    with pytest.raises(ValueError):
        AlgopackClient()


def test_client_sets_bearer_header() -> None:
    client = AlgopackClient(token="dummy-token-123")
    assert client._session.headers["Authorization"] == "Bearer dummy-token-123"


def test_extract_data_block_finds_first_with_data() -> None:
    payload = {
        "metadata": {"foo": 1},
        "tradestats": {"columns": ["a"], "data": [[1]]},
    }
    block = _extract_data_block(payload)
    assert block == {"columns": ["a"], "data": [[1]]}


def test_extract_data_block_handles_empty() -> None:
    assert _extract_data_block({}) is None
    assert _extract_data_block({"x": 1}) is None


def test_normalize_supercandle_index_combines_date_time() -> None:
    df = pd.DataFrame({
        "tradedate": ["2024-01-15", "2024-01-15"],
        "tradetime": ["10:05:00", "10:10:00"],
        "vol": [100, 200],
    })
    out = _normalize_supercandle_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
    assert out.index[0] == pd.Timestamp("2024-01-15 10:05:00", tz="UTC")
    assert out["vol"].iloc[1] == 200


def test_normalize_supercandle_index_empty_passthrough() -> None:
    df = pd.DataFrame()
    assert _normalize_supercandle_index(df).empty


# ---------------------------------------------------------------------------
# Pagination behaviour with mocked _fetch_one_page
# ---------------------------------------------------------------------------

def _make_client_with_pages(pages: list[list[list]], columns: list[str]) -> AlgopackClient:
    """Return a client whose _fetch_one_page returns pages sequentially."""
    client = AlgopackClient(token="x")
    page_iter = iter(pages)

    def fake_fetch(path, params):
        try:
            page = next(page_iter)
        except StopIteration:
            page = []
        return {"tradestats": {"columns": columns, "data": page}}

    client._fetch_one_page = MagicMock(side_effect=fake_fetch)
    return client


def test_paginate_concatenates_pages_until_short() -> None:
    cols = ["tradedate", "tradetime", "vol"]
    pages = [
        [[f"2024-01-{i:02d}", "10:00:00", i] for i in range(1, 11)],   # 10 rows
        [[f"2024-01-{i:02d}", "10:00:00", i] for i in range(11, 14)],  # 3 rows < page_size
    ]
    client = _make_client_with_pages(pages, cols)
    df = client._paginate("path", {}, page_size=10)
    assert len(df) == 13


def test_paginate_returns_empty_when_no_data() -> None:
    client = _make_client_with_pages([[]], ["a", "b"])
    df = client._paginate("path", {}, page_size=10)
    assert df.empty


# ---------------------------------------------------------------------------
# Errors: 401 raises AlgopackError immediately
# ---------------------------------------------------------------------------

def test_401_raises_algopack_error() -> None:
    client = AlgopackClient(token="bad", retries=2, backoff_sec=0.01)

    fake_resp = MagicMock()
    fake_resp.status_code = 401
    fake_resp.raise_for_status.side_effect = AssertionError("should not call")

    with patch.object(client._session, "get", return_value=fake_resp):
        with pytest.raises(AlgopackError):
            client._fetch_one_page("datashop/algopack/eq/tradestats/SBER", {})


# ---------------------------------------------------------------------------
# Product registry sanity
# ---------------------------------------------------------------------------

def test_products_registry_covers_expected_keys() -> None:
    expected = {
        "tradestats_eq", "tradestats_fo", "tradestats_fx",
        "orderstats_eq", "orderstats_fx",
        "obstats_eq", "obstats_fo", "obstats_fx",
        "futoi", "hi2_eq", "megaalerts_eq",
    }
    assert expected <= set(PRODUCTS)


# ---------------------------------------------------------------------------
# Features: tradestats
# ---------------------------------------------------------------------------

def _fake_supercandle_frame(rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-15 10:00", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame(index=idx)


def test_tradestats_features_imbalance_calc() -> None:
    df = _fake_supercandle_frame(2)
    df["vol"] = [100.0, 200.0]
    df["vol_b"] = [70.0, 150.0]
    df["vol_s"] = [30.0, 50.0]
    df["val"] = [10000.0, 20000.0]
    df["val_b"] = [7000.0, 15000.0]
    df["val_s"] = [3000.0, 5000.0]
    df["trades"] = [10, 20]
    df["trades_b"] = [7, 14]
    df["trades_s"] = [3, 6]
    df["disb"] = [0.4, 0.5]
    out = tradestats_features(df)
    assert out["aps_vol_imb"].iloc[0] == pytest.approx(0.40)
    assert out["aps_val_imb"].iloc[0] == pytest.approx(0.40)
    assert out["aps_trades_imb"].iloc[0] == pytest.approx(0.40)
    assert out["aps_disb"].iloc[1] == pytest.approx(0.50)


def test_tradestats_features_handles_zero_volume() -> None:
    df = _fake_supercandle_frame(1)
    df["vol"] = [0.0]
    df["vol_b"] = [0.0]
    df["vol_s"] = [0.0]
    df["val"] = [0.0]
    df["val_b"] = [0.0]
    df["val_s"] = [0.0]
    df["trades"] = [0]
    df["trades_b"] = [0]
    df["trades_s"] = [0]
    out = tradestats_features(df)
    # Деление 0/0 → 0, не NaN.
    assert out["aps_vol_imb"].iloc[0] == 0.0
    assert out["aps_val_imb"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Features: orderstats
# ---------------------------------------------------------------------------

def test_orderstats_features_cancel_ratio() -> None:
    df = _fake_supercandle_frame(2)
    df["put_orders"] = [100.0, 50.0]
    df["put_orders_b"] = [60.0, 30.0]
    df["put_orders_s"] = [40.0, 20.0]
    df["put_vol"] = [10000.0, 5000.0]
    df["put_vol_b"] = [6000.0, 3000.0]
    df["put_vol_s"] = [4000.0, 2000.0]
    df["cancel_orders"] = [50.0, 50.0]
    df["cancel_orders_b"] = [30.0, 25.0]
    df["cancel_orders_s"] = [20.0, 25.0]
    out = orderstats_features(df)
    # cancel / (put + cancel) = 50 / 150 = 0.333
    assert out["aps_cancel_ratio_orders"].iloc[0] == pytest.approx(1/3, abs=1e-3)
    # put orders imbalance (60-40)/100 = 0.2
    assert out["aps_put_orders_imb"].iloc[0] == pytest.approx(0.2)
    assert "aps_cancel_orders_imb" in out.columns


# ---------------------------------------------------------------------------
# Features: obstats
# ---------------------------------------------------------------------------

def test_obstats_features_uses_mid_for_spread_bp() -> None:
    df = _fake_supercandle_frame(1)
    df["spread_bbo"] = [0.10]
    df["spread_lv10"] = [0.50]
    df["mid_price"] = [100.0]
    df["imbalance_vol_bbo"] = [0.3]
    df["imbalance_val_bbo"] = [0.4]
    df["levels_b"] = [10]
    df["levels_s"] = [5]
    df["vwap_b_1mio"] = [100.10]
    df["vwap_s_1mio"] = [99.90]
    out = obstats_features(df)
    # 0.10 / 100 * 1e4 = 10 bp.
    assert out["aps_spread_bbo_bp"].iloc[0] == pytest.approx(10.0)
    assert out["aps_spread_lv10_bp"].iloc[0] == pytest.approx(50.0)
    assert out["aps_imb_vol_bbo"].iloc[0] == pytest.approx(0.3)
    # levels (10-5)/(10+5) = 1/3
    assert out["aps_levels_imb"].iloc[0] == pytest.approx(1/3, abs=1e-3)


# ---------------------------------------------------------------------------
# Combined builder
# ---------------------------------------------------------------------------

def test_build_algopack_features_merges_all_three() -> None:
    ts = _fake_supercandle_frame(3)
    ts["vol"] = [100, 100, 100]
    ts["vol_b"] = [60, 50, 40]
    ts["vol_s"] = [40, 50, 60]
    ts["val"] = [1000, 1000, 1000]
    ts["val_b"] = [600, 500, 400]
    ts["val_s"] = [400, 500, 600]
    ts["trades"] = [10, 10, 10]
    ts["trades_b"] = [5, 5, 5]
    ts["trades_s"] = [5, 5, 5]
    ts["disb"] = [0.1, 0.0, -0.1]
    ts["pr_vwap"] = [100.0, 101.0, 99.0]
    ts["pr_vwap_b"] = [100.05, 101.02, 99.0]
    ts["pr_vwap_s"] = [99.95, 100.98, 99.0]

    os_ = _fake_supercandle_frame(3)
    os_["put_orders"] = [50, 60, 40]
    os_["put_orders_b"] = [25, 30, 20]
    os_["put_orders_s"] = [25, 30, 20]
    os_["put_vol"] = [5000, 6000, 4000]
    os_["put_vol_b"] = [2500, 3000, 2000]
    os_["put_vol_s"] = [2500, 3000, 2000]
    os_["cancel_orders"] = [10, 10, 10]
    os_["cancel_orders_b"] = [5, 5, 5]
    os_["cancel_orders_s"] = [5, 5, 5]

    ob = _fake_supercandle_frame(3)
    ob["mid_price"] = [100.0, 101.0, 99.0]
    ob["spread_bbo"] = [0.05, 0.05, 0.05]
    ob["imbalance_vol_bbo"] = [0.1, -0.1, 0.0]

    out = build_algopack_features(tradestats=ts, orderstats=os_, obstats=ob)
    assert {"aps_vol_imb", "aps_cancel_ratio_orders", "aps_imb_vol_bbo",
            "aps_order_to_trade"} <= set(out.columns)
    assert len(out) == 3


def test_build_algopack_features_empty_returns_empty() -> None:
    out = build_algopack_features()
    assert out.empty


def test_order_to_trade_ratio_outer_join() -> None:
    """orderstats и tradestats могут иметь разные индексы — нужен outer."""
    idx1 = pd.date_range("2024-01-15 10:00", periods=2, freq="5min", tz="UTC")
    idx2 = pd.date_range("2024-01-15 10:05", periods=2, freq="5min", tz="UTC")
    os_ = pd.DataFrame({"put_orders": [50, 60]}, index=idx1)
    ts_ = pd.DataFrame({"trades": [10, 12]}, index=idx2)
    ratio = order_to_trade_ratio(os_, ts_)
    # На совпадающем баре 10:05: put_orders=60 / trades=10 = 6.0.
    assert ratio.loc["2024-01-15 10:05:00+00:00"] == pytest.approx(6.0)
