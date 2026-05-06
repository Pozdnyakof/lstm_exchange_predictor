"""Эндпоинты офлайн-отчёта о тестировании (§3.4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import plots
from .deps import load_report

router = APIRouter(prefix="/api/report", tags=["report"])


def _equity_series(equity_df: pd.DataFrame) -> pd.Series:
    if "timestamp" in equity_df.columns:
        return pd.Series(
            equity_df["equity"].to_numpy(dtype=float),
            index=pd.to_datetime(equity_df["timestamp"], utc=True),
            name="equity",
        )
    series = equity_df.iloc[:, 0]
    series.index = pd.to_datetime(series.index, utc=True)
    return series


@router.get("/backtest")
async def backtest() -> JSONResponse:
    equity_df = load_report("equity")
    metrics = load_report("metrics") or {}
    if not isinstance(equity_df, pd.DataFrame):
        return JSONResponse({"figure": None, "metrics": metrics})
    figure = plots.equity_curve(_equity_series(equity_df))
    return JSONResponse({"figure": json.loads(figure), "metrics": metrics})


@router.get("/per_ticker")
async def per_ticker() -> JSONResponse:
    df = load_report("per_ticker")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return JSONResponse({"figure": None, "rows": []})
    df = df.sort_values("total_return", ascending=False)
    figure = plots.per_ticker_returns(df)
    return JSONResponse(
        {
            "figure": json.loads(figure),
            "rows": df.round(6).to_dict(orient="records"),
        },
    )


@router.get("/random")
async def random_portfolios() -> JSONResponse:
    report = load_report("random")
    if not isinstance(report, dict):
        return JSONResponse({"figure": None, "report": None})
    finals = np.array(report.get("final_returns", []), dtype=float)
    initial = float(report.get("initial_capital", 1.0))
    strategy_return = float(report.get("strategy_final", 0.0)) / initial - 1.0
    threshold_return = float(report.get("threshold_value", 0.0)) / initial - 1.0
    figure = plots.random_portfolios_distribution(
        finals, strategy_return, threshold_return,
    )
    return JSONResponse({"figure": json.loads(figure), "report": report})
