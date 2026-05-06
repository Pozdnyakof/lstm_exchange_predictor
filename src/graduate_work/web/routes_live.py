"""Live-эндпоинты: текущие прогнозы по тикерам."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .deps import get_service

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/predictions")
async def predictions(request: Request) -> JSONResponse:
    service = get_service(request)
    forecasts = service.predict_all()
    return JSONResponse(
        {
            "forecasts": [f.to_json() for f in forecasts],
            "cached_at": service.cached_at(),
        },
    )


@router.get("/alerts")
async def alerts(request: Request) -> JSONResponse:
    service = get_service(request)
    items = service.alerts()
    return JSONResponse({"alerts": [f.to_json() for f in items]})


@router.get("/{ticker}")
async def ticker(ticker: str, request: Request) -> JSONResponse:
    service = get_service(request)
    forecast = service.predict_one(ticker.upper())
    if forecast is None:
        raise HTTPException(404, f"No forecast for {ticker}")
    return JSONResponse(forecast.to_json())


@router.post("/refresh")
async def refresh(request: Request) -> JSONResponse:
    service = get_service(request)
    forecasts = service.predict_all(force_refresh=True)
    return JSONResponse({"refreshed": len(forecasts)})
