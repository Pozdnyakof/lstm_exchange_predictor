"""FastAPI-приложение веб-интерфейса трейдера.

Маршруты разнесены по роутерам:
    * routes_live   — live-прогнозы (§3.1, §3.3)
    * routes_report — отчёт о бэктесте (§3.4)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import default_config
from ..serving import (
    InferenceService,
    LiveFeatureBuilder,
    RefreshScheduler,
    load_artifact,
)
from .routes_live import router as live_router
from .routes_report import router as report_router

logger = logging.getLogger(__name__)


def _init_runtime() -> dict:
    """Загружает модель и стартует scheduler. При отсутствии артефакта
    возвращает пустой state - сервер работает, но live-эндпоинты дают 503."""
    cfg = default_config()
    state: dict = {"cfg": cfg, "service": None, "scheduler": None}
    try:
        loaded = load_artifact(cfg.paths.checkpoints)
    except FileNotFoundError as exc:
        logger.warning("No model artifact found: %s. Live endpoints will return 503.", exc)
        return state

    builder = LiveFeatureBuilder(loaded, cfg.data, cfg.serving)
    service = InferenceService(loaded, cfg.data, cfg.trading, cfg.serving, builder)
    scheduler = RefreshScheduler(service, cfg.serving.refresh_interval_sec)
    scheduler.start()
    state["service"] = service
    state["scheduler"] = scheduler
    logger.info("Live inference service ready: %d tickers", len(loaded.meta.tickers))
    return state


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.runtime = _init_runtime()
    try:
        yield
    finally:
        scheduler = app.state.runtime.get("scheduler")
        if scheduler is not None:
            scheduler.stop()


def _register_pages(app: FastAPI, templates: Jinja2Templates) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def live_page(request: Request) -> HTMLResponse:
        state = getattr(request.app.state, "runtime", {})
        service: InferenceService | None = state.get("service") if isinstance(state, dict) else None
        meta = service.loaded.meta if service is not None else None
        return templates.TemplateResponse(
            request,
            "live.html",
            {
                "tickers": meta.tickers if meta else [],
                "training_date": meta.training_date if meta else None,
                "horizons": meta.horizons if meta else [],
                "service_ready": service is not None,
            },
        )

    @app.get("/report", response_class=HTMLResponse)
    async def report_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "report.html", {})


def create_app() -> FastAPI:
    app = FastAPI(title="Graduate Work - Trading Dashboard", lifespan=_lifespan)
    here = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(here / "templates"))

    static_dir = here / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _register_pages(app, templates)
    app.include_router(live_router)
    app.include_router(report_router)
    return app


app = create_app()
