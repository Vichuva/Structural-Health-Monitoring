from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.services import (
    DATA_ROOT,
    PROJECT_ROOT,
    get_bridge_detail,
    get_fleet_overview,
    get_operation,
    get_report_content,
    list_reports,
    start_crew_report_operation,
    start_bridge_refresh_operation,
    start_pipeline_operation,
)

app = FastAPI(title="Structural Health Monitoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets/data", StaticFiles(directory=DATA_ROOT), name="data-assets")


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def overview() -> dict:
    return get_fleet_overview()


@app.get("/api/bridges/{bridge_id}")
def bridge_detail(bridge_id: str) -> dict:
    try:
        return get_bridge_detail(bridge_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bridges/{bridge_id}/refresh")
def bridge_refresh(bridge_id: str) -> dict:
    try:
        return start_bridge_refresh_operation(bridge_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipeline/run")
def pipeline_run(generate_synthetic_data: bool = False) -> dict:
    return start_pipeline_operation(generate_synthetic_data)


@app.get("/api/operations/{operation_id}")
def operation_status(operation_id: str) -> dict:
    try:
        return get_operation(operation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports")
def reports() -> dict:
    return {"reports": list_reports()}


@app.get("/api/reports/{name}")
def report_detail(name: str) -> dict:
    try:
        return get_report_content(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/reports/generate")
async def report_generate() -> dict:
    return await asyncio.to_thread(start_crew_report_operation)


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/{full_path:path}")
    def frontend_app(full_path: str = ""):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
