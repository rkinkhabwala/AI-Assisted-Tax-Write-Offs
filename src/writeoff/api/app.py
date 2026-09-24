"""FastAPI application. Chat endpoints arrive in phase 9; for now it serves health checks."""

from typing import Literal

import psycopg
from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from writeoff import __doc__ as package_doc
from writeoff.config import get_settings

app = FastAPI(title="WriteOff Assistant", description=package_doc or "")


class Health(BaseModel):
    status: Literal["ok", "unavailable"]


@app.get("/healthz")
async def healthz() -> Health:
    """Liveness: the process is up."""
    return Health(status="ok")


@app.get("/readyz")
async def readyz(response: Response) -> Health:
    """Readiness: the database is reachable."""
    try:
        async with await psycopg.AsyncConnection.connect(
            str(get_settings().database_url), connect_timeout=3
        ) as conn:
            await conn.execute("SELECT 1")
    except psycopg.Error:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Health(status="unavailable")
    return Health(status="ok")
