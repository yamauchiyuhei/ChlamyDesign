"""ChlamyDesign FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import ALLOWED_ORIGINS
from src.api.routers import evaluate, info, predict
from src.api.services.model_manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precompute CSI weights and codon frequencies at startup."""
    manager.precompute()
    yield


app = FastAPI(
    title="ChlamyDesign API",
    description="Codon optimization API for C. reinhardtii chloroplast",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(info.router, prefix="/api")


@app.get("/api/health")
async def health():
    """Health check (does NOT trigger model loading)."""
    return {"status": "ok"}
