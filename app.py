"""
FastAPI app for the HTML-based Mumzworld AI Gift Finder demo.

Run locally:
    uvicorn app:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.gift_finder import find_gifts

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Mumzworld AI Gift Finder", version="1.0.0")


class GiftRequest(BaseModel):
    query: str = Field(default="", max_length=500)


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return (BASE_DIR / "Landing_page.html").read_text(encoding="utf-8")


@app.get("/results", response_class=HTMLResponse)
def results_page():
    return (BASE_DIR / "Results_page.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/find")
def api_find_gifts(request: GiftRequest):
    try:
        return find_gifts(request.query).model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Gift finder failed. Check server logs, API key, and model access.",
        ) from exc
