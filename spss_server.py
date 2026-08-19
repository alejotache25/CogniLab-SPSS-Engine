"""
FastAPI server for the SPSS engine.
Deploy on Railway, Render, or any Python host.
"""
import json
import math
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from spss_engine.cli import run_engine

app = FastAPI(title="CogniLab SPSS Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return str(obj)


@app.post("/api/spss-engine")
async def execute_spss(request: Request):
    """Execute SPSS syntax and return JSON result."""
    data = await request.json()
    syntax = data.get("syntax", "")
    result = run_engine(syntax)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "spss_engine", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)