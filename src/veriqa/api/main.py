"""FastAPI backend. Endpoints: /health /ask /evaluate."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ..service import VeriQAService
from ..config import ROOT

app = FastAPI(title="VeriQA", version="1.0",
              description="Selective extractive QA with a calibrated abstention gate.")
_svc = None

def svc():
    global _svc
    if _svc is None:
        _svc = VeriQAService()
    return _svc

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    k: int | None = Field(None, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "frozen": (ROOT / "results/RESULT_FREEZE.json").exists()}

@app.post("/ask")
def ask(req: AskRequest):
    return svc().ask(req.question, req.threshold, req.k)

@app.get("/evaluate")
def evaluate():
    """Frozen headline results. Never recomputed at request time."""
    p = ROOT / "results/RESULT_FREEZE.json"
    if not p.exists():
        return {"error": "results not frozen"}
    return json.loads(p.read_text())
