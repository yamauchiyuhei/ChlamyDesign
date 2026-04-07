"""POST /api/evaluate — DNA sequence evaluation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import EvaluateRequest, EvaluateResponse
from src.api.services.evaluator import evaluate

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(req: EvaluateRequest) -> EvaluateResponse:
    """Compute evaluation metrics for a DNA sequence."""
    try:
        return evaluate(req.dna_sequence, organism=req.organism)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
