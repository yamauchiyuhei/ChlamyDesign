"""POST /api/predict — DNA sequence prediction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictRequest, PredictResponse
from src.api.services.optimizer import optimize

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """Generate an optimized DNA sequence from a protein sequence."""
    try:
        return optimize(
            protein=req.protein,
            strategy=req.strategy,
            num_sequences=req.num_sequences,
            temperature=req.temperature,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
