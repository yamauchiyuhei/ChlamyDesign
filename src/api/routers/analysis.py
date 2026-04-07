"""POST /api/restriction, POST /api/rna-structure — Sequence analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    RestrictionRequest,
    RestrictionResponse,
    RNAStructureRequest,
    RNAStructureResponse,
)
from src.api.services.sequence_analysis import (
    analyze_restriction_sites,
    predict_rna_structure,
)

router = APIRouter()


@router.post("/restriction", response_model=RestrictionResponse)
async def restriction_endpoint(req: RestrictionRequest) -> RestrictionResponse:
    """Analyze restriction enzyme sites in a DNA sequence."""
    try:
        return analyze_restriction_sites(
            dna=req.dna_sequence,
            enzyme_names=req.enzymes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restriction analysis error: {e}")


@router.post("/rna-structure", response_model=RNAStructureResponse)
async def rna_structure_endpoint(req: RNAStructureRequest) -> RNAStructureResponse:
    """Predict RNA secondary structure."""
    try:
        return predict_rna_structure(
            dna=req.dna_sequence,
            region=req.region,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RNA structure prediction error: {e}")
