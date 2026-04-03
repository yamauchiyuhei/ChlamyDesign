"""GET /api/info — Model and organism information."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.config import CT_MAX_AA, FREQ_MAX_AA, ORGANISM
from src.api.schemas import InfoResponse, ModelStatus
from src.api.services.model_manager import manager

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def info_endpoint() -> InfoResponse:
    """Return model status and organism information."""
    return InfoResponse(
        organism=ORGANISM,
        gc_target=34.0,
        reference_genes=manager.reference_gene_count,
        strategies=["cai_max", "bfc", "base_ct", "chlamyct"],
        models=[
            ModelStatus(
                name="Base CodonTransformer",
                loaded=manager.base_model_loaded,
                available=True,
            ),
            ModelStatus(
                name="ChlamyCodonTransformer",
                loaded=manager.chlamyct_model_loaded,
                available=manager.chlamyct_available,
            ),
        ],
        codon_table="Plastid code (Table 11)",
        max_protein_length_ct=CT_MAX_AA,
        max_protein_length_freq=FREQ_MAX_AA,
    )
