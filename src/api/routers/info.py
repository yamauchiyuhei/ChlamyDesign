"""GET /api/info, GET /api/organisms — Model and organism information."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.config import CT_MAX_AA, DEFAULT_ORGANISM, FREQ_MAX_AA
from src.api.schemas import InfoResponse, ModelStatus, OrganismFrequencies, OrganismInfo
from src.api.services.model_manager import manager

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def info_endpoint() -> InfoResponse:
    """Return model status and organism information."""
    return InfoResponse(
        organism=DEFAULT_ORGANISM,
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
        total_organisms=len(manager.organism_list),
    )


@router.get("/organisms", response_model=list[OrganismInfo])
async def organisms_endpoint() -> list[OrganismInfo]:
    """Return the full list of supported organisms."""
    return [OrganismInfo(**org) for org in manager.organism_list]


@router.get("/organisms/{organism_name}/frequencies", response_model=OrganismFrequencies)
async def organism_frequencies_endpoint(organism_name: str) -> OrganismFrequencies:
    """Return codon frequencies for a specific organism."""
    try:
        freqs = manager.get_organism_frequencies(organism_name)
        return OrganismFrequencies(organism=organism_name, frequencies=freqs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
