"""Pydantic request/response models."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Strategy(str, Enum):
    cai_max = "cai_max"
    bfc = "bfc"
    base_ct = "base_ct"
    chlamyct = "chlamyct"
    chlamydesign = "chlamydesign"


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    protein: str = Field(..., min_length=1, max_length=6200)
    strategy: Strategy = Strategy.cai_max
    organism: str = Field("Chlamydomonas reinhardtii chloroplast")
    num_sequences: int = Field(1, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.1, le=1.0)

    @field_validator("protein")
    @classmethod
    def validate_protein(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.endswith("_") and not v.endswith("*"):
            v += "_"
        v = v.replace("*", "_")
        valid = set("ACDEFGHIKLMNPQRSTVWY_")
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid amino acid characters: {invalid}")
        return v


class PredictResponse(BaseModel):
    predicted_dna: str
    gc_percent: float
    length_nt: int
    length_aa: int
    strategy: str
    organism: str
    num_candidates_generated: int
    start_codon: str
    stop_codon: str
    five_prime_mfe: Optional[float] = None
    base_strategy: Optional[str] = None
    cps_score: Optional[float] = None
    five_prime_context_score: Optional[float] = None


# ---------------------------------------------------------------------------
# /api/evaluate
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    dna_sequence: str = Field(..., min_length=3)
    organism: str = Field("Chlamydomonas reinhardtii chloroplast")

    @field_validator("dna_sequence")
    @classmethod
    def validate_dna(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "").replace("\n", "")
        if not re.fullmatch(r"[ATGC]+", v):
            raise ValueError("DNA must contain only A, T, G, C characters")
        if len(v) % 3 != 0:
            raise ValueError("DNA length must be divisible by 3")
        return v


class EvaluateResponse(BaseModel):
    gc_percent: float
    csi: float
    cfd: float
    minmax_mean: float
    cis_elements: int
    mfe: Optional[float]
    complexity: float
    length_nt: int
    is_valid_orf: bool
    start_codon: str
    stop_codon: str


# ---------------------------------------------------------------------------
# /api/organisms
# ---------------------------------------------------------------------------

class OrganismInfo(BaseModel):
    id: int
    name: str


class OrganismFrequencies(BaseModel):
    organism: str
    frequencies: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# /api/info
# ---------------------------------------------------------------------------

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    available: bool


class InfoResponse(BaseModel):
    organism: str
    gc_target: float
    reference_genes: int
    strategies: list[str]
    models: list[ModelStatus]
    codon_table: str
    max_protein_length_ct: int
    max_protein_length_freq: int
    total_organisms: int


# ---------------------------------------------------------------------------
# /api/restriction — Restriction enzyme analysis
# ---------------------------------------------------------------------------

class RestrictionRequest(BaseModel):
    dna_sequence: str = Field(..., min_length=3)
    enzymes: Optional[list[str]] = Field(
        None,
        description="List of enzyme names to check. If None, checks common enzymes."
    )

    @field_validator("dna_sequence")
    @classmethod
    def validate_dna_restriction(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "").replace("\n", "")
        if not re.fullmatch(r"[ATGCRYSWKMBDHVN]+", v):
            raise ValueError("DNA must contain only valid IUPAC nucleotide characters")
        return v


class RestrictionSite(BaseModel):
    enzyme: str
    recognition_site: str
    cut_positions: list[int]
    num_cuts: int
    fragment_sizes: list[int]


class RestrictionResponse(BaseModel):
    total_enzymes_checked: int
    total_sites_found: int
    length_nt: int
    enzymes: list[RestrictionSite]


# ---------------------------------------------------------------------------
# /api/rna-structure — RNA secondary structure prediction
# ---------------------------------------------------------------------------

class RNAStructureRequest(BaseModel):
    dna_sequence: str = Field(..., min_length=3, max_length=9000)
    region: Optional[str] = Field(
        None,
        description="Region to analyze: 'full', '5utr' (first 100nt), '3utr' (last 100nt)"
    )

    @field_validator("dna_sequence")
    @classmethod
    def validate_dna_rna(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "").replace("\n", "")
        if not re.fullmatch(r"[ATGC]+", v):
            raise ValueError("DNA must contain only A, T, G, C characters")
        return v


class RNAStructureResponse(BaseModel):
    sequence: str
    structure: str
    mfe: float
    length_nt: int
    gc_percent: float
    base_pair_count: int
    region: str
    # ViennaRNA extended fields
    ensemble_energy: Optional[float] = None
    ensemble_diversity: Optional[float] = None
    centroid_structure: Optional[str] = None
    centroid_mfe: Optional[float] = None
    positional_entropy: Optional[list[float]] = None
    mountain_mfe: Optional[list[int]] = None
    mountain_centroid: Optional[list[int]] = None
    mountain_pf: Optional[list[float]] = None
    # Planar 2D layout coordinates (NAView via ViennaRNA) for MFE structure
    layout_x: Optional[list[float]] = None
    layout_y: Optional[list[float]] = None
    layout_type: Optional[str] = None
    # Planar 2D layout coordinates for centroid structure
    layout_centroid_x: Optional[list[float]] = None
    layout_centroid_y: Optional[list[float]] = None
    source: str = "nussinov"
