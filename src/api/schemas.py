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


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    protein: str = Field(..., min_length=1, max_length=6200)
    strategy: Strategy = Strategy.cai_max
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
    num_candidates_generated: int
    start_codon: str
    stop_codon: str


# ---------------------------------------------------------------------------
# /api/evaluate
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    dna_sequence: str = Field(..., min_length=3)

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
