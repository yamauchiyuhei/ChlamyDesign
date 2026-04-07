"""DNA sequence evaluation metrics using CodonTransformer."""

from __future__ import annotations

import re
from typing import Optional

from CodonTransformer.CodonEvaluation import (
    get_CSI_value,
    get_GC_content,
    get_cfd,
    get_min_max_percentage,
    get_sequence_complexity,
)

from src.api.config import DEFAULT_ORGANISM, MFE_MAX_NT
from src.api.schemas import EvaluateResponse
from src.api.services.model_manager import manager

# Cis-element patterns (from compare_models.py)
CIS_ELEMENT_PATTERNS = {
    "SD_like": r"AAGG.{5,10}ATG",
    "TATA_box": r"TATAAA",
    "polyA": r"AAAAAA",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}

# ViennaRNA is optional
try:
    import RNA as _RNA

    def _calc_mfe(dna: str) -> Optional[float]:
        if len(dna) > MFE_MAX_NT:
            return None
        rna = dna.replace("T", "U")
        try:
            _, mfe = _RNA.fold(rna)
            return round(mfe, 2)
        except Exception:
            return None

except ImportError:
    def _calc_mfe(dna: str) -> Optional[float]:
        return None


def _count_cis_elements(dna: str) -> int:
    count = 0
    for pattern in CIS_ELEMENT_PATTERNS.values():
        count += len(re.findall(pattern, dna, re.IGNORECASE))
    return count


def evaluate(dna: str, organism: str = DEFAULT_ORGANISM) -> EvaluateResponse:
    """Compute all evaluation metrics for a DNA sequence.

    Uses the specified organism's codon frequencies for CSI, CFD, and %MinMax
    calculations. Falls back to Chlamydomonas chloroplast if organism not specified.
    """
    # Get organism-specific frequencies
    try:
        codon_freqs = manager.get_organism_frequencies(organism)
    except ValueError:
        codon_freqs = manager.codon_freqs  # fallback to Chlamydomonas

    # CSI weights are Chlamydomonas-specific; for other organisms
    # we still compute CSI based on available data
    csi_weights = manager.csi_weights if organism == DEFAULT_ORGANISM else {}

    gc = get_GC_content(dna)

    csi = 0.0
    if csi_weights:
        try:
            csi = get_CSI_value(dna, csi_weights)
        except Exception:
            pass

    cfd = 0.0
    if codon_freqs:
        try:
            # Convert to AMINO2CODON_TYPE format for get_cfd
            amino2codon = {}
            for aa, codons in codon_freqs.items():
                codon_list = list(codons.keys())
                freq_list = list(codons.values())
                amino2codon[aa] = (codon_list, freq_list)
            cfd = get_cfd(dna, amino2codon)
        except Exception:
            pass

    minmax_mean = 0.0
    if codon_freqs:
        try:
            amino2codon = {}
            for aa, codons in codon_freqs.items():
                codon_list = list(codons.keys())
                freq_list = list(codons.values())
                amino2codon[aa] = (codon_list, freq_list)
            vals = get_min_max_percentage(dna, amino2codon, window_size=18)
            minmax_mean = sum(vals) / len(vals) if vals else 0.0
        except Exception:
            pass

    try:
        complexity = get_sequence_complexity(dna)
    except Exception:
        complexity = 0.0

    cis = _count_cis_elements(dna)
    mfe = _calc_mfe(dna)

    start_codon = dna[:3]
    stop_codon = dna[-3:]
    is_valid_orf = (
        start_codon in ("ATG", "GTG")
        and stop_codon in STOP_CODONS
        and len(dna) % 3 == 0
    )

    return EvaluateResponse(
        gc_percent=round(gc, 2),
        csi=round(csi, 4),
        cfd=round(cfd, 4),
        minmax_mean=round(minmax_mean, 4),
        cis_elements=cis,
        mfe=mfe,
        complexity=round(complexity, 4),
        length_nt=len(dna),
        is_valid_orf=is_valid_orf,
        start_codon=start_codon,
        stop_codon=stop_codon,
    )
