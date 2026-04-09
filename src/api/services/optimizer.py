"""Codon optimization strategies using CodonTransformer."""

from __future__ import annotations

import logging
from typing import Optional

from CodonTransformer.CodonPrediction import (
    get_background_frequency_choice_sequence,
    get_high_frequency_choice_sequence,
    predict_dna_sequence,
)
from CodonTransformer.CodonUtils import ORGANISM2ID

from src.api.config import CT_MAX_AA, DEFAULT_ORGANISM, FREQ_MAX_AA
from src.api.schemas import PredictResponse, Strategy
from src.api.services.codon_pair import score_cps
from src.api.services.evaluator import _calc_mfe
from src.api.services.five_prime_context import score_five_prime_context
from src.api.services.model_manager import manager

logger = logging.getLogger(__name__)

TARGET_GC = 34.0
# Window for 5' end MFE re-ranking (Kudla et al. Science 2009: -4 to +37 nt
# around start codon → ~40 nt within CDS; use 47 nt for safety margin).
FIVE_PRIME_WINDOW_NT = 47
# Minimum candidate count for ChlamyDesign re-ranking to be meaningful.
CHLAMYDESIGN_MIN_CANDIDATES = 5
# Weights for combining z-scored objectives in ChlamyDesign re-ranking.
# All three objectives are equal-weighted by default; sum to 1.0.
CHLAMYDESIGN_W_MFE = 1.0 / 3.0
CHLAMYDESIGN_W_CPS = 1.0 / 3.0
CHLAMYDESIGN_W_5P_CONTEXT = 1.0 / 3.0


def optimize(
    protein: str,
    strategy: Strategy,
    organism: str = DEFAULT_ORGANISM,
    num_sequences: int = 1,
    temperature: float = 0.2,
) -> PredictResponse:
    """Run codon optimization for the given strategy and organism."""
    # Validate organism
    if organism not in ORGANISM2ID:
        raise ValueError(f"Unknown organism: {organism}. Use /api/organisms for the full list.")

    length_aa = len(protein.rstrip("_"))
    five_prime_mfe: Optional[float] = None
    base_strategy: Optional[str] = None
    cps_score: Optional[float] = None
    five_prime_context_score: Optional[float] = None

    if strategy == Strategy.chlamydesign:
        if length_aa > CT_MAX_AA:
            raise ValueError(
                f"Protein too long for CT model ({length_aa} aa > {CT_MAX_AA} aa max)"
            )
        (
            dna,
            five_prime_mfe,
            cps_score,
            five_prime_context_score,
            base_strategy,
            candidates,
        ) = _predict_chlamydesign(
            protein, organism, num_sequences, temperature
        )
    elif strategy in (Strategy.chlamyct, Strategy.base_ct):
        if length_aa > CT_MAX_AA:
            raise ValueError(
                f"Protein too long for CT model ({length_aa} aa > {CT_MAX_AA} aa max)"
            )
        dna = _predict_ct(protein, strategy, organism, num_sequences, temperature)
        candidates = num_sequences
    else:
        if length_aa > FREQ_MAX_AA:
            raise ValueError(
                f"Protein too long ({length_aa} aa > {FREQ_MAX_AA} aa max)"
            )
        dna = _predict_freq(protein, strategy, organism)
        candidates = 1

    gc = sum(1 for b in dna if b in "GC") / len(dna) * 100

    return PredictResponse(
        predicted_dna=dna,
        gc_percent=round(gc, 2),
        length_nt=len(dna),
        length_aa=length_aa,
        strategy=strategy.value,
        organism=organism,
        num_candidates_generated=candidates,
        start_codon=dna[:3],
        stop_codon=dna[-3:],
        five_prime_mfe=five_prime_mfe,
        base_strategy=base_strategy,
        cps_score=cps_score,
        five_prime_context_score=five_prime_context_score,
    )


def _predict_ct(
    protein: str,
    strategy: Strategy,
    organism: str,
    num_sequences: int,
    temperature: float,
) -> str:
    """Predict DNA using CodonTransformer model."""
    if strategy == Strategy.chlamyct:
        model = manager.get_chlamyct_model()
        if model is None:
            raise ValueError("ChlamyCT model is not available")
    else:
        model = manager.get_base_model()

    tokenizer = manager.get_tokenizer()
    deterministic = num_sequences == 1

    result = predict_dna_sequence(
        protein=protein,
        organism=organism,
        device=manager.device,
        model=model,
        tokenizer=tokenizer,
        deterministic=deterministic,
        temperature=temperature,
        num_sequences=num_sequences,
        match_protein=True,
    )

    if num_sequences > 1:
        best = min(
            result,
            key=lambda r: abs(
                sum(1 for b in r.predicted_dna if b in "GC")
                / len(r.predicted_dna) * 100
                - TARGET_GC
            ),
        )
        return best.predicted_dna
    return result.predicted_dna


def _predict_freq(protein: str, strategy: Strategy, organism: str) -> str:
    """Predict DNA using frequency-based strategy."""
    codon_freqs = manager.get_organism_frequencies(organism)
    if not codon_freqs:
        raise ValueError("Codon frequencies not available for this organism")

    # Strip stop symbol — CT freq functions don't handle it
    clean_protein = protein.rstrip("_*")

    # Add stop codon entry to frequency table
    # Convert dict[str, dict[str, float]] to AMINO2CODON_TYPE
    # CodonTransformer expects: Dict[str, Tuple[List[str], List[float]]]
    amino2codon = {}
    for aa, codons in codon_freqs.items():
        codon_list = list(codons.keys())
        freq_list = list(codons.values())
        # Ensure frequencies sum to exactly 1.0
        total = sum(freq_list)
        if total > 0 and abs(total - 1.0) > 1e-9:
            freq_list = [f / total for f in freq_list]
        # Fix floating point: adjust last element
        if freq_list:
            s = sum(freq_list[:-1])
            freq_list[-1] = 1.0 - s
        amino2codon[aa] = (codon_list, freq_list)

    # Add stop codon mapping
    amino2codon["_"] = (["TAA", "TAG", "TGA"], [0.50, 0.30, 0.20])

    if strategy == Strategy.cai_max:
        return get_high_frequency_choice_sequence(clean_protein + "_", amino2codon)
    else:
        return get_background_frequency_choice_sequence(clean_protein + "_", amino2codon)


def _zscores(values: list[float]) -> list[float]:
    """Z-score normalize a list. Returns zeros if variance is zero."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 1e-12:
        return [0.0] * n
    sd = var ** 0.5
    return [(v - mean) / sd for v in values]


def _predict_chlamydesign(
    protein: str,
    organism: str,
    num_sequences: int,
    temperature: float,
) -> tuple[str, Optional[float], Optional[float], Optional[float], str, int]:
    """ChlamyDesign strategy: ChlamyCT + biological inductive biases.

    Phase 1: 5' end mRNA secondary structure minimization (Kudla 2009).
    Phase 2: Codon pair score (CPS) maximization (Coleman 2008), reflecting
             the limited tRNA pool of the *Chlamydomonas* chloroplast.
    Phase 3: 5' codon context preservation (log-odds of 5' codon usage in
             high-expression chloroplast genes vs the chloroplast-wide
             background) — preserves nuclear-encoded translational activator
             binding context that extends into CDS-internal codons.

    Combined re-ranking: each candidate is scored by a weighted sum of the
    z-scored objectives, normalized within the candidate batch. The candidate
    with the highest combined score is returned.

    Returns:
        (best_dna, five_prime_mfe, cps_score, five_prime_context_score,
         base_strategy_name, n_candidates).
    """
    # Choose underlying generator: ChlamyCT preferred, fall back to Base CT.
    model = manager.get_chlamyct_model()
    if model is not None:
        base_strategy = "chlamyct"
    else:
        model = manager.get_base_model()
        base_strategy = "base_ct"
        logger.info("ChlamyDesign: ChlamyCT unavailable, falling back to Base CT")

    # Force at least CHLAMYDESIGN_MIN_CANDIDATES candidates so re-ranking is meaningful.
    n = max(num_sequences, CHLAMYDESIGN_MIN_CANDIDATES)
    tokenizer = manager.get_tokenizer()

    result = predict_dna_sequence(
        protein=protein,
        organism=organism,
        device=manager.device,
        model=model,
        tokenizer=tokenizer,
        deterministic=False,
        temperature=temperature,
        num_sequences=n,
        match_protein=True,
    )

    candidates = result if isinstance(result, list) else [result]
    cps_table = manager.cps_table
    bias_table = manager.five_prime_bias

    # Per-candidate raw metrics.
    dnas: list[str] = [r.predicted_dna for r in candidates]
    mfes: list[Optional[float]] = [_calc_mfe(d[:FIVE_PRIME_WINDOW_NT]) for d in dnas]
    cps_values: list[float] = [score_cps(d, cps_table) for d in dnas]
    ctx_values: list[float] = [score_five_prime_context(d, bias_table) for d in dnas]
    gc_distances: list[float] = [
        abs(sum(1 for b in d if b in "GC") / len(d) * 100 - TARGET_GC) for d in dnas
    ]

    mfe_available = any(m is not None for m in mfes)
    cps_available = bool(cps_table) and any(v != 0.0 for v in cps_values)
    ctx_available = bool(bias_table) and any(v != 0.0 for v in ctx_values)

    # Combined z-score across whichever objectives are available.
    objective_terms: list[tuple[list[float], float]] = []
    if mfe_available:
        valid_mfes = [m for m in mfes if m is not None]
        fallback_mfe = min(valid_mfes) if valid_mfes else 0.0
        mfe_filled = [m if m is not None else fallback_mfe for m in mfes]
        objective_terms.append((_zscores(mfe_filled), CHLAMYDESIGN_W_MFE))
    if cps_available:
        objective_terms.append((_zscores(cps_values), CHLAMYDESIGN_W_CPS))
    if ctx_available:
        objective_terms.append((_zscores(ctx_values), CHLAMYDESIGN_W_5P_CONTEXT))

    if objective_terms:
        # Renormalize weights to sum to 1 over the available objectives.
        total_w = sum(w for _, w in objective_terms)
        combined = [
            sum((zlist[i] * (w / total_w)) for zlist, w in objective_terms)
            for i in range(n)
        ]
        best_idx = max(range(n), key=lambda i: (combined[i], -gc_distances[i]))
    else:
        logger.warning(
            "ChlamyDesign: no biological objectives available; "
            "falling back to GC%% tiebreaker"
        )
        best_idx = min(range(n), key=lambda i: gc_distances[i])

    return (
        dnas[best_idx],
        mfes[best_idx],
        cps_values[best_idx] if cps_available else None,
        ctx_values[best_idx] if ctx_available else None,
        base_strategy,
        n,
    )
