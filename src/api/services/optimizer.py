"""Codon optimization strategies using CodonTransformer."""

from __future__ import annotations

from CodonTransformer.CodonPrediction import (
    get_background_frequency_choice_sequence,
    get_high_frequency_choice_sequence,
    predict_dna_sequence,
)
from CodonTransformer.CodonUtils import ORGANISM2ID

from src.api.config import CT_MAX_AA, DEFAULT_ORGANISM, FREQ_MAX_AA
from src.api.schemas import PredictResponse, Strategy
from src.api.services.model_manager import manager

TARGET_GC = 34.0


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

    if strategy in (Strategy.chlamyct, Strategy.base_ct):
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
