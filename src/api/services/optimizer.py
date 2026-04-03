"""Codon optimization strategies using CodonTransformer."""

from __future__ import annotations

from CodonTransformer.CodonPrediction import (
    get_background_frequency_choice_sequence,
    get_high_frequency_choice_sequence,
    predict_dna_sequence,
)

from src.api.config import CT_MAX_AA, FREQ_MAX_AA, ORGANISM
from src.api.schemas import PredictResponse, Strategy
from src.api.services.model_manager import manager

TARGET_GC = 34.0


def optimize(
    protein: str,
    strategy: Strategy,
    num_sequences: int = 1,
    temperature: float = 0.2,
) -> PredictResponse:
    """Run codon optimization for the given strategy."""
    length_aa = len(protein.rstrip("_"))

    if strategy in (Strategy.chlamyct, Strategy.base_ct):
        if length_aa > CT_MAX_AA:
            raise ValueError(
                f"Protein too long for CT model ({length_aa} aa > {CT_MAX_AA} aa max)"
            )
        dna = _predict_ct(protein, strategy, num_sequences, temperature)
        candidates = num_sequences
    else:
        if length_aa > FREQ_MAX_AA:
            raise ValueError(
                f"Protein too long ({length_aa} aa > {FREQ_MAX_AA} aa max)"
            )
        dna = _predict_freq(protein, strategy)
        candidates = 1

    gc = sum(1 for b in dna if b in "GC") / len(dna) * 100

    return PredictResponse(
        predicted_dna=dna,
        gc_percent=round(gc, 2),
        length_nt=len(dna),
        length_aa=length_aa,
        strategy=strategy.value,
        num_candidates_generated=candidates,
        start_codon=dna[:3],
        stop_codon=dna[-3:],
    )


def _predict_ct(
    protein: str,
    strategy: Strategy,
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
        organism=ORGANISM,
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


def _predict_freq(protein: str, strategy: Strategy) -> str:
    """Predict DNA using frequency-based strategy."""
    codon_freqs = manager.codon_freqs
    if not codon_freqs:
        raise ValueError("Codon frequencies not loaded")

    if strategy == Strategy.cai_max:
        return get_high_frequency_choice_sequence(protein, codon_freqs)
    else:
        return get_background_frequency_choice_sequence(protein, codon_freqs)
